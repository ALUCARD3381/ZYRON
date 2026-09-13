#include "zyron/model/checkpoint.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace zyron::model {
namespace {
constexpr std::uint32_t kMagic = 0x4e52595a; // "ZYRN"
constexpr std::uint32_t kVersion = 4;
constexpr std::uint32_t kLegacyVersion = 1;
constexpr std::uint32_t kPreviousVersion = 2;
constexpr std::uint32_t kDropoutVersion = 3;

void write_exact(std::ostream& out, const void* data, std::size_t bytes) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("ZYRON checkpoint: write failure");
}
void read_exact(std::istream& in, void* data, std::size_t bytes) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("ZYRON checkpoint: corrupted file");
}
void write_u64(std::ostream& out, std::uint64_t v) { write_exact(out, &v, sizeof(v)); }
std::uint64_t read_u64(std::istream& in) { std::uint64_t v=0; read_exact(in,&v,sizeof(v)); return v; }
void write_u32(std::ostream& out, std::uint32_t v) { write_exact(out, &v, sizeof(v)); }
std::uint32_t read_u32(std::istream& in) { std::uint32_t v=0; read_exact(in,&v,sizeof(v)); return v; }
void write_f32(std::ostream& out, float v) { write_exact(out, &v, sizeof(v)); }
float read_f32(std::istream& in) { float v=0.0f; read_exact(in,&v,sizeof(v)); return v; }

void write_config(std::ostream& out, const transformer::Config& c) {
    write_u64(out, c.vocab_size); write_u64(out, c.hidden_size); write_u64(out, c.num_layers);
    write_u64(out, c.num_heads); write_u64(out, c.intermediate_size); write_u64(out, c.max_sequence_length);
    write_f32(out, c.norm_eps); write_u32(out, c.use_rms_norm ? 1u : 0u); write_u32(out, c.causal ? 1u : 0u);
    write_u64(out, c.num_kv_heads); write_u32(out, c.use_rope ? 1u : 0u);
    write_u32(out, c.tie_word_embeddings ? 1u : 0u); write_f32(out, c.rope_theta);
    write_f32(out, c.dropout);
    write_u32(out, c.use_qat ? 1u : 0u);
    write_u64(out, c.qat_bits);
}
transformer::Config read_config(std::istream& in, std::uint32_t version) {
    transformer::Config c;
    c.vocab_size = static_cast<std::size_t>(read_u64(in));
    c.hidden_size = static_cast<std::size_t>(read_u64(in));
    c.num_layers = static_cast<std::size_t>(read_u64(in));
    c.num_heads = static_cast<std::size_t>(read_u64(in));
    c.intermediate_size = static_cast<std::size_t>(read_u64(in));
    c.max_sequence_length = static_cast<std::size_t>(read_u64(in));
    c.norm_eps = read_f32(in); c.use_rms_norm = read_u32(in) != 0; c.causal = read_u32(in) != 0;
    if (version >= 2) {
        c.num_kv_heads = static_cast<std::size_t>(read_u64(in));
        c.use_rope = read_u32(in) != 0;
        c.tie_word_embeddings = read_u32(in) != 0;
        c.rope_theta = read_f32(in);
        if (version >= kDropoutVersion) {
            c.dropout = read_f32(in);
            if (version >= kVersion) {
                c.use_qat = read_u32(in) != 0;
                c.qat_bits = static_cast<std::size_t>(read_u64(in));
            }
        }
    } else {
        c.num_kv_heads = c.num_heads;
        c.use_rope = false;
        c.tie_word_embeddings = false;
    }
    c.validate();
    return c;
}
void write_tensor(std::ostream& out, const Tensor& tensor) {
    const Tensor contiguous = tensor.contiguous();
    write_u64(out, static_cast<std::uint64_t>(contiguous.rank()));
    for (std::size_t d : contiguous.shape().dims()) write_u64(out, static_cast<std::uint64_t>(d));
    write_u64(out, static_cast<std::uint64_t>(contiguous.size()));
    write_exact(out, contiguous.data(), contiguous.size() * sizeof(float));
}
Tensor read_tensor(std::istream& in) {
    const auto rank64 = read_u64(in);
    if (rank64 > 64) throw std::runtime_error("ZYRON checkpoint: invalid tensor rank");
    std::vector<std::size_t> dims;
    dims.reserve(static_cast<std::size_t>(rank64));
    for (std::size_t i=0;i<rank64;++i) dims.push_back(static_cast<std::size_t>(read_u64(in)));
    const Shape shape(std::move(dims));
    const auto count = read_u64(in);
    if (count != shape.size()) throw std::runtime_error("ZYRON checkpoint: tensor size mismatch");
    Tensor t(shape);
    if (count != 0) read_exact(in, t.data(), static_cast<std::size_t>(count) * sizeof(float));
    return t;
}
void write_tensor_vector(std::ostream& out, const std::vector<Tensor>& tensors) {
    write_u64(out, static_cast<std::uint64_t>(tensors.size()));
    for (const auto& t : tensors) write_tensor(out,t);
}
std::vector<Tensor> read_tensor_vector(std::istream& in) {
    const auto n=read_u64(in);
    if (n > 1000000) throw std::runtime_error("ZYRON checkpoint: too many tensors");
    std::vector<Tensor> result; result.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i=0;i<n;++i) result.push_back(read_tensor(in));
    return result;
}
}

void Checkpoint::save(
    const std::string& path,
    const LanguageModel& model,
    const training::AdamW& optimizer,
    const training::WarmupCosineScheduler& scheduler,
    const tokenizer::BPE& tokenizer,
    std::size_t training_step) {

    std::ostringstream tokenizer_stream(std::ios::binary | std::ios::out);
    tokenizer.save(tokenizer_stream);
    const std::string tok = tokenizer_stream.str();

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("ZYRON checkpoint: failed to open: " + path);

    write_u32(out,kMagic); write_u32(out,kVersion); write_config(out,model.config());
    write_u64(out,static_cast<std::uint64_t>(training_step));
    write_u64(out,static_cast<std::uint64_t>(optimizer.step_count()));
    write_f32(out,optimizer.learning_rate());
    write_f32(out,optimizer.beta1()); write_f32(out,optimizer.beta2());
    write_f32(out,optimizer.epsilon()); write_f32(out,optimizer.weight_decay());
    write_u64(out,static_cast<std::uint64_t>(scheduler.step_count()));
    write_f32(out,scheduler.base_learning_rate());
    write_u64(out,static_cast<std::uint64_t>(scheduler.warmup_steps()));
    write_u64(out,static_cast<std::uint64_t>(scheduler.total_steps()));
    write_f32(out,scheduler.min_learning_rate());

    const auto parameters = const_cast<LanguageModel&>(model).parameters();
    std::vector<Tensor> parameter_values; parameter_values.reserve(parameters.size());
    for (const auto* p : parameters) parameter_values.push_back(p->value());
    write_tensor_vector(out,parameter_values);
    write_tensor_vector(out,optimizer.first_moments());
    write_tensor_vector(out,optimizer.second_moments());

    if (tok.size() > std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("ZYRON checkpoint: tokenizer too large");
    write_u64(out,static_cast<std::uint64_t>(tok.size()));
    if (!tok.empty()) write_exact(out,tok.data(),tok.size());
}

CheckpointState Checkpoint::load(const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    if (!in) throw std::runtime_error("ZYRON checkpoint: failed to open: " + path);
    if (read_u32(in)!=kMagic) throw std::runtime_error("ZYRON checkpoint: unsupported format");
    const auto version = read_u32(in);
    if (version != kLegacyVersion && version != kPreviousVersion && version != kDropoutVersion && version != kVersion) {
        throw std::runtime_error("ZYRON checkpoint: unsupported format");
    }

    CheckpointState state;
    state.config=read_config(in, version);
    state.training_step=static_cast<std::size_t>(read_u64(in));
    state.optimizer_step=static_cast<std::size_t>(read_u64(in));
    state.optimizer_learning_rate=read_f32(in);
    state.optimizer_beta1=read_f32(in);
    state.optimizer_beta2=read_f32(in);
    state.optimizer_epsilon=read_f32(in);
    state.optimizer_weight_decay=read_f32(in);
    state.scheduler_step=static_cast<std::size_t>(read_u64(in));
    state.scheduler_base_learning_rate=read_f32(in);
    state.scheduler_warmup_steps=static_cast<std::size_t>(read_u64(in));
    state.scheduler_total_steps=static_cast<std::size_t>(read_u64(in));
    state.scheduler_min_learning_rate=read_f32(in);
    state.model_parameters=read_tensor_vector(in);
    state.first_moments=read_tensor_vector(in);
    state.second_moments=read_tensor_vector(in);
    const auto tok_size=read_u64(in);
    if (tok_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::overflow_error("ZYRON checkpoint: tokenizer size overflow");
    state.tokenizer_data.resize(static_cast<std::size_t>(tok_size));
    if (tok_size) read_exact(in,state.tokenizer_data.data(),static_cast<std::size_t>(tok_size));
    return state;
}

void Checkpoint::restore(
    const CheckpointState& state,
    LanguageModel& model,
    training::AdamW& optimizer,
    training::WarmupCosineScheduler& scheduler,
    tokenizer::BPE& tokenizer) {

    if (model.config().vocab_size != state.config.vocab_size ||
        model.config().hidden_size != state.config.hidden_size ||
        model.config().num_layers != state.config.num_layers ||
        model.config().num_heads != state.config.num_heads ||
        model.config().num_kv_heads != state.config.num_kv_heads ||
        model.config().intermediate_size != state.config.intermediate_size ||
        model.config().max_sequence_length != state.config.max_sequence_length ||
        model.config().norm_eps != state.config.norm_eps ||
        model.config().use_rms_norm != state.config.use_rms_norm ||
        model.config().causal != state.config.causal ||
        model.config().use_rope != state.config.use_rope ||
        model.config().tie_word_embeddings != state.config.tie_word_embeddings ||
        model.config().rope_theta != state.config.rope_theta ||
        model.config().dropout != state.config.dropout ||
        model.config().use_qat != state.config.use_qat ||
        model.config().qat_bits != state.config.qat_bits) {
        throw std::invalid_argument("ZYRON checkpoint: model configuration mismatch");
    }

    auto parameters=model.parameters();
    if (parameters.size()!=state.model_parameters.size()) throw std::invalid_argument("ZYRON checkpoint: parameter count mismatch");
    for (std::size_t i=0;i<parameters.size();++i) {
        if (parameters[i]->value().shape()!=state.model_parameters[i].shape()) throw std::invalid_argument("ZYRON checkpoint: parameter shape mismatch");
        parameters[i]->value()=state.model_parameters[i];
    }

    optimizer.restore_state(
        state.optimizer_step,
        state.optimizer_learning_rate,
        state.optimizer_beta1,
        state.optimizer_beta2,
        state.optimizer_epsilon,
        state.optimizer_weight_decay,
        state.first_moments,
        state.second_moments);
    if (scheduler.base_learning_rate() != state.scheduler_base_learning_rate ||
        scheduler.warmup_steps() != state.scheduler_warmup_steps ||
        scheduler.total_steps() != state.scheduler_total_steps ||
        scheduler.min_learning_rate() != state.scheduler_min_learning_rate) {
        throw std::invalid_argument("ZYRON checkpoint: scheduler configuration mismatch");
    }
    scheduler.restore_step(state.scheduler_step);

    if (state.tokenizer_data.empty()) throw std::runtime_error("ZYRON checkpoint: tokenizer data missing");
    std::string bytes(reinterpret_cast<const char*>(state.tokenizer_data.data()),state.tokenizer_data.size());
    std::istringstream tokenizer_stream(bytes,std::ios::binary | std::ios::in);
    tokenizer.load(tokenizer_stream);
}

} // namespace zyron::model
