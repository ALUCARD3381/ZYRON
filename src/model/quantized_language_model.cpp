#include "zyron/model/quantized_language_model.hpp"

#include "zyron/autograd/autograd.hpp"
#include "zyron/math/ops.hpp"
#include "zyron/transformer/attention.hpp"
#include "zyron/transformer/feed_forward.hpp"
#include "zyron/transformer/transformer_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <limits>
#include <map>
#include <set>
#include <random>
#include <stdexcept>
#include <utility>

namespace zyron::model {
namespace {

Tensor copy_tensor(const Tensor& input) {
    return input.is_contiguous() ? input : input.contiguous();
}

Tensor tensor_from_scalar(float value) {
    Tensor t(Shape{});
    t[0] = value;
    return t;
}

void check_finite_positive(float value, const char* message) {
    if (!(value > 0.0f) || !std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}


Tensor softmax_last_dimension(const Tensor& input) {
    if (input.rank() == 0) throw std::invalid_argument("ZYRON QuantizedLanguageModel: softmax requires rank >= 1");
    const std::size_t width = input.shape()[input.rank() - 1];
    if (width == 0) throw std::invalid_argument("ZYRON QuantizedLanguageModel: softmax width must be > 0");
    Tensor output(input.shape());
    const std::size_t rows = input.size() / width;
    for (std::size_t row = 0; row < rows; ++row) {
        const float* src = input.data() + row * width;
        float* dst = output.data() + row * width;
        float max_value = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < width; ++i) max_value = std::max(max_value, src[i]);
        float sum = 0.0f;
        for (std::size_t i = 0; i < width; ++i) {
            dst[i] = std::exp(src[i] - max_value);
            sum += dst[i];
        }
        const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (std::size_t i = 0; i < width; ++i) dst[i] *= inverse;
    }
    return output;
}

struct Candidate {
    std::size_t id{0};
    float probability{0.0f};
};

void validate_generation_config(const GenerationConfig& config) {
    check_finite_positive(config.temperature,
                          "ZYRON QuantizedLanguageModel: invalid temperature");
    if (!(config.top_p > 0.0f) || config.top_p > 1.0f || !std::isfinite(config.top_p)) {
        throw std::invalid_argument("ZYRON QuantizedLanguageModel: invalid top_p");
    }
    if (!(config.repetition_penalty >= 1.0f) || !std::isfinite(config.repetition_penalty)) {
        throw std::invalid_argument("ZYRON QuantizedLanguageModel: invalid repetition_penalty");
    }
}

std::size_t sample_next(
    const Tensor& logits,
    std::size_t batch_index,
    std::size_t sequence_length,
    const GenerationConfig& config,
    const std::vector<std::size_t>& history,
    std::mt19937& rng) {

    const std::size_t vocab = logits.shape()[2];
    const std::size_t base = batch_index * sequence_length * vocab + (sequence_length - 1) * vocab;
    std::vector<Candidate> candidates;
    candidates.reserve(vocab);

    std::map<std::vector<std::size_t>, std::set<std::size_t>> seen_ngrams;
    if (config.no_repeat_ngram_size > 0 && history.size() >= config.no_repeat_ngram_size) {
        const auto n = config.no_repeat_ngram_size;
        const auto prefix_len = n - 1;
        for (std::size_t start = 0; start + n <= history.size(); ++start) {
            std::vector<std::size_t> prefix(history.begin() + static_cast<std::ptrdiff_t>(start),
                                             history.begin() + static_cast<std::ptrdiff_t>(start + prefix_len));
            seen_ngrams[prefix].insert(history[start + prefix_len]);
        }
    }
    std::vector<std::size_t> current_prefix;
    if (config.no_repeat_ngram_size > 0 && history.size() >= config.no_repeat_ngram_size - 1) {
        const auto prefix_len = config.no_repeat_ngram_size - 1;
        current_prefix.assign(history.end() - static_cast<std::ptrdiff_t>(prefix_len), history.end());
    }
    const auto blocked = seen_ngrams.find(current_prefix);

    float max_logit = -std::numeric_limits<float>::infinity();
    std::vector<float> values(vocab);
    for (std::size_t id = 0; id < vocab; ++id) {
        float value = logits[base + id];
        if (config.repetition_penalty != 1.0f &&
            std::find(history.begin(), history.end(), id) != history.end()) {
            value = value >= 0.0f
                ? value / config.repetition_penalty
                : value * config.repetition_penalty;
        }
        if (blocked != seen_ngrams.end() && blocked->second.count(id)) {
            value = -std::numeric_limits<float>::infinity();
        }
        values[id] = value;
        max_logit = std::max(max_logit, value);
    }
    bool any_finite = false;
    for (float value : values) any_finite = any_finite || std::isfinite(value);
    if (!any_finite && blocked != seen_ngrams.end()) {
        for (std::size_t id = 0; id < vocab; ++id) values[id] = logits[base + id];
        max_logit = *std::max_element(values.begin(), values.end());
    }

    float denominator = 0.0f;
    for (float value : values) denominator += std::exp(value - max_logit);
    if (!(denominator > 0.0f) || !std::isfinite(denominator)) return 0;

    for (std::size_t id = 0; id < vocab; ++id) {
        candidates.push_back({id, std::exp(values[id] - max_logit) / denominator});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.probability > b.probability;
              });

    if (config.top_k != 0 && config.top_k < candidates.size()) {
        candidates.resize(config.top_k);
    }
    if (config.top_p < 1.0f) {
        float cumulative = 0.0f;
        std::size_t keep = 0;
        for (const auto& candidate : candidates) {
            cumulative += candidate.probability;
            ++keep;
            if (cumulative >= config.top_p) break;
        }
        candidates.resize(std::max<std::size_t>(1, keep));
    }

    float probability_sum = 0.0f;
    for (const auto& candidate : candidates) probability_sum += candidate.probability;
    if (!(probability_sum > 0.0f)) return candidates.front().id;

    std::uniform_real_distribution<float> distribution(0.0f, probability_sum);
    const float draw = distribution(rng);
    float cumulative = 0.0f;
    for (const auto& candidate : candidates) {
        cumulative += candidate.probability;
        if (draw <= cumulative) return candidate.id;
    }
    return candidates.back().id;
}

} // namespace

QuantizedLanguageModel::QuantizedLanguageModel(
    const LanguageModel& source,
    quantization::Type type)
    : config_(source.config()),
      type_(type),
      token_embedding_(quantization::quantize(source.token_embedding().weight().value(), type)),
      positional_encoding_(config_.use_rope ? Tensor{} : Tensor(Shape{config_.max_sequence_length, config_.hidden_size})),
      final_norm_weight_(copy_tensor(source.final_norm().weight().value())),
      lm_head_(source.lm_head().effective_weight(),
               source.lm_head().bias() ? &source.lm_head().bias()->value() : nullptr,
               type) {

    config_.validate();
    if (source.lm_head().in_features() != config_.hidden_size ||
        source.lm_head().out_features() != config_.vocab_size) {
        throw std::invalid_argument("ZYRON QuantizedLanguageModel: source LM head shape mismatch");
    }

    if (!config_.use_rope) {
        constexpr float base = 10000.0f;
        const float hidden = static_cast<float>(config_.hidden_size);
        for (std::size_t pos = 0; pos < config_.max_sequence_length; ++pos) {
            for (std::size_t i = 0; i < config_.hidden_size; i += 2) {
                const float exponent = static_cast<float>(i) / hidden;
                const float angle = static_cast<float>(pos) / std::pow(base, exponent);
                positional_encoding_.at({pos, i}) = std::sin(angle);
                if (i + 1 < config_.hidden_size) positional_encoding_.at({pos, i + 1}) = std::cos(angle);
            }
        }
    }

    blocks_.reserve(source.transformer().num_layers());
    for (std::size_t index = 0; index < source.transformer().num_layers(); ++index) {
        const auto& src_block = source.transformer().block(index);
        Block block;

        const auto& norm1 = src_block.norm1();
        if (const auto* rms = dynamic_cast<const nn::RMSNorm*>(&norm1)) {
            block.norm1_rms = true;
            block.norm1_weight = copy_tensor(rms->weight().value());
        } else if (const auto* ln = dynamic_cast<const nn::LayerNorm*>(&norm1)) {
            block.norm1_rms = false;
            block.norm1_weight = copy_tensor(ln->weight().value());
            block.norm1_bias = copy_tensor(ln->bias().value());
        } else {
            throw std::runtime_error("ZYRON QuantizedLanguageModel: unsupported norm1 type");
        }

        const auto& norm2 = src_block.norm2();
        if (const auto* rms = dynamic_cast<const nn::RMSNorm*>(&norm2)) {
            block.norm2_rms = true;
            block.norm2_weight = copy_tensor(rms->weight().value());
        } else if (const auto* ln = dynamic_cast<const nn::LayerNorm*>(&norm2)) {
            block.norm2_rms = false;
            block.norm2_weight = copy_tensor(ln->weight().value());
            block.norm2_bias = copy_tensor(ln->bias().value());
        } else {
            throw std::runtime_error("ZYRON QuantizedLanguageModel: unsupported norm2 type");
        }

        const auto& attn = src_block.attention();
        const auto& ff = src_block.feed_forward();
        block.q_proj = quantization::QuantizedLinear(attn.q_proj().weight().value(),
            attn.q_proj().bias() ? &attn.q_proj().bias()->value() : nullptr, type);
        block.k_proj = quantization::QuantizedLinear(attn.k_proj().weight().value(),
            attn.k_proj().bias() ? &attn.k_proj().bias()->value() : nullptr, type);
        block.v_proj = quantization::QuantizedLinear(attn.v_proj().weight().value(),
            attn.v_proj().bias() ? &attn.v_proj().bias()->value() : nullptr, type);
        block.out_proj = quantization::QuantizedLinear(attn.out_proj().weight().value(),
            attn.out_proj().bias() ? &attn.out_proj().bias()->value() : nullptr, type);
        block.ff_up = quantization::QuantizedLinear(ff.up().weight().value(),
            ff.up().bias() ? &ff.up().bias()->value() : nullptr, type);
        block.ff_down = quantization::QuantizedLinear(ff.down().weight().value(),
            ff.down().bias() ? &ff.down().bias()->value() : nullptr, type);
        blocks_.push_back(std::move(block));
    }

    quantized_bytes_ = token_embedding_.bytes() + positional_encoding_.nbytes() + lm_head_.weight().bytes();
    for (const auto& block : blocks_) {
        quantized_bytes_ += block.q_proj.weight().bytes();
        quantized_bytes_ += block.k_proj.weight().bytes();
        quantized_bytes_ += block.v_proj.weight().bytes();
        quantized_bytes_ += block.out_proj.weight().bytes();
        quantized_bytes_ += block.ff_up.weight().bytes();
        quantized_bytes_ += block.ff_down.weight().bytes();
        quantized_bytes_ += block.norm1_weight.nbytes() + block.norm1_bias.nbytes();
        quantized_bytes_ += block.norm2_weight.nbytes() + block.norm2_bias.nbytes();
    }
    quantized_bytes_ += final_norm_weight_.nbytes();
    if (lm_head_.bias().size() != 0) quantized_bytes_ += lm_head_.bias().nbytes();
    float_reference_bytes_ = count_float_reference_bytes();
}

Tensor QuantizedLanguageModel::token_lookup(const std::vector<std::size_t>& token_ids) const {
    Tensor result(Shape{token_ids.size(), config_.hidden_size});
    for (std::size_t row = 0; row < token_ids.size(); ++row) {
        const std::size_t token = token_ids[row];
        if (token >= config_.vocab_size) throw std::out_of_range("ZYRON QuantizedLanguageModel: token id out of range");
        for (std::size_t hidden = 0; hidden < config_.hidden_size; ++hidden) {
            result.at({row, hidden}) =
                static_cast<float>(token_embedding_.value(hidden, token)) * token_embedding_.scales()[hidden];
        }
    }
    return result;
}

Tensor QuantizedLanguageModel::norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    bool rms) const {

    if (input.rank() < 2) throw std::invalid_argument("ZYRON QuantizedLanguageModel: norm expects rank >= 2");
    const std::size_t hidden = input.shape()[input.rank() - 1];
    if (weight.size() != hidden) throw std::invalid_argument("ZYRON QuantizedLanguageModel: norm weight mismatch");

    Tensor output(input.shape());
    const std::size_t rows = input.size() / hidden;
    for (std::size_t row = 0; row < rows; ++row) {
        const float* src = input.data() + row * hidden;
        float* dst = output.data() + row * hidden;
        float mean = 0.0f;
        if (!rms) {
            for (std::size_t i = 0; i < hidden; ++i) mean += src[i];
            mean /= static_cast<float>(hidden);
        }
        float variance = 0.0f;
        for (std::size_t i = 0; i < hidden; ++i) {
            const float centered = rms ? src[i] : src[i] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(hidden);
        const float inv = 1.0f / std::sqrt(variance + config_.norm_eps);
        for (std::size_t i = 0; i < hidden; ++i) {
            const float centered = rms ? src[i] : src[i] - mean;
            float value = centered * inv * weight[i];
            if (!rms) value += bias[i];
            dst[i] = value;
        }
    }
    return output;
}

Tensor QuantizedLanguageModel::gelu(const Tensor& input) const {
    Tensor output(input.shape());
    constexpr float kC = 0.044715f;
    constexpr float kInvSqrt2Pi = 0.7978845608028654f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        const float x3 = x * x * x;
        output[i] = 0.5f * x * (1.0f + std::tanh(kInvSqrt2Pi * (x + kC * x3)));
    }
    return output;
}

Tensor QuantizedLanguageModel::positional_encoding(std::size_t sequence_length) const {
    if (config_.use_rope) {
        throw std::logic_error("ZYRON QuantizedLanguageModel: sinusoidal positional encoding is disabled when RoPE is enabled");
    }
    if (sequence_length > config_.max_sequence_length) {
        throw std::out_of_range("ZYRON QuantizedLanguageModel: positional encoding exceeds context");
    }
    Tensor result(Shape{sequence_length, config_.hidden_size});
    for (std::size_t i = 0; i < sequence_length; ++i) {
        for (std::size_t j = 0; j < config_.hidden_size; ++j) result.at({i, j}) = positional_encoding_.at({i, j});
    }
    return result;
}

Tensor QuantizedLanguageModel::apply_rope(const Tensor& input, std::size_t start_position) const {
    if (input.rank() != 4) {
        throw std::invalid_argument("ZYRON QuantizedLanguageModel: RoPE expects [batch, heads, sequence, head_dim]");
    }
    const std::size_t sequence = input.shape()[2];
    const std::size_t head_dim = input.shape()[3];
    if ((head_dim % 2) != 0) {
        throw std::invalid_argument("ZYRON QuantizedLanguageModel: RoPE requires even head dimension");
    }
    if (start_position > config_.max_sequence_length ||
        sequence > config_.max_sequence_length - start_position) {
        throw std::out_of_range("ZYRON QuantizedLanguageModel: RoPE position exceeds context");
    }

    Tensor output(input.shape());
    const std::size_t batch = input.shape()[0];
    const std::size_t heads = input.shape()[1];
    const std::size_t half = head_dim / 2;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            for (std::size_t pos = 0; pos < sequence; ++pos) {
                const float absolute = static_cast<float>(start_position + pos);
                const std::size_t base_index = ((b * heads + h) * sequence + pos) * head_dim;
                for (std::size_t pair = 0; pair < half; ++pair) {
                    const float exponent = (2.0f * static_cast<float>(pair)) / static_cast<float>(head_dim);
                    const float angle = absolute / std::pow(config_.rope_theta, exponent);
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const float even = input[base_index + 2 * pair];
                    const float odd = input[base_index + 2 * pair + 1];
                    output[base_index + 2 * pair] = even * c - odd * s;
                    output[base_index + 2 * pair + 1] = even * s + odd * c;
                }
            }
        }
    }
    return output;
}

Tensor QuantizedLanguageModel::repeat_kv_heads(const Tensor& input, std::size_t repeats) const {
    if (input.rank() != 4) {
        throw std::invalid_argument("ZYRON QuantizedLanguageModel: repeat_kv_heads expects [batch, heads, sequence, head_dim]");
    }
    if (repeats == 0) throw std::invalid_argument("ZYRON QuantizedLanguageModel: repeats must be > 0");
    const std::size_t batch = input.shape()[0];
    const std::size_t heads = input.shape()[1];
    const std::size_t sequence = input.shape()[2];
    const std::size_t head_dim = input.shape()[3];
    if (heads > std::numeric_limits<std::size_t>::max() / repeats) {
        throw std::overflow_error("ZYRON QuantizedLanguageModel: KV head repeat overflow");
    }
    Tensor output(Shape{batch, heads * repeats, sequence, head_dim});
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t group = 0; group < repeats; ++group) {
                const std::size_t out_head = head * repeats + group;
                const std::size_t src = (b * heads + head) * sequence * head_dim;
                const std::size_t dst = (b * (heads * repeats) + out_head) * sequence * head_dim;
                std::copy_n(input.data() + src, sequence * head_dim, output.data() + dst);
            }
        }
    }
    return output;
}

Tensor QuantizedLanguageModel::forward_block(const Tensor& input, const Block& block) const {
    const std::size_t batch = input.shape()[0];
    const std::size_t sequence = input.shape()[1];
    const std::size_t hidden = input.shape()[2];
    const std::size_t heads = config_.num_heads;
    const std::size_t kv_heads = config_.resolved_num_kv_heads();
    const std::size_t head_dim = hidden / heads;

    const auto normalized = norm(input, block.norm1_weight, block.norm1_bias, block.norm1_rms);
    auto q = block.q_proj.forward(normalized).reshape(Shape{batch, sequence, heads, head_dim});
    auto k = block.k_proj.forward(normalized).reshape(Shape{batch, sequence, kv_heads, head_dim});
    auto v = block.v_proj.forward(normalized).reshape(Shape{batch, sequence, kv_heads, head_dim});
    q = q.transpose(1, 2).contiguous();
    k = k.transpose(1, 2).contiguous();
    v = v.transpose(1, 2).contiguous();

    if (config_.use_rope) {
        q = apply_rope(q, 0);
        k = apply_rope(k, 0);
    }
    if (kv_heads != heads) {
        const std::size_t repeats = heads / kv_heads;
        k = repeat_kv_heads(k, repeats);
        v = repeat_kv_heads(v, repeats);
    }

    const auto k_transposed = k.transpose(2, 3).contiguous();
    auto scores = math::matmul(q, k_transposed);
    scores = math::mul(scores, tensor_from_scalar(1.0f / std::sqrt(static_cast<float>(head_dim))));
    if (config_.causal) {
        Tensor mask(Shape{sequence, sequence});
        for (std::size_t i = 0; i < sequence; ++i) {
            for (std::size_t j = 0; j < sequence; ++j) {
                mask.at({i, j}) = j <= i ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }
        scores = math::add(scores, mask);
    }
    auto probabilities = softmax_last_dimension(scores);
    auto context = math::matmul(probabilities, v);
    context = context.transpose(1, 2).contiguous().reshape(Shape{batch, sequence, hidden});
    const auto attended = block.out_proj.forward(context);
    const auto residual = math::add(input, attended);

    const auto normalized2 = norm(residual, block.norm2_weight, block.norm2_bias, block.norm2_rms);
    auto feed = block.ff_up.forward(normalized2);
    feed = gelu(feed);
    feed = block.ff_down.forward(feed);
    return math::add(residual, feed);
}

Tensor QuantizedLanguageModel::forward(
    const std::vector<std::size_t>& token_ids) const {
    return forward(std::vector<std::vector<std::size_t>>{token_ids});
}

Tensor QuantizedLanguageModel::forward(
    const std::vector<std::vector<std::size_t>>& batch_token_ids) const {

    if (batch_token_ids.empty() || batch_token_ids.front().empty()) {
        throw std::invalid_argument("ZYRON QuantizedLanguageModel: empty input");
    }
    const std::size_t batch = batch_token_ids.size();
    const std::size_t sequence = batch_token_ids.front().size();
    if (sequence > config_.max_sequence_length) {
        throw std::invalid_argument("ZYRON QuantizedLanguageModel: sequence exceeds context");
    }

    std::vector<std::size_t> flat;
    flat.reserve(batch * sequence);
    for (const auto& ids : batch_token_ids) {
        if (ids.size() != sequence) throw std::invalid_argument("ZYRON QuantizedLanguageModel: inconsistent batch sequence lengths");
        flat.insert(flat.end(), ids.begin(), ids.end());
    }

    auto hidden = token_lookup(flat).reshape(Shape{batch, sequence, config_.hidden_size});
    if (!config_.use_rope) {
        hidden = math::add(hidden, positional_encoding(sequence));
    }
    for (const auto& block : blocks_) hidden = forward_block(hidden, block);
    hidden = norm(hidden, final_norm_weight_, Tensor{}, true);
    return lm_head_.forward(hidden);
}

std::vector<std::size_t> QuantizedLanguageModel::generate(
    const std::vector<std::size_t>& input_ids,
    const GenerationConfig& config) const {
    const auto batch = generate_batch({input_ids}, config);
    return batch.front();
}

std::vector<std::vector<std::size_t>> QuantizedLanguageModel::generate_batch(
    const std::vector<std::vector<std::size_t>>& input_ids,
    const GenerationConfig& config) const {

    validate_generation_config(config);
    if (input_ids.empty()) throw std::invalid_argument("ZYRON QuantizedLanguageModel: batch cannot be empty");
    const std::size_t prompt_length = input_ids.front().size();
    if (prompt_length == 0) throw std::invalid_argument("ZYRON QuantizedLanguageModel: empty prompt");
    for (const auto& ids : input_ids) {
        if (ids.size() != prompt_length) throw std::invalid_argument("ZYRON QuantizedLanguageModel: batch prompts must have equal lengths");
        if (ids.size() > config_.max_sequence_length) throw std::invalid_argument("ZYRON QuantizedLanguageModel: prompt exceeds context");
    }
    if (config.top_k > config_.vocab_size) throw std::invalid_argument("ZYRON QuantizedLanguageModel: top_k exceeds vocabulary");

    std::vector<std::vector<std::size_t>> tokens = input_ids;
    std::vector<bool> finished(tokens.size(), false);
    std::mt19937 rng(config.seed);
    for (std::size_t step = 0; step < config.max_new_tokens; ++step) {
        const auto logits = forward(tokens);
        bool any_active = false;
        std::vector<std::vector<std::size_t>> next_ids(tokens.size(), std::vector<std::size_t>(1, 0));
        for (std::size_t b = 0; b < tokens.size(); ++b) {
            if (finished[b]) {
                next_ids[b][0] = 0;
                continue;
            }
            next_ids[b][0] = sample_next(logits, b, logits.shape()[1], config, tokens[b], rng);
            any_active = true;
        }
        if (!any_active) break;
        for (std::size_t b = 0; b < tokens.size(); ++b) {
            if (!finished[b]) {
                tokens[b].push_back(next_ids[b][0]);
                if (next_ids[b][0] >= config_.vocab_size) throw std::runtime_error("ZYRON QuantizedLanguageModel: sampled token out of range");
            }
        }
        bool all_finished = true;
        for (std::size_t b = 0; b < tokens.size(); ++b) {
            if (!finished[b] && tokens[b].size() >= config_.max_sequence_length) finished[b] = true;
            all_finished = all_finished && finished[b];
        }
        if (all_finished) break;
    }
    return tokens;
}

std::string QuantizedLanguageModel::generate_text(
    const tokenizer::Tokenizer& tokenizer,
    const std::string& prompt,
    const GenerationConfig& config) const {
    return tokenizer.decode(generate(tokenizer.encode(prompt), config), true);
}

namespace {
constexpr std::uint32_t kQuantizedModelMagic = 0x4d4c5a51u; // QZLM
constexpr std::uint32_t kQuantizedModelVersion = 1;

void qwrite(std::ostream& out, const void* data, std::size_t bytes) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("ZYRON quantized model: write failure");
}
void qread(std::istream& in, void* data, std::size_t bytes) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("ZYRON quantized model: corrupted file");
}
void q_u32(std::ostream& out, std::uint32_t v) { qwrite(out, &v, sizeof(v)); }
void q_u64(std::ostream& out, std::uint64_t v) { qwrite(out, &v, sizeof(v)); }
std::uint32_t q_u32(std::istream& in) { std::uint32_t v{}; qread(in,&v,sizeof(v)); return v; }
std::uint64_t q_u64(std::istream& in) { std::uint64_t v{}; qread(in,&v,sizeof(v)); return v; }
void q_f32(std::ostream& out, float v) { qwrite(out,&v,sizeof(v)); }
float q_f32(std::istream& in) { float v{}; qread(in,&v,sizeof(v)); return v; }

void q_write_config(std::ostream& out, const transformer::Config& c) {
    q_u64(out,c.vocab_size); q_u64(out,c.hidden_size); q_u64(out,c.num_layers); q_u64(out,c.num_heads);
    q_u64(out,c.num_kv_heads); q_u64(out,c.intermediate_size); q_u64(out,c.max_sequence_length);
    q_f32(out,c.norm_eps); q_u32(out,c.use_rms_norm); q_u32(out,c.causal); q_u32(out,c.use_rope);
    q_u32(out,c.tie_word_embeddings); q_f32(out,c.rope_theta); q_f32(out,c.dropout); q_u32(out,c.use_qat); q_u64(out,c.qat_bits);
}
transformer::Config q_read_config(std::istream& in) {
    transformer::Config c;
    c.vocab_size=static_cast<std::size_t>(q_u64(in)); c.hidden_size=static_cast<std::size_t>(q_u64(in)); c.num_layers=static_cast<std::size_t>(q_u64(in)); c.num_heads=static_cast<std::size_t>(q_u64(in));
    c.num_kv_heads=static_cast<std::size_t>(q_u64(in)); c.intermediate_size=static_cast<std::size_t>(q_u64(in)); c.max_sequence_length=static_cast<std::size_t>(q_u64(in));
    c.norm_eps=q_f32(in); c.use_rms_norm=q_u32(in)!=0; c.causal=q_u32(in)!=0; c.use_rope=q_u32(in)!=0; c.tie_word_embeddings=q_u32(in)!=0;
    c.rope_theta=q_f32(in); c.dropout=q_f32(in); c.use_qat=q_u32(in)!=0; c.qat_bits=static_cast<std::size_t>(q_u64(in)); c.validate(); return c;
}
void qwrite_tensor(std::ostream& out, const Tensor& tensor) {
    if (tensor.size() == 0) {
        q_u64(out, 0);
        q_u64(out, 0);
        return;
    }
    const auto t=tensor.contiguous();
    q_u64(out,t.rank()); for(auto d:t.shape().dims()) q_u64(out,d); q_u64(out,t.size());
    qwrite(out,t.data(),t.size()*sizeof(float));
}
Tensor qread_tensor(std::istream& in) {
    const auto rank=q_u64(in); if(rank>16) throw std::runtime_error("ZYRON quantized model: invalid tensor rank");
    std::vector<std::size_t> dims; dims.reserve(static_cast<std::size_t>(rank)); for(std::size_t i=0;i<rank;++i) dims.push_back(static_cast<std::size_t>(q_u64(in)));
    const Shape shape(std::move(dims)); const auto size=q_u64(in);
    if(rank==0 && size==0) return Tensor{};
    if(size!=shape.size()) throw std::runtime_error("ZYRON quantized model: tensor size mismatch");
    Tensor t(shape); if(size) qread(in,t.data(),static_cast<std::size_t>(size)*sizeof(float)); return t;
}
void qwrite_linear(std::ostream& out, const quantization::QuantizedLinear& linear) { linear.weight().save(out); qwrite_tensor(out, linear.bias()); }
quantization::QuantizedLinear qread_linear(std::istream& in) {
    auto weight = quantization::QuantizedMatrix::load(in);
    auto bias = qread_tensor(in);
    return quantization::QuantizedLinear(std::move(weight), std::move(bias));
}

} // namespace

void QuantizedLanguageModel::save(const std::string& path) const {
    std::ofstream out(path,std::ios::binary); if(!out) throw std::runtime_error("ZYRON quantized model: failed to open: "+path);
    q_u32(out,kQuantizedModelMagic); q_u32(out,kQuantizedModelVersion); q_u32(out,static_cast<std::uint32_t>(type_)); q_write_config(out,config_);
    token_embedding_.save(out); qwrite_tensor(out,positional_encoding_); qwrite_tensor(out,final_norm_weight_);
    q_u64(out,blocks_.size());
    for(const auto& block:blocks_) {
        q_u32(out,block.norm1_rms); qwrite_tensor(out,block.norm1_weight); qwrite_tensor(out,block.norm1_bias); q_u32(out,block.norm2_rms); qwrite_tensor(out,block.norm2_weight); qwrite_tensor(out,block.norm2_bias);
        qwrite_linear(out,block.q_proj); qwrite_linear(out,block.k_proj); qwrite_linear(out,block.v_proj); qwrite_linear(out,block.out_proj); qwrite_linear(out,block.ff_up); qwrite_linear(out,block.ff_down);
    }
    qwrite_linear(out,lm_head_);
}

QuantizedLanguageModel QuantizedLanguageModel::load(const std::string& path) {
    std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("ZYRON quantized model: failed to open: "+path);
    if(q_u32(in)!=kQuantizedModelMagic || q_u32(in)!=kQuantizedModelVersion) throw std::runtime_error("ZYRON quantized model: unsupported format");
    QuantizedLanguageModel model(LoadTag{}); const auto type=q_u32(in); if(type!=1u&&type!=2u) throw std::runtime_error("ZYRON quantized model: invalid type"); model.type_=static_cast<quantization::Type>(type); model.config_=q_read_config(in);
    model.token_embedding_=quantization::QuantizedMatrix::load(in); model.positional_encoding_=qread_tensor(in); model.final_norm_weight_=qread_tensor(in);
    const auto blocks=q_u64(in); if(blocks!=model.config_.num_layers) throw std::runtime_error("ZYRON quantized model: layer count mismatch"); model.blocks_.reserve(static_cast<std::size_t>(blocks));
    for(std::uint64_t i=0;i<blocks;++i) { Block block; block.norm1_rms=q_u32(in)!=0; block.norm1_weight=qread_tensor(in); block.norm1_bias=qread_tensor(in); block.norm2_rms=q_u32(in)!=0; block.norm2_weight=qread_tensor(in); block.norm2_bias=qread_tensor(in); block.q_proj=qread_linear(in); block.k_proj=qread_linear(in); block.v_proj=qread_linear(in); block.out_proj=qread_linear(in); block.ff_up=qread_linear(in); block.ff_down=qread_linear(in); model.blocks_.push_back(std::move(block)); }
    model.lm_head_=qread_linear(in);
    model.quantized_bytes_ = model.token_embedding_.bytes() + model.positional_encoding_.nbytes() + model.final_norm_weight_.nbytes() + model.lm_head_.weight().bytes() + model.lm_head_.bias().nbytes();
    for (const auto& block : model.blocks_) {
        model.quantized_bytes_ += block.norm1_weight.nbytes() + block.norm1_bias.nbytes() + block.norm2_weight.nbytes() + block.norm2_bias.nbytes();
        model.quantized_bytes_ += block.q_proj.weight().bytes() + block.k_proj.weight().bytes() + block.v_proj.weight().bytes() + block.out_proj.weight().bytes() + block.ff_up.weight().bytes() + block.ff_down.weight().bytes();
    }
    model.float_reference_bytes_ = model.count_float_reference_bytes();
    return model;
}

double QuantizedLanguageModel::compression_ratio() const noexcept {
    if (quantized_bytes_ == 0) return 0.0;
    return static_cast<double>(float_reference_bytes_) / static_cast<double>(quantized_bytes_);
}

std::size_t QuantizedLanguageModel::count_float_reference_bytes() const noexcept {
    std::size_t total = token_embedding_.float_bytes() + positional_encoding_.nbytes() + final_norm_weight_.nbytes();
    total += lm_head_.weight().float_bytes();
    if (lm_head_.bias().size() != 0) total += lm_head_.bias().nbytes();
    for (const auto& block : blocks_) {
        total += block.q_proj.weight().float_bytes() + block.q_proj.bias().nbytes();
        total += block.k_proj.weight().float_bytes() + block.k_proj.bias().nbytes();
        total += block.v_proj.weight().float_bytes() + block.v_proj.bias().nbytes();
        total += block.out_proj.weight().float_bytes() + block.out_proj.bias().nbytes();
        total += block.ff_up.weight().float_bytes() + block.ff_up.bias().nbytes();
        total += block.ff_down.weight().float_bytes() + block.ff_down.bias().nbytes();
        total += block.norm1_weight.nbytes() + block.norm1_bias.nbytes();
        total += block.norm2_weight.nbytes() + block.norm2_bias.nbytes();
    }
    return total;
}

} // namespace zyron::model
