#pragma once

#include "zyron/model/generation.hpp"
#include "zyron/nn/normalization.hpp"
#include "zyron/quantization/quantization.hpp"
#include "zyron/quantization/quantized_linear.hpp"
#include "zyron/tokenizer/tokenizer.hpp"
#include "zyron/transformer/config.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace zyron::model {

// Inference-only language model. Train in float32 with LanguageModel, then
// construct this object from it for INT8/INT4 weight-only inference.
class QuantizedLanguageModel {
public:
    QuantizedLanguageModel(
        const LanguageModel& source,
        quantization::Type type = quantization::Type::Int8);

    [[nodiscard]] Tensor forward(const std::vector<std::size_t>& token_ids) const;
    [[nodiscard]] Tensor forward(const std::vector<std::vector<std::size_t>>& batch_token_ids) const;

    [[nodiscard]] std::vector<std::size_t> generate(
        const std::vector<std::size_t>& input_ids,
        const GenerationConfig& config = {}) const;

    [[nodiscard]] std::vector<std::vector<std::size_t>> generate_batch(
        const std::vector<std::vector<std::size_t>>& input_ids,
        const GenerationConfig& config = {}) const;

    void save(const std::string& path) const;
    [[nodiscard]] static QuantizedLanguageModel load(const std::string& path);

    [[nodiscard]] std::string generate_text(
        const tokenizer::Tokenizer& tokenizer,
        const std::string& prompt,
        const GenerationConfig& config = {}) const;

    [[nodiscard]] const transformer::Config& config() const noexcept { return config_; }
    [[nodiscard]] quantization::Type type() const noexcept { return type_; }
    [[nodiscard]] std::size_t quantized_bytes() const noexcept { return quantized_bytes_; }
    [[nodiscard]] std::size_t float_reference_bytes() const noexcept { return float_reference_bytes_; }
    [[nodiscard]] double compression_ratio() const noexcept;

private:
    struct LoadTag {};
    explicit QuantizedLanguageModel(LoadTag) {}

    struct Block {
        bool norm1_rms{true};
        Tensor norm1_weight;
        Tensor norm1_bias;
        bool norm2_rms{true};
        Tensor norm2_weight;
        Tensor norm2_bias;
        quantization::QuantizedLinear q_proj;
        quantization::QuantizedLinear k_proj;
        quantization::QuantizedLinear v_proj;
        quantization::QuantizedLinear out_proj;
        quantization::QuantizedLinear ff_up;
        quantization::QuantizedLinear ff_down;
    };

    [[nodiscard]] Tensor token_lookup(const std::vector<std::size_t>& token_ids) const;
    [[nodiscard]] Tensor norm(const Tensor& input, const Tensor& weight,
                              const Tensor& bias, bool rms) const;
    [[nodiscard]] Tensor gelu(const Tensor& input) const;
    [[nodiscard]] Tensor positional_encoding(std::size_t sequence_length) const;
    [[nodiscard]] Tensor apply_rope(const Tensor& input, std::size_t start_position) const;
    [[nodiscard]] Tensor repeat_kv_heads(const Tensor& input, std::size_t repeats) const;
    [[nodiscard]] Tensor forward_block(const Tensor& input, const Block& block) const;
    [[nodiscard]] std::size_t count_float_reference_bytes() const noexcept;

    transformer::Config config_;
    quantization::Type type_;
    quantization::QuantizedMatrix token_embedding_;
    Tensor positional_encoding_;
    std::vector<Block> blocks_;
    Tensor final_norm_weight_;
    quantization::QuantizedLinear lm_head_;
    std::size_t quantized_bytes_{0};
    std::size_t float_reference_bytes_{0};
};

} // namespace zyron::model
