#include "zyron/transformer/transformer_block.hpp"

#include "zyron/autograd/autograd.hpp"

#include <stdexcept>
#include <utility>

namespace zyron::transformer {

TransformerBlock::TransformerBlock(
    const Config& config,
    std::uint32_t seed)
    : attention_(config, seed + 10),
      feed_forward_(config, seed + 20) {

    config.validate();

    if (config.use_rms_norm) {
        norm1_ = std::make_unique<nn::RMSNorm>(
            config.hidden_size,
            config.norm_eps);
        norm2_ = std::make_unique<nn::RMSNorm>(
            config.hidden_size,
            config.norm_eps);
    } else {
        norm1_ = std::make_unique<nn::LayerNorm>(
            config.hidden_size,
            config.norm_eps);
        norm2_ = std::make_unique<nn::LayerNorm>(
            config.hidden_size,
            config.norm_eps);
    }
}

autograd::Variable TransformerBlock::forward(
    const autograd::Variable& input) {

    auto normalized = norm1_->forward(input);
    auto attended = attention_.forward(normalized);
    auto residual = autograd::add(input, attended);

    normalized = norm2_->forward(residual);
    auto feed_forward_output = feed_forward_.forward(normalized);
    return autograd::add(residual, feed_forward_output);
}

autograd::Variable TransformerBlock::forward_cached(
    const autograd::Variable& input,
    MultiHeadAttention::KVCache& cache) {

    auto normalized = norm1_->forward(input);
    auto attended = attention_.forward_cached(normalized, cache);
    auto residual = autograd::add(input, attended);

    normalized = norm2_->forward(residual);
    auto feed_forward_output = feed_forward_.forward(normalized);
    return autograd::add(residual, feed_forward_output);
}

std::vector<autograd::Variable*> TransformerBlock::parameters() {
    auto result = norm1_->parameters();
    const auto attention_params = attention_.parameters();
    const auto norm2_params = norm2_->parameters();
    const auto ff_params = feed_forward_.parameters();

    result.insert(result.end(), attention_params.begin(), attention_params.end());
    result.insert(result.end(), norm2_params.begin(), norm2_params.end());
    result.insert(result.end(), ff_params.begin(), ff_params.end());
    return result;
}

void TransformerBlock::set_qat(bool enabled, std::size_t bits) noexcept {
    attention_.set_qat(enabled, bits);
    feed_forward_.set_qat(enabled, bits);
}

void TransformerBlock::train(bool mode) noexcept {
    Layer::train(mode);
    attention_.train(mode);
    feed_forward_.train(mode);
    norm1_->train(mode);
    norm2_->train(mode);
}

nn::Layer& TransformerBlock::norm1() noexcept {
    return *norm1_;
}

nn::Layer& TransformerBlock::norm2() noexcept {
    return *norm2_;
}

const nn::Layer& TransformerBlock::norm1() const noexcept {
    return *norm1_;
}

const nn::Layer& TransformerBlock::norm2() const noexcept {
    return *norm2_;
}

} // namespace zyron::transformer
