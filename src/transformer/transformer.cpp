#include "zyron/transformer/transformer.hpp"

#include "zyron/autograd/autograd.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace zyron::transformer {

namespace {

Config validated_config(Config config) {
    config.validate();
    return config;
}

} // namespace

Transformer::Transformer(
    const Config& config,
    std::uint32_t seed)
    : config_(validated_config(config)) {

    if (!config_.use_rope) {
        positional_encoding_ = Tensor(Shape{config_.max_sequence_length, config_.hidden_size});
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

    blocks_.reserve(config_.num_layers);
    for (std::size_t i = 0; i < config_.num_layers; ++i) {
        blocks_.push_back(std::make_unique<TransformerBlock>(
            config_,
            seed + static_cast<std::uint32_t>(i * 100)));
    }
}

Tensor Transformer::positional_encoding(
    std::size_t start_position,
    std::size_t sequence_length) const {

    if (config_.use_rope) {
        throw std::logic_error("ZYRON Transformer: sinusoidal positional encoding is disabled when RoPE is enabled");
    }
    if (start_position > config_.max_sequence_length ||
        sequence_length > config_.max_sequence_length - start_position) {
        throw std::out_of_range(
            "ZYRON Transformer: positional encoding range outside configured context");
    }

    Tensor result(Shape{sequence_length, config_.hidden_size});
    for (std::size_t i = 0; i < sequence_length; ++i) {
        for (std::size_t j = 0; j < config_.hidden_size; ++j) {
            result.at({i, j}) = positional_encoding_.at({start_position + i, j});
        }
    }
    return result;
}

autograd::Variable Transformer::forward(
    const autograd::Variable& input) {

    if (input.value().rank() != 3) {
        throw std::invalid_argument(
            "ZYRON Transformer: expected [batch, sequence, hidden]");
    }

    const std::size_t sequence = input.value().shape()[1];
    const std::size_t hidden = input.value().shape()[2];

    if (hidden != config_.hidden_size) {
        throw std::invalid_argument(
            "ZYRON Transformer: hidden size mismatch");
    }
    if (sequence == 0 || sequence > config_.max_sequence_length) {
        throw std::invalid_argument(
            "ZYRON Transformer: sequence length outside configured range");
    }

    auto hidden_states = config_.use_rope
        ? input
        : autograd::add(input, autograd::Variable(positional_encoding(0, sequence), false));

    for (auto& block_ptr : blocks_) {
        hidden_states = block_ptr->forward(hidden_states);
    }

    return hidden_states;
}

autograd::Variable Transformer::forward_cached(
    const autograd::Variable& input,
    KVCache& cache) {

    if (!autograd::NoGradGuard::enabled()) {
        throw std::invalid_argument(
            "ZYRON Transformer: forward_cached requires NoGradGuard");
    }
    if (input.value().rank() != 3) {
        throw std::invalid_argument(
            "ZYRON Transformer: cached forward expects [batch, sequence, hidden]");
    }

    const std::size_t batch = input.value().shape()[0];
    const std::size_t sequence = input.value().shape()[1];
    const std::size_t hidden = input.value().shape()[2];

    if (hidden != config_.hidden_size) {
        throw std::invalid_argument(
            "ZYRON Transformer: hidden size mismatch");
    }
    if (sequence == 0) {
        throw std::invalid_argument(
            "ZYRON Transformer: sequence length must be > 0");
    }

    if (!cache.initialized) {
        cache.layers.clear();
        cache.layers.resize(blocks_.size());
        cache.sequence_length = 0;
        cache.batch_size = batch;
        cache.initialized = true;
    } else if (cache.batch_size != batch) {
        throw std::invalid_argument(
            "ZYRON Transformer: cached batch size mismatch");
    }

    if (cache.sequence_length > config_.max_sequence_length) {
        throw std::out_of_range(
            "ZYRON Transformer: invalid cached sequence length");
    }
    if (sequence > config_.max_sequence_length - cache.sequence_length) {
        throw std::out_of_range(
            "ZYRON Transformer: cached sequence exceeds configured context length");
    }

    const std::size_t position_offset = cache.sequence_length;
    auto hidden_states = config_.use_rope
        ? input
        : autograd::add(input, autograd::Variable(positional_encoding(position_offset, sequence), false));

    for (std::size_t index = 0; index < blocks_.size(); ++index) {
        hidden_states = blocks_[index]->forward_cached(
            hidden_states,
            cache.layers[index]);
    }

    cache.sequence_length += sequence;
    return hidden_states;
}

std::vector<autograd::Variable*> Transformer::parameters() {
    std::vector<autograd::Variable*> result;
    for (auto& block_ptr : blocks_) {
        const auto block_params = block_ptr->parameters();
        result.insert(result.end(), block_params.begin(), block_params.end());
    }
    return result;
}

void Transformer::set_qat(bool enabled, std::size_t bits) noexcept {
    for (auto& block_ptr : blocks_) block_ptr->set_qat(enabled, bits);
}

void Transformer::train(bool mode) noexcept {
    Layer::train(mode);
    for (auto& block_ptr : blocks_) block_ptr->train(mode);
}

TransformerBlock& Transformer::block(std::size_t index) {
    if (index >= blocks_.size()) {
        throw std::out_of_range("ZYRON Transformer: block index out of range");
    }
    return *blocks_[index];
}

const TransformerBlock& Transformer::block(std::size_t index) const {
    if (index >= blocks_.size()) {
        throw std::out_of_range("ZYRON Transformer: block index out of range");
    }
    return *blocks_[index];
}

} // namespace zyron::transformer
