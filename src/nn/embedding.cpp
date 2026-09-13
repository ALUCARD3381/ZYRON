#include "zyron/nn/embedding.hpp"

#include <cmath>
#include <stdexcept>

namespace zyron::nn {

namespace {

float validated_init_range(float init_range) {
    if (!(init_range > 0.0f) || !std::isfinite(init_range)) {
        throw std::invalid_argument(
            "ZYRON Embedding: init_range must be finite and > 0");
    }
    return init_range;
}

} // namespace

Embedding::Embedding(
    std::size_t num_embeddings,
    std::size_t embedding_dim,
    std::uint32_t seed,
    float init_range)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      weight_(Tensor::random(
          Shape{num_embeddings, embedding_dim},
          -validated_init_range(init_range),
          validated_init_range(init_range),
          seed),
          true) {

    if (num_embeddings == 0 || embedding_dim == 0) {
        throw std::invalid_argument(
            "ZYRON Embedding: dimensions must be > 0");
    }
}

autograd::Variable Embedding::forward(
    const std::vector<std::size_t>& indices) {
    const auto effective_weight = qat_enabled_
        ? autograd::fake_quantize(weight_, qat_bits_)
        : weight_;
    return autograd::embedding(effective_weight, indices);
}

autograd::Variable Embedding::forward(
    const autograd::Variable& input) {

    if (input.value().rank() != 1) {
        throw std::invalid_argument(
            "ZYRON Embedding: tensor input must be rank-1 token ids");
    }

    std::vector<std::size_t> indices(input.value().size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const float raw = input.value()[i];
        if (!std::isfinite(raw) || raw < 0.0f ||
            std::floor(raw) != raw || raw >= static_cast<float>(num_embeddings_)) {
            throw std::invalid_argument(
                "ZYRON Embedding: token ids must be valid non-negative integers");
        }
        indices[i] = static_cast<std::size_t>(raw);
    }

    return forward(indices);
}

std::vector<autograd::Variable*> Embedding::parameters() {
    return {&weight_};
}

} // namespace zyron::nn
