#include "zyron/nn/dropout.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace zyron::nn {

Dropout::Dropout(float probability, std::uint32_t seed)
    : probability_(probability), rng_(seed) {
    if (!(probability_ >= 0.0f) || probability_ >= 1.0f || !std::isfinite(probability_)) {
        throw std::invalid_argument("ZYRON Dropout: probability must be finite and in [0, 1)");
    }
}

autograd::Variable Dropout::forward(const autograd::Variable& input) {
    if (!training_ || probability_ == 0.0f) return input;

    Tensor mask(input.value().shape());
    const float keep_probability = 1.0f - probability_;
    const float scale = 1.0f / keep_probability;
    std::bernoulli_distribution keep(keep_probability);

    for (std::size_t i = 0; i < mask.size(); ++i) {
        mask[i] = keep(rng_) ? scale : 0.0f;
    }

    return autograd::mul(input, autograd::Variable(std::move(mask), false));
}

} // namespace zyron::nn
