#pragma once

#include "zyron/nn/layer.hpp"

#include <cstddef>
#include <cstdint>
#include <random>

namespace zyron::nn {

class Dropout final : public Layer {
public:
    explicit Dropout(float probability = 0.0f, std::uint32_t seed = 0);

    autograd::Variable forward(const autograd::Variable& input) override;
    std::vector<autograd::Variable*> parameters() override { return {}; }

    [[nodiscard]] float probability() const noexcept { return probability_; }

private:
    float probability_;
    std::mt19937 rng_;
};

} // namespace zyron::nn
