#pragma once

#include "zyron/nn/layer.hpp"

#include <cstddef>
#include <cstdint>

namespace zyron::nn {

class LayerNorm final : public Layer {
public:
    explicit LayerNorm(
        std::size_t normalized_size,
        float eps = 1e-5f);

    autograd::Variable forward(const autograd::Variable& input) override;
    std::vector<autograd::Variable*> parameters() override;

    [[nodiscard]] std::size_t normalized_size() const noexcept { return normalized_size_; }
    [[nodiscard]] float eps() const noexcept { return eps_; }
    [[nodiscard]] autograd::Variable& weight() noexcept { return weight_; }
    [[nodiscard]] const autograd::Variable& weight() const noexcept { return weight_; }
    [[nodiscard]] autograd::Variable& bias() noexcept { return bias_; }
    [[nodiscard]] const autograd::Variable& bias() const noexcept { return bias_; }

private:
    std::size_t normalized_size_;
    float eps_;
    autograd::Variable weight_;
    autograd::Variable bias_;
};

class RMSNorm final : public Layer {
public:
    explicit RMSNorm(
        std::size_t normalized_size,
        float eps = 1e-5f);

    autograd::Variable forward(const autograd::Variable& input) override;
    std::vector<autograd::Variable*> parameters() override;

    [[nodiscard]] std::size_t normalized_size() const noexcept { return normalized_size_; }
    [[nodiscard]] float eps() const noexcept { return eps_; }
    [[nodiscard]] autograd::Variable& weight() noexcept { return weight_; }
    [[nodiscard]] const autograd::Variable& weight() const noexcept { return weight_; }

private:
    std::size_t normalized_size_;
    float eps_;
    autograd::Variable weight_;
};

} // namespace zyron::nn
