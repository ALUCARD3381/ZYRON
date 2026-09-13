#pragma once

#include "zyron/nn/layer.hpp"

#include <cstddef>
#include <cstdint>

namespace zyron::nn {

class Linear final : public Layer {
public:
    Linear(
        std::size_t in_features,
        std::size_t out_features,
        bool use_bias = true,
        std::uint32_t seed = 0);

    // Constructs a linear layer optionally tied to another module's weight.
    // When shared_weight is non-null, no private weight allocation is performed.
    Linear(
        std::size_t in_features,
        std::size_t out_features,
        autograd::Variable* shared_weight,
        bool transpose_weight,
        bool use_bias = true,
        std::uint32_t seed = 0);

    Linear(
        std::size_t in_features,
        std::size_t out_features,
        autograd::Variable& shared_weight,
        bool transpose_weight,
        bool use_bias = true);

    autograd::Variable forward(const autograd::Variable& input) override;
    std::vector<autograd::Variable*> parameters() override;

    [[nodiscard]] std::size_t in_features() const noexcept { return in_features_; }
    [[nodiscard]] std::size_t out_features() const noexcept { return out_features_; }
    [[nodiscard]] bool has_bias() const noexcept { return use_bias_; }

    [[nodiscard]] autograd::Variable& weight() noexcept {
        return tied_weight_ ? *tied_weight_ : weight_;
    }
    [[nodiscard]] const autograd::Variable& weight() const noexcept {
        return tied_weight_ ? *tied_weight_ : weight_;
    }
    [[nodiscard]] Tensor effective_weight() const;
    [[nodiscard]] autograd::Variable* bias() noexcept;
    [[nodiscard]] const autograd::Variable* bias() const noexcept;
    void tie_weight(autograd::Variable& shared_weight, bool transpose = false);
    [[nodiscard]] bool is_weight_tied() const noexcept { return tied_weight_ != nullptr; }
    void set_qat(bool enabled, std::size_t bits = 8) noexcept { qat_enabled_ = enabled; qat_bits_ = bits; }
    [[nodiscard]] bool qat_enabled() const noexcept { return qat_enabled_; }
    [[nodiscard]] std::size_t qat_bits() const noexcept { return qat_bits_; }

private:
    std::size_t in_features_;
    std::size_t out_features_;
    bool use_bias_;
    autograd::Variable weight_;
    autograd::Variable bias_;
    autograd::Variable* tied_weight_{nullptr};
    bool transpose_tied_weight_{false};
    bool qat_enabled_{false};
    std::size_t qat_bits_{8};
};

} // namespace zyron::nn
