#pragma once

#include "zyron/core/tensor.hpp"
#include "zyron/quantization/quantization.hpp"

#include <cstddef>

namespace zyron::quantization {

class QuantizedLinear {
public:
    QuantizedLinear() = default;

    explicit QuantizedLinear(
        const Tensor& weight,
        const Tensor* bias = nullptr,
        Type type = Type::Int8);
    explicit QuantizedLinear(QuantizedMatrix weight, Tensor bias = Tensor{});

    [[nodiscard]] Tensor forward(const Tensor& input) const;

    [[nodiscard]] std::size_t in_features() const noexcept { return weight_.rows(); }
    [[nodiscard]] std::size_t out_features() const noexcept { return weight_.cols(); }
    [[nodiscard]] Type type() const noexcept { return weight_.type(); }
    [[nodiscard]] const QuantizedMatrix& weight() const noexcept { return weight_; }
    [[nodiscard]] const Tensor& bias() const noexcept { return bias_; }

private:
    QuantizedMatrix weight_;
    Tensor bias_;
};

} // namespace zyron::quantization
