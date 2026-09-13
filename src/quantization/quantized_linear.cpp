#include "zyron/quantization/quantized_linear.hpp"

#include <stdexcept>
#include <utility>

namespace zyron::quantization {

QuantizedLinear::QuantizedLinear(QuantizedMatrix weight, Tensor bias)
    : weight_(std::move(weight)), bias_(std::move(bias)) {
    if (weight_.empty()) throw std::invalid_argument("ZYRON QuantizedLinear: weight cannot be empty");
    if (bias_.size() != 0 && (bias_.rank() != 1 || bias_.size() != weight_.cols())) {
        throw std::invalid_argument("ZYRON QuantizedLinear: bias shape mismatch");
    }
}

QuantizedLinear::QuantizedLinear(
    const Tensor& weight,
    const Tensor* bias,
    Type type)
    : weight_(quantize(weight, type)) {

    if (weight.rank() != 2) {
        throw std::invalid_argument("ZYRON QuantizedLinear: weight must be rank-2");
    }
    if (bias) {
        if (bias->rank() != 1 || bias->shape()[0] != weight.shape()[1]) {
            throw std::invalid_argument("ZYRON QuantizedLinear: bias shape mismatch");
        }
        bias_ = *bias;
    }
}

Tensor QuantizedLinear::forward(const Tensor& input) const {
    if (input.rank() < 2) {
        throw std::invalid_argument("ZYRON QuantizedLinear: input rank must be >= 2");
    }
    if (input.shape()[input.rank() - 1] != in_features()) {
        throw std::invalid_argument("ZYRON QuantizedLinear: input feature mismatch");
    }

    auto output = matmul(input, weight_);
    if (bias_.size() == 0) return output;

    const std::size_t n = out_features();
    for (std::size_t row = 0; row < output.size() / n; ++row) {
        float* dst = output.data() + row * n;
        for (std::size_t column = 0; column < n; ++column) dst[column] += bias_[column];
    }
    return output;
}

} // namespace zyron::quantization
