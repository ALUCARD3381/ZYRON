#include "zyron/nn/normalization.hpp"

#include <stdexcept>

namespace zyron::nn {

LayerNorm::LayerNorm(std::size_t normalized_size, float eps)
    : normalized_size_(normalized_size),
      eps_(eps),
      weight_(Tensor::ones(Shape{normalized_size}), true),
      bias_(Tensor::zeros(Shape{normalized_size}), true) {

    if (normalized_size == 0) {
        throw std::invalid_argument(
            "ZYRON LayerNorm: normalized size must be > 0");
    }

    if (!(eps > 0.0f)) {
        throw std::invalid_argument(
            "ZYRON LayerNorm: eps must be > 0");
    }
}

autograd::Variable LayerNorm::forward(const autograd::Variable& input) {
    if (input.value().rank() == 0 ||
        input.value().shape()[input.value().rank() - 1] != normalized_size_) {
        throw std::invalid_argument(
            "ZYRON LayerNorm: input last dimension mismatch");
    }

    return autograd::layer_norm(input, weight_, bias_, eps_);
}

std::vector<autograd::Variable*> LayerNorm::parameters() {
    return {&weight_, &bias_};
}

RMSNorm::RMSNorm(std::size_t normalized_size, float eps)
    : normalized_size_(normalized_size),
      eps_(eps),
      weight_(Tensor::ones(Shape{normalized_size}), true) {

    if (normalized_size == 0) {
        throw std::invalid_argument(
            "ZYRON RMSNorm: normalized size must be > 0");
    }

    if (!(eps > 0.0f)) {
        throw std::invalid_argument(
            "ZYRON RMSNorm: eps must be > 0");
    }
}

autograd::Variable RMSNorm::forward(const autograd::Variable& input) {
    if (input.value().rank() == 0 ||
        input.value().shape()[input.value().rank() - 1] != normalized_size_) {
        throw std::invalid_argument(
            "ZYRON RMSNorm: input last dimension mismatch");
    }

    return autograd::rms_norm(input, weight_, eps_);
}

std::vector<autograd::Variable*> RMSNorm::parameters() {
    return {&weight_};
}

} // namespace zyron::nn
