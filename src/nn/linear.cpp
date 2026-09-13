#include "zyron/nn/linear.hpp"

#include "zyron/math/ops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace zyron::nn {

namespace {

float xavier_limit(std::size_t in_features, std::size_t out_features) {
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument(
            "ZYRON Linear: feature sizes must be > 0");
    }

    if (in_features > std::numeric_limits<std::size_t>::max() - out_features) {
        throw std::overflow_error(
            "ZYRON Linear: feature count overflow");
    }

    const auto total = in_features + out_features;
    return std::sqrt(6.0f / static_cast<float>(total));
}

} // namespace

Linear::Linear(
    std::size_t in_features,
    std::size_t out_features,
    bool use_bias,
    std::uint32_t seed)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_(Tensor::random(
          Shape{in_features, out_features},
          -xavier_limit(in_features, out_features),
          xavier_limit(in_features, out_features),
          seed),
          true),
      bias_(Tensor::zeros(Shape{out_features}), true) {}

Linear::Linear(
    std::size_t in_features,
    std::size_t out_features,
    autograd::Variable* shared_weight,
    bool transpose_weight,
    bool use_bias,
    std::uint32_t seed)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_(shared_weight
          ? autograd::Variable{}
          : autograd::Variable(
                Tensor::random(
                    Shape{in_features, out_features},
                    -xavier_limit(in_features, out_features),
                    xavier_limit(in_features, out_features),
                    seed),
                true)),
      bias_(Tensor::zeros(Shape{out_features}), true),
      tied_weight_(shared_weight),
      transpose_tied_weight_(shared_weight ? transpose_weight : false) {

    if (in_features_ == 0 || out_features_ == 0) {
        throw std::invalid_argument(
            "ZYRON Linear: feature sizes must be > 0");
    }
    if (!shared_weight) return;

    const std::size_t expected_rows = transpose_weight ? out_features_ : in_features_;
    const std::size_t expected_cols = transpose_weight ? in_features_ : out_features_;
    if (shared_weight->value().rank() != 2 ||
        shared_weight->value().shape() != Shape{expected_rows, expected_cols}) {
        throw std::invalid_argument("ZYRON Linear: tied weight shape mismatch");
    }
    if (!shared_weight->requires_grad()) {
        throw std::invalid_argument("ZYRON Linear: tied weight must require gradients");
    }
}

Linear::Linear(
    std::size_t in_features,
    std::size_t out_features,
    autograd::Variable& shared_weight,
    bool transpose_weight,
    bool use_bias)
    : Linear(
          in_features,
          out_features,
          &shared_weight,
          transpose_weight,
          use_bias,
          0) {}

autograd::Variable Linear::forward(const autograd::Variable& input) {
    if (input.value().rank() < 2) {
        throw std::invalid_argument(
            "ZYRON Linear: forward expects rank >= 2 input");
    }

    const std::size_t last_dim = input.value().shape()[input.value().rank() - 1];
    if (last_dim != in_features_) {
        throw std::invalid_argument(
            "ZYRON Linear: input feature dimension mismatch");
    }

    const auto effective_weight = [&]() -> autograd::Variable {
        const auto* source = tied_weight_ ? tied_weight_ : &weight_;
        autograd::Variable result = qat_enabled_
            ? autograd::fake_quantize(*source, qat_bits_)
            : *source;
        return transpose_tied_weight_
            ? autograd::transpose(result, 0, 1)
            : result;
    };

    if (input.value().rank() == 2) {
        auto output = autograd::matmul(input, effective_weight());
        if (use_bias_) output = autograd::add(output, bias_);
        return output;
    }

    const std::size_t outer = input.value().size() / in_features_;
    const auto flat_shape = Shape{outer, in_features_};
    const auto output_shape = [&] {
        auto dims = input.value().shape().dims();
        dims.back() = out_features_;
        return Shape(std::move(dims));
    }();

    auto flat = autograd::reshape(input, flat_shape);
    auto flat_output = autograd::matmul(flat, effective_weight());
    if (use_bias_) flat_output = autograd::add(flat_output, bias_);
    return autograd::reshape(flat_output, output_shape);
}

std::vector<autograd::Variable*> Linear::parameters() {
    autograd::Variable* parameter = tied_weight_ ? tied_weight_ : &weight_;
    if (use_bias_) return {parameter, &bias_};
    return {parameter};
}

Tensor Linear::effective_weight() const {
    if (!tied_weight_) return weight_.value();
    if (!transpose_tied_weight_) return tied_weight_->value();
    return tied_weight_->value().transpose(0, 1).contiguous();
}

void Linear::tie_weight(autograd::Variable& shared_weight, bool transpose) {
    if (shared_weight.value().rank() != 2) {
        throw std::invalid_argument("ZYRON Linear: tied weight must be rank-2");
    }
    const std::size_t expected_rows = transpose ? out_features_ : in_features_;
    const std::size_t expected_cols = transpose ? in_features_ : out_features_;
    if (shared_weight.value().shape() != Shape{expected_rows, expected_cols}) {
        throw std::invalid_argument("ZYRON Linear: tied weight shape mismatch");
    }
    if (!shared_weight.requires_grad()) {
        throw std::invalid_argument("ZYRON Linear: tied weight must require gradients");
    }
    tied_weight_ = &shared_weight;
    transpose_tied_weight_ = transpose;
}

autograd::Variable* Linear::bias() noexcept {
    return use_bias_ ? &bias_ : nullptr;
}

const autograd::Variable* Linear::bias() const noexcept {
    return use_bias_ ? &bias_ : nullptr;
}

} // namespace zyron::nn
