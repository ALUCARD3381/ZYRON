#pragma once

#include "zyron/core/tensor.hpp"

namespace zyron::math {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& x);
Tensor sum(const Tensor& x);
Tensor mean(const Tensor& x);
Tensor max(const Tensor& x);
Tensor min(const Tensor& x);
Tensor exp(const Tensor& x);
Tensor log(const Tensor& x);
Tensor sqrt(const Tensor& x);
Tensor pow(const Tensor& x, float exponent);
Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace zyron::math
