#pragma once

#include "zyron/core/tensor.hpp"

#include <cstddef>

namespace zyron::math::kernels {

[[nodiscard]] bool optimized_matmul_available() noexcept;
[[nodiscard]] const char* matmul_backend() noexcept;

// Optimized 2D contiguous matrix multiplication. Output is [M,N].
// This function assumes all tensors are float32 and shapes are compatible.
void matmul_2d_contiguous(
    const Tensor& a,
    const Tensor& b,
    Tensor& out);

void matmul_batched_contiguous(
    const Tensor& a,
    const Tensor& b,
    Tensor& out);

} // namespace zyron::math::kernels
