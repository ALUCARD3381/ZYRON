#include "zyron/math/kernels.hpp"

#include "zyron/core/runtime.hpp"

#include <algorithm>
#include <cstddef>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace zyron::math::kernels {
namespace {

constexpr std::size_t kTile = 64;

template <typename APtr, typename BPtr, typename CPtr>
void compute_row_block(
    APtr a_row,
    BPtr B,
    CPtr c_row,
    std::size_t k,
    std::size_t n,
    std::size_t j_begin,
    std::size_t j_end) {

    for (std::size_t j0 = j_begin; j0 < j_end; j0 += kTile) {
        const std::size_t j_limit = std::min(n, j0 + kTile);

#if defined(__ARM_NEON)
        if (runtime::cpu_features().neon) {
            std::size_t j = j0;
            for (; j + 4 <= j_limit; j += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (std::size_t kk = 0; kk < k; ++kk) {
                    const float32x4_t bv = vld1q_f32(B + kk * n + j);
                    acc = vmlaq_n_f32(acc, bv, a_row[kk]);
                }
                vst1q_f32(c_row + j, acc);
            }
            for (; j < j_limit; ++j) {
                float value = 0.0f;
                for (std::size_t kk = 0; kk < k; ++kk) {
                    value += a_row[kk] * B[kk * n + j];
                }
                c_row[j] = value;
            }
            continue;
        }
#endif

#if defined(__SSE2__)
        std::size_t j = j0;
        for (; j + 4 <= j_limit; j += 4) {
            __m128 acc = _mm_setzero_ps();
            for (std::size_t kk = 0; kk < k; ++kk) {
                const __m128 bv = _mm_loadu_ps(B + kk * n + j);
                const __m128 av = _mm_set1_ps(a_row[kk]);
                acc = _mm_add_ps(acc, _mm_mul_ps(av, bv));
            }
            _mm_storeu_ps(c_row + j, acc);
        }
        for (; j < j_limit; ++j) {
            float value = 0.0f;
            for (std::size_t kk = 0; kk < k; ++kk) {
                value += a_row[kk] * B[kk * n + j];
            }
            c_row[j] = value;
        }
#else
        for (std::size_t j = j0; j < j_limit; ++j) {
            float value = 0.0f;
            for (std::size_t kk = 0; kk < k; ++kk) {
                value += a_row[kk] * B[kk * n + j];
            }
            c_row[j] = value;
        }
#endif
    }
}

[[maybe_unused]] void scalar_rows(
    const float* A,
    const float* B,
    float* C,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t k,
    std::size_t n) {

    for (std::size_t i = row_begin; i < row_end; ++i) {
        const float* a_row = A + i * k;
        float* c_row = C + i * n;
        for (std::size_t j = 0; j < n; ++j) {
            float value = 0.0f;
            for (std::size_t kk = 0; kk < k; ++kk) {
                value += a_row[kk] * B[kk * n + j];
            }
            c_row[j] = value;
        }
    }
}

void optimized_rows(
    const float* A,
    const float* B,
    float* C,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t k,
    std::size_t n) {

    for (std::size_t i = row_begin; i < row_end; ++i) {
        const float* a_row = A + i * k;
        float* c_row = C + i * n;
        compute_row_block(a_row, B, c_row, k, n, 0, n);
    }
}

} // namespace

bool optimized_matmul_available() noexcept {
    const auto features = runtime::cpu_features();
#if defined(__ARM_NEON)
    return features.neon;
#elif defined(__SSE2__)
    return features.sse2;
#else
    return false;
#endif
}

const char* matmul_backend() noexcept {
    const auto features = runtime::cpu_features();
#if defined(__ARM_NEON)
    return features.neon ? "ARMv7 NEON" : "ARMv7 scalar";
#elif defined(__SSE2__)
    return features.sse2 ? "x86 SSE2" : "scalar";
#else
    return "scalar";
#endif
}

void matmul_2d_contiguous(
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {

    const std::size_t m = a.shape()[0];
    const std::size_t k = a.shape()[1];
    const std::size_t n = b.shape()[1];

    const std::size_t min_grain = 1;
    runtime::parallel_for(
        0,
        m,
        [&](std::size_t begin, std::size_t end) {
#if defined(__ARM_NEON)
            if (runtime::cpu_features().neon) {
                optimized_rows(a.data(), b.data(), out.data(), begin, end, k, n);
            } else {
                scalar_rows(a.data(), b.data(), out.data(), begin, end, k, n);
            }
#elif defined(__SSE2__)
            optimized_rows(a.data(), b.data(), out.data(), begin, end, k, n);
#else
            scalar_rows(a.data(), b.data(), out.data(), begin, end, k, n);
#endif
        },
        min_grain);
}

void matmul_batched_contiguous(
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {

    const std::size_t rank = a.rank();
    const std::size_t m = a.shape()[rank - 2];
    const std::size_t k = a.shape()[rank - 1];
    const std::size_t n = b.shape()[rank - 1];
    const std::size_t rows_per_batch = m;
    const std::size_t batch_count = a.size() / (m * k);
    const std::size_t total_rows = batch_count * rows_per_batch;

    runtime::parallel_for(
        0,
        total_rows,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t global_row = begin; global_row < end; ++global_row) {
                const std::size_t batch = global_row / rows_per_batch;
                const std::size_t row = global_row % rows_per_batch;
                const float* A = a.data() + batch * m * k;
                const float* B = b.data() + batch * k * n;
                float* C = out.data() + batch * m * n;
#if defined(__ARM_NEON)
                if (runtime::cpu_features().neon) {
                    compute_row_block(A + row * k, B, C + row * n, k, n, 0, n);
                } else {
                    for (std::size_t j = 0; j < n; ++j) {
                        float value = 0.0f;
                        for (std::size_t kk = 0; kk < k; ++kk) value += A[row * k + kk] * B[kk * n + j];
                        C[row * n + j] = value;
                    }
                }
#elif defined(__SSE2__)
                compute_row_block(A + row * k, B, C + row * n, k, n, 0, n);
#else
                for (std::size_t j = 0; j < n; ++j) {
                    float value = 0.0f;
                    for (std::size_t kk = 0; kk < k; ++kk) value += A[row * k + kk] * B[kk * n + j];
                    C[row * n + j] = value;
                }
#endif
            }
        },
        1);
}

} // namespace zyron::math::kernels
