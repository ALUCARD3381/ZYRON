#include "zyron/math/ops.hpp"
#include "zyron/math/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace zyron::math {

namespace {
using BinaryFn = float (*)(float, float);

Shape broadcast_shape(const Shape& a, const Shape& b) {
    const std::size_t rank = std::max(a.rank(), b.rank());
    std::vector<std::size_t> dims(rank, 1);

    for (std::size_t out_dim = 0; out_dim < rank; ++out_dim) {
        const std::size_t a_offset = rank - a.rank();
        const std::size_t b_offset = rank - b.rank();
        const std::size_t a_dim = out_dim < a_offset ? 1 : a[out_dim - a_offset];
        const std::size_t b_dim = out_dim < b_offset ? 1 : b[out_dim - b_offset];

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument("ZYRON math: incompatible shapes for broadcasting");
        }
        dims[out_dim] = std::max(a_dim, b_dim);
    }
    return Shape(std::move(dims));
}

std::vector<std::size_t> broadcast_strides(const Tensor& input, const Shape& output_shape) {
    const std::size_t rank = output_shape.rank();
    const std::size_t offset = rank - input.rank();
    std::vector<std::size_t> result(rank, 0);

    for (std::size_t dim = 0; dim < input.rank(); ++dim) {
        const std::size_t out_dim = dim + offset;
        result[out_dim] = input.shape()[dim] == 1 ? 0 : input.strides()[dim];
    }
    return result;
}

std::size_t broadcast_offset(std::size_t linear_index, const Shape& shape, const std::vector<std::size_t>& strides) {
    std::size_t physical_offset = 0;
    for (std::size_t dim = shape.rank(); dim-- > 0;) {
        const std::size_t index = linear_index % shape[dim];
        linear_index /= shape[dim];
        physical_offset += index * strides[dim];
    }
    return physical_offset;
}

Tensor binary_op(const Tensor& a, const Tensor& b, BinaryFn fn) {
    const Shape out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor out(out_shape);

    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = fn(a.data()[i], b.data()[i]);
        return out;
    }

    const auto a_strides = broadcast_strides(a, out_shape);
    const auto b_strides = broadcast_strides(b, out_shape);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = fn(a.data()[broadcast_offset(i, out_shape, a_strides)],
                    b.data()[broadcast_offset(i, out_shape, b_strides)]);
    }
    return out;
}

float fadd(float a, float b) { return a + b; }
float fsub(float a, float b) { return a - b; }
float fmul(float a, float b) { return a * b; }
float fdiv(float a, float b) { return a / b; }
}

Tensor add(const Tensor& a, const Tensor& b) { return binary_op(a, b, fadd); }
Tensor sub(const Tensor& a, const Tensor& b) { return binary_op(a, b, fsub); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary_op(a, b, fmul); }
Tensor div(const Tensor& a, const Tensor& b) { return binary_op(a, b, fdiv); }

Tensor neg(const Tensor& x) {
    Tensor out(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = -x[i];
    return out;
}

Tensor sum(const Tensor& x) {
    if (x.size() == 0) throw std::invalid_argument("ZYRON math: sum of empty tensor");
    Tensor out(Shape{});
    float value = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) value += x[i];
    out[0] = value;
    return out;
}

Tensor mean(const Tensor& x) {
    if (x.size() == 0) throw std::invalid_argument("ZYRON math: mean of empty tensor");
    Tensor out(Shape{});
    float value = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) value += x[i];
    out[0] = value / static_cast<float>(x.size());
    return out;
}

Tensor max(const Tensor& x) {
    if (x.size() == 0) throw std::invalid_argument("ZYRON math: max of empty tensor");
    Tensor out(Shape{});
    float value = x[0];
    for (std::size_t i = 1; i < x.size(); ++i) value = std::max(value, x[i]);
    out[0] = value;
    return out;
}

Tensor min(const Tensor& x) {
    if (x.size() == 0) throw std::invalid_argument("ZYRON math: min of empty tensor");
    Tensor out(Shape{});
    float value = x[0];
    for (std::size_t i = 1; i < x.size(); ++i) value = std::min(value, x[i]);
    out[0] = value;
    return out;
}

Tensor exp(const Tensor& x) {
    Tensor out(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = std::exp(x[i]);
    return out;
}

Tensor log(const Tensor& x) {
    Tensor out(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = std::log(x[i]);
    return out;
}

Tensor sqrt(const Tensor& x) {
    Tensor out(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = std::sqrt(x[i]);
    return out;
}

Tensor pow(const Tensor& x, float exponent) {
    Tensor out(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = std::pow(x[i], exponent);
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.rank() < 2 || b.rank() < 2) {
        throw std::invalid_argument(
            "ZYRON matmul: tensors must have rank >= 2");
    }

    if (a.rank() != b.rank()) {
        throw std::invalid_argument(
            "ZYRON matmul: rank mismatch");
    }

    const std::size_t rank = a.rank();
    for (std::size_t dim = 0; dim + 2 < rank; ++dim) {
        if (a.shape()[dim] != b.shape()[dim]) {
            throw std::invalid_argument(
                "ZYRON matmul: batch dimensions must match");
        }
    }

    const std::size_t m = a.shape()[rank - 2];
    const std::size_t k = a.shape()[rank - 1];
    const std::size_t bk = b.shape()[rank - 2];
    const std::size_t n = b.shape()[rank - 1];

    if (k != bk) {
        throw std::invalid_argument(
            "ZYRON matmul: incompatible matrix dimensions");
    }

    std::vector<std::size_t> output_dims;
    output_dims.reserve(rank);
    for (std::size_t dim = 0; dim + 2 < rank; ++dim) {
        output_dims.push_back(a.shape()[dim]);
    }
    output_dims.push_back(m);
    output_dims.push_back(n);

    Tensor out(Shape(std::move(output_dims)));

    if (a.is_contiguous() && b.is_contiguous() && out.is_contiguous()) {
        if (rank == 2) {
            kernels::matmul_2d_contiguous(a, b, out);
            return out;
        }
        kernels::matmul_batched_contiguous(a, b, out);
        return out;
    }

    std::fill_n(out.data(), out.size(), 0.0f);

    std::size_t batch_count = 1;
    for (std::size_t dim = 0; dim + 2 < rank; ++dim) {
        batch_count *= a.shape()[dim];
    }

    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        std::size_t remaining = batch;
        std::size_t a_base = 0;
        std::size_t b_base = 0;
        std::size_t out_base = 0;

        for (std::size_t dim = rank - 2; dim > 0; --dim) {
            const std::size_t batch_dim = dim - 1;
            const std::size_t index =
                remaining % a.shape()[batch_dim];
            remaining /= a.shape()[batch_dim];
            a_base += index * a.strides()[batch_dim];
            b_base += index * b.strides()[batch_dim];
            out_base += index * out.strides()[batch_dim];
        }

        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t kk = 0; kk < k; ++kk) {
                const float aik = a.data()[
                    a_base + i * a.strides()[rank - 2] +
                    kk * a.strides()[rank - 1]];

                for (std::size_t j = 0; j < n; ++j) {
                    out.data()[
                        out_base + i * out.strides()[rank - 2] +
                        j * out.strides()[rank - 1]] +=
                        aik * b.data()[
                            b_base + kk * b.strides()[rank - 2] +
                            j * b.strides()[rank - 1]];
                }
            }
        }
    }

    return out;
}

} // namespace zyron::math
