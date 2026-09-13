#include "zyron/core/tensor.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace zyron {

bool Tensor::has_contiguous_strides(const Shape& shape, const std::vector<std::size_t>& strides) noexcept {
    if (strides.size() != shape.rank()) return false;
    std::size_t expected = 1;
    for (std::size_t dim = shape.rank(); dim-- > 0;) {
        if (strides[dim] != expected) return false;
        expected *= shape[dim];
    }
    return true;
}

std::vector<std::size_t> Tensor::make_contiguous_strides(const Shape& shape) {
    std::vector<std::size_t> strides(shape.rank(), 1);
    if (shape.rank() > 1) {
        for (std::size_t i = shape.rank() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
    }
    return strides;
}

Tensor::Tensor(Shape shape, DType dtype)
    : shape_(std::move(shape)), strides_(make_contiguous_strides(shape_)),
      memory_(shape_.size() * dtype_size(dtype)), dtype_(dtype), contiguous_(true) {
    validate_dtype();
}

Tensor::Tensor(Shape shape, std::vector<std::size_t> strides, Memory memory, DType dtype)
    : shape_(std::move(shape)), strides_(std::move(strides)), memory_(std::move(memory)), dtype_(dtype) {
    validate_dtype();
    if (strides_.size() != shape_.rank()) {
        throw std::invalid_argument("ZYRON Tensor: stride rank mismatch");
    }

    std::size_t max_offset = 0;
    for (std::size_t dim = 0; dim < shape_.rank(); ++dim) {
        const std::size_t extent = shape_[dim] - 1;
        if (extent != 0 && strides_[dim] > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::overflow_error("ZYRON Tensor: stride offset overflow");
        }
        const std::size_t contribution = extent * strides_[dim];
        if (max_offset > std::numeric_limits<std::size_t>::max() - contribution) {
            throw std::overflow_error("ZYRON Tensor: stride offset overflow");
        }
        max_offset += contribution;
    }

    const std::size_t required_elements = shape_.rank() == 0 ? 1 : max_offset + 1;
    if (required_elements > std::numeric_limits<std::size_t>::max() / dtype_size(dtype_)) {
        throw std::overflow_error("ZYRON Tensor: byte size overflow");
    }
    const auto required_bytes = required_elements * dtype_size(dtype_);
    if (memory_.size_bytes() < required_bytes) {
        throw std::invalid_argument("ZYRON Tensor: insufficient memory for strides");
    }
    contiguous_ = has_contiguous_strides(shape_, strides_) && !memory_.empty();
}

void Tensor::validate_dtype() const {
    if (dtype_ != DType::Float32) {
        throw std::runtime_error("ZYRON Tensor: only float32 is implemented in Phase 1");
    }
}

std::size_t Tensor::size() const {
    if (memory_.empty() && shape_.empty()) return 0; // default/empty Tensor
    return shape_.size();
}

std::size_t Tensor::nbytes() const noexcept {
    return size() * dtype_size(dtype_);
}

bool Tensor::is_contiguous() const noexcept {
    return contiguous_;
}

std::size_t Tensor::offset(const std::vector<std::size_t>& indices) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument("ZYRON Tensor: index rank mismatch");
    }
    std::size_t result = 0;
    for (std::size_t i = 0; i < rank(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("ZYRON Tensor: index out of range");
        }
        result += indices[i] * strides_[i];
    }
    return result;
}

std::size_t Tensor::offset_linear(std::size_t linear_index) const noexcept {
    if (is_contiguous()) return linear_index;

    std::size_t physical_offset = 0;
    for (std::size_t dim = rank(); dim-- > 0;) {
        const std::size_t index = linear_index % shape_[dim];
        linear_index /= shape_[dim];
        physical_offset += index * strides_[dim];
    }
    return physical_offset;
}

float& Tensor::at(const std::vector<std::size_t>& indices) { return data()[offset(indices)]; }
const float& Tensor::at(const std::vector<std::size_t>& indices) const { return data()[offset(indices)]; }

float& Tensor::operator[](std::size_t i) noexcept { return data()[offset_linear(i)]; }
const float& Tensor::operator[](std::size_t i) const noexcept { return data()[offset_linear(i)]; }

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (!is_contiguous()) throw std::invalid_argument("ZYRON Tensor: reshape requires contiguous tensor");
    if (new_shape.size() != size()) throw std::invalid_argument("ZYRON Tensor: reshape changes element count");
    return Tensor(new_shape, make_contiguous_strides(new_shape), memory_, dtype_);
}

Tensor Tensor::transpose(std::size_t dim0, std::size_t dim1) const {
    if (dim0 >= rank() || dim1 >= rank()) throw std::out_of_range("ZYRON Tensor: transpose dimension out of range");
    auto dims = shape_.dims();
    auto strides = strides_;
    std::swap(dims[dim0], dims[dim1]);
    std::swap(strides[dim0], strides[dim1]);
    return Tensor(Shape(std::move(dims)), std::move(strides), memory_, dtype_);
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;

    Tensor out(shape_, dtype_);
    for (std::size_t linear = 0; linear < size(); ++linear) out[linear] = (*this)[linear];
    return out;
}

Tensor Tensor::slice_prefix(std::size_t dim, std::size_t length) const {
    if (dim >= rank()) {
        throw std::out_of_range("ZYRON Tensor: prefix slice dimension out of range");
    }
    if (length > shape_[dim]) {
        throw std::out_of_range("ZYRON Tensor: prefix slice length exceeds dimension");
    }

    auto dims = shape_.dims();
    dims[dim] = length;
    return Tensor(Shape(std::move(dims)), strides_, memory_, dtype_);
}

Tensor Tensor::zeros(const Shape& shape) {
    Tensor out(shape);
    std::fill_n(out.data(), out.size(), 0.0f);
    return out;
}

Tensor Tensor::ones(const Shape& shape) {
    Tensor out(shape);
    std::fill_n(out.data(), out.size(), 1.0f);
    return out;
}

Tensor Tensor::random(const Shape& shape, float low, float high, std::uint32_t seed) {
    if (low > high) throw std::invalid_argument("ZYRON Tensor: random low must be <= high");
    Tensor out(shape);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = dist(rng);
    return out;
}

} // namespace zyron
