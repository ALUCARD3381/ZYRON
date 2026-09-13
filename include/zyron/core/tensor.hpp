#pragma once

#include "zyron/core/dtype.hpp"
#include "zyron/core/memory.hpp"
#include "zyron/core/shape.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace zyron {

class Tensor {
public:
    using index_type = std::size_t;

    Tensor() = default;
    // Allocates storage without initializing its contents. Use zeros(), ones(),
    // or random() when initialized values are required.
    explicit Tensor(Shape shape, DType dtype = DType::Float32);
    Tensor(Shape shape, std::vector<std::size_t> strides, Memory memory, DType dtype = DType::Float32);

    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.rank(); }
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t nbytes() const noexcept;
    [[nodiscard]] std::size_t storage_nbytes() const noexcept { return memory_.size_bytes(); }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] bool is_contiguous() const noexcept;

    float* data() noexcept { return reinterpret_cast<float*>(memory_.data()); }
    const float* data() const noexcept { return reinterpret_cast<const float*>(memory_.data()); }

    float& at(const std::vector<std::size_t>& indices);
    const float& at(const std::vector<std::size_t>& indices) const;
    float& operator[](std::size_t i) noexcept;
    const float& operator[](std::size_t i) const noexcept;

    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(std::size_t dim0, std::size_t dim1) const;
    Tensor contiguous() const;

    // Returns a zero-offset view with a shorter extent along one dimension.
    // The view shares the same storage and does not copy data.
    Tensor slice_prefix(std::size_t dim, std::size_t length) const;

    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor random(const Shape& shape, float low = 0.0f, float high = 1.0f, std::uint32_t seed = 0);

private:
    static std::vector<std::size_t> make_contiguous_strides(const Shape& shape);
    static bool has_contiguous_strides(const Shape& shape, const std::vector<std::size_t>& strides) noexcept;
    std::size_t offset(const std::vector<std::size_t>& indices) const;
    std::size_t offset_linear(std::size_t linear_index) const noexcept;
    void validate_dtype() const;

    Shape shape_{};
    std::vector<std::size_t> strides_{};
    Memory memory_{};
    DType dtype_{DType::Float32};
    bool contiguous_{false};
};

} // namespace zyron
