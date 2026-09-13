#pragma once

#include "zyron/core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace zyron::quantization {

enum class Type : std::uint8_t {
    Int8 = 1,
    Int4 = 2,
};

class QuantizedMatrix {
public:
    QuantizedMatrix() = default;

    [[nodiscard]] Type type() const noexcept { return type_; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t elements() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] std::size_t float_bytes() const noexcept;
    [[nodiscard]] double compression_ratio() const noexcept;
    [[nodiscard]] const std::vector<float>& scales() const noexcept { return scales_; }
    [[nodiscard]] std::int8_t value(std::size_t column, std::size_t row) const;
    [[nodiscard]] bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

    [[nodiscard]] Tensor dequantize() const;

    void save(const std::string& path) const;
    static QuantizedMatrix load(const std::string& path);
    void save(std::ostream& out) const;
    static QuantizedMatrix load(std::istream& in);

private:
    Type type_{Type::Int8};
    std::size_t rows_{0};
    std::size_t cols_{0};
    // Logical shape is [rows, cols], but storage is column-major so each
    // inference dot product reads a contiguous quantized weight vector.
    std::vector<std::int8_t> int8_data_;
    std::vector<std::uint8_t> int4_data_;
    std::vector<float> scales_;

    friend QuantizedMatrix quantize(const Tensor&, Type);
    friend Tensor matmul(const Tensor&, const QuantizedMatrix&);
};

[[nodiscard]] QuantizedMatrix quantize(const Tensor& weights, Type type);

// Inference-only matmul. The activation is dynamically quantized per row and
// the weight uses symmetric per-output-column INT8 or packed INT4 scales.
[[nodiscard]] Tensor matmul(const Tensor& input, const QuantizedMatrix& weights);

} // namespace zyron::quantization
