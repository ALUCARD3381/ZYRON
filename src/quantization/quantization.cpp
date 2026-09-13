#include "zyron/quantization/quantization.hpp"
#include "zyron/core/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace zyron::quantization {
namespace {

constexpr std::uint32_t kMagic = 0x315A5951u; // "QYZ1"
constexpr std::uint32_t kVersion = 1;

void write_exact(std::ostream& out, const void* data, std::size_t bytes) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("ZYRON quantization: write failure");
}

void read_exact(std::istream& in, void* data, std::size_t bytes) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("ZYRON quantization: corrupted file");
}

void write_u32(std::ostream& out, std::uint32_t value) { write_exact(out, &value, sizeof(value)); }
void write_u64(std::ostream& out, std::uint64_t value) { write_exact(out, &value, sizeof(value)); }
std::uint32_t read_u32(std::istream& in) { std::uint32_t value{}; read_exact(in, &value, sizeof(value)); return value; }
std::uint64_t read_u64(std::istream& in) { std::uint64_t value{}; read_exact(in, &value, sizeof(value)); return value; }

std::size_t checked_product(std::size_t a, std::size_t b, const char* message) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(message);
    }
    return a * b;
}

std::int8_t quantize_value(float value, float scale, int limit) {
    if (scale <= 0.0f || !std::isfinite(scale)) return 0;
    const long rounded = std::lround(value / scale);
    const long clamped = std::clamp(rounded, static_cast<long>(-limit), static_cast<long>(limit));
    return static_cast<std::int8_t>(clamped);
}

std::int8_t unpack_int4(std::uint8_t packed, bool high) {
    const std::uint8_t nibble = high
        ? static_cast<std::uint8_t>(packed >> 4)
        : static_cast<std::uint8_t>(packed & 0x0Fu);
    return static_cast<std::int8_t>(nibble < 8u ? nibble : static_cast<int>(nibble) - 16);
}

void pack_int4(std::vector<std::uint8_t>& dst, std::size_t index, std::int8_t value) {
    const std::size_t byte_index = index >> 1;
    const std::uint8_t nibble = static_cast<std::uint8_t>(value) & 0x0Fu;
    if ((index & 1u) == 0u) {
        dst[byte_index] = static_cast<std::uint8_t>((dst[byte_index] & 0xF0u) | nibble);
    } else {
        dst[byte_index] = static_cast<std::uint8_t>((dst[byte_index] & 0x0Fu) | (nibble << 4));
    }
}

#if defined(__ARM_NEON)
std::int32_t dot_int8_neon(const std::int8_t* a, const std::int8_t* b, std::size_t size) {
    int32x4_t accumulator = vdupq_n_s32(0);
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        const int8x8_t va = vld1_s8(a + i);
        const int8x8_t vb = vld1_s8(b + i);
        const int16x8_t product = vmull_s8(va, vb);
        accumulator = vaddq_s32(accumulator, vpaddlq_s16(product));
    }
    const int32x2_t pair = vadd_s32(vget_low_s32(accumulator), vget_high_s32(accumulator));
    const int32x2_t reduced = vpadd_s32(pair, pair);
    std::int32_t result = vget_lane_s32(reduced, 0);
    for (; i < size; ++i) result += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return result;
}
#endif

} // namespace

std::int8_t QuantizedMatrix::value(std::size_t column, std::size_t row) const {
    if (column >= cols_ || row >= rows_) {
        throw std::out_of_range("ZYRON quantization: matrix index out of range");
    }
    const std::size_t linear = checked_product(
        column, rows_, "ZYRON quantization: index overflow") + row;
    if (type_ == Type::Int8) return int8_data_[linear];
    return unpack_int4(int4_data_[linear >> 1], (linear & 1u) != 0u);
}

std::size_t QuantizedMatrix::elements() const noexcept {
    if (rows_ == 0 || cols_ == 0) return 0;
    return rows_ * cols_;
}

std::size_t QuantizedMatrix::bytes() const noexcept {
    const std::size_t payload = type_ == Type::Int8
        ? int8_data_.size()
        : int4_data_.size();
    return payload + scales_.size() * sizeof(float);
}

std::size_t QuantizedMatrix::float_bytes() const noexcept {
    return elements() * sizeof(float);
}

double QuantizedMatrix::compression_ratio() const noexcept {
    const auto qbytes = bytes();
    if (qbytes == 0) return 0.0;
    return static_cast<double>(float_bytes()) / static_cast<double>(qbytes);
}

QuantizedMatrix quantize(const Tensor& weights, Type type) {
    if (weights.rank() != 2) {
        throw std::invalid_argument("ZYRON quantization: weights must be rank-2");
    }
    if (!weights.is_contiguous()) {
        throw std::invalid_argument("ZYRON quantization: weights must be contiguous");
    }

    QuantizedMatrix result;
    result.type_ = type;
    result.rows_ = weights.shape()[0];
    result.cols_ = weights.shape()[1];
    checked_product(result.rows_, result.cols_, "ZYRON quantization: matrix size overflow");
    result.scales_.resize(result.cols_, 1.0f);

    const std::size_t element_count = result.elements();
    if (type == Type::Int8) result.int8_data_.resize(element_count);
    else if (type == Type::Int4) result.int4_data_.assign((element_count + 1) / 2, 0u);
    else throw std::invalid_argument("ZYRON quantization: unsupported type");

    for (std::size_t column = 0; column < result.cols_; ++column) {
        float max_abs = 0.0f;
        for (std::size_t row = 0; row < result.rows_; ++row) {
            max_abs = std::max(max_abs,
                std::fabs(weights.data()[row * result.cols_ + column]));
        }
        const float limit = type == Type::Int8 ? 127.0f : 7.0f;
        result.scales_[column] = max_abs > 0.0f ? max_abs / limit : 1.0f;

        for (std::size_t row = 0; row < result.rows_; ++row) {
            const std::int8_t value = quantize_value(
                weights.data()[row * result.cols_ + column],
                result.scales_[column],
                type == Type::Int8 ? 127 : 7);
            const std::size_t linear = column * result.rows_ + row;
            if (type == Type::Int8) result.int8_data_[linear] = value;
            else pack_int4(result.int4_data_, linear, value);
        }
    }

    return result;
}

Tensor QuantizedMatrix::dequantize() const {
    if (empty()) return Tensor(Shape{rows_, cols_});
    Tensor result(Shape{rows_, cols_});
    for (std::size_t column = 0; column < cols_; ++column) {
        const float scale = scales_[column];
        for (std::size_t row = 0; row < rows_; ++row) {
            result.data()[row * cols_ + column] =
                static_cast<float>(value(column, row)) * scale;
        }
    }
    return result;
}

Tensor matmul(const Tensor& input, const QuantizedMatrix& weights) {
    if (input.rank() < 2) {
        throw std::invalid_argument("ZYRON quantization: input rank must be >= 2");
    }
    if (weights.empty()) {
        throw std::invalid_argument("ZYRON quantization: weights are empty");
    }
    if (input.shape()[input.rank() - 1] != weights.rows_) {
        throw std::invalid_argument("ZYRON quantization: input/weight dimension mismatch");
    }

    const Tensor contiguous_input = input.is_contiguous() ? input : input.contiguous();
    const std::size_t k = weights.rows_;
    const std::size_t n = weights.cols_;
    const std::size_t m = contiguous_input.size() / k;
    Tensor output([&] {
        auto dims = contiguous_input.shape().dims();
        dims.back() = n;
        return Shape(std::move(dims));
    }());

    runtime::parallel_for(
        0,
        m,
        [&](std::size_t begin, std::size_t end) {
            std::vector<std::int8_t> quantized_row(k);
            for (std::size_t row = begin; row < end; ++row) {
                const float* input_row = contiguous_input.data() + row * k;
                float max_abs = 0.0f;
                for (std::size_t kk = 0; kk < k; ++kk) {
                    max_abs = std::max(max_abs, std::fabs(input_row[kk]));
                }
                const float input_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
                for (std::size_t kk = 0; kk < k; ++kk) {
                    quantized_row[kk] = quantize_value(input_row[kk], input_scale, 127);
                }

                float* output_row = output.data() + row * n;
                for (std::size_t column = 0; column < n; ++column) {
                    std::int32_t accumulator = 0;
#if defined(__ARM_NEON)
                    if (weights.type_ == Type::Int8 && runtime::cpu_features().neon) {
                        accumulator = dot_int8_neon(
                            quantized_row.data(),
                            weights.int8_data_.data() + column * k,
                            k);
                    } else
#endif
                    {
                        for (std::size_t kk = 0; kk < k; ++kk) {
                            accumulator += static_cast<std::int32_t>(quantized_row[kk]) *
                                           static_cast<std::int32_t>(weights.value(column, kk));
                        }
                    }
                    output_row[column] = static_cast<float>(accumulator) *
                                         input_scale * weights.scales_[column];
                }
            }
        },
        1);
    return output;
}

void QuantizedMatrix::save(const std::string& path) const {
    if (empty()) throw std::runtime_error("ZYRON quantization: cannot save empty matrix");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("ZYRON quantization: failed to open: " + path);
    save(out);
}

void QuantizedMatrix::save(std::ostream& out) const {
    if (empty()) throw std::runtime_error("ZYRON quantization: cannot save empty matrix");
    write_u32(out, kMagic);
    write_u32(out, kVersion);
    write_u32(out, static_cast<std::uint32_t>(type_));
    write_u64(out, static_cast<std::uint64_t>(rows_));
    write_u64(out, static_cast<std::uint64_t>(cols_));
    write_u64(out, static_cast<std::uint64_t>(scales_.size()));
    write_exact(out, scales_.data(), scales_.size() * sizeof(float));

    const std::size_t payload = type_ == Type::Int8 ? int8_data_.size() : int4_data_.size();
    write_u64(out, static_cast<std::uint64_t>(payload));
    if (type_ == Type::Int8) write_exact(out, int8_data_.data(), int8_data_.size());
    else write_exact(out, int4_data_.data(), int4_data_.size());
}

QuantizedMatrix QuantizedMatrix::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("ZYRON quantization: failed to open: " + path);
    return load(in);
}

QuantizedMatrix QuantizedMatrix::load(std::istream& in) {
    if (read_u32(in) != kMagic || read_u32(in) != kVersion) {
        throw std::runtime_error("ZYRON quantization: unsupported format");
    }

    QuantizedMatrix result;
    const auto type = read_u32(in);
    if (type != static_cast<std::uint32_t>(Type::Int8) && type != static_cast<std::uint32_t>(Type::Int4)) {
        throw std::runtime_error("ZYRON quantization: invalid type");
    }
    result.type_ = static_cast<Type>(type);
    const auto rows64 = read_u64(in);
    const auto cols64 = read_u64(in);
    if (rows64 == 0 || cols64 == 0 ||
        rows64 > std::numeric_limits<std::size_t>::max() ||
        cols64 > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("ZYRON quantization: invalid matrix dimensions");
    }
    result.rows_ = static_cast<std::size_t>(rows64);
    result.cols_ = static_cast<std::size_t>(cols64);
    const std::size_t elements = checked_product(result.rows_, result.cols_, "ZYRON quantization: matrix size overflow");

    const auto scale_count = read_u64(in);
    if (scale_count != result.cols_) throw std::runtime_error("ZYRON quantization: invalid scale count");
    result.scales_.resize(result.cols_);
    read_exact(in, result.scales_.data(), result.scales_.size() * sizeof(float));
    for (float scale : result.scales_) {
        if (!(scale > 0.0f) || !std::isfinite(scale)) throw std::runtime_error("ZYRON quantization: invalid scale");
    }

    const auto payload64 = read_u64(in);
    const std::size_t expected = result.type_ == Type::Int8 ? elements : (elements + 1) / 2;
    if (payload64 != expected) throw std::runtime_error("ZYRON quantization: invalid payload size");

    if (result.type_ == Type::Int8) {
        result.int8_data_.resize(expected);
        read_exact(in, result.int8_data_.data(), expected);
    } else {
        result.int4_data_.resize(expected);
        read_exact(in, result.int4_data_.data(), expected);
    }
    return result;
}

} // namespace zyron::quantization
