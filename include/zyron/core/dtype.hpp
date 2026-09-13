#pragma once

#include <cstddef>
#include <string_view>

namespace zyron {

enum class DType {
    Float32,
};

constexpr std::size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
    }
    return 0;
}

constexpr std::string_view dtype_name(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float32: return "float32";
    }
    return "unknown";
}

} // namespace zyron
