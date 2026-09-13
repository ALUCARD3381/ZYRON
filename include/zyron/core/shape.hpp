#pragma once

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace zyron {

class Shape {
public:
    using value_type = std::size_t;

    Shape() = default;
    Shape(std::initializer_list<value_type> dims) : dims_(dims) { validate(); }
    explicit Shape(std::vector<value_type> dims) : dims_(std::move(dims)) { validate(); }

    [[nodiscard]] std::size_t rank() const noexcept { return dims_.size(); }

    [[nodiscard]] std::size_t size() const {
        if (dims_.empty()) return 1;

        std::size_t result = 1;
        for (const auto dim : dims_) {
            if (dim != 0 && result > std::numeric_limits<std::size_t>::max() / dim) {
                throw std::overflow_error("ZYRON Shape: element count overflow");
            }
            result *= dim;
        }
        return result;
    }

    [[nodiscard]] const std::vector<value_type>& dims() const noexcept { return dims_; }
    [[nodiscard]] value_type operator[](std::size_t i) const { return dims_.at(i); }
    [[nodiscard]] bool empty() const noexcept { return dims_.empty(); }

    friend bool operator==(const Shape& a, const Shape& b) noexcept { return a.dims_ == b.dims_; }
    friend bool operator!=(const Shape& a, const Shape& b) noexcept { return !(a == b); }

private:
    void validate() const {
        std::size_t result = 1;

        for (const auto dim : dims_) {
            if (dim == 0) {
                throw std::invalid_argument(
                    "ZYRON Shape: dimensions must be > 0");
            }

            if (result > std::numeric_limits<std::size_t>::max() / dim) {
                throw std::overflow_error(
                    "ZYRON Shape: element count overflow");
            }

            result *= dim;
        }
    }

    std::vector<value_type> dims_;
};

} // namespace zyron
