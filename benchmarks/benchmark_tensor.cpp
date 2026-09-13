#include "zyron/core/tensor.hpp"
#include "zyron/math/ops.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {
using clock = std::chrono::steady_clock;

struct BenchResult {
    double ms{};
    double sink{};
};

template <typename Fn>
BenchResult measure(Fn&& fn, int iterations) {
    volatile float sink = 0.0f;
    const auto start = clock::now();
    for (int i = 0; i < iterations; ++i) sink += fn();
    const auto end = clock::now();
    return {
        std::chrono::duration<double, std::milli>(end - start).count() / iterations,
        static_cast<double>(sink)
    };
}

void print_result(std::string_view name, const BenchResult& result) {
    std::cout << std::left << std::setw(22) << name
              << std::right << std::setw(12) << std::fixed << std::setprecision(3)
              << result.ms << " ms"
              << "  sink=" << result.sink << '\n';
}

} // namespace

int main() {
    constexpr int iterations = 20;

    const zyron::Shape shape{256, 256};
    auto a = zyron::Tensor::random(shape, -1.0f, 1.0f, 42);
    auto b = zyron::Tensor::random(shape, -1.0f, 1.0f, 43);

    std::cout << "ZYRON Phase 1 benchmark\n"
              << "=======================\n"
              << "Host pointer width: " << (sizeof(void*) * 8) << "-bit\n"
              << "Element type: float32\n"
              << "Iterations: " << iterations << "\n\n";

    print_result("Tensor allocation", measure([&]() {
        zyron::Tensor t(shape);
        return t.nbytes() > 0 ? t.nbytes() % 17 : 0;
    }, iterations));

    print_result("Tensor zeros", measure([&]() {
        auto t = zyron::Tensor::zeros(shape);
        return t[0];
    }, iterations));

    print_result("Add", measure([&]() {
        auto c = zyron::math::add(a, b);
        return c[0];
    }, iterations));

    print_result("Mul", measure([&]() {
        auto c = zyron::math::mul(a, b);
        return c[0];
    }, iterations));

    print_result("Sum", measure([&]() {
        auto c = zyron::math::sum(a);
        return c[0];
    }, iterations));

    std::cout << "\nMatMul baseline\n";
    for (const std::size_t n : {64U, 128U, 256U, 512U}) {
        auto x = zyron::Tensor::random(zyron::Shape{n, n}, -1.0f, 1.0f, 100 + static_cast<std::uint32_t>(n));
        auto y = zyron::Tensor::random(zyron::Shape{n, n}, -1.0f, 1.0f, 200 + static_cast<std::uint32_t>(n));
        const int matmul_iterations = n >= 512 ? 3 : 5;
        const auto result = measure([&]() {
            auto c = zyron::math::matmul(x, y);
            return c[0];
        }, matmul_iterations);
        const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const double gflops = (flops / (result.ms / 1000.0)) / 1e9;
        std::cout << "  " << n << "x" << n << " * " << n << "x" << n
                  << ": " << std::fixed << std::setprecision(3)
                  << result.ms << " ms, " << gflops << " GFLOP/s"
                  << ", sink=" << result.sink << '\n';
    }

    std::cout << "\nMemory (two 256x256 inputs): "
              << (2.0 * shape.size() * sizeof(float) / (1024.0 * 1024.0))
              << " MiB\n";
}
