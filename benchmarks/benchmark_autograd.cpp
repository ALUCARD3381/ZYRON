#include "zyron/autograd/autograd.hpp"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>

using zyron::Shape;
using zyron::Tensor;
using zyron::autograd::Variable;
using zyron::autograd::exp;
using zyron::autograd::matmul;
using zyron::autograd::mean;
using zyron::autograd::mul;
using zyron::autograd::sum;

int main() {
    constexpr std::size_t n = 128;
    constexpr int iterations = 10;

    Tensor a_value = Tensor::random(Shape{n, n}, -0.1f, 0.1f, 11);
    Tensor b_value = Tensor::random(Shape{n, n}, -0.1f, 0.1f, 17);

    double forward_total_ms = 0.0;
    double backward_total_ms = 0.0;
    float sink = 0.0f;

    std::cout << "ZYRON Phase 2 autograd benchmark\n";
    std::cout << "================================\n";
    std::cout << "Matrix: " << n << "x" << n << "\n";
    std::cout << "Iterations: " << iterations << "\n";

    for (int i = 0; i < iterations; ++i) {
        Variable a(a_value, true);
        Variable b(b_value, true);

        const auto forward_start = std::chrono::steady_clock::now();
        auto product = matmul(a, b);
        auto loss = mean(exp(mul(product, product)));
        const auto forward_end = std::chrono::steady_clock::now();

        const auto backward_start = std::chrono::steady_clock::now();
        loss.backward();
        const auto backward_end = std::chrono::steady_clock::now();

        forward_total_ms += std::chrono::duration<double, std::milli>(
            forward_end - forward_start).count();
        backward_total_ms += std::chrono::duration<double, std::milli>(
            backward_end - backward_start).count();
        sink += loss.value()[0] + a.grad()[0] + b.grad()[0];
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Forward average:  "
              << forward_total_ms / iterations << " ms\n";
    std::cout << "Backward average: "
              << backward_total_ms / iterations << " ms\n";
    std::cout << "sink=" << sink << "\n";

    return 0;
}
