#include "zyron/nn/nn.hpp"
#include "zyron/nn/sequential.hpp"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>

using zyron::Shape;
using zyron::Tensor;
using zyron::autograd::Variable;
using zyron::autograd::cross_entropy;
using zyron::nn::Linear;
using zyron::nn::Sequential;

int main() {
    constexpr std::size_t batch = 32;
    constexpr std::size_t input_features = 128;
    constexpr std::size_t hidden = 256;
    constexpr std::size_t classes = 16;
    constexpr int iterations = 10;

    Tensor input_value = Tensor::random(
        Shape{batch, input_features}, -1.0f, 1.0f, 101);
    const std::vector<std::size_t> targets(batch, 3);

    Sequential model;
    model.emplace<Linear>(input_features, hidden, true, 1);
    model.emplace<Linear>(hidden, classes, true, 2);

    double total_ms = 0.0;
    float sink = 0.0f;

    std::cout << "ZYRON Phase 2 NN benchmark\n"
              << "==========================\n"
              << "Batch: " << batch << "\n"
              << "MLP: " << input_features << " -> " << hidden
              << " -> " << classes << "\n"
              << "Iterations: " << iterations << "\n";

    for (int i = 0; i < iterations; ++i) {
        Variable input(input_value, true);

        const auto start = std::chrono::steady_clock::now();
        auto logits = model.forward(input);
        auto loss = cross_entropy(logits, targets);
        loss.backward();
        const auto end = std::chrono::steady_clock::now();

        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        sink += loss.value()[0] + input.grad()[0];

        for (auto* parameter : model.parameters()) {
            parameter->zero_grad();
        }
        input.zero_grad();
    }

    std::cout << std::fixed << std::setprecision(3)
              << "Forward + backward average: "
              << total_ms / iterations << " ms\n"
              << "sink=" << sink << "\n";

    return 0;
}
