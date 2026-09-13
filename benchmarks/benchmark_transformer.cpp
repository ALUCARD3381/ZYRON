#include "zyron/autograd/autograd.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/transformer/config.hpp"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using Clock = std::chrono::steady_clock;

    zyron::transformer::Config config;
    config.vocab_size = 320;
    config.hidden_size = 64;
    config.num_layers = 2;
    config.num_heads = 4;
    config.intermediate_size = 128;
    config.max_sequence_length = 64;
    config.validate();

    zyron::model::LanguageModel model(config, 123);

    const std::vector<std::vector<std::size_t>> input{
        std::vector<std::size_t>(32, 1),
        std::vector<std::size_t>(32, 2),
        std::vector<std::size_t>(32, 3),
        std::vector<std::size_t>(32, 4)};

    const std::vector<std::vector<std::size_t>> target = input;

    constexpr std::size_t iterations = 5;
    double forward_ms = 0.0;
    double training_ms = 0.0;
    float sink = 0.0f;

    for (std::size_t i = 0; i < iterations; ++i) {
        model.eval();
        zyron::autograd::NoGradGuard guard;
        const auto start = Clock::now();
        const auto logits = model.forward(input);
        const auto end = Clock::now();
        forward_ms += std::chrono::duration<double, std::milli>(end - start).count();
        sink += logits.value()[0];
    }

    for (std::size_t i = 0; i < iterations; ++i) {
        model.train();
        const auto start = Clock::now();
        auto loss = model.loss(input, target);
        loss.backward();
        const auto end = Clock::now();
        training_ms += std::chrono::duration<double, std::milli>(end - start).count();
        sink += loss.value()[0];
        for (auto* parameter : model.parameters()) parameter->zero_grad();
    }

    std::cout << "ZYRON Phase 3 Transformer benchmark\n"
              << "===================================\n"
              << "Batch: 4\n"
              << "Sequence: 32\n"
              << "Hidden: 64\n"
              << "Layers: 2\n"
              << "Heads: 4\n"
              << "Iterations: " << iterations << "\n"
              << std::fixed << std::setprecision(3)
              << "Forward average: " << forward_ms / iterations << " ms\n"
              << "Forward + backward average: " << training_ms / iterations << " ms\n"
              << "sink=" << sink << "\n";

    return 0;
}
