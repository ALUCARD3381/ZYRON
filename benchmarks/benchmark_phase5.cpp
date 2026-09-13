#include "zyron/core/memory.hpp"
#include "zyron/core/runtime.hpp"
#include "zyron/core/tensor.hpp"
#include "zyron/math/kernels.hpp"
#include "zyron/math/ops.hpp"
#include "zyron/quantization/quantization.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/model/quantized_language_model.hpp"
#include "zyron/transformer/config.hpp"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
using clock = std::chrono::steady_clock;

template <typename Fn>
double measure_ms(Fn&& fn, int iterations, float& sink) {
    volatile float local_sink = 0.0f;
    const auto start = clock::now();
    for (int i = 0; i < iterations; ++i) local_sink += fn();
    const auto end = clock::now();
    sink = local_sink;
    return std::chrono::duration<double, std::milli>(end - start).count() /
           static_cast<double>(iterations);
}

std::size_t parse_threads(int argc, char** argv) {
    if (argc < 2) return 1;
    const auto value = std::stoull(argv[1]);
    return value == 0 ? 1 : static_cast<std::size_t>(value);
}

void print_memory() {
    const auto stats = zyron::Memory::stats();
    std::cout << "\nZYRON memory buffers\n"
              << "  Live bytes:        " << stats.live_bytes << '\n'
              << "  Peak bytes:        " << stats.peak_bytes << '\n'
              << "  Live allocations:  " << stats.live_allocations << '\n'
              << "  Total allocations: " << stats.total_allocations << '\n'
              << "  Process RSS:       " << zyron::runtime::process_rss_bytes() << " bytes\n";
}

} // namespace

int main(int argc, char** argv) {
    const auto threads = parse_threads(argc, argv);
    zyron::runtime::set_num_threads(threads);
    const auto features = zyron::runtime::cpu_features();

    std::cout << "ZYRON Phase 5 benchmark\n"
              << "=======================\n"
              << "Pointer width: " << sizeof(void*) * 8 << "-bit\n"
              << "Threads: " << zyron::runtime::num_threads() << '\n'
              << "Recommended threads: " << zyron::runtime::recommended_threads() << '\n'
              << "Backend: " << zyron::math::kernels::matmul_backend() << '\n'
              << "ARMv7: " << (features.armv7 ? "yes" : "no") << '\n'
              << "NEON: " << (features.neon ? "yes" : "no") << '\n'
              << "\n";

    for (const std::size_t n : {128U, 256U, 512U}) {
        auto a = zyron::Tensor::random(zyron::Shape{n, n}, -1.0f, 1.0f, 100 + static_cast<std::uint32_t>(n));
        auto b = zyron::Tensor::random(zyron::Shape{n, n}, -1.0f, 1.0f, 200 + static_cast<std::uint32_t>(n));
        const int iterations = n >= 512 ? 3 : 5;
        float sink = 0.0f;
        const double ms = measure_ms([&] {
            auto c = zyron::math::matmul(a, b);
            return c[0];
        }, iterations, sink);
        const double flops = 2.0 * n * n * n;
        const double gflops = (flops / (ms / 1000.0)) / 1e9;
        std::cout << "Float32 MatMul " << n << "x" << n
                  << ": " << std::fixed << std::setprecision(3)
                  << ms << " ms, " << gflops << " GFLOP/s, sink=" << sink << '\n';
    }

    const std::size_t k = 512;
    const std::size_t m = 64;
    const std::size_t n = 512;
    auto input = zyron::Tensor::random(zyron::Shape{m, k}, -1.0f, 1.0f, 501);
    auto weights = zyron::Tensor::random(zyron::Shape{k, n}, -1.0f, 1.0f, 502);
    const auto weight_q8 = zyron::quantization::quantize(weights, zyron::quantization::Type::Int8);
    const auto weight_q4 = zyron::quantization::quantize(weights, zyron::quantization::Type::Int4);
    std::cout << "\nQuantization\n"
              << "  FP32: " << weights.nbytes() << " bytes\n"
              << "  INT8: " << weight_q8.bytes() << " bytes, compression=" << weight_q8.compression_ratio() << "x\n"
              << "  INT4: " << weight_q4.bytes() << " bytes, compression=" << weight_q4.compression_ratio() << "x\n";

    float sink = 0.0f;
    std::cout << "  INT8 matmul: "
              << measure_ms([&] {
                    auto out = zyron::quantization::matmul(input, weight_q8);
                    return out[0];
                 }, 5, sink)
              << " ms, sink=" << sink << '\n';
    std::cout << "  INT4 matmul: "
              << measure_ms([&] {
                    auto out = zyron::quantization::matmul(input, weight_q4);
                    return out[0];
                 }, 5, sink)
              << " ms, sink=" << sink << '\n';


    auto config = zyron::transformer::tiny(64);
    config.hidden_size = 32;
    config.num_layers = 1;
    config.num_heads = 4;
    config.intermediate_size = 64;
    config.max_sequence_length = 32;
    zyron::model::LanguageModel model(config, 7);
    zyron::model::QuantizedLanguageModel q8(model, zyron::quantization::Type::Int8);
    zyron::model::QuantizedLanguageModel q4(model, zyron::quantization::Type::Int4);
    const std::vector<std::size_t> ids{1, 2, 3, 4, 5, 6, 7, 8};

    float qlm_sink = 0.0f;
    const double float_ms = measure_ms([&] {
        const auto out = model.forward(ids).value();
        return out[0];
    }, 3, qlm_sink);
    std::cout << "\nTiny Language Model inference\n"
              << "  FP32: " << float_ms << " ms\n";
    std::cout << "  INT8: "
              << measure_ms([&] {
                    const auto out = q8.forward(ids);
                    return out[0];
                 }, 3, qlm_sink)
              << " ms\n";
    std::cout << "  INT4: "
              << measure_ms([&] {
                    const auto out = q4.forward(ids);
                    return out[0];
                 }, 3, qlm_sink)
              << " ms\n"
              << "  INT8 model storage: " << q8.quantized_bytes() << " bytes\n"
              << "  INT4 model storage: " << q4.quantized_bytes() << " bytes\n"
              << "  FP32 reference: " << q8.float_reference_bytes() << " bytes\n"
              << "  sink=" << qlm_sink << '\n';

    print_memory();
}
