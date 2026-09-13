#include "zyron/core/memory.hpp"
#include "zyron/core/runtime.hpp"
#include "zyron/core/tensor.hpp"
#include "zyron/math/kernels.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/model/quantized_language_model.hpp"
#include "zyron/transformer/config.hpp"
#include "zyron/math/ops.hpp"
#include "zyron/quantization/quantization.hpp"
#include "zyron/quantization/quantized_linear.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void check_close(float a, float b, float eps, const char* message) {
    if (std::fabs(a - b) > eps) throw std::runtime_error(message);
}

void compare_tensors(
    const zyron::Tensor& a,
    const zyron::Tensor& b,
    float eps,
    const char* message) {

    check(a.shape() == b.shape(), message);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            throw std::runtime_error(message);
        }
    }
}

zyron::Tensor reference_matmul(const zyron::Tensor& a, const zyron::Tensor& b) {
    check(a.rank() == 2 && b.rank() == 2, "reference rank");
    check(a.shape()[1] == b.shape()[0], "reference dimensions");
    zyron::Tensor out(zyron::Shape{a.shape()[0], b.shape()[1]});
    for (std::size_t i = 0; i < a.shape()[0]; ++i) {
        for (std::size_t j = 0; j < b.shape()[1]; ++j) {
            float value = 0.0f;
            for (std::size_t k = 0; k < a.shape()[1]; ++k) {
                value += a.at({i, k}) * b.at({k, j});
            }
            out.at({i, j}) = value;
        }
    }
    return out;
}

void test_memory_alignment_and_stats() {
    const auto before = zyron::Memory::stats();
    zyron::Memory memory(1024);
    check(memory.data() != nullptr, "memory data");
    check(reinterpret_cast<std::uintptr_t>(memory.data()) % zyron::Memory::kAlignment == 0,
          "memory alignment");
    const auto during = zyron::Memory::stats();
    check(during.live_bytes >= before.live_bytes + 1024, "memory live bytes");
    check(during.live_allocations >= before.live_allocations + 1, "memory live allocations");
}

void test_runtime() {
    const auto features = zyron::runtime::cpu_features();
    (void)features;
    zyron::runtime::set_num_threads(2);
    check(zyron::runtime::num_threads() == 2, "runtime threads");
    zyron::runtime::set_num_threads(1);
}

void test_optimized_matmul() {
    auto a = zyron::Tensor::random(zyron::Shape{17, 19}, -1.0f, 1.0f, 11);
    auto b = zyron::Tensor::random(zyron::Shape{19, 13}, -1.0f, 1.0f, 12);

    auto reference = reference_matmul(a, b);
    auto actual = zyron::math::matmul(a, b);
    compare_tensors(reference, actual, 1e-4f, "optimized matmul mismatch");

    auto ab = zyron::Tensor::random(zyron::Shape{3, 17, 19}, -1.0f, 1.0f, 13);
    auto bb = zyron::Tensor::random(zyron::Shape{3, 19, 13}, -1.0f, 1.0f, 14);
    auto actual_batch = zyron::math::matmul(ab, bb);
    check(actual_batch.shape() == zyron::Shape{3, 17, 13}, "batched matmul shape");
    check(zyron::math::kernels::matmul_backend() != nullptr, "backend name");
}

void test_quantization() {
    auto weights = zyron::Tensor::random(zyron::Shape{32, 24}, -1.0f, 1.0f, 21);
    auto input = zyron::Tensor::random(zyron::Shape{7, 32}, -1.0f, 1.0f, 22);

    for (const auto type : {zyron::quantization::Type::Int8, zyron::quantization::Type::Int4}) {
        const auto q = zyron::quantization::quantize(weights, type);
        check(q.rows() == 32 && q.cols() == 24, "quantized shape");
        check(q.scales().size() == 24, "quantized scales");
        check(q.bytes() < q.float_bytes(), "quantized storage reduction");

        const auto deq = q.dequantize();
        float max_error = 0.0f;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            max_error = std::max(max_error, std::fabs(weights[i] - deq[i]));
        }
        if (type == zyron::quantization::Type::Int8) {
            check(max_error < 0.02f, "int8 quantization error");
        } else {
            check(max_error < 0.2f, "int4 quantization error");
        }

        const auto qout = zyron::quantization::matmul(input, q);
        const auto reference = zyron::math::matmul(input, deq);
        const float tolerance = type == zyron::quantization::Type::Int8 ? 0.15f : 0.5f;
        compare_tensors(reference, qout, tolerance, "quantized matmul mismatch");

        const auto path = std::filesystem::temp_directory_path() /
            (type == zyron::quantization::Type::Int8 ? "zyron_q8.zyq" : "zyron_q4.zyq");
        q.save(path.string());
        const auto loaded = zyron::quantization::QuantizedMatrix::load(path.string());
        compare_tensors(q.dequantize(), loaded.dequantize(), 0.0f, "quantized serialization mismatch");
        std::filesystem::remove(path);
    }
}

void test_quantized_linear() {
    auto weight = zyron::Tensor::random(zyron::Shape{16, 8}, -0.5f, 0.5f, 31);
    auto bias = zyron::Tensor::random(zyron::Shape{8}, -0.2f, 0.2f, 32);
    auto input = zyron::Tensor::random(zyron::Shape{4, 5, 16}, -1.0f, 1.0f, 33);

    zyron::quantization::QuantizedLinear linear8(
        weight, &bias, zyron::quantization::Type::Int8);
    zyron::quantization::QuantizedLinear linear4(
        weight, &bias, zyron::quantization::Type::Int4);

    auto out8 = linear8.forward(input);
    auto out4 = linear4.forward(input);

    const auto flat_input = input.reshape(zyron::Shape{20, 16});
    auto reference = zyron::math::matmul(
        flat_input,
        weight);
    reference = reference.reshape(zyron::Shape{4, 5, 8});
    for (std::size_t row = 0; row < reference.size() / bias.size(); ++row) {
        for (std::size_t col = 0; col < bias.size(); ++col) {
            reference[row * bias.size() + col] += bias[col];
        }
    }

    compare_tensors(reference, out8, 0.2f, "quantized linear int8 mismatch");
    compare_tensors(reference, out4, 0.6f, "quantized linear int4 mismatch");
}


void test_quantized_language_model() {
    auto config = zyron::transformer::tiny(32);
    config.hidden_size = 16;
    config.num_layers = 1;
    config.num_heads = 4;
    config.intermediate_size = 32;
    config.max_sequence_length = 16;

    zyron::model::LanguageModel model(config, 77);
    zyron::model::QuantizedLanguageModel q8(
        model, zyron::quantization::Type::Int8);
    zyron::model::QuantizedLanguageModel q4(
        model, zyron::quantization::Type::Int4);

    const std::vector<std::size_t> ids{1, 2, 3, 4, 5};
    const auto float_logits = model.forward(ids).value();
    const auto int8_logits = q8.forward(ids);
    const auto int4_logits = q4.forward(ids);

    check(int8_logits.shape() == float_logits.shape(), "quantized LM int8 shape");
    check(int4_logits.shape() == float_logits.shape(), "quantized LM int4 shape");
    check(q8.quantized_bytes() < q8.float_reference_bytes(), "quantized LM int8 size");
    check(q4.quantized_bytes() < q4.float_reference_bytes(), "quantized LM int4 size");
    check(q8.compression_ratio() > 1.0, "quantized LM int8 compression");
    check(q4.compression_ratio() > q8.compression_ratio(), "quantized LM int4 compression");

    for (std::size_t i = 0; i < int8_logits.size(); ++i) {
        check(std::isfinite(int8_logits[i]), "quantized LM int8 finite");
        check(std::isfinite(int4_logits[i]), "quantized LM int4 finite");
    }

    zyron::model::GenerationConfig generation_config;
    generation_config.max_new_tokens = 3;
    generation_config.temperature = 1.0f;
    generation_config.top_k = 5;
    generation_config.seed = 123;
    const auto first = q8.generate(ids, generation_config);
    const auto second = q8.generate(ids, generation_config);
    check(first == second, "quantized LM deterministic generation");
}

} // namespace

int main() {
    try {
        test_memory_alignment_and_stats();
        test_runtime();
        test_optimized_matmul();
        test_quantization();
        test_quantized_linear();
        test_quantized_language_model();
        std::cout << "ZYRON Phase 5 tests: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ZYRON Phase 5 tests: FAIL: " << e.what() << '\n';
        return 1;
    }
}
