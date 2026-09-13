#include "zyron/autograd/autograd.hpp"
#include "zyron/model/generation.hpp"
#include "zyron/math/ops.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/tokenizer/bpe.hpp"
#include "zyron/transformer/attention.hpp"
#include "zyron/transformer/transformer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using zyron::Tensor;
using zyron::Shape;
using zyron::autograd::Variable;
using zyron::model::GenerationConfig;
using zyron::model::LanguageModel;
using zyron::tokenizer::BPE;
using zyron::transformer::Config;
using zyron::transformer::MultiHeadAttention;
using zyron::transformer::Transformer;

static void check(bool condition, const char* msg) {
    if (!condition) throw std::runtime_error(msg);
}

static void check_close(float a, float b, float eps, const char* msg) {
    if (std::fabs(a - b) > eps) throw std::runtime_error(msg);
}

static Config tiny_config() {
    Config config = zyron::transformer::tiny(270);
    config.hidden_size = 16;
    config.num_layers = 2;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.intermediate_size = 32;
    config.max_sequence_length = 16;
    return config;
}

static void test_gqa_and_weight_tying() {
    const Config config = tiny_config();
    LanguageModel model(config, 17);

    const auto& attention = model.transformer().block(0).attention();
    check(attention.num_heads() == 4, "GQA query head count");
    check(attention.num_kv_heads() == 2, "GQA KV head count");
    check(attention.k_proj().out_features() == 8, "GQA K projection size");
    check(attention.v_proj().out_features() == 8, "GQA V projection size");

    check(model.lm_head().is_weight_tied(), "LM head is not weight tied");
    check(model.lm_head().effective_weight().shape() == Shape{config.hidden_size, config.vocab_size},
          "tied LM head effective shape");
    check(model.lm_head().weight().node_handle().get() ==
              model.token_embedding().weight().node_handle().get(),
          "LM head and embedding do not share the same variable");

    std::size_t embedding_occurrences = 0;
    for (const auto* parameter : model.parameters()) {
        if (parameter == &model.token_embedding().weight()) ++embedding_occurrences;
    }
    check(embedding_occurrences == 1, "tied embedding appears more than once in parameters");
}

static void test_rope_changes_attention_positions() {
    Config rope = tiny_config();
    Config legacy = rope;
    legacy.use_rope = false;

    LanguageModel rope_model(rope, 44);
    LanguageModel legacy_model(legacy, 44);

    const std::vector<std::size_t> ids{1, 2, 3, 4, 5};
    zyron::autograd::NoGradGuard guard;
    const auto rope_logits = rope_model.forward(ids).value();
    const auto legacy_logits = legacy_model.forward(ids).value();

    bool differs = false;
    for (std::size_t i = 0; i < rope_logits.size(); ++i) {
        if (std::fabs(rope_logits[i] - legacy_logits[i]) > 1e-7f) {
            differs = true;
            break;
        }
    }
    check(differs, "RoPE does not change model positions");
}

static void test_weight_tying_gradient() {
    const Config config = tiny_config();
    LanguageModel model(config, 55);
    model.zero_grad();
    auto loss = model.loss(
        {{1, 2, 3, 4}, {4, 3, 2, 1}},
        {{2, 3, 4, 5}, {3, 2, 1, 0}});
    loss.backward();

    check(model.token_embedding().weight().has_grad(), "tied embedding gradient missing");
    check(std::isfinite(model.token_embedding().weight().grad()[0]),
          "tied embedding gradient is not finite");
}

static void test_batched_matmul() {
    Tensor a(Shape{2, 2, 3});
    Tensor b(Shape{2, 3, 2});

    for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i + 1);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>(i + 1);

    const Tensor c = zyron::math::matmul(a, b);
    check(c.shape() == Shape{2, 2, 2}, "batched matmul shape");
    check_close(c.at({0, 0, 0}), 22.0f, 1e-6f, "batched matmul c000");
    check_close(c.at({1, 1, 1}), 334.0f, 1e-6f, "batched matmul c111");
}

static void test_batched_matmul_autograd() {
    Tensor ar(Shape{2, 2, 3});
    Tensor br(Shape{2, 3, 2});
    for (std::size_t i = 0; i < ar.size(); ++i) ar[i] = 0.1f * static_cast<float>(i + 1);
    for (std::size_t i = 0; i < br.size(); ++i) br[i] = 0.07f * static_cast<float>(i + 1);

    Variable a(ar, true);
    Variable b(br, true);
    auto y = zyron::autograd::sum(zyron::autograd::matmul(a, b));
    y.backward();

    check(a.has_grad(), "batched matmul grad a");
    check(b.has_grad(), "batched matmul grad b");
    check(std::isfinite(a.grad()[0]), "batched matmul grad finite a");
    check(std::isfinite(b.grad()[0]), "batched matmul grad finite b");

    const float original = ar[0];
    constexpr float h = 1e-3f;
    ar[0] = original + h;
    const float plus = zyron::math::sum(zyron::math::matmul(ar, br))[0];
    ar[0] = original - h;
    const float minus = zyron::math::sum(zyron::math::matmul(ar, br))[0];
    ar[0] = original;
    const float numerical = (plus - minus) / (2.0f * h);
    check_close(a.grad()[0], numerical, 1e-3f, "batched matmul numeric gradient");
}

static void test_attention_shape_and_causality() {
    const Config config = tiny_config();
    MultiHeadAttention attention(config, 42);

    Tensor input(Shape{2, 5, config.hidden_size});
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.01f * static_cast<float>(i + 1);
    }

    Variable x(input, true);
    const auto output = attention.forward(x);
    check(output.value().shape() == Shape{2, 5, 16}, "attention output shape");

    auto loss = zyron::autograd::mean(
        zyron::autograd::mul(output, output));
    loss.backward();
    check(x.has_grad(), "attention input gradient");
}

static void test_transformer() {
    const Config config = tiny_config();
    Transformer transformer(config, 123);

    Tensor input(Shape{2, 6, config.hidden_size});
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.02f * std::sin(static_cast<float>(i));
    }

    Variable x(input, true);
    const auto output = transformer.forward(x);
    check(output.value().shape() == Shape{2, 6, config.hidden_size}, "transformer shape");
    check(transformer.parameters().size() > 0, "transformer parameters");

    auto loss = zyron::autograd::mean(output);
    loss.backward();
    check(x.has_grad(), "transformer input gradient");

    {
        zyron::autograd::NoGradGuard guard;
        Tensor first = input;
        Tensor changed = input;
        // Positions 4 and 5 are changed. With causal masking, positions 0..3
        // must remain unchanged.
        for (std::size_t pos = 4; pos < 6; ++pos) {
            for (std::size_t h = 0; h < config.hidden_size; ++h) {
                changed.at({0, pos, h}) += 1.0f;
            }
        }

        const auto first_output = transformer.forward(Variable(first, false));
        const auto changed_output = transformer.forward(Variable(changed, false));
        for (std::size_t pos = 0; pos < 4; ++pos) {
            for (std::size_t h = 0; h < config.hidden_size; ++h) {
                check_close(
                    first_output.value().at({0, pos, h}),
                    changed_output.value().at({0, pos, h}),
                    1e-5f,
                    "causal mask leakage");
            }
        }
    }

    bool threw = false;
    try {
        Tensor too_long(Shape{1, 17, config.hidden_size});
        Variable bad(too_long, false);
        (void)transformer.forward(bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "max sequence validation");
}

static void test_kv_cache_matches_full_forward() {
    const Config config = tiny_config();
    LanguageModel model(config, 1234);

    const std::vector<std::size_t> prompt{1, 2, 3, 4};
    const std::vector<std::size_t> generated{5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

    zyron::autograd::NoGradGuard guard;
    model.eval();

    LanguageModel::KVCache cache;
    auto cached = model.forward_cached(prompt, cache);
    check(cache.sequence_length == prompt.size(), "KV cache prompt length");

    auto compare_last = [&](const auto& cached_logits, const auto& full_ids, const char* msg) {
        const auto full = model.forward(full_ids);
        check(cached_logits.value().shape()[0] == 1, "cached batch size");
        check(cached_logits.value().shape()[2] == config.vocab_size, "cached vocab size");
        const std::size_t cached_base = (cached_logits.value().shape()[1] - 1) * config.vocab_size;
        const std::size_t full_base = (full.value().shape()[1] - 1) * config.vocab_size;
        for (std::size_t id = 0; id < config.vocab_size; ++id) {
            check_close(
                cached_logits.value()[cached_base + id],
                full.value()[full_base + id],
                1e-5f,
                msg);
        }
    };

    compare_last(cached, prompt, "KV cache prompt logits mismatch");

    std::vector<std::size_t> prefix = prompt;
    for (const auto token : generated) {
        prefix.push_back(token);
        cached = model.forward_cached({token}, cache);
        check(cache.sequence_length == prefix.size(), "KV cache sequence length");
        for (const auto& layer_cache : cache.layers) {
            check(layer_cache.key.shape()[2] >= prefix.size(), "KV cache capacity growth");
            check(layer_cache.value.shape()[2] >= prefix.size(), "KV value cache capacity growth");
        }
        compare_last(cached, prefix, "KV cache incremental logits mismatch");
    }
}


static void test_gqa_cached_zero_weights_are_finite() {
    Config config = tiny_config();
    config.num_layers = 1;
    config.vocab_size = 8;
    config.max_sequence_length = 8;
    config.num_heads = 4;
    config.num_kv_heads = 2;

    LanguageModel model(config, 123);
    for (auto* parameter : model.parameters()) {
        parameter->value() = Tensor::zeros(parameter->value().shape());
    }

    zyron::autograd::NoGradGuard guard;
    model.eval();
    LanguageModel::KVCache cache;
    const auto cached = model.forward_cached({0, 1, 2}, cache);
    check(cache.sequence_length == 3, "GQA zero-weight cache sequence length");
    for (std::size_t i = 0; i < cached.value().size(); ++i) {
        check(std::isfinite(cached.value()[i]), "GQA cached forward produced non-finite logits");
        check(cached.value()[i] == 0.0f, "GQA cached zero-weight logits are not zero");
    }

    const auto full = model.forward(std::vector<std::size_t>{0, 1, 2});
    for (std::size_t i = 0; i < full.value().size(); ++i) {
        check(std::isfinite(full.value()[i]), "GQA full forward produced non-finite logits");
        check(full.value()[i] == 0.0f, "GQA full zero-weight logits are not zero");
    }
}

static void test_no_repeat_ngram_generation() {
    Config config = tiny_config();
    config.vocab_size = 10;
    config.max_sequence_length = 8;
    LanguageModel model(config, 123);

    for (auto* parameter : model.parameters()) {
        Tensor zeros = Tensor::zeros(parameter->value().shape());
        parameter->value() = std::move(zeros);
    }

    GenerationConfig generation;
    generation.max_new_tokens = 4;
    generation.temperature = 1.0f;
    generation.seed = 77;
    generation.no_repeat_ngram_size = 1;

    BPE tokenizer;
    const auto output = zyron::model::generate(
        model, tokenizer, std::vector<std::size_t>{0}, generation);

    check(output.size() == 5, "no-repeat generation length");
    for (std::size_t i = 0; i < output.size(); ++i) {
        for (std::size_t j = i + 1; j < output.size(); ++j) {
            check(output[i] != output[j], "no-repeat 1-gram blocking failed");
        }
    }
}

static void test_language_model_and_loss() {
    const Config config = tiny_config();
    LanguageModel model(config, 99);

    const std::vector<std::vector<std::size_t>> input{
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1}};
    const std::vector<std::vector<std::size_t>> target{
        {2, 3, 4, 5, 6},
        {4, 3, 2, 1, 0}};

    const auto logits = model.forward(input);
    check(logits.value().shape() == Shape{2, 5, config.vocab_size}, "LM logits shape");

    auto loss = model.loss(input, target);
    check(std::isfinite(loss.value()[0]), "LM loss finite");
    loss.backward();

    const auto params = model.parameters();
    check(!params.empty(), "LM parameters");

    bool parameter_has_grad = false;
    for (const auto* parameter : params) {
        if (parameter->has_grad()) {
            parameter_has_grad = true;
            break;
        }
    }
    check(parameter_has_grad, "LM parameter gradients");
}

static void test_generation() {
    const Config config = tiny_config();
    LanguageModel model(config, 7);

    BPE tokenizer;
    tokenizer.train_text("abcdefgabcdefgabcdefg", config.vocab_size);

    GenerationConfig generation;
    generation.max_new_tokens = 4;
    generation.temperature = 0.9f;
    generation.top_k = 8;
    generation.top_p = 0.95f;
    generation.repetition_penalty = 1.05f;
    generation.seed = 5;

    const auto prompt = tokenizer.encode("abc");
    const auto output = zyron::model::generate(
        model,
        tokenizer,
        prompt,
        generation);
    const auto output_repeat = zyron::model::generate(
        model,
        tokenizer,
        prompt,
        generation);

    check(output == output_repeat, "generation deterministic seed");
    check(output.size() >= prompt.size(), "generation length");
    check(output.size() <= config.max_sequence_length, "generation context length");
    for (const auto id : output) {
        check(id < config.vocab_size, "generation token range");
    }
}

int main() {
    try {
        test_batched_matmul();
        test_batched_matmul_autograd();
        test_gqa_and_weight_tying();
        test_rope_changes_attention_positions();
        test_weight_tying_gradient();
        test_attention_shape_and_causality();
        test_transformer();
        test_kv_cache_matches_full_forward();
        test_gqa_cached_zero_weights_are_finite();
        test_language_model_and_loss();
        test_generation();
        test_no_repeat_ngram_generation();
        std::cout << "ZYRON Phase 3 transformer/model tests: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ZYRON Phase 3 transformer/model tests: FAIL: "
                  << e.what() << '\n';
        return 1;
    }
}
