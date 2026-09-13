#include "zyron/nn/nn.hpp"
#include "zyron/nn/sequential.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using zyron::Shape;
using zyron::Tensor;
using zyron::autograd::Variable;
using zyron::autograd::cross_entropy;
using zyron::autograd::embedding;
using zyron::autograd::gelu;
using zyron::autograd::layer_norm;
using zyron::autograd::log_softmax;
using zyron::autograd::matmul;
using zyron::autograd::mean;
using zyron::autograd::mul;
using zyron::autograd::rms_norm;
using zyron::autograd::sigmoid;
using zyron::autograd::silu;
using zyron::autograd::softmax;
using zyron::autograd::tanh;
using zyron::nn::Embedding;
using zyron::nn::LayerNorm;
using zyron::nn::Linear;
using zyron::nn::RMSNorm;
using zyron::nn::Sequential;

static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static void check_close(float actual, float expected, float eps, const char* message) {
    if (std::fabs(actual - expected) > eps) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
static void check_throws(Fn&& fn, const char* message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, message);
}

static Tensor make_tensor(const Shape& shape, const std::vector<float>& values) {
    Tensor result(shape);
    check(result.size() == values.size(), "tensor value count mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) result[i] = values[i];
    return result;
}

static float scalar_value(const Variable& value) {
    check(value.value().size() == 1, "expected scalar variable");
    return value.value()[0];
}

static Tensor clone_tensor(const Tensor& source) {
    Tensor copy(source.shape());
    for (std::size_t i = 0; i < source.size(); ++i) {
        copy[i] = source[i];
    }
    return copy;
}

static float finite_difference(
    const Tensor& source,
    std::size_t index,
    float h,
    const auto& evaluate) {

    Tensor plus = clone_tensor(source);
    Tensor minus = clone_tensor(source);
    plus[index] += h;
    minus[index] -= h;
    return (evaluate(plus) - evaluate(minus)) / (2.0f * h);
}


static void test_dropout() {
    const Tensor input = Tensor::ones(Shape{256});
    zyron::nn::Dropout dropout(0.5f, 12345);

    auto eval_input = Variable(input, false);
    dropout.eval();
    const auto eval_output = dropout.forward(eval_input);
    for (std::size_t i = 0; i < eval_output.value().size(); ++i) {
        check_close(eval_output.value()[i], 1.0f, 1e-6f, "dropout eval must be identity");
    }

    dropout.train();
    auto train_input = Variable(input, true);
    const auto train_output = dropout.forward(train_input);
    std::size_t dropped = 0;
    std::size_t kept = 0;
    for (std::size_t i = 0; i < train_output.value().size(); ++i) {
        const float value = train_output.value()[i];
        if (value == 0.0f) ++dropped;
        else {
            ++kept;
            check_close(value, 2.0f, 1e-6f, "dropout inverted scaling");
        }
    }
    check(dropped > 0 && kept > 0, "dropout mask should contain dropped and kept values");

    auto loss = zyron::autograd::sum(train_output);
    loss.backward();
    for (std::size_t i = 0; i < train_input.grad().size(); ++i) {
        check(train_input.grad()[i] == 0.0f || train_input.grad()[i] == 2.0f,
              "dropout gradient mask");
    }

    check_throws(
        [] { zyron::nn::Dropout invalid(1.0f); },
        "dropout probability validation");
}

static void test_activation_derivatives() {
    const Tensor raw = make_tensor(Shape{4}, {-1.2f, -0.2f, 0.3f, 1.1f});
    const float h = 1e-3f;

    const auto check_activation = [&](const auto& activation,
                                      const auto& evaluate,
                                      const char* message,
                                      float tolerance) {
        Variable x(raw, true);
        auto y = activation(x);
        auto loss = zyron::autograd::sum(y);
        loss.backward();

        for (std::size_t i = 0; i < raw.size(); ++i) {
            const float numerical = finite_difference(raw, i, h, evaluate);
            check_close(x.grad()[i], numerical, tolerance, message);
        }
    };

    check_activation(
        [](const Variable& x) { return gelu(x); },
        [](const Tensor& x) {
            return scalar_value(zyron::autograd::sum(
                gelu(Variable(x, false))));
        },
        "GELU gradient",
        4e-3f);

    check_activation(
        [](const Variable& x) { return silu(x); },
        [](const Tensor& x) {
            return scalar_value(zyron::autograd::sum(
                silu(Variable(x, false))));
        },
        "SiLU gradient",
        4e-3f);

    check_activation(
        [](const Variable& x) { return sigmoid(x); },
        [](const Tensor& x) {
            return scalar_value(zyron::autograd::sum(
                sigmoid(Variable(x, false))));
        },
        "Sigmoid gradient",
        3e-3f);

    check_activation(
        [](const Variable& x) { return tanh(x); },
        [](const Tensor& x) {
            return scalar_value(zyron::autograd::sum(
                tanh(Variable(x, false))));
        },
        "Tanh gradient",
        3e-3f);
}

static void test_relu() {
    auto x = Variable(
        make_tensor(Shape{4}, {-1.0f, 0.0f, 2.0f, 3.0f}),
        true);

    auto loss = zyron::autograd::sum(zyron::autograd::relu(x));
    loss.backward();

    check(x.grad()[0] == 0.0f, "ReLU negative grad");
    check(x.grad()[1] == 0.0f, "ReLU zero grad");
    check(x.grad()[2] == 1.0f, "ReLU positive grad");
    check(x.grad()[3] == 1.0f, "ReLU positive grad 2");
}

static void test_softmax_and_log_softmax() {
    const Tensor raw = make_tensor(Shape{2, 3}, {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 1.5f});

    {
        Variable x(raw, true);
        auto s = softmax(x);
        auto loss = zyron::autograd::sum(mul(
            s,
            Variable(make_tensor(Shape{2, 3}, {0.3f, -0.7f, 1.2f, 0.4f, 0.8f, -0.2f}), false)));
        loss.backward();

        const float sum_row0 = s.value()[0] + s.value()[1] + s.value()[2];
        const float sum_row1 = s.value()[3] + s.value()[4] + s.value()[5];
        check_close(sum_row0, 1.0f, 1e-6f, "softmax row sum 0");
        check_close(sum_row1, 1.0f, 1e-6f, "softmax row sum 1");
        check(std::isfinite(x.grad()[0]), "softmax gradient finite");
    }

    {
        Variable x(raw, true);
        auto ls = log_softmax(x);
        auto loss = zyron::autograd::sum(ls);
        loss.backward();
        check(std::isfinite(x.grad()[0]), "log_softmax gradient finite");
    }
}

static void test_layer_norm_gradient() {
    const Tensor raw = make_tensor(Shape{2, 3}, {1.0f, -2.0f, 0.5f, 2.0f, 1.0f, -1.0f});
    const Tensor gamma_raw = make_tensor(Shape{3}, {1.2f, 0.8f, 1.5f});
    const Tensor beta_raw = make_tensor(Shape{3}, {0.1f, -0.2f, 0.3f});

    Variable x(raw, true);
    Variable gamma(gamma_raw, true);
    Variable beta(beta_raw, true);
    auto out = layer_norm(x, gamma, beta, 1e-5f);
    auto loss = mean(mul(out, out));
    loss.backward();

    const auto evaluate_x = [&](const Tensor& x_value) {
        Variable xv(x_value, false);
        Variable gv(gamma_raw, false);
        Variable bv(beta_raw, false);
        auto y = layer_norm(xv, gv, bv, 1e-5f);
        return scalar_value(mean(mul(y, y)));
    };

    const auto evaluate_gamma = [&](const Tensor& gamma_value) {
        Variable xv(raw, false);
        Variable gv(gamma_value, false);
        Variable bv(beta_raw, false);
        auto y = layer_norm(xv, gv, bv, 1e-5f);
        return scalar_value(mean(mul(y, y)));
    };

    const auto evaluate_beta = [&](const Tensor& beta_value) {
        Variable xv(raw, false);
        Variable gv(gamma_raw, false);
        Variable bv(beta_value, false);
        auto y = layer_norm(xv, gv, bv, 1e-5f);
        return scalar_value(mean(mul(y, y)));
    };

    constexpr float h = 1e-3f;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        check_close(
            x.grad()[i],
            finite_difference(raw, i, h, evaluate_x),
            8e-3f,
            "LayerNorm input gradient");
    }
    for (std::size_t i = 0; i < gamma_raw.size(); ++i) {
        check_close(
            gamma.grad()[i],
            finite_difference(gamma_raw, i, h, evaluate_gamma),
            8e-3f,
            "LayerNorm gamma gradient");
        check_close(
            beta.grad()[i],
            finite_difference(beta_raw, i, h, evaluate_beta),
            8e-3f,
            "LayerNorm beta gradient");
    }
}

static void test_rms_norm_gradient() {
    const Tensor raw = make_tensor(Shape{2, 3}, {1.0f, -2.0f, 0.5f, 2.0f, 1.0f, -1.0f});
    const Tensor gamma_raw = make_tensor(Shape{3}, {1.2f, 0.8f, 1.5f});

    Variable x(raw, true);
    Variable gamma(gamma_raw, true);
    auto out = rms_norm(x, gamma, 1e-5f);
    auto loss = mean(mul(out, out));
    loss.backward();

    const auto evaluate_x = [&](const Tensor& x_value) {
        Variable xv(x_value, false);
        Variable gv(gamma_raw, false);
        auto y = rms_norm(xv, gv, 1e-5f);
        return scalar_value(mean(mul(y, y)));
    };

    const auto evaluate_gamma = [&](const Tensor& gamma_value) {
        Variable xv(raw, false);
        Variable gv(gamma_value, false);
        auto y = rms_norm(xv, gv, 1e-5f);
        return scalar_value(mean(mul(y, y)));
    };

    constexpr float h = 1e-3f;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        check_close(
            x.grad()[i],
            finite_difference(raw, i, h, evaluate_x),
            8e-3f,
            "RMSNorm input gradient");
    }
    for (std::size_t i = 0; i < gamma_raw.size(); ++i) {
        check_close(
            gamma.grad()[i],
            finite_difference(gamma_raw, i, h, evaluate_gamma),
            8e-3f,
            "RMSNorm gamma gradient");
    }
}

static void test_cross_entropy() {
    const Tensor raw = make_tensor(
        Shape{2, 3},
        {1.0f, 2.0f, 0.5f, -1.0f, 2.5f, 0.0f});
    const std::vector<std::size_t> targets{1, 2};

    Variable logits(raw, true);
    auto loss = cross_entropy(logits, targets);
    loss.backward();

    check(std::isfinite(loss.value()[0]), "cross entropy finite");
    check(std::isfinite(logits.grad()[0]), "cross entropy gradient finite");

    const auto evaluate = [&](const Tensor& value) {
        return scalar_value(cross_entropy(Variable(value, false), targets));
    };

    constexpr float h = 1e-3f;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        check_close(
            logits.grad()[i],
            finite_difference(raw, i, h, evaluate),
            8e-3f,
            "cross entropy gradient");
    }
}

static void test_embedding_scatter() {
    const Tensor raw = make_tensor(
        Shape{4, 3},
        {0.1f, 0.2f, 0.3f,
         1.0f, 1.1f, 1.2f,
         2.0f, 2.1f, 2.2f,
         3.0f, 3.1f, 3.2f});

    Variable weight(raw, true);
    auto out = embedding(weight, {2, 1, 2});
    auto loss = zyron::autograd::sum(out);
    loss.backward();

    check_close(weight.grad()[0], 0.0f, 1e-6f, "embedding row 0");
    check_close(weight.grad()[3], 1.0f, 1e-6f, "embedding row 1");
    check_close(weight.grad()[6], 2.0f, 1e-6f, "embedding row 2");
    check_close(weight.grad()[9], 0.0f, 1e-6f, "embedding row 3");

    check_throws([&] {
        (void)embedding(weight, {4});
    }, "embedding bounds");
}

static void test_linear_and_sequential() {
    Linear linear(3, 2, true, 123);
    check(linear.parameters().size() == 2, "Linear parameter count");
    check(linear.in_features() == 3, "Linear in features");
    check(linear.out_features() == 2, "Linear out features");

    Variable input(make_tensor(Shape{2, 3}, {1, 2, 3, 4, 5, 6}), true);
    auto output = linear.forward(input);
    check(output.value().shape() == Shape{2, 2}, "Linear output shape");

    auto loss = zyron::autograd::sum(output);
    loss.backward();

    check(linear.weight().has_grad(), "Linear weight gradient");
    check(linear.bias() != nullptr && linear.bias()->has_grad(), "Linear bias gradient");

    Sequential network;
    network.emplace<Linear>(3, 4, true, 1);
    network.emplace<Linear>(4, 2, true, 2);
    check(network.size() == 2, "Sequential layer count");
    check(network.parameters().size() == 4, "Sequential parameter count");

    auto network_output = network.forward(input);
    check(network_output.value().shape() == Shape{2, 2}, "Sequential output shape");

    network.eval();
    check(!network.is_training(), "Sequential eval state");
    network.train();
    check(network.is_training(), "Sequential train state");
}

static void test_normalization_layers() {
    LayerNorm ln(3);
    RMSNorm rn(3);

    Variable x(make_tensor(Shape{2, 3}, {1, 2, 3, 4, 5, 6}), true);
    auto a = ln.forward(x);
    auto b = rn.forward(x);

    check(a.value().shape() == Shape{2, 3}, "LayerNorm output shape");
    check(b.value().shape() == Shape{2, 3}, "RMSNorm output shape");
    check(ln.parameters().size() == 2, "LayerNorm parameters");
    check(rn.parameters().size() == 1, "RMSNorm parameters");
}

static void test_no_grad_guard() {
    Variable x(make_tensor(Shape{2}, {1.0f, 2.0f}), true);
    {
        zyron::autograd::NoGradGuard guard;
        auto y = gelu(x);
        check(!y.requires_grad(), "NoGrad output requires_grad");
        check(y.node_handle().use_count() >= 1, "NoGrad node exists");
    }

    auto y = gelu(x);
    check(y.requires_grad(), "graph restored after NoGradGuard");
}

int main() {
    try {
        test_relu();
        test_dropout();
        test_activation_derivatives();
        test_softmax_and_log_softmax();
        test_layer_norm_gradient();
        test_rms_norm_gradient();
        test_cross_entropy();
        test_embedding_scatter();
        test_linear_and_sequential();
        test_normalization_layers();
        test_no_grad_guard();

        std::cout << "ZYRON Phase 2 NN tests: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "ZYRON Phase 2 NN tests: FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
