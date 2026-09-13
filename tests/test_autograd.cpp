#include "zyron/autograd/autograd.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

using zyron::Shape;
using zyron::Tensor;
using zyron::autograd::Variable;
using zyron::autograd::add;
using zyron::autograd::div;
using zyron::autograd::exp;
using zyron::autograd::log;
using zyron::autograd::matmul;
using zyron::autograd::mean;
using zyron::autograd::mul;
using zyron::autograd::pow;
using zyron::autograd::sqrt;
using zyron::autograd::sub;
using zyron::autograd::sum;

static void check(bool condition, const char* msg) {
    if (!condition) throw std::runtime_error(msg);
}

static void check_close(float actual, float expected, float eps, const char* msg) {
    if (std::fabs(actual - expected) > eps) throw std::runtime_error(msg);
}

template <typename Fn>
static void check_throws(Fn&& fn, const char* msg) {
    bool threw = false;
    try { fn(); } catch (const std::exception&) { threw = true; }
    check(threw, msg);
}

static Variable make_vector(const std::initializer_list<float>& values, bool requires_grad = false) {
    Tensor x(Shape{values.size()});
    std::size_t i = 0;
    for (float v : values) x[i++] = v;
    return Variable(std::move(x), requires_grad);
}

static float scalar(const Variable& x) {
    check(x.value().size() == 1, "expected scalar");
    return x.value()[0];
}

static void test_leaf_and_backward() {
    auto x = make_vector({2.0f, 3.0f}, true);
    auto y = sum(x);

    y.backward();

    check(x.has_grad(), "leaf gradient missing");
    check_close(x.grad()[0], 1.0f, 1e-6f, "sum grad 0");
    check_close(x.grad()[1], 1.0f, 1e-6f, "sum grad 1");
}

static void test_branch_accumulation_and_zero_grad() {
    auto x = make_vector({2.0f, 3.0f}, true);
    auto y = sum(add(x, x));
    y.backward();

    check_close(x.grad()[0], 2.0f, 1e-6f, "branch grad 0");
    check_close(x.grad()[1], 2.0f, 1e-6f, "branch grad 1");

    x.zero_grad();
    check(!x.has_grad(), "zero_grad");
}

static void test_broadcast_gradient() {
    Tensor a_value(Shape{2, 3});
    for (std::size_t i = 0; i < a_value.size(); ++i) {
        a_value[i] = static_cast<float>(i + 1);
    }

    Tensor b_value(Shape{3});
    b_value[0] = 10.0f;
    b_value[1] = 20.0f;
    b_value[2] = 30.0f;

    Variable a(std::move(a_value), true);
    Variable b(std::move(b_value), true);

    auto loss = sum(mul(a, b));
    loss.backward();

    check_close(a.grad()[0], 10.0f, 1e-6f, "broadcast mul da0");
    check_close(a.grad()[1], 20.0f, 1e-6f, "broadcast mul da1");
    check_close(a.grad()[2], 30.0f, 1e-6f, "broadcast mul da2");
    check_close(a.grad()[3], 10.0f, 1e-6f, "broadcast mul da3");
    check_close(a.grad()[5], 30.0f, 1e-6f, "broadcast mul da5");

    check_close(b.grad()[0], 5.0f, 1e-6f, "broadcast mul db0");
    check_close(b.grad()[1], 7.0f, 1e-6f, "broadcast mul db1");
    check_close(b.grad()[2], 9.0f, 1e-6f, "broadcast mul db2");
}

static void test_scalar_calculus_ops() {
    auto x = make_vector({2.0f, 8.0f}, true);

    auto y = mean(exp(log(x)));
    y.backward();

    const float expected = 0.5f;
    check_close(x.grad()[0], expected, 1e-5f, "exp(log(x)) grad 0");
    check_close(x.grad()[1], expected, 1e-5f, "exp(log(x)) grad 1");

    x.zero_grad();
    auto z = sum(sqrt(x));
    z.backward();
    check_close(x.grad()[0], 1.0f / (2.0f * std::sqrt(2.0f)), 1e-5f, "sqrt grad 0");
    check_close(x.grad()[1], 1.0f / (2.0f * std::sqrt(8.0f)), 1e-5f, "sqrt grad 1");

    x.zero_grad();
    auto p = sum(pow(x, 3.0f));
    p.backward();
    check_close(x.grad()[0], 12.0f, 1e-5f, "pow grad 0");
    check_close(x.grad()[1], 192.0f, 1e-5f, "pow grad 1");
}

static void test_division_and_subtraction() {
    auto x = make_vector({4.0f, 6.0f}, true);
    auto y = make_vector({2.0f, 3.0f}, true);

    auto loss = sum(sub(div(x, y), y));
    loss.backward();

    check_close(x.grad()[0], 0.5f, 1e-6f, "div/sub dx0");
    check_close(x.grad()[1], 1.0f / 3.0f, 1e-6f, "div/sub dx1");
    check_close(y.grad()[0], -2.0f, 1e-6f, "div/sub dy0");
    check_close(y.grad()[1], -5.0f / 3.0f, 1e-6f, "div/sub dy1");
}

static void test_matmul_gradient() {
    Tensor av(Shape{2, 2});
    av[0] = 1.0f; av[1] = 2.0f;
    av[2] = 3.0f; av[3] = 4.0f;

    Tensor bv(Shape{2, 2});
    bv[0] = 5.0f; bv[1] = 6.0f;
    bv[2] = 7.0f; bv[3] = 8.0f;

    Variable a(std::move(av), true);
    Variable b(std::move(bv), true);

    auto loss = sum(matmul(a, b));
    loss.backward();

    check_close(a.grad()[0], 11.0f, 1e-6f, "matmul da00");
    check_close(a.grad()[1], 15.0f, 1e-6f, "matmul da01");
    check_close(a.grad()[2], 11.0f, 1e-6f, "matmul da10");
    check_close(a.grad()[3], 15.0f, 1e-6f, "matmul da11");

    check_close(b.grad()[0], 4.0f, 1e-6f, "matmul db00");
    check_close(b.grad()[1], 4.0f, 1e-6f, "matmul db01");
    check_close(b.grad()[2], 6.0f, 1e-6f, "matmul db10");
    check_close(b.grad()[3], 6.0f, 1e-6f, "matmul db11");
}

static Tensor copy_tensor_values(const Tensor& source) {
    Tensor copy(source.shape());
    for (std::size_t i = 0; i < source.size(); ++i) {
        copy[i] = source[i];
    }
    return copy;
}

static float evaluate_scalar(const Tensor& x_value) {
    Variable x(copy_tensor_values(x_value), false);
    return scalar(sum(exp(mul(x, x))));
}

static void test_gradient_checking() {
    Tensor raw(Shape{3});
    raw[0] = 0.5f;
    raw[1] = -1.25f;
    raw[2] = 2.0f;

    Variable x(raw, true);
    auto loss = sum(exp(mul(x, x)));
    loss.backward();

    constexpr float h = 1e-3f;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        Tensor plus = copy_tensor_values(raw);
        Tensor minus = copy_tensor_values(raw);
        plus[i] += h;
        minus[i] -= h;

        const float numerical =
            (evaluate_scalar(plus) - evaluate_scalar(minus)) / (2.0f * h);

        const float analytical = x.grad()[i];
        check_close(
            analytical,
            numerical,
            2e-2f,
            "gradient checking");
    }
}

static void test_repeated_backward_accumulates_leaf_gradients() {
    auto x = make_vector({2.0f, 3.0f}, true);
    auto y = sum(mul(x, x));

    y.backward();
    const float first0 = x.grad()[0];
    const float first1 = x.grad()[1];

    y.backward();
    check_close(x.grad()[0], 2.0f * first0, 1e-6f, "repeated backward grad 0");
    check_close(x.grad()[1], 2.0f * first1, 1e-6f, "repeated backward grad 1");
}

static void test_backward_validation() {
    auto x = make_vector({1.0f, 2.0f}, true);
    auto y = mul(x, x);

    check_throws([&] { y.backward(); }, "non-scalar backward must throw");

    Tensor bad_gradient(Shape{3});
    check_throws(
        [&] { y.backward(bad_gradient); },
        "gradient shape mismatch must throw");

    Variable no_grad(Tensor::ones(Shape{1}), false);
    check_throws([&] { no_grad.backward(); }, "backward without requires_grad");
}

int main() {
    try {
        test_leaf_and_backward();
        test_branch_accumulation_and_zero_grad();
        test_broadcast_gradient();
        test_scalar_calculus_ops();
        test_division_and_subtraction();
        test_matmul_gradient();
        test_gradient_checking();
        test_repeated_backward_accumulates_leaf_gradients();
        test_backward_validation();

        std::cout << "ZYRON Phase 2 autograd tests: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "ZYRON Phase 2 autograd tests: FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
