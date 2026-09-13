#include "zyron/core/tensor.hpp"
#include "zyron/math/ops.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using zyron::Shape;
using zyron::Tensor;
using zyron::math::add;
using zyron::math::div;
using zyron::math::exp;
using zyron::math::log;
using zyron::math::matmul;
using zyron::math::max;
using zyron::math::mean;
using zyron::math::min;
using zyron::math::mul;
using zyron::math::neg;
using zyron::math::pow;
using zyron::math::sqrt;
using zyron::math::sub;
using zyron::math::sum;

static void check(bool condition, const char* msg) {
    if (!condition) {
        throw std::runtime_error(msg);
    }
}

static void check_close(float actual, float expected, float eps, const char* msg) {
    if (std::fabs(actual - expected) > eps) {
        throw std::runtime_error(msg);
    }
}

template <typename Fn>
static void check_throws(Fn&& fn, const char* msg) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, msg);
}

static void test_shape_and_creation() {
    Tensor x(Shape{2, 3});
    check(x.rank() == 2, "rank");
    check(x.size() == 6, "size");
    check(x.nbytes() == 6 * sizeof(float), "bytes");
    check(x.storage_nbytes() == 6 * sizeof(float), "storage bytes");
    check(x.is_contiguous(), "contiguous");
}

static void test_shape_validation() {
    check_throws([] { (void)Shape{0, 3}; }, "zero dimension must throw");
    check_throws([] {
        (void)Shape{std::numeric_limits<std::size_t>::max(), 2};
    }, "shape overflow must throw");
}

static void test_strided_memory_validation() {
    check_throws([] {
        Tensor invalid(
            Shape{2, 2},
            std::vector<std::size_t>{3, 1},
            zyron::Memory(4 * sizeof(float)));
        (void)invalid;
    }, "strided storage validation");
}

static void test_initializers() {
    auto z = Tensor::zeros(Shape{2, 2});
    auto o = Tensor::ones(Shape{2, 2});
    for (std::size_t i = 0; i < 4; ++i) {
        check(z[i] == 0.0f, "zeros");
        check(o[i] == 1.0f, "ones");
    }
}

static void test_default_tensor_is_empty() {
    Tensor x;
    check(x.size() == 0, "default size");
    check(x.nbytes() == 0, "default bytes");
    check(x.storage_nbytes() == 0, "default storage bytes");
}

static void test_indexing() {
    Tensor x(Shape{2, 3});
    x.at({1, 2}) = 42.0f;
    check(x.at({1, 2}) == 42.0f, "indexing");
    check(x[5] == 42.0f, "linear indexing");
    check_throws([&] { (void)x.at({2, 0}); }, "row bounds");
    check_throws([&] { (void)x.at({0, 3}); }, "column bounds");
    check_throws([&] { (void)x.at({0}); }, "rank mismatch");
}

static void test_reshape() {
    Tensor x(Shape{2, 3});
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i);
    auto y = x.reshape(Shape{3, 2});
    check(y.shape() == Shape{3, 2}, "reshape shape");
    check(y[5] == 5.0f, "reshape data");
    check_throws([&] { (void)x.reshape(Shape{4, 2}); }, "reshape element count");
    auto t = x.transpose(0, 1);
    check_throws([&] { (void)t.reshape(Shape{3, 2}); }, "reshape non-contiguous");
}

static void test_transpose_contiguous_and_linear_view() {
    Tensor x(Shape{2, 3});
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i);

    auto t = x.transpose(0, 1);
    check(t.shape() == Shape{3, 2}, "transpose shape");
    check(!t.is_contiguous(), "transpose non-contiguous");
    check(t.at({0, 1}) == 3.0f, "transpose value");
    check(t[1] == 3.0f, "transpose linear view");

    auto c = t.contiguous();
    check(c.is_contiguous(), "contiguous result");
    check(c.at({0, 1}) == 3.0f, "contiguous value");
    check(c[1] == 3.0f, "contiguous linear value");
}

static void test_view_lifetime() {
    Tensor view;
    {
        Tensor x(Shape{2, 2});
        x[0] = 7.0f;
        view = x.transpose(0, 1);
    }
    check(view.at({0, 0}) == 7.0f, "view lifetime");
}

static void test_elementwise() {
    auto a = Tensor::ones(Shape{2, 3});
    auto b = Tensor::ones(Shape{2, 3});
    auto c = add(a, b);
    for (std::size_t i = 0; i < c.size(); ++i) check(c[i] == 2.0f, "add");
}

static void test_broadcasting() {
    Tensor a(Shape{2, 3});
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i + 1);

    Tensor row(Shape{3});
    row[0] = 10.0f;
    row[1] = 20.0f;
    row[2] = 30.0f;

    auto c = add(a, row);
    check(c.shape() == Shape{2, 3}, "broadcast row shape");
    check(c.at({0, 0}) == 11.0f, "broadcast row c00");
    check(c.at({0, 2}) == 33.0f, "broadcast row c02");
    check(c.at({1, 0}) == 14.0f, "broadcast row c10");
    check(c.at({1, 2}) == 36.0f, "broadcast row c12");

    Tensor col(Shape{2, 1});
    col[0] = 100.0f;
    col[1] = 200.0f;
    auto d = mul(a, col);
    check(d.shape() == Shape{2, 3}, "broadcast column shape");
    check(d.at({0, 1}) == 200.0f, "broadcast column d01");
    check(d.at({1, 2}) == 1200.0f, "broadcast column d12");

    Tensor scalar(Shape{});
    scalar[0] = 2.0f;
    auto e = sub(a, scalar);
    check(e.shape() == a.shape(), "scalar broadcast shape");
    check(e[0] == -1.0f, "scalar broadcast first");
    check(e[5] == 4.0f, "scalar broadcast last");

    Tensor incompatible(Shape{2, 2});
    check_throws([&] { (void)add(a, incompatible); }, "incompatible broadcast");

    auto transposed = a.transpose(0, 1);
    auto doubled = add(transposed, transposed);
    check(doubled.shape() == Shape{3, 2}, "view elementwise shape");
    check(doubled.at({0, 0}) == 2.0f, "view elementwise first");
    check(doubled.at({1, 0}) == 4.0f, "view elementwise second");
}

static void test_basic_math() {
    Tensor a(Shape{2, 2});
    Tensor b(Shape{2, 2});
    const float av[] = {1, 4, 9, 16};
    const float bv[] = {1, 2, 3, 4};
    for (std::size_t i = 0; i < 4; ++i) { a[i] = av[i]; b[i] = bv[i]; }

    auto s = sub(a, b);
    auto m = mul(a, b);
    auto d = div(a, b);
    auto n = neg(b);
    check(s[3] == 12.0f, "sub");
    check(m[2] == 27.0f, "mul");
    check(d[3] == 4.0f, "div");
    check(n[0] == -1.0f, "neg");

    check(sum(b)[0] == 10.0f, "sum");
    check(mean(b)[0] == 2.5f, "mean");
    check(max(b)[0] == 4.0f, "max");
    check(min(b)[0] == 1.0f, "min");

    auto t = b.transpose(0, 1);
    check(sum(t)[0] == 10.0f, "sum view");
    check(mean(t)[0] == 2.5f, "mean view");
    check(max(t)[0] == 4.0f, "max view");
    check(min(t)[0] == 1.0f, "min view");

    auto e = exp(Tensor::zeros(Shape{1}));
    auto l = log(Tensor::ones(Shape{1}));
    auto q = sqrt(Tensor::ones(Shape{1}));
    auto p = pow(b, 2.0f);
    check_close(e[0], 1.0f, 1e-6f, "exp");
    check_close(l[0], 0.0f, 1e-6f, "log");
    check_close(q[0], 1.0f, 1e-6f, "sqrt");
    check(p[3] == 16.0f, "pow");
}

static void test_matmul() {
    Tensor a(Shape{2, 3});
    Tensor b(Shape{3, 2});
    const float av[] = {1, 2, 3, 4, 5, 6};
    const float bv[] = {7, 8, 9, 10, 11, 12};
    for (std::size_t i = 0; i < 6; ++i) { a[i] = av[i]; b[i] = bv[i]; }
    auto c = matmul(a, b);
    check(c.shape() == Shape{2, 2}, "matmul shape");
    check_close(c.at({0, 0}), 58.0f, 1e-6f, "matmul c00");
    check_close(c.at({0, 1}), 64.0f, 1e-6f, "matmul c01");
    check_close(c.at({1, 0}), 139.0f, 1e-6f, "matmul c10");
    check_close(c.at({1, 1}), 154.0f, 1e-6f, "matmul c11");

    auto at = a.transpose(0, 1);
    auto bt = b.transpose(0, 1);
    auto c2 = matmul(at, bt);
    check(c2.shape() == Shape{3, 3}, "matmul transpose shape");
    check_close(c2.at({0, 0}), 39.0f, 1e-6f, "matmul transpose c00");
    check_close(c2.at({2, 2}), 105.0f, 1e-6f, "matmul transpose c22");

    check_throws([&] { (void)matmul(a, Tensor(Shape{4, 2})); }, "matmul incompatible dimensions");
    check_throws([&] { (void)matmul(Tensor(Shape{2, 3, 1}), b); }, "matmul rank mismatch");
}

int main() {
    try {
        test_shape_and_creation();
        test_shape_validation();
        test_strided_memory_validation();
        test_initializers();
        test_default_tensor_is_empty();
        test_indexing();
        test_reshape();
        test_transpose_contiguous_and_linear_view();
        test_view_lifetime();
        test_elementwise();
        test_broadcasting();
        test_basic_math();
        test_matmul();
        std::cout << "ZYRON Phase 1 tests: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "ZYRON Phase 1 tests: FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
