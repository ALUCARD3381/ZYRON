#include "zyron/autograd/autograd.hpp"
#include "zyron/math/ops.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace zyron::autograd {

struct Variable::Node {
    Tensor value;
    Tensor grad;
    bool has_grad{false};
    bool requires_grad{false};
    bool leaf{false};
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(const Tensor&)> backward_fn;
};

namespace {

using NodePtr = std::shared_ptr<Variable::Node>;
thread_local bool g_no_grad = false;

void require_initialized(const Variable& x) {
    if (!x.node_handle()) {
        throw std::logic_error("ZYRON autograd: uninitialized Variable");
    }
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* message) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(message);
    }
}

void accumulate_grad(const NodePtr& node, const Tensor& gradient) {
    if (!node || !node->requires_grad) return;

    if (!node->has_grad) {
        node->grad = gradient;
        node->has_grad = true;
        return;
    }

    node->grad = zyron::math::add(node->grad, gradient);
}

Tensor reduce_to_shape(const Tensor& gradient, const Shape& target_shape) {
    if (gradient.shape() == target_shape) return gradient;

    if (target_shape.rank() > gradient.rank()) {
        throw std::invalid_argument(
            "ZYRON autograd: cannot reduce gradient to higher rank");
    }

    Tensor out = Tensor::zeros(target_shape);
    const std::size_t rank_delta = gradient.rank() - target_shape.rank();

    for (std::size_t linear = 0; linear < gradient.size(); ++linear) {
        std::size_t remaining = linear;
        std::vector<std::size_t> grad_indices(gradient.rank(), 0);

        for (std::size_t dim = gradient.rank(); dim-- > 0;) {
            grad_indices[dim] = remaining % gradient.shape()[dim];
            remaining /= gradient.shape()[dim];
        }

        std::vector<std::size_t> target_indices(target_shape.rank(), 0);
        for (std::size_t dim = 0; dim < target_shape.rank(); ++dim) {
            const std::size_t source_dim = dim + rank_delta;
            target_indices[dim] = target_shape[dim] == 1
                ? 0
                : grad_indices[source_dim];
        }

        if (target_shape.rank() == 0) {
            out[0] += gradient[linear];
        } else {
            out.at(target_indices) += gradient[linear];
        }
    }

    return out;
}

Tensor broadcast_scalar(const Tensor& scalar, const Shape& target_shape) {
    if (scalar.size() != 1) {
        throw std::invalid_argument(
            "ZYRON autograd: expected scalar gradient");
    }

    Tensor out(target_shape);
    std::fill_n(out.data(), out.size(), scalar[0]);
    return out;
}

NodePtr make_node(
    Tensor value,
    bool requires_grad,
    bool leaf,
    std::vector<NodePtr> parents,
    std::function<void(const Tensor&)> backward_fn = {}) {

    auto node = std::make_shared<Variable::Node>();
    node->value = std::move(value);
    node->requires_grad = requires_grad;
    node->leaf = leaf;
    node->parents = std::move(parents);
    node->backward_fn = std::move(backward_fn);
    return node;
}

Variable no_grad_result(Tensor value) {
    return Variable(std::move(value), false);
}

bool should_build_graph(const Variable& x) {
    return !NoGradGuard::enabled() && x.requires_grad();
}

bool should_build_graph(const Variable& a, const Variable& b) {
    return !NoGradGuard::enabled() &&
        (a.requires_grad() || b.requires_grad());
}

void build_topological_order(
    const NodePtr& node,
    std::unordered_set<const Variable::Node*>& visited,
    std::vector<NodePtr>& order) {

    if (!node || !visited.insert(node.get()).second) return;

    for (const auto& parent : node->parents) {
        build_topological_order(parent, visited, order);
    }

    order.push_back(node);
}

void clear_non_leaf_grads(const std::vector<NodePtr>& order) {
    for (const auto& node : order) {
        if (!node->leaf) {
            node->grad = Tensor{};
            node->has_grad = false;
        }
    }
}

void require_rank_at_least(const Tensor& x, std::size_t minimum_rank, const char* message) {
    if (x.rank() < minimum_rank) throw std::invalid_argument(message);
}

std::size_t rows_for_last_dim(const Tensor& x, std::size_t last_dim) {
    if (last_dim == 0 || x.size() % last_dim != 0) {
        throw std::invalid_argument("ZYRON autograd: invalid last dimension");
    }
    return x.size() / last_dim;
}

void check_normalization_shapes(
    const Tensor& x,
    const Tensor& gamma,
    const Tensor* beta) {

    require_rank_at_least(
        x,
        1,
        "ZYRON autograd: normalization requires rank >= 1");

    if (gamma.rank() != 1 || gamma.size() != x.shape()[x.rank() - 1]) {
        throw std::invalid_argument(
            "ZYRON autograd: normalization gamma shape mismatch");
    }

    if (beta && (beta->rank() != 1 || beta->size() != gamma.size())) {
        throw std::invalid_argument(
            "ZYRON autograd: normalization beta shape mismatch");
    }
}

float gelu_derivative(float x) {
    constexpr float inv_sqrt_2 = 0.7071067811865475244f;
    constexpr float inv_sqrt_2pi = 0.3989422804014326779f;
    const float cdf = 0.5f * (1.0f + std::erf(x * inv_sqrt_2));
    const float pdf = std::exp(-0.5f * x * x) * inv_sqrt_2pi;
    return cdf + x * pdf;
}

} // namespace

NoGradGuard::NoGradGuard() noexcept
    : previous_(g_no_grad) {
    g_no_grad = true;
}

NoGradGuard::~NoGradGuard() {
    g_no_grad = previous_;
}

bool NoGradGuard::enabled() noexcept {
    return g_no_grad;
}

Variable::Variable(Tensor value, bool requires_grad)
    : node_(make_node(
        std::move(value),
        requires_grad && !NoGradGuard::enabled(),
        true,
        {})) {}

Variable::~Variable() = default;

Variable::Variable(std::shared_ptr<Node> node)
    : node_(std::move(node)) {}

const Tensor& Variable::value() const {
    require_initialized(*this);
    return node_->value;
}

Tensor& Variable::value() {
    require_initialized(*this);
    return node_->value;
}

bool Variable::requires_grad() const noexcept {
    return node_ && node_->requires_grad;
}

bool Variable::has_grad() const noexcept {
    return node_ && node_->has_grad;
}

const Tensor& Variable::grad() const {
    require_initialized(*this);

    if (!node_->has_grad) {
        throw std::logic_error("ZYRON autograd: gradient is not available");
    }

    return node_->grad;
}

Tensor& Variable::grad() {
    require_initialized(*this);

    if (!node_->has_grad) {
        throw std::logic_error("ZYRON autograd: gradient is not available");
    }

    return node_->grad;
}

void Variable::zero_grad() noexcept {
    if (!node_) return;
    node_->grad = Tensor{};
    node_->has_grad = false;
}

void Variable::backward() {
    require_initialized(*this);

    if (!node_->requires_grad) {
        throw std::invalid_argument(
            "ZYRON autograd: backward requires requires_grad=true");
    }

    if (node_->value.size() != 1) {
        throw std::invalid_argument(
            "ZYRON autograd: backward() without a gradient requires a scalar output");
    }

    backward(Tensor::ones(Shape{}));
}

void Variable::backward(const Tensor& gradient) {
    require_initialized(*this);

    if (!node_->requires_grad) {
        throw std::invalid_argument(
            "ZYRON autograd: backward requires requires_grad=true");
    }

    require_same_shape(
        gradient,
        node_->value,
        "ZYRON autograd: supplied gradient shape mismatch");

    std::unordered_set<const Node*> visited;
    std::vector<NodePtr> order;
    build_topological_order(node_, visited, order);

    clear_non_leaf_grads(order);
    accumulate_grad(node_, gradient);

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const auto& current = *it;
        if (!current->has_grad || !current->backward_fn) continue;
        current->backward_fn(current->grad);
    }
}

Variable add(const Variable& a, const Variable& b) {
    require_initialized(a);
    require_initialized(b);

    Tensor value = zyron::math::add(a.value(), b.value());
    if (!should_build_graph(a, b)) return no_grad_result(std::move(value));

    const NodePtr a_node = a.node_handle();
    const NodePtr b_node = b.node_handle();
    const Shape a_shape = a.value().shape();
    const Shape b_shape = b.value().shape();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {a_node, b_node},
        [a_node, b_node, a_shape, b_shape](const Tensor& gradient) {
            if (a_node->requires_grad) {
                accumulate_grad(a_node, reduce_to_shape(gradient, a_shape));
            }
            if (b_node->requires_grad) {
                accumulate_grad(b_node, reduce_to_shape(gradient, b_shape));
            }
        }));
}

Variable sub(const Variable& a, const Variable& b) {
    require_initialized(a);
    require_initialized(b);

    Tensor value = zyron::math::sub(a.value(), b.value());
    if (!should_build_graph(a, b)) return no_grad_result(std::move(value));

    const NodePtr a_node = a.node_handle();
    const NodePtr b_node = b.node_handle();
    const Shape a_shape = a.value().shape();
    const Shape b_shape = b.value().shape();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {a_node, b_node},
        [a_node, b_node, a_shape, b_shape](const Tensor& gradient) {
            if (a_node->requires_grad) {
                accumulate_grad(a_node, reduce_to_shape(gradient, a_shape));
            }
            if (b_node->requires_grad) {
                auto negative = zyron::math::neg(gradient);
                accumulate_grad(b_node, reduce_to_shape(negative, b_shape));
            }
        }));
}

Variable mul(const Variable& a, const Variable& b) {
    require_initialized(a);
    require_initialized(b);

    Tensor value = zyron::math::mul(a.value(), b.value());
    if (!should_build_graph(a, b)) return no_grad_result(std::move(value));

    const NodePtr a_node = a.node_handle();
    const NodePtr b_node = b.node_handle();
    const Shape a_shape = a.value().shape();
    const Shape b_shape = b.value().shape();
    const Tensor a_value = a.value();
    const Tensor b_value = b.value();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {a_node, b_node},
        [a_node, b_node, a_shape, b_shape, a_value, b_value](const Tensor& gradient) {
            if (a_node->requires_grad) {
                auto grad_a = zyron::math::mul(gradient, b_value);
                accumulate_grad(a_node, reduce_to_shape(grad_a, a_shape));
            }
            if (b_node->requires_grad) {
                auto grad_b = zyron::math::mul(gradient, a_value);
                accumulate_grad(b_node, reduce_to_shape(grad_b, b_shape));
            }
        }));
}

Variable div(const Variable& a, const Variable& b) {
    require_initialized(a);
    require_initialized(b);

    Tensor value = zyron::math::div(a.value(), b.value());
    if (!should_build_graph(a, b)) return no_grad_result(std::move(value));

    const NodePtr a_node = a.node_handle();
    const NodePtr b_node = b.node_handle();
    const Shape a_shape = a.value().shape();
    const Shape b_shape = b.value().shape();
    const Tensor a_value = a.value();
    const Tensor b_value = b.value();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {a_node, b_node},
        [a_node, b_node, a_shape, b_shape, a_value, b_value](const Tensor& gradient) {
            if (a_node->requires_grad) {
                auto grad_a = zyron::math::div(gradient, b_value);
                accumulate_grad(a_node, reduce_to_shape(grad_a, a_shape));
            }
            if (b_node->requires_grad) {
                auto denominator = zyron::math::mul(b_value, b_value);
                auto numerator = zyron::math::mul(gradient, a_value);
                auto grad_b = zyron::math::neg(
                    zyron::math::div(numerator, denominator));
                accumulate_grad(b_node, reduce_to_shape(grad_b, b_shape));
            }
        }));
}

Variable neg(const Variable& x) {
    require_initialized(x);

    Tensor value = zyron::math::neg(x.value());
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node](const Tensor& gradient) {
            accumulate_grad(x_node, zyron::math::neg(gradient));
        }));
}

Variable sum(const Variable& x) {
    require_initialized(x);

    Tensor value = zyron::math::sum(x.value());
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();
    const Shape x_shape = x.value().shape();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node, x_shape](const Tensor& gradient) {
            accumulate_grad(x_node, broadcast_scalar(gradient, x_shape));
        }));
}

Variable mean(const Variable& x) {
    require_initialized(x);

    Tensor value = zyron::math::mean(x.value());
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();
    const Shape x_shape = x.value().shape();
    const float scale = 1.0f / static_cast<float>(x.value().size());

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node, x_shape, scale](const Tensor& gradient) {
            Tensor scaled = Tensor::zeros(Shape{});
            scaled[0] = gradient[0] * scale;
            accumulate_grad(x_node, broadcast_scalar(scaled, x_shape));
        }));
}

Variable exp(const Variable& x) {
    require_initialized(x);

    Tensor value = zyron::math::exp(x.value());
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();
    const Tensor output = value;

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node, output](const Tensor& gradient) {
            accumulate_grad(x_node, zyron::math::mul(gradient, output));
        }));
}

Variable log(const Variable& x) {
    require_initialized(x);

    Tensor value = zyron::math::log(x.value());
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();
    const Tensor input = x.value();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node, input](const Tensor& gradient) {
            accumulate_grad(x_node, zyron::math::div(gradient, input));
        }));
}

Variable sqrt(const Variable& x) {
    require_initialized(x);

    Tensor value = zyron::math::sqrt(x.value());
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();
    const Tensor output = value;

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node, output](const Tensor& gradient) {
            Tensor two = Tensor::zeros(Shape{});
            two[0] = 2.0f;
            auto denominator = zyron::math::mul(two, output);
            accumulate_grad(x_node, zyron::math::div(gradient, denominator));
        }));
}

Variable pow(const Variable& x, float exponent) {
    require_initialized(x);

    Tensor value = zyron::math::pow(x.value(), exponent);
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();
    const Tensor input = x.value();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node, input, exponent](const Tensor& gradient) {
            Tensor coefficient = Tensor::zeros(Shape{});
            coefficient[0] = exponent;
            auto power = zyron::math::pow(input, exponent - 1.0f);
            auto local = zyron::math::mul(coefficient, power);
            auto grad = zyron::math::mul(gradient, local);
            accumulate_grad(x_node, grad);
        }));
}

Variable matmul(const Variable& a, const Variable& b) {
    require_initialized(a);
    require_initialized(b);

    Tensor value = zyron::math::matmul(a.value(), b.value());
    if (!should_build_graph(a, b)) return no_grad_result(std::move(value));

    if (a.value().rank() < 2 || b.value().rank() < 2 ||
        a.value().rank() != b.value().rank()) {
        throw std::invalid_argument(
            "ZYRON autograd: MatMul requires equal ranks >= 2");
    }

    for (std::size_t dim = 0; dim + 2 < a.value().rank(); ++dim) {
        if (a.value().shape()[dim] != b.value().shape()[dim]) {
            throw std::invalid_argument(
                "ZYRON autograd: MatMul batch dimensions must match");
        }
    }

    const NodePtr a_node = a.node_handle();
    const NodePtr b_node = b.node_handle();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {a_node, b_node},
        [a_node, b_node](const Tensor& gradient) {
            const std::size_t last = gradient.rank() - 1;
            const std::size_t second_last = gradient.rank() - 2;

            if (a_node->requires_grad) {
                auto grad_a = zyron::math::matmul(
                    gradient,
                    b_node->value.transpose(second_last, last));
                accumulate_grad(a_node, grad_a);
            }

            if (b_node->requires_grad) {
                auto grad_b = zyron::math::matmul(
                    a_node->value.transpose(second_last, last),
                    gradient);
                accumulate_grad(b_node, grad_b);
            }
        }));
}


Variable reshape(const Variable& x, const Shape& new_shape) {
    require_initialized(x);

    Tensor value = x.value().reshape(new_shape);
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();
    const Shape old_shape = x.value().shape();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node, old_shape](const Tensor& gradient) {
            Tensor contiguous_gradient = gradient.contiguous();
            accumulate_grad(
                x_node,
                contiguous_gradient.reshape(old_shape));
        }));
}

Variable transpose(const Variable& x, std::size_t dim0, std::size_t dim1) {
    require_initialized(x);

    Tensor value = x.value().transpose(dim0, dim1);
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node, dim0, dim1](const Tensor& gradient) {
            accumulate_grad(x_node, gradient.transpose(dim0, dim1));
        }));
}

Variable contiguous(const Variable& x) {
    require_initialized(x);

    Tensor value = x.value().contiguous();
    if (!should_build_graph(x)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();

    return Variable(make_node(
        std::move(value),
        true,
        false,
        {x_node},
        [x_node](const Tensor& gradient) {
            accumulate_grad(x_node, gradient);
        }));
}


Variable rotary_embedding(
    const Variable& x,
    const Tensor& cos_table,
    const Tensor& sin_table,
    std::size_t start_position) {

    require_initialized(x);
    if (x.value().rank() != 4) {
        throw std::invalid_argument(
            "ZYRON autograd: rotary_embedding expects [batch, heads, sequence, head_dim]");
    }
    const std::size_t sequence = x.value().shape()[2];
    const std::size_t head_dim = x.value().shape()[3];
    if ((head_dim % 2) != 0) {
        throw std::invalid_argument("ZYRON autograd: rotary_embedding requires even head_dim");
    }
    if (cos_table.rank() != 2 || sin_table.rank() != 2 ||
        cos_table.shape() != sin_table.shape() ||
        cos_table.shape()[1] != head_dim / 2 ||
        start_position > cos_table.shape()[0] ||
        sequence > cos_table.shape()[0] - start_position) {
        throw std::invalid_argument("ZYRON autograd: invalid rotary embedding tables or range");
    }

    Tensor value(x.value().shape());
    const std::size_t batch = x.value().shape()[0];
    const std::size_t heads = x.value().shape()[1];
    const std::size_t half = head_dim / 2;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < heads; ++h) {
            for (std::size_t pos = 0; pos < sequence; ++pos) {
                const std::size_t base = ((b * heads + h) * sequence + pos) * head_dim;
                for (std::size_t pair = 0; pair < half; ++pair) {
                    const float c = cos_table.at({start_position + pos, pair});
                    const float sn = sin_table.at({start_position + pos, pair});
                    const float even = x.value().at({b, h, pos, 2 * pair});
                    const float odd = x.value().at({b, h, pos, 2 * pair + 1});
                    value[base + 2 * pair] = even * c - odd * sn;
                    value[base + 2 * pair + 1] = even * sn + odd * c;
                }
            }
        }
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node, cos_table, sin_table, start_position, sequence, head_dim](const Tensor& gradient) {
            Tensor grad_x(gradient.shape());
            const std::size_t batch_local = gradient.shape()[0];
            const std::size_t heads_local = gradient.shape()[1];
            const std::size_t half_local = head_dim / 2;
            for (std::size_t b = 0; b < batch_local; ++b) {
                for (std::size_t h = 0; h < heads_local; ++h) {
                    for (std::size_t pos = 0; pos < sequence; ++pos) {
                        const std::size_t base = ((b * heads_local + h) * sequence + pos) * head_dim;
                        for (std::size_t pair = 0; pair < half_local; ++pair) {
                            const float c = cos_table.at({start_position + pos, pair});
                            const float sn = sin_table.at({start_position + pos, pair});
                            const float grad_even = gradient.at({b, h, pos, 2 * pair});
                            const float grad_odd = gradient.at({b, h, pos, 2 * pair + 1});
                            grad_x[base + 2 * pair] = grad_even * c + grad_odd * sn;
                            grad_x[base + 2 * pair + 1] = -grad_even * sn + grad_odd * c;
                        }
                    }
                }
            }
            accumulate_grad(x_node, grad_x);
        }));
}

Variable repeat_kv_heads(const Variable& x, std::size_t repeats) {
    require_initialized(x);
    if (x.value().rank() != 4) {
        throw std::invalid_argument(
            "ZYRON autograd: repeat_kv_heads expects [batch, kv_heads, sequence, head_dim]");
    }
    if (repeats == 0) {
        throw std::invalid_argument("ZYRON autograd: repeat_kv_heads repeats must be > 0");
    }

    const std::size_t batch = x.value().shape()[0];
    const std::size_t kv_heads = x.value().shape()[1];
    const std::size_t sequence = x.value().shape()[2];
    const std::size_t head_dim = x.value().shape()[3];
    if (kv_heads > std::numeric_limits<std::size_t>::max() / repeats) {
        throw std::overflow_error("ZYRON autograd: KV head repeat overflow");
    }
    Tensor value(Shape{batch, kv_heads * repeats, sequence, head_dim});

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t kv = 0; kv < kv_heads; ++kv) {
            for (std::size_t group = 0; group < repeats; ++group) {
                const std::size_t out_head = kv * repeats + group;
                for (std::size_t pos = 0; pos < sequence; ++pos) {
                    const std::size_t dst = ((b * (kv_heads * repeats) + out_head) * sequence + pos) * head_dim;
                    for (std::size_t d = 0; d < head_dim; ++d) {
                        value[dst + d] = x.value().at({b, kv, pos, d});
                    }
                }
            }
        }
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node, batch, kv_heads, sequence, head_dim, repeats](const Tensor& gradient) {
            Tensor grad_x = Tensor::zeros(Shape{batch, kv_heads, sequence, head_dim});
            for (std::size_t b = 0; b < batch; ++b) {
                for (std::size_t kv = 0; kv < kv_heads; ++kv) {
                    for (std::size_t group = 0; group < repeats; ++group) {
                        const std::size_t in_head = kv * repeats + group;
                        for (std::size_t pos = 0; pos < sequence; ++pos) {
                            const std::size_t src = ((b * (kv_heads * repeats) + in_head) * sequence + pos) * head_dim;
                            const std::size_t dst = ((b * kv_heads + kv) * sequence + pos) * head_dim;
                            for (std::size_t d = 0; d < head_dim; ++d) {
                                grad_x.data()[dst + d] += gradient.data()[src + d];
                            }
                        }
                    }
                }
            }
            accumulate_grad(x_node, grad_x);
        }));
}

Variable fake_quantize(const Variable& x, std::size_t bits) {
    require_initialized(x);
    if (bits != 4 && bits != 8) {
        throw std::invalid_argument("ZYRON autograd: fake_quantize bits must be 4 or 8");
    }

    const int qmax = bits == 8 ? 127 : 7;
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < x.value().size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(x.value()[i]));
    }
    const float scale = max_abs > 0.0f ? max_abs / static_cast<float>(qmax) : 1.0f;

    Tensor value(x.value().shape());
    for (std::size_t i = 0; i < x.value().size(); ++i) {
        const long rounded = std::lround(x.value()[i] / scale);
        const long clamped = std::clamp(
            rounded,
            static_cast<long>(-qmax),
            static_cast<long>(qmax));
        value[i] = static_cast<float>(clamped) * scale;
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node](const Tensor& gradient) {
            // Straight-through estimator: forward simulates quantization while
            // backward treats the quantizer as the identity map.
            accumulate_grad(x_node, gradient);
        }));
}

Variable relu(const Variable& x) {
    require_initialized(x);
    Tensor value(x.value().shape());
    for (std::size_t i = 0; i < x.value().size(); ++i) {
        value[i] = std::max(0.0f, x.value()[i]);
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    const Tensor input = x.value();

    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node, input](const Tensor& gradient) {
            Tensor grad_x(input.shape());
            for (std::size_t i = 0; i < input.size(); ++i) {
                grad_x[i] = input[i] > 0.0f ? gradient[i] : 0.0f;
            }
            accumulate_grad(x_node, grad_x);
        }));
}

Variable gelu(const Variable& x) {
    require_initialized(x);
    constexpr float inv_sqrt_2 = 0.7071067811865475244f;
    Tensor value(x.value().shape());

    for (std::size_t i = 0; i < x.value().size(); ++i) {
        const float xi = x.value()[i];
        value[i] = 0.5f * xi * (1.0f + std::erf(xi * inv_sqrt_2));
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    const Tensor input = x.value();

    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node, input](const Tensor& gradient) {
            Tensor grad_x(input.shape());
            for (std::size_t i = 0; i < input.size(); ++i) {
                grad_x[i] = gradient[i] * gelu_derivative(input[i]);
            }
            accumulate_grad(x_node, grad_x);
        }));
}

Variable silu(const Variable& x) {
    require_initialized(x);
    Tensor value(x.value().shape());

    for (std::size_t i = 0; i < x.value().size(); ++i) {
        const float xi = x.value()[i];
        const float s = 1.0f / (1.0f + std::exp(-xi));
        value[i] = xi * s;
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    const Tensor input = x.value();

    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node, input](const Tensor& gradient) {
            Tensor grad_x(input.shape());
            for (std::size_t i = 0; i < input.size(); ++i) {
                const float xi = input[i];
                const float s = 1.0f / (1.0f + std::exp(-xi));
                grad_x[i] = gradient[i] * (s * (1.0f + xi * (1.0f - s)));
            }
            accumulate_grad(x_node, grad_x);
        }));
}

Variable sigmoid(const Variable& x) {
    require_initialized(x);
    Tensor value(x.value().shape());

    for (std::size_t i = 0; i < x.value().size(); ++i) {
        value[i] = 1.0f / (1.0f + std::exp(-x.value()[i]));
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    const Tensor output = value;

    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node, output](const Tensor& gradient) {
            Tensor grad_x(output.shape());
            for (std::size_t i = 0; i < output.size(); ++i) {
                grad_x[i] = gradient[i] * output[i] * (1.0f - output[i]);
            }
            accumulate_grad(x_node, grad_x);
        }));
}

Variable tanh(const Variable& x) {
    require_initialized(x);
    Tensor value(x.value().shape());
    for (std::size_t i = 0; i < x.value().size(); ++i) {
        value[i] = std::tanh(x.value()[i]);
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    const Tensor output = value;

    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node, output](const Tensor& gradient) {
            Tensor grad_x(output.shape());
            for (std::size_t i = 0; i < output.size(); ++i) {
                grad_x[i] = gradient[i] * (1.0f - output[i] * output[i]);
            }
            accumulate_grad(x_node, grad_x);
        }));
}

Variable softmax(const Variable& x) {
    require_initialized(x);
    require_rank_at_least(
        x.value(),
        1,
        "ZYRON autograd: softmax requires rank >= 1");

    const std::size_t classes = x.value().shape()[x.value().rank() - 1];
    const std::size_t rows = rows_for_last_dim(x.value(), classes);
    Tensor value(x.value().shape());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * classes;
        float max_value = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < classes; ++j) {
            max_value = std::max(max_value, x.value()[base + j]);
        }

        float denom = 0.0f;
        for (std::size_t j = 0; j < classes; ++j) {
            value[base + j] = std::exp(x.value()[base + j] - max_value);
            denom += value[base + j];
        }

        for (std::size_t j = 0; j < classes; ++j) {
            value[base + j] /= denom;
        }
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    const Tensor output = value;

    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node, output, classes](const Tensor& gradient) {
            Tensor grad_x(output.shape());
            const std::size_t rows_local = output.size() / classes;

            for (std::size_t row = 0; row < rows_local; ++row) {
                const std::size_t base = row * classes;
                float dot = 0.0f;
                for (std::size_t j = 0; j < classes; ++j) {
                    dot += gradient[base + j] * output[base + j];
                }
                for (std::size_t j = 0; j < classes; ++j) {
                    grad_x[base + j] =
                        output[base + j] * (gradient[base + j] - dot);
                }
            }

            accumulate_grad(x_node, grad_x);
        }));
}

Variable log_softmax(const Variable& x) {
    require_initialized(x);
    require_rank_at_least(
        x.value(),
        1,
        "ZYRON autograd: log_softmax requires rank >= 1");

    const std::size_t classes = x.value().shape()[x.value().rank() - 1];
    const std::size_t rows = rows_for_last_dim(x.value(), classes);
    Tensor value(x.value().shape());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * classes;
        float max_value = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < classes; ++j) {
            max_value = std::max(max_value, x.value()[base + j]);
        }

        float sum_exp = 0.0f;
        for (std::size_t j = 0; j < classes; ++j) {
            sum_exp += std::exp(x.value()[base + j] - max_value);
        }

        const float log_sum_exp = max_value + std::log(sum_exp);
        for (std::size_t j = 0; j < classes; ++j) {
            value[base + j] = x.value()[base + j] - log_sum_exp;
        }
    }

    if (!should_build_graph(x)) return no_grad_result(std::move(value));
    const NodePtr x_node = x.node_handle();
    const Tensor output = value;

    return Variable(make_node(
        std::move(value), true, false, {x_node},
        [x_node, output, classes](const Tensor& gradient) {
            Tensor grad_x(output.shape());
            const std::size_t rows_local = output.size() / classes;

            for (std::size_t row = 0; row < rows_local; ++row) {
                const std::size_t base = row * classes;
                float sum_gradient = 0.0f;
                float sum_weighted = 0.0f;
                for (std::size_t j = 0; j < classes; ++j) {
                    sum_gradient += gradient[base + j];
                    sum_weighted += gradient[base + j] * std::exp(output[base + j]);
                }
                (void)sum_weighted;
                for (std::size_t j = 0; j < classes; ++j) {
                    grad_x[base + j] =
                        gradient[base + j] -
                        std::exp(output[base + j]) * sum_gradient;
                }
            }

            accumulate_grad(x_node, grad_x);
        }));
}

Variable layer_norm(
    const Variable& x,
    const Variable& gamma,
    const Variable& beta,
    float eps) {

    require_initialized(x);
    require_initialized(gamma);
    require_initialized(beta);
    if (!(eps > 0.0f)) {
        throw std::invalid_argument("ZYRON autograd: LayerNorm eps must be > 0");
    }

    check_normalization_shapes(x.value(), gamma.value(), &beta.value());

    const std::size_t hidden = gamma.value().size();
    const std::size_t rows = rows_for_last_dim(x.value(), hidden);
    Tensor value(x.value().shape());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * hidden;
        float mean_value = 0.0f;
        for (std::size_t j = 0; j < hidden; ++j) {
            mean_value += x.value()[base + j];
        }
        mean_value /= static_cast<float>(hidden);

        float variance = 0.0f;
        for (std::size_t j = 0; j < hidden; ++j) {
            const float d = x.value()[base + j] - mean_value;
            variance += d * d;
        }
        variance /= static_cast<float>(hidden);

        const float inv_std = 1.0f / std::sqrt(variance + eps);
        for (std::size_t j = 0; j < hidden; ++j) {
            const float normalized =
                (x.value()[base + j] - mean_value) * inv_std;
            value[base + j] = normalized * gamma.value()[j] + beta.value()[j];
        }
    }

    const bool requires_grad =
        should_build_graph(x, gamma) || (!NoGradGuard::enabled() && beta.requires_grad());

    if (!requires_grad) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();
    const NodePtr gamma_node = gamma.node_handle();
    const NodePtr beta_node = beta.node_handle();
    const Tensor x_value = x.value();
    const Tensor gamma_value = gamma.value();

    return Variable(make_node(
        std::move(value), true, false,
        {x_node, gamma_node, beta_node},
        [x_node, gamma_node, beta_node, x_value, gamma_value, hidden, eps](const Tensor& gradient) {
            const std::size_t rows_local = x_value.size() / hidden;
            Tensor grad_x(x_value.shape());
            Tensor grad_gamma = Tensor::zeros(Shape{hidden});
            Tensor grad_beta = Tensor::zeros(Shape{hidden});

            for (std::size_t row = 0; row < rows_local; ++row) {
                const std::size_t base = row * hidden;
                float mean_value = 0.0f;
                for (std::size_t j = 0; j < hidden; ++j) {
                    mean_value += x_value[base + j];
                }
                mean_value /= static_cast<float>(hidden);

                float variance = 0.0f;
                for (std::size_t j = 0; j < hidden; ++j) {
                    const float d = x_value[base + j] - mean_value;
                    variance += d * d;
                }
                variance /= static_cast<float>(hidden);
                const float inv_std = 1.0f / std::sqrt(variance + eps);

                float sum_dy = 0.0f;
                float sum_dy_xhat = 0.0f;
                for (std::size_t j = 0; j < hidden; ++j) {
                    const float xhat = (x_value[base + j] - mean_value) * inv_std;
                    const float dy = gradient[base + j] * gamma_value[j];
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat;
                    grad_gamma[j] += gradient[base + j] * xhat;
                    grad_beta[j] += gradient[base + j];
                }

                const float n = static_cast<float>(hidden);
                for (std::size_t j = 0; j < hidden; ++j) {
                    const float xhat = (x_value[base + j] - mean_value) * inv_std;
                    const float dy = gradient[base + j] * gamma_value[j];
                    grad_x[base + j] =
                        inv_std / n *
                        (n * dy - sum_dy - xhat * sum_dy_xhat);
                }
            }

            accumulate_grad(x_node, grad_x);
            accumulate_grad(gamma_node, grad_gamma);
            accumulate_grad(beta_node, grad_beta);
        }));
}

Variable rms_norm(const Variable& x, const Variable& gamma, float eps) {
    require_initialized(x);
    require_initialized(gamma);
    if (!(eps > 0.0f)) {
        throw std::invalid_argument("ZYRON autograd: RMSNorm eps must be > 0");
    }

    check_normalization_shapes(x.value(), gamma.value(), nullptr);

    const std::size_t hidden = gamma.value().size();
    const std::size_t rows = rows_for_last_dim(x.value(), hidden);
    Tensor value(x.value().shape());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * hidden;
        float mean_sq = 0.0f;
        for (std::size_t j = 0; j < hidden; ++j) {
            mean_sq += x.value()[base + j] * x.value()[base + j];
        }
        mean_sq /= static_cast<float>(hidden);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (std::size_t j = 0; j < hidden; ++j) {
            value[base + j] = x.value()[base + j] * inv_rms * gamma.value()[j];
        }
    }

    if (!should_build_graph(x, gamma)) return no_grad_result(std::move(value));

    const NodePtr x_node = x.node_handle();
    const NodePtr gamma_node = gamma.node_handle();
    const Tensor x_value = x.value();
    const Tensor gamma_value = gamma.value();

    return Variable(make_node(
        std::move(value), true, false,
        {x_node, gamma_node},
        [x_node, gamma_node, x_value, gamma_value, hidden, eps](const Tensor& gradient) {
            const std::size_t rows_local = x_value.size() / hidden;
            Tensor grad_x(x_value.shape());
            Tensor grad_gamma = Tensor::zeros(Shape{hidden});

            for (std::size_t row = 0; row < rows_local; ++row) {
                const std::size_t base = row * hidden;
                float mean_sq = 0.0f;
                for (std::size_t j = 0; j < hidden; ++j) {
                    mean_sq += x_value[base + j] * x_value[base + j];
                }
                mean_sq /= static_cast<float>(hidden);
                const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

                float mean_dyx = 0.0f;
                for (std::size_t j = 0; j < hidden; ++j) {
                    const float dy = gradient[base + j] * gamma_value[j];
                    mean_dyx += dy * x_value[base + j];
                    grad_gamma[j] += gradient[base + j] * x_value[base + j] * inv_rms;
                }
                mean_dyx /= static_cast<float>(hidden);

                const float inv_rms_sq = inv_rms * inv_rms;
                for (std::size_t j = 0; j < hidden; ++j) {
                    const float dy = gradient[base + j] * gamma_value[j];
                    grad_x[base + j] =
                        inv_rms * (dy - x_value[base + j] * inv_rms_sq * mean_dyx);
                }
            }

            accumulate_grad(x_node, grad_x);
            accumulate_grad(gamma_node, grad_gamma);
        }));
}

Variable cross_entropy(
    const Variable& logits,
    const std::vector<std::size_t>& targets) {

    require_initialized(logits);
    if (logits.value().rank() < 2) {
        throw std::invalid_argument(
            "ZYRON autograd: cross_entropy expects rank >= 2 logits");
    }

    const std::size_t classes =
        logits.value().shape()[logits.value().rank() - 1];
    const std::size_t rows = logits.value().size() / classes;

    if (targets.size() != rows) {
        throw std::invalid_argument(
            "ZYRON autograd: cross_entropy target count mismatch");
    }

    for (const auto target : targets) {
        if (target >= classes) {
            throw std::out_of_range(
                "ZYRON autograd: cross_entropy target out of range");
        }
    }

    if (rows == 0) {
        throw std::invalid_argument(
            "ZYRON autograd: cross_entropy input cannot be empty");
    }

    Tensor value(Shape{});
    float loss = 0.0f;

    for (std::size_t i = 0; i < rows; ++i) {
        const std::size_t base = i * classes;
        float max_value = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < classes; ++j) {
            max_value = std::max(max_value, logits.value()[base + j]);
        }

        float sum_exp = 0.0f;
        for (std::size_t j = 0; j < classes; ++j) {
            sum_exp += std::exp(logits.value()[base + j] - max_value);
        }

        const float log_sum_exp = max_value + std::log(sum_exp);
        loss += log_sum_exp - logits.value()[base + targets[i]];
    }

    value[0] = loss / static_cast<float>(rows);

    if (!should_build_graph(logits)) return no_grad_result(std::move(value));

    const NodePtr logits_node = logits.node_handle();
    const Tensor logits_value = logits.value();

    return Variable(make_node(
        std::move(value), true, false, {logits_node},
        [logits_node, logits_value, targets, rows, classes](const Tensor& gradient) {
            const float upstream = gradient[0] / static_cast<float>(rows);
            Tensor grad_logits(logits_value.shape());

            for (std::size_t i = 0; i < rows; ++i) {
                const std::size_t base = i * classes;
                float max_value = -std::numeric_limits<float>::infinity();
                for (std::size_t j = 0; j < classes; ++j) {
                    max_value = std::max(max_value, logits_value[base + j]);
                }

                float denom = 0.0f;
                for (std::size_t j = 0; j < classes; ++j) {
                    denom += std::exp(logits_value[base + j] - max_value);
                }

                for (std::size_t j = 0; j < classes; ++j) {
                    const float probability =
                        std::exp(logits_value[base + j] - max_value) / denom;
                    grad_logits[base + j] = probability * upstream;
                }

                grad_logits[base + targets[i]] -= upstream;
            }

            accumulate_grad(logits_node, grad_logits);
        }));
}

Variable embedding(
    const Variable& weight,
    const std::vector<std::size_t>& indices) {

    require_initialized(weight);
    if (weight.value().rank() != 2) {
        throw std::invalid_argument(
            "ZYRON autograd: embedding weight must be rank 2");
    }

    const std::size_t vocab = weight.value().shape()[0];
    const std::size_t embedding_dim = weight.value().shape()[1];

    for (const auto index : indices) {
        if (index >= vocab) {
            throw std::out_of_range(
                "ZYRON autograd: embedding index out of range");
        }
    }

    Tensor value(Shape{indices.size(), embedding_dim});
    for (std::size_t i = 0; i < indices.size(); ++i) {
        for (std::size_t j = 0; j < embedding_dim; ++j) {
            value[i * embedding_dim + j] =
                weight.value()[indices[i] * embedding_dim + j];
        }
    }

    if (!should_build_graph(weight)) return no_grad_result(std::move(value));

    const NodePtr weight_node = weight.node_handle();
    const Tensor weight_value = weight.value();

    return Variable(make_node(
        std::move(value), true, false, {weight_node},
        [weight_node, weight_value, indices, embedding_dim](const Tensor& gradient) {
            Tensor grad_weight = Tensor::zeros(weight_value.shape());
            for (std::size_t i = 0; i < indices.size(); ++i) {
                const std::size_t destination = indices[i] * embedding_dim;
                const std::size_t source = i * embedding_dim;
                for (std::size_t j = 0; j < embedding_dim; ++j) {
                    grad_weight[destination + j] += gradient[source + j];
                }
            }
            accumulate_grad(weight_node, grad_weight);
        }));
}

} // namespace zyron::autograd
