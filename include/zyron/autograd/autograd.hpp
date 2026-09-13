#pragma once

#include "zyron/core/tensor.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace zyron::autograd {

class Variable;

class NoGradGuard {
public:
    NoGradGuard() noexcept;
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
    NoGradGuard(NoGradGuard&&) = delete;
    NoGradGuard& operator=(NoGradGuard&&) = delete;

    [[nodiscard]] static bool enabled() noexcept;

private:
    bool previous_{false};
};

Variable add(const Variable& a, const Variable& b);
Variable sub(const Variable& a, const Variable& b);
Variable mul(const Variable& a, const Variable& b);
Variable div(const Variable& a, const Variable& b);
Variable neg(const Variable& x);
Variable sum(const Variable& x);
Variable mean(const Variable& x);
Variable exp(const Variable& x);
Variable log(const Variable& x);
Variable sqrt(const Variable& x);
Variable pow(const Variable& x, float exponent);
Variable matmul(const Variable& a, const Variable& b);
Variable reshape(const Variable& x, const Shape& new_shape);
Variable transpose(const Variable& x, std::size_t dim0, std::size_t dim1);
Variable contiguous(const Variable& x);
Variable rotary_embedding(
    const Variable& x,
    const Tensor& cos_table,
    const Tensor& sin_table,
    std::size_t start_position);
Variable repeat_kv_heads(const Variable& x, std::size_t repeats);
Variable fake_quantize(const Variable& x, std::size_t bits);

Variable relu(const Variable& x);
Variable gelu(const Variable& x);
Variable silu(const Variable& x);
Variable sigmoid(const Variable& x);
Variable tanh(const Variable& x);
Variable softmax(const Variable& x);
Variable log_softmax(const Variable& x);

Variable layer_norm(
    const Variable& x,
    const Variable& gamma,
    const Variable& beta,
    float eps = 1e-5f);

Variable rms_norm(
    const Variable& x,
    const Variable& gamma,
    float eps = 1e-5f);

Variable cross_entropy(
    const Variable& logits,
    const std::vector<std::size_t>& targets);

Variable embedding(
    const Variable& weight,
    const std::vector<std::size_t>& indices);

class Variable {
public:
    struct Node;

    Variable() = default;
    explicit Variable(Tensor value, bool requires_grad = false);
    ~Variable();

    Variable(const Variable&) = default;
    Variable& operator=(const Variable&) = default;
    Variable(Variable&&) noexcept = default;
    Variable& operator=(Variable&&) noexcept = default;

    [[nodiscard]] const Tensor& value() const;
    [[nodiscard]] Tensor& value();
    [[nodiscard]] const std::shared_ptr<Node>& node_handle() const noexcept {
        return node_;
    }

    [[nodiscard]] bool requires_grad() const noexcept;
    [[nodiscard]] bool has_grad() const noexcept;
    [[nodiscard]] const Tensor& grad() const;
    [[nodiscard]] Tensor& grad();

    void zero_grad() noexcept;
    void backward();
    void backward(const Tensor& gradient);

private:
    explicit Variable(std::shared_ptr<Node> node);
    std::shared_ptr<Node> node_{};

    friend Variable add(const Variable&, const Variable&);
    friend Variable sub(const Variable&, const Variable&);
    friend Variable mul(const Variable&, const Variable&);
    friend Variable div(const Variable&, const Variable&);
    friend Variable neg(const Variable&);
    friend Variable sum(const Variable&);
    friend Variable mean(const Variable&);
    friend Variable exp(const Variable&);
    friend Variable log(const Variable&);
    friend Variable sqrt(const Variable&);
    friend Variable pow(const Variable&, float);
    friend Variable matmul(const Variable&, const Variable&);
    friend Variable reshape(const Variable&, const Shape&);
    friend Variable transpose(const Variable&, std::size_t, std::size_t);
    friend Variable contiguous(const Variable&);
    friend Variable rotary_embedding(const Variable&, const Tensor&, const Tensor&, std::size_t);
    friend Variable repeat_kv_heads(const Variable&, std::size_t);
    friend Variable fake_quantize(const Variable&, std::size_t);
    friend Variable relu(const Variable&);
    friend Variable gelu(const Variable&);
    friend Variable silu(const Variable&);
    friend Variable sigmoid(const Variable&);
    friend Variable tanh(const Variable&);
    friend Variable softmax(const Variable&);
    friend Variable log_softmax(const Variable&);
    friend Variable layer_norm(const Variable&, const Variable&, const Variable&, float);
    friend Variable rms_norm(const Variable&, const Variable&, float);
    friend Variable cross_entropy(const Variable&, const std::vector<std::size_t>&);
    friend Variable embedding(const Variable&, const std::vector<std::size_t>&);
};

} // namespace zyron::autograd
