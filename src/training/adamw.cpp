#include "zyron/training/adamw.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace zyron::training {

namespace {
void validate_hyperparameters(float lr, float b1, float b2, float eps, float wd) {
    if (!(lr >= 0.0f) || !std::isfinite(lr)) throw std::invalid_argument("ZYRON AdamW: invalid learning rate");
    if (!(b1 >= 0.0f && b1 < 1.0f) || !std::isfinite(b1)) throw std::invalid_argument("ZYRON AdamW: invalid beta1");
    if (!(b2 >= 0.0f && b2 < 1.0f) || !std::isfinite(b2)) throw std::invalid_argument("ZYRON AdamW: invalid beta2");
    if (!(eps > 0.0f) || !std::isfinite(eps)) throw std::invalid_argument("ZYRON AdamW: invalid epsilon");
    if (!(wd >= 0.0f) || !std::isfinite(wd)) throw std::invalid_argument("ZYRON AdamW: invalid weight decay");
}
}

AdamW::AdamW(
    std::vector<autograd::Variable*> parameters,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay)
    : parameters_(std::move(parameters)),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay) {

    if (parameters_.empty()) throw std::invalid_argument("ZYRON AdamW: no parameters");
    validate_hyperparameters(learning_rate_, beta1_, beta2_, epsilon_, weight_decay_);

    m_.reserve(parameters_.size());
    v_.reserve(parameters_.size());
    for (auto* parameter : parameters_) {
        if (parameter == nullptr || !parameter->requires_grad()) {
            throw std::invalid_argument("ZYRON AdamW: parameter must require gradients");
        }
        m_.push_back(Tensor::zeros(parameter->value().shape()));
        v_.push_back(Tensor::zeros(parameter->value().shape()));
    }
}

void AdamW::zero_grad() noexcept {
    for (auto* parameter : parameters_) {
        if (parameter) parameter->zero_grad();
    }
}

void AdamW::set_learning_rate(float lr) {
    if (!(lr >= 0.0f) || !std::isfinite(lr)) {
        throw std::invalid_argument("ZYRON AdamW: invalid learning rate");
    }
    learning_rate_ = lr;
}

void AdamW::step() {
    ++step_count_;
    const float bias1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    const float bias2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

    if (!(bias1 > 0.0f) || !(bias2 > 0.0f)) {
        throw std::overflow_error("ZYRON AdamW: bias correction overflow");
    }

    for (std::size_t p = 0; p < parameters_.size(); ++p) {
        auto* parameter = parameters_[p];
        if (!parameter || !parameter->has_grad()) continue;

        const Tensor grad = parameter->grad().contiguous();
        Tensor param = parameter->value().contiguous();
        Tensor m = m_[p].contiguous();
        Tensor v = v_[p].contiguous();

        float* param_data = param.data();
        float* m_data = m.data();
        float* v_data = v.data();
        const float* grad_data = grad.data();

        for (std::size_t i = 0; i < param.size(); ++i) {
            const float g = grad_data[i];
            m_data[i] = beta1_ * m_data[i] + (1.0f - beta1_) * g;
            v_data[i] = beta2_ * v_data[i] + (1.0f - beta2_) * g * g;

            const float m_hat = m_data[i] / bias1;
            const float v_hat = v_data[i] / bias2;
            if (weight_decay_ != 0.0f) {
                param_data[i] -= learning_rate_ * weight_decay_ * param_data[i];
            }
            param_data[i] -= learning_rate_ * (m_hat / (std::sqrt(v_hat) + epsilon_));
        }

        parameter->value() = std::move(param);
        m_[p] = std::move(m);
        v_[p] = std::move(v);
    }
}

void AdamW::restore_state(
    std::size_t step,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    const std::vector<Tensor>& first_moments,
    const std::vector<Tensor>& second_moments) {

    if (first_moments.size() != parameters_.size() ||
        second_moments.size() != parameters_.size()) {
        throw std::invalid_argument("ZYRON AdamW: checkpoint state parameter count mismatch");
    }

    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        if (first_moments[i].shape() != parameters_[i]->value().shape() ||
            second_moments[i].shape() != parameters_[i]->value().shape()) {
            throw std::invalid_argument("ZYRON AdamW: checkpoint moment shape mismatch");
        }
    }

    validate_hyperparameters(learning_rate, beta1, beta2, epsilon, weight_decay);
    set_learning_rate(learning_rate);
    beta1_ = beta1;
    beta2_ = beta2;
    epsilon_ = epsilon;
    weight_decay_ = weight_decay;
    step_count_ = step;
    m_ = first_moments;
    v_ = second_moments;
}

float global_grad_norm(const std::vector<autograd::Variable*>& parameters) {
    long double sum_sq = 0.0L;
    for (const auto* parameter : parameters) {
        if (!parameter || !parameter->has_grad()) continue;
        const Tensor grad = parameter->grad().contiguous();
        const float* data = grad.data();
        for (std::size_t i = 0; i < grad.size(); ++i) {
            const long double value = static_cast<long double>(data[i]);
            sum_sq += value * value;
        }
    }
    return static_cast<float>(std::sqrt(sum_sq));
}

float clip_grad_norm(
    const std::vector<autograd::Variable*>& parameters,
    float max_norm) {

    if (!(max_norm > 0.0f) || !std::isfinite(max_norm)) {
        throw std::invalid_argument("ZYRON grad clip: max_norm must be finite and > 0");
    }

    const float norm = global_grad_norm(parameters);
    if (norm > max_norm && norm > 0.0f) {
        const float scale = max_norm / norm;
        for (auto* parameter : parameters) {
            if (!parameter || !parameter->has_grad()) continue;
            Tensor grad = parameter->grad().contiguous();
            float* data = grad.data();
            for (std::size_t i = 0; i < grad.size(); ++i) data[i] *= scale;
            parameter->grad() = std::move(grad);
        }
    }
    return norm;
}

} // namespace zyron::training
