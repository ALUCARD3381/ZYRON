#pragma once

#include "zyron/training/optimizer.hpp"

#include <cstddef>
#include <vector>

namespace zyron::training {

class AdamW final : public Optimizer {
public:
    AdamW(
        std::vector<autograd::Variable*> parameters,
        float learning_rate = 3e-4f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weight_decay = 0.01f);

    void zero_grad() noexcept override;
    void step() override;
    void set_learning_rate(float lr) override;

    [[nodiscard]] float learning_rate() const noexcept override { return learning_rate_; }
    [[nodiscard]] std::size_t step_count() const noexcept override { return step_count_; }

    [[nodiscard]] float beta1() const noexcept { return beta1_; }
    [[nodiscard]] float beta2() const noexcept { return beta2_; }
    [[nodiscard]] float epsilon() const noexcept { return epsilon_; }
    [[nodiscard]] float weight_decay() const noexcept { return weight_decay_; }
    [[nodiscard]] const std::vector<autograd::Variable*>& parameters() const noexcept override { return parameters_; }
    [[nodiscard]] const std::vector<zyron::Tensor>& first_moments() const noexcept { return m_; }
    [[nodiscard]] const std::vector<zyron::Tensor>& second_moments() const noexcept { return v_; }

    void restore_state(
        std::size_t step,
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon,
        float weight_decay,
        const std::vector<zyron::Tensor>& first_moments,
        const std::vector<zyron::Tensor>& second_moments);

private:
    std::vector<autograd::Variable*> parameters_;
    std::vector<zyron::Tensor> m_;
    std::vector<zyron::Tensor> v_;
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    std::size_t step_count_{0};
};

} // namespace zyron::training
