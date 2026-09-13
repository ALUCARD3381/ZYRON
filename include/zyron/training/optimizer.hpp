#pragma once

#include "zyron/autograd/autograd.hpp"

#include <cstddef>
#include <vector>

namespace zyron::training {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void zero_grad() noexcept = 0;
    virtual void step() = 0;
    virtual void set_learning_rate(float lr) = 0;
    [[nodiscard]] virtual float learning_rate() const noexcept = 0;
    [[nodiscard]] virtual std::size_t step_count() const noexcept = 0;
    [[nodiscard]] virtual const std::vector<autograd::Variable*>& parameters() const noexcept = 0;
};

[[nodiscard]] float global_grad_norm(
    const std::vector<autograd::Variable*>& parameters);

float clip_grad_norm(
    const std::vector<autograd::Variable*>& parameters,
    float max_norm);

} // namespace zyron::training
