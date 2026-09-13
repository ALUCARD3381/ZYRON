#pragma once

#include "zyron/training/optimizer.hpp"

#include <cstddef>

namespace zyron::training {

class WarmupCosineScheduler {
public:
    WarmupCosineScheduler(
        float base_learning_rate,
        std::size_t warmup_steps,
        std::size_t total_steps,
        float min_learning_rate = 0.0f);

    [[nodiscard]] float learning_rate(std::size_t step) const noexcept;
    void step(Optimizer& optimizer);

    [[nodiscard]] std::size_t step_count() const noexcept { return step_count_; }
    [[nodiscard]] float base_learning_rate() const noexcept { return base_learning_rate_; }
    [[nodiscard]] std::size_t warmup_steps() const noexcept { return warmup_steps_; }
    [[nodiscard]] std::size_t total_steps() const noexcept { return total_steps_; }
    [[nodiscard]] float min_learning_rate() const noexcept { return min_learning_rate_; }

    void restore_step(std::size_t step);

private:
    float base_learning_rate_;
    std::size_t warmup_steps_;
    std::size_t total_steps_;
    float min_learning_rate_;
    std::size_t step_count_{0};
};

} // namespace zyron::training
