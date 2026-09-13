#include "zyron/training/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace zyron::training {

WarmupCosineScheduler::WarmupCosineScheduler(
    float base_learning_rate,
    std::size_t warmup_steps,
    std::size_t total_steps,
    float min_learning_rate)
    : base_learning_rate_(base_learning_rate),
      warmup_steps_(warmup_steps),
      total_steps_(total_steps),
      min_learning_rate_(min_learning_rate) {

    if (!(base_learning_rate_ >= 0.0f) || !std::isfinite(base_learning_rate_)) {
        throw std::invalid_argument("ZYRON Scheduler: invalid base learning rate");
    }
    if (total_steps_ == 0) throw std::invalid_argument("ZYRON Scheduler: total_steps must be > 0");
    if (warmup_steps_ > total_steps_) throw std::invalid_argument("ZYRON Scheduler: warmup_steps > total_steps");
    if (!(min_learning_rate_ >= 0.0f) || !std::isfinite(min_learning_rate_) || min_learning_rate_ > base_learning_rate_) {
        throw std::invalid_argument("ZYRON Scheduler: invalid minimum learning rate");
    }
}

float WarmupCosineScheduler::learning_rate(std::size_t step) const noexcept {
    if (step == 0) return 0.0f;
    if (warmup_steps_ != 0 && step <= warmup_steps_) {
        return base_learning_rate_ *
            (static_cast<float>(step) / static_cast<float>(warmup_steps_));
    }
    if (step >= total_steps_) return min_learning_rate_;

    const float denominator = static_cast<float>(total_steps_ - warmup_steps_);
    const float progress = denominator > 0.0f
        ? static_cast<float>(step - warmup_steps_) / denominator
        : 1.0f;
    const float cosine = 0.5f * (1.0f + std::cos(3.14159265358979323846f * progress));
    return min_learning_rate_ +
        (base_learning_rate_ - min_learning_rate_) * cosine;
}

void WarmupCosineScheduler::step(Optimizer& optimizer) {
    ++step_count_;
    optimizer.set_learning_rate(learning_rate(step_count_));
}

void WarmupCosineScheduler::restore_step(std::size_t step) {
    step_count_ = std::min(step, total_steps_);
}

} // namespace zyron::training
