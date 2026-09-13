#pragma once

#include "zyron/autograd/autograd.hpp"

#include <vector>

namespace zyron::nn {

class Layer {
public:
    virtual ~Layer() = default;

    virtual autograd::Variable forward(const autograd::Variable& input) = 0;
    virtual std::vector<autograd::Variable*> parameters() = 0;

    virtual void train(bool mode = true) noexcept { training_ = mode; }
    void eval() noexcept { train(false); }

    [[nodiscard]] bool is_training() const noexcept { return training_; }

protected:
    bool training_{true};
};

} // namespace zyron::nn
