#pragma once

#include "zyron/autograd/autograd.hpp"

#include <cstddef>
#include <vector>

namespace zyron::nn {

inline autograd::Variable cross_entropy(
    const autograd::Variable& logits,
    const std::vector<std::size_t>& targets) {
    return autograd::cross_entropy(logits, targets);
}

} // namespace zyron::nn
