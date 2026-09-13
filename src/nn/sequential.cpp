#include "zyron/nn/sequential.hpp"

#include <stdexcept>

namespace zyron::nn {

void Sequential::add(std::unique_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("ZYRON Sequential: layer cannot be null");
    }
    layer->train(is_training());
    layers_.push_back(std::move(layer));
}

autograd::Variable Sequential::forward(const autograd::Variable& input) {
    autograd::Variable output = input;
    for (auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}

std::vector<autograd::Variable*> Sequential::parameters() {
    std::vector<autograd::Variable*> result;
    for (auto& layer : layers_) {
        const auto child_parameters = layer->parameters();
        result.insert(result.end(), child_parameters.begin(), child_parameters.end());
    }
    return result;
}

void Sequential::train(bool mode) noexcept {
    training_ = mode;
    for (auto& layer : layers_) {
        layer->train(mode);
    }
}

} // namespace zyron::nn
