#include "zyron/nn/nn.hpp"
#include "zyron/nn/sequential.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using zyron::Shape;
    using zyron::Tensor;
    using zyron::autograd::Variable;
    using zyron::autograd::cross_entropy;
    using zyron::nn::Linear;
    using zyron::nn::Sequential;
    using zyron::nn::gelu;

    Sequential model;
    model.emplace<Linear>(4, 8, true, 1);
    model.emplace<Linear>(8, 3, true, 2);

    Tensor input_value(Shape{2, 4});
    const float input_data[] = {
        1.0f, 0.5f, -0.5f, 2.0f,
        -1.0f, 1.5f, 0.2f, -0.3f
    };

    for (std::size_t i = 0; i < input_value.size(); ++i) {
        input_value[i] = input_data[i];
    }

    Variable input(input_value, true);
    auto hidden = gelu(model.forward(input));
    const std::vector<std::size_t> targets{0, 2};
    auto loss = cross_entropy(hidden, targets);

    loss.backward();

    std::cout << "ZYRON Phase 2 MLP example\n";
    std::cout << "Loss: " << loss.value()[0] << '\n';
    std::cout << "Parameters: " << model.parameters().size() << '\n';

    return 0;
}
