#pragma once

#include "zyron/nn/layer.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace zyron::nn {

class Sequential final : public Layer {
public:
    Sequential() = default;

    void add(std::unique_ptr<Layer> layer);

    template <typename T, typename... Args>
    T& emplace(Args&&... args) {
        auto layer = std::make_unique<T>(std::forward<Args>(args)...);
        T& reference = *layer;
        add(std::move(layer));
        return reference;
    }

    autograd::Variable forward(const autograd::Variable& input) override;
    std::vector<autograd::Variable*> parameters() override;
    void train(bool mode = true) noexcept override;

    [[nodiscard]] std::size_t size() const noexcept { return layers_.size(); }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace zyron::nn
