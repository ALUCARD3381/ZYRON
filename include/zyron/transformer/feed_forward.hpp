#pragma once

#include "zyron/nn/layer.hpp"
#include "zyron/nn/linear.hpp"
#include "zyron/nn/dropout.hpp"
#include "zyron/transformer/config.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zyron::transformer {

class FeedForward final : public nn::Layer {
public:
    explicit FeedForward(
        const Config& config,
        std::uint32_t seed = 0);

    autograd::Variable forward(const autograd::Variable& input) override;
    std::vector<autograd::Variable*> parameters() override;
    void train(bool mode = true) noexcept override;
    void set_qat(bool enabled, std::size_t bits = 8) noexcept;

    [[nodiscard]] nn::Linear& up() noexcept { return up_; }
    [[nodiscard]] const nn::Linear& up() const noexcept { return up_; }
    [[nodiscard]] nn::Linear& down() noexcept { return down_; }
    [[nodiscard]] const nn::Linear& down() const noexcept { return down_; }

private:
    nn::Linear up_;
    nn::Linear down_;
    nn::Dropout dropout_;
};

} // namespace zyron::transformer
