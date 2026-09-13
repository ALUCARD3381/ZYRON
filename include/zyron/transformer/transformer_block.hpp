#pragma once

#include "zyron/nn/layer.hpp"
#include "zyron/nn/normalization.hpp"
#include "zyron/transformer/attention.hpp"
#include "zyron/transformer/config.hpp"
#include "zyron/transformer/feed_forward.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace zyron::transformer {

class TransformerBlock final : public nn::Layer {
public:
    explicit TransformerBlock(
        const Config& config,
        std::uint32_t seed = 0);

    autograd::Variable forward(const autograd::Variable& input) override;
    autograd::Variable forward_cached(
        const autograd::Variable& input,
        MultiHeadAttention::KVCache& cache);
    std::vector<autograd::Variable*> parameters() override;
    void train(bool mode = true) noexcept override;
    void set_qat(bool enabled, std::size_t bits = 8) noexcept;

    [[nodiscard]] MultiHeadAttention& attention() noexcept { return attention_; }
    [[nodiscard]] const MultiHeadAttention& attention() const noexcept { return attention_; }
    [[nodiscard]] FeedForward& feed_forward() noexcept { return feed_forward_; }
    [[nodiscard]] const FeedForward& feed_forward() const noexcept { return feed_forward_; }
    [[nodiscard]] nn::Layer& norm1() noexcept;
    [[nodiscard]] const nn::Layer& norm1() const noexcept;
    [[nodiscard]] nn::Layer& norm2() noexcept;
    [[nodiscard]] const nn::Layer& norm2() const noexcept;

private:
    std::unique_ptr<nn::Layer> norm1_;
    MultiHeadAttention attention_;
    std::unique_ptr<nn::Layer> norm2_;
    FeedForward feed_forward_;
};

} // namespace zyron::transformer
