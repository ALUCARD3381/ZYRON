#pragma once

#include "zyron/nn/layer.hpp"
#include "zyron/transformer/config.hpp"
#include "zyron/transformer/transformer_block.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace zyron::transformer {

class Transformer final : public nn::Layer {
public:
    struct KVCache {
        std::vector<MultiHeadAttention::KVCache> layers;
        std::size_t sequence_length{0};
        std::size_t batch_size{0};
        bool initialized{false};

        void reset() noexcept {
            for (auto& layer : layers) layer.reset();
            sequence_length = 0;
            batch_size = 0;
            initialized = false;
        }
    };

    explicit Transformer(
        const Config& config,
        std::uint32_t seed = 0);

    autograd::Variable forward(const autograd::Variable& input) override;
    // Inference-only incremental forward. Each call appends new positions to
    // the per-layer KV cache and returns logits/hidden states for only that
    // input chunk.
    autograd::Variable forward_cached(
        const autograd::Variable& input,
        KVCache& cache);
    std::vector<autograd::Variable*> parameters() override;
    void train(bool mode = true) noexcept override;
    void set_qat(bool enabled, std::size_t bits = 8) noexcept;

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t num_layers() const noexcept { return blocks_.size(); }
    [[nodiscard]] TransformerBlock& block(std::size_t index);
    [[nodiscard]] const TransformerBlock& block(std::size_t index) const;

private:
    Tensor positional_encoding(std::size_t start_position, std::size_t sequence_length) const;

    Config config_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    Tensor positional_encoding_;
};

} // namespace zyron::transformer
