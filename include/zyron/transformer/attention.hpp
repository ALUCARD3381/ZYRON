#pragma once

#include "zyron/nn/layer.hpp"
#include "zyron/nn/linear.hpp"
#include "zyron/nn/dropout.hpp"
#include "zyron/transformer/config.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace zyron::transformer {

class MultiHeadAttention final : public nn::Layer {
public:
    struct KVCache {
        Tensor key;
        Tensor value;
        std::size_t sequence_length{0};
        std::size_t batch_size{0};
        bool initialized{false};

        void reset() noexcept {
            key = Tensor{};
            value = Tensor{};
            sequence_length = 0;
            batch_size = 0;
            initialized = false;
        }
    };

    explicit MultiHeadAttention(
        const Config& config,
        std::uint32_t seed = 0);

    autograd::Variable forward(const autograd::Variable& input) override;

    // Inference-only incremental attention. The caller must keep NoGradGuard
    // active. Keys and values are stored in fixed-capacity buffers up to the
    // model context length, so appending one token does not copy the prefix.
    autograd::Variable forward_cached(
        const autograd::Variable& input,
        KVCache& cache);

    std::vector<autograd::Variable*> parameters() override;
    void train(bool mode = true) noexcept override;

    [[nodiscard]] std::size_t hidden_size() const noexcept { return hidden_size_; }
    [[nodiscard]] std::size_t num_heads() const noexcept { return num_heads_; }
    [[nodiscard]] std::size_t num_kv_heads() const noexcept { return num_kv_heads_; }
    [[nodiscard]] std::size_t head_dim() const noexcept { return head_dim_; }
    [[nodiscard]] bool causal() const noexcept { return causal_; }

    void set_qat(bool enabled, std::size_t bits = 8) noexcept;

    [[nodiscard]] nn::Linear& q_proj() noexcept { return q_proj_; }
    [[nodiscard]] const nn::Linear& q_proj() const noexcept { return q_proj_; }
    [[nodiscard]] nn::Linear& k_proj() noexcept { return k_proj_; }
    [[nodiscard]] const nn::Linear& k_proj() const noexcept { return k_proj_; }
    [[nodiscard]] nn::Linear& v_proj() noexcept { return v_proj_; }
    [[nodiscard]] const nn::Linear& v_proj() const noexcept { return v_proj_; }
    [[nodiscard]] nn::Linear& out_proj() noexcept { return out_proj_; }
    [[nodiscard]] const nn::Linear& out_proj() const noexcept { return out_proj_; }

private:
    static Tensor make_causal_mask(std::size_t sequence_length);

    std::size_t hidden_size_;
    std::size_t num_heads_;
    std::size_t num_kv_heads_;
    std::size_t head_dim_;
    std::size_t max_sequence_length_;
    bool causal_;
    nn::Linear q_proj_;
    nn::Linear k_proj_;
    nn::Linear v_proj_;
    nn::Linear out_proj_;
    nn::Dropout attention_dropout_;
    Tensor rope_cos_;
    Tensor rope_sin_;
    bool use_rope_{false};
};

} // namespace zyron::transformer
