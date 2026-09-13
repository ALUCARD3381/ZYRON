#pragma once

#include "zyron/nn/embedding.hpp"
#include "zyron/nn/linear.hpp"
#include "zyron/nn/normalization.hpp"
#include "zyron/transformer/transformer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zyron::model {

class LanguageModel {
public:
    using KVCache = transformer::Transformer::KVCache;

    explicit LanguageModel(
        const transformer::Config& config,
        std::uint32_t seed = 0);

    [[nodiscard]] autograd::Variable forward(
        const std::vector<std::size_t>& token_ids);

    [[nodiscard]] autograd::Variable forward(
        const std::vector<std::vector<std::size_t>>& batch_token_ids);

    // Inference-only incremental forward. On the first call, token_ids is the
    // full prompt; subsequent calls should contain newly appended token(s).
    [[nodiscard]] autograd::Variable forward_cached(
        const std::vector<std::size_t>& token_ids,
        KVCache& cache);

    [[nodiscard]] autograd::Variable forward_cached_batch(
        const std::vector<std::vector<std::size_t>>& batch_token_ids,
        KVCache& cache);

    [[nodiscard]] autograd::Variable loss(
        const std::vector<std::vector<std::size_t>>& input_ids,
        const std::vector<std::vector<std::size_t>>& target_ids);

    [[nodiscard]] std::vector<autograd::Variable*> parameters();
    void zero_grad() noexcept;

    void train(bool mode = true) noexcept;
    void eval() noexcept { train(false); }
    [[nodiscard]] bool is_training() const noexcept { return training_; }

    [[nodiscard]] const transformer::Config& config() const noexcept {
        return config_;
    }
    [[nodiscard]] nn::Embedding& token_embedding() noexcept { return token_embedding_; }
    [[nodiscard]] const nn::Embedding& token_embedding() const noexcept { return token_embedding_; }
    [[nodiscard]] transformer::Transformer& transformer() noexcept { return transformer_; }
    [[nodiscard]] const transformer::Transformer& transformer() const noexcept { return transformer_; }
    [[nodiscard]] nn::RMSNorm& final_norm() noexcept { return final_norm_; }
    [[nodiscard]] const nn::RMSNorm& final_norm() const noexcept { return final_norm_; }
    [[nodiscard]] nn::Linear& lm_head() noexcept { return lm_head_; }
    [[nodiscard]] const nn::Linear& lm_head() const noexcept { return lm_head_; }

private:
    transformer::Config config_;
    nn::Embedding token_embedding_;
    transformer::Transformer transformer_;
    nn::RMSNorm final_norm_;
    nn::Linear lm_head_;
    bool training_{true};
};

} // namespace zyron::model
