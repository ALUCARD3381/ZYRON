#pragma once

#include "zyron/nn/layer.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zyron::nn {

class Embedding final : public Layer {
public:
    Embedding(
        std::size_t num_embeddings,
        std::size_t embedding_dim,
        std::uint32_t seed = 0,
        float init_range = 0.02f);

    autograd::Variable forward(const autograd::Variable& input) override;
    autograd::Variable forward(const std::vector<std::size_t>& indices);
    std::vector<autograd::Variable*> parameters() override;

    [[nodiscard]] std::size_t num_embeddings() const noexcept { return num_embeddings_; }
    [[nodiscard]] std::size_t embedding_dim() const noexcept { return embedding_dim_; }
    [[nodiscard]] autograd::Variable& weight() noexcept { return weight_; }
    [[nodiscard]] const autograd::Variable& weight() const noexcept { return weight_; }
    void set_qat(bool enabled, std::size_t bits = 8) noexcept { qat_enabled_ = enabled; qat_bits_ = bits; }
    [[nodiscard]] bool qat_enabled() const noexcept { return qat_enabled_; }
    [[nodiscard]] std::size_t qat_bits() const noexcept { return qat_bits_; }

private:
    std::size_t num_embeddings_;
    std::size_t embedding_dim_;
    autograd::Variable weight_;
    bool qat_enabled_{false};
    std::size_t qat_bits_{8};
};

} // namespace zyron::nn
