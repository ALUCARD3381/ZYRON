#pragma once

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>

namespace zyron::transformer {

struct Config {
    std::size_t vocab_size = 260;
    std::size_t hidden_size = 128;
    std::size_t num_layers = 2;
    std::size_t num_heads = 4;
    std::size_t num_kv_heads = 0; // 0 means num_heads (classic MHA).
    std::size_t intermediate_size = 512;
    std::size_t max_sequence_length = 128;
    float norm_eps = 1e-5f;
    bool use_rms_norm = true;
    bool causal = true;
    bool use_rope = true;
    bool tie_word_embeddings = true;
    float rope_theta = 10000.0f;
    float dropout = 0.0f;
    bool use_qat = false;
    std::size_t qat_bits = 8;

    [[nodiscard]] std::size_t resolved_num_kv_heads() const noexcept {
        return num_kv_heads == 0 ? num_heads : num_kv_heads;
    }

    void validate() const {
        if (vocab_size == 0 || hidden_size == 0 || num_layers == 0 ||
            num_heads == 0 || intermediate_size == 0 ||
            max_sequence_length == 0) {
            throw std::invalid_argument(
                "ZYRON Transformer: dimensions must be > 0");
        }
        if (hidden_size % num_heads != 0) {
            throw std::invalid_argument(
                "ZYRON Transformer: hidden_size must be divisible by num_heads");
        }
        if (resolved_num_kv_heads() == 0 || resolved_num_kv_heads() > num_heads ||
            num_heads % resolved_num_kv_heads() != 0) {
            throw std::invalid_argument(
                "ZYRON Transformer: num_kv_heads must divide num_heads and be <= num_heads");
        }
        if (use_rope && ((hidden_size / num_heads) % 2 != 0)) {
            throw std::invalid_argument(
                "ZYRON Transformer: RoPE requires an even attention head dimension");
        }
        if (!(norm_eps > 0.0f)) {
            throw std::invalid_argument(
                "ZYRON Transformer: norm_eps must be > 0");
        }
        if (!(rope_theta > 0.0f) || !std::isfinite(rope_theta)) {
            throw std::invalid_argument(
                "ZYRON Transformer: rope_theta must be finite and > 0");
        }
        if (!(dropout >= 0.0f) || dropout >= 1.0f || !std::isfinite(dropout)) {
            throw std::invalid_argument(
                "ZYRON Transformer: dropout must be finite and in [0, 1)");
        }
        if (qat_bits != 4 && qat_bits != 8) {
            throw std::invalid_argument(
                "ZYRON Transformer: qat_bits must be 4 or 8");
        }
    }
};

inline Config tiny(std::size_t vocab_size = 320) {
    Config config;
    config.vocab_size = vocab_size;
    config.hidden_size = 64;
    config.num_layers = 2;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.intermediate_size = 256;
    config.max_sequence_length = 128;
    return config;
}

inline Config small(std::size_t vocab_size = 320) {
    Config config;
    config.vocab_size = vocab_size;
    config.hidden_size = 256;
    config.num_layers = 6;
    config.num_heads = 8;
    config.num_kv_heads = 2;
    config.intermediate_size = 1024;
    config.max_sequence_length = 512;
    return config;
}

inline Config medium(std::size_t vocab_size = 320) {
    Config config;
    config.vocab_size = vocab_size;
    config.hidden_size = 512;
    config.num_layers = 12;
    config.num_heads = 8;
    config.num_kv_heads = 4;
    config.intermediate_size = 2048;
    config.max_sequence_length = 1024;
    return config;
}

} // namespace zyron::transformer
