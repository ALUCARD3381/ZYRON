#include "zyron/transformer/attention.hpp"

#include "zyron/autograd/autograd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include "zyron/core/runtime.hpp"

namespace zyron::transformer {

namespace {

autograd::Variable scalar_variable(float value) {
    Tensor t(Shape{});
    t[0] = value;
    return autograd::Variable(std::move(t), false);
}

Tensor fused_attention_inference(
    const Tensor& q,
    const Tensor& key_cache,
    const Tensor& value_cache,
    std::size_t previous,
    std::size_t total,
    std::size_t num_heads,
    std::size_t num_kv_heads,
    std::size_t head_dim,
    bool causal) {

    const std::size_t batch = q.shape()[0];
    const std::size_t sequence = q.shape()[2];
    Tensor context(Shape{batch, num_heads, sequence, head_dim});
    const std::size_t repeats = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    runtime::parallel_for(0, batch * num_heads * sequence,
        [&](std::size_t begin, std::size_t end) {
            std::vector<float> scores;
            scores.reserve(total);
            for (std::size_t work = begin; work < end; ++work) {
                const std::size_t query = work % sequence;
                const std::size_t head = (work / sequence) % num_heads;
                const std::size_t b = work / (sequence * num_heads);
                const std::size_t kv_head = head / repeats;
                const std::size_t absolute_position = previous + query;

                float max_score = -std::numeric_limits<float>::infinity();
                for (std::size_t key_index = 0; key_index < total; ++key_index) {
                    if (causal && key_index > absolute_position) break;
                    const float* qv = q.data() + ((b * num_heads + head) * sequence + query) * head_dim;
                    const float* kv = key_cache.data() + ((b * num_kv_heads + kv_head) * key_cache.shape()[2] + key_index) * head_dim;
                    float dot = 0.0f;
                    for (std::size_t d = 0; d < head_dim; ++d) dot += qv[d] * kv[d];
                    max_score = std::max(max_score, dot * scale);
                }

                float denom = 0.0f;
                for (std::size_t key_index = 0; key_index < total; ++key_index) {
                    if (causal && key_index > absolute_position) break;
                    const float* qv = q.data() + ((b * num_heads + head) * sequence + query) * head_dim;
                    const float* kv = key_cache.data() + ((b * num_kv_heads + kv_head) * key_cache.shape()[2] + key_index) * head_dim;
                    float dot = 0.0f;
                    for (std::size_t d = 0; d < head_dim; ++d) dot += qv[d] * kv[d];
                    const float weight = std::exp(dot * scale - max_score);
                    denom += weight;
                }

                float* out = context.data() + ((b * num_heads + head) * sequence + query) * head_dim;
                std::fill_n(out, head_dim, 0.0f);
                if (denom <= 0.0f || !std::isfinite(denom)) continue;
                const float inv_denom = 1.0f / denom;
                for (std::size_t key_index = 0; key_index < total; ++key_index) {
                    if (causal && key_index > absolute_position) break;
                    const float* qv = q.data() + ((b * num_heads + head) * sequence + query) * head_dim;
                    const float* kv = key_cache.data() + ((b * num_kv_heads + kv_head) * key_cache.shape()[2] + key_index) * head_dim;
                    float dot = 0.0f;
                    for (std::size_t d = 0; d < head_dim; ++d) dot += qv[d] * kv[d];
                    const float weight = std::exp(dot * scale - max_score) * inv_denom;
                    const float* vv = value_cache.data() + ((b * num_kv_heads + kv_head) * value_cache.shape()[2] + key_index) * head_dim;
                    for (std::size_t d = 0; d < head_dim; ++d) out[d] += weight * vv[d];
                }
            }
        }, 1);
    return context;
}

} // namespace


MultiHeadAttention::MultiHeadAttention(
    const Config& config,
    std::uint32_t seed)
    : hidden_size_(config.hidden_size),
      num_heads_(config.num_heads),
      num_kv_heads_(config.resolved_num_kv_heads()),
      head_dim_(config.num_heads == 0 ? 0 : config.hidden_size / config.num_heads),
      max_sequence_length_(config.max_sequence_length),
      causal_(config.causal),
      q_proj_(config.hidden_size, config.hidden_size, true, seed + 1),
      k_proj_(config.hidden_size, config.resolved_num_kv_heads() * (config.num_heads == 0 ? 0 : config.hidden_size / config.num_heads), true, seed + 2),
      v_proj_(config.hidden_size, config.resolved_num_kv_heads() * (config.num_heads == 0 ? 0 : config.hidden_size / config.num_heads), true, seed + 3),
      out_proj_(config.hidden_size, config.hidden_size, true, seed + 4),
      attention_dropout_(config.dropout, seed + 5),
      rope_cos_(Shape{config.max_sequence_length, (config.num_heads == 0 ? 0 : config.hidden_size / config.num_heads) / 2}),
      rope_sin_(Shape{config.max_sequence_length, (config.num_heads == 0 ? 0 : config.hidden_size / config.num_heads) / 2}),
      use_rope_(config.use_rope) {
    config.validate();
    const std::size_t half = head_dim_ / 2;
    for (std::size_t pos = 0; pos < max_sequence_length_; ++pos) {
        for (std::size_t pair = 0; pair < half; ++pair) {
            const float exponent = static_cast<float>(2 * pair) / static_cast<float>(head_dim_);
            const float angle = static_cast<float>(pos) / std::pow(config.rope_theta, exponent);
            rope_cos_.at({pos, pair}) = std::cos(angle);
            rope_sin_.at({pos, pair}) = std::sin(angle);
        }
    }
}

Tensor MultiHeadAttention::make_causal_mask(
    std::size_t sequence_length) {

    Tensor mask(Shape{sequence_length, sequence_length});
    for (std::size_t i = 0; i < sequence_length; ++i) {
        for (std::size_t j = 0; j < sequence_length; ++j) {
            mask.at({i, j}) =
                j <= i ? 0.0f
                       : -std::numeric_limits<float>::infinity();
        }
    }
    return mask;
}

autograd::Variable MultiHeadAttention::forward(
    const autograd::Variable& input) {

    if (input.value().rank() != 3) {
        throw std::invalid_argument(
            "ZYRON MultiHeadAttention: expected [batch, sequence, hidden]");
    }

    const std::size_t batch = input.value().shape()[0];
    const std::size_t sequence = input.value().shape()[1];
    const std::size_t hidden = input.value().shape()[2];

    if (hidden != hidden_size_) {
        throw std::invalid_argument(
            "ZYRON MultiHeadAttention: hidden size mismatch");
    }
    if (sequence == 0) {
        throw std::invalid_argument(
            "ZYRON MultiHeadAttention: sequence length must be > 0");
    }

    auto q = q_proj_.forward(input);
    auto k = k_proj_.forward(input);
    auto v = v_proj_.forward(input);

    q = autograd::reshape(
        q,
        Shape{batch, sequence, num_heads_, head_dim_});
    k = autograd::reshape(
        k,
        Shape{batch, sequence, num_kv_heads_, head_dim_});
    v = autograd::reshape(
        v,
        Shape{batch, sequence, num_kv_heads_, head_dim_});

    q = autograd::contiguous(autograd::transpose(q, 1, 2));
    k = autograd::contiguous(autograd::transpose(k, 1, 2));
    v = autograd::contiguous(autograd::transpose(v, 1, 2));

    if (use_rope_) {
        q = autograd::rotary_embedding(q, rope_cos_, rope_sin_, 0);
        k = autograd::rotary_embedding(k, rope_cos_, rope_sin_, 0);
    }

    if (num_kv_heads_ != num_heads_) {
        const std::size_t repeats = num_heads_ / num_kv_heads_;
        k = autograd::repeat_kv_heads(k, repeats);
        v = autograd::repeat_kv_heads(v, repeats);
    }

    auto k_transposed = autograd::transpose(k, 2, 3);
    auto scores = autograd::matmul(q, k_transposed);
    scores = autograd::mul(
        scores,
        scalar_variable(1.0f / std::sqrt(static_cast<float>(head_dim_))));

    if (causal_) {
        const Tensor mask = make_causal_mask(sequence);
        scores = autograd::add(
            scores,
            autograd::Variable(mask, false));
    }

    auto probabilities = autograd::softmax(scores);
    probabilities = attention_dropout_.forward(probabilities);
    auto context = autograd::matmul(probabilities, v);
    context = autograd::transpose(context, 1, 2);
    context = autograd::contiguous(context);
    context = autograd::reshape(
        context,
        Shape{batch, sequence, hidden_size_});

    return out_proj_.forward(context);
}

autograd::Variable MultiHeadAttention::forward_cached(
    const autograd::Variable& input,
    KVCache& cache) {

    if (!autograd::NoGradGuard::enabled()) {
        throw std::invalid_argument(
            "ZYRON MultiHeadAttention: forward_cached requires NoGradGuard");
    }
    if (input.value().rank() != 3) {
        throw std::invalid_argument(
            "ZYRON MultiHeadAttention: cached forward expects [batch, sequence, hidden]");
    }

    const std::size_t batch = input.value().shape()[0];
    const std::size_t sequence = input.value().shape()[1];
    const std::size_t hidden = input.value().shape()[2];

    if (hidden != hidden_size_) {
        throw std::invalid_argument(
            "ZYRON MultiHeadAttention: hidden size mismatch");
    }
    if (sequence == 0) {
        throw std::invalid_argument(
            "ZYRON MultiHeadAttention: sequence length must be > 0");
    }
    const std::size_t previous = cache.sequence_length;
    if (previous > std::numeric_limits<std::size_t>::max() - sequence) {
        throw std::overflow_error(
            "ZYRON MultiHeadAttention: cached sequence length overflow");
    }
    const std::size_t total = previous + sequence;
    if (total > max_sequence_length_) {
        throw std::out_of_range(
            "ZYRON MultiHeadAttention: cached sequence exceeds configured context length");
    }

    if (!cache.initialized) {
        // Start from a small capacity and grow geometrically. This keeps cache
        // memory proportional to the actually used context while making append
        // amortized O(1) instead of copying the whole prefix every token.
        const std::size_t initial_capacity = std::min<std::size_t>(
            max_sequence_length_,
            std::max<std::size_t>(8, sequence));
        cache.key = Tensor(Shape{batch, num_kv_heads_, initial_capacity, head_dim_});
        cache.value = Tensor(Shape{batch, num_kv_heads_, initial_capacity, head_dim_});
        cache.batch_size = batch;
        cache.sequence_length = 0;
        cache.initialized = true;
    } else if (cache.batch_size != batch) {
        throw std::invalid_argument(
            "ZYRON MultiHeadAttention: cached batch size mismatch");
    }

    std::size_t capacity = cache.key.shape()[2];
    if (total > capacity) {
        std::size_t new_capacity = capacity;
        while (new_capacity < total) {
            if (new_capacity >= max_sequence_length_ - new_capacity) {
                new_capacity = max_sequence_length_;
                break;
            }
            new_capacity *= 2;
        }
        if (new_capacity > max_sequence_length_) new_capacity = max_sequence_length_;

        Tensor new_key(Shape{batch, num_kv_heads_, new_capacity, head_dim_});
        Tensor new_value(Shape{batch, num_kv_heads_, new_capacity, head_dim_});
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t h = 0; h < num_kv_heads_; ++h) {
                const float* old_k = cache.key.data() +
                    ((b * num_kv_heads_ + h) * capacity) * head_dim_;
                const float* old_v = cache.value.data() +
                    ((b * num_kv_heads_ + h) * capacity) * head_dim_;
                float* dst_k = new_key.data() +
                    ((b * num_kv_heads_ + h) * new_capacity) * head_dim_;
                float* dst_v = new_value.data() +
                    ((b * num_kv_heads_ + h) * new_capacity) * head_dim_;

                for (std::size_t token = 0; token < previous; ++token) {
                    std::copy_n(old_k + token * head_dim_, head_dim_,
                                dst_k + token * head_dim_);
                    std::copy_n(old_v + token * head_dim_, head_dim_,
                                dst_v + token * head_dim_);
                }
            }
        }
        cache.key = std::move(new_key);
        cache.value = std::move(new_value);
        capacity = new_capacity;
    }

    auto q = q_proj_.forward(input);
    auto k = k_proj_.forward(input);
    auto v = v_proj_.forward(input);

    q = autograd::reshape(
        q,
        Shape{batch, sequence, num_heads_, head_dim_});
    k = autograd::reshape(
        k,
        Shape{batch, sequence, num_kv_heads_, head_dim_});
    v = autograd::reshape(
        v,
        Shape{batch, sequence, num_kv_heads_, head_dim_});

    q = autograd::contiguous(autograd::transpose(q, 1, 2));
    k = autograd::contiguous(autograd::transpose(k, 1, 2));
    v = autograd::contiguous(autograd::transpose(v, 1, 2));

    if (use_rope_) {
        q = autograd::rotary_embedding(q, rope_cos_, rope_sin_, previous);
        k = autograd::rotary_embedding(k, rope_cos_, rope_sin_, previous);
    }

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t h = 0; h < num_kv_heads_; ++h) {
            const float* src_k = k.value().data() +
                ((b * num_kv_heads_ + h) * sequence) * head_dim_;
            const float* src_v = v.value().data() +
                ((b * num_kv_heads_ + h) * sequence) * head_dim_;
            float* dst_k = cache.key.data() +
                ((b * num_kv_heads_ + h) * cache.key.shape()[2] + previous) * head_dim_;
            float* dst_v = cache.value.data() +
                ((b * num_kv_heads_ + h) * cache.value.shape()[2] + previous) * head_dim_;

            for (std::size_t token = 0; token < sequence; ++token) {
                std::copy_n(src_k + token * head_dim_, head_dim_,
                            dst_k + token * head_dim_);
                std::copy_n(src_v + token * head_dim_, head_dim_,
                            dst_v + token * head_dim_);
            }
        }
    }

    const Tensor fused_context = fused_attention_inference(
        q.value(), cache.key, cache.value, previous, total,
        num_heads_, num_kv_heads_, head_dim_, causal_);
    auto context = fused_context.transpose(1, 2).contiguous().reshape(
        Shape{batch, sequence, hidden_size_});

    cache.sequence_length = total;
    return out_proj_.forward(autograd::Variable(std::move(context), false));
}


void MultiHeadAttention::set_qat(bool enabled, std::size_t bits) noexcept {
    q_proj_.set_qat(enabled, bits);
    k_proj_.set_qat(enabled, bits);
    v_proj_.set_qat(enabled, bits);
    out_proj_.set_qat(enabled, bits);
}

void MultiHeadAttention::train(bool mode) noexcept {
    Layer::train(mode);
    attention_dropout_.train(mode);
}

std::vector<autograd::Variable*> MultiHeadAttention::parameters() {
    auto result = q_proj_.parameters();
    auto append = [&result](std::vector<autograd::Variable*> values) {
        result.insert(result.end(), values.begin(), values.end());
    };
    append(k_proj_.parameters());
    append(v_proj_.parameters());
    append(out_proj_.parameters());
    return result;
}

} // namespace zyron::transformer
