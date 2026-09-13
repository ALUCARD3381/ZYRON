#include "zyron/model/language_model.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <unordered_set>

namespace zyron::model {

namespace {

std::vector<std::size_t> flatten_ids(
    const std::vector<std::vector<std::size_t>>& batch,
    std::size_t& batch_size,
    std::size_t& sequence_length) {

    if (batch.empty()) {
        throw std::invalid_argument("ZYRON LanguageModel: batch cannot be empty");
    }
    if (batch.front().empty()) {
        throw std::invalid_argument("ZYRON LanguageModel: sequence cannot be empty");
    }

    batch_size = batch.size();
    sequence_length = batch.front().size();

    if (sequence_length == 0) {
        throw std::invalid_argument("ZYRON LanguageModel: sequence cannot be empty");
    }

    std::size_t total = 0;
    if (sequence_length != 0 &&
        batch_size > std::numeric_limits<std::size_t>::max() / sequence_length) {
        throw std::overflow_error("ZYRON LanguageModel: batch size overflow");
    }
    total = batch_size * sequence_length;

    std::vector<std::size_t> result;
    result.reserve(total);

    for (const auto& sequence : batch) {
        if (sequence.size() != sequence_length) {
            throw std::invalid_argument(
                "ZYRON LanguageModel: all sequences must have equal length");
        }
        result.insert(result.end(), sequence.begin(), sequence.end());
    }

    return result;
}

} // namespace

LanguageModel::LanguageModel(
    const transformer::Config& config,
    std::uint32_t seed)
    : config_(config),
      token_embedding_(
          config.vocab_size,
          config.hidden_size,
          seed + 1),
      transformer_(config, seed + 100),
      final_norm_(config.hidden_size, config.norm_eps),
      lm_head_(
          config.hidden_size,
          config.vocab_size,
          config.tie_word_embeddings ? &token_embedding_.weight() : nullptr,
          true,
          true,
          seed + 200) {

    config_.validate();
    token_embedding_.set_qat(config_.use_qat, config_.qat_bits);
    transformer_.set_qat(config_.use_qat, config_.qat_bits);
    lm_head_.set_qat(config_.use_qat, config_.qat_bits);
}

autograd::Variable LanguageModel::forward(
    const std::vector<std::size_t>& token_ids) {
    return forward(std::vector<std::vector<std::size_t>>{token_ids});
}

autograd::Variable LanguageModel::forward(
    const std::vector<std::vector<std::size_t>>& batch_token_ids) {

    std::size_t batch = 0;
    std::size_t sequence = 0;
    const auto flat_ids = flatten_ids(
        batch_token_ids,
        batch,
        sequence);

    for (const auto id : flat_ids) {
        if (id >= config_.vocab_size) {
            throw std::out_of_range(
                "ZYRON LanguageModel: token id out of vocabulary range");
        }
    }

    auto embedded = token_embedding_.forward(flat_ids);
    embedded = autograd::reshape(
        embedded,
        Shape{batch, sequence, config_.hidden_size});

    auto hidden = transformer_.forward(embedded);
    hidden = final_norm_.forward(hidden);
    return lm_head_.forward(hidden);
}

autograd::Variable LanguageModel::forward_cached(
    const std::vector<std::size_t>& token_ids,
    KVCache& cache) {

    if (token_ids.empty()) {
        throw std::invalid_argument(
            "ZYRON LanguageModel: cached input cannot be empty");
    }
    if (cache.sequence_length > config_.max_sequence_length) {
        throw std::out_of_range(
            "ZYRON LanguageModel: cached sequence exceeds context");
    }
    if (token_ids.size() > config_.max_sequence_length - cache.sequence_length) {
        throw std::out_of_range(
            "ZYRON LanguageModel: cached input exceeds context");
    }
    for (const auto id : token_ids) {
        if (id >= config_.vocab_size) {
            throw std::out_of_range(
                "ZYRON LanguageModel: cached token id out of vocabulary range");
        }
    }

    autograd::Variable embedded = token_embedding_.forward(token_ids);
    embedded = autograd::reshape(
        embedded,
        Shape{1, token_ids.size(), config_.hidden_size});

    auto hidden = transformer_.forward_cached(embedded, cache);
    hidden = final_norm_.forward(hidden);
    return lm_head_.forward(hidden);
}

autograd::Variable LanguageModel::forward_cached_batch(
    const std::vector<std::vector<std::size_t>>& batch_token_ids,
    KVCache& cache) {

    std::size_t batch = 0;
    std::size_t sequence = 0;
    const auto flat_ids = flatten_ids(batch_token_ids, batch, sequence);
    for (const auto id : flat_ids) {
        if (id >= config_.vocab_size) {
            throw std::out_of_range("ZYRON LanguageModel: cached token id out of vocabulary range");
        }
    }
    if (!batch_token_ids.empty() && batch_token_ids.size() != cache.batch_size && cache.initialized) {
        // Transformer performs the authoritative cache batch-size validation.
    }

    auto embedded = token_embedding_.forward(flat_ids);
    embedded = autograd::reshape(
        embedded,
        Shape{batch, sequence, config_.hidden_size});
    auto hidden = transformer_.forward_cached(embedded, cache);
    hidden = final_norm_.forward(hidden);
    return lm_head_.forward(hidden);
}

autograd::Variable LanguageModel::loss(
    const std::vector<std::vector<std::size_t>>& input_ids,
    const std::vector<std::vector<std::size_t>>& target_ids) {

    std::size_t input_batch = 0;
    std::size_t input_sequence = 0;
    (void)flatten_ids(input_ids, input_batch, input_sequence);

    std::size_t target_batch = 0;
    std::size_t target_sequence = 0;
    const auto flat_targets = flatten_ids(
        target_ids,
        target_batch,
        target_sequence);

    if (input_batch != target_batch || input_sequence != target_sequence) {
        throw std::invalid_argument(
            "ZYRON LanguageModel: input and target shape mismatch");
    }

    const auto logits = forward(input_ids);
    return autograd::cross_entropy(logits, flat_targets);
}

std::vector<autograd::Variable*> LanguageModel::parameters() {
    std::vector<autograd::Variable*> result;
    std::unordered_set<autograd::Variable*> seen;
    auto append_unique = [&](const std::vector<autograd::Variable*>& params) {
        for (auto* parameter : params) {
            if (parameter && seen.insert(parameter).second) result.push_back(parameter);
        }
    };

    append_unique(token_embedding_.parameters());
    append_unique(transformer_.parameters());
    append_unique(final_norm_.parameters());
    append_unique(lm_head_.parameters());
    return result;
}

void LanguageModel::zero_grad() noexcept {
    const auto params = parameters();
    for (auto* parameter : params) {
        if (parameter) parameter->zero_grad();
    }
}

void LanguageModel::train(bool mode) noexcept {
    training_ = mode;
    token_embedding_.train(mode);
    transformer_.train(mode);
    final_norm_.train(mode);
    lm_head_.train(mode);
}

} // namespace zyron::model
