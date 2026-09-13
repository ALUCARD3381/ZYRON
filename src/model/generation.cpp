#include "zyron/model/generation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <map>
#include <set>

namespace zyron::model {

namespace {

struct Candidate {
    std::size_t id;
    float logit;
    float probability;
};

void validate_config(
    const GenerationConfig& config) {

    if (!(config.temperature > 0.0f) ||
        !std::isfinite(config.temperature)) {
        throw std::invalid_argument(
            "ZYRON generation: temperature must be finite and > 0");
    }
    if (!(config.top_p > 0.0f) || config.top_p > 1.0f ||
        !std::isfinite(config.top_p)) {
        throw std::invalid_argument(
            "ZYRON generation: top_p must be in (0, 1]");
    }
    if (!(config.repetition_penalty >= 1.0f) ||
        !std::isfinite(config.repetition_penalty)) {
        throw std::invalid_argument(
            "ZYRON generation: repetition_penalty must be finite and >= 1");
    }
}

std::size_t sample_next(
    const Tensor& logits,
    std::size_t sequence_length,
    const GenerationConfig& config,
    const std::vector<std::size_t>& history,
    std::mt19937& rng) {

    const std::size_t vocab = logits.shape()[2];
    const std::size_t base = (sequence_length - 1) * vocab;

    for (std::size_t id = 0; id < vocab; ++id) {
        if (!std::isfinite(logits[base + id])) {
            throw std::runtime_error(
                "ZYRON generation: model produced non-finite logits");
        }
    }

    std::vector<Candidate> candidates;
    candidates.reserve(vocab);

    std::map<std::vector<std::size_t>, std::set<std::size_t>> seen_ngrams;
    if (config.no_repeat_ngram_size > 0 && history.size() >= config.no_repeat_ngram_size) {
        const std::size_t n = config.no_repeat_ngram_size;
        const std::size_t prefix_len = n - 1;
        for (std::size_t start = 0; start + n <= history.size(); ++start) {
            std::vector<std::size_t> prefix(history.begin() + static_cast<std::ptrdiff_t>(start),
                                            history.begin() + static_cast<std::ptrdiff_t>(start + prefix_len));
            seen_ngrams[prefix].insert(history[start + prefix_len]);
        }
    }

    std::vector<std::size_t> current_prefix;
    if (config.no_repeat_ngram_size > 0 && history.size() >= config.no_repeat_ngram_size - 1) {
        const std::size_t prefix_len = config.no_repeat_ngram_size - 1;
        current_prefix.assign(history.end() - static_cast<std::ptrdiff_t>(prefix_len), history.end());
    }
    const auto blocked_it = seen_ngrams.find(current_prefix);

    for (std::size_t id = 0; id < vocab; ++id) {
        float value = logits[base + id];

        if (std::find(history.begin(), history.end(), id) != history.end() &&
            config.repetition_penalty != 1.0f) {
            value = value >= 0.0f
                ? value / config.repetition_penalty
                : value * config.repetition_penalty;
        }

        if (blocked_it != seen_ngrams.end() && blocked_it->second.find(id) != blocked_it->second.end()) {
            value = -std::numeric_limits<float>::infinity();
        }

        candidates.push_back(Candidate{id, value, 0.0f});
    }

    bool has_finite_candidate = false;
    for (const auto& candidate : candidates) {
        has_finite_candidate = has_finite_candidate || std::isfinite(candidate.logit);
    }
    if (!has_finite_candidate && blocked_it != seen_ngrams.end()) {
        for (auto& candidate : candidates) {
            candidate.logit = logits[base + candidate.id];
        }
    }

    const float inv_temperature = 1.0f / config.temperature;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (auto& candidate : candidates) {
        candidate.logit *= inv_temperature;
        max_logit = std::max(max_logit, candidate.logit);
    }

    float denominator = 0.0f;
    for (auto& candidate : candidates) {
        candidate.probability = std::exp(candidate.logit - max_logit);
        denominator += candidate.probability;
    }
    for (auto& candidate : candidates) {
        candidate.probability /= denominator;
    }

    if (config.top_k != 0 && config.top_k < candidates.size()) {
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(config.top_k),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.probability > b.probability;
            });
        candidates.resize(config.top_k);
    } else {
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.probability > b.probability;
            });
    }

    if (config.top_p < 1.0f) {
        float cumulative = 0.0f;
        std::size_t keep = 0;
        for (const auto& candidate : candidates) {
            cumulative += candidate.probability;
            ++keep;
            if (cumulative >= config.top_p) break;
        }
        keep = std::max<std::size_t>(1, keep);
        candidates.resize(keep);
    }

    float probability_sum = 0.0f;
    for (const auto& candidate : candidates) probability_sum += candidate.probability;
    if (!(probability_sum > 0.0f)) {
        return candidates.front().id;
    }

    std::uniform_real_distribution<float> distribution(0.0f, probability_sum);
    const float sample = distribution(rng);

    float cumulative = 0.0f;
    for (const auto& candidate : candidates) {
        cumulative += candidate.probability;
        if (sample <= cumulative) return candidate.id;
    }

    return candidates.back().id;
}

} // namespace

std::vector<std::size_t> generate(
    LanguageModel& model,
    const tokenizer::Tokenizer& tokenizer,
    const std::vector<std::size_t>& input_ids,
    const GenerationConfig& config) {

    validate_config(config);

    if (input_ids.empty()) {
        throw std::invalid_argument(
            "ZYRON generation: input prompt cannot be empty");
    }
    if (input_ids.size() > model.config().max_sequence_length) {
        throw std::invalid_argument(
            "ZYRON generation: prompt exceeds model context length");
    }
    if (config.top_k != 0 && config.top_k > model.config().vocab_size) {
        throw std::invalid_argument(
            "ZYRON generation: top_k exceeds vocabulary size");
    }

    std::vector<std::size_t> tokens = input_ids;
    std::mt19937 rng(config.seed);
    const bool was_training = model.is_training();
    model.eval();

    autograd::NoGradGuard no_grad;
    LanguageModel::KVCache cache;
    auto logits = model.forward_cached(input_ids, cache);

    for (std::size_t step = 0; step < config.max_new_tokens; ++step) {
        if (tokens.size() >= model.config().max_sequence_length) break;

        const std::size_t sequence_length = logits.value().shape()[1];
        const std::size_t next = sample_next(
            logits.value(),
            sequence_length,
            config,
            tokens,
            rng);

        tokens.push_back(next);
        if (tokenizer.has_eos() && next == tokenizer.eos_id()) break;
        if (tokens.size() >= model.config().max_sequence_length) break;

        logits = model.forward_cached(std::vector<std::size_t>{next}, cache);
    }

    model.train(was_training);
    return tokens;
}

std::vector<std::vector<std::size_t>> generate_batch(
    LanguageModel& model,
    const tokenizer::Tokenizer& tokenizer,
    const std::vector<std::vector<std::size_t>>& input_ids,
    const GenerationConfig& config) {

    validate_config(config);
    if (input_ids.empty()) throw std::invalid_argument("ZYRON generation: batch cannot be empty");
    const std::size_t prompt_length = input_ids.front().size();
    if (prompt_length == 0) throw std::invalid_argument("ZYRON generation: prompts cannot be empty");
    if (input_ids.size() > model.config().max_sequence_length) {
        // This is not a hard model restriction; retained only to catch absurdly large batches early.
    }
    for (const auto& ids : input_ids) {
        if (ids.empty()) throw std::invalid_argument("ZYRON generation: prompts cannot be empty");
        if (ids.size() != prompt_length) {
            throw std::invalid_argument("ZYRON generation: batch prompts must have equal lengths");
        }
        if (ids.size() > model.config().max_sequence_length) {
            throw std::invalid_argument("ZYRON generation: prompt exceeds model context length");
        }
        for (const auto id : ids) {
            if (id >= model.config().vocab_size) throw std::out_of_range("ZYRON generation: token id out of vocabulary range");
        }
    }
    if (config.top_k != 0 && config.top_k > model.config().vocab_size) {
        throw std::invalid_argument("ZYRON generation: top_k exceeds vocabulary size");
    }

    std::vector<std::vector<std::size_t>> tokens = input_ids;
    std::vector<bool> finished(tokens.size(), false);
    std::mt19937 rng(config.seed);
    const bool was_training = model.is_training();
    model.eval();
    autograd::NoGradGuard no_grad;

    LanguageModel::KVCache cache;
    auto logits = model.forward_cached_batch(input_ids, cache);

    for (std::size_t step = 0; step < config.max_new_tokens; ++step) {
        if (cache.sequence_length >= model.config().max_sequence_length) break;
        std::vector<std::vector<std::size_t>> next_ids(tokens.size(), std::vector<std::size_t>(1, 0));
        bool any_active = false;
        const Tensor& value = logits.value();
        const std::size_t vocab = value.shape()[2];
        for (std::size_t b = 0; b < tokens.size(); ++b) {
            if (finished[b]) {
                next_ids[b][0] = tokenizer.has_eos() ? tokenizer.eos_id() : 0;
                continue;
            }
            std::vector<Candidate> candidates;
            candidates.reserve(vocab);
            const std::size_t base = b * value.shape()[1] * vocab + (value.shape()[1] - 1) * vocab;
            std::map<std::vector<std::size_t>, std::set<std::size_t>> seen_ngrams;
            if (config.no_repeat_ngram_size > 0 && tokens[b].size() >= config.no_repeat_ngram_size) {
                const auto n = config.no_repeat_ngram_size;
                const auto prefix_len = n - 1;
                for (std::size_t start = 0; start + n <= tokens[b].size(); ++start) {
                    std::vector<std::size_t> prefix(tokens[b].begin() + static_cast<std::ptrdiff_t>(start),
                                                     tokens[b].begin() + static_cast<std::ptrdiff_t>(start + prefix_len));
                    seen_ngrams[prefix].insert(tokens[b][start + prefix_len]);
                }
            }
            std::vector<std::size_t> prefix;
            if (config.no_repeat_ngram_size > 0 && tokens[b].size() >= config.no_repeat_ngram_size - 1) {
                const auto prefix_len = config.no_repeat_ngram_size - 1;
                prefix.assign(tokens[b].end() - static_cast<std::ptrdiff_t>(prefix_len), tokens[b].end());
            }
            const auto blocked = seen_ngrams.find(prefix);
            float max_logit = -std::numeric_limits<float>::infinity();
            for (std::size_t id = 0; id < vocab; ++id) {
                float logit = value[base + id];
                if (std::find(tokens[b].begin(), tokens[b].end(), id) != tokens[b].end() && config.repetition_penalty != 1.0f) {
                    logit = logit >= 0.0f ? logit / config.repetition_penalty : logit * config.repetition_penalty;
                }
                if (blocked != seen_ngrams.end() && blocked->second.count(id)) logit = -std::numeric_limits<float>::infinity();
                candidates.push_back({id, logit, 0.0f});
                max_logit = std::max(max_logit, logit);
            }
            bool any_finite = false;
            for (const auto& c : candidates) any_finite = any_finite || std::isfinite(c.logit);
            if (!any_finite && blocked != seen_ngrams.end()) {
                for (auto& c : candidates) c.logit = value[base + c.id];
                max_logit = -std::numeric_limits<float>::infinity();
                for (const auto& c : candidates) max_logit = std::max(max_logit, c.logit);
            }
            const float inv_temp = 1.0f / config.temperature;
            float denom = 0.0f;
            for (auto& c : candidates) { c.logit *= inv_temp; max_logit = std::max(max_logit, c.logit); }
            for (auto& c : candidates) { c.probability = std::exp(c.logit - max_logit); denom += c.probability; }
            for (auto& c : candidates) c.probability /= denom;
            std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.probability > b.probability; });
            if (config.top_k != 0 && config.top_k < candidates.size()) candidates.resize(config.top_k);
            if (config.top_p < 1.0f) {
                float cumulative = 0.0f; std::size_t keep = 0;
                for (const auto& c : candidates) { cumulative += c.probability; ++keep; if (cumulative >= config.top_p) break; }
                candidates.resize(std::max<std::size_t>(1, keep));
            }
            float sum = 0.0f; for (const auto& c : candidates) sum += c.probability;
            std::uniform_real_distribution<float> dist(0.0f, sum);
            const float draw = dist(rng);
            float cumulative = 0.0f;
            std::size_t next = candidates.back().id;
            for (const auto& c : candidates) { cumulative += c.probability; if (draw <= cumulative) { next = c.id; break; } }
            next_ids[b][0] = next;
            any_active = true;
        }
        if (!any_active) break;
        for (std::size_t b = 0; b < tokens.size(); ++b) {
            if (!finished[b]) {
                tokens[b].push_back(next_ids[b][0]);
                if (tokenizer.has_eos() && next_ids[b][0] == tokenizer.eos_id()) finished[b] = true;
            }
        }
        logits = model.forward_cached_batch(next_ids, cache);
        bool all_finished = true;
        for (bool done : finished) all_finished = all_finished && done;
        if (all_finished) break;
    }

    model.train(was_training);
    return tokens;
}

std::vector<std::string> generate_text_batch(
    LanguageModel& model,
    const tokenizer::Tokenizer& tokenizer,
    const std::vector<std::string>& prompts,
    const GenerationConfig& config) {
    std::vector<std::vector<std::size_t>> ids;
    ids.reserve(prompts.size());
    for (const auto& prompt : prompts) ids.push_back(tokenizer.encode(prompt));
    const auto generated = generate_batch(model, tokenizer, ids, config);
    std::vector<std::string> result;
    result.reserve(generated.size());
    for (const auto& sequence : generated) result.push_back(tokenizer.decode(sequence, true));
    return result;
}

std::string generate_text(
    LanguageModel& model,
    const tokenizer::Tokenizer& tokenizer,
    const std::string& prompt,
    const GenerationConfig& config) {

    const auto ids = tokenizer.encode(prompt);
    const auto generated = generate(model, tokenizer, ids, config);
    return tokenizer.decode(generated, true);
}

} // namespace zyron::model
