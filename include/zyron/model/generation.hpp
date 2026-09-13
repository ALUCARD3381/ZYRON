#pragma once

#include "zyron/model/language_model.hpp"
#include "zyron/tokenizer/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zyron::model {

struct GenerationConfig {
    std::size_t max_new_tokens = 32;
    float temperature = 1.0f;
    std::size_t top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    std::size_t no_repeat_ngram_size = 0;
    std::uint32_t seed = 0;
};

[[nodiscard]] std::vector<std::size_t> generate(
    LanguageModel& model,
    const tokenizer::Tokenizer& tokenizer,
    const std::vector<std::size_t>& input_ids,
    const GenerationConfig& config = {});

[[nodiscard]] std::vector<std::vector<std::size_t>> generate_batch(
    LanguageModel& model,
    const tokenizer::Tokenizer& tokenizer,
    const std::vector<std::vector<std::size_t>>& input_ids,
    const GenerationConfig& config = {});

[[nodiscard]] std::vector<std::string> generate_text_batch(
    LanguageModel& model,
    const tokenizer::Tokenizer& tokenizer,
    const std::vector<std::string>& prompts,
    const GenerationConfig& config = {});

[[nodiscard]] std::string generate_text(
    LanguageModel& model,
    const tokenizer::Tokenizer& tokenizer,
    const std::string& prompt,
    const GenerationConfig& config = {});

} // namespace zyron::model
