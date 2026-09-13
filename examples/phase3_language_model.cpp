#include "zyron/model/generation.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/tokenizer/bpe.hpp"
#include "zyron/transformer/config.hpp"

#include <iostream>

int main() {
    zyron::tokenizer::BPE tokenizer;
    tokenizer.train_text(
        "Olá mundo. O gato dorme. O gato corre. Olá mundo. ",
        320);

    zyron::transformer::Config config;
    config.vocab_size = tokenizer.vocab_size();
    config.hidden_size = 64;
    config.num_layers = 2;
    config.num_heads = 4;
    config.intermediate_size = 128;
    config.max_sequence_length = 64;
    config.validate();

    zyron::model::LanguageModel model(config, 42);

    zyron::model::GenerationConfig generation;
    generation.max_new_tokens = 12;
    generation.temperature = 0.8f;
    generation.top_k = 20;
    generation.top_p = 0.95f;
    generation.seed = 123;

    std::cout << zyron::model::generate_text(
        model,
        tokenizer,
        "Olá ",
        generation)
              << '\n';
}
