#include "zyron/model/checkpoint.hpp"
#include "zyron/model/generation.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/tokenizer/bpe.hpp"
#include "zyron/training/adamw.hpp"
#include "zyron/training/scheduler.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace zyron;

namespace {

struct Args {
    std::string checkpoint;
    std::string prompt{""};
    std::size_t max_new_tokens{64};
    float temperature{0.8f};
    std::size_t top_k{40};
    float top_p{1.0f};
    float repetition_penalty{1.1f};
    std::size_t no_repeat_ngram_size{0};
    std::uint32_t seed{0};
};

std::string need(int argc, char** argv, int& i, const char* name) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
    }
    return argv[++i];
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--checkpoint") a.checkpoint = need(argc, argv, i, "--checkpoint");
        else if (arg == "--prompt") a.prompt = need(argc, argv, i, "--prompt");
        else if (arg == "--max-new-tokens") a.max_new_tokens = std::stoull(need(argc, argv, i, "--max-new-tokens"));
        else if (arg == "--temperature") a.temperature = std::stof(need(argc, argv, i, "--temperature"));
        else if (arg == "--top-k") a.top_k = std::stoull(need(argc, argv, i, "--top-k"));
        else if (arg == "--top-p") a.top_p = std::stof(need(argc, argv, i, "--top-p"));
        else if (arg == "--repetition-penalty") a.repetition_penalty = std::stof(need(argc, argv, i, "--repetition-penalty"));
        else if (arg == "--no-repeat-ngram-size") a.no_repeat_ngram_size = std::stoull(need(argc, argv, i, "--no-repeat-ngram-size"));
        else if (arg == "--seed") a.seed = static_cast<std::uint32_t>(std::stoul(need(argc, argv, i, "--seed")));
        else if (arg == "--help") {
            std::cout <<
                "zyron_generate --checkpoint <path.zyron> --prompt \"texto\" "
                "[--max-new-tokens N] [--temperature F] [--top-k N] [--top-p F] "
                "[--repetition-penalty F] [--no-repeat-ngram-size N] [--seed N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (a.checkpoint.empty()) {
        throw std::runtime_error("--checkpoint is required");
    }
    return a;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse(argc, argv);

        std::cout << "Loading checkpoint: " << args.checkpoint << "\n";
        model::CheckpointState state = model::Checkpoint::load(args.checkpoint);

        model::LanguageModel llm(state.config);
        training::AdamW optimizer(llm.parameters());
        training::WarmupCosineScheduler scheduler(
            state.scheduler_base_learning_rate,
            state.scheduler_warmup_steps,
            state.scheduler_total_steps,
            state.scheduler_min_learning_rate);
        tokenizer::BPE tokenizer;

        model::Checkpoint::restore(state, llm, optimizer, scheduler, tokenizer);
        llm.eval();

        std::cout << "Checkpoint carregado. training_step=" << state.training_step
                  << " vocab_size=" << state.config.vocab_size << "\n";

        model::GenerationConfig gen_cfg;
        gen_cfg.max_new_tokens = args.max_new_tokens;
        gen_cfg.temperature = args.temperature;
        gen_cfg.top_k = args.top_k;
        gen_cfg.top_p = args.top_p;
        gen_cfg.repetition_penalty = args.repetition_penalty;
        gen_cfg.no_repeat_ngram_size = args.no_repeat_ngram_size;
        gen_cfg.seed = args.seed;

        std::string output = model::generate_text(llm, tokenizer, args.prompt, gen_cfg);

        std::cout << "\n--- Prompt ---\n" << args.prompt
                  << "\n--- Geracao ---\n" << output << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "zyron_generate error: " << e.what() << "\n";
        return 1;
    }
}
