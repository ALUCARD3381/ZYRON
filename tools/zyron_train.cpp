#include "zyron/core/runtime.hpp"
#include "zyron/model/checkpoint.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/tokenizer/bpe.hpp"
#include "zyron/training/adamw.hpp"
#include "zyron/training/dataloader.hpp"
#include "zyron/training/trainer.hpp"
#include "zyron/training/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace zyron;

namespace {

struct Args {
    std::vector<std::string> data;
    std::string tokenizer_path;
    std::string tokenizer_out{"models/tokenizer.zytok"};
    std::string resume;
    std::size_t vocab_size{320};
    std::size_t seq_len{128};
    std::size_t batch_size{4};
    std::size_t hidden_size{64};
    std::size_t layers{2};
    std::size_t heads{4};
    std::size_t kv_heads{0};
    std::size_t intermediate_size{256};
    float lr{3e-4f};
    float weight_decay{0.01f};
    std::size_t warmup_steps{100};
    std::size_t steps{1000};
    float val_split{0.1f};
    std::size_t workers{1};
    std::size_t prefetch{2};
    std::size_t threads{1};
    std::size_t log_interval{10};
    std::size_t eval_interval{100};
    std::size_t checkpoint_interval{500};
    float grad_clip{1.0f};
    std::size_t gradient_accumulation_steps{1};
    float dropout{0.0f};
    bool qat{false};
    std::size_t qat_bits{8};
    std::string checkpoint_path{"models/checkpoint.zyron"};
    std::string log_path;
};

std::size_t parse_size(const std::string& s) { return static_cast<std::size_t>(std::stoull(s)); }
float parse_float(const std::string& s) { return std::stof(s); }

Args parse(int argc, char** argv) {
    Args a;
    for (int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        auto need=[&](const char* name)->std::string{
            if (i+1>=argc) throw std::invalid_argument(std::string("missing value for ")+name);
            return argv[++i];
        };
        if (arg=="--data") a.data.push_back(need("--data"));
        else if (arg=="--tokenizer") a.tokenizer_path=need("--tokenizer");
        else if (arg=="--tokenizer-out") a.tokenizer_out=need("--tokenizer-out");
        else if (arg=="--resume") a.resume=need("--resume");
        else if (arg=="--vocab-size") a.vocab_size=parse_size(need("--vocab-size"));
        else if (arg=="--seq-len") a.seq_len=parse_size(need("--seq-len"));
        else if (arg=="--batch-size") a.batch_size=parse_size(need("--batch-size"));
        else if (arg=="--hidden-size") a.hidden_size=parse_size(need("--hidden-size"));
        else if (arg=="--layers") a.layers=parse_size(need("--layers"));
        else if (arg=="--heads") a.heads=parse_size(need("--heads"));
        else if (arg=="--kv-heads") a.kv_heads=parse_size(need("--kv-heads"));
        else if (arg=="--intermediate-size") a.intermediate_size=parse_size(need("--intermediate-size"));
        else if (arg=="--lr") a.lr=parse_float(need("--lr"));
        else if (arg=="--weight-decay") a.weight_decay=parse_float(need("--weight-decay"));
        else if (arg=="--warmup-steps") a.warmup_steps=parse_size(need("--warmup-steps"));
        else if (arg=="--steps") a.steps=parse_size(need("--steps"));
        else if (arg=="--val-split") a.val_split=parse_float(need("--val-split"));
        else if (arg=="--workers") a.workers=parse_size(need("--workers"));
        else if (arg=="--prefetch") a.prefetch=parse_size(need("--prefetch"));
        else if (arg=="--threads") a.threads=parse_size(need("--threads"));
        else if (arg=="--log-interval") a.log_interval=parse_size(need("--log-interval"));
        else if (arg=="--eval-interval") a.eval_interval=parse_size(need("--eval-interval"));
        else if (arg=="--checkpoint-interval") a.checkpoint_interval=parse_size(need("--checkpoint-interval"));
        else if (arg=="--grad-clip") a.grad_clip=parse_float(need("--grad-clip"));
        else if (arg=="--gradient-accumulation") a.gradient_accumulation_steps=parse_size(need("--gradient-accumulation"));
        else if (arg=="--dropout") a.dropout=parse_float(need("--dropout"));
        else if (arg=="--qat") a.qat=true;
        else if (arg=="--qat-bits") a.qat_bits=parse_size(need("--qat-bits"));
        else if (arg=="--checkpoint") a.checkpoint_path=need("--checkpoint");
        else if (arg=="--log") a.log_path=need("--log");
        else if (arg=="--help") {
            std::cout << "ZYRON train options:\n"
                      << "  --data FILE [--data FILE ...]\n"
                      << "  --tokenizer FILE | --vocab-size N\n"
                      << "  --seq-len N --batch-size N --hidden-size N --layers N --heads N --kv-heads N\n"
                      << "  --intermediate-size N --lr F --weight-decay F\n"
                      << "  --warmup-steps N --steps N --val-split F\n"
                      << "  --workers N --prefetch N --threads N\n"
                      << "  --log-interval N --eval-interval N --checkpoint-interval N\n"
                      << "  --grad-clip F --gradient-accumulation N --dropout F --qat [--qat-bits 4|8]\n"
                      << "  --checkpoint FILE --log FILE\n"
                      << "  --resume FILE\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown argument: "+arg);
    }
    if (a.data.empty() && a.resume.empty()) throw std::invalid_argument("--data is required for new training");
    if (!(a.val_split >= 0.0f && a.val_split < 1.0f)) throw std::invalid_argument("--val-split must be in [0,1)");
    if (a.gradient_accumulation_steps == 0) a.gradient_accumulation_steps = 1;
    if (!(a.dropout >= 0.0f) || a.dropout >= 1.0f || !std::isfinite(a.dropout)) throw std::invalid_argument("--dropout must be finite and in [0,1)");
    if (a.qat_bits != 4 && a.qat_bits != 8) throw std::invalid_argument("--qat-bits must be 4 or 8");
    if (a.threads == 0) a.threads=1;
    return a;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args=parse(argc,argv);
        runtime::set_num_threads(args.threads);
        tokenizer::BPE tokenizer;
        transformer::Config config;
        std::size_t initial_step=0;

        if (!args.resume.empty()) {
            const auto state=model::Checkpoint::load(args.resume);
            config=state.config;
            model::LanguageModel model(config,42);
            training::AdamW optimizer(
                model.parameters(),
                state.optimizer_learning_rate,
                state.optimizer_beta1,
                state.optimizer_beta2,
                state.optimizer_epsilon,
                state.optimizer_weight_decay);
            training::WarmupCosineScheduler scheduler(
                state.scheduler_base_learning_rate,
                state.scheduler_warmup_steps,
                state.scheduler_total_steps,
                state.scheduler_min_learning_rate);
            model::Checkpoint::restore(state,model,optimizer,scheduler,tokenizer);
            initial_step=state.training_step;

            training::DatasetConfig ds;
            ds.paths=args.data;
            ds.tokenizer=&tokenizer;
            ds.sequence_length=config.max_sequence_length;
            ds.validation_period = args.val_split > 0.0f
                ? std::max<std::size_t>(2, static_cast<std::size_t>(std::llround(1.0f / args.val_split)))
                : 0;

            training::DataLoaderConfig dl;
            dl.dataset=ds; dl.batch_size=args.batch_size; dl.workers=args.workers; dl.prefetch_batches=args.prefetch;
            dl.seed=1234; dl.shuffle=true;
            training::DataLoader train_loader(dl);
            training::DatasetConfig vds=ds; vds.validation_only=true;
            dl.dataset=vds;
            training::DataLoader val_loader(dl);

            training::TrainerConfig tc;
            tc.max_steps=args.steps; tc.log_interval=args.log_interval; tc.eval_interval=args.eval_interval;
        tc.checkpoint_interval=args.checkpoint_interval; tc.gradient_clip_norm=args.grad_clip;
        tc.gradient_accumulation_steps=args.gradient_accumulation_steps;
            tc.checkpoint_path=args.checkpoint_path; tc.log_path=args.log_path;
            training::Trainer trainer(model,tokenizer,optimizer,scheduler,train_loader,&val_loader,tc);
            const auto result = trainer.train(initial_step);
            (void)result;
            return 0;
        }

        if (!args.tokenizer_path.empty()) {
            tokenizer.load(args.tokenizer_path);
        } else {
            tokenizer.train_files(args.data,args.vocab_size);

            // Very small/repetitive corpora can legitimately collapse into too few
            // BPE tokens. For an initial language-model experiment, keep a byte-level
            // fallback so that seq_len+1 tokens are available without loading the
            // entire dataset into RAM.
            bool enough_tokens = false;
            if (!args.data.empty()) {
                std::ifstream probe(args.data.front(), std::ios::binary);
                if (probe) {
                    std::string chunk(1u << 20, '\0');
                    probe.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
                    chunk.resize(static_cast<std::size_t>(probe.gcount()));
                    const auto ids = tokenizer.encode(chunk, true, true);
                    enough_tokens = ids.size() >= args.seq_len + 1;
                }
            }
            if (!enough_tokens) {
                tokenizer.train_files(args.data, 260);
            }

            if (!std::filesystem::path(args.tokenizer_out).parent_path().empty()) std::filesystem::create_directories(std::filesystem::path(args.tokenizer_out).parent_path());
            tokenizer.save(args.tokenizer_out);
        }

        config=transformer::tiny(tokenizer.vocab_size());
        config.hidden_size=args.hidden_size;
        config.num_layers=args.layers;
        config.num_heads=args.heads;
        if (args.kv_heads != 0) config.num_kv_heads=args.kv_heads;
        config.intermediate_size=args.intermediate_size;
        config.max_sequence_length=args.seq_len;
        config.dropout=args.dropout;
        config.use_qat=args.qat;
        config.qat_bits=args.qat_bits;
        config.validate();

        model::LanguageModel model(config,42);
        training::AdamW optimizer(model.parameters(),args.lr,0.9f,0.999f,1e-8f,args.weight_decay);
        training::WarmupCosineScheduler scheduler(args.lr,args.warmup_steps,args.steps);

        const std::size_t validation_period = args.val_split > 0.0f
            ? std::max<std::size_t>(2, static_cast<std::size_t>(std::llround(1.0f / args.val_split)))
            : 0;
        training::DatasetConfig ds;
        ds.paths=args.data; ds.tokenizer=&tokenizer; ds.sequence_length=args.seq_len;
        ds.validation_period=validation_period; ds.validation_only=false;

        training::DataLoaderConfig dl;
        dl.dataset=ds; dl.batch_size=args.batch_size; dl.workers=args.workers; dl.prefetch_batches=args.prefetch; dl.shuffle=true; dl.seed=1234;
        training::DataLoader train_loader(dl);
        training::DatasetConfig vds=ds; vds.validation_only=true;
        dl.dataset=vds;
        training::DataLoader val_loader(dl);

        if (!std::filesystem::path(args.checkpoint_path).parent_path().empty()) std::filesystem::create_directories(std::filesystem::path(args.checkpoint_path).parent_path());
        training::TrainerConfig tc;
        tc.max_steps=args.steps; tc.log_interval=args.log_interval; tc.eval_interval=args.eval_interval;
        tc.checkpoint_interval=args.checkpoint_interval; tc.gradient_clip_norm=args.grad_clip;
            tc.gradient_accumulation_steps=args.gradient_accumulation_steps;
        tc.checkpoint_path=args.checkpoint_path; tc.log_path=args.log_path;
        training::Trainer trainer(model,tokenizer,optimizer,scheduler,train_loader,&val_loader,tc);
        const auto result = trainer.train(0);
        (void)result;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ZYRON train error: " << e.what() << '\n';
        return 1;
    }
}
