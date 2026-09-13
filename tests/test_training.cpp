#include "zyron/model/checkpoint.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/tokenizer/bpe.hpp"
#include "zyron/training/adamw.hpp"
#include "zyron/training/dataloader.hpp"
#include "zyron/training/scheduler.hpp"
#include "zyron/training/trainer.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace zyron;
using namespace zyron::training;

static void check(bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(msg);
}

int main() {
    try {
        const std::string data_path = "phase4_test_dataset.txt";
        {
            std::ofstream out(data_path);
            for (int i = 0; i < 30; ++i) {
                out << "o gato dorme. o gato corre. o cao dorme. "
                       "o gato come. o cao corre.\n";
            }
        }

        tokenizer::BPE tokenizer;
        tokenizer.train_text(
            "o gato dorme. o gato corre. o cao dorme. o gato come. o cao corre.",
            280);

        transformer::Config config = transformer::tiny(tokenizer.vocab_size());
        config.hidden_size = 16;
        config.num_layers = 1;
        config.num_heads = 2;
        config.intermediate_size = 32;
        config.max_sequence_length = 8;
        config.dropout = 0.1f;
        config.validate();

        DatasetConfig train_ds;
        train_ds.paths = {data_path};
        train_ds.tokenizer = &tokenizer;
        train_ds.sequence_length = 8;
        train_ds.chunk_bytes = 32;
        train_ds.add_eos = true;
        train_ds.validation_period = 5;
        train_ds.validation_only = false;

        DatasetConfig val_ds = train_ds;
        val_ds.validation_only = true;

        DataLoaderConfig dl_cfg;
        dl_cfg.dataset = train_ds;
        dl_cfg.batch_size = 2;
        dl_cfg.shuffle = false;
        dl_cfg.prefetch_batches = 0;
        dl_cfg.drop_last = false;

        DataLoader train_loader(dl_cfg);
        dl_cfg.dataset = val_ds;
        DataLoader val_loader(dl_cfg);

        model::LanguageModel model(config, 42);
        AdamW optimizer(model.parameters(), 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);
        WarmupCosineScheduler scheduler(1e-3f, 2, 4, 1e-4f);

        Batch batch;
        check(train_loader.next(batch), "loader next");
        check(!batch.empty(), "non-empty batch");
        check(batch.input_ids.front().size() == 8, "sequence length");

        DataLoaderConfig threaded_cfg = dl_cfg;
        threaded_cfg.prefetch_batches = 2;
        threaded_cfg.workers = 2;
        threaded_cfg.shuffle = true;
        DataLoader threaded_loader(threaded_cfg);
        check(threaded_loader.next(batch), "threaded loader next");
        check(!batch.empty(), "threaded non-empty batch");

        const auto initial = model.loss(batch.input_ids, batch.target_ids).value()[0];
        optimizer.zero_grad();
        auto loss = model.loss(batch.input_ids, batch.target_ids);
        loss.backward();
        const float norm = global_grad_norm(optimizer.parameters());
        check(std::isfinite(norm), "finite grad norm");
        clip_grad_norm(optimizer.parameters(), 1.0f);
        scheduler.step(optimizer);
        optimizer.step();
        const auto after = model.loss(batch.input_ids, batch.target_ids).value()[0];
        check(std::isfinite(initial) && std::isfinite(after), "finite loss");

        const std::string checkpoint = "phase4_test.zyron";
        model.eval();
        const auto after_eval = model.loss(batch.input_ids, batch.target_ids).value()[0];
        model::Checkpoint::save(
            checkpoint, model, optimizer, scheduler, tokenizer, 1);

        const auto state = model::Checkpoint::load(checkpoint);
        check(state.training_step == 1, "training step checkpoint");
        check(state.model_parameters.size() == model.parameters().size(), "checkpoint parameters");
        check(state.first_moments.size() == model.parameters().size(), "checkpoint moments");
        check(state.scheduler_total_steps == 4, "checkpoint scheduler");
        check(std::fabs(state.config.dropout - config.dropout) < 1e-6f, "checkpoint dropout config");

        model::LanguageModel restored(config, 7);
        AdamW restored_optimizer(restored.parameters(), 1e-3f);
        WarmupCosineScheduler restored_scheduler(1e-3f, 2, 4, 1e-4f);
        tokenizer::BPE restored_tokenizer;
        model::Checkpoint::restore(
            state, restored, restored_optimizer, restored_scheduler, restored_tokenizer);

        check(restored_tokenizer.vocab_size() == tokenizer.vocab_size(), "tokenizer checkpoint");
        restored.eval();
        auto restored_loss = restored.loss(batch.input_ids, batch.target_ids);
        check(std::fabs(restored_loss.value()[0] - after_eval) < 1e-5f, "restored weights");

        training::TrainerConfig trainer_config;
        trainer_config.max_steps = 1;
        trainer_config.log_interval = 1;
        trainer_config.eval_interval = 100;
        trainer_config.checkpoint_interval = 100;
        trainer_config.checkpoint_path.clear();
        trainer_config.gradient_accumulation_steps = 2;
        model::LanguageModel accumulated_model(config, 77);
        AdamW accumulated_optimizer(accumulated_model.parameters(), 1e-3f);
        WarmupCosineScheduler accumulated_scheduler(1e-3f, 1, 1);
        DataLoader accumulated_loader(dl_cfg);
        Trainer accumulated_trainer(
            accumulated_model, tokenizer, accumulated_optimizer, accumulated_scheduler,
            accumulated_loader, nullptr, trainer_config);
        const auto accumulated_state = accumulated_trainer.train(0);
        check(accumulated_state.step == 1, "gradient accumulation trainer step");
        check(accumulated_optimizer.step_count() == 1, "gradient accumulation optimizer step");
        check(accumulated_scheduler.step_count() == 1, "gradient accumulation scheduler step");

        std::filesystem::remove(data_path);
        std::filesystem::remove(checkpoint);
        std::cout << "ZYRON Phase 4 training tests: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ZYRON Phase 4 training tests: FAIL: " << e.what() << '\n';
        return 1;
    }
}
