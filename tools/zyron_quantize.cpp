#include "zyron/model/checkpoint.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/model/quantized_language_model.hpp"
#include "zyron/tokenizer/bpe.hpp"
#include "zyron/training/adamw.hpp"
#include "zyron/training/scheduler.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace zyron;

namespace {
struct Args { std::string input; std::string output; quantization::Type type{quantization::Type::Int8}; };
std::string need(int argc,char** argv,int& i,const char* name){ if(i+1>=argc) throw std::runtime_error(std::string("missing value for ")+name); return argv[++i]; }
Args parse(int argc,char** argv){
    Args a;
    for(int i=1;i<argc;++i){
        std::string arg=argv[i];
        if(arg=="--checkpoint") a.input=need(argc,argv,i,"--checkpoint");
        else if(arg=="--output") a.output=need(argc,argv,i,"--output");
        else if(arg=="--bits"){ const auto bits=std::stoull(need(argc,argv,i,"--bits")); if(bits==8)a.type=quantization::Type::Int8; else if(bits==4)a.type=quantization::Type::Int4; else throw std::invalid_argument("--bits must be 4 or 8"); }
        else if(arg=="--help"){ std::cout << "zyron_quantize --checkpoint <train.zyron> --output <model.zyqmodel> [--bits 4|8]\n"; std::exit(0); }
        else throw std::runtime_error("unknown argument: "+arg);
    }
    if(a.input.empty()||a.output.empty()) throw std::runtime_error("--checkpoint and --output are required");
    return a;
}
}

int main(int argc,char** argv){
    try{
        const auto args=parse(argc,argv);
        const auto state=model::Checkpoint::load(args.input);
        model::LanguageModel model(state.config,42);
        training::AdamW optimizer(model.parameters(), state.optimizer_learning_rate, state.optimizer_beta1, state.optimizer_beta2, state.optimizer_epsilon, state.optimizer_weight_decay);
        training::WarmupCosineScheduler scheduler(state.scheduler_base_learning_rate,state.scheduler_warmup_steps,state.scheduler_total_steps,state.scheduler_min_learning_rate);
        tokenizer::BPE tokenizer;
        model::Checkpoint::restore(state,model,optimizer,scheduler,tokenizer);
        model::QuantizedLanguageModel quantized(model,args.type);
        quantized.save(args.output);
        std::cout << "ZYRON quantized model saved: " << args.output << "\n"
                  << "compression=" << quantized.compression_ratio() << "x\n";
        return 0;
    } catch(const std::exception& e){ std::cerr << "zyron_quantize error: " << e.what() << '\n'; return 1; }
}
