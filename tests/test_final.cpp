#include "zyron/autograd/autograd.hpp"
#include "zyron/model/generation.hpp"
#include "zyron/model/language_model.hpp"
#include "zyron/model/quantized_language_model.hpp"
#include "zyron/quantization/quantization.hpp"
#include "zyron/transformer/config.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool ok,const char* msg){ if(!ok) throw std::runtime_error(msg); }
void compare(const zyron::Tensor& a,const zyron::Tensor& b,float eps,const char* msg){ check(a.shape()==b.shape(),msg); for(std::size_t i=0;i<a.size();++i) if(std::fabs(a[i]-b[i])>eps) throw std::runtime_error(msg); }

class DummyTokenizer final : public zyron::tokenizer::Tokenizer {
public:
    std::vector<std::size_t> encode(const std::string&,bool=false,bool=false) const override { return {1,2}; }
    std::string decode(const std::vector<std::size_t>& ids,bool=true) const override { return std::to_string(ids.size()); }
    std::size_t vocab_size() const noexcept override { return 32; }
};

void test_fake_quant_ste(){
    zyron::autograd::Variable x(zyron::Tensor::random(zyron::Shape{8},-1.0f,1.0f,7),true);
    auto q=zyron::autograd::fake_quantize(x,4);
    auto loss=zyron::autograd::sum(q);
    loss.backward();
    check(x.has_grad(),"fake quant gradient missing");
    for(std::size_t i=0;i<x.grad().size();++i) check(std::fabs(x.grad()[i]-1.0f)<1e-6f,"fake quant STE gradient mismatch");
}

void test_qat_model(){
    auto config=zyron::transformer::tiny(32); config.hidden_size=16; config.num_layers=1; config.num_heads=4; config.intermediate_size=32; config.max_sequence_length=16; config.use_qat=true; config.qat_bits=4; config.validate();
    zyron::model::LanguageModel model(config,13); model.train();
    const std::vector<std::vector<std::size_t>> input{{1,2,3,4,5}}; const auto targets=input;
    auto loss=model.loss(input,targets); loss.backward();
    bool any_grad=false; for(auto* p:model.parameters()) any_grad=any_grad||p->has_grad(); check(any_grad,"QAT produced no gradients");
}

void test_batch_generation(){
    auto config=zyron::transformer::tiny(32); config.hidden_size=16; config.num_layers=1; config.num_heads=4; config.intermediate_size=32; config.max_sequence_length=12;
    zyron::model::LanguageModel model(config,21); DummyTokenizer tokenizer;
    zyron::model::GenerationConfig gc; gc.max_new_tokens=2; gc.temperature=1.0f; gc.top_k=3; gc.seed=99;
    const std::vector<std::vector<std::size_t>> prompts{{1,2,3},{4,5,6}};
    const auto out=zyron::model::generate_batch(model,tokenizer,prompts,gc);
    check(out.size()==2,"batch generation count"); check(out[0].size()==5 && out[1].size()==5,"batch generation length");
}

void test_quantized_checkpoint(){
    auto config=zyron::transformer::tiny(32); config.hidden_size=16; config.num_layers=1; config.num_heads=4; config.intermediate_size=32; config.max_sequence_length=16;
    zyron::model::LanguageModel model(config,31); zyron::model::QuantizedLanguageModel q(model,zyron::quantization::Type::Int8);
    const auto path=(std::filesystem::temp_directory_path()/"zyron_final_q.zyqmodel").string(); q.save(path);
    auto loaded=zyron::model::QuantizedLanguageModel::load(path);
    const auto a=q.forward(std::vector<std::size_t>{1,2,3,4}); const auto b=loaded.forward(std::vector<std::size_t>{1,2,3,4}); compare(a,b,0.0f,"quantized checkpoint mismatch");
    check(loaded.quantized_bytes()==q.quantized_bytes(),"quantized size mismatch"); std::filesystem::remove(path);
}

void test_quantized_batch_generation(){
    auto config=zyron::transformer::tiny(32); config.hidden_size=16; config.num_layers=1; config.num_heads=4; config.intermediate_size=32; config.max_sequence_length=12;
    zyron::model::LanguageModel model(config,41); zyron::model::QuantizedLanguageModel q(model,zyron::quantization::Type::Int8);
    zyron::model::GenerationConfig gc; gc.max_new_tokens=2; gc.temperature=1.0f; gc.top_k=3; gc.seed=5;
    const auto out=q.generate_batch({{1,2,3},{4,5,6}},gc); check(out.size()==2,"quantized batch count"); check(out[0].size()==5 && out[1].size()==5,"quantized batch length");
}
}

int main(){ try{ test_fake_quant_ste(); test_qat_model(); test_batch_generation(); test_quantized_checkpoint(); test_quantized_batch_generation(); std::cout<<"ZYRON final tests: PASS\n"; return 0; } catch(const std::exception& e){ std::cerr<<"ZYRON final tests: FAIL: "<<e.what()<<'\n'; return 1; } }
