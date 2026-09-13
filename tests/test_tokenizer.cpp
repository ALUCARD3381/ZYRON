#include "zyron/tokenizer/bpe.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using zyron::tokenizer::BPE;
using zyron::tokenizer::SpecialTokens;

static void check(bool condition, const char* msg) {
    if (!condition) throw std::runtime_error(msg);
}

static void test_bpe_train_encode_decode() {
    const std::string text =
        "o gato dorme. o gato corre. o gato dorme. "
        "o cachorro corre. o gato dorme.";

    BPE tokenizer;
    tokenizer.train_text(text, 280);

    check(tokenizer.vocab_size() <= 280, "vocab limit");
    check(tokenizer.vocab_size() > 260, "merges learned");

    const auto ids = tokenizer.encode("o gato dorme");
    const auto decoded = tokenizer.decode(ids);
    check(decoded == "o gato dorme", "roundtrip decode");

    const auto special_ids = tokenizer.encode(
        "<BOS>olá<EOS>", false, false);
    check(special_ids.size() >= 3, "special encoding");
    check(special_ids.front() == tokenizer.bos_id(), "BOS id");
    check(special_ids.back() == tokenizer.eos_id(), "EOS id");

    const auto with_flags = tokenizer.encode("olá", true, true);
    check(with_flags.front() == tokenizer.bos_id(), "add BOS");
    check(with_flags.back() == tokenizer.eos_id(), "add EOS");
}

static void test_bpe_save_load() {
    const std::string path = "/tmp/zyron_test_tokenizer.zytok";

    BPE a;
    a.train_text("banana banana bandana banana", 270);
    const auto ids = a.encode("banana bandana");
    const auto decoded = a.decode(ids);
    check(decoded == "banana bandana", "pre-save decode");

    a.save(path);

    BPE b;
    b.load(path);

    check(a.vocab_size() == b.vocab_size(), "loaded vocab size");
    const auto loaded_ids = b.encode("banana bandana");
    check(loaded_ids == ids, "loaded ids");
    check(b.decode(loaded_ids) == decoded, "loaded decode");

    std::remove(path.c_str());
}


static void test_bpe_train_files_chunked() {
    const std::string path = "/tmp/zyron_bpe_train.txt";
    {
        std::ofstream out(path, std::ios::binary);
        out << "abcdefabcdefabcdef\n";
        out << "abcdefabcdefabcdef\n";
    }

    BPE tokenizer;
    tokenizer.train_files({path}, 265, {}, 7);
    const auto ids = tokenizer.encode("abcdef");
    check(tokenizer.decode(ids) == "abcdef", "chunked file roundtrip");

    std::remove(path.c_str());
}

static void test_bpe_custom_specials() {
    SpecialTokens specials;
    specials.pad = "<padx>";
    specials.unk = "<unkx>";
    specials.bos = "<bosx>";
    specials.eos = "<eosx>";

    BPE tokenizer;
    tokenizer.train_text("abc abc abc", 265, specials);

    const auto ids = tokenizer.encode("<bosx>abc<eosx>");
    check(ids.front() == tokenizer.bos_id(), "custom BOS");
    check(ids.back() == tokenizer.eos_id(), "custom EOS");
}

int main() {
    try {
        test_bpe_train_encode_decode();
        test_bpe_save_load();
        test_bpe_train_files_chunked();
        test_bpe_custom_specials();
        std::cout << "ZYRON Phase 3 tokenizer tests: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ZYRON Phase 3 tokenizer tests: FAIL: "
                  << e.what() << '\n';
        return 1;
    }
}
