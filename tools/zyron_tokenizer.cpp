#include "zyron/tokenizer/bpe.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage() {
    std::cerr
        << "Usage: zyron_tokenizer --train --vocab-size N --output FILE DATA...\n";
}

std::string arg_value(
    int argc,
    char** argv,
    int& index,
    const char* name) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
    ++index;
    return argv[index];
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 1;
        }

        bool train = false;
        std::size_t vocab_size = 0;
        std::string output;
        std::vector<std::string> data_files;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--train") {
                train = true;
            } else if (arg == "--vocab-size") {
                vocab_size = std::stoull(arg_value(argc, argv, i, "--vocab-size"));
            } else if (arg == "--output") {
                output = arg_value(argc, argv, i, "--output");
            } else {
                data_files.push_back(arg);
            }
        }

        if (!train || vocab_size < 260 || output.empty() || data_files.empty()) {
            usage();
            return 1;
        }

        zyron::tokenizer::BPE tokenizer;
        tokenizer.train_files(data_files, vocab_size);
        tokenizer.save(output);

        std::cout << "ZYRON tokenizer saved\n"
                  << "Vocabulary: " << tokenizer.vocab_size() << "\n"
                  << "Output: " << output << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "zyron_tokenizer: " << e.what() << '\n';
        return 1;
    }
}
