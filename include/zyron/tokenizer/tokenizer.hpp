#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace zyron::tokenizer {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    [[nodiscard]] virtual std::vector<std::size_t> encode(
        const std::string& text,
        bool add_bos = false,
        bool add_eos = false) const = 0;

    [[nodiscard]] virtual std::string decode(
        const std::vector<std::size_t>& ids,
        bool skip_special_tokens = true) const = 0;

    [[nodiscard]] virtual std::size_t vocab_size() const noexcept = 0;

    [[nodiscard]] virtual bool has_bos() const noexcept { return false; }
    [[nodiscard]] virtual std::size_t bos_id() const {
        throw std::logic_error("ZYRON Tokenizer: BOS token is not available");
    }

    [[nodiscard]] virtual bool has_eos() const noexcept { return false; }

    [[nodiscard]] virtual std::size_t eos_id() const {
        throw std::logic_error("ZYRON Tokenizer: EOS token is not available");
    }
};

} // namespace zyron::tokenizer
