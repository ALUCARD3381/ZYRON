#pragma once

#include "zyron/tokenizer/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace zyron::tokenizer {

struct SpecialTokens {
    std::string pad = "<PAD>";
    std::string unk = "<UNK>";
    std::string bos = "<BOS>";
    std::string eos = "<EOS>";
};

class BPE final : public Tokenizer {
public:
    BPE() = default;

    void train_text(
        const std::string& text,
        std::size_t target_vocab_size,
        const SpecialTokens& specials = {});

    void train_files(
        const std::vector<std::string>& paths,
        std::size_t target_vocab_size,
        const SpecialTokens& specials = {},
        std::size_t chunk_bytes = 1u << 20);

    [[nodiscard]] std::vector<std::size_t> encode(
        const std::string& text,
        bool add_bos = false,
        bool add_eos = false) const override;

    [[nodiscard]] std::string decode(
        const std::vector<std::size_t>& ids,
        bool skip_special_tokens = true) const override;

    [[nodiscard]] std::size_t vocab_size() const noexcept override {
        return vocabulary_.size();
    }

    [[nodiscard]] std::size_t token_id(std::string_view token) const;
    [[nodiscard]] const std::string& token(std::size_t id) const;

    [[nodiscard]] const SpecialTokens& special_tokens() const noexcept {
        return specials_;
    }

    [[nodiscard]] std::size_t bos_id() const noexcept { return bos_id_; }
    [[nodiscard]] bool has_bos() const noexcept override { return true; }
    [[nodiscard]] std::size_t eos_id() const noexcept { return eos_id_; }
    [[nodiscard]] bool has_eos() const noexcept override { return true; }
    [[nodiscard]] std::size_t pad_id() const noexcept { return pad_id_; }
    [[nodiscard]] std::size_t unk_id() const noexcept { return unk_id_; }

    void save(const std::string& path) const;
    void load(const std::string& path);
    void save(std::ostream& out) const;
    void load(std::istream& in);

private:
    using TokenId = std::uint32_t;
    using PairKey = std::uint64_t;

    struct Merge {
        TokenId left;
        TokenId right;
        TokenId result;
    };

    struct PairHash {
        std::size_t operator()(PairKey key) const noexcept;
    };

    static PairKey pair_key(TokenId left, TokenId right) noexcept;
    static std::vector<TokenId> byte_ids(const std::string& text);
    static std::string read_chunk(std::istream& in, std::size_t chunk_bytes);

    void reset(const SpecialTokens& specials);
    void rebuild_maps();
    void learn_from_sequences(
        std::vector<std::vector<TokenId>>& sequences,
        std::size_t target_vocab_size);
    void apply_merge_to_sequence(
        std::vector<TokenId>& sequence,
        TokenId left,
        TokenId right,
        TokenId result) const;

    [[nodiscard]] std::vector<TokenId> encode_bytes(
        std::string_view bytes) const;
    [[nodiscard]] std::size_t find_special_at(
        std::string_view text,
        std::size_t pos,
        std::size_t& length) const;

    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::vector<Merge> merges_;
    std::unordered_map<PairKey, TokenId, PairHash> merge_map_;
    SpecialTokens specials_{};
    TokenId pad_id_{256};
    TokenId unk_id_{257};
    TokenId bos_id_{258};
    TokenId eos_id_{259};
};

} // namespace zyron::tokenizer
