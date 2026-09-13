#include "zyron/tokenizer/bpe.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace zyron::tokenizer {

namespace {

constexpr std::uint32_t kMagic = 0x4e42595a; // "ZYBN"
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kMaxMergeTokenBytes = 64;


} // namespace

std::size_t BPE::PairHash::operator()(PairKey key) const noexcept {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return static_cast<std::size_t>(key);
}

BPE::PairKey BPE::pair_key(TokenId left, TokenId right) noexcept {
    return (static_cast<PairKey>(left) << 32) |
           static_cast<PairKey>(right);
}

void BPE::reset(const SpecialTokens& specials) {
    specials_ = specials;
    if (specials_.pad.empty() || specials_.unk.empty() ||
        specials_.bos.empty() || specials_.eos.empty()) {
        throw std::invalid_argument(
            "ZYRON BPE: special tokens must not be empty");
    }
    vocabulary_.clear();
    token_to_id_.clear();
    merges_.clear();
    merge_map_.clear();

    vocabulary_.reserve(260);
    token_to_id_.reserve(512);

    for (std::size_t i = 0; i < 256; ++i) {
        std::string value(1, static_cast<char>(i));
        vocabulary_.push_back(value);
        token_to_id_.emplace(value, static_cast<TokenId>(i));
    }

    const std::array<std::string, 4> special_values{
        specials_.pad,
        specials_.unk,
        specials_.bos,
        specials_.eos};

    std::unordered_map<std::string, bool> seen_specials;
    seen_specials.reserve(4);
    for (const auto& value : special_values) {
        if (token_to_id_.find(value) != token_to_id_.end() ||
            seen_specials.find(value) != seen_specials.end()) {
            throw std::invalid_argument(
                "ZYRON BPE: special token collision");
        }
        seen_specials.emplace(value, true);
        const auto id = static_cast<TokenId>(vocabulary_.size());
        vocabulary_.push_back(value);
        token_to_id_.emplace(value, id);
    }

    pad_id_ = 256;
    unk_id_ = 257;
    bos_id_ = 258;
    eos_id_ = 259;
}

std::vector<BPE::TokenId> BPE::byte_ids(const std::string& text) {
    std::vector<TokenId> ids;
    ids.reserve(text.size());
    for (const unsigned char byte : text) {
        ids.push_back(static_cast<TokenId>(byte));
    }
    return ids;
}

void BPE::apply_merge_to_sequence(
    std::vector<TokenId>& sequence,
    TokenId left,
    TokenId right,
    TokenId result) const {

    if (sequence.size() < 2) return;

    std::vector<TokenId> merged;
    merged.reserve(sequence.size());

    for (std::size_t i = 0; i < sequence.size();) {
        if (i + 1 < sequence.size() &&
            sequence[i] == left &&
            sequence[i + 1] == right) {
            merged.push_back(result);
            i += 2;
        } else {
            merged.push_back(sequence[i]);
            ++i;
        }
    }

    sequence.swap(merged);
}

void BPE::learn_from_sequences(
    std::vector<std::vector<TokenId>>& sequences,
    std::size_t target_vocab_size) {

    while (vocabulary_.size() < target_vocab_size) {
        std::unordered_map<PairKey, std::size_t, PairHash> counts;
        counts.reserve(4096);

        for (const auto& sequence : sequences) {
            if (sequence.size() < 2) continue;
            for (std::size_t i = 1; i < sequence.size(); ++i) {
                const auto key = pair_key(sequence[i - 1], sequence[i]);
                ++counts[key];
            }
        }

        if (counts.empty()) break;

        PairKey best_key = 0;
        std::size_t best_count = 0;
        bool found = false;

        for (const auto& [key, count] : counts) {
            const TokenId left_candidate = static_cast<TokenId>(key >> 32);
            const TokenId right_candidate = static_cast<TokenId>(key & 0xffffffffu);
            if (left_candidate >= vocabulary_.size() || right_candidate >= vocabulary_.size()) continue;
            if (vocabulary_[left_candidate].size() + vocabulary_[right_candidate].size() > kMaxMergeTokenBytes) continue;
            if (!found || count > best_count || (count == best_count && key < best_key)) {
                best_key = key;
                best_count = count;
                found = true;
            }
        }

        if (!found || best_count == 0) break;

        const TokenId left = static_cast<TokenId>(best_key >> 32);
        const TokenId right = static_cast<TokenId>(best_key & 0xffffffffu);
        const TokenId result = static_cast<TokenId>(vocabulary_.size());

        if (merge_map_.find(best_key) != merge_map_.end()) {
            break;
        }

        vocabulary_.push_back(
            vocabulary_.at(left) + vocabulary_.at(right));
        merges_.push_back(Merge{left, right, result});
        merge_map_.emplace(best_key, result);
        token_to_id_[vocabulary_.back()] = result;

        for (auto& sequence : sequences) {
            apply_merge_to_sequence(sequence, left, right, result);
        }
    }
}

void BPE::train_text(
    const std::string& text,
    std::size_t target_vocab_size,
    const SpecialTokens& specials) {

    if (target_vocab_size < 260) {
        throw std::invalid_argument(
            "ZYRON BPE: target vocabulary must be at least 260");
    }

    reset(specials);
    if (target_vocab_size == vocabulary_.size() || text.empty()) return;

    std::vector<std::vector<TokenId>> sequences;
    sequences.push_back(byte_ids(text));
    learn_from_sequences(sequences, target_vocab_size);
    rebuild_maps();
}

void BPE::train_files(
    const std::vector<std::string>& paths,
    std::size_t target_vocab_size,
    const SpecialTokens& specials,
    std::size_t chunk_bytes) {

    if (paths.empty()) {
        throw std::invalid_argument("ZYRON BPE: no training files");
    }
    if (target_vocab_size < 260) {
        throw std::invalid_argument(
            "ZYRON BPE: target vocabulary must be at least 260");
    }
    if (chunk_bytes == 0) {
        throw std::invalid_argument(
            "ZYRON BPE: chunk_bytes must be > 0");
    }

    reset(specials);

    // Memory-conscious baseline: tokenize one bounded chunk at a time for each
    // merge pass. This keeps peak memory bounded but intentionally favors
    // correctness and simplicity over training throughput.
    while (vocabulary_.size() < target_vocab_size) {
        std::unordered_map<PairKey, std::size_t, PairHash> counts;
        counts.reserve(4096);

        bool any_data = false;
        for (const auto& path : paths) {
            std::ifstream input(path, std::ios::binary);
            if (!input) {
                throw std::runtime_error(
                    "ZYRON BPE: failed to open training file: " + path);
            }

            while (input) {
                std::string chunk = read_chunk(input, chunk_bytes);
                if (chunk.empty()) break;
                any_data = true;

                auto sequence = encode_bytes(chunk);
                for (std::size_t i = 1; i < sequence.size(); ++i) {
                    ++counts[pair_key(sequence[i - 1], sequence[i])];
                }
            }
        }

        if (!any_data || counts.empty()) break;

        PairKey best_key = 0;
        std::size_t best_count = 0;
        bool found = false;
        for (const auto& [key, count] : counts) {
            const TokenId left_candidate = static_cast<TokenId>(key >> 32);
            const TokenId right_candidate = static_cast<TokenId>(key & 0xffffffffu);
            if (left_candidate >= vocabulary_.size() || right_candidate >= vocabulary_.size()) continue;
            if (vocabulary_[left_candidate].size() + vocabulary_[right_candidate].size() > kMaxMergeTokenBytes) continue;
            if (!found || count > best_count || (count == best_count && key < best_key)) {
                best_key = key;
                best_count = count;
                found = true;
            }
        }

        if (!found || best_count == 0) break;

        const TokenId left = static_cast<TokenId>(best_key >> 32);
        const TokenId right = static_cast<TokenId>(best_key & 0xffffffffu);
        const TokenId result = static_cast<TokenId>(vocabulary_.size());

        vocabulary_.push_back(
            vocabulary_.at(left) + vocabulary_.at(right));
        merges_.push_back(Merge{left, right, result});
        merge_map_[best_key] = result;
        token_to_id_[vocabulary_.back()] = result;
    }

    rebuild_maps();
}

std::string BPE::read_chunk(std::istream& in, std::size_t chunk_bytes) {
    std::string chunk(chunk_bytes, '\0');
    in.read(chunk.data(), static_cast<std::streamsize>(chunk_bytes));
    chunk.resize(static_cast<std::size_t>(in.gcount()));
    return chunk;
}

std::vector<BPE::TokenId> BPE::encode_bytes(
    std::string_view bytes) const {

    std::vector<TokenId> tokens;
    tokens.reserve(bytes.size());

    for (const unsigned char byte : bytes) {
        tokens.push_back(static_cast<TokenId>(byte));
    }

    for (const auto& merge : merges_) {
        apply_merge_to_sequence(
            tokens,
            merge.left,
            merge.right,
            merge.result);
    }

    return tokens;
}

std::size_t BPE::find_special_at(
    std::string_view text,
    std::size_t pos,
    std::size_t& length) const {

    struct Candidate {
        const std::string* value;
        TokenId id;
    };

    const std::array<Candidate, 4> candidates{
        Candidate{&specials_.pad, pad_id_},
        Candidate{&specials_.unk, unk_id_},
        Candidate{&specials_.bos, bos_id_},
        Candidate{&specials_.eos, eos_id_}};

    std::size_t best_id = std::numeric_limits<std::size_t>::max();
    length = 0;

    for (const auto& candidate : candidates) {
        if (candidate.value->empty() ||
            pos + candidate.value->size() > text.size()) {
            continue;
        }

        if (text.substr(pos, candidate.value->size()) == *candidate.value &&
            candidate.value->size() > length) {
            length = candidate.value->size();
            best_id = candidate.id;
        }
    }

    return best_id;
}

std::vector<std::size_t> BPE::encode(
    const std::string& text,
    bool add_bos,
    bool add_eos) const {

    if (vocabulary_.size() < 260) {
        throw std::logic_error("ZYRON BPE: tokenizer is not initialized");
    }

    std::vector<std::size_t> result;
    result.reserve(text.size() + static_cast<std::size_t>(add_bos) +
                   static_cast<std::size_t>(add_eos));

    if (add_bos) result.push_back(bos_id_);

    std::string ordinary;
    ordinary.reserve(text.size());

    const auto flush_ordinary = [&]() {
        if (ordinary.empty()) return;
        const auto tokens = encode_bytes(ordinary);
        result.insert(result.end(), tokens.begin(), tokens.end());
        ordinary.clear();
    };

    for (std::size_t pos = 0; pos < text.size();) {
        std::size_t special_length = 0;
        const std::size_t special_id =
            find_special_at(text, pos, special_length);

        if (special_length != 0) {
            flush_ordinary();
            result.push_back(special_id);
            pos += special_length;
        } else {
            ordinary.push_back(text[pos]);
            ++pos;
        }
    }

    flush_ordinary();

    if (add_eos) result.push_back(eos_id_);
    return result;
}

std::string BPE::decode(
    const std::vector<std::size_t>& ids,
    bool skip_special_tokens) const {

    std::string result;

    for (const auto id : ids) {
        if (id >= vocabulary_.size()) {
            throw std::out_of_range("ZYRON BPE: token id out of range");
        }

        const bool is_special =
            id == pad_id_ || id == unk_id_ ||
            id == bos_id_ || id == eos_id_;

        if (is_special && skip_special_tokens) continue;
        result += vocabulary_[id];
    }

    return result;
}

std::size_t BPE::token_id(std::string_view token) const {
    const auto it = token_to_id_.find(std::string(token));
    if (it == token_to_id_.end()) {
        throw std::out_of_range("ZYRON BPE: token not found");
    }
    return it->second;
}

const std::string& BPE::token(std::size_t id) const {
    if (id >= vocabulary_.size()) {
        throw std::out_of_range("ZYRON BPE: token id out of range");
    }
    return vocabulary_[id];
}

void BPE::rebuild_maps() {
    token_to_id_.clear();
    token_to_id_.reserve(vocabulary_.size() * 2 + 1);
    for (std::size_t id = 0; id < vocabulary_.size(); ++id) {
        token_to_id_[vocabulary_[id]] = static_cast<TokenId>(id);
    }

    merge_map_.clear();
    merge_map_.reserve(merges_.size() * 2 + 1);
    for (const auto& merge : merges_) {
        merge_map_[pair_key(merge.left, merge.right)] = merge.result;
    }
}

void BPE::save(const std::string& path) const {
    if (vocabulary_.size() < 260) {
        throw std::logic_error("ZYRON BPE: tokenizer is not initialized");
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("ZYRON BPE: failed to open save path: " + path);
    }
    save(out);
}

void BPE::save(std::ostream& out) const {
    if (vocabulary_.size() < 260) {
        throw std::logic_error("ZYRON BPE: tokenizer is not initialized");
    }
    const auto write_u32 = [&out](std::uint32_t value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    const auto write_string = [&out, &write_u32](const std::string& value) {
        if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("ZYRON BPE: string too large to serialize");
        }
        write_u32(static_cast<std::uint32_t>(value.size()));
        out.write(value.data(), static_cast<std::streamsize>(value.size()));
    };
    if (vocabulary_.size() > std::numeric_limits<std::uint32_t>::max() ||
        merges_.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("ZYRON BPE: tokenizer too large to serialize");
    }
    write_u32(kMagic);
    write_u32(kVersion);
    write_u32(static_cast<std::uint32_t>(vocabulary_.size()));
    write_u32(static_cast<std::uint32_t>(merges_.size()));
    write_string(specials_.pad);
    write_string(specials_.unk);
    write_string(specials_.bos);
    write_string(specials_.eos);
    for (const auto& value : vocabulary_) write_string(value);
    for (const auto& merge : merges_) {
        write_u32(merge.left);
        write_u32(merge.right);
        write_u32(merge.result);
    }
    if (!out) throw std::runtime_error("ZYRON BPE: tokenizer save failed");
}

void BPE::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("ZYRON BPE: failed to open tokenizer: " + path);
    }
    load(in);
}

void BPE::load(std::istream& in) {
    const auto read_u32 = [&in]() -> std::uint32_t {
        std::uint32_t value = 0;
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!in) throw std::runtime_error("ZYRON BPE: corrupted tokenizer file");
        return value;
    };
    const auto read_string = [&in, &read_u32]() {
        const std::uint32_t length = read_u32();
        std::string value(length, '\0');
        if (length != 0) {
            in.read(value.data(), static_cast<std::streamsize>(length));
            if (!in) throw std::runtime_error("ZYRON BPE: corrupted tokenizer string");
        }
        return value;
    };
    if (read_u32() != kMagic || read_u32() != kVersion) {
        throw std::runtime_error("ZYRON BPE: unsupported tokenizer format");
    }
    const auto vocab_size = read_u32();
    const auto merge_count = read_u32();
    if (vocab_size < 260) throw std::runtime_error("ZYRON BPE: invalid vocabulary size");
    SpecialTokens specials;
    specials.pad = read_string();
    specials.unk = read_string();
    specials.bos = read_string();
    specials.eos = read_string();
    reset(specials);
    vocabulary_.clear();
    vocabulary_.reserve(vocab_size);
    for (std::uint32_t i=0;i<vocab_size;++i) vocabulary_.push_back(read_string());
    merges_.clear();
    merges_.reserve(merge_count);
    for (std::uint32_t i=0;i<merge_count;++i) {
        merges_.push_back(Merge{read_u32(),read_u32(),read_u32()});
    }
    if (!in) throw std::runtime_error("ZYRON BPE: tokenizer load failed");
    pad_id_=256; unk_id_=257; bos_id_=258; eos_id_=259;
    rebuild_maps();
}

} // namespace zyron::tokenizer
