#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace zyron {

struct MemoryStats {
    std::size_t live_bytes{0};
    std::size_t peak_bytes{0};
    std::size_t live_allocations{0};
    std::size_t total_allocations{0};
};

class Memory {
public:
    static constexpr std::size_t kAlignment = 64;

    Memory() = default;

    explicit Memory(std::size_t bytes)
        : size_(bytes), data_(allocate(bytes)) {}

    [[nodiscard]] std::byte* data() noexcept { return data_.get(); }
    [[nodiscard]] const std::byte* data() const noexcept { return data_.get(); }
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] static MemoryStats stats() noexcept;
    static void reset_peak() noexcept;

private:
    static std::shared_ptr<std::byte[]> allocate(std::size_t bytes);

    std::size_t size_{0};
    std::shared_ptr<std::byte[]> data_{};
};

} // namespace zyron
