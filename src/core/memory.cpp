#include "zyron/core/memory.hpp"

#include <algorithm>
#include <new>

namespace zyron {
namespace {

std::atomic<std::size_t> g_live_bytes{0};
std::atomic<std::size_t> g_peak_bytes{0};
std::atomic<std::size_t> g_live_allocations{0};
std::atomic<std::size_t> g_total_allocations{0};

void update_peak(std::size_t value) noexcept {
    std::size_t current = g_peak_bytes.load(std::memory_order_relaxed);
    while (current < value &&
           !g_peak_bytes.compare_exchange_weak(
               current, value,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {}
}

} // namespace

std::shared_ptr<std::byte[]> Memory::allocate(std::size_t bytes) {
    if (bytes == 0) return {};

    auto* raw = static_cast<std::byte*>(
        ::operator new(bytes, std::align_val_t(kAlignment)));

    g_live_bytes.fetch_add(bytes, std::memory_order_relaxed);
    g_live_allocations.fetch_add(1, std::memory_order_relaxed);
    g_total_allocations.fetch_add(1, std::memory_order_relaxed);
    update_peak(g_live_bytes.load(std::memory_order_relaxed));

    return std::shared_ptr<std::byte[]>(
        raw,
        [bytes](std::byte* ptr) noexcept {
            ::operator delete(ptr, std::align_val_t(kAlignment));
            g_live_bytes.fetch_sub(bytes, std::memory_order_relaxed);
            g_live_allocations.fetch_sub(1, std::memory_order_relaxed);
        });
}

MemoryStats Memory::stats() noexcept {
    return {
        g_live_bytes.load(std::memory_order_relaxed),
        g_peak_bytes.load(std::memory_order_relaxed),
        g_live_allocations.load(std::memory_order_relaxed),
        g_total_allocations.load(std::memory_order_relaxed)
    };
}

void Memory::reset_peak() noexcept {
    g_peak_bytes.store(
        g_live_bytes.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
}

} // namespace zyron
