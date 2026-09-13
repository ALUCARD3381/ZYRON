#include "zyron/core/runtime.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

#if defined(__linux__) && (defined(__arm__) || defined(__aarch64__))
#include <sys/auxv.h>
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
#endif

namespace zyron::runtime {
namespace {

std::atomic<std::size_t> g_threads{1};

} // namespace

void set_num_threads(std::size_t threads) {
    if (threads == 0) threads = 1;
    const std::size_t max_threads = std::max<std::size_t>(1, recommended_threads() * 4);
    g_threads.store(std::min(threads, max_threads), std::memory_order_relaxed);
}

std::size_t num_threads() noexcept {
    return std::max<std::size_t>(1, g_threads.load(std::memory_order_relaxed));
}

std::size_t recommended_threads() noexcept {
    const unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) return 1;
    // Mobile-friendly recommendation: never suggest an unbounded core count.
    return std::min<std::size_t>(static_cast<std::size_t>(hc), 4);
}

CpuFeatures cpu_features() noexcept {
    CpuFeatures features{};
#if defined(__arm__) && !defined(__aarch64__)
    features.armv7 = true;
#endif
#if defined(__aarch64__)
    features.armv8 = true;
#endif
#if defined(__SSE2__)
    features.sse2 = true;
#endif

#if defined(__linux__) && (defined(__arm__) || defined(__aarch64__))
    unsigned long hwcap = 0;
#if defined(AT_HWCAP)
    hwcap = getauxval(AT_HWCAP);
#endif
#if defined(HWCAP_NEON)
    features.neon = (hwcap & HWCAP_NEON) != 0;
#elif defined(__ARM_NEON)
    features.neon = true;
#endif
#elif defined(__ARM_NEON)
    features.neon = true;
#endif
    return features;
}

std::size_t process_rss_bytes() noexcept {
#if defined(__linux__)
    std::ifstream in("/proc/self/status");
    std::string key;
    std::size_t value = 0;
    std::string unit;
    while (in >> key >> value >> unit) {
        if (key == "VmRSS:") return value * 1024u;
    }
#endif
#if defined(__linux__)
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<std::size_t>(usage.ru_maxrss) * 1024u;
    }
#endif
    return 0;
}

void parallel_for(
    std::size_t begin,
    std::size_t end,
    const std::function<void(std::size_t, std::size_t)>& fn,
    std::size_t min_grain) {

    if (end <= begin) return;
    if (!fn) return;

    const std::size_t length = end - begin;
    const std::size_t threads = num_threads();
    if (threads <= 1 || length <= std::max<std::size_t>(1, min_grain)) {
        fn(begin, end);
        return;
    }

    const std::size_t worker_count = std::min(threads, length);
    const std::size_t chunk = std::max(
        std::max<std::size_t>(1, min_grain),
        (length + worker_count - 1) / worker_count);

    std::vector<std::thread> workers;
    workers.reserve(worker_count - 1);

    std::size_t cursor = begin;
    for (std::size_t worker = 0; worker + 1 < worker_count; ++worker) {
        const std::size_t task_begin = cursor;
        const std::size_t task_end = std::min(end, task_begin + chunk);
        cursor = task_end;
        workers.emplace_back([task_begin, task_end, &fn] {
            fn(task_begin, task_end);
        });
    }

    if (cursor < end) fn(cursor, end);
    for (auto& worker : workers) worker.join();
}

} // namespace zyron::runtime
