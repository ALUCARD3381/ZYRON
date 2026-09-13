#pragma once

#include <cstddef>
#include <functional>

namespace zyron::runtime {

struct CpuFeatures {
    bool armv7{false};
    bool armv8{false};
    bool neon{false};
    bool sse2{false};
};

// Conservative by default: one worker avoids oversubscription on mobile CPUs.
void set_num_threads(std::size_t threads);
[[nodiscard]] std::size_t num_threads() noexcept;
[[nodiscard]] std::size_t recommended_threads() noexcept;
[[nodiscard]] CpuFeatures cpu_features() noexcept;
[[nodiscard]] std::size_t process_rss_bytes() noexcept;

// Coarse-grained parallel loop used by compute kernels. The callback receives
// [begin, end) and must be thread-safe for independent ranges.
void parallel_for(
    std::size_t begin,
    std::size_t end,
    const std::function<void(std::size_t, std::size_t)>& fn,
    std::size_t min_grain = 1);

} // namespace zyron::runtime
