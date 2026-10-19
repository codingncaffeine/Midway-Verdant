#include "core/guest_clock.h"

#include <algorithm>

#include "core/clock.h"

namespace mv {

namespace {
// Guest cycles per picosecond: 733.333 MHz = 2200/3 MHz = 11/15000 cycles per ps.
constexpr u64 kCyclesPerPsNum = 11, kCyclesPerPsDen = 15'000;
}  // namespace

void GuestClock::set_scale_milli(u32 scale_milli) { scale_milli_ = std::max<u32>(scale_milli, 1); }

u32 GuestClock::tsc_khz() const { return static_cast<u32>(muldiv64(clk::kCpuHzNum / 1000, scale_milli_, clk::kCpuHzDen * 1000)); }

void GuestClock::add_host_ns(u64 ns) {
    // ns of host time × scale = ns × scale_milli ps.
    guest_ps_.store(guest_ps_.load(std::memory_order_relaxed) + ns * scale_milli_, std::memory_order_relaxed);
}

Cycles GuestClock::cycles_from_ps(u64 ps) { return muldiv64(ps, kCyclesPerPsNum, kCyclesPerPsDen); }

void GuestClock::publish(bool inside) {
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    pub_entered_cpu_ns_.store(entered_cpu_ns_, std::memory_order_relaxed);
    pub_inside_.store(inside, std::memory_order_relaxed);
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}

GuestClock::Exit GuestClock::classify(u64 wall_ns) {
    const u64 stretch = wall_ns > out_since_ns_ ? wall_ns - out_since_ns_ : 0;
    classified_at_ns_ = wall_ns;
    pending_long_ = stretch > kShortExitNs;
    excluded_to_cpu_ = out_since_cpu_;
    if (!pending_long_) return {};
    // The guest TSC ran through the whole stretch at tsc_khz; take off no more than that (floor), so it never steps
    // below a value the guest read before the exit.
    return {true, muldiv64(stretch, tsc_khz(), 1'000'000)};
}

void GuestClock::excluded_until(u64 cpu_ns) { excluded_to_cpu_ = std::max(cpu_ns, out_since_cpu_); }

void GuestClock::entered(u64 cpu_ns, u64 wall_ns) {
    // Counted: the thread's CPU time since the exit, but for a long stretch only what came after its exclusion
    // (the rewind and the entry, which the TSC runs through as well).
    const u64 from = pending_long_ ? excluded_to_cpu_ : out_since_cpu_;
    const u64 counted = cpu_ns > from ? cpu_ns - from : 0;
    if (pending_long_) {
        const u64 stretch = classified_at_ns_ > out_since_ns_ ? classified_at_ns_ - out_since_ns_ : 0;
        stats_.long_ns += stretch;
        ++stats_.long_exits;
        stats_.longest_exit_ns = std::max(stats_.longest_exit_ns, stretch);
    } else {
        ++stats_.short_exits;
        const u64 stretch = wall_ns > out_since_ns_ ? wall_ns - out_since_ns_ : 0;
        if (stretch > counted) stats_.short_excess_ns += stretch - counted;
    }
    stats_.out_counted_ns += counted;
    pending_long_ = false;
    entered_cpu_ns_ = cpu_ns;
    entered_wall_ns_ = wall_ns;
    // The count and the bracket's start change together for `estimate`.
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    add_host_ns(counted);
    pub_entered_cpu_ns_.store(entered_cpu_ns_, std::memory_order_relaxed);
    pub_inside_.store(true, std::memory_order_relaxed);
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}

void GuestClock::left(u64 cpu_ns, u64 wall_ns) {
    const u64 cpu = cpu_ns > entered_cpu_ns_ ? cpu_ns - entered_cpu_ns_ : 0;
    const u64 wall = wall_ns > entered_wall_ns_ ? wall_ns - entered_wall_ns_ : 0;
    stats_.in_guest_ns += cpu;
    if (wall > cpu) stats_.preempted_ns += wall - cpu;
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    add_host_ns(cpu);
    pub_inside_.store(false, std::memory_order_relaxed);
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    out_since_ns_ = wall_ns;
    out_since_cpu_ = cpu_ns;
}

void GuestClock::mark_out(u64 wall_ns) {
    out_since_ns_ = wall_ns;
    pending_long_ = false;
    publish(false);
}

void GuestClock::thread_started(u64 cpu_ns) {
    out_since_cpu_ = cpu_ns;
    excluded_to_cpu_ = cpu_ns;
}

Cycles GuestClock::warp_to(Cycles target) {
    const Cycles before = now();
    if (target <= before) return 0;
    const u64 ps = muldiv64_ceil(target, kCyclesPerPsDen, kCyclesPerPsNum);
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    guest_ps_.store(std::max(ps, guest_ps_.load(std::memory_order_relaxed)), std::memory_order_relaxed);
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    const Cycles skipped = now() - before;
    stats_.warped_cycles += skipped;
    return skipped;
}

Cycles GuestClock::now() const { return cycles_from_ps(guest_ps_.load(std::memory_order_relaxed)); }

}  // namespace mv
