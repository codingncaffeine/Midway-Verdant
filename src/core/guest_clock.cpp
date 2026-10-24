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

GuestClock::Exit GuestClock::classify(u64 wall_ns, Cycles device_cycles, bool counted) {
    const u64 stretch = wall_ns > out_since_ns_ ? wall_ns - out_since_ns_ : 0;
    classified_at_ns_ = wall_ns;
    pending_long_ = stretch > kShortExitNs;
    pending_excluded_ = pending_long_ || (exact_ && !counted);
    excluded_to_cpu_ = out_since_cpu_;
    device_pending_ = exact_ && !counted ? device_cycles : 0;
    if (!pending_excluded_) return {};
    // A short stretch with exact exits is rewound by its CPU time, in excluded_until: its wall time starts at a clock
    // read made just after the exit, where the host runs slowly (its branch predictors just flushed), so the wall
    // stretch outgrew the CPU time the run had counted and the TSC fell behind the clock by ~0.15 us an exit.
    if (!pending_long_) return {false, true, 0};
    // The guest TSC ran through the whole stretch at tsc_khz, and through the exit's cost inside KVM_RUN taken off
    // the run; take off no more than that (floor), so it never steps below a value the guest read before the exit.
    // The device access moves guest time and the TSC alike.
    return {true, true, rewind_for(stretch)};
}

u64 GuestClock::rewind_for(u64 stretch_ns) const {
    const u64 ran = muldiv64(stretch_ns + overhead_pending_ns_, tsc_khz(), 1'000'000);
    return ran > device_pending_ ? ran - device_pending_ : 0;
}

u64 GuestClock::excluded_until(u64 cpu_ns) {
    excluded_to_cpu_ = std::max(cpu_ns, out_since_cpu_);
    // The CPU time since the exit never exceeds the wall time the guest did not run in: a safe rewind.
    return pending_excluded_ && !pending_long_ ? rewind_for(excluded_to_cpu_ - out_since_cpu_) : 0;
}

void GuestClock::entered(u64 cpu_ns, u64 wall_ns) {
    // Counted: the thread's CPU time since the exit, but for a long stretch only what came after its exclusion
    // (the rewind and the entry, which the TSC runs through as well).
    const u64 from = pending_excluded_ ? excluded_to_cpu_ : out_since_cpu_;
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
        if (pending_excluded_) stats_.excluded_short_ns += classified_at_ns_ > out_since_ns_ ? classified_at_ns_ - out_since_ns_ : 0;
    }
    stats_.out_counted_ns += counted;
    pending_long_ = false;
    if (pending_excluded_) overhead_pending_ns_ = 0;
    pending_excluded_ = false;
    const Cycles device = device_pending_;
    device_pending_ = 0;
    stats_.device_cycles += device;
    entered_cpu_ns_ = cpu_ns;
    entered_wall_ns_ = wall_ns;
    // The count and the bracket's start change together for `estimate`.
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    add_host_ns(counted);
    if (device) guest_ps_.store(guest_ps_.load(std::memory_order_relaxed) + muldiv64_ceil(device, kCyclesPerPsDen, kCyclesPerPsNum), std::memory_order_relaxed);
    pub_entered_cpu_ns_.store(entered_cpu_ns_, std::memory_order_relaxed);
    pub_inside_.store(true, std::memory_order_relaxed);
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}

void GuestClock::left(u64 cpu_ns, u64 wall_ns, u64 overhead_cap_ns) {
    const u64 cpu = cpu_ns > entered_cpu_ns_ ? cpu_ns - entered_cpu_ns_ : 0;
    const u64 overhead = exact_ ? std::min(overhead_cap_ns, cpu) : 0;
    stats_.overhead_ns += overhead;
    overhead_pending_ns_ += overhead;
    const u64 wall = wall_ns > entered_wall_ns_ ? wall_ns - entered_wall_ns_ : 0;
    stats_.in_guest_ns += cpu;
    if (wall > cpu) stats_.preempted_ns += wall - cpu;
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    add_host_ns(cpu - overhead);
    pub_inside_.store(false, std::memory_order_relaxed);
    seq_.store(seq_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    out_since_ns_ = wall_ns;
    out_since_cpu_ = cpu_ns;
}

void GuestClock::mark_out(u64 wall_ns) {
    out_since_ns_ = wall_ns;
    pending_long_ = false;
    pending_excluded_ = false;
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
