#pragma once
#include <atomic>

#include "core/types.h"

namespace mv {

// The fast mode's guest clock (D16): guest time is the work the virtual CPU did, never the host's wall clock. It
// advances while the vCPU executes guest code (the vCPU thread's CPU time inside KVM_RUN) and through short exits
// (the device accesses the guest makes as it runs), `scale` times faster than the host time they took, because a
// host core is many times a 733 MHz Pentium III. A long exit (the GPU's pusher and renderer, a paused frame, the
// thread descheduled outside the guest) and the time the host runs something else while the vCPU is inside the
// guest cost the guest nothing. So host load slows the guest down as a whole instead of stretching its CPU against
// its own timers.
//
// The guest's time stamp counter is the hypervisor's: it runs `scale` times 733.333 MHz on the host clock, in step
// with guest time while the guest runs and through short exits. A long exit would leave it ahead, so each one
// rewinds it by the time spent outside: the guest did not run in between and cannot see the step.
//
// Arithmetic only: the caller reads the clocks (the thread's CPU time, the steady wall clock, in ns, always the
// wall clock first at a boundary) and applies the rewinds, so the bookkeeping is testable without a hypervisor.
// Guest time counts the vCPU thread's CPU time end to end, every stretch once, except the long exits'; a long exit
// is classified by its wall time and rewound by it. So the TSC leads the clock by exactly the time the thread was
// preempted outside a long exit, never less. Guest time is kept in picoseconds, exact for every scale in
// thousandths, and read as CPU cycles at 733.333 MHz.
//
// Exact exits (D33, `set_exact_exits`): a host whose CPU flushes its branch predictors on every VM exit (the IBPB
// that the SRSO and VMSCAPE mitigations put there) spends ~4.5 us on each exit inside KVM_RUN plus ~1.7 us outside,
// and a title that writes the APU voice by voice made 800 exits a frame, twelve times its real CPU work. With
// exact exits no exit costs guest time: every stretch outside is excluded and rewound, the exit's own cost inside
// KVM_RUN (`left`'s overhead, below the host's measured floor) comes off the guest time and the TSC, and a device
// write costs the guest what the console's bus would (`classify`'s device cycles). A device read keeps the host's
// price: guests poll by reading, and a poll at the bus price takes many times the host time per console second.
class GuestClock {
public:
    // A stretch outside the guest shorter than this counts as guest time: it is the cost of a device access, the
    // TSC ran through it, and a rewind takes a hypervisor call as long as the stretch itself.
    static constexpr u64 kShortExitNs = 5'000;
    static constexpr u32 kDefaultScaleMilli = 2'000;

    explicit GuestClock(u32 scale_milli = kDefaultScaleMilli) { set_scale_milli(scale_milli); }
    void set_scale_milli(u32 scale_milli);
    u32 scale_milli() const { return scale_milli_; }
    // The TSC's rate on the host clock in kHz: 733,333 kHz of guest time, `scale` times faster.
    u32 tsc_khz() const;

    // The vCPU thread around each KVM_RUN, in this order:
    //   classify(wall) → [excluded: excluded_until(cpu); apply the rewind] → entered(cpu, wall) → KVM_RUN → left(cpu, wall)
    // classify may be called again before entered (the caller looped); entered consumes the stretch.
    struct Exit {
        bool long_exit{false};  // longer than kShortExitNs (the io thread re-plans on the running clock)
        bool excluded{false};   // not guest time: excluded_until(cpu), then the rewind
        u64 tsc_rewind{0};      // guest TSC cycles to take off before entering
    };
    // `device_cycles`: what the exit's device access costs the guest (exact exits only). `counted`: the exit counts at
    // the host's price even with exact exits (a device read: guests poll by reading, and a poll at the bus's price
    // took 30 times the host time per console millisecond: a frame limiter reading the PM timer).
    Exit classify(u64 wall_ns, Cycles device_cycles = 0, bool counted = false);
    void set_exact_exits(bool on) { exact_ = on; }
    bool exact_exits() const { return exact_; }
    // An excluded stretch: its CPU time up to here is not guest time. Returns the TSC rewind of a short one (exact
    // exits), to add to classify's.
    u64 excluded_until(u64 cpu_ns);
    void entered(u64 cpu_ns, u64 wall_ns);
    // `overhead_cap_ns`: the exit's cost inside KVM_RUN, taken off the run (exact exits; at most the run).
    void left(u64 cpu_ns, u64 wall_ns, u64 overhead_cap_ns = 0);
    // Start a stretch the guest did not run in (start-up, a reset): the next entry rewinds the TSC by it. The vCPU
    // thread gives its CPU clock when it starts (`thread_started`); until then the stretch is wall time only.
    void mark_out(u64 wall_ns);
    void thread_started(u64 cpu_ns);
    // HLT with nothing to do: guest time jumps to `target` (never back). Returns the cycles skipped; the TSC must
    // move as far.
    Cycles warp_to(Cycles target);

    // Any thread: guest time in CPU cycles at the last exit (or warp).
    Cycles now() const;
    // Any thread: guest time now, with the vCPU's time in the guest so far when it is inside: `vcpu_cpu_ns` reads
    // the vCPU thread's CPU clock (called only then). Another thread fires device events on this.
    template <class ReadCpu>
    Cycles estimate(ReadCpu&& vcpu_cpu_ns) const {
        for (;;) {
            const u64 s1 = seq_.load(std::memory_order_acquire);
            if (s1 & 1) continue;  // the vCPU thread is writing
            u64 ps = guest_ps_.load(std::memory_order_relaxed);
            const u64 since = pub_entered_cpu_ns_.load(std::memory_order_relaxed);
            const bool inside = pub_inside_.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) != s1) continue;
            if (inside) {
                const u64 cpu = vcpu_cpu_ns();
                if (cpu > since) ps += (cpu - since) * scale_milli_;
            }
            return cycles_from_ps(ps);
        }
    }

    struct Stats {
        u64 in_guest_ns{0};     // vCPU CPU time inside KVM_RUN
        u64 out_counted_ns{0};  // time outside the guest that counted: short exits whole, long ones after classification
        u64 short_exits{0};
        u64 long_ns{0};         // long exits up to their classification, not counted
        u64 long_exits{0};
        u64 longest_exit_ns{0};
        u64 preempted_ns{0};    // inside KVM_RUN on the wall clock but not on the CPU: the host ran something else
        u64 warped_cycles{0};
        u64 short_excess_ns{0};  // short exits' wall time beyond their CPU time: the TSC ran through it, guest time did not
        u64 overhead_ns{0};       // exact exits: the exits' cost inside KVM_RUN, not guest time
        u64 excluded_short_ns{0}; // exact exits: short stretches outside, not guest time
        u64 device_cycles{0};     // exact exits: device accesses at the console's cost
    };
    const Stats& stats() const { return stats_; }

private:
    void add_host_ns(u64 ns);
    static Cycles cycles_from_ps(u64 ps);
    u64 rewind_for(u64 stretch_ns) const;
    // The vCPU thread publishes its bracket for `estimate` under a sequence count (odd while writing).
    void publish(bool inside);
    std::atomic<u64> seq_{0};
    std::atomic<u64> pub_entered_cpu_ns_{0};
    std::atomic<bool> pub_inside_{false};

    u32 scale_milli_{kDefaultScaleMilli};
    std::atomic<u64> guest_ps_{0};
    u64 out_since_ns_{0};         // wall time of the last exit (or the start of a stretch the guest did not run in)
    u64 out_since_cpu_{0};        // the thread's CPU time then
    u64 classified_at_ns_{0};     // wall time of the classification of the pending stretch
    u64 excluded_to_cpu_{0};      // a long stretch's CPU time is not counted up to here
    bool pending_long_{false};
    bool pending_excluded_{false};
    bool exact_{false};
    u64 overhead_pending_ns_{0};  // taken off the last run, rewound at the next entry
    Cycles device_pending_{0};    // the pending exit's device cost, added at the entry
    u64 entered_cpu_ns_{0}, entered_wall_ns_{0};
    Stats stats_;
};

}  // namespace mv
