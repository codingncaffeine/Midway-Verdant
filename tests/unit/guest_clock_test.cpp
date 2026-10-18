#include <gtest/gtest.h>

#include "core/clock.h"
#include "core/guest_clock.h"

using namespace mv;

namespace {
// The hypervisor's guest TSC as the guest clock's caller drives it: it runs at tsc_khz on the host clock, and the
// caller moves it by the rewinds (through the TSC offset).
struct TscModel {
    double offset{0};
    u32 khz{0};
    double at(u64 wall_ns) const { return offset + static_cast<double>(wall_ns) * khz / 1e6; }
};

// The vCPU thread's trips in and out of the guest as the caller drives the clock: the host clock (`wall`) and the
// thread's CPU time (`cpu`), which stands still while the thread is preempted or blocked.
struct Driver {
    GuestClock clock;
    TscModel tsc;
    u64 wall{1'000'000}, cpu{500'000};
    double last_seen{0};  // the largest TSC the guest could have read: its value at the last exit
    explicit Driver(u32 scale_milli) : clock(scale_milli) {
        tsc.khz = clock.tsc_khz();
        tsc.offset = -tsc.at(wall);  // TSC zero now, at guest time zero
        clock.mark_out(wall);
        clock.thread_started(cpu);
    }
    void enter() {
        const GuestClock::Exit e = clock.classify(wall);
        if (e.long_exit) {
            clock.excluded_until(cpu);
            tsc.offset -= static_cast<double>(e.tsc_rewind);
        }
        wall += 300;  // the rewind's own call and the entry, which the TSC runs through
        cpu += 300;
        clock.entered(cpu, wall);
        EXPECT_GE(tsc.at(wall) + 1e-6, last_seen) << "the TSC stepped below a value the guest could have read";
    }
    void run(u64 ns, u64 preempt = 0) {
        wall += ns;
        cpu += ns - preempt;
        clock.left(cpu, wall);
        last_seen = tsc.at(wall);
    }
    // A stretch outside the guest: the thread works through it (an exit handler, the pusher) unless `blocked` ns
    // of it it waited or was preempted.
    void outside(u64 ns, u64 blocked = 0) {
        wall += ns;
        cpu += ns - blocked;
    }
    // TSC minus the guest clock, in cycles.
    double lead() const { return tsc.at(wall) - static_cast<double>(clock.now()); }
};
}  // namespace

TEST(GuestClock, TscRateIsScaledGuestRate) {
    EXPECT_EQ(GuestClock(1000).tsc_khz(), 733'333u);
    EXPECT_EQ(GuestClock(4000).tsc_khz(), 2'933'333u);
    EXPECT_EQ(GuestClock(2500).tsc_khz(), 1'833'333u);
}

TEST(GuestClock, TimeInTheGuestCountsScaledAndPreemptionDoesNot) {
    Driver d(4000);
    d.enter();
    d.run(1'000'000);  // 1 ms of host CPU in the guest = 4 ms of guest time
    EXPECT_EQ(d.clock.now(), cycles_from_ns(4 * (300 + 1'000'000)));
    d.outside(1'000);  // a short exit: counts as well, scaled
    d.enter();
    d.run(1'000'000, 600'000);  // 1 ms in the guest on the wall clock, 0.6 ms of it preempted
    EXPECT_EQ(d.clock.now(), cycles_from_ns(4 * (300 + 1'000'000 + 1'000 + 300 + 400'000)));
    EXPECT_EQ(d.clock.stats().preempted_ns, 600'000u);
    EXPECT_EQ(d.clock.stats().short_exits, 2u);
}

TEST(GuestClock, LongExitIsNotGuestTimeAndComesOffTheTsc) {
    Driver d(4000);
    d.enter();
    d.run(2'000'000);
    const Cycles before = d.clock.now();
    d.outside(3'000'000);  // the pusher: 3 ms outside the guest
    const GuestClock::Exit e = d.clock.classify(d.wall);
    EXPECT_TRUE(e.long_exit);
    EXPECT_EQ(e.tsc_rewind, muldiv64(3'000'000, d.clock.tsc_khz(), 1'000'000));
    d.enter();
    // Only the tail after the classification counted.
    EXPECT_EQ(d.clock.now() - before, cycles_from_ns(4 * 300));
    EXPECT_EQ(d.clock.stats().long_exits, 1u);
    EXPECT_EQ(d.clock.stats().long_ns, 3'000'000u);
    // The TSC agrees with the guest clock again (to the rounding of a cycle or two).
    EXPECT_NEAR(d.lead(), 0.0, 4.0);
}

TEST(GuestClock, ClassifyingTwiceCountsTheStretchOnce) {
    Driver d(2000);
    d.enter();
    d.run(100'000);
    const Cycles before = d.clock.now();
    d.outside(2'000);
    EXPECT_FALSE(d.clock.classify(d.wall).long_exit);
    d.outside(10'000);  // the caller looped (a pacing wait): the same stretch, now long
    d.enter();
    EXPECT_EQ(d.clock.stats().long_exits, 1u);
    EXPECT_EQ(d.clock.stats().long_ns, 12'000u);
    EXPECT_EQ(d.clock.now() - before, cycles_from_ns(2 * 300));
}

TEST(GuestClock, TscNeverStepsBackAndTracksTheClockWithoutPreemption) {
    // A run of mixed exits: the guest never sees the TSC go back, and without preemption it ends where the clock is.
    Driver d(4000);
    for (int i = 0; i < 2000; ++i) {
        d.enter();
        d.run(5'000 + static_cast<u64>(i % 7) * 3'000);
        d.outside(i % 50 == 0 ? 400'000u : (i % 3 == 0 ? 4'999u : 800u));
    }
    d.enter();
    EXPECT_NEAR(d.lead(), 0.0, 2000.0 * 2);  // at most a cycle or two of rounding per rewind
    EXPECT_EQ(d.clock.stats().preempted_ns, 0u);
}

TEST(GuestClock, PreemptionInTheGuestLeavesTheTscAhead) {
    // The one stretch a rewind may not take back: the guest may have read the TSC after it.
    Driver d(4000);
    d.enter();
    d.run(1'000'000, 250'000);
    d.outside(1'000);
    d.enter();
    EXPECT_NEAR(d.lead(), 250'000.0 * d.clock.tsc_khz() / 1e6, 8.0);
}

TEST(GuestClock, TheTscLeadIsThePreemptionOutsideLongExitsAndNeverNegative) {
    // Preempted in a short exit: counted neither as guest time nor taken off the TSC. Blocked in a long one (a
    // paused frame): rewound whole. The lead is exactly the first.
    Driver d(2000);
    d.enter();
    d.run(50'000);
    d.outside(4'000, 3'000);  // short on the wall clock, 3 us of it preempted
    d.enter();
    d.run(50'000);
    d.outside(2'000'000, 2'000'000);  // a paused frame
    d.enter();
    d.run(50'000);
    d.outside(800'000, 100'000);  // the pusher, preempted for 0.1 ms of it
    d.enter();
    EXPECT_NEAR(d.lead(), 3'000.0 * d.clock.tsc_khz() / 1e6, 4.0);
}

TEST(GuestClock, WarpOnlyMovesForward) {
    GuestClock c(4000);
    c.mark_out(0);
    c.classify(0);
    c.entered(0, 0);
    c.left(1'000, 1'000);  // 4 us of guest time
    const Cycles now = c.now();
    EXPECT_EQ(c.warp_to(now - 1), 0u);
    EXPECT_EQ(c.now(), now);
    const Cycles skipped = c.warp_to(now + 733'333);
    EXPECT_EQ(skipped, 733'333u);
    EXPECT_EQ(c.now(), now + 733'333);
    EXPECT_EQ(c.stats().warped_cycles, 733'333u);
}
