#include <gtest/gtest.h>

#include "core/clock.h"
#include "core/scheduler.h"

using namespace mv;

TEST(Clock, MulDivIsExactAtLargeValues) {
    // 2^63 * 3 / 7 would overflow a 64-bit product; the answer is checked against a long-hand constant.
    EXPECT_EQ(muldiv64(1ull << 63, 3, 7), 3952873730080618203ull);
    EXPECT_EQ(muldiv64(0, 12345, 7), 0u);
    EXPECT_EQ(muldiv64(1'000'000'000, 2'200'000'000ull, 3'000'000'000ull), 733'333'333ull);
}

TEST(Clock, CeilingConversionNeverLandsEarly) {
    // A deadline converted to cycles and back must not fall short of its tick: that is the loop the PIT hit.
    for (u64 n = 1; n < 5000; ++n) {
        const Cycles c = cycles_from_ticks_ceil(n, clk::kPitHz);
        EXPECT_GE(ticks_from_cycles(c, clk::kPitHz), n);
        EXPECT_LT(ticks_from_cycles(c - 1, clk::kPitHz), n);
    }
    EXPECT_EQ(muldiv64_ceil(10, 3, 5), 6u);
    EXPECT_EQ(muldiv64_ceil(10, 5, 5), 10u);
}

TEST(Clock, CyclesAndTicksRoundTrip) {
    // One second of the master clock is 733,333,333 cycles (truncated).
    EXPECT_EQ(cycles_from_ns(1'000'000'000), 733'333'333u);
    EXPECT_EQ(ns_from_cycles(733'333'333u), 999'999'999u);  // floor
    // One PIT tick at 1.125 MHz is 651.85 cycles; 1125 ticks are exactly 1 ms = 733,333 cycles (floor).
    EXPECT_EQ(cycles_from_ticks(1125, clk::kPitHz), 733'333u);
    EXPECT_EQ(ticks_from_cycles(cycles_from_ticks(1125, clk::kPitHz) + 1, clk::kPitHz), 1125u);
    // GPU cycles: 7 GPU cycles = 22 CPU cycles exactly.
    EXPECT_EQ(cycles_from_ticks(7, clk::kGpuHzNum / clk::kGpuHzDen), 22u);  // 233,333,333 Hz, one part in 7e8 under the exact ratio
    EXPECT_EQ(muldiv64(7, 2'200'000'000ull, 700'000'000ull), 22u);
}

TEST(Scheduler, FiresInTimeOrderAndSupportsReschedule) {
    Scheduler s;
    std::vector<int> order;
    TimerEvent a, b, c;
    a.fire = [&](Cycles) { order.push_back(1); };
    b.fire = [&](Cycles) { order.push_back(2); };
    c.fire = [&](Cycles now) {
        order.push_back(3);
        s.schedule(c, now + 10);  // periodic
    };
    s.schedule(b, 200);
    s.schedule(a, 100);
    s.schedule(c, 150);
    EXPECT_EQ(s.next_deadline(), 100u);
    EXPECT_EQ(s.advance_to(99), 0u);
    EXPECT_TRUE(order.empty());
    EXPECT_EQ(s.advance_to(175), 4u);  // a@100, c@150, c@160, c@170
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], 1);
    EXPECT_EQ(order[1], 3);
    EXPECT_EQ(s.now(), 175u);
    s.cancel(c);
    EXPECT_FALSE(c.scheduled());
    EXPECT_EQ(s.advance_to(1000), 1u);
    EXPECT_EQ(order.back(), 2);
    EXPECT_FALSE(s.has_pending());
    // Rescheduling an already scheduled event moves it rather than duplicating it.
    s.schedule(a, 2000);
    s.schedule(a, 1500);
    EXPECT_EQ(s.pending_count(), 1u);
    EXPECT_EQ(s.next_deadline(), 1500u);
}

TEST(Scheduler, EventInThePastFiresOnNextRun) {
    Scheduler s;
    int fired = 0;
    TimerEvent e;
    e.fire = [&](Cycles) { ++fired; };
    s.advance_to(500);
    s.schedule(e, 100);
    EXPECT_EQ(s.run_due(), 1u);
    EXPECT_EQ(fired, 1);
    EXPECT_EQ(s.now(), 500u);
}
