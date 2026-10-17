#include <gtest/gtest.h>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"
#include "mcpx/isa_misc.h"
#include "mcpx/pic.h"
#include "mcpx/pit.h"
#include "mcpx/rtc.h"

using namespace mv;

namespace {
void init_pics(Bus& bus, u8 master_base, u8 slave_base) {
    bus.pio_write(0x20, 1, 0x11);  // ICW1: edge, cascade, ICW4
    bus.pio_write(0x21, 1, master_base);
    bus.pio_write(0x21, 1, 0x04);  // slave on IRQ2
    bus.pio_write(0x21, 1, 0x01);  // 8086 mode
    bus.pio_write(0xA0, 1, 0x11);
    bus.pio_write(0xA1, 1, slave_base);
    bus.pio_write(0xA1, 1, 0x02);
    bus.pio_write(0xA1, 1, 0x01);
    bus.pio_write(0x21, 1, 0x00);  // unmask all
    bus.pio_write(0xA1, 1, 0x00);
}
}  // namespace

TEST(Pic, EdgeRequestsPriorityAndEoi) {
    Bus bus(4096);
    Pic pic(bus);
    int intr_changes = 0;
    pic.set_on_intr([&](bool) { ++intr_changes; });
    init_pics(bus, 0x30, 0x38);
    EXPECT_EQ(pic.pending_vector(), -1);
    pic.set_irq(5, true);
    EXPECT_EQ(pic.pending_vector(), 0x35);
    EXPECT_TRUE(pic.intr_asserted());
    pic.set_irq(1, true);  // higher priority arrives
    EXPECT_EQ(pic.pending_vector(), 0x31);
    EXPECT_EQ(pic.acknowledge(), 0x31);
    EXPECT_EQ(pic.master().isr, 0x02);
    // IRQ5 is still pending but lower priority than the one in service.
    EXPECT_EQ(pic.pending_vector(), -1);
    bus.pio_write(0x20, 1, 0x20);  // non-specific EOI
    EXPECT_EQ(pic.pending_vector(), 0x35);
    EXPECT_EQ(pic.acknowledge(), 0x35);
    bus.pio_write(0x20, 1, 0x20);
    EXPECT_EQ(pic.pending_vector(), -1);
    EXPECT_FALSE(pic.intr_asserted());
    // Edge mode: a line held high does not re-request after acknowledge; dropping and raising does.
    pic.set_irq(5, false);
    pic.set_irq(5, true);
    EXPECT_EQ(pic.pending_vector(), 0x35);
    EXPECT_GE(intr_changes, 4);
}

TEST(Pic, SlaveCascadesThroughIrq2AndMaskingWorks) {
    Bus bus(4096);
    Pic pic(bus);
    init_pics(bus, 0x30, 0x38);
    pic.set_irq(11, true);
    EXPECT_EQ(pic.pending_vector(), 0x3B);
    EXPECT_EQ(pic.acknowledge(), 0x3B);
    EXPECT_EQ(pic.slave().isr, 0x08);
    EXPECT_EQ(pic.master().isr, 0x04);
    bus.pio_write(0xA0, 1, 0x20);  // EOI slave
    bus.pio_write(0x20, 1, 0x20);  // EOI master
    EXPECT_EQ(pic.pending_vector(), -1);
    // Masked requests stay pending and reappear when unmasked.
    bus.pio_write(0x21, 1, 0x01);  // mask IRQ0
    pic.set_irq(0, true);
    pic.set_irq(0, false);
    EXPECT_EQ(pic.pending_vector(), -1);
    bus.pio_write(0x21, 1, 0x00);
    EXPECT_EQ(pic.pending_vector(), 0x30);
    // IRR/ISR reads through OCW3.
    bus.pio_write(0x20, 1, 0x0A);
    EXPECT_EQ(bus.pio_read(0x20, 1), 0x01u);
}

TEST(Pit, ModeTwoPeriodProducesIrqEdgesOnTheTimeline) {
    Bus bus(4096);
    Scheduler sched;
    std::vector<Cycles> edges;
    Pit pit(bus, sched, [&](bool level) {
        if (level) edges.push_back(sched.now());
    });
    bus.pio_write(0x43, 1, 0x34);  // channel 0, LSB/MSB, mode 2
    bus.pio_write(0x40, 1, 0x65);  // 1125 = 0x465: 1 ms at 1.125 MHz
    bus.pio_write(0x40, 1, 0x04);
    EXPECT_EQ(pit.channel(0).reload, 1125u);
    sched.advance_to(cycles_from_ns(10'500'000));  // 10.5 ms
    ASSERT_EQ(edges.size(), 10u);
    // Each period is 1125 ticks = 733,333 cycles (floor), within one tick of 1 ms.
    for (std::size_t i = 1; i < edges.size(); ++i) {
        const Cycles period = edges[i] - edges[i - 1];
        EXPECT_GE(period, 733'333u - 652u);
        EXPECT_LE(period, 733'333u + 652u);
    }
    // The count read back is within range and the latch freezes it.
    const u16 c1 = pit.current_count(0);
    EXPECT_GE(c1, 1u);
    EXPECT_LE(c1, 1125u);
    bus.pio_write(0x43, 1, 0x00);  // latch channel 0
    const u32 lo = bus.pio_read(0x40, 1), hi = bus.pio_read(0x40, 1);
    EXPECT_EQ(static_cast<u16>(lo | (hi << 8)), c1);
}

TEST(Pit, ModeThreeSquareWaveAndModeZeroOneShot) {
    Bus bus(4096);
    Scheduler sched;
    int edges = 0;
    Pit pit(bus, sched, [&](bool level) { edges += level ? 1 : 0; });
    bus.pio_write(0x43, 1, 0x36);  // mode 3
    bus.pio_write(0x40, 1, 0x00);
    bus.pio_write(0x40, 1, 0x00);  // 65536 ticks = 58.25 ms
    sched.advance_to(cycles_from_ns(200'000'000));
    EXPECT_EQ(edges, 3);  // 58.25, 116.5, 174.8 ms
    bus.pio_write(0x43, 1, 0x30);  // mode 0
    bus.pio_write(0x40, 1, 0x10);
    bus.pio_write(0x40, 1, 0x27);  // 10000 ticks = 8.9 ms
    edges = 0;
    sched.advance_to(sched.now() + cycles_from_ns(100'000'000));
    EXPECT_EQ(edges, 1);
    EXPECT_TRUE(pit.out(0));
}

TEST(Rtc, TimeRegistersFollowTheTimelineAndPeriodicInterruptsFire) {
    Bus bus(4096);
    Scheduler sched;
    int irq_raised = 0;
    Rtc rtc(bus, sched, [&](bool level) { irq_raised += level ? 1 : 0; });
    rtc.set_base_time(1'000'000'000);  // 2001-09-09 01:46:40 UTC
    auto read = [&](u8 reg) {
        bus.pio_write(0x70, 1, reg);
        return bus.pio_read(0x71, 1);
    };
    EXPECT_EQ(read(0x00), 40u);  // seconds (binary mode)
    EXPECT_EQ(read(0x02), 46u);
    EXPECT_EQ(read(0x04), 1u);
    EXPECT_EQ(read(0x07), 9u);
    EXPECT_EQ(read(0x08), 9u);
    EXPECT_EQ(read(0x09), 1u);  // year 01
    EXPECT_EQ(read(0x32), 20u);
    sched.advance_to(cycles_from_ns_ceil(25'000'000'000ull));  // 25 s
    EXPECT_EQ(read(0x00), 5u);
    EXPECT_EQ(read(0x02), 47u);
    // Periodic at 1024 Hz (rate 6) with PIE: ~1024 interrupts per second; reading C clears the level.
    bus.pio_write(0x70, 1, 0x0A);
    bus.pio_write(0x71, 1, 0x26);
    bus.pio_write(0x70, 1, 0x0B);
    bus.pio_write(0x71, 1, 0x46);  // PIE | 24h | binary
    const Cycles t0 = sched.now();
    int count = 0;
    while (sched.now() < t0 + cycles_from_ns(100'000'000)) {
        sched.advance_to(sched.next_deadline());
        bus.pio_write(0x70, 1, 0x0C);
        if (bus.pio_read(0x71, 1) & 0x40) ++count;
    }
    EXPECT_GE(count, 100);
    EXPECT_LE(count, 105);
    EXPECT_GE(irq_raised, 100);
}
