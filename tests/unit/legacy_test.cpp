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

// The slave's INT output is a wire into master line 2: it must rise again when the slave's EOI (or a mask write)
// uncovers a request that arrived while a higher-priority slave interrupt was in service. The console's kernel
// hit the lost case with the SMBus (IRQ 11, level) in service when the IDE (IRQ 14, edge) completed.
TEST(Pic, SlaveEoiUncoversALowerRequestAndRaisesTheCascadeAgain) {
    Bus bus(4096);
    Pic pic(bus);
    init_pics(bus, 0x30, 0x38);
    bus.pio_write(0x4D1, 1, 0x18);  // IRQ 11 and 12 level-triggered, as the kernel programs the ELCR
    pic.set_irq(11, true);
    EXPECT_EQ(pic.acknowledge(), 0x3B);
    // While the SMBus handler runs, the IDE completes.
    pic.set_irq(14, true);
    EXPECT_EQ(pic.pending_vector(), -1) << "IRQ 14 does not beat IRQ 11 in service";
    // The handler reads the SMBus status (the line drops), then EOIs the slave and the master.
    pic.set_irq(11, false);
    bus.pio_write(0xA0, 1, 0x20);
    bus.pio_write(0x20, 1, 0x20);
    EXPECT_EQ(pic.master().isr, 0);
    EXPECT_EQ(pic.slave().isr, 0);
    EXPECT_EQ(pic.pending_vector(), 0x3E) << "the IDE request must now reach the CPU through the cascade";
    EXPECT_TRUE(pic.intr_asserted());
    EXPECT_EQ(pic.acknowledge(), 0x3E);
    bus.pio_write(0xA0, 1, 0x20);
    bus.pio_write(0x20, 1, 0x20);
    EXPECT_EQ(pic.pending_vector(), -1);
    EXPECT_FALSE(pic.intr_asserted());
    // The same through a mask: a request behind a masked line is delivered when the mask lifts.
    bus.pio_write(0xA1, 1, 0x40);  // mask IRQ 14
    pic.set_irq(14, false);
    pic.set_irq(14, true);
    EXPECT_EQ(pic.pending_vector(), -1);
    bus.pio_write(0xA1, 1, 0x00);
    EXPECT_EQ(pic.pending_vector(), 0x3E);
}

TEST(Pic, TheResetLineDropsPendingRequestsAndMasksEverything) {
    Bus bus(4096);
    Pic pic(bus);
    init_pics(bus, 0x30, 0x38);
    pic.set_irq(5, true);
    pic.set_irq(14, true);
    ASSERT_EQ(pic.pending_vector(), 0x3E) << "the cascade outranks IRQ 5";
    pic.reset();
    EXPECT_EQ(pic.pending_vector(), -1);
    EXPECT_FALSE(pic.intr_asserted());
    EXPECT_EQ(pic.master().imr, 0xFF);
    EXPECT_EQ(pic.slave().irr, 0);
    // Re-initialised, a fresh edge is seen again.
    init_pics(bus, 0x30, 0x38);
    pic.set_irq(5, false);
    pic.set_irq(5, true);
    EXPECT_EQ(pic.pending_vector(), 0x35);
}

namespace {
// The PIT on the PIC's IRQ0 as the machine wires it; with `policy` the fast mode's lost-tick policy is on.
struct TickRig {
    Bus bus{4096};
    Scheduler sched;
    Pic pic{bus};
    Pit pit{bus, sched, [this](bool level) { pic.set_irq(0, level); }};
    explicit TickRig(bool policy) {
        init_pics(bus, 0x30, 0x38);
        if (policy) {
            pit.set_lost_tick_policy([this] { return pic.request_pending(0); });
            pic.set_on_acknowledge([this](unsigned irq) {
                if (irq == 0) pit.on_irq0_acknowledged();
            });
        }
        bus.pio_write(0x43, 1, 0x34);  // 1 ms ticks
        bus.pio_write(0x40, 1, 0x65);
        bus.pio_write(0x40, 1, 0x04);
    }
    // The guest's ISR: take every timer interrupt on offer (acknowledge + EOI); returns how many it took.
    int drain() {
        int taken = 0;
        while (pic.pending_vector() == 0x30) {
            EXPECT_EQ(pic.acknowledge(), 0x30);
            bus.pio_write(0x20, 1, 0x20);
            ++taken;
        }
        return taken;
    }
};
}  // namespace

TEST(Pit, LostTickPolicyDeliversEveryTickTheGuestWasLateFor) {
    // Ten edges arrive while the guest cannot take interrupts (a vCPU that was not running).
    TickRig hw(false), fast(true);
    hw.sched.advance_to(cycles_from_ns(10'500'000));
    fast.sched.advance_to(cycles_from_ns(10'500'000));
    // The bare 8259 merges them: one interrupt for ten ticks (the guest's clock loses nine milliseconds).
    EXPECT_EQ(hw.drain(), 1);
    // The policy keeps the nine it could not latch and raises one per acknowledge.
    EXPECT_EQ(fast.pit.tick_backlog(), 9u);
    EXPECT_EQ(fast.drain(), 10);
    EXPECT_EQ(fast.pit.ticks_delayed(), 9u);
    EXPECT_EQ(fast.pit.tick_backlog(), 0u);
    // A guest that keeps up never builds a backlog: one tick, one interrupt.
    for (int ms = 11; ms <= 20; ++ms) {
        fast.sched.advance_to(cycles_from_ns(static_cast<u64>(ms) * 1'000'000 + 500'000));
        EXPECT_EQ(fast.drain(), 1) << "at " << ms << " ms";
    }
    EXPECT_EQ(fast.pit.ticks_delayed(), 9u);
}

TEST(Pit, LostTickBacklogIsBoundedAndVoidedByANewRate) {
    TickRig fast(true);
    // A long host stall: 300 ticks pass untaken; the backlog keeps the newest bound and drops the rest.
    fast.sched.advance_to(cycles_from_ns(300'500'000));
    EXPECT_EQ(fast.pit.tick_backlog(), Pit::kMaxTickBacklog);
    EXPECT_EQ(fast.pit.ticks_dropped(), 300u - 1u - Pit::kMaxTickBacklog);
    // Reprogramming channel 0 voids the late ticks of the old rate.
    fast.bus.pio_write(0x43, 1, 0x34);
    fast.bus.pio_write(0x40, 1, 0x65);
    fast.bus.pio_write(0x40, 1, 0x04);
    EXPECT_EQ(fast.pit.tick_backlog(), 0u);
    EXPECT_EQ(fast.drain(), 1);  // the one latched in the 8259 is still there
}
