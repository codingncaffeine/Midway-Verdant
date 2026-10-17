#include <gtest/gtest.h>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"
#include "smbus/smbus.h"
#include "smbus/smc.h"

using namespace mv;

namespace {
struct Rig {
    Bus bus{4096};
    Scheduler sched;
    int irq{0};
    SmbusController ctrl{bus, sched, [this](bool level) { irq += level ? 1 : 0; }};
    Smc smc;
    Eeprom eeprom;
    Rig() {
        ctrl.config_write(0x14, 4, 0xC001);  // BAR1 -> 0xC000
        ctrl.attach(0x10, smc);
        ctrl.attach(0x54, eeprom);
    }
    // The transaction as the console's software does it: address, command, data, clear status, start, poll.
    bool write_byte(u8 addr7, u8 cmd, u8 v) {
        bus.pio_write(0xC004, 1, static_cast<u32>(addr7 << 1));
        bus.pio_write(0xC008, 1, cmd);
        bus.pio_write(0xC006, 2, v);
        bus.pio_write(0xC000, 2, bus.pio_read(0xC000, 2));
        bus.pio_write(0xC002, 1, 0x0A);
        return finish();
    }
    std::optional<u16> read(u8 addr7, u8 cmd, bool word) {
        bus.pio_write(0xC004, 1, static_cast<u32>((addr7 << 1) | 1));
        bus.pio_write(0xC008, 1, cmd);
        bus.pio_write(0xC000, 2, bus.pio_read(0xC000, 2));
        bus.pio_write(0xC002, 1, word ? 0x0B : 0x0A);
        if (!finish()) return std::nullopt;
        return static_cast<u16>(bus.pio_read(0xC006, 2));
    }
    bool finish() {
        int spins = 0;
        while (bus.pio_read(0xC000, 1) & SmbusController::kStatusBusy) {
            sched.advance_to(sched.next_deadline());
            if (++spins > 10) return false;
        }
        // Done means the completion bit without abort, protocol error or timeout (the AMD-756 layout).
        const u32 st = bus.pio_read(0xC000, 1);
        return (st & SmbusController::kStatusComplete) && !(st & (SmbusController::kStatusAbort | SmbusController::kStatusProtocolError | SmbusController::kStatusTimeout));
    }
};
}  // namespace

TEST(Smbus, CyclesTakeBusTimeAndReachTheRightSlave) {
    Rig r;
    const Cycles t0 = r.sched.now();
    r.eeprom.image()[0x40] = 0x12;
    r.eeprom.image()[0x41] = 0x34;
    auto v = r.read(0x54, 0x40, true);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 0x3412u);
    // A word read is five bytes of nine bits at 100 kHz: 450 us.
    const u64 ns = ns_from_cycles(r.sched.now() - t0);
    EXPECT_GE(ns, 400'000u);
    EXPECT_LE(ns, 500'000u);
    EXPECT_TRUE(r.write_byte(0x54, 0x10, 0xAB));
    EXPECT_EQ(r.eeprom.image()[0x10], 0xABu);
    EXPECT_TRUE(r.eeprom.dirty());
    // No slave at 0x45 in this rig: protocol error, no completion bit.
    EXPECT_FALSE(r.read(0x45, 0x00, false).has_value());
    EXPECT_TRUE(r.bus.pio_read(0xC000, 1) & SmbusController::kStatusProtocolError);
    EXPECT_EQ(r.ctrl.transactions(), 3u);
}

TEST(Smbus, SmcAnswersTheBootloaderAndTheVersionCursor) {
    Rig r;
    // The watchdog challenge is never all zeros, and the response is accepted.
    u32 challenge = 0;
    for (u8 i = 0; i < 4; ++i) {
        auto b = r.read(0x10, static_cast<u8>(Smc::kRegChallenge0 + i), false);
        ASSERT_TRUE(b.has_value());
        challenge |= static_cast<u32>(*b) << (8 * i);
    }
    EXPECT_NE(challenge, 0u);
    EXPECT_FALSE(r.smc.challenge_answered());
    EXPECT_TRUE(r.write_byte(0x10, Smc::kRegResponse0, 0x33));
    EXPECT_TRUE(r.write_byte(0x10, Smc::kRegResponse0 + 1, 0x44));
    EXPECT_TRUE(r.smc.challenge_answered());
    EXPECT_EQ(r.smc.response(), 0x4433u);
    // The version string reads out one character per read, from the start after a write to 0x01.
    EXPECT_TRUE(r.write_byte(0x10, Smc::kRegVersion, 0));
    EXPECT_EQ(*r.read(0x10, Smc::kRegVersion, false), 'P');
    EXPECT_EQ(*r.read(0x10, Smc::kRegVersion, false), '0');
    EXPECT_EQ(*r.read(0x10, Smc::kRegVersion, false), '1');
    EXPECT_EQ(*r.read(0x10, Smc::kRegVersion, false), 'P');
    // Tray and AV pack report the configured state.
    EXPECT_EQ(*r.read(0x10, Smc::kRegTrayState, false), static_cast<u8>(Smc::Tray::ClosedNoMedia));
    r.smc.set_av_pack(Smc::AvPack::Composite);
    EXPECT_EQ(*r.read(0x10, Smc::kRegAvPack, false), 0x06u);
    // Power-off requests reach the machine.
    u8 got = 0;
    r.smc.set_on_reset_request([&](u8 bits) { got = bits; });
    EXPECT_TRUE(r.write_byte(0x10, Smc::kRegReset, Smc::kResetBitPowerOff));
    EXPECT_EQ(got, Smc::kResetBitPowerOff);
}

TEST(Smbus, InterruptFollowsCompletionWhenEnabled) {
    Rig r;
    r.bus.pio_write(0xC004, 1, 0x54 << 1);
    r.bus.pio_write(0xC008, 1, 0x00);
    r.bus.pio_write(0xC006, 2, 0x00);
    r.bus.pio_write(0xC002, 1, 0x1A);  // byte write, start, interrupt enabled
    EXPECT_EQ(r.irq, 0);
    r.sched.advance_to(r.sched.next_deadline());
    EXPECT_EQ(r.irq, 1);
    r.bus.pio_write(0xC000, 1, 0xFF);  // clear status drops the line
    r.bus.pio_write(0xC002, 1, 0x00);
}

TEST(Smbus, TheResetLineUnmapsTheControllerAndClearsItsRegisters) {
    Rig r;
    r.bus.pio_write(0xC004, 1, 0x20);
    r.ctrl.reset_config();
    EXPECT_EQ(r.ctrl.bar_base(1), 0u);
    EXPECT_EQ(r.bus.pio_read(0xC004, 1), 0xFFu) << "nothing answers at the old block: open bus";
    r.ctrl.config_write(0x14, 4, 0xC001);
    r.ctrl.reset();
    EXPECT_EQ(r.bus.pio_read(0xC004, 1), 0u) << "the address register reads clear after the reset";
    EXPECT_TRUE(r.write_byte(0x10, 0x07, 0x01)) << "and the bus works again";
}
