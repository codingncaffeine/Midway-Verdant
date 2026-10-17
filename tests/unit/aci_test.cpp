#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "aci/aci.h"
#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"

using namespace mv;

namespace {

struct Rig {
    static constexpr PhysAddr kBar = 0xFEC00000;
    static constexpr u32 kPo = Aci::kBusMasterBase + Aci::kChannelOffset[Aci::PcmOut];
    Bus bus{64 * 1024};
    Scheduler sched;
    bool irq{false};
    int edges{0};
    Aci aci{bus, sched, [this](bool level) {
                irq = level;
                ++edges;
            }};
    std::vector<u8> played;
    Rig() {
        aci.config_write(0x18, 4, kBar);
        aci.set_sample_sink([this](Aci::Channel c, const u8* d, u32 n) {
            if (c == Aci::PcmOut) played.insert(played.end(), d, d + n);
        });
    }
    u32 rd(u32 off, unsigned size) { return bus.read(kBar + off, size); }
    void wr(u32 off, unsigned size, u32 v) { bus.write(kBar + off, size, v); }
    void descriptor(PhysAddr list, u32 index, u32 buffer, u32 samples, bool ioc) {
        const u32 d[2] = {buffer, samples | (ioc ? Aci::kDescIoc : 0u)};
        ASSERT_TRUE(bus.write_ram(list + index * 8, d, 8));
    }
    static Cycles samples(u64 n) { return cycles_from_ticks_ceil(n, Aci::kSamplesPerSecond); }
};

}  // namespace

TEST(Aci, PcmOutWalksItsDescriptorsAtTheSampleClockAndInterruptsOnCompletion) {
    Rig r;
    // Two buffers: 64 samples at 0x4000 and 32 at 0x5000, both interrupting; patterns to recognise in the sink.
    for (u32 i = 0; i < 128; ++i) r.bus.ram()[0x4000 + i] = static_cast<u8>(i);
    for (u32 i = 0; i < 64; ++i) r.bus.ram()[0x5000 + i] = static_cast<u8>(0x80 + i);
    r.descriptor(0x1000, 0, 0x4000, 64, true);
    r.descriptor(0x1000, 1, 0x5000, 32, true);
    r.wr(Rig::kPo + Aci::kBdbar, 4, 0x1000);
    r.wr(Rig::kPo + Aci::kLvi, 1, 1);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kSr, 2), Aci::kSrDch) << "halted before run";
    r.wr(Rig::kPo + Aci::kCr, 1, Aci::kCrRpbm | Aci::kCrIoce | Aci::kCrLvbie);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kSr, 2) & Aci::kSrDch, 0u);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kCiv, 1), 0u);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kPiv, 1), 1u);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kPicb, 2), 64u);
    EXPECT_FALSE(r.irq);

    r.sched.advance_to(Rig::samples(64) - 1);
    EXPECT_FALSE(r.irq) << "not before the buffer has played";
    EXPECT_GE(r.rd(Rig::kPo + Aci::kPicb, 2), 1u) << "position counts down with time";
    r.sched.advance_to(Rig::samples(64));
    EXPECT_TRUE(r.irq) << "interrupt on completion";
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kSr, 2) & (Aci::kSrBcis | Aci::kSrDch | Aci::kSrLvbci), Aci::kSrBcis);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kCiv, 1), 1u) << "on to the next buffer";
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kPicb, 2), 32u);
    EXPECT_TRUE(r.aci.glob_sta() & Aci::kGlobStaChannelInt[Aci::PcmOut]);
    ASSERT_EQ(r.played.size(), 128u);
    EXPECT_EQ(r.played[5], 5u);
    r.wr(Rig::kPo + Aci::kSr, 2, Aci::kSrBcis);
    EXPECT_FALSE(r.irq) << "status is write one to clear";
    EXPECT_FALSE(r.aci.glob_sta() & Aci::kGlobStaChannelInt[Aci::PcmOut]);

    r.sched.advance_to(Rig::samples(64) + Rig::samples(32));
    EXPECT_TRUE(r.irq);
    const u32 sr = r.rd(Rig::kPo + Aci::kSr, 2);
    EXPECT_TRUE(sr & Aci::kSrBcis);
    EXPECT_TRUE(sr & Aci::kSrLvbci) << "the last valid buffer completed";
    EXPECT_TRUE(sr & Aci::kSrDch) << "and the channel halted";
    EXPECT_TRUE(sr & Aci::kSrCelv);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kCiv, 1), 1u) << "stays on the last buffer";
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kPicb, 2), 0u);
    ASSERT_EQ(r.played.size(), 192u);
    EXPECT_EQ(r.played[128], 0x80u);
    EXPECT_EQ(r.aci.channel(Aci::PcmOut).buffers_completed, 2u);
    r.wr(Rig::kPo + Aci::kSr, 2, Aci::kSrBcis | Aci::kSrLvbci);
    EXPECT_FALSE(r.irq);

    // Naming a further buffer resumes the halted channel from the prefetched index.
    r.descriptor(0x1000, 2, 0x4000, 16, false);
    r.wr(Rig::kPo + Aci::kLvi, 1, 2);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kSr, 2) & Aci::kSrDch, 0u);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kCiv, 1), 2u);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kPicb, 2), 16u);
}

TEST(Aci, ChannelResetHaltsAndClearsAndTheGlobalStatusShowsTheCodecReady) {
    Rig r;
    r.descriptor(0x1000, 0, 0x4000, 64, true);
    r.wr(Rig::kPo + Aci::kBdbar, 4, 0x1000);
    r.wr(Rig::kPo + Aci::kLvi, 1, 3);
    r.wr(Rig::kPo + Aci::kCr, 1, Aci::kCrRpbm | Aci::kCrIoce);
    r.sched.advance_to(Rig::samples(10));
    r.wr(Rig::kPo + Aci::kCr, 1, Aci::kCrRr);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kCr, 1), 0u) << "the reset bit does not stick";
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kSr, 2), Aci::kSrDch);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kCiv, 1), 0u);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kLvi, 1), 0u);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kPicb, 2), 0u);
    r.sched.advance_to(Rig::samples(200));
    EXPECT_FALSE(r.irq) << "nothing completes after a reset";
    EXPECT_EQ(r.aci.channel(Aci::PcmOut).buffers_completed, 0u);
    EXPECT_EQ(r.rd(Aci::kBusMasterBase + Aci::kGlobSta, 4) & Aci::kGlobStaPcr, Aci::kGlobStaPcr);
    EXPECT_EQ(r.rd(Aci::kBusMasterBase + Aci::kCas, 1), 0u) << "the codec semaphore is free";
    // Mixer: 16-bit registers store; the power-down register reports every section ready.
    r.wr(0x18, 2, 0x0808);
    EXPECT_EQ(r.rd(0x18, 2), 0x0808u);
    r.wr(0x19, 1, 0x1F);
    EXPECT_EQ(r.rd(0x18, 2), 0x1F08u) << "a byte write lands in its half";
    EXPECT_EQ(r.rd(Aci::kMixPowerdown, 2) & 0xF, Aci::kMixPowerdownReady);
    r.wr(0x02, 2, 0x8000);
    r.wr(0x00, 2, 0);
    EXPECT_EQ(r.rd(0x02, 2), 0u) << "a write to the reset register resets the codec";
}

TEST(Aci, AnEightKibBufferPlaysOutInTheBandOfItsSampleCount) {
    Rig r;
    r.descriptor(0x1000, 0, 0x4000, 0x1000, true);
    r.wr(Rig::kPo + Aci::kBdbar, 4, 0x1000);
    r.wr(Rig::kPo + Aci::kLvi, 1, 0);
    r.wr(Rig::kPo + Aci::kCr, 1, Aci::kCrRpbm | Aci::kCrIoce);
    ASSERT_TRUE(r.sched.has_pending());
    const double ms = static_cast<double>(ns_from_cycles(r.sched.next_deadline())) / 1e6;
    EXPECT_GE(ms, 42.6) << "4096 16-bit samples at 48 kHz stereo";
    EXPECT_LE(ms, 42.8);
}

TEST(Aci, ReArmingTheSameLastValidIndexReplaysTheSingleBufferAndTheSpdifMirrorStaysSilent) {
    Rig r;
    // The kernel's shape: one 8 KiB buffer, last valid index 0, on both PCM out and the S/PDIF mirror.
    r.descriptor(0x1000, 0, 0x4000, 0x1000, true);
    r.descriptor(0x2000, 0, 0x4000, 0x1000, true);
    constexpr u32 kSp = Aci::kBusMasterBase + Aci::kChannelOffset[Aci::Spdif];
    r.wr(kSp + Aci::kBdbar, 4, 0x2000);
    r.wr(kSp + Aci::kLvi, 1, 0);
    r.wr(kSp + Aci::kCr, 1, Aci::kCrRpbm | Aci::kCrIoce | Aci::kCrLvbie);
    r.wr(Rig::kPo + Aci::kBdbar, 4, 0x1000);
    r.wr(Rig::kPo + Aci::kLvi, 1, 0);
    r.wr(Rig::kPo + Aci::kCr, 1, Aci::kCrRpbm | Aci::kCrIoce | Aci::kCrLvbie);
    const Cycles first = Rig::samples(0x1000);
    r.sched.advance_to(first);
    EXPECT_TRUE(r.irq);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kSr, 2), Aci::kSrBcis | Aci::kSrLvbci | Aci::kSrDch | Aci::kSrCelv);
    EXPECT_EQ(r.rd(Aci::kBusMasterBase + Aci::kGlobSta, 4), Aci::kGlobStaPcr | Aci::kGlobStaChannelInt[Aci::PcmOut])
        << "only the PCM out channel shows; the S/PDIF mirror completed too but reports nothing";
    EXPECT_EQ(r.aci.channel(Aci::Spdif).buffers_completed, 1u);
    // The kernel's service: clear the status, write the same last valid index.
    r.wr(Rig::kPo + Aci::kSr, 2, 0xF);
    EXPECT_FALSE(r.irq);
    r.wr(Rig::kPo + Aci::kLvi, 1, 0);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kSr, 2) & Aci::kSrDch, 0u) << "running again";
    r.sched.advance_to(first + 64);
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kCiv, 1), 0u) << "the 31 empty descriptors completed at once and the list wrapped";
    EXPECT_EQ(r.rd(Rig::kPo + Aci::kPicb, 2), 0x1000u);
    EXPECT_FALSE(r.irq) << "empty descriptors do not interrupt";
    r.sched.advance_to(first + 64 + Rig::samples(0x1000));
    EXPECT_TRUE(r.irq) << "the buffer played again, one period later";
    EXPECT_EQ(r.aci.channel(Aci::PcmOut).buffers_completed, 33u);
    EXPECT_EQ(r.aci.channel(Aci::PcmOut).bytes_played, 2u * 0x2000u);
}
