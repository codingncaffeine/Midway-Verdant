#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include <initializer_list>

#include "apu/apu.h"
#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"

using namespace mv;

namespace {

// The APU on a small machine: 64 KiB of RAM (16 pages) and the register window where the kernel puts it.
struct Rig {
    static constexpr PhysAddr kBar = 0xFE800000;
    Bus bus{64 * 1024};
    Scheduler sched;
    bool irq{false};
    int irq_edges{0};
    Apu apu{bus, sched, [this](bool level) {
                irq = level;
                ++irq_edges;
            }};
    // The frame clock starts as the kernel starts it: the setup engine's count mode on (SECTL = 0xF), at time 0.
    Rig() {
        apu.config_write(0x10, 4, kBar);
        wr(Apu::kSectl, 0xF);
    }
    u32 rd(u32 off) { return bus.read(kBar + off, 4); }
    void wr(u32 off, u32 v) { bus.write(kBar + off, 4, v); }
    void ram32(PhysAddr at, u32 v) { ASSERT_TRUE(bus.write_ram(at, &v, 4)); }
    u32 ram32(PhysAddr at) {
        u32 v = 0;
        bus.read_ram(at, &v, 4);
        return v;
    }
    // A scatter-gather table as the kernel builds it: 8-byte entries, the page's physical address first.
    void table(PhysAddr at, std::initializer_list<PhysAddr> pages) {
        u32 i = 0;
        for (PhysAddr p : pages) {
            ram32(at + i * 8, p);
            ram32(at + i * 8 + 4, 0);
            ++i;
        }
    }
    static Cycles frame(u64 n) { return cycles_from_ticks(n, clk::kApuFramesPerSecond); }
};

}  // namespace

TEST(ApuRegisters, InterruptStatusIsWriteOneToClearAndTheLineFollowsTheSummaryBit) {
    Rig r;
    r.wr(Apu::kIen, Apu::kIstsFrontEndTrap);
    r.apu.raise(Apu::kIstsFrontEndTrap);
    EXPECT_EQ(r.rd(Apu::kIsts), Apu::kIstsFrontEndTrap);
    EXPECT_FALSE(r.irq) << "the summary bit is not enabled, so the line stays low";
    r.wr(Apu::kIen, Apu::kIstsGlobal | Apu::kIstsFrontEndTrap);
    EXPECT_EQ(r.rd(Apu::kIsts), Apu::kIstsGlobal | Apu::kIstsFrontEndTrap) << "the summary bit appears with the enable";
    EXPECT_TRUE(r.irq);
    r.wr(Apu::kIsts, 0xFFFFFFFFu);
    EXPECT_EQ(r.rd(Apu::kIsts), 0u);
    EXPECT_FALSE(r.irq);
    EXPECT_EQ(r.irq_edges, 2);
    // Clearing only the summary bit leaves the source pending, so the summary comes straight back.
    r.apu.raise(Apu::kIstsFrontEndTrap);
    r.wr(Apu::kIsts, Apu::kIstsGlobal);
    EXPECT_EQ(r.rd(Apu::kIsts), Apu::kIstsGlobal | Apu::kIstsFrontEndTrap);
    EXPECT_TRUE(r.irq);
    // A byte-sized write clears only the bits in that byte.
    r.wr(Apu::kIen, 0x0111u);
    r.apu.raise(0x0100u);
    r.bus.write(Rig::kBar + Apu::kIsts + 1, 1, 0x01);
    EXPECT_EQ(r.rd(Apu::kIsts), Apu::kIstsGlobal | Apu::kIstsFrontEndTrap) << "bit 8 cleared, bit 4 still pending";
    r.bus.write(Rig::kBar + Apu::kIsts, 1, 0x10);
    EXPECT_EQ(r.rd(Apu::kIsts), 0u);
    EXPECT_FALSE(r.irq);
}

TEST(ApuRegisters, GlobalRegistersStoreAndTheVoiceProcessorWindowReportsFreeEntries) {
    Rig r;
    r.wr(Apu::kGpScratchTable, 0x1000);
    r.wr(Apu::kSectl, 0xF);
    EXPECT_EQ(r.rd(Apu::kGpScratchTable), 0x1000u);
    EXPECT_EQ(r.rd(Apu::kSectl), 0xFu);
    EXPECT_EQ(r.rd(Apu::kVpBlock + Apu::kVpFree), 0x80u) << "the kernel polls this before each method";
    r.wr(Apu::kVpBlock + 0x1000, 1);
    EXPECT_EQ(r.rd(Apu::kVpBlock + 0x1000), 1u) << "methods the engine does not model are kept";
    EXPECT_EQ(r.apu.reads(), 4u);
    EXPECT_EQ(r.apu.writes(), 4u) << "three here and the rig's own SECTL write";
}

TEST(ApuGp, MemoriesAppearInTheWindowTwentyFourBitsWide) {
    Rig r;
    r.wr(Apu::kGpBlock + Apu::kXMem + 0x200 * 4, 0xFF123456u);
    EXPECT_EQ(r.rd(Apu::kGpBlock + Apu::kXMem + 0x200 * 4), 0x123456u);
    EXPECT_EQ(r.apu.gp().core().read_x(0x200), 0x123456u);
    r.wr(Apu::kGpBlock + Apu::kMixBuf, 0x111111u);
    EXPECT_EQ(r.apu.gp().core().read_x(Apu::kMixBufWord), 0x111111u) << "the mix buffer is X:$1400";
    r.wr(Apu::kGpBlock + Apu::kYMem + 4, 0x222222u);
    EXPECT_EQ(r.apu.gp().core().read_y(1), 0x222222u);
    r.wr(Apu::kGpBlock + Apu::kPMem + 8, 0x333333u);
    EXPECT_EQ(r.apu.gp().core().read_p(2), 0x333333u);
    EXPECT_EQ(r.rd(Apu::kGpBlock + Apu::kPMem + 8), 0x333333u);
    EXPECT_EQ(r.rd(Apu::kGpBlock + Apu::kReset), 0u);
}

TEST(ApuGp, ReleasingTheResetLoadsTheProgramFromScratchAndEveryFrameRunsItIntoTheFifoRing) {
    Rig r;
    // The kernel's tables: scratch pages at 0x6000 and 0x7000, FIFO pages at 0x4000 and 0x5000.
    r.table(0x1000, {0x6000, 0x7000});
    r.table(0x2000, {0x4000, 0x5000});
    r.wr(Apu::kGpScratchTable, 0x1000);
    r.wr(Apu::kGpScratchMaxSge, 1);
    r.wr(Apu::kGpFifoTable, 0x2000);
    r.wr(Apu::kGpFifoMaxSge, 1);
    // Output FIFO 0 is the whole 8 KiB FIFO space, with the current pointer 8 bytes before the end.
    r.wr(Apu::kGpFifoRegs + Apu::kFifoBase, 0);
    r.wr(Apu::kGpFifoRegs + Apu::kFifoEnd, 0x2000);
    r.wr(Apu::kGpFifoRegs + Apu::kFifoCurrent, 0x1FF8);
    // The program, in scratch as 32-bit words: set the DMA list head, start it, halt; next frame: jump back.
    const u32 program[] = {0x08F494, 0x000040,   // MOVEP #$40,X:$FFFFD4
                           0x08F496, 0x000001,   // MOVEP #1,X:$FFFFD6 (start)
                           0x08F484, 0x000001,   // MOVEP #1,X:$FFFFC4 (halt until the next frame)
                           0x0C0000};            // JMP $0
    for (u32 i = 0; i < 7; ++i) r.ram32(0x6000 + i * 4, program[i]);

    r.wr(Apu::kGpBlock + Apu::kReset, Apu::kResetBlock);
    EXPECT_FALSE(r.apu.gp().running()) << "the block is on but the core is held";
    r.wr(Apu::kGpBlock + Apu::kReset, Apu::kResetBlock | Apu::kResetDsp);
    ASSERT_TRUE(r.apu.gp().running()) << r.apu.gp().last_problem();
    EXPECT_EQ(r.rd(Apu::kGpBlock + Apu::kPMem), 0x08F494u) << "P:0 came from scratch through the table";
    EXPECT_EQ(r.rd(Apu::kGpBlock + Apu::kReset), 3u);

    // The DMA command block at X:$40: four 24-bit samples from X:$200 to output FIFO 0, end of list.
    const u32 block[] = {0x4000, (1u << 1) | (0u << 5) | (6u << 10), 4, 0x200, 0, 0, 0};
    for (u32 i = 0; i < 7; ++i) r.wr(Apu::kGpBlock + Apu::kXMem + (0x40 + i) * 4, block[i]);
    const u32 samples[] = {0x111111, 0x222222, 0x333333, 0x444444};
    for (u32 i = 0; i < 4; ++i) r.wr(Apu::kGpBlock + Apu::kXMem + (0x200 + i) * 4, samples[i]);

    r.sched.advance_to(Rig::frame(1));
    ASSERT_FALSE(r.apu.gp().core().error()) << r.apu.gp().core().error_text();
    EXPECT_EQ(r.apu.frames(), 1u);
    EXPECT_TRUE(r.apu.gp().core().halted()) << "the program halts itself at the end of its frame";
    EXPECT_EQ(r.apu.gp().core().instructions(), 3u);
    EXPECT_EQ(r.ram32(0x5FF8), 0x111111u) << "the first two samples land at the end of the ring";
    EXPECT_EQ(r.ram32(0x5FFC), 0x222222u);
    EXPECT_EQ(r.ram32(0x4000), 0x333333u) << "the ring wrapped to its base";
    EXPECT_EQ(r.ram32(0x4004), 0x444444u);
    EXPECT_EQ(r.rd(Apu::kGpFifoRegs + Apu::kFifoCurrent), 8u);
    EXPECT_TRUE(r.apu.gp().dma().last_error().empty()) << r.apu.gp().dma().last_error();

    r.sched.advance_to(Rig::frame(2));
    ASSERT_FALSE(r.apu.gp().core().error()) << r.apu.gp().core().error_text();
    EXPECT_EQ(r.apu.frames(), 2u);
    EXPECT_EQ(r.apu.gp().core().instructions(), 7u) << "the jump back and the three moves";
    EXPECT_EQ(r.ram32(0x4008), 0x111111u);
    EXPECT_EQ(r.rd(Apu::kGpFifoRegs + Apu::kFifoCurrent), 24u);
    EXPECT_EQ(r.apu.gp_stats().frames_run, 2u);

    // Holding the reset stops the frames; releasing it again reloads the program.
    r.wr(Apu::kGpBlock + Apu::kReset, Apu::kResetBlock);
    EXPECT_FALSE(r.apu.gp().running());
    r.sched.advance_to(Rig::frame(3));
    EXPECT_EQ(r.rd(Apu::kGpFifoRegs + Apu::kFifoCurrent), 24u) << "nothing moved while held";
    EXPECT_EQ(r.apu.frames(), 3u) << "the frame clock runs regardless";
}

TEST(ApuGp, AnOutputFifoWhoseCurrentPointerIsPastTheEndRefusesTheTransferAndNamesIt) {
    Rig r;
    r.table(0x2000, {0x4000});
    r.wr(Apu::kGpFifoTable, 0x2000);
    r.wr(Apu::kGpFifoMaxSge, 0);
    r.wr(Apu::kGpFifoRegs + Apu::kFifoBase, 0);
    r.wr(Apu::kGpFifoRegs + Apu::kFifoEnd, 0x1000);
    r.wr(Apu::kGpFifoRegs + Apu::kFifoCurrent, 0x1000);
    // Drive the processor's DMA engine directly: a block to FIFO 0.
    auto& core = r.apu.gp().core();
    const u32 block[] = {0x4000, (1u << 1) | (6u << 10), 1, 0x200, 0, 0, 0};
    for (u32 i = 0; i < 7; ++i) core.write_x(0x40 + i, block[i]);
    core.write_x(0x200, 0x123456);
    r.apu.gp().dma().write(dsp::Dma::NextBlock, 0x40);
    r.apu.gp().dma().write(dsp::Dma::Control, 1);
    EXPECT_FALSE(r.apu.gp().dma().last_error().empty()) << "the write is refused rather than hanging";
    EXPECT_EQ(r.rd(Apu::kGpFifoRegs + Apu::kFifoCurrent), 0x1000u) << "the pointer is untouched";
    // A current pointer below the base starts at the base.
    r.wr(Apu::kGpFifoRegs + Apu::kFifoBase, 0x100);
    r.wr(Apu::kGpFifoRegs + Apu::kFifoCurrent, 0);
    r.apu.gp().dma().write(dsp::Dma::NextBlock, 0x40);
    r.apu.gp().dma().write(dsp::Dma::Control, 1);
    EXPECT_EQ(r.ram32(0x4100), 0x123456u);
    EXPECT_EQ(r.rd(Apu::kGpFifoRegs + Apu::kFifoCurrent), 0x104u);
}

namespace {
// The voice side: voice structures at RAM 0x8000, idle-voice traps requested, the trap and summary bits enabled.
struct VoiceRig : Rig {
    static constexpr u32 kVoiceBase = 0x8000;
    VoiceRig() {
        wr(apu::SetupEngine::kRegVoiceBase, kVoiceBase);
        for (u32 top : apu::SetupEngine::kRegListTop) wr(top, apu::SetupEngine::kNoVoice);  // empty lists, as the kernel writes them
        wr(apu::SetupEngine::kRegTrapForce1, apu::SetupEngine::kTrapForceIdleVoice);
        wr(Apu::kIen, Apu::kIstsGlobal | Apu::kIstsFrontEndTrap);
    }
    void vp(u32 method, u32 value) { wr(Apu::kVpBlock + method, value); }
    u32 field(u32 handle, u32 off) { return apu.setup_engine().field(handle, off); }
    u32 cbo(u32 handle) { return field(handle, apu::SetupEngine::kFieldOffset) & apu::SetupEngine::kOffsetMask; }
    u32 next_of(u32 handle) { return field(handle, apu::SetupEngine::kFieldPitchLink) & 0xFFFFu; }
    // The kernel's way out of a trap: acknowledge the status, then halt and free the front end.
    void service_trap() {
        wr(Apu::kIsts, rd(Apu::kIsts));
        const u32 fectl = rd(Apu::kFectl);
        wr(Apu::kFectl, (fectl & ~0x60u) | 0x80u);
        wr(Apu::kFectl, fectl & ~Apu::kFectlMethodMode);
    }
};
}  // namespace

TEST(ApuVoices, VoiceOnLinksAtTheTopOfItsListAndTheEngineRetiresItAtTheBufferEndWithATrap) {
    using SE = apu::SetupEngine;
    VoiceRig r;
    r.vp(SE::kMethodSetCurrentVoice, 5);
    r.vp(0x304, 0x5000E0E6u);  // 16-bit mono, no loop
    r.vp(SE::kMethodBufEbo, 63);
    r.vp(SE::kMethodBufLbo, 0);
    r.vp(0x37C, 0);  // unity pitch
    r.vp(SE::kMethodSetAntecedentVoice, 0x1FFFFu);  // top of the 2D list
    r.vp(SE::kMethodVoiceOn, 5);
    EXPECT_EQ(r.rd(SE::kRegListTop[0]), 5u);
    EXPECT_EQ(r.next_of(5), SE::kNoVoice);
    EXPECT_TRUE(r.field(5, SE::kFieldState) & SE::kStateActive);
    EXPECT_EQ(r.field(5, SE::kFieldFmt), 0x5000E0E6u) << "the configuration landed in the voice structure";

    r.sched.advance_to(Rig::frame(1));
    EXPECT_EQ(r.cbo(5), 32u) << "32 samples per frame at unity pitch";
    EXPECT_EQ(r.rd(Apu::kIsts), 0u);
    r.sched.advance_to(Rig::frame(2));
    EXPECT_EQ(r.rd(SE::kRegListTop[0]), SE::kNoVoice) << "the voice ran off its end and left the list";
    EXPECT_FALSE(r.field(5, SE::kFieldState) & SE::kStateActive);
    EXPECT_EQ(r.cbo(5), 63u);
    EXPECT_EQ(r.rd(Apu::kFectl) & (Apu::kFectlMethodMode | Apu::kFectlTrapReason), Apu::kFectlTrapped | Apu::kFectlTrapRequested);
    EXPECT_EQ(r.rd(Apu::kFeDecMethod), SE::kSe2FeIdleVoice);
    EXPECT_EQ(r.rd(Apu::kFeDecParam), 5u);
    EXPECT_EQ(r.rd(Apu::kIsts), Apu::kIstsGlobal | Apu::kIstsFrontEndTrap);
    EXPECT_TRUE(r.irq);
    EXPECT_TRUE(r.apu.front_end_trapped());
    r.service_trap();
    EXPECT_FALSE(r.irq);
    EXPECT_FALSE(r.apu.front_end_trapped());
    EXPECT_EQ(r.apu.setup_engine().voices_retired(), 1u);

    // Pitch: one octave up doubles the step.
    r.vp(SE::kMethodSetCurrentVoice, 6);
    r.vp(0x304, 0x5000E0E6u);
    r.vp(SE::kMethodBufEbo, 1000);
    r.vp(0x37C, 4096u << 16);
    r.vp(SE::kMethodVoiceOn, 6);
    EXPECT_EQ(r.rd(SE::kRegListTop[0]), 6u);
    r.sched.advance_to(Rig::frame(3));
    EXPECT_EQ(r.cbo(6), 64u);
}

TEST(ApuVoices, ALoopingVoiceWrapsAndRetiresOnlyWhenTurnedOffAndQueuedTrapsWaitForTheFrontEnd) {
    using SE = apu::SetupEngine;
    VoiceRig r;
    // The kernel's timing voice: 32 samples, looping, on the 2D list, unpaused right after it is turned on.
    r.vp(SE::kMethodSetCurrentVoice, 0x40);
    r.vp(0x304, 0x5200E0E6u);
    r.vp(SE::kMethodBufEbo, 0x1F);
    r.vp(SE::kMethodBufLbo, 0);
    r.vp(SE::kMethodSetAntecedentVoice, 0x1FFFFu);
    r.vp(SE::kMethodVoiceOn, 0x40);
    r.vp(SE::kMethodVoicePause, 0x40);
    // A second, finite voice on the 3D list.
    r.vp(SE::kMethodSetCurrentVoice, 7);
    r.vp(0x304, 0x5000E0E6u);
    r.vp(SE::kMethodBufEbo, 31);
    r.vp(SE::kMethodSetAntecedentVoice, 0x2FFFFu);
    r.vp(SE::kMethodVoiceOn, 7);
    EXPECT_EQ(r.rd(SE::kRegListTop[1]), 7u);

    r.sched.advance_to(Rig::frame(1));
    EXPECT_EQ(r.cbo(0x40), 0u) << "wrapped to the loop start";
    EXPECT_TRUE(r.field(0x40, SE::kFieldState) & SE::kStateActive);
    EXPECT_EQ(r.rd(SE::kRegListTop[0]), 0x40u);
    EXPECT_EQ(r.rd(Apu::kFeDecParam), 7u) << "the finite voice ended and trapped";
    EXPECT_TRUE(r.irq);

    r.vp(SE::kMethodVoiceOff, 0x40);
    r.sched.advance_to(Rig::frame(2));
    EXPECT_EQ(r.rd(SE::kRegListTop[0]), SE::kNoVoice) << "turned off: retired at the next frame";
    EXPECT_EQ(r.rd(Apu::kFeDecParam), 7u) << "the front end is still trapped on the first report";
    EXPECT_EQ(r.apu.traps(), 1u);
    r.service_trap();
    EXPECT_FALSE(r.irq);
    r.sched.advance_to(Rig::frame(3));
    EXPECT_EQ(r.rd(Apu::kFeDecParam), 0x40u) << "the queued report is delivered once the front end is free";
    EXPECT_TRUE(r.irq);
    EXPECT_EQ(r.apu.traps(), 2u);
    EXPECT_EQ(r.apu.setup_engine().voices_started(), 2u);
    EXPECT_EQ(r.apu.setup_engine().voices_retired(), 2u);
    r.service_trap();
    EXPECT_FALSE(r.irq);
}

TEST(ApuVoices, AStreamVoicePlaysItsSegmentListsInTurnAndReportsEachThroughItsNotifier) {
    using SE = apu::SetupEngine;
    VoiceRig r;
    constexpr PhysAddr kSslTable = 0x9000, kNotifiers = 0xA000;
    r.wr(SE::kRegSslTable, kSslTable);
    r.wr(SE::kRegNotifierBase, kNotifiers);
    r.wr(Apu::kIen, Apu::kIstsGlobal | Apu::kIstsFrontEndTrap | SE::kIstsNotify);
    // List A at page 0x40: two segments of 32 samples; list B at page 0x48: one of 64 (16-bit containers).
    r.vp(SE::kMethodSetCurrentSsl, 0x40);
    r.vp(0x600, 0x1000);
    r.vp(0x604, 0x00010020);
    r.vp(0x608, 0x2000);
    r.vp(0x60C, 0x00010020);
    // B's page 0x48 is entry 8 of the same 64-entry window: the current SSL page is 64-aligned.
    r.vp(0x640, 0x3000);
    r.vp(0x644, 0x00010040);
    EXPECT_EQ(r.ram32(kSslTable + 0x41 * 8), 0x2000u) << "segment 1 of list A in the table";
    EXPECT_EQ(r.ram32(kSslTable + 0x48 * 8 + 4), 0x00010040u);

    r.vp(SE::kMethodSetCurrentVoice, 5);
    r.vp(0x304, 0x5180E0E6u);  // 16-bit, stream, persistent
    r.vp(0x37C, 0);
    r.vp(SE::kMethodVoiceSslA, (0x40u << 8) | 2);
    r.vp(SE::kMethodVoiceSslB, (0x48u << 8) | 1);
    r.vp(SE::kMethodSetAntecedentVoice, 0x1FFFFu);
    r.vp(SE::kMethodVoiceOn, 5);
    const PhysAddr notify_a = kNotifiers + 16 * (SE::kNotifierBase + 5 * SE::kNotifiersPerVoice + 0);
    const PhysAddr notify_b = notify_a + 16;

    r.sched.advance_to(Rig::frame(1));
    EXPECT_EQ(r.ram32(notify_a + 12) >> 24, 0u) << "one segment of A played: nothing to report yet";
    r.sched.advance_to(Rig::frame(2));
    EXPECT_EQ(r.ram32(notify_a + 12) >> 24, SE::kNotifierDone) << "A's two segments done";
    EXPECT_EQ(r.rd(Apu::kIsts) & SE::kIstsNotify, SE::kIstsNotify);
    EXPECT_TRUE(r.irq);
    r.wr(Apu::kIsts, r.rd(Apu::kIsts));
    r.vp(SE::kMethodSetCurrentVoice, 5);
    r.vp(SE::kMethodVoiceSslA, 0x40u << 8);  // the driver has nothing more for A
    r.sched.advance_to(Rig::frame(3));
    EXPECT_EQ(r.ram32(notify_b + 12) >> 24, 0u) << "B half played";
    r.sched.advance_to(Rig::frame(4));
    EXPECT_EQ(r.ram32(notify_b + 12) >> 24, SE::kNotifierDone) << "B done";
    EXPECT_EQ(r.apu.setup_engine().notifications(), 3u) << "B done, and the voice reached the emptied A at once: starved";
    r.sched.advance_to(Rig::frame(6));
    EXPECT_EQ(r.apu.setup_engine().notifications(), 3u) << "the starvation is reported once, not every frame";
    EXPECT_TRUE(r.field(5, SE::kFieldState) & SE::kStateActive) << "a persistent voice waits for more";

    // Control: a voice without PERSIST ends on the empty list instead.
    r.vp(SE::kMethodSetCurrentVoice, 6);
    r.vp(0x304, 0x5100E0E6u);
    r.vp(SE::kMethodVoiceSslA, 0x40u << 8);
    r.vp(SE::kMethodVoiceOn, 6);
    r.sched.advance_to(Rig::frame(7));
    EXPECT_FALSE(r.field(6, SE::kFieldState) & SE::kStateActive);
}

namespace {
// Sample synthesis: voice 0x40 (a 2D voice) reads a buffer through the scatter-gather table at 0xB000, whose entry 0
// names the page at 0x4000; only V0 is audible (bin 0), the other seven volumes are all ones (silent).
struct SynthRig : VoiceRig {
    static constexpr u32 kVoice = 0x40;
    static constexpr PhysAddr kSge = 0xB000, kPage = 0x4000;
    SynthRig() {
        wr(apu::SetupEngine::kRegSgeTable, kSge);
        ram32(kSge, kPage);
    }
    static i16 pcm(u32 i) { return static_cast<i16>(static_cast<i32>(i) * 1000 - 20000); }
    void fill_pcm(u32 count) {
        for (u32 i = 0; i < count; ++i) {
            const u16 v = static_cast<u16>(pcm(i));
            ASSERT_TRUE(bus.write_ram(kPage + i * 2, &v, 2));
        }
    }
    // A voice on the 2D list: format word, end and loop offsets, pitch, V0's volume, the VOICE_ON envelope phases.
    void start(u32 fmt, u32 ebo, u32 lbo, i32 pitch, u32 vol0, u32 envelope_phases = 0) {
        using SE = apu::SetupEngine;
        vp(SE::kMethodSetCurrentVoice, kVoice);
        vp(0x300, 0);                                 // VBIN: V0-V5 all bin 0 (silenced below except V0)
        vp(0x304, fmt | 31u | (31u << 5));            // V6, V7 in bin 31
        vp(0x360, 0xFFF00000u | 0x000F000Fu | (vol0 << 4));  // VOLA: V0 = vol0, V1 silent, V6/V7 nibbles all ones
        vp(0x364, 0xFFFFFFFFu);
        vp(0x368, 0xFFFFFFFFu);
        vp(0x37C, static_cast<u32>(pitch) << 16);
        vp(SE::kMethodBufBase, 0);
        vp(SE::kMethodBufEbo, ebo);
        vp(SE::kMethodBufLbo, lbo);
        vp(SE::kMethodSetAntecedentVoice, 0x1FFFFu);
        vp(SE::kMethodVoiceOn, kVoice | envelope_phases);
    }
    float bin(u32 b, u32 i) { return apu.setup_engine().mix()[b][i]; }
};
constexpr u32 kPcm16Mono = 0x50000000u;  // container 16-bit, sample size S16
}  // namespace

TEST(ApuSynthesis, ABufferVoicePlaysItsSamplesIntoItsBinAtItsVolumeAndHeadroom) {
    using SE = apu::SetupEngine;
    SynthRig r;
    r.fill_pcm(64);
    r.start(kPcm16Mono, 63, 0, 0, 0);
    r.sched.advance_to(Rig::frame(1));
    for (u32 i = 0; i < 32; ++i) EXPECT_FLOAT_EQ(r.bin(0, i), SynthRig::pcm(i) / 32768.0f) << "sample " << i;
    for (u32 b = 1; b < SE::kBins; ++b) EXPECT_EQ(r.bin(b, 5), 0.0f) << "bin " << b;
    // The bins land in the GP's MIXBUF as 24-bit words.
    const auto& x = r.apu.gp().core().x_memory();
    EXPECT_EQ(x[Apu::kMixBufWord + 3], (static_cast<u32>(static_cast<i32>(SynthRig::pcm(3)) * 256)) & 0xFFFFFFu) << "a 16-bit sample, exactly, shifted to 24 bits";
    // 64 units of attenuation are 1 dB; a headroom of 1 halves what reaches the bin.
    r.vp(SE::kMethodSetCurrentVoice, SynthRig::kVoice);
    r.vp(0x360, 0xFFF00000u | 0x000F000Fu | (64u << 4));
    r.vp(SE::kMethodSubmixHeadroomFirst, 1);
    r.sched.advance_to(Rig::frame(2));
    const float g = std::pow(10.0f, -1.0f / 20.0f) / 2.0f;
    for (u32 i = 0; i < 32; ++i) EXPECT_NEAR(r.bin(0, i), SynthRig::pcm(32 + i) / 32768.0f * g, 1e-6f) << "sample " << (32 + i);
    EXPECT_EQ(r.apu.setup_engine().voices_retired(), 1u) << "the voice played its last sample and ended";
}

TEST(ApuSynthesis, PitchStepsThroughTheBufferWithLinearInterpolation) {
    SynthRig r;
    r.fill_pcm(64);
    r.start(kPcm16Mono, 63, 0, -4096, 0);  // one octave down: half a source sample per output sample
    r.sched.advance_to(Rig::frame(1));
    for (u32 i = 0; i < 32; ++i) {
        const float a = SynthRig::pcm(i / 2) / 32768.0f, b = SynthRig::pcm(i / 2 + 1) / 32768.0f;
        EXPECT_NEAR(r.bin(0, i), (i & 1) ? (a + b) / 2 : a, 1e-6f) << "sample " << i;
    }
}

TEST(ApuSynthesis, XboxAdpcmBlocksDecodeAsImaAdpcm) {
    SynthRig r;
    // One mono block: predictor 0, step index 0, then 64 nibbles of 4 (+step + step/8 each time).
    u8 block[36]{};
    for (u32 i = 4; i < 36; ++i) block[i] = 0x44;
    ASSERT_TRUE(r.bus.write_ram(SynthRig::kPage, block, sizeof block));
    r.start(0x80000000u, 64, 0, 0, 0);  // ADPCM container
    r.sched.advance_to(Rig::frame(1));
    // Steps 7, 9, 11, 13 at indices 0, 2, 4, 6: the sample rises by 7, 10, 12, 14.
    const float expect[5] = {0, 7, 17, 29, 43};
    for (u32 i = 0; i < 5; ++i) EXPECT_FLOAT_EQ(r.bin(0, i), expect[i] / 32768.0f) << "sample " << i;
}

TEST(ApuSynthesis, TheAmplitudeEnvelopeRampsInItsAttackAndTheVoiceEndsWithItsRelease) {
    using SE = apu::SetupEngine;
    SynthRig r;
    for (u32 i = 0; i < 64; ++i) {
        const u16 v = 16384;  // a constant half-scale signal
        ASSERT_TRUE(r.bus.write_ram(SynthRig::kPage + i * 2, &v, 2));
    }
    r.vp(SE::kMethodSetCurrentVoice, SynthRig::kVoice);
    r.vp(0x308, 1);           // CFG_ENV0: attack rate 1 = 16 frames
    r.vp(0x30C, 0xFF000000u); // CFG_ENVA: sustain 255, no hold or decay
    r.vp(0x36C, 1);           // TAR_LFO_ENV: release rate 1 = 16 frames
    r.start(kPcm16Mono | SE::kFmtLoop, 63, 0, 0, 0, SE::kAttack << SE::kStateEaShift);
    r.sched.advance_to(Rig::frame(9));
    EXPECT_NEAR(r.bin(0, 0), 0.5f * 8.0f / 16.0f, 1e-6f) << "the ninth frame of a 16-frame attack is at half level";
    r.sched.advance_to(Rig::frame(20));
    EXPECT_NEAR(r.bin(0, 0), 0.5f, 1e-6f) << "attack done, sustaining at full level";
    r.vp(SE::kMethodVoiceRelease, SynthRig::kVoice);
    r.sched.advance_to(Rig::frame(30));
    EXPECT_GT(r.bin(0, 0), 0.0f) << "still releasing";
    EXPECT_LT(r.bin(0, 0), 0.5f);
    EXPECT_EQ(r.apu.setup_engine().voices_retired(), 0u);
    r.sched.advance_to(Rig::frame(38));
    EXPECT_EQ(r.apu.setup_engine().voices_retired(), 1u) << "the release ends after its 16 frames and the voice retires";
    EXPECT_FALSE(r.field(SynthRig::kVoice, SE::kFieldState) & SE::kStateActive);
}

TEST(ApuSynthesis, TheVoiceBufferTableIsWrittenThroughItsMethods) {
    using SE = apu::SetupEngine;
    SynthRig r;
    r.ram32(SynthRig::kSge, 0);  // empty, as after reset: the methods must fill it
    // Entry 0 names the page at 0x4000 (the offset word keeps bits 31:12 only); entry 3 another page.
    r.vp(SE::kMethodSetCurrentInbufSge, 0);
    r.vp(SE::kMethodSetCurrentInbufSgeOffset, SynthRig::kPage | 0x123u);
    r.vp(SE::kMethodSetCurrentInbufSge, 3);
    r.vp(SE::kMethodSetCurrentInbufSgeOffset, 0x6000);
    EXPECT_EQ(r.ram32(SynthRig::kSge), SynthRig::kPage);
    EXPECT_EQ(r.ram32(SynthRig::kSge + 3 * 8), 0x6000u);
    // A voice reading through entry 0 hears the page's samples.
    r.fill_pcm(64);
    r.start(kPcm16Mono, 63, 0, 0, 0);
    r.sched.advance_to(Rig::frame(1));
    EXPECT_FLOAT_EQ(r.bin(0, 7), SynthRig::pcm(7) / 32768.0f);
}

TEST(ApuSynthesis, TheLowPassFilterTakesTheTopOffAMonoVoiceInFilterMode1) {
    using SE = apu::SetupEngine;
    // A full-scale alternating signal (24 kHz, the highest the rate holds) through a voice, without and with
    // the filter: the cutoff word for about 1 kHz (4096 * log2(2 sin(pi 1000 / 48000)) = -14.5 octaves * 256).
    auto energy = [](bool filtered) {
        SynthRig r;
        for (u32 i = 0; i < 64; ++i) {
            const u16 v = (i & 1) ? 0xC000 : 0x4000;
            EXPECT_TRUE(r.bus.write_ram(SynthRig::kPage + i * 2, &v, 2));
        }
        r.vp(SE::kMethodSetCurrentVoice, SynthRig::kVoice);
        if (filtered) {
            r.vp(0x318, 1u << 16);                                   // CFG_MISC: FMODE 1
            const i16 fc = static_cast<i16>(std::lround(4096.0 * std::log2(2.0 * std::sin(std::numbers::pi * 1000.0 / 48000.0))));
            r.vp(0x374, (0x8000u << 16) | static_cast<u16>(fc));     // TAR_FCA: cutoff, no resonance
        }
        r.start(kPcm16Mono | SE::kFmtLoop, 63, 0, 0, 0);
        double e = 0;
        for (u64 f = 1; f <= 8; ++f) {
            r.sched.advance_to(Rig::frame(f));
            if (f > 4)  // past the filter's settling
                for (u32 i = 0; i < 32; ++i) e += static_cast<double>(r.bin(0, i)) * static_cast<double>(r.bin(0, i));
        }
        return e;
    };
    const double dry = energy(false), wet = energy(true);
    EXPECT_GT(dry, 0.0);
    EXPECT_LT(wet, dry * 0.01) << "a 24 kHz tone through a 1 kHz low-pass loses more than 20 dB";
}

TEST(ApuGp, TheMixBufferAnswersAtXC00AsWellAsX1400ForInstructionsAndDma) {
    Rig r;
    auto& gp = r.apu.gp();
    auto& x = gp.core().x_memory();
    x[Apu::kMixBufWord + 0x25] = 0x123456;  // bin 1, sample 5
    EXPECT_EQ(gp.core().read_x(0xC25), 0x123456u) << "the speaker bins as the boot program addresses them";
    gp.core().write_x(0xC80, 0x654321);        // bin 4, sample 0
    EXPECT_EQ(x[Apu::kMixBufWord + 0x80], 0x654321u);
    EXPECT_EQ(gp.dma().read_dsp(0xC25), 0x123456u) << "the DMA engine sees the same words";
    // Below the alias the GP's own X RAM is untouched, and the EP has no such window.
    gp.core().write_x(0xBFF, 0x111111);
    EXPECT_EQ(x[0xBFF], 0x111111u);
    EXPECT_NE(x[Apu::kMixBufWord + 0x3FF], 0x111111u);
    r.apu.ep().core().x_memory()[0x400 - 1] = 0;
    EXPECT_EQ(r.apu.ep().core().read_x(0xC25), 0u) << "the EP has 3K words and nothing at X:0xC00";
}

TEST(ApuFrames, TheFrameClockRunsOnlyInCountModeAndTheEpFlagRisesEveryEighthFrameOfTheCount) {
    // The host's start sequence: count mode off, XGSCNT zeroed, the processors released, count mode on.
    Rig r;
    r.wr(Apu::kSectl, 0x7);  // count mode (bits 4:3) off
    const u64 before = r.apu.frames();
    r.sched.advance_to(Rig::frame(20));
    EXPECT_EQ(r.apu.frames(), before) << "no frames while the count mode is off";
    r.wr(Apu::kXgsCnt, 0);
    EXPECT_EQ(r.rd(Apu::kXgsCnt), 0u);
    // An EP program that counts the start-of-frame flags it sees: JCLR #1,X:$FFFFC5,* (wait), MOVEP #2 to $FFFFC5
    // (clear), ADD #1,A, halt, BRA back to the wait.
    const u32 program[] = {0x0A8581, 0x000000,  // JCLR #1,X:$FFFFC5,$0 (wait for the flag)
                           0x08F485, 0x000002,  // MOVEP #2,X:$FFFFC5 (clear it)
                           0x014180,            // ADD #1,A
                           0x08F484, 0x000001,  // MOVEP #1,X:$FFFFC4 (halt until the next frame)
                           0x0C0000};           // JMP $0
    r.table(0x1000, {0x6000, 0x7000});  // the boot ROM reads 8 KiB
    for (u32 i = 0; i < 8; ++i) r.ram32(0x6000 + i * 4, program[i]);
    r.wr(Apu::kEpScratchTable, 0x1000);
    r.wr(Apu::kEpScratchMaxSge, 1);
    r.wr(Apu::kEpBlock + Apu::kReset, Apu::kResetBlock | Apu::kResetDsp);
    ASSERT_TRUE(r.apu.ep().running()) << r.apu.ep().last_problem();
    const Cycles start = r.sched.now();
    r.wr(Apu::kSectl, 0xF);  // count mode on: the first frame one period later
    r.sched.advance_to(start + Rig::frame(1) - 1);
    EXPECT_EQ(r.rd(Apu::kXgsCnt), 0u) << "no frame yet";
    r.sched.advance_to(start + Rig::frame(1));
    EXPECT_EQ(r.rd(Apu::kXgsCnt), 32u) << "32 samples a frame";
    r.sched.advance_to(start + Rig::frame(24));
    EXPECT_EQ(r.rd(Apu::kXgsCnt), 24u * 32);
    EXPECT_EQ(r.apu.ep().core().regs().a >> 24, 3) << "flags at counts 0, 8 and 16: three EP frames in 24";
}
