#include <gtest/gtest.h>

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
    Rig() { apu.config_write(0x10, 4, kBar); }
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
    r.wr(Apu::kVpBlock + 0x200, 1);
    EXPECT_EQ(r.rd(Apu::kVpBlock + 0x200), 1u) << "methods are kept until the voice processor is modelled";
    EXPECT_EQ(r.apu.reads(), 4u);
    EXPECT_EQ(r.apu.writes(), 3u);
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
