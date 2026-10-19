#include <gtest/gtest.h>

#include <cstring>
#include <format>
#include <string>
#include <vector>

#include "apu/dsp/core.h"

using namespace mv;
using namespace mv::dsp;

namespace {

struct FakePeripherals : Peripherals {
    std::vector<std::pair<u32, u32>> writes;
    u32 read_value{0};
    bool frame_static{false};
    // A DMA stand-in: bit 7 reads set from core cycle `eol_at` on (when `core` is set); `until` is what the core is told.
    const Core* core{nullptr};
    u64 eol_at{0}, until{0};
    u64 reads{0};
    bool static_until_next_frame(u32, u32) const override { return frame_static; }
    u64 static_until_cycle(u32, u32 bit) const override { return bit == 7 ? until : 0; }
    u32 peripheral_read(u32) override {
        ++reads;
        return read_value | (core && core->cycles() >= eol_at ? 0x80u : 0u);
    }
    void peripheral_write(u32 address, u32 value) override { writes.emplace_back(address, value); }
};

struct Machine {
    FakePeripherals periph;
    Core core{Core::Config{}, &periph};
    u32 at{0};
    Machine() { core.reset(); }
    void emit(u32 word) { core.write_p(at++, word); }
    void emit(u32 word, u32 ext) {
        core.write_p(at++, word);
        core.write_p(at++, ext);
    }
    void run(u32 steps) {
        for (u32 i = 0; i < steps && !core.error(); ++i) core.step();
        ASSERT_TRUE(!core.error()) << core.error_text();
    }
};

}  // namespace

TEST(DspCore, ImmediateMovesAndStoresWithPostIncrement) {
    Machine m;
    m.emit(0x300000);          // MOVE #0,R0
    m.emit(0x44F400, 0x123456); // MOVE #$123456,X0
    m.emit(0x445800);          // MOVE X0,X:(R0)+
    m.emit(0x44F400, 0x800000);
    m.emit(0x445800);
    m.run(5);
    EXPECT_EQ(m.core.read_x(0), 0x123456u);
    EXPECT_EQ(m.core.read_x(1), 0x800000u);
    EXPECT_EQ(m.core.regs().r[0], 2u);
    EXPECT_EQ(m.core.regs().pc, 7u);
}

TEST(DspCore, FractionalMultiplyAccumulateAndRounding) {
    Machine m;
    // X0 = 0.5 ($400000), Y0 = 0.5: MPY X0,Y0,A -> A = 0.25 = $00 200000 000000; MAC adds another 0.25.
    m.emit(0x44F400, 0x400000);  // MOVE #$400000,X0
    m.emit(0x46F400, 0x400000);  // MOVE #$400000,Y0
    m.emit(0x2000D0);            // MPY X0,Y0,A  (1 101 0 000: QQQ=101 Y0,X0; d=0; k=000)
    m.emit(0x2000D2);            // MAC X0,Y0,A
    m.run(4);
    EXPECT_EQ(m.core.regs().a, i64{0x00400000000000});
    // A1 read into X1 is the upper word: 0.5.
    m.emit(0x21C500);  // hmm: MOVE A,X1 — register move eeeee=01110 (A) ddddd=00101 (X1): 0010 00ee eeed dddd
    m.run(1);
    EXPECT_EQ(m.core.regs().x1, 0x400000u);
    // Rounding: A = $00 000000 800001 (just above a half in A0) rounds A1 up; exactly a half with an even A1 rounds down.
    m.core.regs().a = 0x000000800001;
    m.emit(0x200011);  // RND A (JJJ=001 kkk=001, D=0)
    m.run(1);
    EXPECT_EQ(m.core.regs().a, i64{0x000001000000});
    m.core.regs().a = 0x000002800000;
    m.emit(0x200011);
    m.run(1);
    EXPECT_EQ(m.core.regs().a, i64{0x000002000000}) << "convergent: even stays";
    m.core.regs().a = 0x000003800000;
    m.emit(0x200011);
    m.run(1);
    EXPECT_EQ(m.core.regs().a, i64{0x000004000000}) << "convergent: odd rounds up";
}

TEST(DspCore, LimiterSaturatesAccumulatorReads) {
    Machine m;
    m.core.regs().a = i64{0x01000000000000};  // 2.0: not representable in 24 bits
    m.emit(0x21C400);                         // MOVE A,X0
    m.run(1);
    EXPECT_EQ(m.core.regs().x0, 0x7FFFFFu);
    EXPECT_TRUE(m.core.regs().sr & Core::L);
    m.core.regs().b = Core::Regs{}.b - i64{0x01000000000000};  // -2.0 in B
    m.emit(0x21E500);  // MOVE B,X1 (eeeee=01111 B, ddddd=00101 X1)
    m.run(1);
    EXPECT_EQ(m.core.regs().x1, 0x800000u);
}

TEST(DspCore, HardwareLoopRepeatsAndRestoresState) {
    Machine m;
    // R0 = 0; DO #4, end ; MOVE X0,X:(R0)+ ; end: NOP
    m.emit(0x300000);            // MOVE #0,R0
    m.emit(0x44F400, 0x000007);  // MOVE #7,X0
    m.emit(0x060480, 0x000006);  // DO #4,$0006 (LA = expr - 1 = 5: the one-word MOVE below is the last instruction)
    m.emit(0x445800);            // P:5  MOVE X0,X:(R0)+
    m.emit(0x000000);            // P:6  NOP
    m.run(8);  // two moves, DO, four passes, NOP
    EXPECT_EQ(m.core.regs().r[0], 4u);
    EXPECT_EQ(m.core.read_x(3), 7u);
    EXPECT_EQ(m.core.read_x(4), 0u);
    EXPECT_EQ(m.core.regs().pc, 7u);
    EXPECT_FALSE(m.core.regs().sr & Core::LF) << "the loop flag is restored at the end";
    EXPECT_EQ(m.core.regs().sp, 0u);
}

TEST(DspCore, BranchesAreRelativeToTheInstruction) {
    Machine m;
    // P:0 BRA +3 -> P:3; P:1 NOP; P:2 NOP; P:3 BSR +2 -> P:5 (RTS); back at P:4: MOVE #1,R1
    m.emit(0x050C03);  // BRA xxx: 0000 0101 0000 11 aaaa 0 aaaaa with displacement 3 (aaaa=0000, aaaaa=00011)
    m.emit(0x000000);
    m.emit(0x000000);
    m.emit(0x050802);  // BSR xxx displacement 2 -> P:5
    m.emit(0x310100);  // P:4 MOVE #1,R1 (immediate short in bits 15-8)
    m.emit(0x00000C);  // P:5 RTS
    m.run(4);
    EXPECT_EQ(m.core.regs().pc, 5u);
    EXPECT_EQ(m.core.regs().r[1], 1u);
}

TEST(DspCore, PeripheralMovesAndBitBranches) {
    Machine m;
    // MOVEP #1,X:$FFFFC4 (08F484 000001); then BRCLR #7,X:$FFFFC5,back until the bit is set.
    m.emit(0x08F484, 0x000001);
    m.emit(0x0CC507, 0x000000);  // P:2 BRCLR #7,X:$FFFFC5,+0 -> branch to itself while clear (as the kernel's program does)
    m.emit(0x000000);            // P:4 NOP
    m.run(1);
    ASSERT_EQ(m.periph.writes.size(), 1u);
    EXPECT_EQ(m.periph.writes[0].first, 0xFFFFC4u);
    EXPECT_EQ(m.periph.writes[0].second, 1u);
    m.periph.read_value = 0;
    m.run(3);
    EXPECT_EQ(m.core.regs().pc, 2u) << "still waiting";
    m.periph.read_value = 0x80;
    m.run(1);
    EXPECT_EQ(m.core.regs().pc, 4u);
}

TEST(DspCore, ModuloAddressingWraps) {
    Machine m;
    m.core.regs().r[2] = 0x100;
    m.core.regs().m[2] = 3;  // modulo 4 buffer at $100
    m.core.regs().x0 = 0xAB;
    for (int i = 0; i < 5; ++i) m.emit(0x445A00);  // MOVE X0,X:(R2)+
    m.run(5);
    EXPECT_EQ(m.core.regs().r[2], 0x101u) << "five increments in a modulo-4 buffer";
    EXPECT_EQ(m.core.read_x(0x103), 0xABu);
}

TEST(DspCore, UnknownInstructionStopsWithAMessage) {
    Machine m;
    m.emit(0x000005);  // ILLEGAL
    m.core.step();
    EXPECT_TRUE(m.core.error());
    EXPECT_NE(m.core.error_text().find("ILLEGAL"), std::string::npos) << m.core.error_text();
}

TEST(DspCore, RegisterFormPeripheralMovesWriteWithWSetAndReadWithWClear) {
    Machine m;
    // MOVEP A,X:$FFFFD4 (08CE14: W=1, d=A, p=$14) writes the peripheral; MOVEP X:$FFFFD4,A (084E14: W=0) reads it.
    m.core.regs().a = i64{0x00004000000000};  // $4000 in A1, the word a register move carries
    m.emit(0x08CE14);
    m.run(1);
    ASSERT_EQ(m.periph.writes.size(), 1u) << "W = 1 moves the register to the peripheral";
    EXPECT_EQ(m.periph.writes[0].first, 0xFFFFD4u);
    EXPECT_EQ(m.periph.writes[0].second, 0x4000u);
    m.periph.read_value = 0x123456;
    m.emit(0x084E14);
    m.run(1);
    EXPECT_EQ(m.periph.writes.size(), 1u) << "W = 0 does not write";
    EXPECT_EQ(m.core.regs().a, i64{0x00123456000000}) << "the peripheral word lands in A1";
}

TEST(DspCore, CountLeadingBitsThenNormaliseFastAsTheManualShows) {
    Machine m;
    m.core.regs().a = i64{0x20} << 48;  // A = $20:000000:000000
    m.emit(0x0C1E01);                   // CLB A,B: two leading zeros -> 9 - 2 = 7 in B1
    m.emit(0x0C1E26);                   // NORMF B1,A: positive count -> A shifted right 7
    m.run(1);
    EXPECT_EQ(m.core.regs().b, i64{7} << 24);
    m.run(1);
    EXPECT_EQ(m.core.regs().a, i64{0x400000} << 24);
}

TEST(DspCore, UnsignedCompareSeesTopBitAsMagnitude) {
    Machine m;
    m.core.regs().a = i64{0x800000} << 24;  // A1 = $800000: negative signed, large unsigned
    m.core.regs().x0 = 0x7FFFFF;
    m.emit(0x0C1FF8);  // CMPU X0,A
    m.run(1);
    EXPECT_EQ(m.core.regs().sr & (Core::C | Core::Z), 0u);  // $800000 > $7FFFFF unsigned
    m.core.regs().x0 = 0x900000;
    m.emit(0x0C1FF8);
    m.run(1);
    EXPECT_EQ(m.core.regs().sr & Core::C, static_cast<u32>(Core::C));  // borrow
    m.core.regs().x0 = 0x800000;
    m.emit(0x0C1FF8);
    m.run(1);
    EXPECT_EQ(m.core.regs().sr & Core::Z, static_cast<u32>(Core::Z));
}

TEST(DspCore, TwentyFourDivideStepsGiveTheFractionalQuotient) {
    Machine m;
    m.core.regs().a = i64{0x200000} << 24;  // 0.25
    m.core.regs().x0 = 0x400000;            // 0.5
    for (int i = 0; i < 24; ++i) m.emit(0x018040);  // DIV X0,A
    m.run(24);
    EXPECT_EQ(static_cast<u32>(m.core.regs().a) & 0xFFFFFFu, 0x400000u);  // 0.5 in A0
}

TEST(DspCore, BitSetAboveBitFifteenOnARegister) {
    Machine m;
    m.emit(0x0AC474);  // BSET #20,X0 (the 5-bit bit number)
    m.run(1);
    EXPECT_EQ(m.core.regs().x0, 0x100000u);
}

TEST(DspCore, ExtractSignExtendsExtractuDoesNotInsertPlaces) {
    Machine m;
    m.core.regs().a = i64{0xABCDEF} << 24;
    m.emit(0x0C1881, 0x00801C);  // EXTRACTU #CO,A,B: width 8 offset 28 -> $DE
    m.emit(0x0C1801, 0x00801C);  // EXTRACT  #CO,A,B: the same field sign-extended
    m.run(1);
    EXPECT_EQ(m.core.regs().b, i64{0xDE});
    m.run(1);
    EXPECT_EQ(m.core.regs().b, i64{0xDE} - 0x100);
    m.core.regs().a = 0;
    m.core.regs().x0 = 0x0000AB;
    m.emit(0x0C1940, 0x008004);  // INSERT #CO,X0,A: width 8 offset 4
    m.run(1);
    EXPECT_EQ(m.core.regs().a, i64{0xAB0});
}

TEST(DspCore, MultiplyByPowerOfTwoAndUnsignedSquare) {
    Machine m;
    m.core.regs().x0 = 0x400000;  // 0.5
    m.emit(0x0101D0);             // MPY X0,#1,A: 0.5 * 2^-1
    m.emit(0x0110D0);             // MPY X0,#16,A: the fifth shift bit
    m.run(1);
    EXPECT_EQ(m.core.regs().a, i64{0x200000} << 24);
    m.run(1);
    EXPECT_EQ(m.core.regs().a, (i64{0x400000} << 24) >> 16);
    m.core.regs().x0 = 0xFFFFFF;
    m.emit(0x0127C0);  // MPY uu X0,X0,A: $FFFFFF unsigned squared (signed it would be -1 * -1)
    m.run(1);
    EXPECT_EQ(m.core.regs().a, (i64{0xFFFFFF} * 0xFFFFFF) << 1);
}

TEST(DspCore, TestWithLongImmediateMoveSetsZeroForTheBranch) {
    // An audio program's state check: MOVEM P:$0001,A; TST A #$000BC0,R0; BNE +9. P:1 holds zero, so the branch
    // falls through and R0 takes the immediate.
    Machine m;
    m.core.write_p(1, 0);
    m.at = 0x10;
    m.emit(0x07F08E, 0x000001);  // MOVEM P:$0001,A
    m.emit(0x60F403, 0x000BC0);  // TST A #$000BC0,R0
    m.emit(0x052409);            // BNE *+9
    m.core.regs().pc = 0x10;
    m.core.regs().a = 0x00000001000000;  // as the live program has it: A = 1, carry set
    m.core.regs().sr = 0x311;
    m.core.regs().b = 0x00000001000000;  // B non-zero, so testing the wrong accumulator shows
    m.run(3);
    EXPECT_EQ(m.core.regs().r[0], 0xBC0u);
    EXPECT_EQ(m.core.regs().pc, 0x15u);
}

TEST(DspCore, CarryFollowsEachInstructionsRule) {
    constexpr u32 kCarry = 1u << 0, kOverflow = 1u << 1, kZero = 1u << 2;
    // CMP #xx computes the borrow: A = 1, CMP #2 borrows (C set); CMP #0 does not.
    {
        Machine m;
        m.core.regs().a = 0x00000001000000;
        m.emit(0x014285);  // CMP #2,A
        m.run(1);
        EXPECT_TRUE(m.core.regs().sr & kCarry);
        m.emit(0x014085);  // CMP #0,A
        m.run(1);
        EXPECT_FALSE(m.core.regs().sr & kCarry);
    }
    // TFR changes no condition code; MPY and CLR keep C.
    {
        Machine m;
        m.core.regs().sr = kCarry | kZero;
        m.core.regs().b = 0x00000005000000;
        m.emit(0x200001);  // TFR B,A
        m.run(1);
        EXPECT_EQ(m.core.regs().a, 0x00000005000000);
        EXPECT_EQ(m.core.regs().sr & (kCarry | kZero), kCarry | kZero);
        m.emit(0x2000D0);  // MPY X0,Y0,A
        m.emit(0x200013);  // CLR A
        m.run(2);
        EXPECT_TRUE(m.core.regs().sr & kCarry);
    }
    // INC carries out of bit 55; ASL by 4 takes C from the last bit out and V from a sign change.
    {
        Machine m;
        m.core.regs().a = -1;  // $FF FFFFFF FFFFFF
        m.emit(0x000008);      // INC A
        m.run(1);
        EXPECT_EQ(m.core.regs().a, 0);
        EXPECT_TRUE(m.core.regs().sr & kCarry);
        m.core.regs().a = 0x08000000000000;  // bit 51 set: ASL #4 moves it to bit 55
        m.emit(0x0C1D08);                    // ASL #4,A,A
        m.run(1);
        EXPECT_FALSE(m.core.regs().sr & kCarry);
        EXPECT_TRUE(m.core.regs().sr & kOverflow);
    }
}

TEST(DspCore, AWaitOnAFrameStaticBitSkipsToTheBudgetsEndWithTheSameResult) {
    // An EP's idle loop: CLR A; ADD #1,A; JCLR #1,X:$FFFFC5,$0001 — counting while it waits for the frame flag.
    // Run it with the skip and without; every register and both counters must match, for several budgets.
    for (u64 budget : {7u, 8u, 100u, 101u, 106'666u, 853'333u}) {  // some end mid-instruction
        Machine with, without;
        for (Machine* m : {&with, &without}) {
            m->emit(0x200013);            // CLR A
            m->emit(0x014180);            // ADD #1,A
            m->emit(0x0A8581, 0x000001);  // JCLR #1,X:$FFFFC5,$0001
        }
        with.periph.frame_static = true;
        with.core.run(budget);
        without.core.run(budget);
        EXPECT_EQ(with.core.regs().a, without.core.regs().a) << budget;
        EXPECT_EQ(with.core.regs().pc, without.core.regs().pc) << budget;
        EXPECT_EQ(with.core.regs().sr, without.core.regs().sr) << budget;
        EXPECT_EQ(with.core.cycles(), without.core.cycles()) << budget;
        EXPECT_EQ(with.core.instructions(), without.core.instructions()) << budget;
    }
    // A jump that waits on itself skips the same way.
    Machine self, plain;
    for (Machine* m : {&self, &plain}) m->emit(0x0A8581, 0x000000);  // JCLR #1,X:$FFFFC5,$0000
    self.periph.frame_static = true;
    self.core.run(50'000);
    plain.core.run(50'000);
    EXPECT_EQ(self.core.cycles(), plain.core.cycles());
    EXPECT_EQ(self.core.regs().pc, plain.core.regs().pc);
    EXPECT_EQ(self.core.instructions(), plain.core.instructions());
}

TEST(DspCore, APollOnARunningDmaSkipsToItsEndWithTheSameResult) {
    // A processor waiting for its DMA list's end: BRCLR #7,X:$FFFFC5,* alone, and counting (CLR A; ADD #1,A; BRCLR
    // back to the ADD). Told when the bit will rise, the core skips the polls before it; every register, both
    // counters and the instruction the loop leaves from must match a run that polls every time.
    for (u64 eol_at : {3u, 10u, 11u, 12u, 1001u, 50'000u}) {
        for (bool counting : {false, true}) {
            Machine with, without;
            for (Machine* m : {&with, &without}) {
                m->periph.core = &m->core;
                m->periph.eol_at = eol_at;
                if (counting) {
                    m->emit(0x200013);            // CLR A
                    m->emit(0x014180);            // ADD #1,A
                    m->emit(0x0CC507, 0xFFFFFF);  // BRCLR #7,X:$FFFFC5,-1 (to the ADD)
                } else {
                    m->emit(0x0CC507, 0x000000);  // BRCLR #7,X:$FFFFC5,* (to itself)
                }
                m->emit(0x014180);  // ADD #1,A once the wait is over, then NOPs
            }
            with.periph.until = eol_at;
            with.core.run(60'000);
            without.core.run(60'000);
            const std::string at = std::format("eol at {} counting {}", eol_at, counting);
            EXPECT_EQ(with.core.regs().a, without.core.regs().a) << at;
            EXPECT_EQ(with.core.regs().pc, without.core.regs().pc) << at;
            EXPECT_EQ(with.core.regs().sr, without.core.regs().sr) << at;
            EXPECT_EQ(with.core.cycles(), without.core.cycles()) << at;
            EXPECT_EQ(with.core.instructions(), without.core.instructions()) << at;
            if (eol_at >= 1001) {
                EXPECT_LT(with.periph.reads * 10, without.periph.reads) << at << ": the polls were skipped";
            }
        }
    }
}

TEST(DspCore, TheCompiledPathMatchesTheReferencePathOnRandomInstructionsAndStates) {
    // Every instruction word the decoder knows, on random register and memory states: one step through the compiled
    // form (fields packed into slots, a handler per move kind or operation class) and one through the full decoded
    // form must leave the whole state identical. Small memories keep the snapshots cheap; addresses past them read 0.
    const dsp::Core::Config cfg{256, 256, 64};
    dsp::Core fast(cfg, nullptr), ref(cfg, nullptr);
    ref.set_reference(true);
    u64 seed = 0x9E3779B97F4A7C15ull;
    auto next = [&] {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    u32 compared = 0;
    for (int i = 0; i < 200000; ++i) {
        dsp::Core& base = fast;
        base.reset();
        for (u32& w : base.x_memory()) w = static_cast<u32>(next()) & 0xFFFFFFu;
        for (u32& w : base.y_memory()) w = static_cast<u32>(next()) & 0xFFFFFFu;
        auto& r = base.regs();
        r.a = static_cast<i64>(next() << 8) >> 8;
        r.b = static_cast<i64>(next() << 8) >> 8;
        r.x0 = static_cast<u32>(next()) & 0xFFFFFFu;
        r.x1 = static_cast<u32>(next()) & 0xFFFFFFu;
        r.y0 = static_cast<u32>(next()) & 0xFFFFFFu;
        r.y1 = static_cast<u32>(next()) & 0xFFFFFFu;
        for (u32 k = 0; k < 8; ++k) {
            r.r[k] = static_cast<u32>(next()) & 0x1FFu;  // mostly inside the memories
            r.n[k] = static_cast<u32>(next()) & 0x1Fu;
            r.m[k] = (next() & 3u) ? 0xFFFFFFu : static_cast<u32>(next()) & 0xFFFFu;  // linear, modulo, reverse carry
        }
        r.sr = static_cast<u32>(next()) & 0xFFu;  // the condition codes
        r.pc = 8;
        base.write_p(8, static_cast<u32>(next()) & 0xFFFFFFu);
        base.write_p(9, static_cast<u32>(next()) & 0xFFFFFFu);
        const std::vector<u32> start = base.snapshot();
        ASSERT_TRUE(ref.restore(start));
        fast.run(1);
        ref.run(1);
        const auto a = fast.snapshot(), b = ref.snapshot();
        if (a != b || fast.error_text() != ref.error_text()) {
            std::size_t w = 0;
            while (w < a.size() && a[w] == b[w]) ++w;
            FAIL() << "word " << std::hex << start[6 + 256 + 256 + 8] << " ext " << start[6 + 256 + 256 + 9] << ": state word " << std::dec << w
                   << " differs (compiled " << std::hex << a[w] << ", reference " << b[w] << "); errors '" << fast.error_text() << "' / '" << ref.error_text() << "'";
        }
        ++compared;
    }
    EXPECT_EQ(compared, 200000u);
}
