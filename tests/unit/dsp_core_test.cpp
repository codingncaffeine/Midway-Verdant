#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "apu/dsp/core.h"

using namespace mv;
using namespace mv::dsp;

namespace {

struct FakePeripherals : Peripherals {
    std::vector<std::pair<u32, u32>> writes;
    u32 read_value{0};
    u32 peripheral_read(u32) override { return read_value; }
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
