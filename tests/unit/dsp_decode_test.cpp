#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>

#include "apu/dsp/decode.h"
#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pgraph/replay.h"

using namespace mv;
using namespace mv::dsp;

TEST(DspDecode, KnownWordsFromTheManual) {
    // JSR with a 24-bit address extension: 0000 1011 1111 0000 1000 0000 = ea mode 110000 (absolute).
    Decoded j = decode(0x0BF080);
    EXPECT_EQ(mnemonic(j), "JSR");
    EXPECT_TRUE(j.extension);
    EXPECT_EQ(disassemble(j, 0xAC).find("JSR ea"), 0u) << disassemble(j, 0xAC);
    // A parallel move: MOVE #imm,X0 (01 00 0 100 W=1 1 110100).
    Decoded m = decode(0x44F400);
    EXPECT_EQ(m.move, Move::X);
    EXPECT_TRUE(m.extension);
    EXPECT_EQ(m.alu, Alu::None);
    EXPECT_EQ(disassemble(m, 0x4000), "MOVE  #$004000,X0");
    // MOVE X0,X:(R0)+
    Decoded s = decode(0x445800);
    EXPECT_FALSE(s.extension);
    EXPECT_EQ(disassemble(s, 0), "MOVE  X0,X:(R0)+");
    // NOP, RTS, and the no-move parallel form of CLR A (0010 0000 0000 0000 0001 0011).
    EXPECT_EQ(mnemonic(decode(0x000000)), "NOP");
    EXPECT_EQ(mnemonic(decode(0x00000C)), "RTS");
    Decoded c = decode(0x200013);
    EXPECT_EQ(c.move, Move::None);
    EXPECT_EQ(c.alu, Alu::Clr);
    EXPECT_EQ(disassemble(c, 0), "CLR A");
    // MAC X0,Y0,A with X:(R0)+,X0 Y:(R4)+,Y0: 1 w=1 mm=11 ee=00 ff=00 W=1 rr=00 MM=11 RRR=000 | 1QQQdkkk: QQQ=101? (Y0,X0) ...
    // The multiply byte 1 101 0 010 = 0xD2 means MAC Y0,X0,A (negate 0, mac 1, round 0).
    Decoded k = decode(0xF098D2);
    EXPECT_EQ(k.move, Move::XY);
    EXPECT_EQ(k.alu, Alu::Mac);
    EXPECT_EQ(k.alu_src, 5u);
    EXPECT_EQ(disassemble(k, 0), "MAC Y0,X0,A  X:(R0)+,X0 Y:(R4)+,Y0");
    // DO #count,expr has an extension word; REP #count has none.
    EXPECT_TRUE(decode(0x060080).extension);
    EXPECT_FALSE(decode(0x0600A0).extension);
    EXPECT_EQ(mnemonic(decode(0x0600A0)), "REP");
    // Bit test and jump: JCLR #n,X:ea,xxxx (0000 1010 01 MMMRRR 0S0bbbbb) carries an extension.
    Decoded jc = decode(0x0A6080);
    EXPECT_EQ(mnemonic(jc), "JCLR");
    EXPECT_TRUE(jc.extension);
    // MOVEP X:(R0),X:$FFFFD4-ish: 0000100s W1MMMRRR 1Spppppp
    Decoded mp = decode(0x08E0D4);
    EXPECT_EQ(mnemonic(mp), "MOVEP");
}

TEST(DspDecode, EveryWordOfTheKernelsGpProgramDecodes) {
    std::filesystem::path capture = std::filesystem::path(MV_SOURCE_DIR) / "scratch" / "boot-anim.mvcap";
    if (const char* env = std::getenv("MV_BOOT_CAPTURE")) capture = env;
    if (!std::filesystem::exists(capture)) GTEST_SKIP() << "no boot-animation capture at " << capture;
    pgraph::Replay r;
    std::string error;
    ASSERT_TRUE(r.open(capture, error)) << error;
    pgraph::Kelvin k;
    ASSERT_TRUE(r.run(k, nullptr, error, 0)) << error;
    // The kernel copies the GP program to scratch offset 0 (page 0x03994000 in this capture); the boot ROM DMAs it in.
    const u8* p = r.memory().at(0x03994000, 0x800 * 4);
    ASSERT_NE(p, nullptr);
    std::map<std::string, u32> histogram;
    u32 unknown = 0, instructions = 0, last_nonzero = 0;
    for (u32 i = 0; i < 0x800; ++i) {
        u32 w;
        std::memcpy(&w, p + i * 4, 4);
        if (w & 0xFFFFFFu) last_nonzero = i;
    }
    const auto listing_path = std::filesystem::path(MV_SOURCE_DIR) / "scratch" / "gp-program.lst";
    std::FILE* lst = std::fopen(listing_path.string().c_str(), "w");
    for (u32 i = 0; i <= last_nonzero;) {
        u32 w, ext = 0;
        std::memcpy(&w, p + i * 4, 4);
        const Decoded d = decode(w);
        if (d.extension && i + 1 <= last_nonzero) std::memcpy(&ext, p + (i + 1) * 4, 4);
        ++histogram[mnemonic(d)];
        ++instructions;
        if (d.pattern == -2) ++unknown;
        if (lst) std::fprintf(lst, "P:%04x  %06x%s  %s\n", i, w & 0xFFFFFFu, d.extension ? std::format(" {:06x}", ext & 0xFFFFFFu).c_str() : "       ", disassemble(d, ext).c_str());
        i += d.extension ? 2 : 1;
    }
    if (lst) std::fclose(lst);
    std::printf("GP program: %u words, %u instructions, %u unknown; listing in %s\n", last_nonzero + 1, instructions, unknown, listing_path.string().c_str());
    for (const auto& [m, n] : histogram) std::printf("  %-8s x%u\n", m.c_str(), n);
    EXPECT_EQ(unknown, 0u);
}

// Two ordinary parallel-move multiplies that damaged lines of the manual's text once listed as fixed encodings of
// their own; they must decode through the parallel-move tables like any other word.
TEST(DspDecode, ParallelMultipliesAreNotShadowedByDamagedTableRows) {
    const Decoded r = decode(0x98CA91);
    EXPECT_EQ(r.alu, Alu::Mpyr) << mnemonic(r);
    EXPECT_NE(r.move, Move::None);
    const Decoded m = decode(0x654880);
    EXPECT_EQ(m.alu, Alu::Mpy) << mnemonic(m);
    EXPECT_NE(m.move, Move::None);
}
