// Which instruction encodings the interpreted Pentium III takes and which it answers with an invalid-opcode
// exception: every one-byte and two-byte opcode under each mandatory prefix with every ModRM group value, and the
// whole x87 escape space, executed in flat protected mode at ring 0. The expected set is the processor's own
// instruction set (Intel's manual, Vol. 2: the opcode maps and each instruction's page; MMX and SSE, no SSE2).
#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "core/bus.h"
#include "cpu/p6/cpu.h"

using namespace mv;
using namespace mv::p6;

namespace {

constexpr u32 kCode = 0x1000, kGdt = 0x2000, kIdt = 0x3000, kHandlers = 0x4000, kData = 0x10000;

struct Machine {
    Bus bus{4u << 20};
    Cpu cpu{bus};
    Machine() {
        auto descriptor = [&](u32 at, u32 limit, u8 access, u8 flags) {
            const u8 d[8] = {static_cast<u8>(limit), static_cast<u8>(limit >> 8), 0, 0, 0, access, static_cast<u8>(((limit >> 16) & 0xF) | (flags << 4)), 0};
            bus.write_ram(at, d, 8);
        };
        descriptor(kGdt, 0, 0, 0);
        descriptor(kGdt + 8, 0xFFFFF, 0x9B, 0xC);
        descriptor(kGdt + 16, 0xFFFFF, 0x93, 0xC);
        for (u32 n = 0; n < 256; ++n) {
            const u32 target = kHandlers + n * 8;
            const u8 gate[8] = {static_cast<u8>(target), static_cast<u8>(target >> 8), 0x08, 0x00, 0x00, 0x8E, 0x00, 0x00};
            bus.write_ram(kIdt + n * 8, gate, 8);
        }
    }
    // One instruction from a fresh processor; the vector of the exception it raised, or -1.
    int run(const std::vector<u8>& bytes) {
        cpu.reset(0x68A);
        CpuState& s = cpu.state();
        s.cr0 |= cr0::PE;
        s.cr4 |= cr4::OSFXSR | cr4::OSXMMEXCPT;
        s.seg[kCs] = Segment{0x08, 0, 0xFFFFFFFFu, 0xC09B};
        for (unsigned g : {kSs, kDs, kEs, kFs, kGs}) s.seg[g] = Segment{0x10, 0, 0xFFFFFFFFu, 0xC093};
        s.gdtr = {kGdt, 0x17};
        s.idtr = {kIdt, 0x7FF};
        for (u32& r : s.gpr) r = kData;
        s.gpr[kEcx] = 1;
        s.gpr[kEsp] = 0x8000;
        s.eip = kCode;
        std::vector<u8> code = bytes;
        code.resize(16, 0x00);
        bus.write_ram(kCode, code.data(), code.size());
        cpu.step();
        const u32 eip = cpu.state().eip;
        if (eip >= kHandlers && eip < kHandlers + 256 * 8 && (eip - kHandlers) % 8 == 0) return static_cast<int>((eip - kHandlers) / 8);
        return -1;
    }
};

// The Pentium III's instruction set as validity rules (true = the processor has the encoding).
bool one_byte_valid(unsigned op, bool mem, unsigned reg) {
    switch (op) {
        case 0x62: case 0x8D: case 0xC4: case 0xC5: return mem;  // BOUND, LEA, LES, LDS: a memory operand
        case 0x8C: return reg <= 5;                              // six segment registers
        case 0x8E: return reg <= 5 && reg != 1;                  // and CS is not loaded by MOV
        case 0x8F: case 0xC6: case 0xC7: return reg == 0;        // POP Ev, MOV Eb/Ev,imm: one member each
        case 0xFE: return reg <= 1;
        case 0xFF: return reg != 7 && (mem || (reg != 3 && reg != 5));  // far CALL/JMP take a memory pointer
        default: return true;
    }
}

bool two_byte_valid(unsigned op, bool mem, unsigned reg) {
    auto in = [&](unsigned lo, unsigned hi) { return op >= lo && op <= hi; };
    switch (op) {
        case 0x00: return reg <= 5;
        case 0x01: return mem ? reg != 5 : (reg == 4 || reg == 6);
        case 0x13: case 0x17: case 0x2B: case 0xE7: case 0xB2: case 0xB4: case 0xB5: return mem;
        case 0x50: case 0xC5: case 0xD7: case 0xF7: return !mem;
        case 0x20: case 0x22: return reg == 0 || reg == 2 || reg == 3 || reg == 4;
        case 0x71: case 0x72: return !mem && (reg == 2 || reg == 4 || reg == 6);
        case 0x73: return !mem && (reg == 2 || reg == 6);
        case 0xAE: return mem ? reg <= 3 : reg == 7;  // FXSAVE FXRSTOR LDMXCSR STMXCSR; SFENCE
        case 0xBA: return reg >= 4;
        case 0xC7: return mem && reg == 1;  // CMPXCHG8B
        default: break;
    }
    return op == 0x02 || op == 0x03 || op == 0x06 || op == 0x08 || op == 0x09 || in(0x10, 0x1F) || op == 0x21 || op == 0x23 || in(0x28, 0x2F) ||
           in(0x30, 0x35) || in(0x40, 0x4F) || in(0x51, 0x59) || in(0x5C, 0x5F) || in(0x60, 0x6B) || op == 0x6E || op == 0x6F || op == 0x70 ||
           in(0x74, 0x77) || op == 0x7E || op == 0x7F || in(0x80, 0x9F) || in(0xA0, 0xA5) || op == 0xA8 || op == 0xA9 || in(0xAB, 0xAD) || op == 0xAF ||
           op == 0xB0 || op == 0xB1 || op == 0xB3 || op == 0xB6 || op == 0xB7 || in(0xBB, 0xBF) || in(0xC0, 0xC2) || op == 0xC4 || op == 0xC6 ||
           in(0xC8, 0xCF) || in(0xD1, 0xD3) || op == 0xD5 || in(0xD8, 0xDF) || in(0xE0, 0xE5) || in(0xE8, 0xEF) || in(0xF1, 0xF3) || op == 0xF5 ||
           op == 0xF6 || in(0xF8, 0xFA) || in(0xFC, 0xFE);
}

bool x87_valid(unsigned op, unsigned modrm) {
    const unsigned reg = (modrm >> 3) & 7u;
    if (modrm < 0xC0) {
        switch (op) {
            case 0xD9: case 0xDF: return reg != 1;
            case 0xDB: return reg == 0 || reg == 2 || reg == 3 || reg == 5 || reg == 7;
            case 0xDD: return reg != 1 && reg != 5;
            default: return true;
        }
    }
    auto in = [&](unsigned lo, unsigned hi) { return modrm >= lo && modrm <= hi; };
    switch (op) {
        case 0xD9: return !(in(0xD1, 0xD7) || modrm == 0xE2 || modrm == 0xE3 || modrm == 0xE6 || modrm == 0xE7 || modrm == 0xEF);
        case 0xDA: return in(0xC0, 0xDF) || modrm == 0xE9;
        case 0xDB: return in(0xC0, 0xE4) || in(0xE8, 0xF7);
        case 0xDD: return in(0xC0, 0xEF);
        case 0xDE: return !(modrm == 0xD8 || in(0xDA, 0xDF));
        case 0xDF: return in(0xC0, 0xE0) || in(0xE8, 0xF7);
        default: return true;  // D8 and DC: every register form
    }
}

}  // namespace

TEST(P6Coverage, TheOneByteMapIsThePentiumIIIs) {
    Machine m;
    for (unsigned op = 0; op < 256; ++op) {
        if (op == 0x0F || op == 0x26 || op == 0x2E || op == 0x36 || op == 0x3E || (op >= 0x64 && op <= 0x67) || op == 0xF0 || op == 0xF2 || op == 0xF3 || (op >= 0xD8 && op <= 0xDF)) continue;
        for (unsigned mod = 0; mod < 4; mod += 3)
            for (unsigned reg = 0; reg < 8; ++reg) {
                const bool ud = m.run({static_cast<u8>(op), static_cast<u8>((mod << 6) | (reg << 3) | (mod == 3 ? 1 : 0))}) == 6;
                EXPECT_EQ(!ud, one_byte_valid(op, mod != 3, reg)) << std::hex << "opcode " << op << (mod == 3 ? " register form /" : " memory form /") << reg;
            }
    }
}

TEST(P6Coverage, TheTwoByteMapIsThePentiumIIIs) {
    Machine m;
    for (unsigned op = 0; op < 256; ++op) {
        // 0F 0D (another vendor's prefetch) is left out: the sources disagree on what this processor does with it.
        if (op == 0x0D) continue;
        for (unsigned mod = 0; mod < 4; mod += 3)
            for (unsigned reg = 0; reg < 8; ++reg) {
                const bool ud = m.run({0x0F, static_cast<u8>(op), static_cast<u8>((mod << 6) | (reg << 3) | (mod == 3 ? 1 : 0))}) == 6;
                EXPECT_EQ(!ud, two_byte_valid(op, mod != 3, reg)) << std::hex << "opcode 0F " << op << (mod == 3 ? " register form /" : " memory form /") << reg;
            }
    }
    // The scalar single-precision forms behind F3, and none of the SSE2 forms behind 66 or F2.
    for (unsigned op : {0x10u, 0x11u, 0x2Au, 0x2Cu, 0x2Du, 0x51u, 0x52u, 0x53u, 0x58u, 0x59u, 0x5Cu, 0x5Du, 0x5Eu, 0x5Fu, 0xC2u}) {
        EXPECT_NE(m.run({0xF3, 0x0F, static_cast<u8>(op), 0xC1}), 6) << std::hex << "F3 0F " << op;
        EXPECT_EQ(m.run({0xF2, 0x0F, static_cast<u8>(op), 0xC1}), 6) << std::hex << "F2 0F " << op;
    }
    for (unsigned op : {0x10u, 0x28u, 0x54u, 0x58u, 0x6Fu, 0x7Fu, 0xD4u, 0xEFu, 0xFBu}) EXPECT_EQ(m.run({0x66, 0x0F, static_cast<u8>(op), 0xC1}), 6) << std::hex << "66 0F " << op;
}

TEST(P6Coverage, TheX87EscapesAreThePentiumIIIs) {
    Machine m;
    for (unsigned op = 0xD8; op <= 0xDF; ++op)
        for (unsigned modrm = 0; modrm < 256; ++modrm) {
            if (modrm < 0xC0 && (modrm & 0xC7) != 0x00) continue;  // a memory form depends on its reg field only
            const bool ud = m.run({static_cast<u8>(op), static_cast<u8>(modrm)}) == 6;
            EXPECT_EQ(!ud, x87_valid(op, modrm)) << std::hex << "escape " << op << " modrm " << modrm;
        }
}

// The reserved second encodings do what the documented ones do.
TEST(P6Coverage, X87SecondEncodingsRunAsTheDocumentedForms) {
    Machine m;
    auto run2 = [&](std::initializer_list<u8> second) {
        // fninit (every exception masked, as programs run); fld1; fldz; then the instruction under test: ST(0) = 0,
        // ST(1) = 1 before it.
        std::vector<u8> code{0xDB, 0xE3, 0xD9, 0xE8, 0xD9, 0xEE};
        code.insert(code.end(), second);
        m.run(code);
        m.cpu.step();
        m.cpu.step();
        m.cpu.step();
    };
    run2({0xDD, 0xC9});  // FXCH ST(1) by its second encoding
    EXPECT_EQ(m.cpu.state().fpu.st[m.cpu.state().fpu.top()], 1.0L);
    run2({0xDF, 0xC9});
    EXPECT_EQ(m.cpu.state().fpu.st[m.cpu.state().fpu.top()], 1.0L);
    run2({0xD9, 0xD9});  // FSTP ST(1): the zero replaces the one, and is then the top
    EXPECT_EQ(m.cpu.state().fpu.st[m.cpu.state().fpu.top()], 0.0L);
    EXPECT_EQ(m.cpu.state().fpu.top(), 7u);
    run2({0xDF, 0xD1});
    EXPECT_EQ(m.cpu.state().fpu.top(), 7u);
    run2({0xDF, 0xD9});
    EXPECT_EQ(m.cpu.state().fpu.top(), 7u);
    run2({0xDE, 0xD1});  // FCOMP ST(1): 0 < 1 sets C0 alone, one pop
    EXPECT_EQ(m.cpu.state().fpu.fsw & 0x4500u, 0x0100u);
    EXPECT_EQ(m.cpu.state().fpu.top(), 7u);
}
