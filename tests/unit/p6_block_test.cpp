// The block engine (D26): blocks executed through the interpreter's own `execute` must leave the same machine as the
// per-instruction step — on generated programs (integer forms, memory through a base register, short branches),
// on code that rewrites itself inside the block that runs it, and on code a device rewrites. The block schedule's
// rules are pinned on asymmetric cases.
#include <gtest/gtest.h>

#include <cstring>
#include <initializer_list>
#include <vector>

#include "core/bus.h"
#include "cpu/p6/block.h"
#include "cpu/p6/cpu.h"
#include "cpu/p6/timing.h"

using namespace mv;
using namespace mv::p6;

namespace {

constexpr u32 kCode = 0x1000, kData = 0x20000, kStack = 0x8000;

struct Rig {
    Bus bus{4u << 20};
    Cpu cpu{bus};
    explicit Rig(bool paged = false) {
        cpu.reset(0x68A);
        // Big real mode with flat 32-bit code, stack and data (as the P6 unit tests), code at 0x1000.
        cpu.state().seg[kCs] = Segment{0, 0, 0xFFFFFFFFu, 0xC09B};
        cpu.state().seg[kSs] = Segment{0, 0, 0xFFFFFFFFu, 0xC093};
        cpu.state().seg[kDs] = cpu.state().seg[kEs] = Segment{0, 0, 0xFFFFFFFFu, 0xC093};
        cpu.state().eip = kCode;
        cpu.state().gpr[kEsp] = kStack;
        if (paged) {
            // Protected mode with paging: one 4 MiB page maps the RAM 1:1 (PSE), flat ring-0 segments as the
            // kernel runs (the recompiler runs only here).
            constexpr u32 kPageDir = 0x100000;
            const u32 pde = 0x83;  // present, writable, 4 MiB
            std::memcpy(bus.ram() + kPageDir, &pde, 4);
            cpu.state().cr3 = kPageDir;
            cpu.state().cr4 |= cr4::PSE;
            cpu.state().cr0 |= cr0::PE | cr0::PG;
            cpu.state().seg[kCs].selector = 0x08;
            cpu.state().seg[kSs].selector = cpu.state().seg[kDs].selector = cpu.state().seg[kEs].selector = 0x10;
            cpu.memory().flush_tlb(false);
        }
    }
    void load(const std::vector<u8>& code, u32 at = kCode) { std::memcpy(bus.ram() + at, code.data(), code.size()); }
};

// Run to the program's HLT (or a step budget) one instruction at a time.
void run_steps(Rig& r) {
    for (unsigned i = 0; i < 100000 && !r.cpu.state().halted; ++i) r.cpu.step();
}

// The same through the block engine, as the backend's loop drives it (no events, no interrupts).
void run_blocks(Rig& r, BlockEngine& e) {
    for (unsigned i = 0; i < 100000 && !r.cpu.state().halted; ++i) {
        e.apply_device_writes();
        if (Block* b = e.find()) e.run(*b, ~u64{0});
        else r.cpu.step();
    }
}

void expect_same(Rig& a, Rig& b, const char* what, bool cycles = false);

// The recompiler in lock step with the block interpreter: after each of its dispatches (a chain of blocks), the
// interpreter catches up to the same instruction count and the whole state must match — a fault a later
// instruction would overwrite (a flag, C1) is seen where it happens.
void run_lockstep(Rig& b, BlockEngine& eb, Rig& c, BlockEngine& ec, const char* what) {
    for (unsigned n = 0; n < 100000 && !c.cpu.state().halted; ++n) {
        ec.apply_device_writes();
        if (Block* blk = ec.find()) ec.run(*blk, ~u64{0});
        else c.cpu.step();
        for (unsigned k = 0; k < 100000 && b.cpu.instructions() < c.cpu.instructions() && !b.cpu.state().halted; ++k) {
            eb.apply_device_writes();
            if (Block* blk = eb.find()) eb.run(*blk, ~u64{0});
            else b.cpu.step();
        }
        expect_same(b, c, what, true);
        if (::testing::Test::HasFailure()) return;
    }
}

void expect_same(Rig& a, Rig& b, const char* what, bool cycles) {
    if (cycles) {
        EXPECT_EQ(a.cpu.cycles(), b.cpu.cycles()) << what << " cycles";
    }
    for (unsigned g = 0; g < 8; ++g) EXPECT_EQ(a.cpu.state().gpr[g], b.cpu.state().gpr[g]) << what << " gpr " << g;
    EXPECT_EQ(a.cpu.state().eip, b.cpu.state().eip) << what;
    EXPECT_EQ(a.cpu.state().eflags(), b.cpu.state().eflags()) << what;
    EXPECT_EQ(a.cpu.instructions(), b.cpu.instructions()) << what;
    const FpuState& fa = a.cpu.state().fpu;
    const FpuState& fb = b.cpu.state().fpu;
    for (unsigned r = 0; r < 8; ++r) EXPECT_EQ(0, std::memcmp(&fa.st[r], &fb.st[r], 10)) << what << " st phys " << r;
    EXPECT_EQ(fa.fsw, fb.fsw) << what << " fsw";
    EXPECT_EQ(fa.ftw, fb.ftw) << what << " ftw";
    EXPECT_EQ(fa.fcw, fb.fcw) << what << " fcw";
    EXPECT_EQ(fa.fop, fb.fop) << what << " fop";
    EXPECT_EQ(fa.fip, fb.fip) << what << " fip";
    EXPECT_EQ(fa.fdp, fb.fdp) << what << " fdp";
    EXPECT_EQ(fa.fcs, fb.fcs) << what << " fcs";
    EXPECT_EQ(fa.fds, fb.fds) << what << " fds";
    EXPECT_EQ(fa.mxcsr, fb.mxcsr) << what << " mxcsr";
    for (unsigned r = 0; r < 8; ++r) EXPECT_EQ(fa.xmm[r], fb.xmm[r]) << what << " xmm" << r;
    EXPECT_EQ(fa.mxcsr, fb.mxcsr) << what << " mxcsr";
    for (unsigned r = 0; r < 8; ++r) EXPECT_EQ(fa.xmm[r], fb.xmm[r]) << what << " xmm" << r;
    if (cycles) {
        const TimingStats& ta = a.cpu.cache_timing().stats();
        const TimingStats& tb = b.cpu.cache_timing().stats();
        EXPECT_EQ(ta.l1d_misses, tb.l1d_misses) << what << " L1D misses";
        EXPECT_EQ(ta.l2_misses, tb.l2_misses) << what << " L2 misses";
        EXPECT_EQ(ta.l1i_misses, tb.l1i_misses) << what << " L1I misses";
    }
    EXPECT_EQ(0, std::memcmp(a.bus.ram() + kData, b.bus.ram() + kData, 0x200)) << what << " data";
    EXPECT_EQ(0, std::memcmp(a.bus.ram() + kStack - 0x100, b.bus.ram() + kStack - 0x100, 0x100)) << what << " stack";
}

// A program generator: integer instructions on EAX ECX EDX EBP ESI EDI (EBX stays the data pointer, ESP the stack),
// memory at [EBX + d8], balanced pushes and pops, short forward branches.
struct Gen {
    u32 seed;
    bool x87{false};
    bool simd{false};
    int depth{0};  // the x87 stack depth the program believes in (overflow and underflow now and then on purpose)
    u32 next() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    }
    u8 reg() {
        static constexpr u8 r[] = {0, 1, 2, 5, 6, 7};
        return r[next() % 6];
    }
    u8 reg8() {
        static constexpr u8 r[] = {0, 1, 2, 4, 5, 6};  // AL CL DL AH CH DH (not BL/BH: EBX is the pointer)
        return r[next() % 6];
    }
    u8 any() { return static_cast<u8>(next() & 7); }
    u8 src() {
        const u8 r = any();
        return r == 4 ? 3 : r;  // ESP's value differs per run only if pushed; keep it out of sources
    }
    // x87 forms: loads, stores and arithmetic with singles at [EBX + d8], register forms, FNSTSW AX, and FILD/FISTP
    // (interpreter-only) between them. Singles from the data area are random bits: NaNs and denormals appear.
    // MMX and SSE forms on mm0-7 and xmm0-7, memory at [EBX + d8]: 16-aligned for the packed forms the interpreter
    // checks (a misaligned one would fault there), any alignment for MOVUPS, MOVSS and the 64-bit moves.
    void simd_one(std::vector<u8>& c) {
        const u8 x = static_cast<u8>(next() % 8), y = static_cast<u8>(next() % 8);
        const u8 a16 = static_cast<u8>((next() % 8) * 16), d8 = static_cast<u8>(next() & 0xF8);
        const bool m = next() % 3 == 0;
        auto modrm = [&](u8 disp) {
            if (m) c.insert(c.end(), {static_cast<u8>(0x43 | x << 3), disp});
            else c.push_back(static_cast<u8>(0xC0 | x << 3 | y));
        };
        switch (next() % 16) {
            case 0: { static constexpr u8 ops[] = {0x58, 0x59, 0x5C, 0x5E, 0x5D, 0x5F, 0x51}; c.insert(c.end(), {0x0F, ops[next() % 7]}); modrm(a16); break; }
            case 1: { static constexpr u8 ops[] = {0x58, 0x59, 0x5C, 0x5E, 0x5D, 0x5F, 0x51}; c.insert(c.end(), {0xF3, 0x0F, ops[next() % 7]}); modrm(d8); break; }
            case 2: { static constexpr u8 ops[] = {0x54, 0x55, 0x56, 0x57, 0x14, 0x15}; c.insert(c.end(), {0x0F, ops[next() % 6]}); modrm(a16); break; }
            case 3: c.insert(c.end(), {0x0F, 0xC6}); modrm(a16); c.push_back(static_cast<u8>(next())); break;      // shufps
            case 4: c.insert(c.end(), {0x0F, 0xC2}); modrm(a16); c.push_back(static_cast<u8>(next() % 8)); break;  // cmpps
            case 5: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0x28 : 0x29)}); modrm(a16); break;     // movaps
            case 6: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0x10 : 0x11)}); modrm(d8); break;      // movups
            case 7: c.insert(c.end(), {0xF3, 0x0F, static_cast<u8>(next() & 1 ? 0x10 : 0x11)}); modrm(d8); break; // movss
            case 8: {                                                                                             // 64-bit moves
                static constexpr u8 ops[] = {0x12, 0x13, 0x16, 0x17};
                const u8 op = ops[next() % 4];
                c.insert(c.end(), {0x0F, op});
                if (m || op == 0x13 || op == 0x17) c.insert(c.end(), {static_cast<u8>(0x43 | x << 3), d8});
                else c.push_back(static_cast<u8>(0xC0 | x << 3 | y));
                break;
            }
            case 9: c.insert(c.end(), {0x0F, 0x50, static_cast<u8>(0xC0 | reg() << 3 | y)}); break;  // movmskps
            case 10: {                                                                              // MMX arithmetic, logic
                static constexpr u8 ops[] = {0xFD, 0xFE, 0xF9, 0xEC, 0xD9, 0xDB, 0xEB, 0xEF, 0xDF, 0xD5, 0xE5, 0xF5, 0x74, 0x65,
                                             0x60, 0x69, 0x6A, 0x63, 0x67, 0x6B, 0xD1, 0xE2, 0xF3, 0xF6, 0xE0, 0xDA, 0xEE};
                c.insert(c.end(), {0x0F, ops[next() % 27]});
                modrm(d8);
                break;
            }
            case 11: {                                                                              // MMX shifts by an immediate
                static constexpr u8 g[] = {2, 4, 6};
                const u8 op = static_cast<u8>(0x71 + next() % 3);
                const u8 dg = op == 0x73 ? static_cast<u8>(next() & 1 ? 2 : 6) : g[next() % 3];
                c.insert(c.end(), {0x0F, op, static_cast<u8>(0xC0 | dg << 3 | y), static_cast<u8>(next() % 40)});
                break;
            }
            case 12: {                                                                              // movd in and out
                const u8 op = static_cast<u8>(next() & 1 ? 0x6E : 0x7E);
                c.insert(c.end(), {0x0F, op});
                if (m) c.insert(c.end(), {static_cast<u8>(0x43 | x << 3), d8});
                else c.push_back(static_cast<u8>(0xC0 | x << 3 | (op == 0x7E ? reg() : src())));
                break;
            }
            case 13: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0x6F : 0x7F)}); modrm(d8); break;  // movq
            case 14: c.insert(c.end(), {0x0F, 0x70, static_cast<u8>(0xC0 | x << 3 | y), static_cast<u8>(next())}); break;  // pshufw
            case 15: c.insert(c.end(), {0x0F, 0x77}); break;                                       // emms
        }
    }
    void fpu(std::vector<u8>& c) {
        const bool reckless = next() % 32 == 0;
        const u8 d8 = static_cast<u8>(next() & 0xFC);
        const u8 i = static_cast<u8>(next() % static_cast<u32>(depth > 1 ? depth : 2));
        switch (next() % 14) {
            case 0: case 1:
                if (depth >= 8 && !reckless) return;
                c.insert(c.end(), {0xD9, 0x43, d8});  // fld dword [ebx+d8]
                ++depth;
                return;
            case 2:
                if (depth <= 0 && !reckless) return;
                c.insert(c.end(), {0xD9, static_cast<u8>(next() & 1 ? 0x5B : 0x53), d8});  // fstp / fst dword [ebx+d8]
                if (c[c.size() - 2] == 0x5B) --depth;
                return;
            case 3: case 4: {
                if (depth <= 0 && !reckless) return;
                static constexpr u8 r[] = {0, 1, 4, 5, 6, 7};
                c.insert(c.end(), {0xD8, static_cast<u8>(0x43 | r[next() % 6] << 3), d8});  // fadd/fmul/fsub/fsubr/fdiv/fdivr dword
                return;
            }
            case 5:
                if (depth <= 0 && !reckless) return;
                c.insert(c.end(), {0xD8, static_cast<u8>(next() & 1 ? 0x5B : 0x53), d8});  // fcomp / fcom dword
                if (c[c.size() - 2] == 0x5B) --depth;
                return;
            case 6: {
                if (depth < 2 && !reckless) return;
                static constexpr u8 r[] = {0, 1, 4, 5, 6, 7};
                static constexpr u8 escs[] = {0xD8, 0xDC, 0xDE};
                const u8 e = escs[next() % 3];
                c.insert(c.end(), {e, static_cast<u8>(0xC0 | r[next() % 6] << 3 | i)});
                if (e == 0xDE) --depth;
                return;
            }
            case 7:
                if ((depth <= 0 || depth >= 8) && !reckless) return;
                c.insert(c.end(), {0xD9, static_cast<u8>(0xC0 | i)});  // fld st(i)
                ++depth;
                return;
            case 8:
                if (depth < 2 && !reckless) return;
                c.insert(c.end(), {0xD9, static_cast<u8>(0xC8 | i)});  // fxch st(i)
                return;
            case 9:
                if (depth <= 0 && !reckless) return;
                c.insert(c.end(), {0xD9, static_cast<u8>(next() & 1 ? 0xE0 : 0xE1)});  // fchs / fabs
                return;
            case 10:
                if (depth >= 8 && !reckless) return;
                c.insert(c.end(), {0xD9, static_cast<u8>(next() & 1 ? 0xE8 : 0xEE)});  // fld1 / fldz
                ++depth;
                return;
            case 11:
                if (depth <= 0 && !reckless) return;
                c.insert(c.end(), {0xDD, static_cast<u8>(next() & 1 ? 0xD8 | i : 0xD0 | i)});  // fstp / fst st(i)
                if ((c.back() & 0xF8) == 0xD8) --depth;
                return;
            case 12: c.insert(c.end(), {0xDF, 0xE0}); return;  // fnstsw ax
            case 13:
                if (next() & 1) {
                    if (depth >= 8 && !reckless) return;
                    c.insert(c.end(), {0xDB, 0x43, d8});  // fild dword (the interpreter's)
                    ++depth;
                } else {
                    if (depth <= 0 && !reckless) return;
                    c.insert(c.end(), {0xDB, 0x5B, d8});  // fistp dword
                    --depth;
                }
                return;
        }
    }
    void one(std::vector<u8>& c) {
        static constexpr u8 alu[] = {0x01, 0x09, 0x11, 0x19, 0x21, 0x29, 0x31, 0x39, 0x85};
        const u8 d = reg(), s = src();
        if (x87 && next() % 3 == 0) {
            fpu(c);
            return;
        }
        if (simd && next() % 3 == 0) {
            simd_one(c);
            return;
        }
        switch (next() % 40) {
            case 0: case 1: case 2: c.insert(c.end(), {alu[next() % 9], static_cast<u8>(0xC0 | s << 3 | d)}); break;
            case 3: c.insert(c.end(), {static_cast<u8>(alu[next() % 8] - 1), static_cast<u8>(0xC0 | reg8() << 3 | reg8())}); break;
            case 4: c.insert(c.end(), {0x66, alu[next() % 9], static_cast<u8>(0xC0 | s << 3 | d)}); break;
            case 5: {
                const u32 imm = next() * 2654435761u;
                c.insert(c.end(), {0x81, static_cast<u8>(0xC0 | (next() % 8) << 3 | d), static_cast<u8>(imm), static_cast<u8>(imm >> 8), static_cast<u8>(imm >> 16),
                                   static_cast<u8>(imm >> 24)});
                break;
            }
            case 6: c.insert(c.end(), {0x83, static_cast<u8>(0xC0 | (next() % 8) << 3 | d), static_cast<u8>(next())}); break;
            case 7: c.push_back(static_cast<u8>((next() & 1 ? 0x40 : 0x48) + d)); break;
            case 8: c.insert(c.end(), {0xF7, static_cast<u8>(0xC0 | (next() & 1 ? 3 : 2) << 3 | d)}); break;
            case 9: {
                const u32 imm = next() * 40503u;
                c.insert(c.end(), {static_cast<u8>(0xB8 + d), static_cast<u8>(imm), static_cast<u8>(imm >> 8), static_cast<u8>(imm >> 16), static_cast<u8>(imm >> 24)});
                break;
            }
            case 10: c.insert(c.end(), {0x8B, static_cast<u8>(0x43 | d << 3), static_cast<u8>(next() & 0xFC)}); break;
            case 11: c.insert(c.end(), {0x89, static_cast<u8>(0x43 | s << 3), static_cast<u8>(next() & 0xFF)}); break;
            case 12: c.insert(c.end(), {alu[next() % 8], static_cast<u8>(0x43 | s << 3), static_cast<u8>(next() & 0xFF)}); break;
            case 13: {
                static constexpr u8 sh[] = {0, 1, 4, 5, 7};
                c.insert(c.end(), {0xC1, static_cast<u8>(0xC0 | sh[next() % 5] << 3 | d), static_cast<u8>(next() & 31)});
                break;
            }
            case 14: c.insert(c.end(), {0xD3, static_cast<u8>(0xC0 | (next() & 1 ? 4 : 7) << 3 | d)}); break;
            case 15: c.insert(c.end(), {0x0F, 0xAF, static_cast<u8>(0xC0 | d << 3 | s)}); break;
            case 16: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0xB6 : 0xBE), static_cast<u8>(0xC0 | d << 3 | reg8())}); break;
            case 17: c.insert(c.end(), {0x0F, static_cast<u8>(0x90 + (next() & 15)), static_cast<u8>(0xC0 | reg8())}); break;
            case 18: c.insert(c.end(), {0x0F, static_cast<u8>(0x40 + (next() & 15)), static_cast<u8>(0xC0 | d << 3 | s)}); break;
            case 19: c.insert(c.end(), {0x8D, static_cast<u8>(0x44 | d << 3), static_cast<u8>((next() % 4) << 6 | 7 << 3 | 6), static_cast<u8>(next())}); break;
            case 20: c.insert(c.end(), {0x87, static_cast<u8>(0xC0 | reg() << 3 | d)}); break;
            case 21: c.insert(c.end(), {static_cast<u8>(0x50 + s), static_cast<u8>(0x58 + d)}); break;
            case 22: c.insert(c.end(), {static_cast<u8>(0x70 + (next() & 15)), 0x02, static_cast<u8>(0x40 + d), static_cast<u8>(0x48 + reg())}); break;
            case 23: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0xC1 : 0xA3), static_cast<u8>(0xC0 | reg() << 3 | d)}); break;
            case 24: c.insert(c.end(), {0x8A, static_cast<u8>(0x43 | reg8() << 3), static_cast<u8>(next())}); break;  // mov r8, [ebx+d8]
            case 25: c.insert(c.end(), {0x88, static_cast<u8>(0x43 | reg8() << 3), static_cast<u8>(next())}); break;  // mov [ebx+d8], r8
            case 26: c.insert(c.end(), {0xFF, static_cast<u8>(0x43 | (next() & 1) << 3), static_cast<u8>(next())}); break;  // inc/dec [ebx+d8]
            case 27: c.insert(c.end(), {0x83, static_cast<u8>(0x43 | (next() % 8) << 3), static_cast<u8>(next()), static_cast<u8>(next())}); break;  // alu [ebx+d8], imm8
            case 28: c.insert(c.end(), {0x66, 0x89, static_cast<u8>(0x43 | s << 3), static_cast<u8>(next())}); break;  // mov [ebx+d8], r16
            case 29: c.insert(c.end(), {0x66, 0x8B, static_cast<u8>(0x43 | d << 3), static_cast<u8>(next() & 0xFE)}); break;  // mov r16, [ebx+d8]
            case 30: {
                static constexpr u8 sh[] = {4, 5, 7};
                c.insert(c.end(), {0xC1, static_cast<u8>(0x43 | sh[next() % 3] << 3), static_cast<u8>(next()), static_cast<u8>(next() & 31)});  // shift [ebx+d8], imm
                break;
            }
            case 32: {  // cdq or cwde, then a safe division by a register or a memory operand
                const u8 r = reg();
                const u32 v = (next() % 0x7FFFFFF0u) + 2;
                c.insert(c.end(), {static_cast<u8>(next() & 1 ? 0x99 : 0x98)});
                if (r == 2) break;  // the divisor must not be EDX
                c.insert(c.end(), {static_cast<u8>(0xB8 + r), static_cast<u8>(v), static_cast<u8>(v >> 8), static_cast<u8>(v >> 16), static_cast<u8>(v >> 24)});
                if (next() & 1) c.insert(c.end(), {0x99, 0xF7, static_cast<u8>(0xF8 | r)});           // cdq; idiv r
                else c.insert(c.end(), {0x31, 0xD2, 0xF7, static_cast<u8>(0xF0 | r)});               // xor edx, edx; div r
                break;
            }
            case 33: {  // a division by memory
                const u32 v = (next() % 0x7FFFFFF0u) + 2;
                const u8 dd = static_cast<u8>(next() & 0xFC);
                c.insert(c.end(), {0xC7, 0x43, dd, static_cast<u8>(v), static_cast<u8>(v >> 8), static_cast<u8>(v >> 16), static_cast<u8>(v >> 24)});
                if (next() & 1) c.insert(c.end(), {0x99, 0xF7, 0x7B, dd});
                else c.insert(c.end(), {0x31, 0xD2, 0xF7, 0x73, dd});
                break;
            }
            case 34: case 35: {  // an indirect CALL or JMP to the next instruction, through a register or memory
                const u8 r = reg();
                const bool call = next() & 1, viamem = next() & 1;
                const u8 dd = static_cast<u8>(next() & 0xFC);
                const u32 at = kCode + static_cast<u32>(c.size());
                const u32 len = viamem ? 7u + 3u : 5u + 2u;  // the store/load of the target, then the branch
                const u32 target = at + len;
                if (viamem) c.insert(c.end(), {0xC7, 0x43, dd, static_cast<u8>(target), static_cast<u8>(target >> 8), static_cast<u8>(target >> 16), static_cast<u8>(target >> 24), 0xFF, static_cast<u8>(call ? 0x53 : 0x63), dd});
                else c.insert(c.end(), {static_cast<u8>(0xB8 + r), static_cast<u8>(target), static_cast<u8>(target >> 8), static_cast<u8>(target >> 16), static_cast<u8>(target >> 24), 0xFF, static_cast<u8>((call ? 0xD0 : 0xE0) | r)});
                if (call) c.push_back(static_cast<u8>(0x58 + reg()));  // pop the return address
                break;
            }
            case 36: c.insert(c.end(), {0xFF, 0x73, static_cast<u8>(next()), static_cast<u8>(0x58 + reg())}); break;  // push [ebx+d8]; pop r
            case 37: c.insert(c.end(), {0x55, 0x89, 0xE5, 0xC9}); break;  // push ebp; mov ebp, esp; leave
            case 38: c.insert(c.end(), {0xC1, static_cast<u8>(0xC0 | (next() & 1) << 3 | d), static_cast<u8>(next() & 31)}); break;  // rol/ror r, imm
            case 39: c.insert(c.end(), {0x0F, 0x18, static_cast<u8>(0x43 | (next() % 4) << 3), static_cast<u8>(next())}); break;  // prefetch
            case 31: {
                static constexpr u8 sh[] = {4, 5, 7};
                const u8 op = static_cast<u8>(next() % 3 == 0 ? 0xD2 : 0xD3);  // byte or dword
                const u8 r = op == 0xD2 ? reg8() : reg();  // byte registers for the byte form, never ESP
                c.insert(c.end(), {op, static_cast<u8>(next() & 1 ? (0x43 | sh[next() % 3] << 3) : (0xC0 | sh[next() % 3] << 3 | r)), static_cast<u8>(next())});
                if ((c[c.size() - 2] & 0xC0) == 0xC0) c.pop_back();  // the register form has no displacement
                break;
            }
        }
    }
};

}  // namespace

// The three executors on the same programs: the step (architecture), the block interpreter, the recompiled blocks
// (architecture, and the clock and instruction count exactly: both run block timing).
TEST(P6Block, GeneratedProgramsMatchTheStep) {
    u64 native = 0, paged_insns = 0, chains = 0;
    for (u32 program = 0; program < 300; ++program) {
        Gen g{program * 7919u + 1};
        g.x87 = program % 3 == 0;
        g.simd = program % 5 < 2;
        static constexpr u16 kCw[] = {0x37F, 0x27F, 0x07F, 0xF7F, 0x77F};  // precisions and roundings, all masked
        // Five passes over the body (a counter on the stack the forms leave alone): blocks run again and chain.
        std::vector<u8> code = {0x6A, 0x05};  // push 5
        while (code.size() < 900) g.one(code);
        const i32 back = 2 - static_cast<i32>(code.size() + 3 + 6);
        code.insert(code.end(), {0xFF, 0x0C, 0x24, 0x0F, 0x85, static_cast<u8>(back), static_cast<u8>(back >> 8), static_cast<u8>(back >> 16),
                                 static_cast<u8>(back >> 24), 0x58});  // dec dword [esp]; jnz body; pop eax
        code.push_back(0xF4);  // hlt
        const bool paged = (program & 1) != 0;
        Rig a(paged), b(paged), c(paged);
        for (Rig* r : {&a, &b, &c}) {
            r->load(code);
            r->cpu.state().gpr[kEbx] = kData;
            for (unsigned i = 0; i < 0x200; ++i) r->bus.ram()[kData + i] = static_cast<u8>(i * 37 + program);
            r->cpu.state().gpr[kEax] = program * 0x01010101u;
            r->cpu.state().gpr[kEsi] = kData + 0x40;
            r->cpu.state().gpr[kEdi] = 0x10;
            r->cpu.state().fpu.fcw = kCw[(program / 3) % 5];
            r->cpu.state().fpu.mxcsr = 0x1F80u | ((program / 5) % 4) << 13 | ((program / 20) % 2) << 15;  // roundings, FZ
            r->cpu.state().cr4 |= cr4::OSFXSR | cr4::OSXMMEXCPT;
        }
        run_steps(a);
        b.cpu.set_block_timing(true);
        BlockEngine e(b.cpu, b.bus);
        e.disable_jit();
        run_blocks(b, e);
        expect_same(a, b, "block interpreter against the step");
        ASSERT_FALSE(::testing::Test::HasFailure()) << "program " << program;
        EXPECT_GT(e.stats().block_insns, a.cpu.instructions() / 2);  // blocks ran it, not the fallback
        // Again, in lock step: a second block interpreter beside the recompiler.
        Rig b2(paged);
        b2.load(code);
        b2.cpu.state().gpr[kEbx] = kData;
        for (unsigned k = 0; k < 0x200; ++k) b2.bus.ram()[kData + k] = static_cast<u8>(k * 37 + program);
        b2.cpu.state().gpr[kEax] = program * 0x01010101u;
        b2.cpu.state().gpr[kEsi] = kData + 0x40;
        b2.cpu.state().gpr[kEdi] = 0x10;
        b2.cpu.state().fpu.fcw = kCw[(program / 3) % 5];
        b2.cpu.state().fpu.mxcsr = 0x1F80u | ((program / 5) % 4) << 13 | ((program / 20) % 2) << 15;
        b2.cpu.state().cr4 |= cr4::OSFXSR | cr4::OSXMMEXCPT;
        b2.cpu.set_block_timing(true);
        BlockEngine e2(b2.cpu, b2.bus);
        e2.disable_jit();
        c.cpu.set_block_timing(true);
        BlockEngine ej(c.cpu, c.bus);
        ej.set_jit_threshold(0);
        ej.set_chain_budget((program / 2) % 2 ? 0 : 1024);  // one block a dispatch (compared after each) or chained
        run_lockstep(b2, e2, c, ej, "recompiled against the block interpreter, in lock step");
        expect_same(b, c, "recompiled against the block interpreter", true);
        ASSERT_FALSE(::testing::Test::HasFailure()) << "program " << program << (paged ? " (paged)" : "");
        native += ej.stats().native_insns;
        chains += ej.stats().chains;
        if (paged) paged_insns += c.cpu.instructions();
    }
    EXPECT_GT(native * 2, paged_insns);  // the recompiler ran most of the paged programs natively
    EXPECT_GT(chains, 1000u);            // the passes chained
    std::printf("generated programs: %llu native instructions of %llu, %llu chains\n", static_cast<unsigned long long>(native), static_cast<unsigned long long>(paged_insns), static_cast<unsigned long long>(chains));
}

// A block writes the instruction after the next one inside its own page: the new bytes must run (the step sees them).
TEST(P6Block, CodeWrittenByItsOwnBlockRuns) {
    // mov byte [0x100c], 0x41 (inc ecx -> its slot); nop; ...; at 0x100c: inc eax (0x40), to be rewritten.
    const std::vector<u8> code = {0xC6, 0x05, 0x0C, 0x10, 0x00, 0x00, 0x41,  // mov byte [0x100c], 0x41
                                  0x90, 0x90, 0x90, 0x90, 0x90,              // 0x1007..0x100b
                                  0x40,                                      // 0x100c: inc eax (becomes inc ecx)
                                  0xF4};                                     // hlt
    Rig a, b;
    a.load(code);
    b.load(code);
    // Warm the block engine on the original code first: run the bytes at 0x100c once as they were.
    b.cpu.set_block_timing(true);
    BlockEngine e(b.cpu, b.bus);
    b.cpu.state().eip = 0x100c;
    run_blocks(b, e);
    b.cpu.state().halted = false;
    b.cpu.state().eip = kCode;
    b.cpu.state().gpr[kEax] = 0;
    a.cpu.state().gpr[kEax] = 0;
    run_steps(a);
    run_blocks(b, e);
    EXPECT_EQ(a.cpu.state().gpr[kEcx], 1u);
    EXPECT_EQ(a.cpu.state().gpr[kEax], 0u);
    EXPECT_EQ(b.cpu.state().gpr[kEcx], a.cpu.state().gpr[kEcx]);
    EXPECT_EQ(b.cpu.state().gpr[kEax], a.cpu.state().gpr[kEax]);
    EXPECT_GE(e.stats().killed, 1u);
}

// A device writes over code the engine holds (DMA through the bus): the next run sees the new code.
TEST(P6Block, CodeWrittenByADeviceRuns) {
    Rig r;
    r.load({0x40, 0x40, 0xF4});  // inc eax; inc eax; hlt
    r.cpu.set_block_timing(true);
    BlockEngine e(r.cpu, r.bus);
    run_blocks(r, e);
    EXPECT_EQ(r.cpu.state().gpr[kEax], 2u);
    const u8 dec[] = {0x48, 0x48};  // dec eax; dec eax
    ASSERT_TRUE(r.bus.write_ram(kCode, dec, sizeof dec));
    EXPECT_TRUE(r.bus.code_writes_pending());
    r.cpu.state().halted = false;
    r.cpu.state().eip = kCode;
    run_blocks(r, e);
    EXPECT_EQ(r.cpu.state().gpr[kEax], 0u);
    EXPECT_EQ(e.stats().device_page_writes, 1u);
    // A control: a write elsewhere queues nothing.
    ASSERT_TRUE(r.bus.write_ram(kData, dec, sizeof dec));
    EXPECT_FALSE(r.bus.code_writes_pending());
}

// The schedule: a dependency chain costs more than the same instructions on independent registers; offsets never
// decrease and the total is at least a cycle.
TEST(P6Block, ScheduleCountsDependencies) {
    auto cost = [](std::vector<u8> bytes) {
        std::vector<Insn> insns;
        struct Src : CodeSource {
            const std::vector<u8>* b;
            unsigned at;
            u8 byte(unsigned i) override { return (*b)[at + i]; }
        } src;
        src.b = &bytes;
        src.at = 0;
        while (src.at < bytes.size()) {
            Insn in;
            decode(src, true, in);
            insns.push_back(in);
            src.at += in.len;
        }
        std::vector<const Insn*> list;
        for (auto& in : insns) list.push_back(&in);
        std::vector<u16> after(insns.size());
        Timing t;
        const u32 total = schedule_block(t, list.data(), static_cast<unsigned>(list.size()), after.data());
        for (std::size_t i = 1; i < after.size(); ++i) EXPECT_GE(after[i], after[i - 1]);
        EXPECT_GE(total, 1u);
        return total;
    };
    // imul eax, eax x4 (a chain, 4-cycle latency) against imul eax/ecx/edx/esi on themselves (independent).
    const u32 chain = cost({0x0F, 0xAF, 0xC0, 0x0F, 0xAF, 0xC0, 0x0F, 0xAF, 0xC0, 0x0F, 0xAF, 0xC0});
    const u32 independent = cost({0x0F, 0xAF, 0xC0, 0x0F, 0xAF, 0xC9, 0x0F, 0xAF, 0xD2, 0x0F, 0xAF, 0xF6});
    EXPECT_GT(chain, independent);
    EXPECT_GE(chain, 12u);
    EXPECT_GE(cost({0x90}), 1u);
}

// Directed x87 cases where each result is observable at once (C1 right after the instruction that sets or clears
// it, a stack fault, NaN and denormal operands, compares): the step against the recompiler under several control
// words. Singles at [EBX]: +0 1/3, +4 3.0, +8 1.0, +12 a denormal, +16 a quiet NaN, +20 a signalling NaN, +24 1e38.
TEST(P6Block, X87FormsMatchTheInterpreter) {
    const std::vector<std::vector<u8>> snippets = {
        {0xD9, 0xE8, 0xD9, 0x43, 0x04, 0xDE, 0xF9, 0xD9, 0xE0, 0xF4},              // fld1; fld 3.0; fdivp; fchs
        {0xD9, 0x03, 0xD8, 0x0B, 0xD9, 0xE1, 0xF4},                                // fld 1/3; fmul 1/3; fabs
        {0xD9, 0x03, 0xD8, 0x4B, 0x04, 0xD9, 0x5B, 0x40, 0xDF, 0xE0, 0xF4},        // fld; fmul 3.0; fstp [x]; fnstsw ax
        {0xD9, 0xEE, 0xD9, 0xE8, 0xD9, 0xC9, 0xD8, 0x5B, 0x08, 0xDF, 0xE0, 0xF4},  // fldz; fld1; fxch; fcomp 1.0; fnstsw
        {0xD9, 0x03, 0xD9, 0xC0, 0xDE, 0xC1, 0xDD, 0xD8, 0xF4},                    // fld; fld st0; faddp; fstp st0
        {0xD9, 0xE8, 0xD9, 0xE8, 0xD9, 0xE8, 0xD9, 0xE8, 0xD9, 0xE8, 0xD9, 0xE8, 0xD9, 0xE8, 0xD9, 0xE8, 0xD9, 0xE8, 0xF4},  // 9 pushes
        {0xD9, 0x43, 0x0C, 0xD8, 0x0B, 0xF4},                                      // fld denormal; fmul
        {0xD9, 0x43, 0x10, 0xD8, 0x0B, 0xD9, 0x5B, 0x40, 0xF4},                    // fld qNaN; fmul; fstp
        {0xD9, 0x43, 0x14, 0xF4},                                                  // fld sNaN
        {0xD9, 0x43, 0x18, 0xD8, 0x4B, 0x18, 0xD9, 0x5B, 0x40, 0xDF, 0xE0, 0xF4},  // 1e38 * 1e38 -> fstp single: overflow
        {0xDD, 0xD8, 0xF4},                                                        // fstp st0 on an empty stack
        {0xD9, 0xE8, 0xD8, 0x73, 0x00, 0xD9, 0xEE, 0xDE, 0xF9, 0xDF, 0xE0, 0xF4},  // 1 / (1/3); fldz; fdivp: zero divide
    };
    const u32 singles[] = {0x3EAAAAABu, 0x40400000u, 0x3F800000u, 0x00000001u, 0x7FC00000u, 0x7F800001u, 0x7E967699u};
    for (const u16 cw : std::initializer_list<u16>{0x037F, 0x027F, 0x007F, 0x0F7F, 0x077F}) {
        for (std::size_t s = 0; s < snippets.size(); ++s) {
            Rig a(true), c(true);
            for (Rig* r : {&a, &c}) {
                r->load(snippets[s]);
                std::memcpy(r->bus.ram() + kData, singles, sizeof singles);
                r->cpu.state().gpr[kEbx] = kData;
                r->cpu.state().fpu.fcw = cw;
            }
            run_steps(a);
            c.cpu.set_block_timing(true);
            BlockEngine e(c.cpu, c.bus);
            e.set_jit_threshold(0);
            run_blocks(c, e);
            expect_same(a, c, "x87 snippet");
            ASSERT_FALSE(::testing::Test::HasFailure()) << "snippet " << s << " control word " << cw;
        }
    }
}

// Write tracking under the recompiler: a watched page reports its first write once, then is not watched; every
// written page is reported when every page is logged. Native and interpreted runs report the same pages.
TEST(P6Block, WriteTrackingMatchesTheInterpreter) {
    // A loop writing [ebx] and [ebx+0x1000] eight times: mov [ebx], eax; mov [ebx+0x1000], eax; dec ecx; jnz.
    const std::vector<u8> code = {0x89, 0x03, 0x89, 0x83, 0x00, 0x10, 0x00, 0x00, 0x49, 0x75, 0xF5, 0xF4};
    for (bool selective : {true, false}) {
        std::vector<u32> pages[2];
        for (int jit = 0; jit < 2; ++jit) {
            Rig r(true);
            r.load(code);
            r.cpu.state().gpr[kEbx] = kData;
            r.cpu.state().gpr[kEcx] = 8;
            r.cpu.memory().set_dirty_tracking(true, selective);
            r.cpu.memory().watch(kData / 4096, 1);  // only the first page is watched (selective)
            r.cpu.set_block_timing(true);
            BlockEngine e(r.cpu, r.bus);
            if (!jit) e.disable_jit();
            e.set_jit_threshold(0);
            run_blocks(r, e);
            EXPECT_EQ(r.cpu.state().gpr[kEcx], 0u);
            r.cpu.memory().take_dirty_pages(pages[jit]);
            if (jit) {
                EXPECT_GT(e.stats().native_insns, 0u);
            }
        }
        EXPECT_EQ(pages[0], pages[1]) << (selective ? "selective" : "every page");
        if (selective) {
            EXPECT_EQ(pages[1], std::vector<u32>{kData / 4096});
        } else {
            EXPECT_EQ(pages[1], (std::vector<u32>{kData / 4096, kData / 4096 + 1, 0x100000 / 4096}));  // + the walk's A/D bits
        }
    }
}

// A native exit whose target no block holds (an instruction crossing a page runs as a fallback step) and the block
// after it: the exit must not be linked to that next block, or chaining would skip the step.
TEST(P6Block, ChainsOnlyToTheExitTarget) {
    Rig a(true), c(true);
    for (Rig* r : {&a, &c}) {
        r->load({0xE9, 0xF9, 0x0F, 0x00, 0x00});            // 0x1000: jmp 0x1ffe
        r->load({0x05, 0x01, 0x00, 0x00, 0x00}, 0x1FFE);    // 0x1ffe: add eax, 1 (crosses into 0x2000)
        r->load({0x49, 0x0F, 0x85, 0xF6, 0xEF, 0xFF, 0xFF, 0xF4}, 0x2003);  // dec ecx; jnz 0x1000; hlt
        r->cpu.state().gpr[kEcx] = 20;
    }
    run_steps(a);
    c.cpu.set_block_timing(true);
    BlockEngine e(c.cpu, c.bus);
    e.set_jit_threshold(0);
    run_blocks(c, e);
    EXPECT_EQ(a.cpu.state().gpr[kEax], 20u);
    expect_same(a, c, "a jump to a fallback step");
    EXPECT_GT(e.stats().native_insns, 20u);
}

// Interrupts between blocks, as the backend delivers them: an event every 997 cycles is a run's limit; at it the
// dispatcher delivers vector 0x20 (a handler that counts and returns). A native exit followed by a delivery must not
// be linked to the handler's block (a chain would skip the delivery's effects). The recompiler against the block
// interpreter: the same deliveries at the same instruction boundaries, the same state and clock.
TEST(P6Block, ChainsAcrossInterruptsMatchTheInterpreter) {
    auto setup = [](Rig& r) {
        const u64 gdt[3] = {0, 0x00CF9B000000FFFFull, 0x00CF93000000FFFFull};
        std::memcpy(r.bus.ram() + 0x5000, gdt, sizeof gdt);
        r.cpu.state().gdtr = DescTable{0x5000, 0x17};
        const u32 gate[2] = {(0x08u << 16) | 0x1800u, 0x00008E00u};  // vector 0x20: interrupt gate to 0x1800 (the loop's page)
        std::memcpy(r.bus.ram() + 0x3000 + 0x20 * 8, gate, sizeof gate);
        r.cpu.state().idtr = DescTable{0x3000, 0x7FF};
        r.load({0x83, 0xC6, 0x01, 0xCF}, 0x1800);  // handler: add esi, 1; iret
        // 0x1000: jmp 0x1010 | 0x1010: add eax, 1 | 0x1013: dec ecx; jnz 0x1000 | hlt
        r.load({0xEB, 0x0E});
        r.load({0x83, 0xC0, 0x01, 0x49, 0x0F, 0x85, 0xE7, 0xFF, 0xFF, 0xFF, 0xF4}, 0x1010);
        r.cpu.state().gpr[kEcx] = 3000;
        r.cpu.state().set_eflags(fl::IF);
    };
    auto run = [](Rig& r, BlockEngine& e) {
        u64 next_event = 997;
        for (unsigned n = 0; n < 200000 && !r.cpu.state().halted; ++n) {
            e.apply_device_writes();
            if (r.cpu.cycles() >= next_event) {
                next_event += 997;
                if (r.cpu.state().flags & fl::IF) r.cpu.deliver_external(0x20);
            }
            if (Block* b = e.find()) e.run(*b, next_event);
            else r.cpu.step();
        }
    };
    Rig b(true), c(true);
    setup(b);
    setup(c);
    b.cpu.set_block_timing(true);
    BlockEngine eb(b.cpu, b.bus);
    eb.disable_jit();
    run(b, eb);
    c.cpu.set_block_timing(true);
    BlockEngine ec(c.cpu, c.bus);
    ec.set_jit_threshold(0);
    run(c, ec);
    EXPECT_EQ(b.cpu.state().gpr[kEax], 3000u);
    EXPECT_GT(b.cpu.state().gpr[kEsi], 10u);  // interrupts were taken
    expect_same(b, c, "deliveries between native blocks", true);
    EXPECT_GT(ec.stats().chains, 100u);
}

// The boot's case: an exit to a REP string block (never translated) and then the next block on the same page. The
// exit must not be linked past the REP block (a chain would skip the copy).
TEST(P6Block, ChainsDoNotSkipAStringBlock) {
    Rig a(true), c(true);
    for (Rig* r : {&a, &c}) {
        r->load({0xB9, 0x01, 0x00, 0x00, 0x00, 0xEB, 0x09});                   // 0x1000: mov ecx, 1; jmp 0x1010
        r->load({0xF3, 0xA5, 0x4D, 0x0F, 0x85, 0xE7, 0xFF, 0xFF, 0xFF, 0xF4}, 0x1010);  // rep movsd; dec ebp; jnz 0x1000; hlt
        r->cpu.state().gpr[kEbp] = 3000;
        r->cpu.state().gpr[kEcx] = 0;
        r->cpu.state().gpr[kEsi] = 0x20000;
        r->cpu.state().gpr[kEdi] = 0x30000;
    }
    run_steps(a);
    c.cpu.set_block_timing(true);
    BlockEngine e(c.cpu, c.bus);
    e.set_jit_threshold(0);
    run_blocks(c, e);
    expect_same(a, c, "a REP block between an exit and the next block");
    EXPECT_GT(e.stats().chains, 100u);
}

// The spin skip: a loop polling a word that an "event" (the test's dispatcher, at fixed cycles, as a device would
// at a deadline) sets; the recompiler skips whole passes, the block interpreter runs them all. Same state, clock and
// instruction count; the skip happened.
TEST(P6Block, SpinSkipIsExact) {
    auto setup = [](Rig& r) {
        // 0x1000: mov eax, [0x20000]; test eax, eax; jz 0x1000 | inc ecx; mov dword [0x20000], 0; dec ebp; jnz 0x1000 | hlt
        r.load({0xA1, 0x00, 0x00, 0x02, 0x00, 0x85, 0xC0, 0x74, 0xF7, 0x41, 0xC7, 0x05, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4D, 0x75,
                0xE9, 0xF4});
        r.cpu.state().gpr[kEbp] = 20;
    };
    auto run = [](Rig& r, BlockEngine& e) {
        u64 next_event = 20011;
        for (unsigned n = 0; n < 2000000 && !r.cpu.state().halted; ++n) {
            e.apply_device_writes();
            if (r.cpu.cycles() >= next_event) {
                next_event += 20011;
                const u32 one = 1;
                std::memcpy(r.bus.ram() + 0x20000, &one, 4);
            }
            if (Block* b = e.find()) e.run(*b, next_event);
            else r.cpu.step();
        }
    };
    Rig b(true), c(true);
    setup(b);
    setup(c);
    b.cpu.set_block_timing(true);
    BlockEngine eb(b.cpu, b.bus);
    eb.disable_jit();
    run(b, eb);
    c.cpu.set_block_timing(true);
    BlockEngine ec(c.cpu, c.bus);
    ec.set_jit_threshold(0);
    run(c, ec);
    EXPECT_EQ(b.cpu.state().gpr[kEcx], 20u);
    expect_same(b, c, "a skipped spin", true);
    EXPECT_GT(ec.stats().spin_skips, 10u);
    std::printf("spin skip: %llu skips over %llu cycles of %llu\n", static_cast<unsigned long long>(ec.stats().spin_skips),
                static_cast<unsigned long long>(ec.stats().spin_cycles), static_cast<unsigned long long>(c.cpu.cycles()));
}
