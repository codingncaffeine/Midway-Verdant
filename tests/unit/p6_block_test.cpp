// The block engine (D26): blocks executed through the interpreter's own `execute` must leave the same machine as the
// per-instruction step — on generated programs (integer forms, memory through a base register, short branches),
// on code that rewrites itself inside the block that runs it, and on code a device rewrites. The block schedule's
// rules are pinned on asymmetric cases.
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <format>
#include <initializer_list>
#include <memory>
#include <string>
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
    // The CPU on the heap: it is ~380 KB, and tests keeping four rigs overflowed Windows' 1 MiB main-thread stack.
    std::unique_ptr<Cpu> cpu_storage{std::make_unique<Cpu>(bus)};
    Cpu& cpu{*cpu_storage};
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

// Run to the program's HLT (or a step budget) one instruction at a time. Instrument: P6DBG=1 prints the first faults
// (the handler they reach, the faulting bytes) and the wide test writes each program to p6-wide-N.bin in the working
// directory (objdump -D -b binary -m i386 --adjust-vma=0x1000).
void run_steps(Rig& r) {
    static const bool dbg = std::getenv("P6DBG") != nullptr;
    unsigned shown = 0;
    for (unsigned i = 0; i < 100000 && !r.cpu.state().halted; ++i) {
        const u32 at = r.cpu.state().eip;
        r.cpu.step();
        if (dbg && r.cpu.last_faulted() && shown < 12) {
            ++shown;
            std::string bytes;
            for (u32 k = 0; k < 8; ++k) bytes += std::format(" {:02x}", r.bus.ram()[at + k]);
            std::printf("fault to %x at %x:%s\n", r.cpu.state().eip, at, bytes.c_str());
        }
    }
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
        const u32 at = c.cpu.state().eip;
        const u32 mxcsr_before = c.cpu.state().fpu.mxcsr;
        if (Block* blk = ec.find()) ec.run(*blk, ~u64{0});
        else c.cpu.step();
        for (unsigned k = 0; k < 100000 && b.cpu.instructions() < c.cpu.instructions() && !b.cpu.state().halted; ++k) {
            eb.apply_device_writes();
            if (Block* blk = eb.find()) eb.run(*blk, ~u64{0});
            else b.cpu.step();
        }
        expect_same(b, c, what, true);
        if (::testing::Test::HasFailure()) {
            // Where: the dispatch's start and its first bytes (the code is identity-mapped at its linear address).
            std::string bytes;
            for (u32 i = 0; i < 48 && at + i < c.bus.ram_size(); ++i) bytes += std::format(" {:02x}", c.bus.ram()[at + i]);
            ADD_FAILURE() << what << ": dispatch " << n << " from eip " << std::hex << at << " (mxcsr before " << mxcsr_before << ", after " << c.cpu.state().fpu.mxcsr
                          << " vs " << b.cpu.state().fpu.mxcsr << "), native instructions so far " << std::dec << ec.stats().native_insns << "; bytes" << bytes;
            return;
        }
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
    for (unsigned r = 0; kHostX87 && r < 8; ++r) EXPECT_EQ(0, std::memcmp(&fa.st[r], &fb.st[r], 10)) << what << " st phys " << r;
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
        // MMX shares the x87 registers: without an 80-bit long double (MSVC) the interpreter cannot hold them.
        switch (next() % (kHostX87 ? 16u : 10u)) {
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
                c.insert(c.end(), {static_cast<u8>(next() & 1 ? 0xDD : 0xD9), 0x43, d8});  // fld qword / dword [ebx+d8]
                ++depth;
                return;
            case 2:
                if (depth <= 0 && !reckless) return;
                c.insert(c.end(), {static_cast<u8>(next() & 2 ? 0xDD : 0xD9), static_cast<u8>(next() & 1 ? 0x5B : 0x53), d8});  // fstp / fst qword or dword
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
    // The wide forms (the wide test): every remaining native integer form, SIB and absolute addressing, FS-relative
    // accesses (FS = the data area, limit 0x1FF), x87 and SSE control changes inside a block, more x87 and SSE forms,
    // and faults at fixed lengths the test's handlers skip: #DE (3 bytes), #UD (2), #GP by FS's limit (6), #PF (5).
    bool wide{false};
    void wide_one(std::vector<u8>& c) {
        const u8 d = reg(), s = src(), d8 = static_cast<u8>(next() & 0xFC);
        auto imm32 = [&](u32 v) { c.insert(c.end(), {static_cast<u8>(v), static_cast<u8>(v >> 8), static_cast<u8>(v >> 16), static_cast<u8>(v >> 24)}); };
        switch (next() % 44) {
            case 0: c.insert(c.end(), {0xF7, static_cast<u8>(0x43 | (next() & 1 ? 2 : 3) << 3), d8}); break;               // not/neg [ebx+d8]
            case 1: c.insert(c.end(), {0xF7, static_cast<u8>(0xC0 | (next() & 1 ? 4 : 5) << 3 | s)}); break;               // mul/imul r (EDX:EAX)
            case 2: c.insert(c.end(), {0xF7, static_cast<u8>(0x43 | (next() & 1 ? 4 : 5) << 3), d8}); break;               // mul/imul [ebx+d8]
            case 3: c.insert(c.end(), {0x6B, static_cast<u8>(0xC0 | d << 3 | s), static_cast<u8>(next())}); break;         // imul r, r, imm8
            case 4: c.insert(c.end(), {0x69, static_cast<u8>(0x43 | d << 3), d8}); imm32(next() * 2654435761u); break;     // imul r, [ebx+d8], imm32
            case 5: c.insert(c.end(), {0x66, 0x69, static_cast<u8>(0xC0 | d << 3 | s), static_cast<u8>(next()), static_cast<u8>(next())}); break;
            case 6: c.insert(c.end(), {static_cast<u8>(next() & 1 ? 0x11 : 0x19), static_cast<u8>(0x43 | s << 3), d8}); break;  // adc/sbb [ebx+d8], r
            case 7: c.insert(c.end(), {static_cast<u8>(next() & 1 ? 0x13 : 0x1B), static_cast<u8>(0x43 | d << 3), d8}); break;  // adc/sbb r, [ebx+d8]
            case 8: c.insert(c.end(), {0x0F, 0xC1, static_cast<u8>(0x43 | d << 3), d8}); break;                            // xadd [ebx+d8], r
            case 9: c.insert(c.end(), {0x0F, static_cast<u8>(0xB6 + (next() & 1) + (next() & 1 ? 8 : 0)), static_cast<u8>(0x43 | d << 3), static_cast<u8>(next() & 0xFE)}); break;  // movzx/movsx mem
            case 10: c.insert(c.end(), {0x0F, static_cast<u8>(0x90 + (next() & 15)), 0x43, static_cast<u8>(next())}); break;  // setcc [ebx+d8]
            case 11: c.insert(c.end(), {0x0F, static_cast<u8>(0x40 + (next() & 15)), static_cast<u8>(0x43 | d << 3), d8}); break;  // cmovcc r, [ebx+d8]
            case 12: c.insert(c.end(), {0xF7, static_cast<u8>(0xC0 | d)}); imm32(next() * 40503u); break;                // test r, imm32
            case 13: c.insert(c.end(), {0xF6, 0x43, static_cast<u8>(next()), static_cast<u8>(next())}); break;            // test byte [ebx+d8], imm8
            case 14: c.insert(c.end(), {0x0F, 0xBA, static_cast<u8>(0xC0 | (4 + next() % 4) << 3 | d), static_cast<u8>(next())}); break;  // bt/bts/btr/btc r, imm
            case 15: c.insert(c.end(), {0x0F, 0xBA, static_cast<u8>(0x43 | (4 + next() % 4) << 3), d8, static_cast<u8>(next())}); break;  // the same on [ebx+d8]
            case 16: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0xA4 : 0xAC), static_cast<u8>(0xC0 | s << 3 | d), static_cast<u8>(next() % 40)}); break;  // shld/shrd imm
            case 17: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0xA5 : 0xAD), static_cast<u8>(0xC0 | s << 3 | d)}); break;  // shld/shrd cl
            case 18: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0xBC : 0xBD), static_cast<u8>(0xC0 | d << 3 | s)}); break;  // bsf/bsr
            case 19: c.insert(c.end(), {0x0F, 0xB1, static_cast<u8>(next() & 1 ? (0x43 | s << 3) : (0xC0 | s << 3 | d))}); if ((c.back() & 0xC0) == 0x40) c.push_back(d8); break;  // cmpxchg
            case 20: c.insert(c.end(), {0x87, static_cast<u8>(0x43 | d << 3), d8}); break;                                // xchg [ebx+d8], r
            case 21: c.insert(c.end(), {0xC6, 0x43, static_cast<u8>(next()), static_cast<u8>(next())}); break;            // mov byte [ebx+d8], imm8
            case 22: c.insert(c.end(), {0x66, static_cast<u8>(next() & 1 ? 0xF7 : 0xD1), static_cast<u8>(0xC0 | (next() & 1 ? 3 : 4) << 3 | d)}); break;  // neg r16 / shl r16,1
            case 23: c.insert(c.end(), {0xF0, static_cast<u8>(next() & 1 ? 0x01 : 0x29), static_cast<u8>(0x43 | s << 3), d8}); break;  // lock add/sub
            case 24: c.push_back(static_cast<u8>(0x9E + (next() & 1))); break;                                           // sahf/lahf
            case 25: { static constexpr u8 f[] = {0xF8, 0xF9, 0xF5}; c.push_back(f[next() % 3]); break; }                 // clc/stc/cmc
            case 26: c.insert(c.end(), {0xC1, static_cast<u8>(0xC0 | (next() & 1 ? 2 : 3) << 3 | d), static_cast<u8>(next() & 31)}); break;  // rcl/rcr imm
            case 27: c.insert(c.end(), {0xD1, static_cast<u8>(0xC0 | (next() % 8) << 3 | d)}); break;                   // shifts and rotates by 1
            case 28: {                                                                                                  // SIB: [ebx + edi*s + d8], EDI masked first
                c.insert(c.end(), {0x83, 0xE7, 0x3C});
                const u8 op = static_cast<u8>(next() & 1 ? 0x8B : 0x89);
                c.insert(c.end(), {op, static_cast<u8>(0x44 | (op == 0x8B ? d : s) << 3), static_cast<u8>((next() % 4) << 6 | 7 << 3 | 3), d8});
                break;
            }
            case 29: { const u8 op = static_cast<u8>(next() & 1 ? 0xA1 : 0xA3); c.push_back(op); imm32(0x20000u + (next() % 0x1FC)); break; }  // mov eax <-> [abs]
            case 30: c.insert(c.end(), {0x8B, static_cast<u8>(0x05 | d << 3)}); imm32(0x20000u + (next() % 0x1FC)); break; // mov r, [abs]
            case 31: { const u8 op = static_cast<u8>(next() & 1 ? 0x8B : 0x89); c.insert(c.end(), {0x64, op, static_cast<u8>(0x05 | (op == 0x8B ? d : s) << 3)}); imm32(next() % 0x1FC); break; }  // fs:[disp32]
            case 32: c.insert(c.end(), {0x64, 0x01, static_cast<u8>(0x05 | s << 3)}); imm32(next() % 0x1FC); break;     // add fs:[disp32], r
            case 33: c.insert(c.end(), {0xFF, 0x73, static_cast<u8>(next()), 0x8F, 0x43, d8}); break;                  // push [m]; pop [m]
            case 34: c.insert(c.end(), {0x60, 0x61}); break;                                                             // pushad; popad
            case 35: c.insert(c.end(), {0x6A, static_cast<u8>(next()), static_cast<u8>(0x58 + reg())}); break;           // push imm8; pop r
            case 36: c.insert(c.end(), {0x0F, static_cast<u8>(0x80 + (next() & 15)), 0x01, 0x00, 0x00, 0x00, static_cast<u8>(0x40 + d)}); break;  // jcc rel32 over an inc
            case 37:                                                                                                     // #DE: div by a zeroed dword
                c.insert(c.end(), {0x31, 0xD2, 0xC7, 0x43, d8, 0, 0, 0, 0, 0xF7, 0x73, d8});
                break;
            case 38:                                                                                                     // #DE: idiv overflow
                c.insert(c.end(), {0xB8, 0, 0, 0, 0x80, 0x99, 0xC7, 0x43, d8, 0xFF, 0xFF, 0xFF, 0xFF, 0xF7, 0x7B, d8});
                break;
            case 39: c.insert(c.end(), {0x0F, 0x0B}); break;                                                             // #UD
            case 40: c.insert(c.end(), {0x64, 0xA1}); imm32(0x200u + next() % 0x100); break;                             // #GP: past FS's limit
            case 41: c.push_back(static_cast<u8>(next() & 1 ? 0xA1 : 0xA3)); imm32(0x00800000u + (next() & 0xFFFC)); break;  // #PF: unmapped
            case 42:                                                                                                     // fldcw: a new precision/rounding, all masked
                if (!x87) break;
                c.insert(c.end(), {0x66, 0xC7, 0x43, d8, static_cast<u8>(0x7F), static_cast<u8>((next() % 16) & 0x0F), 0xD9, 0x6B, d8});
                break;
            case 43:                                                                                                     // ldmxcsr: a new rounding/FZ, all masked
                if (!simd) break;
                c.insert(c.end(), {0xC7, 0x43, d8});
                imm32(0x1F80u | (next() % 4) << 13 | (next() % 2) << 15 | (next() & 0x3F));
                c.insert(c.end(), {0x0F, 0xAE, 0x53, d8});
                if (next() & 1) c.insert(c.end(), {0x0F, 0xAE, 0x5B, static_cast<u8>(next() & 0xFC)});  // stmxcsr
                break;
        }
    }
    // More x87 forms for the wide test (the depth bookkeeping as fpu()).
    void fpu_wide(std::vector<u8>& c) {
        const u8 d8 = static_cast<u8>(next() & 0xF8);
        const u8 i = static_cast<u8>(next() % static_cast<u32>(depth > 1 ? depth : 2));
        switch (next() % 19) {
            case 0: if (depth >= 8) return; c.insert(c.end(), {0xDF, static_cast<u8>(next() & 1 ? 0x43 : 0x6B), d8}); ++depth; return;  // fild m16 / m64
            case 1: if (depth <= 0) return; c.insert(c.end(), {0xDF, static_cast<u8>(next() & 1 ? 0x5B : 0x7B), d8}); --depth; return;  // fistp m16 / m64
            case 2: if (depth <= 0) return; c.insert(c.end(), {0xDB, 0x53, d8}); return;                                     // fist m32
            case 3: if (depth < 2) return; c.insert(c.end(), {0xDD, static_cast<u8>(0xE0 | i)}); return;                       // fucom st(i)
            case 4: if (depth < 2) return; c.insert(c.end(), {0xDB, static_cast<u8>((next() & 1 ? 0xF0 : 0xE8) | i)}); return;  // fcomi / fucomi
            case 5: if (depth < 2) return; c.insert(c.end(), {0xDF, static_cast<u8>((next() & 1 ? 0xF0 : 0xE8) | i)}); --depth; return;  // fcomip / fucomip
            case 6: if (depth < 2) return; c.insert(c.end(), {static_cast<u8>(next() & 1 ? 0xDA : 0xDB), static_cast<u8>(0xC0 | (next() % 4) << 3 | i)}); return;  // fcmovcc
            case 7: if (depth <= 0) return; { static constexpr u8 u[] = {0xFA, 0xFC, 0xE4, 0xE5, 0xFD, 0xF8}; c.insert(c.end(), {0xD9, u[next() % 6]}); } return;  // fsqrt frndint ftst fxam fscale fprem
            case 8: if (depth <= 0) return; c.insert(c.end(), {0xDC, static_cast<u8>(0x43 | (next() % 8) << 3), static_cast<u8>(next() & 0xFC)}); if (((c[c.size() - 2] >> 3) & 7) == 3) --depth; return;  // arith m64 (fcomp pops)
            case 9: if (depth <= 0) return; c.insert(c.end(), {0xDA, static_cast<u8>(0x43 | (next() % 8) << 3), d8}); if (((c[c.size() - 2] >> 3) & 7) == 3) --depth; return;  // fiadd.. m32
            case 10: if (depth <= 0) return; c.insert(c.end(), {0xDB, 0x7B, static_cast<u8>(next() % 0xF0)}); --depth; return;  // fstp m80
            case 11: if (depth >= 8) return; c.insert(c.end(), {0xDB, 0x6B, static_cast<u8>(next() % 0xF0)}); ++depth; return;  // fld m80
            case 12: if (depth < 2) return; c.insert(c.end(), {0xDA, 0xE9}); depth -= 2; return;                              // fucompp
            case 13: if (depth >= 8) return; { static constexpr u8 k[] = {0xE9, 0xEA, 0xEB, 0xEC, 0xED}; c.insert(c.end(), {0xD9, k[next() % 5]}); } ++depth; return;  // constants
            case 14: c.insert(c.end(), {0xD9, 0x7B, d8}); return;                                                             // fnstcw
            case 15: c.insert(c.end(), {0xDB, 0xE2}); return;                                                                 // fnclex
            case 16: if (depth >= 8) return; c.insert(c.end(), {0xDB, 0x43, static_cast<u8>(next() % 0xF0)}); ++depth; return;  // fild m32 (any byte)
            case 17: if (depth <= 0) return; c.insert(c.end(), {0xDB, 0x5B, static_cast<u8>(next() % 0xF0)}); --depth; return;  // fistp m32 (any byte)
            case 18: if (depth < 2) return; c.insert(c.end(), {0xD8, static_cast<u8>((next() & 1 ? 0xD8 : 0xD0) | i)}); if ((c.back() & 0xF8) == 0xD8) --depth; return;  // fcom(p) st(i)
        }
    }
    // More SSE and MMX forms for the wide test.
    void simd_wide(std::vector<u8>& c) {
        const u8 x = static_cast<u8>(next() % 8), y = static_cast<u8>(next() % 8), d8 = static_cast<u8>(next() & 0xF0);
        const bool m = next() % 3 == 0;
        auto modrm = [&](u8 disp) {
            if (m) c.insert(c.end(), {static_cast<u8>(0x43 | x << 3), disp});
            else c.push_back(static_cast<u8>(0xC0 | x << 3 | y));
        };
        // Scalar forms on words that are 4- but not 8-aligned, and the compares into EFLAGS (then a SETcc reads them).
        if (next() % 3 == 0) {
            const u8 d4 = static_cast<u8>((next() & 0xF8) | 4);
            const u8 dany = static_cast<u8>(next() % 0xF0);  // any byte: elements across cache lines
            switch (next() % (kHostX87 ? 6u : 5u)) {
                case 0: { static constexpr u8 ops[] = {0x58, 0x59, 0x5C, 0x5E, 0x5D, 0x5F, 0x51}; c.insert(c.end(), {0xF3, 0x0F, ops[next() % 7], static_cast<u8>(0x43 | x << 3), d4}); break; }
                case 1: c.insert(c.end(), {0xF3, 0x0F, static_cast<u8>(next() & 1 ? 0x10 : 0x11), static_cast<u8>(0x43 | x << 3), next() & 1 ? d4 : dany}); break;
                case 2: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0x2E : 0x2F), static_cast<u8>(0x43 | x << 3), d4}); break;
                case 3: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0x2E : 0x2F), static_cast<u8>(0xC0 | x << 3 | y)}); break;
                case 4: { static constexpr u8 ops[] = {0x12, 0x13, 0x16, 0x17, 0x10, 0x11}; c.insert(c.end(), {0x0F, ops[next() % 6], static_cast<u8>(0x43 | x << 3), dany}); break; }  // 64-bit moves, movups
                case 5: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0x6F : 0x7F), static_cast<u8>(0x43 | x << 3), dany}); break;  // movq mm
            }
            if (c[c.size() - 2] == 0x2E || c[c.size() - 2] == 0x2F || c[c.size() - 3] == 0x2E || c[c.size() - 3] == 0x2F)
                c.insert(c.end(), {0x0F, static_cast<u8>(0x90 + (next() & 15)), static_cast<u8>(0xC0 | reg8())});  // setcc: the flags are used
            return;
        }
        switch (next() % (kHostX87 ? 14u : 9u)) {
            case 0: c.insert(c.end(), {0xF3, 0x0F, 0x2A}); if (m) c.insert(c.end(), {static_cast<u8>(0x43 | x << 3), d8}); else c.push_back(static_cast<u8>(0xC0 | x << 3 | src())); break;  // cvtsi2ss
            case 1: c.insert(c.end(), {0xF3, 0x0F, static_cast<u8>(next() & 1 ? 0x2C : 0x2D)}); if (m) c.insert(c.end(), {static_cast<u8>(0x43 | reg() << 3), static_cast<u8>(next() % 0xF0)}); else c.push_back(static_cast<u8>(0xC0 | reg() << 3 | y)); break;  // cvt(t)ss2si
            case 2: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0x53 : 0x52)}); modrm(d8); break;             // rcpps/rsqrtps
            case 3: c.insert(c.end(), {0xF3, 0x0F, static_cast<u8>(next() & 1 ? 0x53 : 0x52)}); modrm(d8); break;       // rcpss/rsqrtss
            case 4: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0x2E : 0x2F)}); modrm(d8); break;             // ucomiss/comiss
            case 5: c.insert(c.end(), {0xF3, 0x0F, 0xC2}); modrm(d8); c.push_back(static_cast<u8>(next() % 8)); break;  // cmpss
            case 6: c.insert(c.end(), {0x0F, 0x2B, static_cast<u8>(0x43 | x << 3), d8}); break;                        // movntps
            case 7: c.insert(c.end(), {0x0F, 0xAE, 0xF8}); break;                                                      // sfence
            case 8: c.insert(c.end(), {0x0F, 0x14}); modrm(d8); break;                                                 // unpcklps
            // MMX-sharing forms (need the 80-bit registers)
            case 9: c.insert(c.end(), {0x0F, 0x2A}); if (m) c.insert(c.end(), {static_cast<u8>(0x43 | x << 3), d8}); else c.push_back(static_cast<u8>(0xC0 | x << 3 | y)); break;  // cvtpi2ps
            case 10: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0x2C : 0x2D), static_cast<u8>(0xC0 | x << 3 | y)}); break;  // cvt(t)ps2pi
            case 11: c.insert(c.end(), {0x0F, 0xC4, static_cast<u8>(0xC0 | x << 3 | src()), static_cast<u8>(next())}); break;  // pinsrw
            case 12: c.insert(c.end(), {0x0F, static_cast<u8>(next() & 1 ? 0xC5 : 0xD7), static_cast<u8>(0xC0 | reg() << 3 | y)}); if (c[c.size() - 2] == 0xC5) c.push_back(static_cast<u8>(next())); break;  // pextrw / pmovmskb
            case 13: { static constexpr u8 ops[] = {0xE0, 0xE3, 0xF6, 0xDA, 0xDE, 0xEA, 0xEE, 0xE4}; c.insert(c.end(), {0x0F, ops[next() % 8]}); modrm(d8); break; }  // pavg psadbw pmin pmax pmulhuw
        }
    }
    void one(std::vector<u8>& c) {
        static constexpr u8 alu[] = {0x01, 0x09, 0x11, 0x19, 0x21, 0x29, 0x31, 0x39, 0x85};
        const u8 d = reg(), s = src();
        if (wide && next() % 3 == 0) {
            const u32 k = next() % 4;
            if (k == 0 && x87) fpu_wide(c);
            else if (k == 1 && simd) simd_wide(c);
            else wide_one(c);
            return;
        }
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
        SCOPED_TRACE(std::format("program {}", program));
        Gen g{program * 7919u + 1};
        g.x87 = kHostX87 && program % 3 == 0;  // MSVC: no 80-bit long double, the interpreter's x87 only approximates
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
    // Without an 80-bit long double (MSVC) the interpreter's x87 is an approximation, and native x87 is off.
    if (!kHostX87) GTEST_SKIP() << "no 80-bit x87 type on this compiler";
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

// The spin probe: a polling loop that calls a function each pass (a return address and a saved register on the
// stack, a flag stored back with the value it holds) leaves memory as it found it; a loop that counts its passes in
// memory does not. Both against the block interpreter; only the first is skipped.
TEST(P6Block, SpinSkipWithStoresThatUndoThemselves) {
    for (int counting = 0; counting < 2; ++counting) {
        auto setup = [counting](Rig& r) {
            // 0x1000: mov eax, [0x20000]; test eax, eax; jnz 0x1012 | call 0x1100 | jmp 0x1000
            // 0x1012: inc ecx; mov dword [0x20000], 0; dec ebp; jnz 0x1000 | hlt
            // 0x1100: push ebx; mov dword [0x20004], 7; (counting: inc dword [0x20008];) pop ebx; ret
            r.load({0xA1, 0x00, 0x00, 0x02, 0x00, 0x85, 0xC0, 0x75, 0x09, 0xE8, 0xF2, 0x00, 0x00, 0x00, 0xEB, 0xF0, 0x90, 0x90});
            r.load({0x41, 0xC7, 0x05, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4D, 0x75, 0xE0, 0xF4}, 0x1012);
            if (counting) r.load({0x53, 0xC7, 0x05, 0x04, 0x00, 0x02, 0x00, 0x07, 0x00, 0x00, 0x00, 0xFF, 0x05, 0x08, 0x00, 0x02, 0x00, 0x5B, 0xC3}, 0x1100);
            else r.load({0x53, 0xC7, 0x05, 0x04, 0x00, 0x02, 0x00, 0x07, 0x00, 0x00, 0x00, 0x5B, 0xC3}, 0x1100);
            const u32 seven = 7;
            std::memcpy(r.bus.ram() + 0x20004, &seven, 4);
            r.cpu.state().gpr[kEbp] = 12;
        };
        auto run = [](Rig& r, BlockEngine& e) {
            u64 next_event = 30011;
            for (unsigned n = 0; n < 3000000 && !r.cpu.state().halted; ++n) {
                e.apply_device_writes();
                if (r.cpu.cycles() >= next_event) {
                    next_event += 30011;
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
        EXPECT_EQ(b.cpu.state().gpr[kEcx], 12u);
        expect_same(b, c, counting ? "a counting loop" : "a calling spin", true);
        EXPECT_EQ(0, std::memcmp(b.bus.ram() + 0x20000, c.bus.ram() + 0x20000, 16));
        if (counting) {
            EXPECT_EQ(ec.stats().spin_skips, 0u);
        } else {
            EXPECT_GT(ec.stats().spin_skips, 5u);
        }
        std::printf("%s: %llu skips over %llu cycles of %llu\n", counting ? "counting" : "calling", static_cast<unsigned long long>(ec.stats().spin_skips),
                    static_cast<unsigned long long>(ec.stats().spin_cycles), static_cast<unsigned long long>(c.cpu.cycles()));
    }
}

// A polling loop whose inner scan mispredicts every pass in the same pattern (a 4-entry table, one entry differs):
// the predictors end each pass as they began, so the probe lets it skip. Against the block interpreter.
TEST(P6Block, SpinSkipWithPeriodicMispredicts) {
    auto setup = [](Rig& r) {
        // 0x1000: mov esi, 0x20100
        // 0x1005: cmp dword [esi], 1; jne 0x100d; nop
        // 0x100d: add esi, 4; cmp esi, 0x20110; jl 0x1005
        // 0x1019: mov eax, [0x20000]; test eax, eax; jz 0x1000
        // 0x1022: inc ecx; mov dword [0x20000], 0; dec ebp; jnz 0x1000; hlt
        r.load({0xBE, 0x00, 0x01, 0x02, 0x00, 0x83, 0x3E, 0x01, 0x75, 0x03, 0x90, 0x90, 0x90, 0x83, 0xC6, 0x04, 0x81, 0xFE, 0x10, 0x01, 0x02, 0x00,
                0x7C, 0xED, 0xA1, 0x00, 0x00, 0x02, 0x00, 0x85, 0xC0, 0x74, 0xDF, 0x41, 0xC7, 0x05, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x4D, 0x75, 0xD1, 0xF4});
        const u32 table[4] = {1, 0, 0, 0};
        std::memcpy(r.bus.ram() + 0x20100, table, sizeof table);
        r.cpu.state().gpr[kEbp] = 10;
    };
    auto run = [](Rig& r, BlockEngine& e) {
        u64 next_event = 40009;
        for (unsigned n = 0; n < 3000000 && !r.cpu.state().halted; ++n) {
            e.apply_device_writes();
            if (r.cpu.cycles() >= next_event) {
                next_event += 40009;
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
    EXPECT_EQ(b.cpu.state().gpr[kEcx], 10u);
    expect_same(b, c, "a spin with periodic mispredicts", true);
    EXPECT_GT(ec.stats().spin_skips, 3u);
    std::printf("refused: %llu %llu %llu %llu %llu %llu %llu\n", static_cast<unsigned long long>(ec.stats().spin_refused[0]), static_cast<unsigned long long>(ec.stats().spin_refused[1]), static_cast<unsigned long long>(ec.stats().spin_refused[2]), static_cast<unsigned long long>(ec.stats().spin_refused[3]), static_cast<unsigned long long>(ec.stats().spin_refused[4]), static_cast<unsigned long long>(ec.stats().spin_refused[5]), static_cast<unsigned long long>(ec.stats().spin_refused[6]));
    std::printf("periodic: %llu skips over %llu cycles of %llu; mispredicts %llu\n", static_cast<unsigned long long>(ec.stats().spin_skips),
                static_cast<unsigned long long>(ec.stats().spin_cycles), static_cast<unsigned long long>(c.cpu.cycles()),
                static_cast<unsigned long long>(eb.stats().mispredicts));
}

// Accesses that straddle a page (and ones that end exactly at its last byte): the recompiler's single tag compare
// must send the straddling ones to the interpreter and keep the others native. Against the step.
TEST(P6Block, PageStraddlingAccessesMatchTheStep) {
    // EBX = 0x20FFC: mov eax, [ebx] (fits) / [ebx+1..3] (straddle); mov [ebx+3], ax (straddles); add [ebx+2], ecx;
    // movzx edx, word [ebx+3]; mov [ebx+4], eax (next page); hlt.
    const std::vector<u8> code = {0x8B, 0x03, 0x8B, 0x4B, 0x01, 0x8B, 0x53, 0x02, 0x8B, 0x73, 0x03, 0x66, 0x89, 0x43, 0x03, 0x01, 0x4B, 0x02,
                                  0x0F, 0xB7, 0x53, 0x03, 0x89, 0x43, 0x04, 0xF4};
    Rig a(true), c(true);
    for (Rig* r : {&a, &c}) {
        r->load(code);
        r->cpu.state().gpr[kEbx] = 0x20FFC;
        r->cpu.state().gpr[kEcx] = 0x01020304;
        for (unsigned k = 0; k < 16; ++k) r->bus.ram()[0x20FF8 + k] = static_cast<u8>(0x11 * (k + 1));
    }
    run_steps(a);
    c.cpu.set_block_timing(true);
    BlockEngine e(c.cpu, c.bus);
    e.set_jit_threshold(0);
    run_blocks(c, e);
    expect_same(a, c, "page-straddling accesses");
    EXPECT_EQ(0, std::memcmp(a.bus.ram() + 0x20FF0, c.bus.ram() + 0x20FF0, 32));
    EXPECT_GT(e.stats().native_helpers, 2u);  // the straddling ones went to the interpreter
}

// The wide programs: the generator's wide forms (every native integer form, SIB, absolute and FS-relative addressing,
// control-word and MXCSR changes inside blocks, more x87 and SSE forms) and faults inside blocks at fixed lengths —
// #DE, #UD, #GP (FS's limit), #PF (an unmapped page) — whose handlers count them, record CR2 and skip the instruction.
// The step against the block interpreter (architecture); the block interpreter against the recompiler in lock step
// (state, clock, caches); then both again with a timer interrupt every 997 cycles: the same deliveries at the same
// boundaries.
namespace {
constexpr u32 kFaultCounts = 0x2F00;
void wide_setup(Rig& r, const std::vector<u8>& code, u32 program) {
    r.load(code);
    // GDT: null, code 0x08, data 0x10, FS 0x18 = the data area with a 0x1FF limit.
    const u64 gdt[4] = {0, 0x00CF9B000000FFFFull, 0x00CF93000000FFFFull, 0x00409302000001FFull};
    std::memcpy(r.bus.ram() + 0x5000, gdt, sizeof gdt);
    r.cpu.state().gdtr = DescTable{0x5000, 0x1F};
    r.cpu.state().seg[kFs] = Segment{0x18, kData, 0x1FF, 0x4093};
    // Handlers at 0x2000 + 0x40 * n: (pop the error code), skip the faulting instruction, count, record CR2 for #PF.
    struct H { u8 vector; bool code; u8 skip; };
    static constexpr H hs[] = {{0, false, 3}, {6, false, 2}, {13, true, 6}, {14, true, 5}, {0x20, false, 0}};
    for (unsigned n = 0; n < 5; ++n) {
        const u32 at = 0x2000 + 0x40 * n;
        std::vector<u8> h;
        if (hs[n].code) h.insert(h.end(), {0x83, 0xC4, 0x04});
        if (hs[n].skip) h.insert(h.end(), {0x83, 0x04, 0x24, hs[n].skip});
        const u32 counter = kFaultCounts + 4 * n;
        h.insert(h.end(), {0xFF, 0x05, static_cast<u8>(counter), static_cast<u8>(counter >> 8), static_cast<u8>(counter >> 16), static_cast<u8>(counter >> 24)});
        if (hs[n].vector == 14) {
            const u32 cr2 = kFaultCounts + 0x40;
            h.insert(h.end(), {0x52, 0x0F, 0x20, 0xD2, 0x89, 0x15, static_cast<u8>(cr2), static_cast<u8>(cr2 >> 8), static_cast<u8>(cr2 >> 16), static_cast<u8>(cr2 >> 24), 0x5A});
        }
        h.push_back(0xCF);  // iret
        r.load(h, at);
        const u32 gate[2] = {(0x08u << 16) | at, 0x00008E00u};
        std::memcpy(r.bus.ram() + 0x3000 + hs[n].vector * 8, gate, sizeof gate);
    }
    r.cpu.state().idtr = DescTable{0x3000, 0x7FF};
    r.cpu.state().gpr[kEbx] = kData;
    for (unsigned i = 0; i < 0x200; ++i) r.bus.ram()[kData + i] = static_cast<u8>(i * 37 + program);
    r.cpu.state().gpr[kEax] = program * 0x01010101u;
    r.cpu.state().gpr[kEsi] = kData + 0x40;
    r.cpu.state().gpr[kEdi] = 0x10;
    static constexpr u16 kCw[] = {0x37F, 0x27F, 0x07F, 0xF7F, 0x77F};
    r.cpu.state().fpu.fcw = kCw[(program / 3) % 5];
    r.cpu.state().fpu.mxcsr = 0x1F80u | ((program / 5) % 4) << 13 | ((program / 20) % 2) << 15;
    r.cpu.state().cr4 |= cr4::OSFXSR | cr4::OSXMMEXCPT;
    r.cpu.state().set_eflags(fl::IF);
    r.cpu.set_block_timing(true);
}
// A run with the timer: at each event boundary vector 0x20 when IF is set; every delivery's point recorded.
std::vector<std::string> run_with_timer(Rig& r, BlockEngine& e) {
    std::vector<std::string> points;
    u64 next_event = 997;
    for (unsigned n = 0; n < 400000 && !r.cpu.state().halted; ++n) {
        e.apply_device_writes();
        if (r.cpu.cycles() >= next_event) {
            next_event += 997;
            if (r.cpu.state().flags & fl::IF) {
                points.push_back(std::format("{} {} {:x} {:x}", r.cpu.instructions(), r.cpu.cycles(), r.cpu.state().eip, r.cpu.state().eflags()));
                r.cpu.deliver_external(0x20);
            }
        }
        if (Block* b = e.find()) e.run(*b, r.cpu.state().inhibit_interrupts ? r.cpu.cycles() : next_event);
        else r.cpu.step();
    }
    return points;
}
void expect_same_wide(Rig& a, Rig& b, const char* what) {
    EXPECT_EQ(0, std::memcmp(a.bus.ram() + kData, b.bus.ram() + kData, 0x400)) << what << " data and beyond";
    EXPECT_EQ(0, std::memcmp(a.bus.ram() + kFaultCounts, b.bus.ram() + kFaultCounts, 0x80)) << what << " fault counts";
    EXPECT_EQ(a.cpu.state().cr2, b.cpu.state().cr2) << what << " cr2";
}
}  // namespace

TEST(P6Block, WideProgramsMatchTheInterpreter) {
    u64 native = 0, insns = 0, faults[4] = {}, deliveries = 0;
    for (u32 program = 0; program < 240; ++program) {
        SCOPED_TRACE(std::format("wide program {}", program));
        Gen g{program * 104729u + 7};
        g.wide = true;
        g.x87 = kHostX87 && program % 3 != 2;
        g.simd = program % 4 != 3;
        std::vector<u8> code = {0x6A, 0x05};  // push 5
        while (code.size() < 900) g.one(code);
        const i32 back = 2 - static_cast<i32>(code.size() + 3 + 6);
        code.insert(code.end(), {0xFF, 0x0C, 0x24, 0x0F, 0x85, static_cast<u8>(back), static_cast<u8>(back >> 8), static_cast<u8>(back >> 16),
                                 static_cast<u8>(back >> 24), 0x58, 0xF4});  // dec dword [esp]; jnz body; pop eax; hlt
        if (std::getenv("P6DBG")) {
            if (std::FILE* f = std::fopen(std::format("p6-wide-{}.bin", program).c_str(), "wb")) {
                std::fwrite(code.data(), 1, code.size(), f);
                std::fclose(f);
            }
        }
        // The step and the block interpreter: the architecture.
        Rig a(true), b(true);
        wide_setup(a, code, program);
        a.cpu.set_block_timing(false);
        wide_setup(b, code, program);
        if (const char* t = std::getenv("P6DBG_TRACE"); t && std::strtoul(t, nullptr, 10) == program) {
            // Instrument: the step's x87 state after every instruction of this program (raw registers, TOP, tags, status).
            for (unsigned k = 0; k < 100000 && !a.cpu.state().halted; ++k) {
                const u32 at = a.cpu.state().eip;
                a.cpu.step();
                const FpuState& f = a.cpu.state().fpu;
                std::string line = std::format("{:x} sw {:04x} tw {:04x} cw {:04x} mx {:04x}", at, f.fsw, f.ftw, f.fcw, f.mxcsr);
                for (unsigned r = 0; r < 8; ++r) {
                    u8 raw[10];
                    std::memcpy(raw, &f.st[r], 10);
                    line += " ";
                    for (int byte = 9; byte >= 0; --byte) line += std::format("{:02x}", raw[byte]);
                }
                std::printf("%s\n", line.c_str());
            }
        }
        run_steps(a);
        BlockEngine eb(b.cpu, b.bus);
        eb.disable_jit();
        run_blocks(b, eb);
        expect_same(a, b, "wide: block interpreter against the step");
        expect_same_wide(a, b, "wide: block interpreter against the step");
        ASSERT_FALSE(::testing::Test::HasFailure()) << "wide program " << program;
        // The recompiler in lock step with a second block interpreter.
        Rig b2(true), c(true);
        wide_setup(b2, code, program);
        wide_setup(c, code, program);
        BlockEngine e2(b2.cpu, b2.bus);
        e2.disable_jit();
        BlockEngine ej(c.cpu, c.bus);
        ej.set_jit_threshold(0);
        ej.set_chain_budget(program % 2 ? 0 : 1024);
        run_lockstep(b2, e2, c, ej, "wide: recompiled against the block interpreter, in lock step");
        expect_same(b, c, "wide: recompiled against the block interpreter", true);
        expect_same_wide(b, c, "wide: recompiled against the block interpreter");
        ASSERT_FALSE(::testing::Test::HasFailure()) << "wide program " << program;
        native += ej.stats().native_insns;
        insns += c.cpu.instructions();
        for (unsigned k = 0; k < 4; ++k) {
            u32 n;
            std::memcpy(&n, c.bus.ram() + kFaultCounts + 4 * k, 4);
            faults[k] += n;
        }
        // Both again under the timer.
        Rig t1(true), t2(true);
        wide_setup(t1, code, program);
        wide_setup(t2, code, program);
        BlockEngine et1(t1.cpu, t1.bus);
        et1.disable_jit();
        BlockEngine et2(t2.cpu, t2.bus);
        et2.set_jit_threshold(0);
        const auto p1 = run_with_timer(t1, et1);
        const auto p2 = run_with_timer(t2, et2);
        ASSERT_EQ(p1.size(), p2.size()) << "deliveries";
        for (std::size_t k = 0; k < p1.size(); ++k) ASSERT_EQ(p1[k], p2[k]) << "delivery " << k;
        expect_same(t1, t2, "wide, timer: recompiled against the block interpreter", true);
        expect_same_wide(t1, t2, "wide, timer: recompiled against the block interpreter");
        ASSERT_FALSE(::testing::Test::HasFailure()) << "wide program " << program;
        deliveries += p1.size();
    }
    EXPECT_GT(native * 3, insns);  // a good share ran natively
    for (unsigned k = 0; k < 4; ++k) EXPECT_GT(faults[k], 20u) << "fault kind " << k;  // every fault kind happened
    EXPECT_GT(deliveries, 1000u);
    std::printf("wide programs: %llu native instructions of %llu; faults #DE %llu #UD %llu #GP %llu #PF %llu; %llu timer deliveries\n",
                static_cast<unsigned long long>(native), static_cast<unsigned long long>(insns), static_cast<unsigned long long>(faults[0]),
                static_cast<unsigned long long>(faults[1]), static_cast<unsigned long long>(faults[2]), static_cast<unsigned long long>(faults[3]),
                static_cast<unsigned long long>(deliveries));
}

// A function called from two places in turn: its RET's link always remembers the other caller, so the chain looks the
// real return block up (the fast cache) and relinks instead of going back to the dispatcher. Against the block
// interpreter in lock step; the relinks happened.
TEST(P6Block, ChainsRelinkAReturnToItsOtherCaller) {
    // 0x1000: mov ecx, 2000
    // 0x1005: call 0x1100 | 0x100A: add eax, 3 | 0x100D: call 0x1100 | 0x1012: add edx, 5 | 0x1015: dec ecx; jnz 0x1005 | hlt
    // 0x1100: add esi, eax; xor esi, edx; ret
    const std::vector<u8> main = {0xB9, 0xD0, 0x07, 0x00, 0x00, 0xE8, 0xF6, 0x00, 0x00, 0x00, 0x83, 0xC0, 0x03, 0xE8, 0xEE, 0x00, 0x00, 0x00,
                                  0x83, 0xC2, 0x05, 0x49, 0x0F, 0x85, 0xE9, 0xFF, 0xFF, 0xFF, 0xF4};
    const std::vector<u8> fn = {0x01, 0xC6, 0x31, 0xD6, 0xC3};
    Rig b(true), c(true);
    for (Rig* r : {&b, &c}) {
        r->load(main);
        r->load(fn, 0x1100);
        r->cpu.set_block_timing(true);
    }
    BlockEngine eb(b.cpu, b.bus);
    eb.disable_jit();
    BlockEngine ec(c.cpu, c.bus);
    ec.set_jit_threshold(0);
    run_lockstep(b, eb, c, ec, "a return to alternating callers");
    expect_same(b, c, "a return to alternating callers", true);
    EXPECT_EQ(c.cpu.state().gpr[kEax], 6000u);
    EXPECT_GT(ec.stats().relinks, 1000u) << "the returns relinked";
}

// More lines through one L1 data set than it has ways, with hits on lines that are not the set's most recent: the
// model's LRU order decides every victim, so the generated code's own hit handling (ages, the clock) must be the
// model's to the cycle. (The generated programs touch too few lines for any eviction.)
TEST(P6Block, CacheSetPressureMatchesTheInterpreter) {
    // 0x1000: mov ecx, 3000
    // 0x1005: mov eax, [ebx]; mov eax, [ebx+0x1000]; mov eax, [ebx]; mov eax, [ebx+0x2000]; mov eax, [ebx+0x3000];
    //         mov eax, [ebx+0x1000]; mov edx, ecx; and edx, 7; shl edx, 12; mov eax, [edx+ebx]; mov [ebx+0x5000], eax;
    //         dec ecx; jnz 0x1005 | hlt
    const std::vector<u8> code = {0xB9, 0xB8, 0x0B, 0x00, 0x00,
                                  0x8B, 0x03, 0x8B, 0x83, 0x00, 0x10, 0x00, 0x00, 0x8B, 0x03, 0x8B, 0x83, 0x00, 0x20, 0x00, 0x00,
                                  0x8B, 0x83, 0x00, 0x30, 0x00, 0x00, 0x8B, 0x83, 0x00, 0x10, 0x00, 0x00, 0x89, 0xCA, 0x83, 0xE2, 0x07,
                                  0xC1, 0xE2, 0x0C, 0x8B, 0x04, 0x1A, 0x89, 0x83, 0x00, 0x50, 0x00, 0x00, 0x49, 0x0F, 0x85, 0xCC, 0xFF, 0xFF, 0xFF, 0xF4};
    Rig b(true), c(true);
    for (Rig* r : {&b, &c}) {
        r->load(code);
        r->cpu.state().gpr[kEbx] = kData;
        r->cpu.set_block_timing(true);
    }
    BlockEngine eb(b.cpu, b.bus);
    eb.disable_jit();
    BlockEngine ec(c.cpu, c.bus);
    ec.set_jit_threshold(0);
    run_lockstep(b, eb, c, ec, "lines through one set");
    expect_same(b, c, "lines through one set", true);
    EXPECT_EQ(b.cpu.cache_timing().cache_hash(), c.cpu.cache_timing().cache_hash()) << "the caches' contents and order";
    EXPECT_TRUE(c.cpu.state().halted);
    EXPECT_GT(c.cpu.cache_timing().stats().l1d_misses, 3000u) << "the set was under pressure";
    EXPECT_GT(ec.stats().native_insns, 30000u) << "the loop ran natively";
}

// Code lines through one L1 instruction set: six blocks 4 KiB apart (one set), chained by direct jumps, two of them
// entered again from the line of an earlier one (a hit on a way that is not the set's most recent). The chain's own
// fetch lookup must age and evict exactly as the model.
TEST(P6Block, CodeSetPressureMatchesTheInterpreter) {
    auto jmp = [](u32 at, u32 to) {
        const u32 rel = to - (at + 5);
        return std::vector<u8>{0xE9, static_cast<u8>(rel), static_cast<u8>(rel >> 8), static_cast<u8>(rel >> 16), static_cast<u8>(rel >> 24)};
    };
    // add eax, k; jmp next — A 0x1000 -> B 0x2000 -> A2 0x1010 -> C 0x3000 -> D 0x4000 -> B2 0x2010 -> E 0x5000 -> F 0x6000
    const u32 at[] = {0x1000, 0x2000, 0x1010, 0x3000, 0x4000, 0x2010, 0x5000};
    const u32 to[] = {0x2000, 0x1010, 0x3000, 0x4000, 0x2010, 0x5000, 0x6000};
    Rig b(true), c(true);
    for (Rig* r : {&b, &c}) {
        for (unsigned k = 0; k < 7; ++k) {
            std::vector<u8> code = {0x83, 0xC0, static_cast<u8>(k + 1)};
            const auto j = jmp(at[k] + 3, to[k]);
            code.insert(code.end(), j.begin(), j.end());
            r->load(code, at[k]);
        }
        // F 0x6000: dec ecx; jnz 0x1000; hlt
        const u32 rel = 0x1000u - 0x6007u;
        r->load({0x49, 0x0F, 0x85, static_cast<u8>(rel), static_cast<u8>(rel >> 8), static_cast<u8>(rel >> 16), static_cast<u8>(rel >> 24), 0xF4}, 0x6000);
        r->cpu.state().gpr[kEcx] = 2000;
        r->cpu.set_block_timing(true);
    }
    BlockEngine eb(b.cpu, b.bus);
    eb.disable_jit();
    BlockEngine ec(c.cpu, c.bus);
    ec.set_jit_threshold(0);
    run_lockstep(b, eb, c, ec, "code lines through one set");
    expect_same(b, c, "code lines through one set", true);
    EXPECT_EQ(b.cpu.cache_timing().cache_hash(), c.cpu.cache_timing().cache_hash()) << "the caches' contents and order";
    EXPECT_TRUE(c.cpu.state().halted);
    EXPECT_EQ(c.cpu.state().gpr[kEax], 2000u * 28u);
    EXPECT_GT(c.cpu.cache_timing().stats().l1i_misses, 2000u) << "the set was under pressure";
    EXPECT_GT(ec.stats().chains, 10000u) << "the blocks chained";
}
