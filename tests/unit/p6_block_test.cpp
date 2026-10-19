// The block engine (D26): blocks executed through the interpreter's own `execute` must leave the same machine as the
// per-instruction step — on generated programs (integer forms, memory through a base register, short branches),
// on code that rewrites itself inside the block that runs it, and on code a device rewrites. The block schedule's
// rules are pinned on asymmetric cases.
#include <gtest/gtest.h>

#include <cstring>
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

void expect_same(Rig& a, Rig& b, const char* what, bool cycles = false) {
    if (cycles) {
        EXPECT_EQ(a.cpu.cycles(), b.cpu.cycles()) << what << " cycles";
    }
    for (unsigned g = 0; g < 8; ++g) EXPECT_EQ(a.cpu.state().gpr[g], b.cpu.state().gpr[g]) << what << " gpr " << g;
    EXPECT_EQ(a.cpu.state().eip, b.cpu.state().eip) << what;
    EXPECT_EQ(a.cpu.state().eflags(), b.cpu.state().eflags()) << what;
    EXPECT_EQ(a.cpu.instructions(), b.cpu.instructions()) << what;
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
    void one(std::vector<u8>& c) {
        static constexpr u8 alu[] = {0x01, 0x09, 0x11, 0x19, 0x21, 0x29, 0x31, 0x39, 0x85};
        const u8 d = reg(), s = src();
        switch (next() % 30) {
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
        }
    }
};

}  // namespace

// The three executors on the same programs: the step (architecture), the block interpreter, the recompiled blocks
// (architecture, and the clock and instruction count exactly: both run block timing).
TEST(P6Block, GeneratedProgramsMatchTheStep) {
    u64 native = 0, paged_insns = 0;
    for (u32 program = 0; program < 300; ++program) {
        Gen g{program * 7919u + 1};
        std::vector<u8> code;
        while (code.size() < 900) g.one(code);
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
        }
        run_steps(a);
        b.cpu.set_block_timing(true);
        BlockEngine e(b.cpu, b.bus);
        e.disable_jit();
        run_blocks(b, e);
        expect_same(a, b, "block interpreter against the step");
        ASSERT_FALSE(::testing::Test::HasFailure()) << "program " << program;
        EXPECT_GT(e.stats().block_insns, a.cpu.instructions() / 2);  // blocks ran it, not the fallback
        c.cpu.set_block_timing(true);
        BlockEngine ej(c.cpu, c.bus);
        ej.set_jit_threshold(0);
        run_blocks(c, ej);
        expect_same(b, c, "recompiled against the block interpreter", true);
        ASSERT_FALSE(::testing::Test::HasFailure()) << "program " << program << (paged ? " (paged)" : "");
        native += ej.stats().native_insns;
        if (paged) paged_insns += c.cpu.instructions();
    }
    EXPECT_GT(native * 2, paged_insns);  // the recompiler ran most of the paged programs natively
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
