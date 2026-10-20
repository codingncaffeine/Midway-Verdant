#pragma once
// The recompiler (D26): a hot block becomes host code. Guest registers and flags stay where the interpreter keeps
// them (CpuState, with the same lazy-flag representation), so an instruction the translator does not cover is
// executed by the interpreter in the middle of native code, and the state is never converted. A native instruction
// first checks that every memory access it will make is a plain TLB hit on RAM (inside its segment, not crossing a
// page, not into translated code); otherwise the interpreter executes it from the start, so faults, page walks and
// device accesses are always the interpreter's. Each access is looked up in the block timing's cache model in the
// interpreter's order (inline when it hits the set's most recent line, which changes no LRU order). So a block run
// natively leaves the machine, the clock and the caches exactly as the block interpreter does — the oracle.
#include <array>
#include <cstddef>
#include <exception>
#include <memory>

#include "core/types.h"
#include "cpu/p6/block.h"

namespace mv::p6 {

// What the generated code and its helpers share (offsets fixed: the code addresses it from RBX). One per engine, for
// its life (a device thread may zero `chain_limit` at any time: see Bus::set_code_write_kill).
struct JitCtx {
    CpuState* st{};          // 0
    Cpu* cpu{};              // 8
    BlockEngine* engine{};   // 16
    Block* block{};          // 24
    u64 base{};              // 32: the run's start plus the penalties so far (cycles)
    u64 limit{};             // 40: the cycle the run must not pass (the next event, the stop cycle)
    u64 devices{};           // 48: the memory unit's device-access count at the run's start
    u32 helpers{};           // 56: instructions the interpreter executed for the native code
    u8 exit{};               // 60: leave after the current instruction (1), or hand the rest of the block to the
                           //     interpreter after it (2: a penalty may make a later boundary reach the limit)
    u8 pad[3]{};             // 61
    u64 scratch[12]{};       // 64: slots for the generated code (see jit_x64.cpp)
    // The generated code's chain goes on into the next block only while that block ends before this cycle: the limit,
    // or sooner the end of the chain budget; 0 sends every exit through the engine (a device wrote into code, a helper
    // walked the page tables, a logging probe). The engine restores it where it chains.
    u64 chain_limit{};
    u64 budget_end{};
    std::exception_ptr* error{};  // where a helper leaves what the interpreter threw (an unimplemented form)
    u64* helper_ops{};  // statistics: the interpreter's instructions by operation
    // Chains: whether the run's last exit was already costed (by the engine) and the cycle it ended at; the
    // instructions of the blocks before the current one in this run; the link slot the last exit tried; the penalties
    // added (statistics).
    u8 ended{};
    u64 end_cycles{};
    u64 insns_done{};
    Block::Link* last_slot{};
    u32 last_target{};
    u64 penalties{};
    // The spin skip's evidence (see BlockEngine::chain): whether the native code made a store since the probe was armed
    // (1), mispredicts, where the run began, the blocks a run may chain past its budget to come back to the probe.
    u32 stores{};
    u32 mispredicts{};
    u32 start_eip{};
    u32 chain_extra{};
    // The spin probe: armed where the run first went past its chain budget (or at its start, for a known loop or a
    // logging probe) at EIP `probe_eip`, with the cycle, the instruction count and the counters there; the generated
    // code's chain leaves to the engine when its target is `probe_eip` (~0: none armed).
    u32 probe_eip{0xFFFFFFFFu};
    u8 probe{};         // 0 none, 1 armed, 2 back at the armed state
    u8 probe_misses{};  // arrivals at probe_eip with another state
    u64 probe_cycle{}, probe_insns{}, probe_penalties{};
    u32 probe_helpers{}, probe_mispredicts{};
    // Statistics of the chains the generated code took (folded into the engine's after the run).
    u32 inline_chains{};
    u32 inline_mispredicts{};
    // The spin probe: with `logging` set, each native store logs its host address, size and the bytes it replaces
    // (16-byte stores as two entries); past 256 entries only the count grows.
    u8 logging{};
    u32 log_count{};
    struct LogEntry {
        u64 where;  // the host address, the size in the top byte
        u64 old;
    } log[256];  // not initialised: read only up to log_count (zeroing 4 KiB a run was a measurable cost)
    u8 stop_why{};  // instrument: which test in BlockEngine::chain ended the run (0: none)
    // Where returns and indirect jumps chain inside the generated code: by target EIP (direct-mapped), each block the
    // engine chained such an exit into, with what its checks need (filled by BlockEngine::chain, cleared when the block
    // goes or is translated again). 64 bytes an entry, offsets fixed.
    struct Target {
        u32 eip;
        u32 key;     // the block's snapshot (segments, write tracking): it must be the exit's own
        u32 cycles;
        u32 tlb;     // the TLB entry's offset, its tag and physical page for the block's page
        u32 tag;
        u32 page;
        u32 line[2];  // the block's code lines plus one (the second 0: one line)
        i64 mru[2];   // each line's set in L1I, as an offset from the L1D MRU array (R15)
        void* body;
        Block* block;
    };
    static constexpr unsigned kTargets = 4096;
    static unsigned target_index(u32 eip) { return (eip ^ (eip >> 12)) & (kTargets - 1); }
    Target targets[kTargets]{};
};
static_assert(sizeof(JitCtx::Target) == 64);

// The native function's result: (instruction index << 2) | kind.
enum JitResult : u32 { kJitEnd = 0, kJitExit = 1, kJitFault = 3 };

class Jit {
public:
    Jit();
    ~Jit();
    Jit(const Jit&) = delete;
    Jit& operator=(const Jit&) = delete;

    // True when the host can run generated code (an x86-64 host with executable memory).
    bool available() const { return buffer_ != nullptr; }
    // Translate `b` for the machine as it is now (its entry EIP, segments, write tracking). False if it cannot
    // be (the block stays interpreted).
    bool compile(Block& b, Cpu& cpu, BlockEngine& engine);
    // Whether `b`'s native code may run from the current state.
    static bool eligible(const Block& b, Cpu& cpu);
    using Fn = u32 (*)(JitCtx*);
    static Fn entry(const Block& b) { return reinterpret_cast<Fn>(b.native); }

    u64 compiled() const { return compiled_; }
    u64 code_bytes() const { return used_; }
    u64 flushes() const { return flushes_; }
    // Drop every block's native code (the buffer is full, or the machine reset).
    void flush(BlockEngine& engine);
    // Exit `slot` of `from`'s last instruction patched to chain into `to`'s native code inside the generated code (its
    // chain cell gets `to`'s cycles, its page's translation unless it is `from`'s page, its code lines unless they are
    // among `from`'s, and the jump). False when the cell cannot hold the link (none for this exit, `to`'s page not in
    // the TLB as `to` was made for, more than four lines). The link's conditions (same segments, write tracking, not
    // the idle loop) are the caller's.
    static bool patch_link(Block& from, unsigned slot, const Block& to, Cpu& cpu);
    static void unpatch_link(Block& from, unsigned slot);

private:
    u8* buffer_{nullptr};
    std::size_t capacity_{0}, used_{0};
    u64 compiled_{0}, flushes_{0};
};

}  // namespace mv::p6
