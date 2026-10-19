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

// What the generated code and its helpers share (offsets fixed: the code addresses it from RBX).
struct JitCtx {
    CpuState* st;          // 0
    Cpu* cpu;              // 8
    BlockEngine* engine;   // 16
    Block* block;          // 24
    u64 base;              // 32: the run's start plus the penalties so far (cycles)
    u64 limit;             // 40: the cycle the run must not pass
    u64 devices;           // 48: the memory unit's device-access count at the run's start
    u32 helpers;           // 56: instructions the interpreter executed for the native code
    u8 exit;               // 60: leave after the current instruction (1), or hand the rest of the block to the
                           //     interpreter after it (2: a penalty may make a later boundary reach the limit)
    u8 pad[3];             // 61
    u64 scratch[12];       // 64: slots for the generated code (see jit_x64.cpp)
    std::exception_ptr* error;  // where a helper leaves what the interpreter threw (an unimplemented form)
    u64* helper_ops;  // statistics: the interpreter's instructions by operation
    // Chains (used by C++ only): the current block's entry EIP; whether its exit was already costed and the cycle it
    // ended at; the instructions of the blocks before it in this run; how many more blocks may chain; the link slot
    // the last exit tried; the penalties added (statistics).
    u32 entry_eip;
    u8 ended;
    u64 end_cycles;
    u64 insns_done;
    u32 chain_budget;
    Block::Link* last_slot;
    u32 last_target;
    u64 penalties;
    // The spin skip's evidence (see BlockEngine::run_native): stores the native code made, mispredicts, where the
    // run began, the blocks a run may chain past its budget to end where it began.
    u32 stores;
    u32 mispredicts;
    u32 start_eip;
    u32 chain_extra;
    // Statistics of the chains the generated code took (folded into the engine's after the run).
    u32 inline_chains;
    u32 inline_mispredicts;
    u64 inline_sched;
    u64 inline_insns;
    // The spin probe: with `logging` set, each native store logs its host address, size and the bytes it replaces
    // (16-byte stores as two entries); past 256 entries only the count grows.
    u8 logging;
    u32 log_count;
    struct LogEntry {
        u64 where;  // the host address, the size in the top byte
        u64 old;
    } log[256];
};

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

private:
    u8* buffer_{nullptr};
    std::size_t capacity_{0}, used_{0};
    u64 compiled_{0}, flushes_{0};
};

}  // namespace mv::p6
