#pragma once
// The block engine (D26): the interpreted Pentium III run as translated blocks. A block is a straight run of decoded
// instructions inside one RAM page, ending at a control transfer, at an instruction that can change what the next
// one may assume (segments, interrupt flag, control registers, port I/O), at the page's end or after 32
// instructions. Its cycles come from a schedule computed once (block timing); device latencies, REP element counts
// and the final branch's prediction are added as they happen. The engine keeps the blocks by physical address,
// drops a page's blocks when the CPU or a device writes into it, and runs a block's instructions through the
// interpreter's own `execute` — the reference the native recompiler is checked against.
#include <array>
#include <cstdio>
#include <exception>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/bus.h"
#include "core/types.h"
#include "cpu/p6/cpu.h"
#include "cpu/p6/decode.h"
#include "cpu/p6/memory.h"

namespace mv::p6 {

inline constexpr unsigned kMaxBlockInsns = 32;

enum class BranchKind : u8 { None, Conditional, Direct, Indirect, Return, Call, CallIndirect };

struct Block {
    u32 ram_page{0};   // RAM page index of the code (a block never leaves its page)
    u16 page_off{0};   // offset of the first byte in the page
    u8 mode{0};        // bit 0: 32-bit code; bit 1: 32-bit stack
    u8 n{0};           // instructions
    u16 bytes{0};      // length of the code
    bool rep{false};   // a lone REP string instruction: costed per element
    u8 countdown{0xFF};  // a two-instruction loop on itself that counts this register down by one (DEC/SUB 1; JNZ), else 0xFF
    BranchKind branch{BranchKind::None};  // what the final instruction is
    u8 exit_counter{1};  // the final conditional branch's two-bit predictor (taken at 2 and 3)
    bool exit_seen{false};
    u32 exit_target{0};  // the last target of an indirect final branch
    u32 cycles{0};       // the schedule's total
    std::array<u16, kMaxBlockInsns> after{};  // cycles from the start to each instruction's retirement
    std::vector<Insn> insns;
    std::array<u16, kMaxBlockInsns> off{};  // each instruction's offset from the first byte
    void* native{nullptr};  // the recompiled form, when there is one
    // The recompiled form's assumptions, checked before it runs: the entry EIP, all six segment registers (base,
    // limit and rights; native code never changes one, so a chain between blocks with equal snapshots needs no
    // check), whether write tracking was on.
    u32 jit_eip{0};
    // What a chain into this block reads (the generated code addresses it through a link: plain data, fixed layout).
    struct Entry {
        void* body{nullptr};          // past the prologue of the native code: where a chain enters (null: none)
        const u8* host_page{nullptr};  // the code's RAM page on the host (the fetch translation must still give it)
        u64 runs{0};
        u32 cycles{0};                 // the schedule's total (a copy)
        u32 fetch_line{0};             // the first code line (physical address / 32) plus one, as the L1I MRU keeps it
        u8 fetch_lines{0};             // lines the code spans
        u8 n{0};
        u32 jit_eip{0};                // the EIP the native code was made for (a copy)
    } entry;
    // Where each exit of the final instruction went last (a hint: every condition is checked when a chain is taken).
    struct Link {
        Block* to{nullptr};
        u32 epoch{0};
    };
    std::array<Link, 2> links{};
    // The link slots (of other blocks or its own) that name this block, and the fast-cache entries that held it: a
    // kill clears exactly these (BlockEngine::forget), so a write into one code page leaves every other link and entry.
    std::vector<Link*> incoming;
    std::vector<u32> fast_slots;
    bool jit_dirty{false};
    bool jit_failed{false};
    u8 jit_compiles{0};
    std::array<Segment, kSegCount> jit_seg{};
    u32 lin{0};  // the linear address it was first built at (reports)
};

struct BlockStats {
    u64 built{0}, killed{0}, pages_written{0}, device_page_writes{0};
    u64 code_page_writes_kept{0};  // CPU writes near code that overlapped no block (every block kept)
    u64 empty_blocks{0};  // places no block can start (a page-straddling or undecodable first instruction)
    u64 lookups{0}, fast_hits{0};  // find(): calls, and those the direct-mapped cache answered
    u64 relinks{0};  // chains that found their block in the fast cache when the exit's link was missing or elsewhere
    // Every native run's end: [0..15] the chain test that stopped it at a normal block end (0: none was tried, the
    // exit went to the dispatcher), [16] an exit mid-block (a device access, a fault, the limit), [17] a fault.
    std::array<u64, 18> run_end{};
    // Exits that reached the engine's chain instead of the generated code's (the first inline condition that failed):
    // 0 no valid link, 1 budget, 2 limit, 3 pending work, 4 fetch translation, 5 fetch lines not the most recent, 6 an
    // indirect exit or a return (no inline chain), 7 none (the inline chain would have taken it).
    // 8 the spin probe's EIP (the engine compares the state).
    std::array<u64, 9> slow_chain{};
    u64 whole_runs{0}, fallback_steps{0}, block_insns{0};
    u64 mispredicts{0};
    u64 native_runs{0}, native_insns{0}, native_helpers{0}, native_ineligible{0}, native_handoffs{0}, chains{0}, links_refused{0};
    u64 spin_skips{0}, spin_cycles{0};  // loops whose passes were skipped, and the cycles they covered
    u64 countdown_skips{0}, countdown_passes{0};  // countdown loops whose passes were skipped, and the passes
    u64 spin_short_runs{0};  // runs from a known spin loop, ended after one pass
    std::array<u64, 16> spin_elsewhere_why{};  // runs ended away from their start: the chain test that stopped them     // of those, runs that measured their pass from their first return to the start
    // Why a long run was not a skippable pass: an interpreted instruction, a cache penalty, a mispredict, ended
    // elsewhere, the state changed, the probe's log overflowed, a store left memory changed.
    std::array<u64, 7> spin_refused{};
    // Where the cycles went: the schedules, the penalties (caches, devices), the exits' mispredicts, REP elements.
    u64 schedule_cycles{0}, penalty_cycles{0}, exit_cycles{0}, rep_cycles{0};
    std::array<u64, 256 + 256 + 2048> helper_ops{};  // instructions the interpreter ran inside native code, by operation; x87 and SSE by form
};

class Jit;
struct JitCtx;

class BlockEngine : public CodeWatcher {
public:
    BlockEngine(Cpu& cpu, Bus& bus);
    ~BlockEngine() override;

    // The block that starts at the current CS:EIP, built if needed; nullptr when none can be (the code is outside
    // RAM, crosses the code segment's limit, or its first instruction straddles the page's end or faults on fetch or
    // decode): the caller steps.
    Block* find();

    enum class Next : u8 { Continue, Exit };
    // One run of a block: where it started, the penalties (caches, devices) its instructions added so far, and the
    // cycle it must not pass without returning to the caller (the next timeline deadline, the stop cycle).
    struct Run {
        Block* b;
        u64 start;
        u32 entry_eip;
        u64 extra;
        u64 limit;
    };
    // Start a run at the current clock and EIP: the code fetch is looked up in the instruction cache.
    Run begin(Block& b, u64 limit);
    // Execute instruction `i` of the run. Continue: the clock holds the next instruction's start, which is before the
    // limit. Exit: the block is over (its last instruction, a fault, a device access, a write into its own page, the
    // limit reached at the next boundary) and the clock holds the cycle it ended at. So the caller's checks between
    // runs fall on the instruction boundaries a per-instruction loop would make them on.
    Next exec(Run& r, unsigned i);
    // Run a block until it exits; returns the instructions executed.
    unsigned run(Block& b, u64 limit);

    // A step outside any block (the fallback) ran: blocks keep no state across it.
    void note_fallback() { ++stats_.fallback_steps; }

    // Device writes into code pages, applied before the next block runs.
    void apply_device_writes() {
        if (bus_.code_writes_pending()) take_device_writes();
    }
    // Drop every block (a reset).
    void clear();

    // The recompiler (D26): on unless MV_P6_JIT=0 or the host cannot run generated code.
    Jit* jit() { return jit_.get(); }
    void disable_jit();
    // Runs before a block is translated (tests use 0: translate at once).
    void set_jit_threshold(u64 runs) { hot_ = runs; }
    // Blocks a native run may chain into (tests use 0: one block a dispatch, to compare after each).
    void set_chain_budget(u32 blocks) { chain_budget_ = blocks; }
    bool running_dead() const { return running_dead_; }
    // Forget every block's native code (the code buffer was flushed).
    void drop_native();
    // The end of a native block (the generated code calls it): its exit costed, then the next block's native entry if
    // the run may go on there without the dispatcher, else null. See block.cpp.
    void* chain(JitCtx& c, Block::Link* slot);
    // The kernel idle loop's check addresses: never chained into (the dispatcher halts there).
    void set_idle_eips(u32 a, u32 b) {
        idle_ = {a, b};
        ++link_epoch_;  // links made before are not checked against them inline
    }
    // What the generated code reads and updates in the engine (stable addresses for the engine's life).
    struct NativeView {
        u32* link_epoch;
        BlockStats* stats;
        u32* rsb;
        unsigned* rsb_top;
        const u8* code_granules;  // see CodeWatcher
    };
    NativeView native_view() { return {&link_epoch_, &stats_, rsb_.data(), &rsb_top_, code_granules_.data()}; }

    void code_written(u32 ram_page, unsigned off, unsigned len) override;
    const BlockStats& stats() const { return stats_; }
    std::size_t live_blocks() const { return blocks_.size(); }
    // The blocks that took the most cycles (runs x schedule), for reports.
    std::string hottest(unsigned count) const;

private:
    Block* build(const u8* page, unsigned off, u32 ram_page, u8 mode);
    void kill_page(u32 ram_page);
    // The blocks of a page whose code overlaps [off, off + len) dropped (a CPU write); the page's granules redone.
    void kill_range(u32 ram_page, unsigned off, unsigned len);
    void kill_block(Block* b);
    void mark_granules(u32 ram_page);
    // A link slot names `to` (null: nothing), kept in to's incoming list.
    void set_link(Block::Link& slot, Block* to);
    // Every link into and out of `b` cleared (its translation or its assumptions changed: links are learned under
    // equal snapshots, and the generated code's chain does not compare them again).
    void drop_links(Block& b);
    // `b` is going away: links into and out of it, and its fast-cache entries, cleared.
    void forget(Block& b);
    void take_device_writes();
    u32 exit_cost(Block& b, u32 entry_eip);
    unsigned run_native(Run& r);
    void spin_skip(const JitCtx& c, u64 instructions);
    void countdown_skip(Block& b, u64 limit);
    bool stores_undone(const JitCtx& c) const;
    bool predictors_undone() const;
    // Arm the spin probe at EIP `eip`, cycle `at`, after `insns` instructions of the run: the state is copied.
    void arm_probe(JitCtx& c, u32 eip, u64 at, u64 insns);
    // Probe EIPs whose runs did not come back to their state: no more chaining past the budget from there.
    bool not_a_fixed_point(u32 eip) const { return not_fixed_[(eip ^ (eip >> 9)) & 255] == eip; }
    void remember_not_fixed(u32 eip) { not_fixed_[(eip ^ (eip >> 9)) & 255] = eip; }
    std::array<u32, 256> not_fixed_{};
    // Loops the spin skip has skipped: a run starting at one is probed from its start, so it stops after its first
    // pass and the skip is tried at once, instead of after the chain budget — events end such runs thousands of times
    // a second.
    bool known_spin(u32 eip) const { return known_spin_[(eip ^ (eip >> 9)) & 255] == eip; }
    void remember_spin(u32 eip) { known_spin_[(eip ^ (eip >> 9)) & 255] = eip; }
    std::array<u32, 256> known_spin_{};
    static u64 key(u32 ram_page, unsigned off, u8 mode) { return (static_cast<u64>(ram_page) << 14) | (static_cast<u64>(off) << 2) | mode; }
    static unsigned fast_index(u32 lin, u8 mode) { return ((lin ^ (lin >> 16)) * 2 + mode) & (kFastSize - 1); }

    static constexpr unsigned kFastSize = 1u << 16;
    struct FastEntry {
        u32 lin{0xFFFFFFFFu};
        u8 mode{0};
        u32 epoch{0};
        Block* block{nullptr};
    };

    Cpu& cpu_;
    Bus& bus_;
    std::unordered_map<u64, std::unique_ptr<Block>> blocks_;
    std::vector<FastEntry> fast_;
    u32 epoch_{1};  // a kill makes every fast entry stale at once
    u32 link_epoch_{1};  // a kill or a new translation makes every chain link stale at once
    std::vector<std::vector<Block*>> page_blocks_;
    std::vector<u8> code_granules_;  // see CodeWatcher
    std::vector<std::unique_ptr<Block>> graveyard_;
    Block* running_{nullptr};
    bool running_dead_{false};
    std::array<u32, 16> rsb_{};
    unsigned rsb_top_{0};
    std::vector<u32> written_;
    BlockStats stats_;
    std::unique_ptr<Jit> jit_;
    std::exception_ptr jit_error_;
    u64 hot_{8};
    u32 chain_budget_{1024};  // the dispatcher's batch and its lock are not kept waiting for long
    CpuState spin_start_{};  // the state where the probe was armed
    u32 probe_eip_{0};  // where the next run probes a spin loop with stores (0: none)
    bool probing_{false};
    struct ProbeByte {
        const u8* at;
        u32 order;
        u8 original;
    };
    mutable std::vector<ProbeByte> probe_bytes_;
    struct ProbedExit {
        Block* block;
        u8 counter;
        bool seen;
        u32 target;
    };
    std::vector<ProbedExit> probed_exits_;
    std::array<u32, 16> rsb_at_probe_{};
    unsigned rsb_top_at_probe_{0};
    std::array<u32, 2> idle_{};
    JitCtx* current_{nullptr};  // the native run in progress
    Block::Link* pending_slot_{nullptr};  // the last native exit's link slot, filled by the next block run
    u32 pending_target_{0};  // where that exit went (the next block run must start there to be linked)
    Block* pending_from_{nullptr};
    // MV_P6_BLOCK_LOG=FILE:FROM:TO: a line per block run whose start lies in the cycle window (its schedule; start,
    // EIP, a hash of the registers and flags, the cache counters, a hash of the caches' contents and order, the limit),
    // then every cache-model access it makes: to find where two executors part.
    std::FILE* block_log_{nullptr};
    u64 log_from_{0}, log_to_{0};
};

}  // namespace mv::p6
