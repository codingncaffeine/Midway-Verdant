#pragma once
// The cycle model of the Pentium III (Coppermine) core: a P6 pipeline accounting — the three decoders (4-1-1),
// register renaming into a 40-entry reorder buffer, issue to the five ports with the µop counts and latencies
// of Agner Fog's Pentium II/III tables, in-order retirement three µops a cycle, the L1 instruction and data
// caches and the on-die L2 as tag arrays with LRU, a branch target buffer with two-bit counters and a return
// stack, and the uncached latencies of device and port accesses. A cycle COUNT whose observable orderings are
// checked against hardware where ground truth exists; it does not claim exactness.
#include <array>
#include <cstdio>

#include "core/types.h"
#include "cpu/p6/decode.h"

namespace mv::p6 {

// Registers of the dependency scoreboard.
inline constexpr unsigned kRegFlags = 8, kRegX87 = 9, kRegXmm = 17, kRegSeg = 25, kRegCount = 31;

struct MemAccess {
    PhysAddr phys;
    u8 size;
    u8 kind;  // 0 load, 1 store, 2 device load, 3 device store, 4 port in, 5 port out
};

struct InsnTiming {
    const Insn* in;
    PhysAddr fetch_phys;
    u8 len;
    u32 reads, writes;  // scoreboard register bitmasks
    bool is_branch, taken, is_call, is_ret, is_indirect;
    u32 branch_eip, target;
    const MemAccess* accesses;
    unsigned naccesses;
    unsigned rep_iterations;  // string elements this step ran (0 for other instructions)
    bool interrupt_delivery;  // an external interrupt was delivered instead of an instruction
};

struct TimingStats {
    u64 instructions{0}, uops{0}, cycles{0};
    u64 l1i_misses{0}, l1d_misses{0}, l2_misses{0};
    u64 branches{0}, mispredicts{0};
    u64 device_loads{0}, device_stores{0}, port_ios{0};
    u64 stall_dependency{0}, stall_rob{0}, stall_fetch{0};
};

// Model parameters (cycles at 733 MHz). Marked [verify] in local-notes/hw/cpu-p6.md where guessed.
namespace tp {
inline constexpr unsigned kFetchToDecode = 3;    // IFU stages
inline constexpr unsigned kDecodeToIssue = 3;    // ID2, RAT, RS
inline constexpr unsigned kMispredictBubble = 9; // front-end restart after a resolved mispredict
inline constexpr unsigned kLoadLatency = 3;      // L1 hit
inline constexpr unsigned kL2Latency = 7;        // added on an L1 miss [verify]
inline constexpr unsigned kDramLatency = 90;     // added on an L2 miss [verify]
inline constexpr unsigned kDeviceLoad = 160;     // an uncached read from the chipset [verify]
inline constexpr unsigned kDeviceStore = 8;      // posted
inline constexpr unsigned kPortIo = 300;         // IN/OUT [verify]
inline constexpr unsigned kInterruptDelivery = 40;
inline constexpr unsigned kRobEntries = 40;
inline constexpr unsigned kRetireWidth = 3;
}  // namespace tp

class Timing {
public:
    Timing();
    void reset();
    // Account for one retired instruction (or an interrupt delivery); returns its retirement cycle.
    u64 retire(const InsnTiming& t);
    // The core slept to `cycle` (HLT): every pipeline clock moves there.
    void skip_to(u64 cycle);
    u64 now() const { return retire_cycle_; }
    // A hash of the caches' contents and LRU order (instruments: where two runs' caches part).
    u64 cache_hash() const;
    // The data cache's most-recently-used lines (see Cache::mru_) for the recompiler.
    const u32* l1d_mru() const { return l1d_.mru_.data(); }
    const u32* l1i_mru() const { return l1i_.mru_.data(); }
    unsigned l1i_sets() const { return l1i_.sets_; }
    unsigned l1d_sets() const { return l1d_.sets_; }
    // The L1 data cache's tags (line plus one, ways of a set together) and ages, and the clock that ages them: the
    // recompiler's generated code settles a hit on any way itself (the model's own update: the way's age from the
    // clock, the set's most recent line).
    const u32* l1d_tags() const { return l1d_.tags_.data(); }
    u64* l1d_last_use() { return l1d_.last_use_.data(); }
    unsigned l1d_ways() const { return l1d_.ways_; }
    u64* lru_clock_word() { return &lru_clock_; }
    // The same for the L1 instruction cache (the chain's code fetch).
    const u32* l1i_tags() const { return l1i_.tags_.data(); }
    u64* l1i_last_use() { return l1i_.last_use_.data(); }
    unsigned l1i_ways() const { return l1i_.ways_; }
    u32* l1i_mru_mutable() { return l1i_.mru_.data(); }
    u32* l1d_mru_mutable() { return l1d_.mru_.data(); }
    bool ideal() const { return ideal_; }
    const TimingStats& stats() const { return stats_; }
    // Block timing (D26): an ideal pipeline (code and data hit L1, branches are predicted) that schedules a translated
    // block once; its caches and predictor are never touched.
    void set_ideal(bool on) { ideal_ = on; }
    // The pipeline's clocks back to zero; caches, predictor and statistics kept.
    void reset_pipeline();
    // Block timing's cache terms (D26), on the same L1/L2 tag arrays: a load's penalty over an L1 hit (a store's is
    // posted: 0), and a block's code fetch. LRU ages from an access count (deterministic).
    // A hit on a set's most recent line changes no LRU order (the clock only grows here), so it skips the model.
    unsigned data_access(PhysAddr phys, bool store) {
        if (trace_) std::fprintf(trace_, "  %c %08x\n", store ? 'w' : 'r', phys);
        const u32 line = phys >> 5;
        if (l1d_.mru_[line & (l1d_.sets_ - 1)] == line + 1) return 0;
        return memory_penalty(phys, ++lru_clock_, store);
    }
    std::FILE* trace_{nullptr};  // instruments: every block-timing data access, when set
    unsigned code_fetch(PhysAddr phys, unsigned len);
    // Every register ready (the next instructions do not depend on the earlier ones).
    void clear_dependencies() { reg_ready_.fill(0); }

private:
    struct Cache {
        Cache(unsigned sets, unsigned ways);
        bool access(PhysAddr phys, u64 now) { return ways_ == 4 ? access_ways<4>(phys, now) : access_ways<8>(phys, now); }  // true on a hit; allocates on a miss
        template <unsigned W>
        bool access_ways(PhysAddr phys, u64 now);
        // The most recently used line of each set, plus one (0: none): a hit on it changes no LRU order, so the
        // recompiler's generated code checks it inline and skips the model.
        std::array<u32, 512> mru_{};
        unsigned sets_, ways_;
        std::array<u32, 4096> tags_{};  // the line plus one (0: the way is empty)
        std::array<u64, 4096> last_use_{};
    };
    struct Btb {
        u32 tag{0};
        u32 target{0};
        u8 counter{0};
    };

    u64 alloc_port(unsigned port, u64 earliest);
    unsigned memory_penalty(PhysAddr phys, u64 now, bool store);
    u64 fetch_ready(PhysAddr phys, unsigned len, u64 now);

    u64 decode_cycle_{0};
    unsigned decode_slot_{0}, decode_bytes_{0};
    u64 front_end_restart_{0};
    std::array<u64, 5> port_free_{};
    std::array<u64, kRegCount> reg_ready_{};
    u64 retire_cycle_{0};
    unsigned retire_slot_{0};
    std::array<u64, tp::kRobEntries> rob_{};
    unsigned rob_pos_{0};
    u64 int_div_free_{0}, fp_div_free_{0};
    Cache l1i_{128, 4}, l1d_{128, 4}, l2_{512, 8};
    std::array<Btb, 512> btb_{};
    std::array<u32, 16> rsb_{};
    unsigned rsb_top_{0};
    TimingStats stats_;
    bool ideal_{false};
    u64 lru_clock_{0};
};

// The scoreboard registers an instruction reads and writes (x87 registers by physical number from `top`).
void regs_of(const Insn& in, unsigned top, u32& reads, u32& writes);

// The block schedule (D26): `after[i]` = cycles from the block's start to instruction i's retirement in the steady
// state (the block scheduled twice back to back on the ideal pipeline, the second pass measured with its registers
// ready, so the first pass leaves only its use of the decoders, ports and reorder buffer behind; x87 registers by
// stack position, the pushes and pops inside the block not followed). Returns the block's cost, at least one cycle.
u32 schedule_block(Timing& ideal, const Insn* const* insns, unsigned n, u16* after);

// Block timing's dynamic terms, added where the run time knows them (cycles at 733 MHz).
namespace tp {
inline constexpr unsigned kBlockDeviceLoad = kDeviceLoad - kLoadLatency;  // over the schedule's L1 hit
inline constexpr unsigned kBlockDeviceStore = kDeviceStore;
inline constexpr unsigned kBlockPortIo = kPortIo;
inline constexpr unsigned kBlockMispredict = kMispredictBubble + kDecodeToIssue;  // [verify]
}  // namespace tp

}  // namespace mv::p6
