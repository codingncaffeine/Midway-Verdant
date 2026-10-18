#pragma once
// The interpreter's view of memory: segmentation checks, paging with a TLB, and the bus underneath (RAM through
// its host pointer, everything else through the bus's MMIO dispatch). In oracle mode writes are logged instead
// of applied so the KVM oracle sees the memory the instruction started from.
#include <array>
#include <functional>
#include <vector>

#include "core/bus.h"
#include "core/types.h"
#include "cpu/p6/state.h"

namespace mv::p6 {

enum class Access : u8 { Read, Write, Fetch };

struct WriteRecord {
    PhysAddr phys;
    u8 size;
    bool mmio;
    u32 value;  // or the low word of a wider record (wide writes are split into 4-byte records)
};

struct IoRecord {
    bool is_write;
    bool is_mmio;  // else port I/O
    u32 addr;
    u8 size;
    u32 value;
};

class Memory {
public:
    Memory(Bus& bus, CpuState& st);

    // Before every device access the owner is told, so the timeline can be brought up to date first.
    void set_device_hook(std::function<void()> hook) { device_hook_ = std::move(hook); }

    // Linear-address accesses (after segmentation). Sizes 1, 2, 4; `read_bytes`/`write_bytes` for the wide
    // x87/SSE operands. They throw #PF.
    u32 read(u32 lin, unsigned size);
    void write(u32 lin, unsigned size, u32 value);
    void read_bytes(u32 lin, void* out, unsigned n);
    void write_bytes(u32 lin, const void* in, unsigned n);
    // Check that a write to [lin, lin+n) would not fault (translation with write intent) without writing.
    void probe_write(u32 lin, unsigned n);

    // Translate a linear address; throws #PF. `user` = the access is made at CPL 3.
    PhysAddr translate(u32 lin, Access access);
    // The host pointer behind a linear address when it is RAM (for straight-line code fetch), else nullptr.
    const u8* host_pointer(u32 lin, Access access);

    // A read-only memory range the bus serves through a handler (the flash window): reads are memory, not
    // device traffic, so they are neither logged nor timed; writes are device writes.
    void set_rom_range(PhysAddr base, u64 size) {
        rom_base_ = base;
        rom_size_ = size;
    }

    // Physical accesses (page-table walks, descriptor reads, the oracle). RAM direct; MMIO via the bus.
    u32 read_phys(PhysAddr phys, unsigned size);
    void write_phys(PhysAddr phys, unsigned size, u32 value);
    u64 read_phys64(PhysAddr phys);

    // Port I/O.
    u32 io_read(u16 port, unsigned size);
    void io_write(u16 port, unsigned size, u32 value);

    // TLB maintenance.
    void flush_tlb(bool keep_global);
    void flush_page(u32 lin);

    // Oracle mode: memory writes are recorded, not applied; device reads and writes are logged. `commit`
    // applies the recorded writes (when the oracle is absent for an instruction), `discard` drops them.
    void set_logging(bool on) { logging_ = on; }
    bool logging() const { return logging_; }
    const std::vector<WriteRecord>& writes() const { return writes_; }
    const std::vector<IoRecord>& io_log() const { return io_log_; }
    void begin_instruction() {
        writes_.clear();
        io_log_.clear();
        naccesses_ = 0;
    }
    void commit_writes();

    // The physical accesses of the current instruction, for the cycle model's caches (capped; a long string
    // step is scaled by its element count instead).
    struct AccessRecord {
        PhysAddr phys;
        u8 size;
        u8 kind;  // 0 load, 1 store, 2 device load, 3 device store, 4 port in, 5 port out
    };
    static constexpr unsigned kMaxAccesses = 24;
    const AccessRecord* accesses() const { return accesses_.data(); }
    unsigned access_count() const { return naccesses_; }

    // Dirty-page tracking for the capture instruments (RAM pages the CPU wrote).
    void set_dirty_tracking(bool on);
    void take_dirty_pages(std::vector<u32>& pages);

    Bus& bus() { return bus_; }

    u64 tlb_misses() const { return tlb_misses_; }
    u64 page_walks() const { return page_walks_; }

private:
    struct TlbEntry {
        u32 tag{0xFFFFFFFFu};  // linear page number, or ~0 when empty
        PhysAddr phys_page{0};
        u8* host{nullptr};  // RAM page, or nullptr for MMIO
        bool writable{false};  // a write walk succeeded (the dirty bit is set)
        bool user{false};
        bool global{false};
    };
    static constexpr unsigned kTlbBits = 10;
    static constexpr unsigned kTlbSize = 1u << kTlbBits;

    TlbEntry& lookup(u32 lin, Access access);
    TlbEntry& walk(u32 lin, Access access);
    [[noreturn]] void page_fault(u32 lin, Access access, bool present, bool reserved);
    void mark_dirty(PhysAddr phys, unsigned n);
    u8* ram_at(PhysAddr phys, unsigned n) { return bus_.ram_pointer(phys, n); }
    void write_phys_logged(PhysAddr phys, unsigned size, u32 value, bool mmio);
    bool read_from_log(PhysAddr phys, unsigned size, u32& value) const;

    Bus& bus_;
    CpuState& st_;
    std::array<TlbEntry, kTlbSize> tlb_{};
    std::function<void()> device_hook_;
    bool logging_{false};
    std::vector<WriteRecord> writes_;
    std::vector<IoRecord> io_log_;
    bool dirty_tracking_{false};
    std::vector<u64> dirty_;
    u64 tlb_misses_{0}, page_walks_{0};
    PhysAddr rom_base_{0xFF000000u};
    u64 rom_size_{16u * 1024 * 1024};
    bool in_rom(PhysAddr phys) const { return phys >= rom_base_ && phys - rom_base_ < rom_size_; }
    std::array<AccessRecord, kMaxAccesses> accesses_{};
    unsigned naccesses_{0};
    void note(PhysAddr phys, unsigned size, u8 kind) {
        if (naccesses_ < kMaxAccesses) accesses_[naccesses_++] = AccessRecord{phys, static_cast<u8>(size), kind};
    }
};

}  // namespace mv::p6
