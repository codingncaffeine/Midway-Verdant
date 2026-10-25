#pragma once
#include <atomic>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>

#include "core/aligned_buffer.h"
#include "core/types.h"

namespace mv {

// A device reachable through memory-mapped registers. `addr` is the full physical address; sizes are 1, 2, 4
// (and 8 for the rare 64-bit MMIO, split by the bus into two 32-bit accesses unless the device overrides).
class MmioHandler {
public:
    virtual ~MmioHandler() = default;
    virtual std::string_view name() const = 0;
    virtual u32 mmio_read(PhysAddr addr, unsigned size) = 0;
    virtual void mmio_write(PhysAddr addr, unsigned size, u32 value) = 0;
};

// A device reachable through x86 port I/O.
class PioHandler {
public:
    virtual ~PioHandler() = default;
    virtual std::string_view name() const = 0;
    virtual u32 pio_read(u16 port, unsigned size) = 0;
    virtual void pio_write(u16 port, unsigned size, u32 value) = 0;
};

// The physical address space and the port space of the machine. RAM is a host buffer; MMIO regions are
// registered in 4 KiB pages; the GPU-side alias of RAM at 0xF0000000 is served by the bus itself.
// Accesses to nothing are logged (coalesced per page) and read as all ones, like an open bus.
class Bus {
public:
    static constexpr PhysAddr kRamAliasBase = 0xF0000000u;
    static constexpr u64 kRamAliasSize = 128u * 1024 * 1024;
    static constexpr u64 kPageSize = 4096;

    explicit Bus(std::size_t ram_size);
    ~Bus();
    Bus(const Bus&) = delete;
    Bus& operator=(const Bus&) = delete;

    u8* ram() { return ram_.data(); }
    const u8* ram() const { return ram_.data(); }
    std::size_t ram_size() const { return ram_size_; }

    // Regions are page-aligned and page-sized; overlapping an existing mapping replaces it.
    void map_mmio(PhysAddr base, u64 size, MmioHandler* handler);
    void unmap_mmio(PhysAddr base, u64 size);
    MmioHandler* mmio_at(PhysAddr addr) const;
    // A device's hint: its writes to [base, base + size) may reach it later, in order, at the CPU's next exit or the
    // io thread's next event pass (Machine::set_before_events), without an exit each (KVM's coalesced MMIO). Only for
    // registers whose writes nothing observes before the guest's next access to the device: that access exits, and
    // the writes before it land first. The CPU backend that can take writes so installs the hook.
    using CoalesceHook = std::function<void(PhysAddr base, u32 size, bool on)>;
    void coalesce_writes(PhysAddr base, u32 size, bool on);
    void set_coalesce_hook(CoalesceHook hook);

    void map_pio(u16 base, u32 count, PioHandler* handler);
    void unmap_pio(u16 base, u32 count);
    PioHandler* pio_at(u16 port) const { return pio_[port]; }

    // Guest-visible accesses. Sizes 1, 2, 4; `read64`/`write64` serve 8-byte MMIO.
    u32 read(PhysAddr addr, unsigned size);
    void write(PhysAddr addr, unsigned size, u32 value);
    u64 read64(PhysAddr addr);
    void write64(PhysAddr addr, u64 value);
    // Bulk access to RAM through the alias too; returns false if any byte is outside RAM.
    bool read_ram(PhysAddr addr, void* out, std::size_t len) const;
    bool write_ram(PhysAddr addr, const void* in, std::size_t len);

    u32 pio_read(u16 port, unsigned size);
    void pio_write(u16 port, unsigned size, u32 value);

    // Where an address lands in RAM (direct or through the alias), or nullptr.
    u8* ram_pointer(PhysAddr addr, std::size_t len);
    const u8* ram_pointer(PhysAddr addr, std::size_t len) const;

    // Dirty-page tracking for the capture instruments: when on, every write that lands in RAM through the bus
    // (device DMA, semaphores, notifiers; the CPU backends keep their own record) marks its 4 KiB pages.
    // `take_dirty_pages` appends the marked page indices and clears the record. Devices on other threads (the APU's
    // processors) mark pages too, so the record is atomic words.
    void set_dirty_tracking(bool on);
    bool dirty_tracking() const { return dirty_tracking_; }
    void take_dirty_pages(std::vector<u32>& pages);
    void mark_dirty(PhysAddr addr, std::size_t len);

    // Pages holding translated code (D26), by RAM page index. A write into one through the bus (device DMA, a device's
    // own RAM writes, from any thread) is recorded for the CPU thread, which invalidates the page's blocks before it
    // runs another; the CPU's own writes are checked by its memory unit.
    void set_code_tracking(bool on);
    bool code_tracking() const { return code_tracking_; }
    void set_code_page(u32 page, bool on) {
        const u64 bit = u64{1} << (page % 64);
        if (on) code_[page / 64].fetch_or(bit, std::memory_order_relaxed);
        else code_[page / 64].fetch_and(~bit, std::memory_order_relaxed);
    }
    bool code_page(u32 page) const { return code_tracking_ && (code_[page / 64].load(std::memory_order_relaxed) & (u64{1} << (page % 64))) != 0; }
    // The RAM page index of a host pointer into RAM.
    u32 ram_page(const u8* p) const { return static_cast<u32>(static_cast<u64>(p - ram_.data()) / kPageSize); }
    // The code-page bitmap as plain words (the recompiler's generated code tests it).
    const u64* code_map() const { return reinterpret_cast<const u64*>(code_.get()); }
    bool code_writes_pending() const { return code_pending_.load(std::memory_order_acquire); }
    // The pending flag's byte, for generated code (read with a plain load: x86 orders it as acquire).
    const void* code_pending_byte() const { return &code_pending_; }
    void take_code_writes(std::vector<u32>& pages);
    // A word a write into code zeroes, after it records the write (the CPU's run limit for chaining: D29 S1). The CPU
    // thread, after it writes the word, looks at code_writes_pending() again (a full fence between), so a write is
    // never missed: either it sees the write pending, or the write's zero lands after its own store.
    void set_code_write_kill(u64* word) { code_write_kill_.store(word, std::memory_order_seq_cst); }

    // Unknown-access bookkeeping: (page, is_write) -> count; printed by the runner at the end.
    struct Unknown {
        bool is_write;
        PhysAddr page;
        u64 count;
    };
    std::vector<Unknown> unknown_mmio() const;
    std::vector<Unknown> unknown_pio() const;

private:
    void note_unknown_mmio(PhysAddr addr, unsigned size, bool is_write, u32 value);
    void note_unknown_pio(u16 port, unsigned size, bool is_write, u32 value);
    void note_code_write(const u8* p, std::size_t len);

    AlignedBuffer ram_;
    std::size_t ram_size_;
    std::vector<MmioHandler*> pages_;  // one entry per 4 KiB page of the 4 GiB space
    std::vector<PioHandler*> pio_;     // 65536 ports
    std::vector<std::pair<PhysAddr, u32>> coalesced_;  // the zones devices asked for (coalesce_writes)
    CoalesceHook coalesce_hook_;
    std::map<std::pair<PhysAddr, bool>, u64> unknown_mmio_;
    std::map<std::pair<u16, bool>, u64> unknown_pio_;
    bool dirty_tracking_{false};
    std::unique_ptr<std::atomic<u64>[]> dirty_;  // one bit per RAM page
    std::size_t dirty_words_{0};
    bool code_tracking_{false};
    std::unique_ptr<std::atomic<u64>[]> code_, code_written_;
    std::size_t code_words_{0};
    std::atomic<bool> code_pending_{false};
    std::atomic<u64*> code_write_kill_{nullptr};
};

}  // namespace mv
