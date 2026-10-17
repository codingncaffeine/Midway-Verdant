#pragma once
#include <cstddef>
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

    AlignedBuffer ram_;
    std::size_t ram_size_;
    std::vector<MmioHandler*> pages_;  // one entry per 4 KiB page of the 4 GiB space
    std::vector<PioHandler*> pio_;     // 65536 ports
    std::map<std::pair<PhysAddr, bool>, u64> unknown_mmio_;
    std::map<std::pair<u16, bool>, u64> unknown_pio_;
};

}  // namespace mv
