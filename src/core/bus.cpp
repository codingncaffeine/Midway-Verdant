#include "core/bus.h"

#include <cstdlib>
#include <cstring>
#include <new>

#include "debug/trace.h"

namespace mv {

Bus::Bus(std::size_t ram_size) : ram_(ram_size), ram_size_(ram_size), pages_(1u << 20, nullptr), pio_(1u << 16, nullptr) {}

Bus::~Bus() = default;

void Bus::map_mmio(PhysAddr base, u64 size, MmioHandler* handler) {
    const u64 first = base / kPageSize;
    const u64 count = (size + kPageSize - 1) / kPageSize;
    for (u64 p = first; p < first + count && p < pages_.size(); ++p) pages_[static_cast<std::size_t>(p)] = handler;
}

void Bus::unmap_mmio(PhysAddr base, u64 size) { map_mmio(base, size, nullptr); }

MmioHandler* Bus::mmio_at(PhysAddr addr) const { return pages_[addr / kPageSize]; }

void Bus::map_pio(u16 base, u32 count, PioHandler* handler) {
    for (u32 p = base; p < static_cast<u32>(base) + count && p < pio_.size(); ++p) pio_[p] = handler;
}

void Bus::unmap_pio(u16 base, u32 count) { map_pio(base, count, nullptr); }

u8* Bus::ram_pointer(PhysAddr addr, std::size_t len) {
    u64 off;
    if (addr < ram_size_) {
        off = addr;
    } else if (addr >= kRamAliasBase && addr - kRamAliasBase < ram_size_) {
        off = addr - kRamAliasBase;
    } else {
        return nullptr;
    }
    if (off + len > ram_size_) return nullptr;
    return ram_.data() + off;
}

const u8* Bus::ram_pointer(PhysAddr addr, std::size_t len) const {
    return const_cast<Bus*>(this)->ram_pointer(addr, len);
}

bool Bus::read_ram(PhysAddr addr, void* out, std::size_t len) const {
    const u8* p = ram_pointer(addr, len);
    if (!p) return false;
    std::memcpy(out, p, len);
    return true;
}

bool Bus::write_ram(PhysAddr addr, const void* in, std::size_t len) {
    u8* p = ram_pointer(addr, len);
    if (!p) return false;
    std::memcpy(p, in, len);
    return true;
}

u32 Bus::read(PhysAddr addr, unsigned size) {
    if (u8* p = ram_pointer(addr, size)) {
        u32 v = 0;
        std::memcpy(&v, p, size);
        return v;
    }
    if (MmioHandler* h = mmio_at(addr)) return h->mmio_read(addr, size);
    note_unknown_mmio(addr, size, false, 0);
    return size == 4 ? 0xFFFFFFFFu : (size == 2 ? 0xFFFFu : 0xFFu);
}

void Bus::write(PhysAddr addr, unsigned size, u32 value) {
    if (u8* p = ram_pointer(addr, size)) {
        std::memcpy(p, &value, size);
        return;
    }
    if (MmioHandler* h = mmio_at(addr)) {
        h->mmio_write(addr, size, value);
        return;
    }
    note_unknown_mmio(addr, size, true, value);
}

u64 Bus::read64(PhysAddr addr) {
    if (u8* p = ram_pointer(addr, 8)) {
        u64 v;
        std::memcpy(&v, p, 8);
        return v;
    }
    return static_cast<u64>(read(addr, 4)) | (static_cast<u64>(read(addr + 4, 4)) << 32);
}

void Bus::write64(PhysAddr addr, u64 value) {
    if (u8* p = ram_pointer(addr, 8)) {
        std::memcpy(p, &value, 8);
        return;
    }
    write(addr, 4, static_cast<u32>(value));
    write(addr + 4, 4, static_cast<u32>(value >> 32));
}

u32 Bus::pio_read(u16 port, unsigned size) {
    if (PioHandler* h = pio_[port]) return h->pio_read(port, size);
    note_unknown_pio(port, size, false, 0);
    return size == 4 ? 0xFFFFFFFFu : (size == 2 ? 0xFFFFu : 0xFFu);
}

void Bus::pio_write(u16 port, unsigned size, u32 value) {
    if (PioHandler* h = pio_[port]) {
        h->pio_write(port, size, value);
        return;
    }
    note_unknown_pio(port, size, true, value);
}

void Bus::note_unknown_mmio(PhysAddr addr, unsigned size, bool is_write, u32 value) {
    auto& n = unknown_mmio_[{addr & ~PhysAddr{kPageSize - 1}, is_write}];
    if (n < 8) {
        if (is_write)
            trace::log(trace::Cat::Mmio, "unmapped write {:08x} size {} value {:08x}", addr, size, value);
        else
            trace::log(trace::Cat::Mmio, "unmapped read  {:08x} size {}", addr, size);
    }
    ++n;
}

void Bus::note_unknown_pio(u16 port, unsigned size, bool is_write, u32 value) {
    auto& n = unknown_pio_[{port, is_write}];
    if (n < 8) {
        if (is_write)
            trace::log(trace::Cat::Pio, "unmapped out {:04x} size {} value {:08x}", port, size, value);
        else
            trace::log(trace::Cat::Pio, "unmapped in  {:04x} size {}", port, size);
    }
    ++n;
}

std::vector<Bus::Unknown> Bus::unknown_mmio() const {
    std::vector<Unknown> out;
    for (const auto& [key, count] : unknown_mmio_) out.push_back({key.second, key.first, count});
    return out;
}

std::vector<Bus::Unknown> Bus::unknown_pio() const {
    std::vector<Unknown> out;
    for (const auto& [key, count] : unknown_pio_) out.push_back({key.second, key.first, count});
    return out;
}

}  // namespace mv
