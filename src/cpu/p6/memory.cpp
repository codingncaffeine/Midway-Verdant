#include "cpu/p6/memory.h"

#include <bit>
#include <cstring>

namespace mv::p6 {

namespace {
constexpr u32 kPageMask = 0xFFFu;
// Page-table entry bits (SDM Vol. 3, paging).
constexpr u32 kPtePresent = 1u << 0, kPteWrite = 1u << 1, kPteUser = 1u << 2, kPteAccessed = 1u << 5, kPteDirty = 1u << 6,
              kPdePageSize = 1u << 7, kPteGlobal = 1u << 8;
}  // namespace

Memory::Memory(Bus& bus, CpuState& st) : bus_(bus), st_(st) {}

void Memory::flush_tlb(bool keep_global) {
    for (auto& e : tlb_) {
        if (keep_global && e.global && e.tag != 0xFFFFFFFFu) continue;
        e.tag = 0xFFFFFFFFu;
    }
}

void Memory::flush_page(u32 lin) { tlb_[(lin >> 12) & (kTlbSize - 1)].tag = 0xFFFFFFFFu; }

void Memory::page_fault(u32 lin, Access access, bool present, bool reserved) {
    st_.cr2 = lin;
    u32 code = 0;
    if (present) code |= 1;
    if (access == Access::Write) code |= 2;
    if (st_.cpl() == 3) code |= 4;
    if (reserved) code |= 8;
    raise(kPF, code);
}

Memory::TlbEntry& Memory::walk(u32 lin, Access access) {
    ++page_walks_;
    const bool user = st_.cpl() == 3;
    const bool write = access == Access::Write;
    const bool wp = (st_.cr0 & cr0::WP) != 0;
    const PhysAddr pde_addr = (st_.cr3 & ~kPageMask) + ((lin >> 22) << 2);
    u32 pde = read_phys(pde_addr, 4);
    if (!(pde & kPtePresent)) page_fault(lin, access, false, false);
    PhysAddr phys_page;
    u32 pte = 0;
    PhysAddr pte_addr = 0;
    bool large = false;
    if ((pde & kPdePageSize) && (st_.cr4 & cr4::PSE)) {
        large = true;
        // PSE-36: bits 16..13 of the PDE are address bits 35..32; anything there is beyond the 32-bit bus.
        if (pde & 0x1E000u) page_fault(lin, access, true, true);
        phys_page = (pde & 0xFFC00000u) | (lin & 0x003FF000u);
        pte = pde;
    } else {
        pte_addr = (pde & ~kPageMask) + (((lin >> 12) & 0x3FF) << 2);
        pte = read_phys(pte_addr, 4);
        if (!(pte & kPtePresent)) page_fault(lin, access, false, false);
        phys_page = pte & ~kPageMask;
    }
    // Permission: user pages need U/S at both levels; writes need R/W at both levels for user, and for
    // supervisor when CR0.WP is set.
    const u32 both = large ? pde : (pde & pte);
    if (user) {
        if (!(both & kPteUser)) page_fault(lin, access, true, false);
        if (write && !(both & kPteWrite)) page_fault(lin, access, true, false);
    } else if (write && wp && !(both & kPteWrite)) {
        page_fault(lin, access, true, false);
    }
    // Accessed and dirty bits, as the hardware sets them (the PDE's A on every walk; D in the leaf on a write).
    if (!(pde & kPteAccessed)) {
        pde |= kPteAccessed;
        if (large && write) pde |= kPteDirty;
        write_phys(pde_addr, 4, pde);
    } else if (large && write && !(pde & kPteDirty)) {
        write_phys(pde_addr, 4, pde | kPteDirty);
    }
    if (!large) {
        u32 updated = pte;
        if (!(updated & kPteAccessed)) updated |= kPteAccessed;
        if (write && !(updated & kPteDirty)) updated |= kPteDirty;
        if (updated != pte) write_phys(pte_addr, 4, updated);
    }
    TlbEntry& e = tlb_[(lin >> 12) & (kTlbSize - 1)];
    e.tag = lin >> 12;
    e.phys_page = phys_page;
    e.host = ram_at(phys_page, 4096);
    const bool dirty_set = large ? ((pde & kPteDirty) != 0 || write) : ((pte & kPteDirty) != 0 || write);
    const bool writable_perm = user ? (both & kPteWrite) != 0 : (!wp || (both & kPteWrite) != 0);
    e.writable = writable_perm && dirty_set;
    e.user = (both & kPteUser) != 0;
    e.global = (st_.cr4 & cr4::PGE) != 0 && (pte & kPteGlobal) != 0;
    return e;
}

inline Memory::TlbEntry& Memory::lookup(u32 lin, Access access) {
    TlbEntry& e = tlb_[(lin >> 12) & (kTlbSize - 1)];
    if (e.tag == (lin >> 12) && (access != Access::Write || e.writable)) return e;
    ++tlb_misses_;
    return walk(lin, access);
}

PhysAddr Memory::translate(u32 lin, Access access) {
    if (!st_.paging()) return lin;
    const TlbEntry& e = lookup(lin, access);
    return e.phys_page | (lin & kPageMask);
}

bool Memory::read_from_log(PhysAddr phys, unsigned size, u32& value) const {
    // The newest record wins; a partial overlap is merged byte by byte.
    bool any = false;
    for (unsigned i = 0; i < size; ++i) {
        for (auto it = writes_.rbegin(); it != writes_.rend(); ++it) {
            if (phys + i >= it->phys && phys + i < it->phys + it->size) {
                const u32 byte = (it->value >> (8 * (phys + i - it->phys))) & 0xFF;
                value = (value & ~(0xFFu << (8 * i))) | (byte << (8 * i));
                any = true;
                break;
            }
        }
    }
    return any;
}

const u8* Memory::host_pointer(u32 lin, Access access) {
    if (!st_.paging()) return ram_at(lin, 1);
    const TlbEntry& e = lookup(lin, access);
    return e.host ? e.host + (lin & kPageMask) : nullptr;
}

u32 Memory::read_phys(PhysAddr phys, unsigned size) {
    if (const u8* p = ram_at(phys, size)) {
        u32 v = 0;
        std::memcpy(&v, p, size);
        if (logging_ && !writes_.empty()) read_from_log(phys, size, v);
        note(phys, size, 0);
        return v;
    }
    if (in_rom(phys)) {
        note(phys, size, 0);
        return bus_.read(phys, size);
    }
    if (device_hook_) device_hook_();
    note(phys, size, 2);
    const u32 v = bus_.read(phys, size);
    if (logging_) io_log_.push_back({false, true, phys, static_cast<u8>(size), v});
    return v;
}

u64 Memory::read_phys64(PhysAddr phys) {
    const u64 lo = read_phys(phys, 4);
    return lo | (static_cast<u64>(read_phys(phys + 4, 4)) << 32);
}

void Memory::write_phys_logged(PhysAddr phys, unsigned size, u32 value, bool mmio) {
    writes_.push_back({phys, static_cast<u8>(size), mmio, value});
}

void Memory::write_phys(PhysAddr phys, unsigned size, u32 value) {
    if (u8* p = ram_at(phys, size)) {
        note(phys, size, 1);
        if (logging_) {
            write_phys_logged(phys, size, value, false);
            return;
        }
        std::memcpy(p, &value, size);
        if (dirty_tracking_) mark_dirty(phys, size);
        return;
    }
    if (device_hook_) device_hook_();
    note(phys, size, 3);
    if (logging_) io_log_.push_back({true, true, phys, static_cast<u8>(size), value});
    bus_.write(phys, size, value);
}

void Memory::commit_writes() {
    for (const auto& w : writes_) {
        if (u8* p = ram_at(w.phys, w.size)) {
            std::memcpy(p, &w.value, w.size);
            if (dirty_tracking_) mark_dirty(w.phys, w.size);
        }
    }
    writes_.clear();
}

u32 Memory::read(u32 lin, unsigned size) {
    if ((lin & kPageMask) + size > 4096) {
        // Crosses a page: both pages are translated before anything is read.
        const u32 first = 4096 - (lin & kPageMask);
        u32 v = 0;
        for (unsigned i = 0; i < size; ++i) {
            const PhysAddr p = translate(lin + i, Access::Read);
            v |= read_phys(p, 1) << (8 * i);
        }
        (void)first;
        return v;
    }
    if (!st_.paging()) return read_phys(lin, size);
    const TlbEntry& e = lookup(lin, Access::Read);
    if (e.host && !(logging_ && !writes_.empty())) {
        u32 v = 0;
        std::memcpy(&v, e.host + (lin & kPageMask), size);
        note(e.phys_page | (lin & kPageMask), size, 0);
        return v;
    }
    return read_phys(e.phys_page | (lin & kPageMask), size);
}

void Memory::write(u32 lin, unsigned size, u32 value) {
    if ((lin & kPageMask) + size > 4096) {
        PhysAddr phys[4];
        for (unsigned i = 0; i < size; ++i) phys[i] = translate(lin + i, Access::Write);
        for (unsigned i = 0; i < size; ++i) write_phys(phys[i], 1, (value >> (8 * i)) & 0xFF);
        return;
    }
    if (!st_.paging()) {
        write_phys(lin, size, value);
        return;
    }
    const TlbEntry& e = lookup(lin, Access::Write);
    if (e.host && !logging_) {
        std::memcpy(e.host + (lin & kPageMask), &value, size);
        if (dirty_tracking_) mark_dirty(e.phys_page, 1);
        note(e.phys_page | (lin & kPageMask), size, 1);
        return;
    }
    write_phys(e.phys_page | (lin & kPageMask), size, value);
}

void Memory::read_bytes(u32 lin, void* out, unsigned n) {
    u8* o = static_cast<u8*>(out);
    unsigned i = 0;
    while (i < n) {
        const unsigned chunk = (n - i >= 4 && ((lin + i) & 3) == 0) ? 4 : 1;
        const u32 v = read(lin + i, chunk);
        std::memcpy(o + i, &v, chunk);
        i += chunk;
    }
}

void Memory::write_bytes(u32 lin, const void* in, unsigned n) {
    probe_write(lin, n);
    const u8* s = static_cast<const u8*>(in);
    unsigned i = 0;
    while (i < n) {
        const unsigned chunk = (n - i >= 4 && ((lin + i) & 3) == 0) ? 4 : 1;
        u32 v = 0;
        std::memcpy(&v, s + i, chunk);
        write(lin + i, chunk, v);
        i += chunk;
    }
}

void Memory::probe_write(u32 lin, unsigned n) {
    if (!st_.paging() || n == 0) return;
    for (u32 page = lin & ~kPageMask; page <= ((lin + n - 1) & ~kPageMask); page += 4096) {
        lookup(page, Access::Write);
        if (page == 0xFFFFF000u) break;
    }
}

u32 Memory::io_read(u16 port, unsigned size) {
    if (device_hook_) device_hook_();
    note(port, size, 4);
    const u32 v = bus_.pio_read(port, size);
    if (logging_) io_log_.push_back({false, false, port, static_cast<u8>(size), v});
    return v;
}

void Memory::io_write(u16 port, unsigned size, u32 value) {
    if (device_hook_) device_hook_();
    note(port, size, 5);
    if (logging_) io_log_.push_back({true, false, port, static_cast<u8>(size), value});
    bus_.pio_write(port, size, value);
}

void Memory::set_dirty_tracking(bool on) {
    dirty_tracking_ = on;
    dirty_.assign(on ? (bus_.ram_size() / 4096 + 63) / 64 : 0, 0);
}

void Memory::mark_dirty(PhysAddr phys, unsigned n) {
    const u8* p = ram_at(phys, n);
    if (!p) return;
    const u64 start = static_cast<u64>(p - bus_.ram());
    const u64 first = start / 4096;
    const u64 last = (start + n - 1) / 4096;
    for (u64 page = first; page <= last; ++page) dirty_[page / 64] |= u64{1} << (page % 64);
}

void Memory::take_dirty_pages(std::vector<u32>& pages) {
    for (std::size_t w = 0; w < dirty_.size(); ++w) {
        u64 bits = dirty_[w];
        while (bits) {
            const unsigned bit = static_cast<unsigned>(std::countr_zero(bits));
            pages.push_back(static_cast<u32>(w * 64 + bit));
            bits &= bits - 1;
        }
        dirty_[w] = 0;
    }
}

}  // namespace mv::p6
