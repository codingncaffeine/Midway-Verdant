#include "nv2a/nv2a_stub.h"

#include "debug/trace.h"

namespace mv {

Nv2aStub::Nv2aStub(Bus& bus) : PciDevice("nv2a", 0x10DE, 0x02A0, 0x030000, 0xA1), bus_(bus) {
    define_bar(0, BarKind::Memory, kMmioSize);
    define_bar(1, BarKind::Memory, 128u * 1024 * 1024, true);
    define_bar(2, BarKind::Memory, 512u * 1024, true);
    set_cfg8(0x3C, 3);  // interrupt line
    set_cfg8(0x3D, 1);
    regs_[0x000000] = kPmcBoot0;
}

void Nv2aStub::on_bar_changed(int index, u32 old_base, u32 new_base) {
    (void)old_base;
    if (index == 0) {
        if (mapped_at_) bus_.unmap_mmio(mapped_at_, kMmioSize);
        mapped_at_ = new_base;
        if (new_base) bus_.map_mmio(new_base, kMmioSize, this);
        trace::log(trace::Cat::Gpu, "registers at {:08x}", new_base);
    } else if (index == 1 && new_base != Bus::kRamAliasBase) {
        trace::log(trace::Cat::Gpu, "frame buffer window moved to {:08x} (the bus serves it only at f0000000)", new_base);
    }
}

u32 Nv2aStub::reg(u32 offset) const {
    auto it = regs_.find(offset & ~3u);
    return it == regs_.end() ? 0 : it->second;
}

u32 Nv2aStub::mmio_read(PhysAddr addr, unsigned size) {
    const u32 off = addr - mapped_at_;
    const u32 word = reg(off);
    const unsigned shift = 8u * (off & 3u);
    const u32 v = size == 4 ? word : (size == 2 ? ((word >> shift) & 0xFFFFu) : ((word >> shift) & 0xFFu));
    ++reads_;
    trace::log(trace::Cat::Gpu, "read  +{:06x} size {} -> {:08x}", off, size, v);
    return v;
}

void Nv2aStub::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 off = addr - mapped_at_;
    const u32 key = off & ~3u;
    if (size == 4) {
        regs_[key] = value;
    } else {
        const unsigned shift = 8u * (off & 3u);
        const u32 mask = (size == 2 ? 0xFFFFu : 0xFFu) << shift;
        regs_[key] = (reg(key) & ~mask) | ((value << shift) & mask);
    }
    ++writes_;
    trace::log(trace::Cat::Gpu, "write +{:06x} size {} <- {:08x}", off, size, value);
}

}  // namespace mv
