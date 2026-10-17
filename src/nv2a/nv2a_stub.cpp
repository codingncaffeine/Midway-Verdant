#include "nv2a/nv2a_stub.h"

#include "debug/trace.h"

namespace mv {

Nv2aStub::Nv2aStub(Bus& bus, Scheduler& sched, std::function<void(bool)> irq)
    : PciDevice("nv2a", 0x10DE, 0x02A0, 0x030000, 0xA1), bus_(bus), sched_(sched), irq_(std::move(irq)) {
    define_bar(0, BarKind::Memory, kMmioSize);
    define_bar(1, BarKind::Memory, 128u * 1024 * 1024, true);
    define_bar(2, BarKind::Memory, 512u * 1024, true);
    set_cfg8(0x3C, 3);  // interrupt line
    set_cfg8(0x3D, 1);
    regs_[0x000000] = kPmcBoot0;
    regs_[kPtimerNumerator] = 1;
    regs_[kPtimerDenominator] = 1;
    vblank_.name = "nv2a.vblank";
    vblank_.fire = [this](Cycles) {
        ++vblanks_;
        regs_[kPcrtcIntr0] |= 1u;
        update_irq();
        schedule_vblank();
    };
    schedule_vblank();
}

void Nv2aStub::schedule_vblank() {
    // 59.94 Hz: the NTSC field rate the encoder puts out; one event per frame on the master clock.
    const u64 frame = muldiv64(clk::kCpuHzNum * 1001, 1, clk::kCpuHzDen * 60000);
    const Cycles now = sched_.now();
    sched_.schedule(vblank_, (now / frame + 1) * frame);
}

void Nv2aStub::update_irq() {
    u32 intr = 0;
    if (regs_[kPcrtcIntr0] & regs_[kPcrtcIntrEn0] & 1u) intr |= kPmcIntrPcrtc;
    if (regs_[kPtimerIntr0] & regs_[kPtimerIntrEn0] & 1u) intr |= kPmcIntrPtimer;
    regs_[kPmcIntr0] = intr;
    const bool level = (regs_[kPmcIntrEn0] & 1u) && intr != 0;
    if (level != irq_level_) {
        irq_level_ = level;
        if (irq_) irq_(level);
    }
}

u64 Nv2aStub::ptimer_ticks() const {
    // GPU cycles scaled by denominator / numerator, as the kernel programs them [verify against hardware].
    const u64 gpu_cycles = muldiv64(sched_.now(), clk::kGpuHzNum * clk::kCpuHzDen, clk::kCpuHzNum * clk::kGpuHzDen);
    const u32 num = reg(kPtimerNumerator) ? reg(kPtimerNumerator) : 1;
    const u32 den = reg(kPtimerDenominator) ? reg(kPtimerDenominator) : 1;
    return muldiv64(gpu_cycles, den, num);
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
    const u32 key = off & ~3u;
    u32 word;
    switch (key) {
        case kPtimerTime0: word = static_cast<u32>((ptimer_ticks() & 0x7FFFFFFu) << 5); break;
        case kPtimerTime1: word = static_cast<u32>((ptimer_ticks() >> 27) & 0x1FFFFFFFu); break;
        default: word = reg(key); break;
    }
    const unsigned shift = 8u * (off & 3u);
    const u32 v = size == 4 ? word : (size == 2 ? ((word >> shift) & 0xFFFFu) : ((word >> shift) & 0xFFu));
    ++reads_;
    if (key != kPtimerTime0 && key != kPtimerTime1) trace::log(trace::Cat::Gpu, "read  +{:06x} size {} -> {:08x}", off, size, v);
    return v;
}

void Nv2aStub::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 off = addr - mapped_at_;
    const u32 key = off & ~3u;
    u32 word;
    if (size == 4) {
        word = value;
    } else {
        const unsigned shift = 8u * (off & 3u);
        const u32 mask = (size == 2 ? 0xFFFFu : 0xFFu) << shift;
        word = (reg(key) & ~mask) | ((value << shift) & mask);
    }
    ++writes_;
    trace::log(trace::Cat::Gpu, "write +{:06x} size {} <- {:08x}", off, size, value);
    switch (key) {
        case kPfbWbc:
            // Bit 16 starts a frame-buffer cache operation and clears when it is done; done at once here.
            regs_[key] = word & ~(1u << 16);
            return;
        case kPcrtcIntr0:
        case kPtimerIntr0:
            regs_[key] &= ~word;  // write one to clear
            update_irq();
            return;
        case kPcrtcIntrEn0:
        case kPtimerIntrEn0:
        case kPmcIntrEn0:
            regs_[key] = word;
            update_irq();
            return;
        case kPmcIntr0:
            return;  // read-only aggregate
        default:
            regs_[key] = word;
            return;
    }
}

RegisterStoreDevice::RegisterStoreDevice(Bus& bus, std::string_view name, u16 device, u32 class_code, u8 revision, int bar,
                                         u32 bar_size, trace::Cat category)
    : PciDevice(name, 0x10DE, device, class_code, revision), bus_(bus), bar_(bar), bar_size_(bar_size), category_(category) {
    define_bar(bar, BarKind::Memory, bar_size);
}

void RegisterStoreDevice::on_bar_changed(int index, u32 old_base, u32 new_base) {
    (void)old_base;
    if (index != bar_) return;
    if (mapped_at_) bus_.unmap_mmio(mapped_at_, bar_size_);
    mapped_at_ = new_base;
    if (new_base) bus_.map_mmio(new_base, bar_size_, this);
    trace::log(category_, "{} registers at {:08x}", PciDevice::name(), new_base);
}

u32 RegisterStoreDevice::reg(u32 offset) const {
    auto f = fixed_.find(offset & ~3u);
    if (f != fixed_.end()) return f->second;
    auto it = regs_.find(offset & ~3u);
    return it == regs_.end() ? 0 : it->second;
}

u32 RegisterStoreDevice::mmio_read(PhysAddr addr, unsigned size) {
    const u32 off = addr - mapped_at_;
    const u32 word = reg(off);
    const unsigned shift = 8u * (off & 3u);
    const u32 v = size == 4 ? word : (size == 2 ? ((word >> shift) & 0xFFFFu) : ((word >> shift) & 0xFFu));
    ++reads_;
    trace::log(category_, "{} read  +{:05x} size {} -> {:08x}", PciDevice::name(), off, size, v);
    return v;
}

void RegisterStoreDevice::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 off = addr - mapped_at_;
    const u32 key = off & ~3u;
    if (size == 4) {
        regs_[key] = value;
    } else {
        const unsigned shift = 8u * (off & 3u);
        const u32 mask = (size == 2 ? 0xFFFFu : 0xFFu) << shift;
        regs_[key] = (reg(key) & ~mask) | ((value << shift) & mask);
    }
    if (auto sc = self_clearing_.find(key); sc != self_clearing_.end()) regs_[key] &= ~sc->second;
    ++writes_;
    trace::log(category_, "{} write +{:05x} size {} <- {:08x}", PciDevice::name(), off, size, value);
}

}  // namespace mv
