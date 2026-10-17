#pragma once
#include <functional>
#include <unordered_map>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"
#include "debug/trace.h"
#include "pci/pci.h"

namespace mv {

// PCI 1:0.0, the NV2A, as a register store with the few behaviours the boot chain polls for: the chip identity,
// the PFB command bit that clears itself, the PTIMER counting on the master clock, the vertical-blank interrupt
// through PCRTC and PMC. Every other register is remembered and read back, every access logged. The real engines
// replace this block by block in Phase 4.
class Nv2aStub : public PciDevice, public MmioHandler {
public:
    static constexpr u32 kMmioSize = 16u * 1024 * 1024;
    static constexpr u32 kPmcBoot0 = 0x02A000A1;  // chip identity read by firmware [verify]
    // Register offsets (public NV20-family layout, see local-notes/hw/nv2a.md).
    static constexpr u32 kPmcIntr0 = 0x000100, kPmcIntrEn0 = 0x000140, kPmcEnable = 0x000200;
    static constexpr u32 kPtimerIntr0 = 0x009100, kPtimerIntrEn0 = 0x009140, kPtimerNumerator = 0x009200,
                         kPtimerDenominator = 0x009210, kPtimerTime0 = 0x009400, kPtimerTime1 = 0x009410, kPtimerAlarm0 = 0x009420;
    static constexpr u32 kPfbWbc = 0x100410;
    static constexpr u32 kPcrtcIntr0 = 0x600100, kPcrtcIntrEn0 = 0x600140, kPcrtcStart = 0x600800;
    static constexpr u32 kPmcIntrPfifo = 1u << 8, kPmcIntrPgraph = 1u << 12, kPmcIntrPtimer = 1u << 20, kPmcIntrPcrtc = 1u << 24;

    Nv2aStub(Bus& bus, Scheduler& sched, std::function<void(bool)> irq);
    std::string_view name() const override { return "nv2a"; }

    u32 mmio_read(PhysAddr addr, unsigned size) override;
    void mmio_write(PhysAddr addr, unsigned size, u32 value) override;

    u32 reg(u32 offset) const;
    u64 reads() const { return reads_; }
    u64 writes() const { return writes_; }
    u64 vblanks() const { return vblanks_; }
    u32 framebuffer_start() const { return reg(kPcrtcStart); }

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    u64 ptimer_ticks() const;
    void update_irq();
    void schedule_vblank();

    Bus& bus_;
    Scheduler& sched_;
    std::function<void(bool)> irq_;
    u32 mapped_at_{0};
    std::unordered_map<u32, u32> regs_;
    u64 reads_{0}, writes_{0}, vblanks_{0};
    bool irq_level_{false};
    TimerEvent vblank_;
};

// A PCI function that exists for the bus scan and stores its configuration, with its regions unmapped: the
// bus logs any access to them. Used for every MCPX function whose model is a later phase.
class PciPlaceholder : public PciDevice {
public:
    PciPlaceholder(std::string_view name, u16 device, u32 class_code, u8 revision, u8 header_type = 0)
        : PciDevice(name, 0x10DE, device, class_code, revision, header_type) {}
};

// A memory-mapped register block that remembers what was written and answers reads from a table of known
// values: the shape of the APU and the AC97 controller until their real models arrive (Phases 5 and 6).
class RegisterStoreDevice : public PciDevice, public MmioHandler {
public:
    RegisterStoreDevice(Bus& bus, std::string_view name, u16 device, u32 class_code, u8 revision, int bar, u32 bar_size,
                        trace::Cat category);
    std::string_view name() const override { return PciDevice::name(); }
    u32 mmio_read(PhysAddr addr, unsigned size) override;
    void mmio_write(PhysAddr addr, unsigned size, u32 value) override;
    // Reads of `offset` always return `value` (status bits the firmware waits for).
    void set_fixed(u32 offset, u32 value) { fixed_[offset] = value; }
    // Bits of `offset` (a byte offset; the mask is in that byte's position) that clear as soon as they are
    // written: command bits hardware finishes at once at this level of modelling.
    void set_self_clearing(u32 offset, u32 mask) { self_clearing_[offset & ~3u] |= mask << (8u * (offset & 3u)); }
    u32 reg(u32 offset) const;
    u64 reads() const { return reads_; }
    u64 writes() const { return writes_; }

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    Bus& bus_;
    int bar_;
    u32 bar_size_;
    trace::Cat category_;
    u32 mapped_at_{0};
    std::unordered_map<u32, u32> regs_;
    std::unordered_map<u32, u32> fixed_;
    std::unordered_map<u32, u32> self_clearing_;
    u64 reads_{0}, writes_{0};
};

}  // namespace mv
