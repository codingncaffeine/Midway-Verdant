#pragma once
#include <functional>
#include <unordered_map>
#include <vector>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"
#include "debug/trace.h"
#include "nv2a/capture.h"
#include "nv2a/pushbuffer.h"
#include "pci/pci.h"

namespace mv {

class CpuBackend;

// PCI 1:0.0, the NV2A, as a register store with the few behaviours the boot chain polls for: the chip identity,
// the PFB command bit that clears itself, the PTIMER counting on the master clock, the vertical-blank interrupt
// through PCRTC and PMC. Every other register is remembered and read back, every access logged. The real engines
// replace this block by block in Phase 4.
class Nv2aStub : public PciDevice, public MmioHandler {
public:
    static constexpr u32 kMmioSize = 16u * 1024 * 1024;
    // Chip identity: a retail console's GPU is silicon revision A2. The BIOS X-codes read the low byte and give an
    // A1 chip a 191 MHz core clock (coefficient 0x11701) and every other revision 233 MHz (0x11C01), which the
    // homebrew runtime checks for before it runs; the kernel derives its timer ratio from the same coefficient.
    static constexpr u32 kPmcBoot0 = 0x02A000A2;
    // Register offsets (public NV20-family layout, see local-notes/hw/nv2a.md).
    static constexpr u32 kPmcIntr0 = 0x000100, kPmcIntrEn0 = 0x000140, kPmcEnable = 0x000200;
    static constexpr u32 kPtimerIntr0 = 0x009100, kPtimerIntrEn0 = 0x009140, kPtimerNumerator = 0x009200,
                         kPtimerDenominator = 0x009210, kPtimerTime0 = 0x009400, kPtimerTime1 = 0x009410, kPtimerAlarm0 = 0x009420;
    static constexpr u32 kPfbWbc = 0x100410;
    static constexpr u32 kPcrtcIntr0 = 0x600100, kPcrtcIntrEn0 = 0x600140, kPcrtcStart = 0x600800;
    // PRAMDAC +0x6A0: the kernel's display-mode set toggles bit 0 and waits until bit 4 reads the same — the
    // hardware mirrors the request bit at the next frame boundary (measured from the kernel's behaviour).
    static constexpr u32 kPramdacModeLatch = 0x6806A0;
    // PGRAPH: interrupt status (NOTIFY bit 0, MISSING_HW 4, CONTEXT_SWITCH 12, ERROR 20), the error source, the
    // trapped method and its data, and the FIFO access bit software clears to hold the engine and sets to resume.
    static constexpr u32 kPgraphNsource = 0x400108, kPgraphStatus = 0x400700, kPgraphTrappedAddr = 0x400704, kPgraphTrappedData = 0x400708,
                         kPgraphFifo = 0x400720;
    static constexpr u32 kPgraphIntrError = 1u << 20, kNsourceDataError = 1u << 1;
    static constexpr u32 kMethodNoOperation = 0x0100;
    // Two 3D-class methods whose values software reads back from the engine's register file: the kernel passes a
    // software method's arguments through them (a register offset or a flip address, and a value) [verify the
    // register names against the public listing].
    static constexpr u32 kMethodZStencilClearValue = 0x1D8C, kMethodColorClearValue = 0x1D90;
    static constexpr u32 kPgraphShadowZStencilClear = 0x401A88, kPgraphShadowColorClear = 0x40186C;
    static constexpr u32 kPfifoIntr0 = 0x002100, kPfifoIntrEn0 = 0x002140, kPgraphIntr = 0x400100, kPgraphIntrEn = 0x400140,
                         kPvideoIntr = 0x008100, kPvideoIntrEn = 0x008140;
    static constexpr u32 kPfifoRamht = 0x002210, kPfifoRamfc = 0x002214;
    static constexpr u32 kPfifoRunoutStatus = 0x002400, kCache1Status = 0x003214, kCache1DmaPush = 0x003220, kCache1DmaInstance = 0x00322C, kCache1DmaPut = 0x003240,
                         kCache1DmaGet = 0x003244;
    static constexpr u32 kPramin = 0x700000, kUser = 0x800000;
    static constexpr u32 kPrmcioInputStatus1 = 0x6013DA;  // the VGA input status register: retrace bits
    static constexpr u32 kPrmcioCrtcIndex = 0x6013D4, kPrmcioCrtcData = 0x6013D5;  // the CRTC register file
    static constexpr u32 kPmcIntrPfifo = 1u << 8, kPmcIntrPgraph = 1u << 12, kPmcIntrPtimer = 1u << 20, kPmcIntrPcrtc = 1u << 24;

    Nv2aStub(Bus& bus, Scheduler& sched, std::function<void(bool)> irq);
    std::string_view name() const override { return "nv2a"; }

    u32 mmio_read(PhysAddr addr, unsigned size) override;
    void mmio_write(PhysAddr addr, unsigned size, u32 value) override;

    u32 reg(u32 offset) const;
    u64 reads() const { return reads_; }
    u64 writes() const { return writes_; }
    u64 vblanks() const { return vblanks_; }
    // The reset line: registers at power-on, no objects bound, the engine free; the frame timer keeps running.
    void reset();
    // The mode the CRTC was programmed with (VGA-style registers through PRMCIO index/data, with NVIDIA's
    // extended bits): the displayed width and height in pixels and the line pitch in bytes; zero until set.
    u32 scanout_width() const { return (static_cast<u32>(crtc_[0x01]) + 1u) * 8u; }
    u32 scanout_height() const {
        return (static_cast<u32>(crtc_[0x12]) | ((crtc_[0x07] & 0x02u) << 7) | ((crtc_[0x07] & 0x40u) << 3) | ((crtc_[0x25] & 0x02u) << 9)) + 1u;
    }
    u32 scanout_pitch() const {
        return 8u * (static_cast<u32>(crtc_[0x13]) | ((crtc_[0x19] & 0xE0u) << 3) | ((crtc_[0x25] & 0x20u) << 6));
    }
    bool scanout_programmed() const { return crtc_[0x13] != 0; }
    u32 framebuffer_start() const { return reg(kPcrtcStart); }
    const PushbufferWalker& pushbuffer() const { return walker_; }
    // Objects bound to the channel's eight subchannels (class numbers, 0 = none).
    std::array<u32, 8> bound_classes() const { return bound_; }
    u32 ramin32(u32 offset) const { return reg(kPramin + offset); }

    // RAMHT lookup: the context word for a handle on channel 0, or 0.
    u32 lookup_handle(u32 handle) const;
    PushbufferWalker::DmaObject dma_object(u32 instance) const;

    // The pushbuffer capture instrument: every method, the RAM pages written before each pusher run (from the
    // CPU backend's dirty log and the bus's), the objects behind the handles, the scanout register writes.
    void set_capture(capture::Writer* writer, CpuBackend* cpu) {
        capture_ = writer;
        cpu_ = cpu;
    }

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    u64 ptimer_ticks() const;
    Cycles frame_period() const;
    bool in_vertical_retrace() const;
    void update_irq();
    void schedule_vblank();
    void run_pusher();
    void on_method(const PushbufferMethod& m);
    void capture_dirty_pages();
    void capture_object(u32 handle);

    Bus& bus_;
    Scheduler& sched_;
    std::function<void(bool)> irq_;
    u32 mapped_at_{0};
    std::unordered_map<u32, u32> regs_;
    std::array<u8, 256> crtc_{};
    u8 crtc_index_{0};
    u64 reads_{0}, writes_{0}, vblanks_{0};
    bool irq_level_{false};
    TimerEvent vblank_;
    PushbufferWalker walker_;
    std::array<u32, 8> bound_{};
    std::array<u32, 8> bound_handles_{};
    u64 method_log_lines_{0};
    // The synchronisation state the CPU side observes: per subchannel, the notifier and semaphore DMA objects.
    struct SyncState {
        u32 notifier_handle{0};
        u32 semaphore_handle{0};
        u32 semaphore_offset{0};
    };
    std::array<SyncState, 8> sync_{};
    u64 semaphore_releases_{0}, notifies_{0}, software_methods_{0};
    bool pgraph_stalled_{false};
    capture::Writer* capture_{nullptr};
    CpuBackend* cpu_{nullptr};
    std::vector<u32> dirty_scratch_;
public:
    u64 semaphore_releases() const { return semaphore_releases_; }
    u64 notifies() const { return notifies_; }
    u64 software_methods() const { return software_methods_; }
    bool pgraph_stalled() const { return pgraph_stalled_; }
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
    // An interrupt status register (write one to clear) and its enable register; the line follows status & enable.
    void set_interrupt_pair(u32 status_offset, u32 enable_offset, std::function<void(bool)> irq) {
        intr_status_ = status_offset;
        intr_enable_ = enable_offset;
        irq_ = std::move(irq);
    }
    // Raise status bits from a device-side event.
    void raise(u32 bits);
    // Some status registers clear on read instead of on write.
    void set_status_read_clears(bool on) { status_read_clears_ = on; }
    bool irq_level() const { return irq_level_; }
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
    u32 intr_status_{0xFFFFFFFF}, intr_enable_{0xFFFFFFFF};
    std::function<void(bool)> irq_;
    bool irq_level_{false};
    u32 previous_status_{0};
    bool status_read_clears_{false};
    void update_irq();
};

}  // namespace mv
