#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <memory>
#include <optional>
#include <functional>
#include <unordered_map>
#include <vector>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"
#include "debug/trace.h"
#include "nv2a/overlay.h"
#include "nv2a/capture.h"
#include "nv2a/pgraph/twod.h"
#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pushbuffer.h"
#include "pci/pci.h"

namespace mv {

class CpuBackend;
namespace pgraph {
class VulkanRenderer;
}

// PCI 1:0.0, the NV2A, as a register store with the few behaviours the boot chain polls for: the chip identity,
// the PFB command bit that clears itself, the PTIMER counting on the master clock, the vertical-blank interrupt
// through PCRTC and PMC. Every other register is remembered and read back, every access logged. The real engines
// replace this block by block in Phase 4.
class Nv2aStub : public PciDevice, public MmioHandler, public pgraph::HandleResolver {
public:
    static constexpr u32 kMmioSize = 16u * 1024 * 1024;
    // The live GPU: a renderer attached here receives every Kelvin method through the state machine as the
    // pushbuffer is consumed, and ends a frame at each semaphore release (the kernel's frame marker), which
    // writes the render targets back into guest RAM for the scanout. Draws run on the thread that runs the
    // pusher, under the machine lock — correct first, a render thread later.
    void set_renderer(pgraph::VulkanRenderer* renderer);
    pgraph::VulkanRenderer* renderer() const { return renderer_; }
    u64 rendered_frames() const { return rendered_frames_; }
    // HandleResolver: a DMA object as the engine reads it from instance memory.
    pgraph::DmaObject resolve(u32 handle) const override;
    // Chip identity: a retail console's GPU is silicon revision A2. The BIOS X-codes read the low byte and give an
    // A1 chip a 191 MHz core clock (coefficient 0x11701) and every other revision 233 MHz (0x11C01), which the
    // homebrew runtime checks for before it runs; the kernel derives its timer ratio from the same coefficient.
    static constexpr u32 kPmcBoot0 = 0x02A000A2;
    // Register offsets (public NV20-family layout, see local-notes/hw/nv2a.md).
    static constexpr u32 kPmcIntr0 = 0x000100, kPmcIntrEn0 = 0x000140, kPmcEnable = 0x000200;
    static constexpr u32 kPtimerIntr0 = 0x009100, kPtimerIntrEn0 = 0x009140, kPtimerNumerator = 0x009200,
                         kPtimerDenominator = 0x009210, kPtimerTime0 = 0x009400, kPtimerTime1 = 0x009410, kPtimerAlarm0 = 0x009420;
    static constexpr u32 kPfbWbc = 0x100410;
    static constexpr u32 kPcrtcIntr0 = 0x600100, kPcrtcIntrEn0 = 0x600140, kPcrtcStart = 0x600800, kPcrtcRaster = 0x600808;
    static constexpr u32 kPfbCstatus = 0x10020C, kPramdacGeneralControl = 0x680600;
    // PGRAPH SURFACE holds the three flip indices (write bits 22:20, read 26:24, modulo 30:28); a write of bit 1 to
    // INCREMENT advances the read index (the display has moved on to the next buffer) and releases a FLIP_STALL.
    static constexpr u32 kPgraphSurface = 0x400710, kPgraphIncrement = 0x40071C;
    // PRAMDAC +0x6A0: the kernel's display-mode set toggles bit 0 and waits until bit 4 reads the same — the
    // hardware mirrors the request bit at the next frame boundary (measured from the kernel's behaviour).
    static constexpr u32 kPramdacModeLatch = 0x6806A0;
    // PGRAPH: interrupt status (NOTIFY bit 0, MISSING_HW 4, CONTEXT_SWITCH 12, ERROR 20), the error source, the
    // trapped method and its data, and the FIFO access bit software clears to hold the engine and sets to resume.
    static constexpr u32 kPgraphNsource = 0x400108, kPgraphStatus = 0x400700, kPgraphTrappedAddr = 0x400704, kPgraphTrappedData = 0x400708,
                         kPgraphFifo = 0x400720;
    static constexpr u32 kPgraphIntrError = 1u << 20, kNsourceDataError = 1u << 1;
    static constexpr u32 kMethodNoOperation = 0x0100;
    // The pattern class's SET_MONOCHROME_COLOR0 lands in PGRAPH PATT_COLOR0, which software reads back: Direct3D pushes
    // a value through it and polls the register to know the engine has reached that point.
    static constexpr u32 kClassPattern = 0x44, kMethodPatternColor0 = 0x0310, kPgraphPattColor0 = 0x400B10;
    // Two 3D-class methods whose values software reads back from the engine's register file: the kernel passes a
    // software method's arguments through them (a register offset or a flip address, and a value) [verify the
    // register names against the public listing].
    static constexpr u32 kMethodZStencilClearValue = 0x1D8C, kMethodColorClearValue = 0x1D90;
    static constexpr u32 kPgraphShadowZStencilClear = 0x401A88, kPgraphShadowColorClear = 0x40186C;
    static constexpr u32 kPfifoIntr0 = 0x002100, kPfifoIntrEn0 = 0x002140, kPgraphIntr = 0x400100, kPgraphIntrEn = 0x400140,
                         kPvideoIntr = 0x008100, kPvideoIntrEn = 0x008140,
                         kPvideoBuffer = 0x008700, kPvideoStop = 0x008704, kPvideoBase = 0x008900, kPvideoColorKey = 0x008B00;
    static constexpr u32 kPfifoRamht = 0x002210, kPfifoRamfc = 0x002214;
    static constexpr u32 kCache1DmaCtl = 0x003230, kCache1Ref = 0x003248, kPramdacPllTestCounter = 0x680514;
    static constexpr u32 kPllLocked = (1u << 27) | (1u << 29) | (1u << 30) | (1u << 31);  // VPLL2, NVPLL, MPLL, VPLL
    // The object a subchannel is bound to, as PGRAPH caches it (the instance's four words and the instance), and the
    // current method's subchannel's copy in CTX_SWITCH1-5, where software-method handlers read the class.
    static constexpr u32 kPgraphCtxSwitch1 = 0x40014C, kPgraphCtxCache1 = 0x400160;
    static constexpr u32 kPfifoRunoutStatus = 0x002400, kCache1Status = 0x003214, kCache1DmaPush = 0x003220, kCache1DmaInstance = 0x00322C, kCache1DmaPut = 0x003240,
                         kCache1DmaGet = 0x003244;
    static constexpr u32 kPramin = 0x700000, kUser = 0x800000;
    static constexpr u32 kPrmcioInputStatus1 = 0x6013DA;  // the VGA input status register: retrace bits
    static constexpr u32 kPrmcioCrtcIndex = 0x6013D4, kPrmcioCrtcData = 0x6013D5;  // the CRTC register file
    static constexpr u32 kPmcIntrPfifo = 1u << 8, kPmcIntrPgraph = 1u << 12, kPmcIntrPtimer = 1u << 20, kPmcIntrPcrtc = 1u << 24;

    Nv2aStub(Bus& bus, Scheduler& sched, std::function<void(bool)> irq);
    ~Nv2aStub() override;
    Nv2aStub(const Nv2aStub&) = delete;
    Nv2aStub& operator=(const Nv2aStub&) = delete;

    // The FIFO thread (PLAN D21): the pusher walks on a thread of its own, under the GPU's own lock, which is always
    // taken after the machine lock. A kick wakes it; it publishes GET as it goes; the registers that report its
    // progress or idleness wait until it is idle when read, every other register reads the live state. An interrupt
    // it raises is handed over through `request` (called on the FIFO thread with no lock held: set a flag, kick the
    // CPU), and the machine applies it with service() under its lock. Not started: the pusher runs inline on the
    // thread that kicked it, as before.
    void start_fifo_thread(std::function<void()> request);
    void stop_fifo_thread();
    bool fifo_threaded() const { return fifo_threaded_; }
    bool fifo_busy_now() const { return fifo_busy_.load(std::memory_order_relaxed); }  // instrument: the FIFO thread has work in hand
    void service();
    // Until the pusher has nothing left it can run (a halted CPU calls it before it skips guest time).
    void wait_fifo_idle();
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
    // The scanout depth: CR28 bits 1:0 (3 = 32 bits a pixel, 2 = 16; unprogrammed reads as 32) and, in 16-bit modes,
    // the layout the RAMDAC's general control picks (bit 12 set: R5G6B5, clear: X1R5G5B5).
    u32 scanout_bytes_per_pixel() const { return (crtc_[0x28] & 3u) == 2u ? 2u : 4u; }
    bool scanout_rgb565() const { return (reg(kPramdacGeneralControl) & (1u << 12)) != 0; }
    u32 framebuffer_start() const { return reg(kPcrtcStart); }
    // The video overlay as it stands, when a buffer is in use (read under the machine lock).
    std::optional<Overlay> overlay() const;
    // The RAMDAC's colour table, written through the VGA-style DAC ports (a write index, then red, green, blue
    // bytes; the index advances after each triple). In the console's direct-colour modes it is the gamma ramp every
    // scanout pixel goes through. Empty while it is the identity (the kernel's default), so callers skip the work.
    std::optional<std::array<u8, 768>> gamma_lut() const;
    static constexpr u32 kDacReadIndex = 0x6813C7, kDacWriteIndex = 0x6813C8, kDacData = 0x6813C9;
    const PushbufferWalker& pushbuffer() const { return walker_; }
    // Objects bound to the channel's eight subchannels (class numbers, 0 = none).
    std::array<u32, 8> bound_classes() const { return bound_; }
    u32 ramin32(u32 offset) const { return reg(kPramin + offset); }

    // RAMHT lookup: the context word for a handle on channel 0, or 0.
    u32 lookup_handle(u32 handle) const;
    PushbufferWalker::DmaObject dma_object(u32 instance) const;

    // The pushbuffer capture instrument: every method, the RAM pages written before each pusher run (from the
    // CPU backend's dirty log and the bus's), the objects behind the handles, the scanout register writes.
    // Called at every vblank once the scanout's frame buffer is in guest RAM, with the guest time (an instrument hook).
    void set_vblank_hook(std::function<void(Cycles)> hook) { vblank_hook_ = std::move(hook); }
    void set_capture(capture::Writer* writer, CpuBackend* cpu) {
        capture_ = writer;
        cpu_ = cpu;
    }
    // CPU writes the GPU must see: with tracking on, every pusher run first drains the CPU backend's and the bus's
    // dirty-page logs and stamps each written 4 KiB page with a new generation. A cache built from RAM (a decoded
    // texture, a palette, a surface's upload) remembers the generation it was built at and is stale when any page of
    // its range is newer. Several entries may share a page, so nothing is ever cleared by a reader.
    void set_write_tracking(CpuBackend* cpu);
    u32 write_generation() const override { return write_generation_; }
    u32 newest_write(u32 phys, u32 len) const override;
    void collect_writes();

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    u64 ptimer_ticks() const;
    Cycles frame_period() const;
    bool in_vertical_retrace() const;
    u32 raster_line() const;
    u32 flip_field(unsigned shift) const { return (reg(kPgraphSurface) >> shift) & 7u; }
    void set_flip_field(unsigned shift, u32 v) { regs_[kPgraphSurface] = (reg(kPgraphSurface) & ~(7u << shift)) | ((v & 7u) << shift); }
    void advance_flip_field(unsigned shift);
    bool flip_released() const { return flip_field(24) != flip_field(20); }
    void write_report(u32 parameter);
    void update_irq();
    void schedule_vblank();
    void schedule_alarm();
    void run_pusher();
    void run_pusher_inline();
    bool pusher_runnable() const;
    void fifo_main();
    // The GPU lock for an entry point (an empty lock without the FIFO thread).
    std::unique_lock<std::mutex> lock_gpu() const;
    void wait_idle(std::unique_lock<std::mutex>& lock);
    // Guest time as the GPU sees it: the scheduler's on the machine's threads, the last register access's on the FIFO thread.
    Cycles gpu_now() const;
    void on_method(const PushbufferMethod& m);

    void capture_object(u32 handle);

    Bus& bus_;
    Scheduler& sched_;
    std::function<void(bool)> irq_;
    u32 mapped_at_{0};
    // The 16 MiB register window as a flat array of words (indexed by byte offset), PRAMIN included: every method,
    // handle lookup and register access goes through it.
    struct RegisterFile {
        std::vector<u32> words = std::vector<u32>(kMmioSize / 4);
        u32& operator[](u32 offset) { return words[(offset & (kMmioSize - 1)) >> 2]; }
        u32 operator[](u32 offset) const { return words[(offset & (kMmioSize - 1)) >> 2]; }
        void clear() { std::fill(words.begin(), words.end(), 0u); }
    } regs_;
    std::array<u8, 256> crtc_{};
    u8 crtc_index_{0};
    std::array<u8, 768> dac_{};
    u8 dac_write_index_{0}, dac_read_index_{0}, dac_write_component_{0}, dac_read_component_{0};
    bool dac_identity_{true};
    u64 reads_{0}, writes_{0}, vblanks_{0};
    bool irq_level_{false};
    TimerEvent vblank_;
    TimerEvent alarm_;  // PTIMER ALARM_0: the next time TIME_0 reaches the alarm's bits 31:5
    u64 ptimer_offset_{0};  // what software's TIME_0/TIME_1 writes added to the counter, in ticks
    u32 ctx_switch_subchannel_{8};
    std::function<void(Cycles)> vblank_hook_;
    PushbufferWalker walker_;
    pgraph::Kelvin kelvin_;
    u64 methods_at_vblank_{0};  // Kelvin methods seen at the last vblank (an idle frame writes the surfaces back)
    pgraph::VulkanRenderer* renderer_{nullptr};
    std::unique_ptr<pgraph::TwoD> twod_;  // the 2D surfaces and image-blit classes
    u64 rendered_frames_{0};
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
    u64 pusher_wall_ns_{0};
    std::chrono::steady_clock::time_point last_run_end_{};
    u64 last_run_ns_{0}, overlap_ns_{0}, sync_points_{0};
    bool run_open_{false};
    std::unordered_map<u32, std::pair<u64, u32>> read_counts_;
    std::unordered_map<u32, std::pair<u64, u32>> busy_read_counts_;  // reads answered while the FIFO thread had work
    u64 collect_calls_{0}, collect_pages_{0}, collect_ns_{0};  // dirty-page harvests before pusher runs, their pages and host time
    // `why`: the register offset (bit 31 set for a write), or one of the codes below.
    static constexpr u32 kSyncVblank = 0xFFFFFFF0u, kSyncNextRun = 0xFFFFFFF1u;
    void sync_point(u32 why);
    std::unordered_map<u32, std::pair<u64, u64>> sync_reasons_;  // why -> (windows ended, pusher ns not overlapped)
    bool pgraph_stalled_{false};
    // FLIP_STALL: the puller waits until the display has taken the buffer just finished (the read index moves off
    // the write index).
    bool flip_stalled_{false};
    u32 report_handle_{0};
    u64 flip_stalls_{0}, reports_{0};
    u64 display_flips_{0};  // INCREMENT writes that moved the display on to the next buffer
    capture::Writer* capture_{nullptr};
    CpuBackend* cpu_{nullptr};
    std::vector<u32> dirty_scratch_;
    bool write_tracking_{false};
    u32 write_generation_{1};
    std::vector<u32> page_generation_;  // per 4 KiB RAM page: the generation of the last write seen
    // The FIFO thread's state, all under gpu_mutex_ but the atomics.
    mutable std::mutex gpu_mutex_;
    mutable std::atomic<u32> gpu_waiters_{0};  // entry points waiting for the lock: the FIFO thread lets them in between chunks
    std::condition_variable fifo_cv_, idle_cv_;
    std::thread fifo_thread_;
    std::thread::id fifo_thread_id_{};
    bool fifo_threaded_{false}, fifo_stop_{false};
    std::atomic<bool> fifo_busy_{false};
    bool stuck_{false};  // the last walk made no progress: nothing to run until the next kick
    u32 chunk_methods_{0};
    static constexpr u32 kChunkMethods = 128;  // methods per walk between letting a waiting register access in
    std::atomic<bool> irq_deferred_{false};
    std::atomic<Cycles> guest_now_{0};
    std::function<void()> service_request_;
    // The lead (PLAN D21): the CPU may run this far past a kick the FIFO thread has not finished, then waits for it,
    // so in guest time the GPU never takes longer than this over a kick (MV_FIFO_LEAD_US, microseconds of guest time).
    TimerEvent lead_;
    Cycles lead_cycles_{0};
    u64 lead_waits_{0}, lead_wait_ns_{0};
    u64 vblank_waits_{0}, vblank_wait_ns_{0};  // vblanks that waited for the FIFO thread to finish what was queued
public:
    u64 semaphore_releases() const { return semaphore_releases_; }
    u64 pusher_wall_ns() const { return pusher_wall_ns_; }  // instrument: host time the vCPU spent inside the pusher (methods + renderer)
    // Instrument for a FIFO thread: of the pusher's time, how much the guest could have run alongside it — the host
    // time from the end of each run to the guest's next access that would need the GPU's result, capped at the run.
    u64 pusher_overlap_ns() const { return overlap_ns_; }
    u64 pusher_sync_points() const { return sync_points_; }
    // Instrument: register reads by offset, with the last value read, busiest first (a title spinning on a register).
    std::string busiest_reads(std::size_t top, bool while_busy = false) const;
    std::string harvest_report() const { return std::format("{} dirty-page harvests, {} pages, {:.3f} s", collect_calls_, collect_pages_, static_cast<double>(collect_ns_) / 1e9); }
    // Instrument: what ended the overlap windows, by the pusher time a FIFO thread could not have hidden, most first.
    std::string sync_reasons(std::size_t top) const;
    u64 notifies() const { return notifies_; }
    u64 software_methods() const { return software_methods_; }
    bool pgraph_stalled() const { return pgraph_stalled_; }
    bool flip_stalled() const { return flip_stalled_; }
    u64 flip_stalls() const { return flip_stalls_; }
    u64 display_flips() const { return display_flips_; }
    u64 fifo_lead_waits() const { return lead_waits_; }
    u64 fifo_lead_wait_ns() const { return lead_wait_ns_; }
    u64 fifo_vblank_waits() const { return vblank_waits_; }
    u64 fifo_vblank_wait_ns() const { return vblank_wait_ns_; }
    u64 reports() const { return reports_; }
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
