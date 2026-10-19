#include "nv2a/nv2a_stub.h"

#include "nv2a/vulkan/vk_renderer.h"

#include <chrono>
#include <algorithm>

#include "cpu/backend.h"
#include "debug/trace.h"

namespace mv {

Nv2aStub::Nv2aStub(Bus& bus, Scheduler& sched, std::function<void(bool)> irq)
    : PciDevice("nv2a", 0x10DE, 0x02A0, 0x030000, 0xA1), bus_(bus), sched_(sched), irq_(std::move(irq)), walker_(bus) {
    walker_.set_sink([this](const PushbufferMethod& m) { on_method(m); });
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
        // The scanout reads the frame buffer now: a rendered one comes down from the device first.
        if (renderer_) renderer_->write_back_at(reg(kPcrtcStart));
        // A whole frame without a method: the GPU is idle, so what it drew is in RAM on the hardware; put it there.
        if (renderer_ && kelvin_.methods_seen() == methods_at_vblank_) renderer_->write_back_all();
        methods_at_vblank_ = kelvin_.methods_seen();
        if (vblank_hook_) vblank_hook_(sched_.now());
        regs_[kPcrtcIntr0] |= 1u;
        // The mode latch acknowledges software's toggle at the frame boundary: bit 4 takes bit 0.
        const u32 latch = reg(kPramdacModeLatch);
        regs_[kPramdacModeLatch] = (latch & ~0x10u) | ((latch & 1u) << 4);
        update_irq();
        schedule_vblank();
    };
    alarm_.name = "nv2a.ptimer-alarm";
    alarm_.fire = [this](Cycles) {
        regs_[kPtimerIntr0] |= 1u;
        update_irq();
        schedule_alarm();
    };
    schedule_vblank();
    for (u32 i = 0; i < 768; ++i) dac_[i] = static_cast<u8>(i / 3);  // the colour table powers up as the identity
    kelvin_.set_vertex_memory(this, {bus_.ram(), static_cast<u32>(bus_.ram_size())});
    twod_ = std::make_unique<pgraph::TwoD>(bus_.ram(), bus_.ram_size(), *this);
    twod_->set_hooks([this](u32 phys, u32 len) { if (renderer_) renderer_->write_back_range(phys, len); },
                     [this](u32 phys, u32 len) { if (renderer_) renderer_->invalidate(phys, len); });
}

void Nv2aStub::reset() {
    regs_.clear();
    regs_[0x000000] = kPmcBoot0;
    regs_[kPtimerNumerator] = 1;
    regs_[kPtimerDenominator] = 1;
    bound_ = {};
    bound_handles_ = {};
    sync_ = {};
    crtc_ = {};
    crtc_index_ = 0;
    for (u32 i = 0; i < 768; ++i) dac_[i] = static_cast<u8>(i / 3);
    dac_write_index_ = dac_read_index_ = dac_write_component_ = dac_read_component_ = 0;
    dac_identity_ = true;
    pgraph_stalled_ = false;
    flip_stalled_ = false;
    report_handle_ = 0;
    ptimer_offset_ = 0;
    ctx_switch_subchannel_ = 8;
    sched_.cancel(alarm_);
    kelvin_ = pgraph::Kelvin();
    kelvin_.set_sink(renderer_);
    kelvin_.set_vertex_memory(this, {bus_.ram(), static_cast<u32>(bus_.ram_size())});
    update_irq();
}

Cycles Nv2aStub::frame_period() const {
    // 59.94 Hz: the NTSC field rate the encoder puts out.
    return muldiv64(clk::kCpuHzNum * 1001, 1, clk::kCpuHzDen * 60000);
}

bool Nv2aStub::in_vertical_retrace() const {
    // The vertical blanking interval: about 20 of 262.5 lines a field, at the start of each frame on our clock.
    const Cycles frame = frame_period();
    const Cycles phase = sched_.now() % frame;
    return phase < frame * 20 / 262;
}

// PCRTC_RASTER: the line the display is reading, counting from the top of the picture through the blanking interval
// to the frame's total and wrapping. Our frame starts with the vertical blank, so it starts at the first line below the
// picture. The total comes from the CRTC's vertical total (CR06 with its overflow bits in CR07 and CR25, plus two).
u32 Nv2aStub::raster_line() const {
    const u32 visible = scanout_programmed() ? scanout_height() : 480u;
    u32 total = (static_cast<u32>(crtc_[0x06]) | ((crtc_[0x07] & 0x01u) << 8) | ((crtc_[0x07] & 0x20u) << 4) | ((crtc_[0x25] & 0x01u) << 10)) + 2u;
    if (crtc_[0x06] == 0 || total <= visible) total = visible + visible * 45u / 480u;
    const Cycles frame = frame_period();
    const u64 into = muldiv64(sched_.now() % frame, total, frame);
    return static_cast<u32>((visible + into) % total);
}

void Nv2aStub::advance_flip_field(unsigned shift) {
    const u32 modulo = flip_field(28);
    set_flip_field(shift, modulo ? (flip_field(shift) + 1u) % modulo : flip_field(shift) + 1u);
}

// GET_REPORT: a 16-byte report at the report DMA object + the offset in bits 23:0 — the PTIMER time, the count, and a
// status word of zero (done). The count is the z-pass pixel count since CLEAR_REPORT_VALUE; the renderer has no
// occlusion queries yet, so a report says every pixel of the screen passed (a visibility test finds its object
// visible) instead of leaving the report unwritten, which a title polling for it waits on forever.
void Nv2aStub::write_report(u32 parameter) {
    const auto dma = dma_object(lookup_handle(report_handle_) & 0xFFFFu);
    const u32 offset = parameter & 0x00FFFFFFu;
    if (!dma.valid || offset + 16 > dma.limit + 1) {
        trace::log(trace::Cat::Gpu, "report with no report object (handle {:08x}, offset {:x})", report_handle_, offset);
        return;
    }
    u8 report[16]{};
    const u64 t = ptimer_ticks();
    for (int i = 0; i < 8; ++i) report[i] = static_cast<u8>(t >> (8 * i));
    const u32 count = (scanout_programmed() ? scanout_width() * scanout_height() : 640u * 480u);
    for (int i = 0; i < 4; ++i) report[8 + i] = static_cast<u8>(count >> (8 * i));
    bus_.write_ram(dma.base + offset, report, 16);
    ++reports_;
}

void Nv2aStub::schedule_vblank() {
    const u64 frame = frame_period();
    const Cycles now = sched_.now();
    sched_.schedule(vblank_, (now / frame + 1) * frame);
}

void Nv2aStub::update_irq() {
    u32 intr = 0;
    if (regs_[kPcrtcIntr0] & regs_[kPcrtcIntrEn0] & 1u) intr |= kPmcIntrPcrtc;
    if (regs_[kPtimerIntr0] & regs_[kPtimerIntrEn0] & 1u) intr |= kPmcIntrPtimer;
    if (regs_[kPfifoIntr0] & regs_[kPfifoIntrEn0]) intr |= kPmcIntrPfifo;
    if (regs_[kPgraphIntr] & regs_[kPgraphIntrEn]) intr |= kPmcIntrPgraph;
    if (regs_[kPvideoIntr] & regs_[kPvideoIntrEn]) intr |= 1u << 16;
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
    return muldiv64(gpu_cycles, den, num) + ptimer_offset_;
}

// PTIMER ALARM_0: an interrupt when TIME_0 (the tick count in bits 31:5) next reaches the alarm's bits 31:5 — a
// comparator, so an alarm already passed when it is armed waits for the counter to wrap [verify on hardware].
void Nv2aStub::schedule_alarm() {
    sched_.cancel(alarm_);
    if (!(reg(kPtimerIntrEn0) & 1u)) return;
    const u32 now_low = static_cast<u32>(ptimer_ticks() << 5) & ~0x1Fu;
    u64 distance = static_cast<u32>((reg(kPtimerAlarm0) & ~0x1Fu) - now_low);
    if (distance == 0) distance = 1ull << 32;
    const u32 num = reg(kPtimerNumerator) ? reg(kPtimerNumerator) : 1;
    const u32 den = reg(kPtimerDenominator) ? reg(kPtimerDenominator) : 1;
    const u64 gpu_cycles = muldiv64_ceil(distance >> 5, num, den);
    const Cycles cycles = muldiv64_ceil(gpu_cycles, clk::kCpuHzNum * clk::kGpuHzDen, clk::kGpuHzNum * clk::kCpuHzDen);
    sched_.schedule(alarm_, sched_.now() + std::max<Cycles>(cycles, 1));
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

u32 Nv2aStub::reg(u32 offset) const { return regs_[offset]; }

u32 Nv2aStub::lookup_handle(u32 handle) const {
    // RAMHT: base and size from PFIFO_RAMHT; the hash folds the handle in slices of the table's index width
    // and mixes in the channel (0 here). Entries are {handle, context} pairs.
    const u32 ramht = reg(kPfifoRamht);
    const u32 base = (ramht & 0x1F0u) << 8;
    const u32 bits = ((ramht >> 16) & 3u) + 9;
    u32 hash = 0;
    u32 h = handle;
    while (h) {
        hash ^= h & ((1u << bits) - 1);
        h >>= bits;
    }
    const u32 entry = base + hash * 8;
    const u32 found = ramin32(entry);
    if (found != handle) {
        // Collisions are rare on the console; a linear scan of the table settles them.
        const u32 size = 1u << (bits + 3);
        for (u32 off = 0; off < size; off += 8) {
            if (ramin32(base + off) == handle) return ramin32(base + off + 4);
        }
        return 0;
    }
    return ramin32(entry + 4);
}

PushbufferWalker::DmaObject Nv2aStub::dma_object(u32 instance) const {
    PushbufferWalker::DmaObject d;
    const u32 at = instance * 16;
    const u32 dw0 = ramin32(at), dw1 = ramin32(at + 4), dw2 = ramin32(at + 8);
    const u32 cls = dw0 & 0xFFu;
    if (cls != 0x02 && cls != 0x03 && cls != 0x3D) return d;  // DMA object classes
    d.base = (dw2 & 0xFFFFF000u) | ((dw0 >> 20) & 0xFFFu);
    d.limit = dw1;
    d.valid = true;
    return d;
}

pgraph::DmaObject Nv2aStub::resolve(u32 handle) const {
    const u32 instance = lookup_handle(handle) & 0xFFFFu;
    const PushbufferWalker::DmaObject d = dma_object(instance);
    pgraph::DmaObject r;
    r.base = d.base;
    r.limit = d.limit;
    r.cls = ramin32(instance * 16) & 0xFFu;
    r.valid = d.valid;
    return r;
}

void Nv2aStub::set_renderer(pgraph::VulkanRenderer* renderer) {
    renderer_ = renderer;
    kelvin_.set_sink(renderer);
}

void Nv2aStub::on_method(const PushbufferMethod& m) {
    if (capture_) {
        capture_->method(m.subchannel, m.method == 0 ? 0 : bound_[m.subchannel], m.method, m.value);
        // The objects a handle names are recorded when the stream binds them: SET_OBJECT, and the context DMA
        // methods every class keeps at 0x180..0x1A8 (the 2D classes name surface objects there too).
        if (m.method == 0 || (m.method >= 0x180 && m.method <= 0x1A8)) capture_object(m.value);
    }
    if (m.method == 0) {
        bound_handles_[m.subchannel] = m.value;
        const u32 ctx = lookup_handle(m.value);
        const u32 instance = ctx & 0xFFFFu;
        // The class is the context's low byte; the bits above it are flags (endianness, the chid, a patch config).
        bound_[m.subchannel] = ramin32(instance * 16) & 0xFFu;
        for (u32 w = 0; w < 4; ++w) regs_[kPgraphCtxCache1 + 0x20 * w + 4 * m.subchannel] = ramin32(instance * 16 + 4 * w);
        regs_[kPgraphCtxCache1 + 0x80 + 4 * m.subchannel] = instance;
        ctx_switch_subchannel_ = 8;  // the next method copies the new binding into CTX_SWITCH
        trace::log(trace::Cat::Gpu, "fifo: subchannel {} <- object {:08x} (class {:04x}, instance {:04x})", m.subchannel, m.value,
                   bound_[m.subchannel], instance);
        return;
    }
    if (m.subchannel != ctx_switch_subchannel_) {
        ctx_switch_subchannel_ = m.subchannel;
        for (u32 w = 0; w < 5; ++w) regs_[kPgraphCtxSwitch1 + 4 * w] = regs_[kPgraphCtxCache1 + 0x20 * w + 4 * m.subchannel];
    }
    // SET_REFERENCE: the puller's reference counter, which software reads back through the USER window to see
    // how far the GPU has come.
    if (m.method == 0x0050) {
        regs_[kCache1Ref] = m.value;
        return;
    }
    if (method_log_lines_ < 200000) {
        ++method_log_lines_;
        trace::log(trace::Cat::Gpu, "fifo: [{:04x}] {:04x} <- {:08x}", bound_[m.subchannel], m.method, m.value);
    }
    // A no-operation carrying data is a software method: a legal method with data the engine cannot take, so it
    // raises ERROR with the DATA_ERROR source, holds its FIFO, and the host reads the trapped method and data
    // (the kernel queues its display flips this way; an illegal-method source would only be acknowledged).
    if (bound_[m.subchannel] == 0x97) {
        if (m.method == kMethodZStencilClearValue) regs_[kPgraphShadowZStencilClear] = m.value;
        else if (m.method == kMethodColorClearValue) regs_[kPgraphShadowColorClear] = m.value;
    }
    if (m.method == kMethodNoOperation && m.value != 0) {
        regs_[kPgraphTrappedAddr] = (m.subchannel << 16) | m.method;  // method in bits 12:0, subchannel in 18:16
        regs_[kPgraphTrappedData] = m.value;
        regs_[kPgraphNsource] |= kNsourceDataError;
        regs_[kPgraphIntr] |= kPgraphIntrError;
        regs_[kPgraphFifo] = 0;
        pgraph_stalled_ = true;
        ++software_methods_;
        walker_.request_stop();
        trace::log(trace::Cat::Gpu, "pgraph: software method {:04x} data {:08x} on subchannel {} (FIFO held)", m.method, m.value, m.subchannel);
        update_irq();
        return;
    }
    // The flip indices and the stall that keeps the GPU from drawing into the buffer still on the screen: titles set
    // a modulo (the number of buffers), advance the write index when a frame is finished and stall until the
    // vblank handler has advanced the read index through PGRAPH INCREMENT; and the visibility-test reports.
    if (bound_[m.subchannel] == 0x97) {
        switch (m.method) {
            case 0x0120: set_flip_field(24, m.value); break;  // SET_FLIP_READ
            case 0x0124: set_flip_field(20, m.value); break;  // SET_FLIP_WRITE
            case 0x0128: set_flip_field(28, m.value); break;  // SET_FLIP_MODULO
            case 0x012C: advance_flip_field(20); break;       // FLIP_INCREMENT_WRITE
            case 0x0130:                                      // FLIP_STALL
                if (!flip_released()) {
                    flip_stalled_ = true;
                    ++flip_stalls_;
                    walker_.request_stop();
                }
                break;
            case 0x01A8: report_handle_ = m.value; break;  // SET_CONTEXT_DMA_REPORT
            case 0x17D0: write_report(m.value); break;     // GET_REPORT
            default: break;
        }
    }
    // The live GPU: every Kelvin method goes through the state machine to the renderer; the semaphore release is
    // the frame marker, so the frame is rendered and written back to RAM before the CPU can see the semaphore.
    if (renderer_ && bound_[m.subchannel] == 0x97) {
        kelvin_.method(m.method, m.value);
        if (m.method == 0x1D70) {
            renderer_->flush_frame();
            ++rendered_frames_;
        }
    }
    // The 2D engine: the surfaces object and the image blit.
    if (bound_[m.subchannel] == pgraph::TwoD::kClassSurfaces || bound_[m.subchannel] == pgraph::TwoD::kClassBlit)
        twod_->method(bound_[m.subchannel], m.method, m.value);
    // The synchronisation methods every class shares, and Kelvin's semaphore: these are what the CPU waits on,
    // so they are executed here while every drawing method is only logged (the real engine arrives in Phase 4).
    SyncState& s = sync_[m.subchannel];
    switch (m.method) {
        case 0x0180:  // SET_CONTEXT_DMA_NOTIFIES
            s.notifier_handle = m.value;
            return;
        case 0x0104: {  // NOTIFY: write the notifier (timestamp, info, status = done) when the next method completes
            const auto dma = dma_object(lookup_handle(s.notifier_handle) & 0xFFFFu);
            if (!dma.valid) {
                trace::log(trace::Cat::Gpu, "notify with no notifier object (handle {:08x})", s.notifier_handle);
                return;
            }
            u8 notifier[16]{};
            const u64 t = ptimer_ticks();
            for (int i = 0; i < 8; ++i) notifier[i] = static_cast<u8>(t >> (8 * i));
            bus_.write_ram(dma.base, notifier, 16);
            ++notifies_;
            return;
        }
        case 0x01A4:  // SET_CONTEXT_DMA_SEMAPHORE
            s.semaphore_handle = m.value;
            return;
        case 0x1D6C:  // SET_SEMAPHORE_OFFSET
            s.semaphore_offset = m.value;
            return;
        case 0x1D70: {  // BACK_END_WRITE_SEMAPHORE_RELEASE
            const auto dma = dma_object(lookup_handle(s.semaphore_handle) & 0xFFFFu);
            if (!dma.valid || s.semaphore_offset + 4 > dma.limit + 1) {
                trace::log(trace::Cat::Gpu, "semaphore release with no object (handle {:08x})", s.semaphore_handle);
                return;
            }
            bus_.write_ram(dma.base + s.semaphore_offset, &m.value, 4);
            ++semaphore_releases_;
            trace::log(trace::Cat::Gpu, "semaphore release: {:08x} -> {:08x}", m.value, dma.base + s.semaphore_offset);
            return;
        }
        default:
            return;
    }
}

void Nv2aStub::capture_object(u32 handle) {
    const u32 instance = lookup_handle(handle) & 0xFFFFu;
    u32 dw[4];
    for (u32 i = 0; i < 4; ++i) dw[i] = ramin32(instance * 16 + i * 4);
    capture_->object(handle, dw);
}

void Nv2aStub::set_write_tracking(CpuBackend* cpu) {
    cpu_ = cpu;
    write_tracking_ = true;
    page_generation_.assign(bus_.ram_size() / capture::kPageSize, 0u);
}

u32 Nv2aStub::newest_write(u32 phys, u32 len) const {
    if (!write_tracking_ || len == 0) return 0;
    const std::size_t pages = page_generation_.size();
    const std::size_t first = phys / capture::kPageSize, last = (static_cast<std::size_t>(phys) + len - 1) / capture::kPageSize;
    u32 newest = 0;
    for (std::size_t p = first; p <= last && p < pages; ++p) newest = std::max(newest, page_generation_[p]);
    return newest;
}

void Nv2aStub::collect_writes() {
    if (!write_tracking_ && !capture_) return;
    // Everything written to RAM since the last run, by the guest (the backend's log) or by a device (the bus's).
    dirty_scratch_.clear();
    if (cpu_) cpu_->take_dirty_pages(dirty_scratch_);
    bus_.take_dirty_pages(dirty_scratch_);
    if (dirty_scratch_.empty()) return;
    std::sort(dirty_scratch_.begin(), dirty_scratch_.end());
    dirty_scratch_.erase(std::unique(dirty_scratch_.begin(), dirty_scratch_.end()), dirty_scratch_.end());
    if (write_tracking_) {
        ++write_generation_;
        for (u32 p : dirty_scratch_)
            if (p < page_generation_.size()) page_generation_[p] = write_generation_;
    }
    if (!capture_) return;
    // The capture records them coalesced into contiguous runs of pages.
    const u32 page_count = static_cast<u32>(bus_.ram_size() / capture::kPageSize);
    std::size_t i = 0;
    while (i < dirty_scratch_.size()) {
        const u32 first = dirty_scratch_[i];
        std::size_t j = i + 1;
        while (j < dirty_scratch_.size() && dirty_scratch_[j] == first + static_cast<u32>(j - i)) ++j;
        const u32 count = static_cast<u32>(j - i);
        if (first < page_count) capture_->pages(first, std::min(count, page_count - first), bus_.ram() + static_cast<std::size_t>(first) * capture::kPageSize);
        i = j;
    }
}

void Nv2aStub::run_pusher() {
    if (!(reg(kCache1DmaPush) & 1u)) return;
    if (pgraph_stalled_) return;  // held on a trapped method until software sets the FIFO access bit
    if (flip_stalled_) {
        if (!flip_released()) return;
        flip_stalled_ = false;
    }
    const u32 put = reg(kCache1DmaPut);
    u32 get = reg(kCache1DmaGet);
    if (get == put) return;
    const auto dma = dma_object(reg(kCache1DmaInstance) & 0xFFFFu);
    if (!dma.valid) {
        trace::log(trace::Cat::Gpu, "pusher: no DMA object bound (instance {:04x})", reg(kCache1DmaInstance));
        return;
    }
    collect_writes();
    if (capture_) capture_->run(sched_.now(), get, put);
    const auto t0 = std::chrono::steady_clock::now();
    get = walker_.run(dma, get, put);
    pusher_wall_ns_ += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    regs_[kCache1DmaGet] = get;
}

u32 Nv2aStub::mmio_read(PhysAddr addr, unsigned size) {
    const u32 off = addr - mapped_at_;
    const u32 key = off & ~3u;
    if (off == kPrmcioInputStatus1 && size == 1) {
        // Bit 3: vertical retrace in progress; bit 0: display inactive (blanking).
        const bool vr = in_vertical_retrace();
        ++reads_;
        return vr ? 0x09u : 0x00u;
    }
    if (size == 1 && off == kPrmcioCrtcIndex) {
        ++reads_;
        return crtc_index_;
    }
    if (size == 1 && off == kPrmcioCrtcData) {
        ++reads_;
        return crtc_[crtc_index_];
    }
    if (size == 1 && off == kDacData) {
        ++reads_;
        const u8 v = dac_[dac_read_index_ * 3u + dac_read_component_];
        if (++dac_read_component_ == 3) {
            dac_read_component_ = 0;
            ++dac_read_index_;
        }
        return v;
    }
    u32 word;
    switch (key) {
        case kPtimerTime0: word = static_cast<u32>((ptimer_ticks() & 0x7FFFFFFu) << 5); break;
        case kPtimerTime1: word = static_cast<u32>((ptimer_ticks() >> 27) & 0x1FFFFFFFu); break;
        case kPcrtcRaster: word = raster_line(); break;
        case kPfbCstatus: word = static_cast<u32>(bus_.ram_size()); break;  // the RAM size in bytes
        case kUser + 0x40: word = reg(kCache1DmaPut); break;
        case kUser + 0x44: word = reg(kCache1DmaGet); break;
        case kUser + 0x48: word = reg(kCache1Ref); break;
        case kCache1DmaCtl: word = reg(key) | (1u << 31); break;  // the pushbuffer translation is always valid
        case kPramdacPllTestCounter: word = reg(key) | kPllLocked; break;  // the PLLs lock at once
        // The cache is always drained here: LOW_MARK (empty) set, HIGH_MARK (full) clear.
        case kCache1Status: word = 0x10; break;
        case kPfifoRunoutStatus: word = 0x10; break;  // runout buffer empty
        // DMA pusher: access bit as written, state idle, buffer empty.
        case kCache1DmaPush: word = (reg(key) & 1u) | 0x100u; break;
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
    if (size == 1 && off == kPrmcioCrtcIndex) {
        crtc_index_ = static_cast<u8>(value);
        return;
    }
    if (size == 1 && off == kPrmcioCrtcData) {
        crtc_[crtc_index_] = static_cast<u8>(value);
        return;
    }
    if (size == 1 && (off == kDacWriteIndex || off == kDacReadIndex || off == kDacData)) {
        if (off == kDacWriteIndex) {
            dac_write_index_ = static_cast<u8>(value);
            dac_write_component_ = 0;
        } else if (off == kDacReadIndex) {
            dac_read_index_ = static_cast<u8>(value);
            dac_read_component_ = 0;
        } else {
            dac_[dac_write_index_ * 3u + dac_write_component_] = static_cast<u8>(value);
            if (++dac_write_component_ == 3) {
                dac_write_component_ = 0;
                ++dac_write_index_;  // wraps at 256
            }
            dac_identity_ = true;
            for (u32 i = 0; i < 768 && dac_identity_; ++i) dac_identity_ = dac_[i] == static_cast<u8>(i / 3);
        }
        return;
    }
    // The scanout blocks (PCRTC, PRAMDAC) go into the capture: a replay needs to know which surface is displayed.
    if (capture_ && ((key >= 0x600000 && key < 0x601000) || (key >= 0x680000 && key < 0x681000))) capture_->reg(key, word);
    switch (key) {
        case kPfbWbc:
            // Bit 16 starts a frame-buffer cache operation and clears when it is done; done at once here.
            regs_[key] = word & ~(1u << 16);
            return;
        case kPcrtcIntr0:
        case kPtimerIntr0:
        case kPfifoIntr0:
        case kPgraphIntr:
            regs_[key] &= ~word;  // interrupt status: write one to clear
            if (word & kPgraphIntrError) regs_[kPgraphNsource] = 0;
            update_irq();
            return;
        case kPvideoStop:
            // A stop request tears the overlay down at bit 0; the register itself reads back zero.
            if (word & 1u) regs_[kPvideoBuffer] = 0;
            return;
        case kPvideoIntr:
            regs_[key] &= ~word;
            update_irq();
            return;
        case kPcrtcStart:
            regs_[key] = word & 0x07FFFFFFu;  // a RAM address: the top bits are not decoded
            return;
        case kPgraphIncrement:
            if (word & 2u) {
                advance_flip_field(24);
                if (flip_stalled_ && flip_released()) run_pusher();
            }
            return;
        case kPgraphFifo:
            regs_[key] = word;
            if ((word & 1u) && pgraph_stalled_) {
                pgraph_stalled_ = false;
                trace::log(trace::Cat::Gpu, "pgraph: FIFO released, the walk resumes");
                run_pusher();
            }
            return;
        case kPcrtcIntrEn0:
        case kPtimerIntrEn0:
        case kPfifoIntrEn0:
        case kPgraphIntrEn:
        case kPvideoIntrEn:
        case kPmcIntrEn0:
            regs_[key] = word;
            update_irq();
            if (key == kPtimerIntrEn0) schedule_alarm();
            return;
        case kPmcIntr0:
            return;  // read-only aggregate
        case kUser + 0x44:  // DMA_GET through the USER window: software rewinds the pusher
            regs_[kCache1DmaGet] = word;
            return;
        case kPtimerAlarm0:
        case kPtimerNumerator:
        case kPtimerDenominator:
            regs_[key] = word;
            schedule_alarm();
            return;
        case kPtimerTime0:
        case kPtimerTime1: {
            // Writing the time moves the counter: TIME_0 holds tick bits 26:0 in its bits 31:5, TIME_1 bits 55:27.
            const u64 now = ptimer_ticks();
            const u64 target = key == kPtimerTime0 ? (now & ~0x7FFFFFFull) | (word >> 5)
                                                   : (now & 0x7FFFFFFull) | (static_cast<u64>(word & 0x1FFFFFFFu) << 27);
            ptimer_offset_ += target - now;
            schedule_alarm();
            return;
        }
        case kUser + 0x40:  // channel 0 DMA_PUT through the USER window
        case kCache1DmaPut:
            regs_[kCache1DmaPut] = word;
            run_pusher();
            return;
        case kCache1DmaPush:
            regs_[key] = word;
            run_pusher();
            return;
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
    if (status_read_clears_ && (off & ~3u) == intr_status_) {
        regs_[intr_status_] = 0;
        update_irq();
    }
    return v;
}

void RegisterStoreDevice::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 off = addr - mapped_at_;
    const u32 key = off & ~3u;
    if (key == intr_status_) previous_status_ = reg(key);
    if (size == 4) {
        regs_[key] = value;
    } else {
        const unsigned shift = 8u * (off & 3u);
        const u32 mask = (size == 2 ? 0xFFFFu : 0xFFu) << shift;
        regs_[key] = (reg(key) & ~mask) | ((value << shift) & mask);
    }
    if (auto sc = self_clearing_.find(key); sc != self_clearing_.end()) regs_[key] &= ~sc->second;
    if (key == intr_status_) {
        // Interrupt status is write-one-to-clear: the store above is undone and the written bits are cleared.
        const u32 written = (size == 4) ? value : ((value & ((size == 2) ? 0xFFFFu : 0xFFu)) << (8u * (off & 3u)));
        regs_[key] = previous_status_ & ~written;
        update_irq();
    }
    if (key == intr_enable_) update_irq();
    ++writes_;
    trace::log(category_, "{} write +{:05x} size {} <- {:08x}", PciDevice::name(), off, size, value);
}

void RegisterStoreDevice::raise(u32 bits) {
    if (intr_status_ == 0xFFFFFFFF) return;
    regs_[intr_status_] |= bits;
    previous_status_ = regs_[intr_status_];
    update_irq();
}

void RegisterStoreDevice::update_irq() {
    if (intr_status_ == 0xFFFFFFFF) return;
    previous_status_ = regs_[intr_status_];
    const bool level = (regs_[intr_status_] & reg(intr_enable_)) != 0;
    if (level != irq_level_) {
        irq_level_ = level;
        if (irq_) irq_(level);
    }
}

}  // namespace mv

namespace mv {

std::optional<Overlay> Nv2aStub::overlay() const {
    const u32 use = reg(kPvideoBuffer);
    if (!(use & 0x11u)) return std::nullopt;
    const u32 i = (use & 1u) ? 0u : 1u;  // buffer 0 when both are in use
    auto r = [&](u32 offset) { return reg(kPvideoBase + offset + 4 * i); };
    Overlay ov;
    ov.address = (r(0x00) + r(0x20)) & 0x07FFFFFFu;  // BASE + OFFSET
    ov.limit = r(0x08);
    const u32 size_in = r(0x28), point_in = r(0x30), point_out = r(0x48), size_out = r(0x50), format = r(0x58);
    ov.in_w = size_in & 0x7FFu;
    ov.in_h = (size_in >> 16) & 0x7FFu;
    ov.s0 = point_in & 0x7FFFu;
    ov.t0 = (point_in >> 16) & 0xFFFEu;
    ov.dsdx = r(0x38) ? r(0x38) : (1u << 20);
    ov.dtdy = r(0x40) ? r(0x40) : (1u << 20);
    ov.out_x = point_out & 0xFFFu;
    ov.out_y = (point_out >> 16) & 0xFFFu;
    ov.out_w = size_out & 0xFFFu;
    ov.out_h = (size_out >> 16) & 0xFFFu;
    ov.pitch = format & 0x1FFFu;
    ov.yuy2 = ((format >> 16) & 3u) == 1u;
    ov.color_key = (format >> 20) & 1u;
    ov.key = reg(kPvideoColorKey);
    return ov;
}

}  // namespace mv

namespace mv {
std::optional<std::array<u8, 768>> Nv2aStub::gamma_lut() const {
    if (dac_identity_) return std::nullopt;
    return dac_;
}
}  // namespace mv
