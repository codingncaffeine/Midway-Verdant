#include "nv2a/nv2a_stub.h"

#include "nv2a/vulkan/vk_renderer.h"

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
        regs_[kPcrtcIntr0] |= 1u;
        // The mode latch acknowledges software's toggle at the frame boundary: bit 4 takes bit 0.
        const u32 latch = reg(kPramdacModeLatch);
        regs_[kPramdacModeLatch] = (latch & ~0x10u) | ((latch & 1u) << 4);
        update_irq();
        schedule_vblank();
    };
    schedule_vblank();
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
    pgraph_stalled_ = false;
    kelvin_ = pgraph::Kelvin();
    kelvin_.set_sink(renderer_);
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
        bound_[m.subchannel] = ramin32(instance * 16) & 0xFFFFu;
        trace::log(trace::Cat::Gpu, "fifo: subchannel {} <- object {:08x} (class {:04x}, instance {:04x})", m.subchannel, m.value,
                   bound_[m.subchannel], instance);
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
        regs_[kPgraphTrappedAddr] = (m.subchannel << 13) | m.method;
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
    // The live GPU: every Kelvin method goes through the state machine to the renderer; the semaphore release is
    // the frame marker, so the frame is rendered and written back to RAM before the CPU can see the semaphore.
    if (renderer_ && bound_[m.subchannel] == 0x97) {
        kelvin_.method(m.method, m.value);
        if (m.method == 0x1D70) {
            renderer_->end_frame();
            ++rendered_frames_;
        }
    }
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

void Nv2aStub::capture_dirty_pages() {
    // Everything written to RAM since the last run, by the guest (the backend's log) or by a device (the bus's),
    // coalesced into contiguous runs of pages.
    dirty_scratch_.clear();
    if (cpu_) cpu_->take_dirty_pages(dirty_scratch_);
    bus_.take_dirty_pages(dirty_scratch_);
    std::sort(dirty_scratch_.begin(), dirty_scratch_.end());
    dirty_scratch_.erase(std::unique(dirty_scratch_.begin(), dirty_scratch_.end()), dirty_scratch_.end());
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
    const u32 put = reg(kCache1DmaPut);
    u32 get = reg(kCache1DmaGet);
    if (get == put) return;
    const auto dma = dma_object(reg(kCache1DmaInstance) & 0xFFFFu);
    if (!dma.valid) {
        trace::log(trace::Cat::Gpu, "pusher: no DMA object bound (instance {:04x})", reg(kCache1DmaInstance));
        return;
    }
    if (capture_) {
        capture_dirty_pages();
        capture_->run(sched_.now(), get, put);
    }
    get = walker_.run(dma, get, put);
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
    u32 word;
    switch (key) {
        case kPtimerTime0: word = static_cast<u32>((ptimer_ticks() & 0x7FFFFFFu) << 5); break;
        case kPtimerTime1: word = static_cast<u32>((ptimer_ticks() >> 27) & 0x1FFFFFFFu); break;
        case kUser + 0x40: word = reg(kCache1DmaPut); break;
        case kUser + 0x44: word = reg(kCache1DmaGet); break;
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
        case kPvideoIntr:
            regs_[key] &= ~word;
            update_irq();
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
            return;
        case kPmcIntr0:
            return;  // read-only aggregate
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
