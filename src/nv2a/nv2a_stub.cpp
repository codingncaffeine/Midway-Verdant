#include "nv2a/nv2a_stub.h"

#include "nv2a/gpu_timing.h"

#include "nv2a/vulkan/vk_renderer.h"

#include <chrono>
#include <algorithm>
#include <cstdlib>
#if defined(__linux__)
#include <pthread.h>
#endif

#include "cpu/backend.h"
#include "debug/trace.h"

namespace mv {

namespace {
// The registers that report how far the pusher has come, or that it is idle: a read waits until the FIFO thread has
// nothing left to run, so a guest polling them sees what it saw when the pusher ran inline. PATT_COLOR0 is a fence
// Direct3D pushes a value through and polls.
bool reports_progress(u32 key) {
    switch (key) {
        case Nv2aStub::kUser + 0x44:
        case Nv2aStub::kUser + 0x48:
        case Nv2aStub::kCache1DmaGet:
        case Nv2aStub::kCache1Ref:
        case Nv2aStub::kCache1Status:
        case Nv2aStub::kCache1DmaPush:
        case Nv2aStub::kPfifoRunoutStatus:
        case Nv2aStub::kPgraphStatus:
        case Nv2aStub::kPgraphPattColor0: return true;
        default: return false;
    }
}
}  // namespace

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
        auto gpu = lock_gpu();
        // What was queued before the vblank is drawn by it, as when the pusher runs inline: the FIFO thread works on the
        // host clock while the console runs on the CPU's, so a frame still in its hands at the vblank missed the flip and
        // the title fell to half rate. The CPU waits here (a long exit, not guest time); within the frame the overlap stays.
        if (fifo_threaded_ && fifo_busy_) {
            const auto t0 = std::chrono::steady_clock::now();
            wait_idle(gpu);
            ++vblank_waits_;
            vblank_wait_ns_ += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        }
        guest_now_.store(sched_.now(), std::memory_order_relaxed);
        ++vblanks_;
        sync_point(kSyncVblank);
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
    pusher_event_.name = "nv2a.pusher";
    pusher_event_.fire = [this](Cycles) {
        auto gpu = lock_gpu();
        // The GPU clock has reached the observable method the walk stopped at: it runs now, then the walk goes on.
        if (held_) {
            // The draws before it may have pixels not yet counted: with them, its time may lie further on.
            if (!pending_draws_.empty()) resolve_pixels();
            if (const Cycles start = gpu_timing::cycles_from_gpu(gpu_clock_); start > sched_.now()) {
                sched_.schedule(pusher_event_, start);
                return;
            }
            const PushbufferMethod m = *held_;
            held_.reset();
            charge(m);
            execute_method(m);
            walker_.clear_stop();
            push_mark(gpu_clock_, walker_.position());
        }
        run_pusher_inline();
    };
    alarm_.name = "nv2a.ptimer-alarm";
    alarm_.fire = [this](Cycles) {
        auto gpu = lock_gpu();
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
    auto gpu = lock_gpu();
    wait_idle(gpu);
    regs_.clear();
    wbc_.store(0, std::memory_order_relaxed);
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
    sched_.cancel(pusher_event_);
    held_.reset();
    get_marks_.clear();
    get_visible_ = 0;
    gpu_clock_ = 0;
    methods_since_mark_ = 0;
    marks_pushed_ = 0;
    pending_draws_.clear();
    zpass_count_ = 0;
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
// status word of zero (done). The count is the z-pass sample count since CLEAR_REPORT_VALUE. A report says every pixel
// of the screen passed, so a visibility test finds its object visible instead of a title polling for an unwritten
// report for ever. MV_ZPASS_REPORTS=1 (accurate core with a renderer) writes the renderer's occlusion counts instead
// (every draw before the report is counted by then: it waits for the GPU clock); not the default yet: one title's flare
// tests (a 4-vertex fan, depth LEQUAL, writes off, z cull mode) pass no sample in our renderer even with the depth
// test and culling off, so every flare vanished (session 18; the transformed vertices are the next thing to look at).
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
    static const bool real_counts = [] {
        const char* z = std::getenv("MV_ZPASS_REPORTS");
        return z && *z == '1';
    }();
    const u32 count = (real_counts && timed_ && renderer_) ? static_cast<u32>(std::min<u64>(zpass_count_, 0xFFFFFFFFu))
                                            : (scanout_programmed() ? scanout_width() * scanout_height() : 640u * 480u);
    for (int i = 0; i < 4; ++i) report[8 + i] = static_cast<u8>(count >> (8 * i));
    if (debug_zpass()) std::fprintf(stderr, "zpass: GET_REPORT %08x -> %u\n", parameter, count);
    bus_.write_ram(dma.base + offset, report, 16);
    ++reports_;
}

void Nv2aStub::schedule_vblank() {
    const u64 frame = frame_period();
    const Cycles now = sched_.now();
    sched_.schedule(vblank_, (now / frame + 1) * frame);
}

void Nv2aStub::update_irq() {
    if (fifo_threaded_ && std::this_thread::get_id() == fifo_thread_id_) {
        // The FIFO thread holds only the GPU lock: the machine applies the change at its next exit (service).
        irq_deferred_.store(true, std::memory_order_release);
        if (service_request_) service_request_();
        return;
    }
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
    const u64 gpu_cycles = muldiv64(gpu_now(), clk::kGpuHzNum * clk::kCpuHzDen, clk::kCpuHzNum * clk::kGpuHzDen);
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

// Instrument: MV_DEBUG_ZPASS=1 prints the z-pass methods, each counted draw's samples and every report to stderr.
bool Nv2aStub::debug_zpass() {
    static const bool on = [] {
        const char* z = std::getenv("MV_DEBUG_ZPASS");
        return z && *z == '1';
    }();
    return on;
}

// Methods whose effect the CPU can see: they wait for the GPU clock in the timed pusher.
bool Nv2aStub::observable(const PushbufferMethod& m) const {
    if (m.method == 0x0050 || m.method == 0x0104 || m.method == kMethodNoOperation) return m.method != kMethodNoOperation || m.value != 0;
    const u32 cls = bound_[m.subchannel];
    if (cls == kClassPattern && m.method == kMethodPatternColor0) return true;
    if (cls == 0x97) {
        switch (m.method) {
            case 0x012C: case 0x0130: case 0x0120: case 0x0124: case 0x17D0: case 0x1D70: return true;  // flips, reports, semaphore
            default: return false;
        }
    }
    return false;
}

// What a method costs the GPU (gpu_timing.h): the front end, and for Kelvin's draws and clears their work.
u64 Nv2aStub::method_clocks(const PushbufferMethod& m) const {
    u64 c = gpu_timing::kMethodClocks;
    if (renderer_ && bound_[m.subchannel] == 0x97) {
        if (m.method == pgraph::mthd::kBeginEnd && m.value == 0 && kelvin_.in_begin()) c += gpu_timing::draw_clocks(kelvin_.state(), kelvin_.batch());
        else if (m.method == pgraph::mthd::kClearSurface) c += gpu_timing::clear_clocks(kelvin_.state());
    }
    return c;
}

void Nv2aStub::charge(const PushbufferMethod& m) {
    gpu_clock_ += method_clocks(m);
    if (++methods_since_mark_ >= 32) {
        methods_since_mark_ = 0;
        push_mark(gpu_clock_, walker_.position());
    }
}

bool Nv2aStub::gpu_busy() {
    if (held_ || gpu_timing::cycles_from_gpu(gpu_clock_) > sched_.now()) return true;
    if (pending_draws_.empty()) return false;
    resolve_pixels();  // idle without the pixels: they decide
    return gpu_timing::cycles_from_gpu(gpu_clock_) > sched_.now();
}

void Nv2aStub::push_mark(u64 clock, u32 get) { get_marks_.push_back({clock, get, marks_pushed_++}); }

// The pending draws' pixels from the renderer (it waits for the GPU only for counts not yet in): the GPU clock and
// every mark pushed after each draw move on by what that draw's pixels took.
void Nv2aStub::resolve_pixels() {
    ++pixel_resolves_;
    u64 delta = 0;
    auto mk = get_marks_.begin();
    for (const PendingDraw& d : pending_draws_) {
        for (; mk != get_marks_.end() && mk->index < d.mark_index; ++mk) mk->clock += delta;
        if (d.ticket == 0) {
            zpass_count_ = 0;
            continue;
        }
        const auto counts = renderer_ ? renderer_->draw_counts(d.ticket) : pgraph::VulkanRenderer::DrawCounts{};
        if (d.zpass) zpass_count_ += counts.samples;
        if (debug_zpass()) std::fprintf(stderr, "zpass: draw %llu samples %llu pixels %llu%s\n", static_cast<unsigned long long>(d.ticket),
                                        static_cast<unsigned long long>(counts.samples), static_cast<unsigned long long>(counts.fragments), d.zpass ? " counted" : "");
        const u64 c = gpu_timing::pixel_clocks(counts.fragments, d.passes, d.bytes);
        delta += c;
        timed_pixels_ += counts.fragments;
        pixel_clocks_ += c;
    }
    for (; mk != get_marks_.end(); ++mk) mk->clock += delta;
    gpu_clock_ += delta;
    pending_draws_.clear();
}

// GET as the guest sees it in the timed pusher: the position the GPU had reached at the guest's present.
u32 Nv2aStub::visible_get() {
    const Cycles now = sched_.now();
    // A mark after a draw whose pixels are not counted may move on: count them before trusting one in the present.
    if (!pending_draws_.empty()) {
        const u64 first = pending_draws_.front().mark_index;
        for (const GetMark& mk : get_marks_)
            if (mk.index >= first) {
                if (gpu_timing::cycles_from_gpu(mk.clock) <= now) resolve_pixels();
                break;
            }
    }
    while (!get_marks_.empty() && gpu_timing::cycles_from_gpu(get_marks_.front().clock) <= now) {
        get_visible_ = get_marks_.front().get;
        get_marks_.pop_front();
    }
    if (get_marks_.empty() && !held_) get_visible_ = reg(kCache1DmaGet);
    return get_visible_;
}

void Nv2aStub::on_method(const PushbufferMethod& m) {
    if (timed_) {
        // An observable method the GPU clock has not reached: the walk stops here and an event runs it at its time.
        Cycles start = gpu_timing::cycles_from_gpu(gpu_clock_);
        // Observable and reached without the pending draws' pixels: with them it may not be reached yet.
        if (observable(m) && start <= sched_.now() && !pending_draws_.empty()) {
            resolve_pixels();
            start = gpu_timing::cycles_from_gpu(gpu_clock_);
        }
        if (observable(m) && start > sched_.now()) {
            held_ = m;
            walker_.request_stop();
            push_mark(gpu_clock_, walker_.position() - 4);  // GET stands at this method until it runs
            sched_.cancel(pusher_event_);
            sched_.schedule(pusher_event_, start);
            return;
        }
        charge(m);
    }
    execute_method(m);
}

void Nv2aStub::execute_method(const PushbufferMethod& m) {
    // On the FIFO thread the walk stops every few methods to let a waiting register access in (the rest of the header
    // follows on the next chunk).
    if (fifo_threaded_ && ++chunk_methods_ >= kChunkMethods) walker_.request_stop();
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
    if (bound_[m.subchannel] == kClassPattern && m.method == kMethodPatternColor0) regs_[kPgraphPattColor0] = m.value;
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
        const bool draw_end = timed_ && m.method == pgraph::mthd::kBeginEnd && m.value == 0 && kelvin_.in_begin();
        const u64 passes = draw_end ? gpu_timing::pixel_passes(kelvin_.state()) : 0;
        const u64 bytes = draw_end ? gpu_timing::pixel_bytes(kelvin_.state()) : 0;
        const u64 ticket_before = renderer_->last_draw_ticket();
        const unsigned debug_primitive = draw_end ? static_cast<unsigned>(kelvin_.batch().primitive) : 0u;
        const u64 debug_vertices = draw_end && debug_zpass() ? gpu_timing::vertex_count(kelvin_.state(), kelvin_.batch()) : 0;
        kelvin_.method(m.method, m.value);
        // A draw the renderer counted: its pixels join the GPU clock when they are needed (stage B).
        if (draw_end) {
            const u64 ticket = renderer_->last_draw_ticket();
            if (ticket && ticket != ticket_before)
                pending_draws_.push_back({ticket, passes, bytes, marks_pushed_, (kelvin_.state().reg(kMethodZpassEnable) & 1u) != 0});
            if (debug_zpass() && (kelvin_.state().reg(kMethodZpassEnable) & 1u)) {
                const auto& st = kelvin_.state();
                std::fprintf(stderr, "zpass: counted draw ticket %llu (before %llu): primitive %u, %llu vertices, depth test %u func %04x write %u, stencil %u, colour mask %08x, cull %u face %04x, program mode %u, clip %08x %08x, zmin/max control %08x\n",
                             static_cast<unsigned long long>(ticket), static_cast<unsigned long long>(ticket_before), debug_primitive,
                             static_cast<unsigned long long>(debug_vertices), st.depth_test() ? 1u : 0u, st.reg(pgraph::mthd::kDepthFunc), st.depth_write() ? 1u : 0u,
                             st.stencil_test() ? 1u : 0u, st.color_mask(), st.cull_face() ? 1u : 0u, st.reg(pgraph::mthd::kCullFace), st.transform_mode(),
                             st.reg(pgraph::mthd::kSurfaceClipH), st.reg(pgraph::mthd::kSurfaceClipV), st.reg(kMethodZminMaxControl));
            }
        }
        if (debug_zpass() && (m.method == kMethodClearReport || m.method == kMethodZpassEnable)) std::fprintf(stderr, "zpass: method %04x <- %08x\n", m.method, m.value);
        if (timed_ && m.method == kMethodClearReport && m.value == 1) {
            if (pending_draws_.empty()) zpass_count_ = 0;
            else pending_draws_.push_back({0, 0, 0, marks_pushed_, false});
        }
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

void Nv2aStub::watch_writes(u32 phys, u32 len) const {
    if (!write_tracking_ || !cpu_ || len == 0) return;
    const u32 first = phys / capture::kPageSize, last = static_cast<u32>((static_cast<u64>(phys) + len - 1) / capture::kPageSize);
    cpu_->watch_pages(first, last - first + 1);
}

u32 Nv2aStub::newest_write(u32 phys, u32 len) const {
    if (!write_tracking_ || len == 0) return 0;
    const std::size_t pages = page_generation_.size();
    const std::size_t first = phys / capture::kPageSize, last = (static_cast<std::size_t>(phys) + len - 1) / capture::kPageSize;
    u32 newest = 0;
    for (std::size_t p = first; p <= last && p < pages; ++p) newest = std::max(newest, page_generation_[p]);
    return newest;
}

// The logs read: the CPU's (KVM's dirty pages) and the bus's. With the FIFO thread this runs at the kick on the CPU's
// thread, out of the guest: reading and re-protecting the log while the guest runs flushed its TLB and cost it
// guest time (a racing title's CPU time per frame 2.4 -> 3.5 ms).
void Nv2aStub::harvest_writes() {
    if (!write_tracking_ && !capture_) return;
    const auto harvest_t0 = std::chrono::steady_clock::now();
    ++collect_calls_;
    if (cpu_) cpu_->take_dirty_pages(dirty_pending_);
    bus_.take_dirty_pages(dirty_pending_);
    collect_ns_ += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - harvest_t0).count());
}

void Nv2aStub::collect_writes() {
    if (!write_tracking_ && !capture_) return;
    if (!fifo_threaded_) harvest_writes();
    // What was written since the last run, by the guest or a device.
    dirty_scratch_.swap(dirty_pending_);
    dirty_pending_.clear();
    collect_pages_ += dirty_scratch_.size();
    if (dirty_scratch_.empty()) return;
    std::sort(dirty_scratch_.begin(), dirty_scratch_.end());
    dirty_scratch_.erase(std::unique(dirty_scratch_.begin(), dirty_scratch_.end()), dirty_scratch_.end());
    if (renderer_) renderer_->cpu_wrote(dirty_scratch_);
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

void Nv2aStub::sync_point(u32 why) {
    if (!run_open_) return;
    run_open_ = false;
    ++sync_points_;
    const u64 gap = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - last_run_end_).count());
    overlap_ns_ += std::min(gap, last_run_ns_);
    auto& r = sync_reasons_[why];
    ++r.first;
    r.second += last_run_ns_ - std::min(gap, last_run_ns_);
}

void Nv2aStub::run_pusher() {
    if (fifo_threaded_) {
        // A kick: the FIFO thread walks it; until it has, the registers that report progress wait when read.
        stuck_ = false;
        if (pusher_runnable()) {
            harvest_writes();  // on the CPU's thread: the FIFO thread applies them
            fifo_busy_ = true;
            fifo_cv_.notify_one();
            if (lead_cycles_ && !lead_.scheduled()) sched_.schedule(lead_, sched_.now() + lead_cycles_);
        }
        return;
    }
    run_pusher_inline();
}

void Nv2aStub::run_pusher_inline() {
    // Timed: an observable method waits for the GPU clock; its event goes on with the walk (a kick meanwhile only moved PUT).
    if (timed_ && held_) return;
    chunk_methods_ = 0;
    const u32 get_before = reg(kCache1DmaGet);
    const u64 methods_before = walker_.methods_seen();
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
    sync_point(kSyncNextRun);  // a run still open: the next one would queue behind it
    if (timed_) gpu_clock_ = std::max(gpu_clock_, gpu_timing::gpu_from_cycles(sched_.now()));  // an idle GPU starts now
    const auto t0 = std::chrono::steady_clock::now();
    get = walker_.run(dma, get, put);
    if (timed_) push_mark(gpu_clock_, get);
    const auto t1 = std::chrono::steady_clock::now();
    last_run_ns_ = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    pusher_wall_ns_ += last_run_ns_;
    last_run_end_ = t1;
    run_open_ = true;
    regs_[kCache1DmaGet] = get;
    // A walk that moved nothing (a fetch outside its object, an unknown header) would be retried for ever by the FIFO
    // thread; it waits for the next kick instead, as the inline pusher does.
    if (get == get_before && walker_.methods_seen() == methods_before) stuck_ = true;
}

u32 Nv2aStub::mmio_read(PhysAddr addr, unsigned size) {
    const u32 off = addr - mapped_at_;
    const u32 key = off & ~3u;
    if (key == kPfbWbc) {
        wbc_accesses_.fetch_add(1, std::memory_order_relaxed);
        const u32 word = wbc_.load(std::memory_order_relaxed);
        const unsigned shift = 8u * (off & 3u);
        return size == 4 ? word : (size == 2 ? ((word >> shift) & 0xFFFFu) : ((word >> shift) & 0xFFu));
    }
    auto gpu = lock_gpu();
    guest_now_.store(sched_.now(), std::memory_order_relaxed);
    if (fifo_busy_ && (key == kUser + 0x44 || key == kCache1DmaGet)) timed_wait_progress(gpu, key);
    else if (fifo_busy_ && key == kPgraphPattColor0) timed_wait_change(gpu, key, patt_seen_, key);
    else if (fifo_busy_ && reports_progress(key)) timed_wait_idle(gpu, key);
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
    // The write-back cache flush is the CPU's own write combining: it needs nothing from the pusher.
    if (key != kPtimerTime0 && key != kPtimerTime1 && key != kPcrtcRaster && key != kPfbWbc) sync_point(key);
    u32 word;
    switch (key) {
        case kPtimerTime0: word = static_cast<u32>((ptimer_ticks() & 0x7FFFFFFu) << 5); break;
        case kPtimerTime1: word = static_cast<u32>((ptimer_ticks() >> 27) & 0x1FFFFFFFu); break;
        case kPcrtcRaster: word = raster_line(); break;
        case kPfbCstatus: word = static_cast<u32>(bus_.ram_size()); break;  // the RAM size in bytes
        case kUser + 0x40: word = reg(kCache1DmaPut); break;
        case kUser + 0x44: word = timed_ ? visible_get() : reg(kCache1DmaGet); break;
        case kCache1DmaGet: word = timed_ ? visible_get() : reg(kCache1DmaGet); break;
        case kPgraphStatus: word = (timed_ && gpu_busy()) ? (reg(key) | 1u) : reg(key); break;  // bit 0: the engine is busy
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
    {
        auto& rc = fifo_busy_ ? busy_read_counts_[key] : read_counts_[key];
        ++rc.first;
        rc.second = word;
    }
    ++reads_;
    if (key != kPtimerTime0 && key != kPtimerTime1) trace::log(trace::Cat::Gpu, "read  +{:06x} size {} -> {:08x}", off, size, v);
    return v;
}

void Nv2aStub::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 off = addr - mapped_at_;
    const u32 key = off & ~3u;
    if (key == kPfbWbc) {
        // Bit 16 starts a frame-buffer cache operation and clears when it is done; done at once here.
        wbc_accesses_.fetch_add(1, std::memory_order_relaxed);
        u32 word = value;
        if (size != 4) {
            const unsigned shift = 8u * (off & 3u);
            const u32 mask = (size == 2 ? 0xFFFFu : 0xFFu) << shift;
            word = (wbc_.load(std::memory_order_relaxed) & ~mask) | ((value << shift) & mask);
        }
        wbc_.store(word & ~(1u << 16), std::memory_order_relaxed);
        return;
    }
    auto gpu = lock_gpu();
    guest_now_.store(sched_.now(), std::memory_order_relaxed);
    // The FIFO's own registers change under a walk in progress only by a kick (PUT, the push enable); any other write
    // there (a rewound GET, a new DMA object) waits for it to finish.
    if (fifo_busy_ && ((key >= 0x002000 && key < 0x004000 && key != kCache1DmaPut && key != kCache1DmaPush) || key == kUser + 0x44)) timed_wait_idle(gpu, key | 0x80000000u);
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
    if (size != 1 && key != kCache1DmaPut && key != kUser + 0x40 && key != kCache1DmaPush && key != kPfbWbc) sync_point(key | 0x80000000u);
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
                ++display_flips_;
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

namespace mv {
std::string Nv2aStub::busiest_reads(std::size_t top, bool while_busy) const {
    const auto& counts = while_busy ? busy_read_counts_ : read_counts_;
    std::vector<std::pair<u32, std::pair<u64, u32>>> v(counts.begin(), counts.end());
    std::ranges::sort(v, std::greater{}, [](const auto& e) { return e.second.first; });
    std::string r;
    for (std::size_t i = 0; i < v.size() && i < top; ++i) r += std::format(" +{:06x} x{} (last {:08x})", v[i].first, v[i].second.first, v[i].second.second);
    if (!while_busy) r += std::format("; +{:06x} (lock-free) read and written x{}", kPfbWbc, wbc_accesses_.load(std::memory_order_relaxed));
    return r;
}
}  // namespace mv

namespace mv {
std::string Nv2aStub::sync_reasons(std::size_t top) const {
    std::vector<std::pair<u32, std::pair<u64, u64>>> v(sync_reasons_.begin(), sync_reasons_.end());
    std::ranges::sort(v, std::greater{}, [](const auto& e) { return e.second.second; });
    std::string r;
    for (std::size_t i = 0; i < v.size() && i < top; ++i) {
        const u32 why = v[i].first;
        const std::string name = why == kSyncVblank ? "vblank" : why == kSyncNextRun ? "next run" : std::format("{} +{:06x}", (why >> 31) ? "write" : "read", why & 0x7FFFFFFFu);
        r += std::format(" {} x{} {:.3f} s;", name, v[i].second.first, static_cast<double>(v[i].second.second) / 1e9);
    }
    return r;
}
}  // namespace mv

namespace mv {

// ── the FIFO thread (PLAN D21) ──────────────────────────────────────────────────────────────────────────────────

Nv2aStub::~Nv2aStub() { stop_fifo_thread(); }

std::unique_lock<std::mutex> Nv2aStub::lock_gpu() const {
    if (!fifo_threaded_) return {};
    gpu_waiters_.fetch_add(1, std::memory_order_acq_rel);
    std::unique_lock l(gpu_mutex_, std::try_to_lock);
    if (!l.owns_lock()) {
        const auto t0 = std::chrono::steady_clock::now();
        l.lock();
        lock_wait_ns_.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
        lock_waits_.fetch_add(1, std::memory_order_relaxed);
    }
    gpu_waiters_.fetch_sub(1, std::memory_order_acq_rel);
    return l;
}

void Nv2aStub::wait_idle(std::unique_lock<std::mutex>& lock) {
    if (!fifo_threaded_ || !lock.owns_lock()) return;
    idle_cv_.wait(lock, [this] { return !fifo_busy_; });
}

void Nv2aStub::timed_wait_change(std::unique_lock<std::mutex>& lock, u32 key, u32& seen, u32 why) {
    const u32 last = seen;
    if (reg(key) == last) {
        const auto t0 = std::chrono::steady_clock::now();
        idle_cv_.wait(lock, [&] { return !fifo_busy_ || reg(key) != last; });
        auto& w = idle_waits_[why];
        ++w.first;
        w.second += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }
    seen = reg(key);
}

void Nv2aStub::timed_wait_idle(std::unique_lock<std::mutex>& lock, u32 why) {
    const auto t0 = std::chrono::steady_clock::now();
    wait_idle(lock);
    auto& w = idle_waits_[why];
    ++w.first;
    w.second += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

std::string Nv2aStub::fifo_wait_report() const {
    std::vector<std::pair<u32, std::pair<u64, u64>>> v(idle_waits_.begin(), idle_waits_.end());
    std::ranges::sort(v, std::greater{}, [](const auto& e) { return e.second.second; });
    std::string r = std::format("the lock {} times ({:.3f} s);", lock_waits_.load(), static_cast<double>(lock_wait_ns_.load()) / 1e9);
    for (std::size_t i = 0; i < v.size() && i < 8; ++i)
        r += std::format(" {} +{:06x} x{} {:.3f} s;", (v[i].first >> 31) ? "write" : (v[i].first & 0x40000000u) ? "poll" : "read", v[i].first & 0x3FFFFFFFu, v[i].second.first, static_cast<double>(v[i].second.second) / 1e9);
    return r;
}

void Nv2aStub::wait_fifo_idle() {
    auto gpu = lock_gpu();
    wait_idle(gpu);
}

Cycles Nv2aStub::gpu_now() const {
    if (fifo_threaded_ && std::this_thread::get_id() == fifo_thread_id_) return guest_now_.load(std::memory_order_relaxed);
    return sched_.now();
}

bool Nv2aStub::pusher_runnable() const {
    if (stuck_ || !(reg(kCache1DmaPush) & 1u) || pgraph_stalled_) return false;
    if (flip_stalled_ && !flip_released()) return false;
    if (reg(kCache1DmaPut) == reg(kCache1DmaGet)) return false;
    return dma_object(reg(kCache1DmaInstance) & 0xFFFFu).valid;
}

void Nv2aStub::start_fifo_thread(std::function<void()> request) {
    if (fifo_threaded_) return;
    {
        const char* l = std::getenv("MV_FIFO_LEAD_US");
        const u64 us = l && *l ? std::strtoull(l, nullptr, 10) : 2000;
        lead_cycles_ = cycles_from_ns(us * 1000);
    }
    lead_.name = "nv2a.fifo-lead";
    lead_.fire = [this](Cycles) {
        // Under the machine lock: the CPU stops at its next exit until the GPU has caught up (a long exit, not guest time).
        auto gpu = lock_gpu();
        if (!fifo_busy_) return;
        const auto t0 = std::chrono::steady_clock::now();
        wait_idle(gpu);
        ++lead_waits_;
        lead_wait_ns_ += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    };
    service_request_ = std::move(request);
    guest_now_.store(sched_.now(), std::memory_order_relaxed);
    fifo_stop_ = false;
    fifo_threaded_ = true;
    fifo_thread_ = std::thread([this] { fifo_main(); });
}

void Nv2aStub::stop_fifo_thread() {
    if (!fifo_threaded_) return;
    {
        std::lock_guard g(gpu_mutex_);
        fifo_stop_ = true;
    }
    fifo_cv_.notify_all();
    if (fifo_thread_.joinable()) fifo_thread_.join();
    fifo_threaded_ = false;
    fifo_busy_ = false;
    sched_.cancel(lead_);
    // An interrupt the thread left for the machine is applied here, on the machine's side.
    if (irq_deferred_.exchange(false)) update_irq();
}

void Nv2aStub::service() {
    auto gpu = lock_gpu();
    if (irq_deferred_.exchange(false, std::memory_order_acq_rel)) update_irq();
}

void Nv2aStub::fifo_main() {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "mv-nv2a-fifo");
#endif
    std::unique_lock g(gpu_mutex_);
    fifo_thread_id_ = std::this_thread::get_id();
    for (;;) {
        fifo_cv_.wait(g, [this] { return fifo_stop_ || pusher_runnable(); });
        if (fifo_stop_) break;
        fifo_busy_ = true;
        const u32 get_before = reg(kCache1DmaGet);
        run_pusher_inline();
        if (reg(kCache1DmaGet) != get_before) idle_cv_.notify_all();  // a poll of GET waits for progress
        if (!pusher_runnable()) {
            fifo_busy_ = false;
            idle_cv_.notify_all();
            continue;
        }
        // Between chunks, a register access waiting for the lock goes first.
        if (gpu_waiters_.load(std::memory_order_acquire)) {
            g.unlock();
            while (gpu_waiters_.load(std::memory_order_acquire)) std::this_thread::yield();
            g.lock();
        }
    }
    fifo_busy_ = false;
    idle_cv_.notify_all();
}

}  // namespace mv
