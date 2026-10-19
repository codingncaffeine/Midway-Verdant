#include "apu/apu.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#if defined(__linux__)
#include <pthread.h>
#endif

#include "core/clock.h"
#include "debug/trace.h"

namespace mv {

namespace {
// GP: 4096 words of X plus the 1024-word MIXBUF at X:$1400, 2048 of Y, 4096 of P. EP: 3072 X, 256 Y, 4096 P.
// The GP's mix buffer (the voice processor's 32 bins) is X:0x1400-0x17FF, and the same words answer at X:0xC00-0xFFF:
// the GP has 3K words of its own X RAM, as the EP does. Inferred from the kernel's boot program, which adds its reverb
// into X:0xC00/0xC20/0xC80/0xCA0 (bins 0, 1, 4 and 5 from 0xC00: the front and rear speaker bins) and outputs
// X:0xC00/0xC20, while it reads its effects send, bin 11, at X:0x1560 [verify on hardware].
constexpr dsp::Core::Config kGpConfig{0x1800, 0x800, 0x1000, 0xC00, 0x1400};
constexpr dsp::Core::Config kEpConfig{0xC00, 0x100, 0x1000};
constexpr u32 kWord = 0xFFFFFFu;
}  // namespace

Apu::Apu(Bus& bus, Scheduler& sched, std::function<void(bool)> irq)
    : PciDevice("apu", 0x10DE, 0x01B0, 0x040100, 0xB1),
      bus_(bus),
      sched_(sched),
      irq_(std::move(irq)),
      regs_(kGpBlock / 4, 0),
      gp_memory_(*this, kGpScratchTable, kGpScratchMaxSge, kGpFifoTable, kGpFifoMaxSge, kGpFifoRegs),
      ep_memory_(*this, kEpScratchTable, kEpScratchMaxSge, kEpFifoTable, kEpFifoMaxSge, kEpFifoRegs),
      gp_(kGpConfig, gp_memory_),
      ep_(kEpConfig, ep_memory_),
      se_(bus, *this) {
    define_bar(0, BarKind::Memory, kBarSize);
    set_cfg8(0x3C, 5);  // interrupt line 5, pin A
    set_cfg8(0x3D, 1);
    frame_.name = "apu.frame";
    frame_.fire = [this](Cycles now) { on_frame(now); };
    if (const char* t = std::getenv("MV_APU_THREAD"); t && *t == '0') dsp_threaded_ = false;
    if (dsp_threaded_) dsp_thread_ = std::thread([this] { dsp_worker(); });
    if (const char* h = std::getenv("MV_DEBUG_DSP_PCHIST")) {
        if (std::strstr(h, "gp")) {
            gp_hist_.assign(gp_.core().p_memory().size(), 0);
            gp_.core().set_pc_histogram(&gp_hist_);
        }
        if (std::strstr(h, "ep")) {
            ep_hist_.assign(ep_.core().p_memory().size(), 0);
            ep_.core().set_pc_histogram(&ep_hist_);
        }
    }
}

Apu::~Apu() {
    if (dsp_thread_.joinable()) {
        {
            std::lock_guard g(dsp_mutex_);
            dsp_stop_ = true;
        }
        dsp_cv_.notify_all();
        dsp_thread_.join();
    }
}

void Apu::dsp_sync() const {
    if (!dsp_busy_.load(std::memory_order_acquire)) return;
    std::unique_lock l(dsp_mutex_);
    dsp_cv_.wait(l, [this] { return !dsp_busy_.load(std::memory_order_relaxed); });
}

void Apu::dsp_worker() {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "mv-apu-dsp");
#endif
    std::unique_lock l(dsp_mutex_);
    for (;;) {
        dsp_cv_.wait(l, [this] { return dsp_stop_ || dsp_busy_.load(std::memory_order_relaxed); });
        if (dsp_stop_) return;
        const u64 frame = dsp_job_frame_;
        const bool ep_start = dsp_job_ep_start_;
        l.unlock();
        run_processors(frame, ep_start);
        l.lock();
        dsp_busy_.store(false, std::memory_order_release);
        dsp_cv_.notify_all();
    }
}

void Apu::on_bar_changed(int index, u32 old_base, u32 new_base) {
    (void)old_base;
    if (index != 0) return;
    if (mapped_at_) bus_.unmap_mmio(mapped_at_, kBarSize);
    mapped_at_ = new_base;
    if (new_base) bus_.map_mmio(new_base, kBarSize, this);
    trace::log(trace::Cat::Apu, "apu registers at {:08x}", new_base);
}

void Apu::reset() {
    dsp_sync();
    apply_reset(gp_, gp_reset_, 0, "gp");
    apply_reset(ep_, ep_reset_, 0, "ep");
    for (auto& r : regs_) r = 0;
    other_.clear();
    fe_trapped_ = false;
    xgscnt_ = 0;
    update_frame_clock();  // SECTL is clear: the frame clock stops
    update_irq();
}

// ── the register window ─────────────────────────────────────────────────────────────────────────────────────────


u32 Apu::processor_word(apu::Processor& p, u32 offset, u32 reset_word) {
    auto& core = p.core();
    if (offset < kYMem) {
        const u32 w = offset / 4;
        return w < core.x_memory().size() ? core.x_memory()[w] : 0;
    }
    if (offset < kPMem) {
        const u32 w = (offset - kYMem) / 4;
        return w < core.y_memory().size() ? core.y_memory()[w] : 0;
    }
    if (offset < kPMem + core.p_memory().size() * 4) return core.read_p((offset - kPMem) / 4);
    if (offset == kReset) return reset_word;
    return 0;
}

bool Apu::processor_write(apu::Processor& p, u32 offset, u32 value) {
    auto& core = p.core();
    if (offset < kYMem) {
        const u32 w = offset / 4;
        if (w < core.x_memory().size()) core.x_memory()[w] = value & kWord;
        return true;
    }
    if (offset < kPMem) {
        const u32 w = (offset - kYMem) / 4;
        if (w < core.y_memory().size()) core.y_memory()[w] = value & kWord;
        return true;
    }
    if (offset < kPMem + core.p_memory().size() * 4) {
        core.write_p((offset - kPMem) / 4, value & kWord);
        return true;
    }
    return false;
}

u32 Apu::read_word(u32 offset) {
    if (offset == kXgsCnt) return xgscnt_;
    if (offset < kVpBlock) return regs_[offset / 4];
    if (offset < kGpBlock) {
        if (offset - kVpBlock == kVpFree) return 0x80;  // the queue is always drained at this level of modelling
        return regs_[offset / 4];
    }
    if (offset >= kGpBlock && offset < kGpBlock + kBlockSize) {
        const u32 in = offset - kGpBlock;
        if (in < kPMem + gp_.core().p_memory().size() * 4 || in == kReset) return processor_word(gp_, in, gp_reset_);
    }
    if (offset >= kEpBlock && offset < kEpBlock + kBlockSize) {
        const u32 in = offset - kEpBlock;
        if (in < kPMem + ep_.core().p_memory().size() * 4 || in == kReset) return processor_word(ep_, in, ep_reset_);
    }
    auto it = other_.find(offset);
    return it == other_.end() ? 0 : it->second;
}

void Apu::write_word(u32 offset, u32 value) {
    if (offset < kVpBlock) {
        if (offset == kIsts) {
            regs_[kIsts / 4] &= ~value;  // write one to clear
            update_irq();
            return;
        }
        regs_[offset / 4] = value;
        if (offset == kIen) update_irq();
        if (offset == kSectl) update_frame_clock();
        if (offset == kXgsCnt) xgscnt_ = value;
        if (offset == kFeMemData) {
            // The setup engine takes each voice method as it is written, so every method before the fence is done.
            const u32 at = regs_[kFeMemAddr / 4];
            if (!bus_.write_ram(at, &value, 4)) trace::log(trace::Cat::Apu, "fe: memory fence at {:08x} is outside RAM", at);
            ++fence_writes_;
        }
        // The host resumes a trapped front end by writing any other method mode.
        if (offset == kFectl && (value & kFectlMethodMode) != kFectlTrapped && fe_trapped_) {
            fe_trapped_ = false;
            trace::log(trace::Cat::Apu, "fe: resumed ({:08x}), {} trap(s) queued", value, trap_queue_.size());
        }
        return;
    }
    if (offset < kGpBlock) {
        // The voice processor's method window: the setup engine takes the voice methods; the rest is kept.
        const u32 m = offset - kVpBlock;
        if (m != kVpFree && !se_.method(m, value)) regs_[offset / 4] = value;
        return;
    }
    if (offset >= kGpBlock && offset < kGpBlock + kBlockSize) {
        const u32 in = offset - kGpBlock;
        if (in == kReset) {
            apply_reset(gp_, gp_reset_, value, "gp");
            return;
        }
        if (processor_write(gp_, in, value)) return;
    }
    if (offset >= kEpBlock && offset < kEpBlock + kBlockSize) {
        const u32 in = offset - kEpBlock;
        if (in == kReset) {
            apply_reset(ep_, ep_reset_, value, "ep");
            return;
        }
        if (processor_write(ep_, in, value)) return;
    }
    other_[offset] = value;
}

u32 Apu::mmio_read(PhysAddr addr, unsigned size) {
    const u32 off = addr - mapped_at_;
    if (!in_vp_window(off)) dsp_sync();
    const u32 word = read_word(off & ~3u);
    const unsigned shift = 8u * (off & 3u);
    const u32 v = size == 4 ? word : (size == 2 ? ((word >> shift) & 0xFFFFu) : ((word >> shift) & 0xFFu));
    ++reads_;
    trace::log(trace::Cat::Apu, "apu read  +{:05x} size {} -> {:08x}", off, size, v);
    return v;
}

void Apu::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 off = addr - mapped_at_;
    if (!in_vp_window(off)) dsp_sync();
    const u32 key = off & ~3u;
    u32 word = value;
    if (size != 4) {
        const unsigned shift = 8u * (off & 3u);
        const u32 mask = (size == 2 ? 0xFFFFu : 0xFFu) << shift;
        // A partial write to the status register clears only the bits it names.
        word = key == kIsts ? ((value << shift) & mask) : ((read_word(key) & ~mask) | ((value << shift) & mask));
    }
    write_word(key, word);
    ++writes_;
    trace::log(trace::Cat::Apu, "apu write +{:05x} size {} <- {:08x}", off, size, value);
}

void Apu::raise(u32 bits) {
    regs_[kIsts / 4] |= bits;
    update_irq();
}

void Apu::update_irq() {
    // The summary bit is derived: pending while another enabled bit is set and the summary itself is enabled.
    u32& ists = regs_[kIsts / 4];
    const u32 ien = regs_[kIen / 4];
    const bool level = (ien & kIstsGlobal) && ((ists & ien & ~kIstsGlobal) != 0);
    if (level) ists |= kIstsGlobal;
    else ists &= ~kIstsGlobal;
    if (level != irq_level_) {
        irq_level_ = level;
        if (irq_) irq_(level);
    }
}

// ── the processors ──────────────────────────────────────────────────────────────────────────────────────────────

void Apu::apply_reset(apu::Processor& p, u32& reset_word, u32 value, std::string_view which) {
    reset_word = value;
    const bool release = (value & (kResetBlock | kResetDsp)) == (kResetBlock | kResetDsp);
    if (release && !p.running()) {
        p.release_reset();
        if (&p == &ep_) {
            ++ep_releases_;
            ep_runs_since_release_ = 0;
        }
        if (p.running()) trace::log(trace::Cat::Apu, "{}: reset released, program loaded from scratch, P:0 = {:06x}", which, p.core().read_p(0));
        else trace::log(trace::Cat::Apu, "{}: reset released but the bootstrap failed: {}", which, p.last_problem());
    } else if (!release && p.running()) {
        p.hold_reset();
        trace::log(trace::Cat::Apu, "{}: reset held ({:x})", which, value);
    }
}

void Apu::update_frame_clock() {
    const bool on = (regs_[kSectl / 4] & kSectlCountMode) != 0;
    if (on == counting_) return;
    counting_ = on;
    if (on) {
        // The first frame one period after the count starts [verify on hardware].
        frame_epoch_ = sched_.now();
        epoch_frames_ = 0;
        sched_.schedule(frame_, frame_epoch_ + cycles_from_ticks(1, clk::kApuFramesPerSecond));
    } else {
        sched_.cancel(frame_);
    }
    trace::log(trace::Cat::Apu, "frame clock {} (sectl {:x}, xgscnt {:x})", on ? "on" : "off", regs_[kSectl / 4], xgscnt_);
}

void Apu::run_frame(apu::Processor& p, FrameStats& stats, std::string_view which, bool start_of_frame) {
    if (!p.running()) return;
    const auto t0 = std::chrono::steady_clock::now();
    const u64 used = p.run_frame(kDspHz / clk::kApuFramesPerSecond, start_of_frame);
    stats.host_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    stats.last_cycles = used;
    stats.total_cycles += used;
    stats.max_cycles = std::max(stats.max_cycles, used);
    ++stats.frames_run;
    bool& logged = &p == &gp_ ? gp_problem_logged_ : ep_problem_logged_;
    if (!logged && (p.core().error() || !p.dma().last_error().empty())) {
        logged = true;
        trace::log(trace::Cat::Apu, "{}: frame {}: {}{}", which, frames_, p.core().error() ? p.core().error_text() : "", p.dma().last_error());
    }
}

void Apu::on_frame(Cycles now) {
    (void)now;
    dsp_sync();  // the last frame's processor run, before this frame touches their memories
    ++frames_;
    // The setup engine walks the voice lists first; its idle reports trap to the host once the front end is free.
    {
        const auto t0 = std::chrono::steady_clock::now();
        se_.frame();
        se_host_ns_ += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }
    if (!fe_trapped_ && !trap_queue_.empty()) {
        const auto [method, parameter] = trap_queue_.front();
        trap_queue_.pop_front();
        deliver_trap(method, parameter);
    }
    // The voice processor's 32 bins land in the GP's MIXBUF as 24-bit samples: full scale is 2^23, clamped, so a
    // 16-bit source sample arrives exactly as itself shifted left by 8.
    auto& x = gp_.core().x_memory();
    if (x.size() >= kMixBufWord + kMixBufWords) {
        const auto& mix = se_.mix();
        for (u32 bin = 0; bin < apu::SetupEngine::kBins; ++bin)
            for (u32 i = 0; i < apu::SetupEngine::kSamplesPerFrame; ++i) {
                const i32 s = static_cast<i32>(std::clamp(std::lround(static_cast<double>(mix[bin][i]) * 8388608.0), -8388608L, 8388607L));
                x[kMixBufWord + bin * apu::SetupEngine::kSamplesPerFrame + i] = static_cast<u32>(s) & 0xFFFFFFu;
            }
        for (u32 i = 0; i < apu::SetupEngine::kSamplesPerFrame; ++i) {
            const u32 l = x[kMixBufWord + i], r = x[kMixBufWord + apu::SetupEngine::kSamplesPerFrame + i];
            if (l != 0 || r != 0) {
                ++mix_samples_;
                if (l != r) ++mix_differ_;
            }
        }
    }
    // Instrument, twice a second under the APU trace: the RMS level each processor wrote to its output FIFO 0.
    if (frames_ % 750 == 0 && trace::on(trace::Cat::Apu)) {
        auto level = [](Memory& m) {
            const double v = m.out0_samples_ ? 10 * std::log10(m.out0_energy_ / static_cast<double>(m.out0_samples_) / (32768.0 * 32768.0) + 1e-12) : -120.0;
            m.out0_energy_ = 0;
            m.out0_samples_ = 0;
            return v;
        };
        const double gp_db = level(gp_memory_), ep_db = level(ep_memory_);
        trace::log(trace::Cat::Apu, "fifo0 rms dB: gp {:.1f} ep {:.1f}", gp_db, ep_db);
        // Where a stereo source turns mono: the frames whose two channels differ in MIXBUF bins 0/1 (counted every
        // frame below) and in what the GP and the EP wrote to their output FIFO 0.
        trace::log(trace::Cat::Apu, "stereo L != R: mixbuf bins 0/1 {} of {} samples, gp fifo0 {} of {} frames, ep fifo0 {} of {} frames", mix_differ_, mix_samples_,
                   gp_memory_.out0_differ_, gp_memory_.out0_frames_, ep_memory_.out0_differ_, ep_memory_.out0_frames_);
        mix_differ_ = mix_samples_ = 0;
        gp_memory_.out0_differ_ = gp_memory_.out0_frames_ = ep_memory_.out0_differ_ = ep_memory_.out0_frames_ = 0;
    }
    // The processors run on their own thread, beside the CPU (the hardware runs them alongside it too): anything that
    // reads or writes their state from here on waits for this frame first (dsp_sync), so what the CPU observes is
    // what it observed when the frame ran inline. MV_APU_THREAD=0 runs them inline.
    // Their output to RAM is not waited for: the AC'97 plays the EP's ring milliseconds behind where the EP writes it
    // (the hardware runs the processors alongside the CPU too, so titles keep that lead), and every frame before
    // this one has finished.
    const bool ep_start = (xgscnt_ / apu::SetupEngine::kSamplesPerFrame) % kEpFrameDivider == 0;
    if (dsp_threaded_) {
        std::lock_guard g(dsp_mutex_);
        dsp_job_frame_ = frames_;
        dsp_job_ep_start_ = ep_start;
        dsp_busy_.store(true, std::memory_order_release);
        dsp_cv_.notify_all();
    } else {
        run_processors(frames_, ep_start);
    }
    xgscnt_ += apu::SetupEngine::kSamplesPerFrame;
    ++epoch_frames_;
    sched_.schedule(frame_, frame_epoch_ + cycles_from_ticks(epoch_frames_ + 1, clk::kApuFramesPerSecond));
}

// One frame of the global processor, then the encode processor (on the processors' thread unless MV_APU_THREAD=0).
void Apu::run_processors(u64 frame, bool ep_start) {
    // Instrument: MV_DEBUG_GP_FRAME=N writes the GP's instructions during frame N to scratch/gp-frame-N.trace.
    static const long debug_frame = std::getenv("MV_DEBUG_GP_FRAME") ? std::atol(std::getenv("MV_DEBUG_GP_FRAME")) : -1;
    std::FILE* gp_trace = nullptr;
    if (debug_frame >= 0 && frame == static_cast<u64>(debug_frame)) {
        gp_trace = std::fopen(std::format("scratch/gp-frame-{}.trace", debug_frame).c_str(), "w");
        gp_.core().set_trace(gp_trace);
        gp_.dma().set_trace(gp_trace);
        if (gp_trace) {
            const auto& xm = gp_.core().x_memory();
            for (u32 a = 0; a < 0x200 && a < xm.size(); a += 8)
                std::fprintf(gp_trace, "X:%04x %06x %06x %06x %06x %06x %06x %06x %06x\n", a, xm[a], xm[a + 1], xm[a + 2], xm[a + 3], xm[a + 4], xm[a + 5], xm[a + 6], xm[a + 7]);
        }
    }
    run_frame(gp_, gp_stats_, "gp", true);
    if (gp_trace) {
        gp_.core().set_trace(nullptr);
        gp_.dma().set_trace(nullptr);
        std::fclose(gp_trace);
    }
    // Instrument: MV_DEBUG_EP_FRAME=N does the same for the EP during frame N (scratch/ep-frame-N.trace), and
    // MV_DEBUG_EP_RUNS=K for its first K runs after each reset release (scratch/ep-run-R-K.trace: the fast mode's
    // frame numbers move from run to run, the EP's own runs do not).
    static const long debug_ep_frame = std::getenv("MV_DEBUG_EP_FRAME") ? std::atol(std::getenv("MV_DEBUG_EP_FRAME")) : -1;
    static const long debug_ep_runs = std::getenv("MV_DEBUG_EP_RUNS") ? std::atol(std::getenv("MV_DEBUG_EP_RUNS")) : 0;
    std::FILE* ep_trace = nullptr;
    std::string trace_name;
    if (debug_ep_frame >= 0 && frame == static_cast<u64>(debug_ep_frame)) trace_name = std::format("scratch/ep-frame-{}", debug_ep_frame);
    if (ep_.running() && ep_runs_since_release_ < static_cast<u64>(debug_ep_runs))
        trace_name = std::format("scratch/ep-run-{}-{}", ep_releases_, ep_runs_since_release_);
    if (ep_.running()) ++ep_runs_since_release_;
    if (!trace_name.empty()) {
        ep_trace = std::fopen((trace_name + ".trace").c_str(), "w");
        // With it, the program as it stands (P with raw words) and the X and Y memories, for reading the program offline.
        if (std::FILE* lst = std::fopen((trace_name + ".lst").c_str(), "w")) {
            auto& core = ep_.core();
            for (u32 a = 0; a < core.p_memory().size(); ++a) std::fprintf(lst, "P:%04x %06x  %s\n", a, core.read_p(a), core.disassemble_at(a).c_str());
            for (u32 a = 0; a < core.x_memory().size(); a += 8) {
                std::fprintf(lst, "X:%04x", a);
                for (u32 i = 0; i < 8 && a + i < core.x_memory().size(); ++i) std::fprintf(lst, " %06x", core.x_memory()[a + i]);
                std::fprintf(lst, "\n");
            }
            for (u32 a = 0; a < core.y_memory().size(); a += 8) {
                std::fprintf(lst, "Y:%04x", a);
                for (u32 i = 0; i < 8 && a + i < core.y_memory().size(); ++i) std::fprintf(lst, " %06x", core.y_memory()[a + i]);
                std::fprintf(lst, "\n");
            }
            std::fclose(lst);
        }
        ep_.core().set_trace(ep_trace);
        ep_.dma().set_trace(ep_trace);
    }
    run_frame(ep_, ep_stats_, "ep", ep_start);
    if (ep_trace) {
        ep_.core().set_trace(nullptr);
        ep_.dma().set_trace(nullptr);
        std::fclose(ep_trace);
    }
}

// ── scratch and FIFO spaces ─────────────────────────────────────────────────────────────────────────────────────

bool Apu::Memory::translate(u32 table_reg, u32 max_reg, u32 offset, PhysAddr& phys) {
    const u32 page = offset / static_cast<u32>(Bus::kPageSize);
    if (page > apu_.read_word(max_reg)) return false;
    const PhysAddr table = apu_.read_word(table_reg);
    u32 entry = 0;
    if (!apu_.bus_.read_ram(table + static_cast<PhysAddr>(page) * 8, &entry, 4)) return false;
    phys = (entry & ~static_cast<u32>(Bus::kPageSize - 1)) + offset % static_cast<u32>(Bus::kPageSize);
    return true;
}

bool Apu::Memory::copy(u32 table_reg, u32 max_reg, u32 offset, u8* out, const u8* in, u32 len) {
    while (len) {
        PhysAddr phys;
        if (!translate(table_reg, max_reg, offset, phys)) {
            if (problem_.empty()) problem_ = std::format("offset {:x} is outside the table at {:08x}", offset, apu_.read_word(table_reg));
            return false;
        }
        const u32 chunk = std::min<u32>(len, static_cast<u32>(Bus::kPageSize) - offset % static_cast<u32>(Bus::kPageSize));
        const bool ok = out ? apu_.bus_.read_ram(phys, out, chunk) : apu_.bus_.write_ram(phys, in, chunk);
        if (!ok) {
            if (problem_.empty()) problem_ = std::format("page {:08x} is outside RAM", phys);
            return false;
        }
        if (out) out += chunk;
        if (in) in += chunk;
        offset += chunk;
        len -= chunk;
    }
    return true;
}

bool Apu::Memory::scratch_read(u32 offset, u8* out, u32 len) { return copy(scratch_table_, scratch_max_, offset, out, nullptr, len); }
bool Apu::Memory::scratch_write(u32 offset, const u8* in, u32 len) { return copy(scratch_table_, scratch_max_, offset, nullptr, in, len); }

// A ring in the FIFO space: the current pointer advances and returns to the base at the end; a transfer longer
// than the ring keeps going round. A current pointer below the base starts at the base; one at or past the end
// stalls the hardware, which we refuse instead.
bool Apu::Memory::ring_transfer(u32 regs, u8* out, const u8* in, u32 len) {
    const u32 base = apu_.read_word(regs + kFifoBase), end = apu_.read_word(regs + kFifoEnd);
    u32 cur = apu_.read_word(regs + kFifoCurrent);
    if (end <= base) {
        if (problem_.empty()) problem_ = std::format("FIFO at +{:04x} has no room (base {:x}, end {:x})", regs, base, end);
        return false;
    }
    if (cur < base) cur = base;
    if (cur >= end) {
        if (problem_.empty()) problem_ = std::format("FIFO at +{:04x}: current {:x} is past the end {:x}", regs, cur, end);
        return false;
    }
    while (len) {
        const u32 chunk = std::min(len, end - cur);
        if (!copy(fifo_table_, fifo_max_, cur, out, in, chunk)) return false;
        if (out) out += chunk;
        if (in) in += chunk;
        cur += chunk;
        if (cur >= end) cur = base;
        len -= chunk;
    }
    apu_.write_word(regs + kFifoCurrent, cur);
    return true;
}

bool Apu::Memory::fifo_read(u32 fifo, u8* out, u32 len) {
    if (fifo > 1) return false;
    in_bytes_[fifo] += len;
    return ring_transfer(fifo_regs_ + kInputFifoRegs + fifo * kFifoStride, out, nullptr, len);
}

bool Apu::Memory::fifo_write(u32 fifo, const u8* in, u32 len) {
    if (fifo > 3) return false;
    out_bytes_[fifo] += len;
    if (fifo == 0)  // instrument: the level of what goes out (16-bit samples, as the output rings carry)
        for (u32 i = 0; i + 1 < len; i += 2) {
            i16 s;
            std::memcpy(&s, in + i, 2);
            out0_energy_ += static_cast<double>(s) * s;
            ++out0_samples_;
        }
    if (fifo == 0)  // instrument: stereo frames (L, R as 16-bit pairs) whose channels differ
        for (u32 i = 0; i + 3 < len; i += 4) {
            i16 l, r;
            std::memcpy(&l, in + i, 2);
            std::memcpy(&r, in + i + 2, 2);
            ++out0_frames_;
            if (l != r) ++out0_differ_;
        }
    trace::log(trace::Cat::Apu, "fifo write: regs +{:04x} out{} at {:x}, {} bytes", fifo_regs_, fifo, apu_.read_word(fifo_regs_ + fifo * kFifoStride + kFifoCurrent), len);
    return ring_transfer(fifo_regs_ + fifo * kFifoStride, nullptr, in, len);
}

}  // namespace mv

namespace mv {

// ── the front end's traps ───────────────────────────────────────────────────────────────────────────────────────

void Apu::front_end_trap(u32 method, u32 parameter) {
    // A report the host already has pending, or queued, is not raised twice.
    if (fe_trapped_ && regs_[kFeDecMethod / 4] == method && regs_[kFeDecParam / 4] == parameter) return;
    // An idle report while the front end is busy is dropped, never queued: the voice reports again on the next frame
    // it is still listed, and a queued report could reach the host after it took the voice off and freed it (one
    // title's handler then read through a null object and bug-checked). The reference emulator re-walks the lists
    // each frame and reports only what is listed then.
    if (fe_trapped_ && method == apu::SetupEngine::kSe2FeIdleVoice && se_.repeats_idle_reports()) return;
    if (std::ranges::find(trap_queue_, std::pair{method, parameter}) != trap_queue_.end()) return;
    if (fe_trapped_) {
        trap_queue_.emplace_back(method, parameter);
        return;
    }
    deliver_trap(method, parameter);
}

void Apu::deliver_trap(u32 method, u32 parameter) {
    fe_trapped_ = true;
    u32& fectl = regs_[kFectl / 4];
    fectl = (fectl & ~(kFectlMethodMode | kFectlTrapReason)) | kFectlTrapped | kFectlTrapRequested;
    regs_[kFeDecMethod / 4] = method;
    regs_[kFeDecParam / 4] = parameter;
    ++traps_;
    trace::log(trace::Cat::Apu, "fe: trap method {:04x} parameter {:x} (frame {})", method, parameter, frames_);
    raise(kIstsFrontEndTrap);
}

}  // namespace mv

namespace mv {

std::string Apu::pc_histogram_report(std::size_t top) const {
    std::string r;
    const auto one = [&](std::string_view which, const std::vector<u64>& hist, const apu::Processor& p, const FrameStats& st) {
        if (hist.empty()) return;
        u64 total = 0;
        std::vector<u32> order;
        for (u32 a = 0; a < hist.size(); ++a)
            if (hist[a]) {
                total += hist[a];
                order.push_back(a);
            }
        std::ranges::sort(order, std::greater{}, [&](u32 a) { return hist[a]; });
        r += std::format("{}: {} frames, {:.0f} cycles a frame ({:.0f} executed, the rest idle-wait skips)\n", which, st.frames_run,
                         st.frames_run ? static_cast<double>(st.total_cycles) / static_cast<double>(st.frames_run) : 0.0,
                         st.frames_run ? static_cast<double>(total) / static_cast<double>(st.frames_run) : 0.0);
        for (std::size_t i = 0; i < order.size() && i < top; ++i)
            r += std::format("  P:{:04x} {:5.2f}%  {}\n", order[i], 100.0 * static_cast<double>(hist[order[i]]) / static_cast<double>(total), p.core().disassemble_at(order[i]));
    };
    one("gp", gp_hist_, gp_, gp_stats_);
    one("ep", ep_hist_, ep_, ep_stats_);
    return r;
}

}  // namespace mv
