#include "apu/apu.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "core/clock.h"
#include "debug/trace.h"

namespace mv {

namespace {
// GP: 4096 words of X plus the 1024-word MIXBUF at X:$1400, 2048 of Y, 4096 of P. EP: 3072 X, 256 Y, 4096 P.
constexpr dsp::Core::Config kGpConfig{0x1800, 0x800, 0x1000};
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
    sched_.schedule(frame_, cycles_from_ticks(1, clk::kApuFramesPerSecond));
}

void Apu::on_bar_changed(int index, u32 old_base, u32 new_base) {
    (void)old_base;
    if (index != 0) return;
    if (mapped_at_) bus_.unmap_mmio(mapped_at_, kBarSize);
    mapped_at_ = new_base;
    if (new_base) bus_.map_mmio(new_base, kBarSize, this);
    trace::log(trace::Cat::Apu, "apu registers at {:08x}", new_base);
}

// ── the register window ─────────────────────────────────────────────────────────────────────────────────────────

u32 Apu::reg(u32 offset) { return read_word(offset & ~3u); }

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
    const u32 word = read_word(off & ~3u);
    const unsigned shift = 8u * (off & 3u);
    const u32 v = size == 4 ? word : (size == 2 ? ((word >> shift) & 0xFFFFu) : ((word >> shift) & 0xFFu));
    ++reads_;
    trace::log(trace::Cat::Apu, "apu read  +{:05x} size {} -> {:08x}", off, size, v);
    return v;
}

void Apu::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 off = addr - mapped_at_;
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
        if (p.running()) trace::log(trace::Cat::Apu, "{}: reset released, program loaded from scratch, P:0 = {:06x}", which, p.core().read_p(0));
        else trace::log(trace::Cat::Apu, "{}: reset released but the bootstrap failed: {}", which, p.last_problem());
    } else if (!release && p.running()) {
        p.hold_reset();
        trace::log(trace::Cat::Apu, "{}: reset held ({:x})", which, value);
    }
}

void Apu::run_frame(apu::Processor& p, FrameStats& stats, std::string_view which) {
    if (!p.running()) return;
    const u64 used = p.run_frame(kDspHz / clk::kApuFramesPerSecond);
    stats.last_cycles = used;
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
    ++frames_;
    // The setup engine walks the voice lists first; its idle reports trap to the host once the front end is free.
    se_.frame();
    if (!fe_trapped_ && !trap_queue_.empty()) {
        const auto [method, parameter] = trap_queue_.front();
        trap_queue_.pop_front();
        deliver_trap(method, parameter);
    }
    // The voice processor is not modelled yet: the mix buffer the GP reads holds silence.
    auto& x = gp_.core().x_memory();
    if (x.size() >= kMixBufWord + kMixBufWords) std::fill(x.begin() + kMixBufWord, x.begin() + kMixBufWord + kMixBufWords, 0u);
    run_frame(gp_, gp_stats_, "gp");
    run_frame(ep_, ep_stats_, "ep");
    sched_.schedule(frame_, cycles_from_ticks(frames_ + 1, clk::kApuFramesPerSecond));
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
    return ring_transfer(fifo_regs_ + kInputFifoRegs + fifo * kFifoStride, out, nullptr, len);
}

bool Apu::Memory::fifo_write(u32 fifo, const u8* in, u32 len) {
    if (fifo > 3) return false;
    return ring_transfer(fifo_regs_ + fifo * kFifoStride, nullptr, in, len);
}

}  // namespace mv

namespace mv {

// ── the front end's traps ───────────────────────────────────────────────────────────────────────────────────────

void Apu::front_end_trap(u32 method, u32 parameter) {
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
