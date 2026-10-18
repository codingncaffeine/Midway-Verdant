#include "aci/aci.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "core/clock.h"
#include "debug/trace.h"

namespace mv {

Aci::Aci(Bus& bus, Scheduler& sched, std::function<void(bool)> irq)
    : PciDevice("aci", 0x10DE, 0x01B1, 0x040100, 0xB1), bus_(bus), sched_(sched), irq_(std::move(irq)) {
    define_bar(0, BarKind::Io, kMixerSize);
    define_bar(1, BarKind::Io, kBusMasterSize);
    define_bar(2, BarKind::Memory, kMmioSize);
    set_cfg8(0x3C, 6);  // interrupt line 6, pin A
    set_cfg8(0x3D, 1);
    for (u32 c = 0; c < kChannels; ++c) {
        ch_[c].done.name = "aci.buffer";
        ch_[c].done.fire = [this, c](Cycles) { buffer_done(static_cast<Channel>(c)); };
    }
}

void Aci::reset() {
    mixer_ = {};
    for (u32 c = 0; c < kChannels; ++c) channel_reset(static_cast<Channel>(c));
    glob_cnt_ = 0;
}

void Aci::on_bar_changed(int index, u32 old_base, u32 new_base) {
    switch (index) {
        case 0:
            if (old_base) bus_.unmap_pio(static_cast<u16>(old_base), kMixerSize);
            mixer_io_ = static_cast<u16>(new_base);
            if (new_base) bus_.map_pio(mixer_io_, kMixerSize, this);
            break;
        case 1:
            if (old_base) bus_.unmap_pio(static_cast<u16>(old_base), kBusMasterSize);
            bus_master_io_ = static_cast<u16>(new_base);
            if (new_base) bus_.map_pio(bus_master_io_, kBusMasterSize, this);
            break;
        case 2:
            if (mmio_at_) bus_.unmap_mmio(mmio_at_, kMmioSize);
            mmio_at_ = new_base;
            if (new_base) bus_.map_mmio(new_base, kMmioSize, this);
            trace::log(trace::Cat::Aci, "aci registers at {:08x}", new_base);
            break;
        default: break;
    }
}

// ── access paths ────────────────────────────────────────────────────────────────────────────────────────────────

u32 Aci::mmio_read(PhysAddr addr, unsigned size) {
    const u32 off = addr - mmio_at_;
    const u32 v = read(off, size);
    ++reads_;
    trace::log(trace::Cat::Aci, "aci read  +{:05x} size {} -> {:08x}", off, size, v);
    return v;
}

void Aci::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 off = addr - mmio_at_;
    write(off, size, value);
    ++writes_;
    trace::log(trace::Cat::Aci, "aci write +{:05x} size {} <- {:08x}", off, size, value);
}

u32 Aci::pio_read(u16 port, unsigned size) {
    const u32 off = (bus_master_io_ && port >= bus_master_io_ && port < bus_master_io_ + kBusMasterSize) ? kBusMasterBase + (port - bus_master_io_)
                                                                                                           : static_cast<u32>(port - mixer_io_);
    ++reads_;
    return read(off, size);
}

void Aci::pio_write(u16 port, unsigned size, u32 value) {
    const u32 off = (bus_master_io_ && port >= bus_master_io_ && port < bus_master_io_ + kBusMasterSize) ? kBusMasterBase + (port - bus_master_io_)
                                                                                                           : static_cast<u32>(port - mixer_io_);
    ++writes_;
    write(off, size, value);
}

u32 Aci::read(u32 offset, unsigned size) {
    if (offset < kMixerSize) {
        const u32 word = mixer_read16(offset & ~1u);
        if (size == 1) return (offset & 1u) ? (word >> 8) : (word & 0xFFu);
        if (size == 2) return word;
        return word | (mixer_read16((offset + 2) & 0xFEu) << 16);
    }
    u32 v = 0;
    for (unsigned i = 0; i < size; ++i) v |= read_byte(offset + i) << (8u * i);
    return v;
}

void Aci::write(u32 offset, unsigned size, u32 value) {
    if (offset < kMixerSize) {
        // Codec registers are 16 bits wide; a byte write lands in its half.
        const u32 reg = offset & ~1u;
        if (size == 1) {
            const u16 old = static_cast<u16>(mixer_read16(reg));
            mixer_write16(reg, (offset & 1u) ? static_cast<u16>((old & 0x00FFu) | (value << 8)) : static_cast<u16>((old & 0xFF00u) | (value & 0xFFu)));
        } else {
            mixer_write16(reg, static_cast<u16>(value));
            if (size == 4) mixer_write16((reg + 2) & 0xFEu, static_cast<u16>(value >> 16));
        }
        return;
    }
    for (unsigned i = 0; i < size; ++i) write_byte(offset + i, static_cast<u8>(value >> (8u * i)));
}

// ── the codec's mixer registers ─────────────────────────────────────────────────────────────────────────────────

u32 Aci::mixer_read16(u32 reg) {
    if (reg == kMixPowerdown) return (mixer_[reg / 2] & 0xFF00u) | kMixPowerdownReady;  // every section ready at once
    return mixer_[reg / 2];
}

void Aci::mixer_write16(u32 reg, u16 value) {
    if (reg == 0x00) {  // a write to the reset register resets the codec
        mixer_.fill(0);
        return;
    }
    mixer_[reg / 2] = value;
}

// ── the bus master ──────────────────────────────────────────────────────────────────────────────────────────────

u32 Aci::glob_sta() {
    u32 v = kGlobStaPcr;
    for (u32 c = 0; c < kChannels; ++c)
        if (channel_interrupting(ch_[c])) v |= kGlobStaChannelInt[c];
    return v;
}

u32 Aci::read_byte(u32 offset) {
    const u32 rel = offset - kBusMasterBase;
    for (u32 c = 0; c < kChannels; ++c)
        if (rel >= kChannelOffset[c] && rel < kChannelOffset[c] + 0x10) return channel_read_byte(static_cast<Channel>(c), rel - kChannelOffset[c]);
    if (rel >= kGlobCnt && rel < kGlobCnt + 4) return (glob_cnt_ >> (8u * (rel - kGlobCnt))) & 0xFFu;
    if (rel >= kGlobSta && rel < kGlobSta + 4) return (glob_sta() >> (8u * (rel - kGlobSta))) & 0xFFu;
    if (rel == kCas) return 0;  // the codec access semaphore: never busy at this level of modelling
    return 0;
}

void Aci::write_byte(u32 offset, u8 value) {
    const u32 rel = offset - kBusMasterBase;
    for (u32 c = 0; c < kChannels; ++c) {
        if (rel >= kChannelOffset[c] && rel < kChannelOffset[c] + 0x10) {
            channel_write_byte(static_cast<Channel>(c), rel - kChannelOffset[c], value);
            return;
        }
    }
    if (rel >= kGlobCnt && rel < kGlobCnt + 4) {
        const u32 shift = 8u * (rel - kGlobCnt);
        glob_cnt_ = (glob_cnt_ & ~(0xFFu << shift)) | (static_cast<u32>(value) << shift);
        glob_cnt_ &= ~kGlobCntWarmReset;  // a warm reset finishes at once
        return;
    }
}

u16 Aci::picb(const ChannelState& s) const {
    if ((s.sr & kSrDch) || !s.buffer_samples) return 0;
    const u64 played = ticks_from_cycles(sched_.now() - s.buffer_started, kSamplesPerSecond);
    return static_cast<u16>(s.buffer_samples - std::min<u64>(played, s.buffer_samples));
}

u32 Aci::channel_read_byte(Channel c, u32 rel) {
    const ChannelState& s = ch_[c];
    switch (rel) {
        case kBdbar + 0: case kBdbar + 1: case kBdbar + 2: case kBdbar + 3: return (s.bdbar >> (8u * rel)) & 0xFFu;
        case kCiv: return s.civ;
        case kLvi: return s.lvi;
        case kSr: return s.sr & 0xFFu;
        case kSr + 1: return s.sr >> 8;
        case kPicb: return picb(s) & 0xFFu;
        case kPicb + 1: return picb(s) >> 8;
        case kPiv: return s.piv;
        case kCr: return s.cr;
        default: return 0;
    }
}

void Aci::channel_reset(Channel c) {
    ChannelState& s = ch_[c];
    sched_.cancel(s.done);
    s.civ = s.lvi = s.piv = 0;
    s.cr = 0;
    s.sr = kSrDch;
    s.buffer_samples = 0;
    s.buffer_address = 0;
    update_irq();
}

void Aci::channel_write_byte(Channel c, u32 rel, u8 value) {
    ChannelState& s = ch_[c];
    switch (rel) {
        case kBdbar + 0: case kBdbar + 1: case kBdbar + 2: case kBdbar + 3: {
            const u32 shift = 8u * rel;
            s.bdbar = ((s.bdbar & ~(0xFFu << shift)) | (static_cast<u32>(value) << shift)) & ~7u;  // 8-byte aligned
            return;
        }
        case kLvi:
            s.lvi = value & 0x1Fu;
            // A running channel halted at its last valid buffer continues from the prefetched index when software
            // writes the index again, even with the same value: the kernel re-arms its single buffer that way, and
            // the empty descriptors in between complete at once.
            if ((s.cr & kCrRpbm) && (s.sr & kSrDch) && (s.sr & kSrCelv)) {
                s.sr &= static_cast<u16>(~(kSrDch | kSrCelv));
                s.civ = s.piv;
                start_buffer(c);
            }
            return;
        case kSr:
            s.sr &= static_cast<u16>(~(value & (kSrLvbci | kSrBcis | kSrFifoe)));  // write one to clear
            update_irq();
            return;
        case kCr: {
            if (value & kCrRr) {
                channel_reset(c);
                trace::log(trace::Cat::Aci, "aci channel {} reset", static_cast<u32>(c));
                return;
            }
            const u8 old = s.cr;
            s.cr = value & (kCrRpbm | kCrLvbie | kCrFeie | kCrIoce);
            if ((s.cr & kCrRpbm) && !(old & kCrRpbm)) {
                // Run: fetch the current descriptor and start consuming it.
                if (s.sr & kSrDch) {
                    s.sr &= static_cast<u16>(~kSrDch);
                    start_buffer(c);
                }
            } else if (!(s.cr & kCrRpbm) && (old & kCrRpbm)) {
                // Pause: the buffer in flight is dropped and restarts on the next run [verify: hardware resumes mid-buffer].
                sched_.cancel(s.done);
                s.sr |= kSrDch;
            }
            update_irq();
            return;
        }
        default: return;
    }
}

void Aci::start_buffer(Channel c) {
    ChannelState& s = ch_[c];
    u32 desc[2]{};
    if (!bus_.read_ram(s.bdbar + s.civ * 8u, desc, 8)) {
        trace::log(trace::Cat::Aci, "aci channel {}: descriptor {} at {:08x} is outside RAM", static_cast<u32>(c), s.civ, s.bdbar + s.civ * 8u);
        s.sr |= kSrDch;
        return;
    }
    s.buffer_address = desc[0] & ~1u;
    s.buffer_samples = desc[1] & kDescLength;
    s.buffer_ioc = (desc[1] & kDescIoc) != 0;
    s.piv = static_cast<u8>((s.civ + 1) & 0x1Fu);
    s.buffer_started = sched_.now();
    // The buffer plays out at the sample clock; an empty buffer completes on the next cycle.
    const Cycles length = s.buffer_samples ? cycles_from_ticks_ceil(s.buffer_samples, kSamplesPerSecond) : 1;
    sched_.schedule(s.done, s.buffer_started + length);
    trace::log(trace::Cat::Aci, "aci channel {}: buffer {} at {:08x}, {} samples{}", static_cast<u32>(c), s.civ, s.buffer_address, s.buffer_samples,
               s.buffer_ioc ? ", interrupt on completion" : "");
}

void Aci::buffer_done(Channel c) {
    ChannelState& s = ch_[c];
    if (s.buffer_samples && sink_) {
        std::vector<u8> data(static_cast<std::size_t>(s.buffer_samples) * 2);
        if (bus_.read_ram(s.buffer_address, data.data(), data.size())) sink_(c, data.data(), static_cast<u32>(data.size()));
    }
    s.bytes_played += static_cast<u64>(s.buffer_samples) * 2;
    ++s.buffers_completed;
    if (s.buffer_ioc) s.sr |= kSrBcis;
    // Finishing the last valid buffer is reported (LVBCI), but the MCPX's controller does not halt there as the
    // AMD-768 and Intel controllers do (datasheet 24467, ACx6 BMCH): it carries on through the list, the empty
    // descriptors completing at once, back round to the first. The kernel depends on it: it programs one descriptor
    // over the APU's 8 KiB output ring, never advances the last valid index while the boot sound plays, and only
    // acknowledges the status [inferred from the kernel; verify on hardware].
    if (s.civ == s.lvi) s.sr |= kSrLvbci;
    s.civ = s.piv;
    start_buffer(c);
    update_irq();
}

bool Aci::channel_interrupting(const ChannelState& s) const {
    // The S/PDIF channel mirrors the PCM output and raises nothing: the kernel runs it with interrupts enabled
    // and never acknowledges it, so a line from it would never drop [verify on hardware].
    if (&s == &ch_[Spdif]) return false;
    return ((s.sr & kSrBcis) && (s.cr & kCrIoce)) || ((s.sr & kSrLvbci) && (s.cr & kCrLvbie)) || ((s.sr & kSrFifoe) && (s.cr & kCrFeie));
}

void Aci::update_irq() {
    bool level = false;
    for (const auto& s : ch_) level = level || channel_interrupting(s);
    if (level != irq_level_) {
        irq_level_ = level;
        if (irq_) irq_(level);
    }
}

}  // namespace mv
