#include "apu/setup_engine.h"

#include <cmath>

#include "debug/trace.h"

namespace mv::apu {

PhysAddr SetupEngine::voice_address(u32 handle) const { return host_.global_reg(kRegVoiceBase) + handle * kVoiceSize; }

u32 SetupEngine::field(u32 handle, u32 offset) const {
    u32 v = 0;
    bus_.read_ram(voice_address(handle) + offset, &v, 4);
    return v;
}

void SetupEngine::set_field(u32 handle, u32 offset, u32 value) { bus_.write_ram(voice_address(handle) + offset, &value, 4); }

bool SetupEngine::method(u32 offset, u32 value) {
    switch (offset) {
        case kMethodSetCurrentVoice: current_ = value & kHandleMask; return true;
        case kMethodVoiceLock: return true;  // the kernel locks a voice while it writes its fields; nothing races here
        case kMethodSetNotifyDma: case kMethodSetSslDma: return true;  // the notifier and SSL tables are FENADDR / VPSSLADDR
        case kMethodSetCurrentSsl: current_ssl_page_ = value & kSslBasePageMask; return true;
        case kMethodSetAntecedentVoice: antecedent_ = value; return true;
        case kMethodVoiceOn: {
            const u32 handle = value & kHandleMask;
            if (handle >= kMaxVoices) return true;
            link_on(handle);
            // The envelope start states ride in the top byte: filter envelope bits 27:24, amplitude 31:28.
            u32 state = (field(handle, kFieldState) & 0x00FFFFFFu) | (value & 0xFF000000u);
            state = (state | kStateActive | kStateNew) & ~kStatePaused;
            set_field(handle, kFieldState, state);
            book_[handle].off_pending = false;
            book_[handle].fraction = 0;
            book_[handle].ssl_index = 0;
            book_[handle].ssl_segment = 0;
            book_[handle].starved_reported = false;
            ++started_;
            return true;
        }
        case kMethodVoiceOff: case kMethodVoiceRelease: {
            // Off retires the voice at the next frame; release should run the envelope's release first [verify].
            const u32 handle = value & kHandleMask;
            if (handle < kMaxVoices) book_[handle].off_pending = true;
            return true;
        }
        case kMethodVoicePause: {
            const u32 handle = value & kHandleMask;
            if (handle >= kMaxVoices) return true;
            u32 state = field(handle, kFieldState);
            state = (value & kVoicePauseAction) ? (state | kStatePaused) : (state & ~kStatePaused);
            set_field(handle, kFieldState, state);
            return true;
        }
        default: break;
    }
    if (offset >= kMethodSslSegmentFirst && offset <= kMethodSslSegmentLast) {
        // One word of a segment of the current SSL: its RAM offset, then length (15:0), container (17:16),
        // samples per block - 1 (22:18) and stereo (23), into the segment table at VPSSLADDR.
        const PhysAddr at = host_.global_reg(kRegSslTable) + current_ssl_page_ * 8 + (offset - kMethodSslSegmentFirst);
        bus_.write_ram(at, &value, 4);
        return true;
    }
    if (current_ >= kMaxVoices) return false;
    if (offset == kMethodVoiceSslA || offset == kMethodVoiceSslB) {
        const u32 n = offset == kMethodVoiceSslA ? 0 : 1;
        book_[current_].ssl_base[n] = value >> kSslBaseShift;
        book_[current_].ssl_count[n] = value & kSslCountMask;
        return true;
    }
    if (offset >= kMethodCfgFirst && offset <= kMethodCfgLast) {
        // The configuration and target fields sit in the structure at the method's own offset within the block;
        // the pitch shares its word with the list link and keeps the link.
        const u32 f = offset - kMethodCfgFirst;
        if (f == kFieldPitchLink) set_field(current_, f, (value & 0xFFFF0000u) | (field(current_, f) & kHandleMask));
        else set_field(current_, f, value);
        return true;
    }
    switch (offset) {
        case kMethodBufBase: set_field(current_, kFieldPslStart, (field(current_, kFieldPslStart) & ~kOffsetMask) | (value & kOffsetMask)); return true;
        case kMethodBufLbo: set_field(current_, kFieldPshSample, (field(current_, kFieldPshSample) & ~kOffsetMask) | (value & kOffsetMask)); return true;
        case kMethodBufCbo:
            set_field(current_, kFieldOffset, (field(current_, kFieldOffset) & ~kOffsetMask) | (value & kOffsetMask));
            book_[current_].fraction = 0;
            return true;
        case kMethodBufEbo: set_field(current_, kFieldNext, (field(current_, kFieldNext) & ~kOffsetMask) | (value & kOffsetMask)); return true;
        default: return false;
    }
}

// Insert the voice where the antecedent says: at the top of a list, or after another voice.
void SetupEngine::link_on(u32 handle) {
    const u32 list = (antecedent_ >> kAntecedentListShift) & 3u;
    const u32 after = antecedent_ & kAntecedentHandle;
    auto set_next = [&](u32 h, u32 next) { set_field(h, kFieldPitchLink, (field(h, kFieldPitchLink) & 0xFFFF0000u) | (next & kHandleMask)); };
    if (list == 0) {
        if (after >= kMaxVoices) return;  // inherit from no voice: nowhere to link [verify]
        set_next(handle, field(after, kFieldPitchLink) & kHandleMask);
        set_next(after, handle);
    } else {
        const u32 top_reg = kRegListTop[list - 1];
        set_next(handle, host_.global_reg(top_reg) & kHandleMask);
        host_.set_global_reg(top_reg, handle);
    }
}

void SetupEngine::frame() {
    for (u32 list = 0; list < kRegListTop.size(); ++list) walk(list);
}

void SetupEngine::walk(u32 list) {
    const u32 top_reg = kRegListTop[list];
    u32 prev = kNoVoice;
    u32 handle = host_.global_reg(top_reg) & kHandleMask;
    for (u32 guard = 0; handle < kMaxVoices && guard < kMaxVoices; ++guard) {
        const u32 link = field(handle, kFieldPitchLink);
        const u32 next = link & kHandleMask;
        const u32 state = field(handle, kFieldState);
        bool retire = book_[handle].off_pending;
        if (!retire && (state & kStateActive) && !(state & kStatePaused)) {
            // Advance through the buffer: 32 samples per frame at unity pitch, scaled by 2^(pitch / 4096).
            const i32 pitch = static_cast<i16>(link >> 16);
            const double step = kSamplesPerFrame * std::exp2(pitch / 4096.0);
            if (field(handle, kFieldFmt) & kFmtStream) {
                if (!advance_stream(handle, step)) retire = true;
            } else {
                const u32 ebo = field(handle, kFieldNext) & kOffsetMask, lbo = field(handle, kFieldPshSample) & kOffsetMask;
                double pos = (field(handle, kFieldOffset) & kOffsetMask) + book_[handle].fraction + step;
                if (pos > ebo) {
                    const u32 fmt = field(handle, kFieldFmt);
                    if ((fmt & kFmtLoop) && ebo >= lbo) {
                        const double length = static_cast<double>(ebo) + 1 - lbo;
                        pos = lbo + std::fmod(pos - (ebo + 1), length);
                    } else {
                        pos = ebo;
                        retire = true;
                    }
            }
            const u32 whole = static_cast<u32>(pos);
            book_[handle].fraction = pos - whole;
            set_field(handle, kFieldOffset, (field(handle, kFieldOffset) & ~kOffsetMask) | (whole & kOffsetMask));
            }
        }
        if (retire) {
            if (prev == kNoVoice) host_.set_global_reg(top_reg, next);
            else set_field(prev, kFieldPitchLink, (field(prev, kFieldPitchLink) & 0xFFFF0000u) | next);
            set_field(handle, kFieldState, state & ~(kStateActive | kStateNew));
            book_[handle].off_pending = false;
            ++retired_;
            if (host_.global_reg(kRegTrapForce1) & kTrapForceIdleVoice) host_.front_end_trap(kSe2FeIdleVoice, handle);
        } else {
            prev = handle;
        }
        handle = next;
    }
}


// A stream voice plays segment after segment of its current SSL; a finished SSL is reported and the voice moves to
// the other one (A, B, A, ...), whose segments the driver refills meanwhile. An SSL with no segments starves the
// voice: a persistent voice waits (reporting once), any other ends. The position carries across segment ends.
bool SetupEngine::advance_stream(u32 handle, double step) {
    Bookkeeping& b = book_[handle];
    const u32 fmt = field(handle, kFieldFmt);
    const PhysAddr table = host_.global_reg(kRegSslTable);
    double pos = (field(handle, kFieldOffset) & kOffsetMask) + b.fraction + step;
    for (u32 guard = 0; guard < 4 * kSslCountMask; ++guard) {
        if (b.ssl_count[b.ssl_index] == 0) {
            b.fraction = 0;
            b.ssl_segment = 0;
            set_field(handle, kFieldOffset, field(handle, kFieldOffset) & ~kOffsetMask);
            if (!(fmt & kFmtPersist)) return false;
            if (!b.starved_reported) {
                notify(handle, b.ssl_index);
                b.starved_reported = true;
            }
            return true;
        }
        b.starved_reported = false;
        u32 length_word = 0;
        bus_.read_ram(table + (b.ssl_base[b.ssl_index] + b.ssl_segment) * 8 + 4, &length_word, 4);
        const double length = length_word & 0xFFFFu;  // samples in the segment
        if (pos < length) {
            const u32 whole = static_cast<u32>(pos);
            b.fraction = pos - whole;
            set_field(handle, kFieldOffset, (field(handle, kFieldOffset) & ~kOffsetMask) | (whole & kOffsetMask));
            return true;
        }
        pos -= length;
        if (++b.ssl_segment >= b.ssl_count[b.ssl_index]) {
            notify(handle, b.ssl_index);
            b.ssl_segment = 0;
            b.ssl_index ^= 1;
        }
    }
    return true;
}

void SetupEngine::notify(u32 handle, u32 ssl) {
    const PhysAddr at = host_.global_reg(kRegNotifierBase) + 16 * (kNotifierBase + handle * kNotifiersPerVoice + ssl);
    const u8 tail[2] = {1, kNotifierDone};  // bytes 14 and 15: the status last
    bus_.write_ram(at + 14, tail, 2);
    trace::log(trace::Cat::Apu, "vp: voice {:x} SSL {} {} (notifier at {:08x})", handle, ssl == 0 ? "A" : "B",
               book_[handle].ssl_count[ssl] ? "done" : "starved", at);
    ++notifications_;
    host_.raise_interrupt(kIstsNotify);
}
}  // namespace mv::apu
