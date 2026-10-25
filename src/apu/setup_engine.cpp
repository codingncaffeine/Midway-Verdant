#include "apu/setup_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>

#include "debug/trace.h"

namespace mv::apu {

namespace {

// IMA ADPCM tables (the Xbox format is IMA ADPCM with 64 nibbles per channel and block).
constexpr std::array<int, 16> kAdpcmIndexStep{-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8};
constexpr std::array<int, 89> kAdpcmStep{7,     8,     9,     10,    11,    12,    13,    14,    16,    17,    19,    21,    23,    25,    28,
                                         31,    34,    37,    41,    45,    50,    55,    60,    66,    73,    80,    88,    97,    107,   118,
                                         130,   143,   157,   173,   190,   209,   230,   253,   279,   307,   337,   371,   408,   449,   494,
                                         544,   598,   658,   724,   796,   876,   963,   1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,
                                         2272,  2499,  2749,  3024,  3327,  3660,  4026,  4428,  4871,  5358,  5894,  6484,  7132,  7845,  8630,
                                         9493,  10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767};

// One channel of a block: the header (predictor, step index) then 4-byte words of 8 nibbles each, low nibble first;
// stereo interleaves the channels a word at a time. `bytes` is the block (36 mono, 72 stereo).
void decode_adpcm_block(const u8* bytes, u32 channels, std::array<i16, SetupEngine::kAdpcmSamplesPerBlock * 2>& out) {
    for (u32 c = 0; c < channels; ++c) {
        const u8* h = bytes + c * 4;
        int predictor = static_cast<i16>(h[0] | (h[1] << 8));
        int index = std::clamp<int>(h[2], 0, 88);
        out[c] = static_cast<i16>(predictor);
        u32 n = 1;
        for (u32 word = 0; word < 8; ++word) {
            const u8* w = bytes + channels * 4 + (word * channels + c) * 4;
            for (u32 b = 0; b < 4; ++b)
                for (u32 half = 0; half < 2; ++half) {
                    const int nibble = half ? (w[b] >> 4) : (w[b] & 0xF);
                    const int step = kAdpcmStep[static_cast<std::size_t>(index)];
                    int diff = step >> 3;
                    if (nibble & 1) diff += step >> 2;
                    if (nibble & 2) diff += step >> 1;
                    if (nibble & 4) diff += step;
                    predictor = std::clamp(predictor + ((nibble & 8) ? -diff : diff), -32768, 32767);
                    index = std::clamp(index + kAdpcmIndexStep[static_cast<std::size_t>(nibble)], 0, 88);
                    out[n * 2 + c] = static_cast<i16>(predictor);
                    ++n;
                }
        }
    }
    if (channels == 1)
        for (u32 n = 0; n < SetupEngine::kAdpcmSamplesPerBlock; ++n) out[n * 2 + 1] = out[n * 2];  // mono: both channels
}

// A 12-bit volume is an attenuation in 1/64 dB; all ones is silence.
float attenuation(u32 vol) {
    vol &= 0xFFFu;
    return vol == 0xFFFu ? 0.0f : std::pow(10.0f, static_cast<float>(vol) / (64.0f * -20.0f));
}

}  // namespace

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
        case kMethodSetCurrentInbufSge: inbuf_sge_ = value; return true;
        case kMethodSetCurrentInbufSgeOffset: {
            const u32 page = value & 0xFFFFF000u;
            bus_.write_ram(static_cast<PhysAddr>(host_.global_reg(kRegSgeTable) + inbuf_sge_ * 8u), &page, 4);
            return true;
        }
        case kMethodSetAntecedentVoice: antecedent_ = value; return true;
        case kMethodHrtfHeadroom: hrtf_headroom_ = static_cast<u8>(value & 7u); return true;
        case kMethodHrtfSubmixes:
            for (u32 i = 0; i < 4; ++i) hrtf_submix_[i] = static_cast<u8>((value >> (8 * i)) & 0x1Fu);
            return true;
        case kMethodVoiceOn: {
            const u32 handle = value & kHandleMask;
            if (handle >= kMaxVoices) return true;
            link_on(handle);
            // The envelope start phases ride in the top byte: filter envelope bits 27:24, amplitude 31:28. Their
            // counters start from the phase's time; playback starts at the buffer's beginning.
            u32 state = (field(handle, kFieldState) & 0x00FFFFFFu) | (value & 0xFF000000u);
            state = (state | kStateActive | kStateNew) & ~kStatePaused;
            set_field(handle, kFieldState, state);
            auto start_count = [&](u32 phase, u32 delay_word, u32 hold_word) -> u32 {
                if (phase == kDelay) return ((delay_word >> 12) & 0xFFFu) * 16;
                if (phase == kHold) return ((hold_word >> 12) & 0xFFFu) * 16;
                return 0;
            };
            const u32 ea = start_count(value >> 28, field(handle, kFieldEnv0), field(handle, kFieldEnvA));
            const u32 ef = start_count((value >> 24) & 0xFu, field(handle, kFieldEnv1), field(handle, kFieldEnvF));
            set_field(handle, kFieldEcnt, (ef << 16) | (ea & 0xFFFFu));
            set_field(handle, kFieldOffset, field(handle, kFieldOffset) & ~kOffsetMask);  // CBO = 0 [verify]
            Bookkeeping& b = book_[handle];
            b.off_pending = false;
            b.fraction = 0;
            b.ssl_index = 0;
            b.ssl_segment = 0;
            b.starved_reported = false;
            b.adpcm_key = ~0ull;
            b.svf = {};
            ++started_;
            if (std::none_of(handles_started_.begin(), handles_started_.end(), [handle](const Started& s) { return s.handle == handle; }))
                handles_started_.push_back({handle, trace::now()});
            if (trace::on(trace::Cat::Apu) && !(field(handle, kFieldFmt) & kFmtStream)) {
                // What the voice will play: its buffer's level and peak, over the whole buffer up to a million samples.
                const u32 fmt = field(handle, kFieldFmt);
                Source src = source_for(fmt);
                src.base = field(handle, kFieldPslStart) & kOffsetMask;
                const u32 ebo = field(handle, kFieldNext) & kOffsetMask, n = std::min(ebo + 1, 1u << 20);
                double sum = 0, peak = 0;
                u32 crossings = 0, differ = 0;
                float last = 0.0f;
                for (u32 i = 0; i < n; ++i) {
                    float s[2];
                    read_sample(handle, src, i, s);
                    if (i > 0 && (s[0] < 0.0f) != (last < 0.0f)) ++crossings;
                    if (s[0] != s[1]) ++differ;
                    last = s[0];
                    sum += static_cast<double>(s[0]) * static_cast<double>(s[0]);
                    peak = std::max(peak, static_cast<double>(std::fabs(s[0])));
                }
                trace::log(trace::Cat::Apu, "vp: voice {:x} on: buffer +{:x} ebo {:x} lbo {:x} fmt {:08x}: rms {:.1f} dB peak {:.1f} dB, {} zero crossings, L != R in {} of {} samples", handle,
                           src.base, ebo, field(handle, kFieldPshSample) & kOffsetMask, fmt, 10 * std::log10(sum / n + 1e-12), 20 * std::log10(peak + 1e-9), crossings, differ, n);
                trace::log(trace::Cat::Apu, "vp: voice {:x} on: vbin {:08x} vol {:08x} {:08x} {:08x} misc {:08x} env0 {:08x} enva {:08x} pitch {:08x}", handle, field(handle, kFieldVbin),
                           field(handle, kFieldVolA), field(handle, kFieldVolB), field(handle, kFieldVolC), field(handle, kFieldMisc), field(handle, 0x08), field(handle, 0x0C),
                           field(handle, kFieldPitchLink));
                // Instrument: MV_DEBUG_VOICE_DUMP=HEX writes that voice's buffer as it stands (dump_voice), here and twice a
                // second while it plays (frame()), to measure what the engine reads and whether the title rewrites it.
                if (dump_handle() == handle) dump_voice(handle);
                b.adpcm_key = ~0ull;  // the reads above filled the ADPCM block cache; playback starts from a clean one
            }
            return true;
        }
        case kMethodVoiceOff: {
            const u32 handle = value & kHandleMask;
            if (handle < kMaxVoices) book_[handle].off_pending = true;
            return true;
        }
        case kMethodVoiceRelease: {
            // Both envelopes enter their release: the amplitude one at the TAR_LFO_ENV rate, the filter one at the
            // CFG_MISC rate; the voice retires when the amplitude release ends (never for a rate of zero).
            const u32 handle = value & kHandleMask;
            if (handle >= kMaxVoices) return true;
            const u32 ea = (field(handle, kFieldLfoEnv) & 0xFFFu) * 16, ef = (field(handle, kFieldMisc) & 0xFFFu) * 16;
            trace::log(trace::Cat::Apu, "vp: voice {:x} release: amplitude rate {:x}, filter rate {:x}, envelope phases {:08x}, level {}", handle, ea / 16, ef / 16,
                       field(handle, kFieldState), field(handle, kFieldOffset) >> 24);
            set_field(handle, kFieldEcnt, (ef << 16) | (ea & 0xFFFFu));
            // A zero-rate release of a voice whose amplitude envelope never ran (phase off: a buffer without an
            // envelope) ends it at once, as both reference emulators do and as one title's recording shows (its
            // looping ambience stops dead at the release); a running envelope's zero-rate release holds (below).
            // [INFERRED from the two recordings; hw/apu.md]
            const u32 state = field(handle, kFieldState);
            const u32 ea_phase = ((state >> kStateEaShift) & 0xFu) == kOff && ea == 0 ? kForceRelease : kRelease;
            set_field(handle, kFieldState, (state & 0x00FFFFFFu) | (ea_phase << kStateEaShift) | (kRelease << kStateEfShift));
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
    if (offset >= kMethodSubmixHeadroomFirst && offset <= kMethodSubmixHeadroomLast) {
        submix_headroom_[(offset - kMethodSubmixHeadroomFirst) / 4] = static_cast<u8>(value & 7u);
        return true;
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
        case kMethodBufEbo: {
            set_field(current_, kFieldNext, (field(current_, kFieldNext) & ~kOffsetMask) | (value & kOffsetMask));
            // Programming a playing one-shot buffer's end starts it again from its beginning. The kernel replays a
            // sound this way, without VOICE_ON, on a voice it still counts as playing: the boot sound's closing chime
            // and the echoes of its jingle, both in the recording of a real console. [INFERRED]
            if (!(field(current_, kFieldFmt) & (kFmtLoop | kFmtStream))) {
                Bookkeeping& b = book_[current_];
                b.fraction = 0;
                set_field(current_, kFieldOffset, field(current_, kFieldOffset) & ~kOffsetMask);
            }
            return true;
        }
        default: return false;
    }
}

bool SetupEngine::on_a_list(u32 handle) const {
    for (u32 list = 0; list < kRegListTop.size(); ++list)
        for (u32 h = host_.global_reg(kRegListTop[list]) & kHandleMask, guard = 0; h < kMaxVoices && guard < kMaxVoices;
             h = field(h, kFieldPitchLink) & kHandleMask, ++guard)
            if (h == handle) return true;
    return false;
}

// Insert the voice where the antecedent says: at the top of a list, or after another voice. A voice still on a list
// (one the host starts again while it plays) stays where it is: linking it twice would loop the list.
void SetupEngine::link_on(u32 handle) {
    if (on_a_list(handle)) return;
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
    for (auto& bin : mix_) bin.fill(0.0f);
    direct_on_ = direct_mix();
    for (auto& s : direct_) s = {0.0f, 0.0f};
    direct_skip_ = direct_on_ ? multipass_bins() : 0u;
    for (u32 list = 0; list < kRegListTop.size(); ++list) walk(list);
    for (u32 b = 0; b < kBins; ++b)
        for (float s : mix_[b]) bin_energy_[b] += static_cast<double>(s) * static_cast<double>(s);
    // Twice a second, under the APU trace: every voice on a list with its position, envelope level and V0/V1 volumes.
    if (++frames_ % 750 == 0 && trace::on(trace::Cat::Apu)) {
        std::string line;
        for (u32 list = 0; list < kRegListTop.size(); ++list)
            for (u32 h = host_.global_reg(kRegListTop[list]) & kHandleMask, guard = 0; h < kMaxVoices && guard < kMaxVoices; h = field(h, kFieldPitchLink) & kHandleMask, ++guard)
                line += std::format(" {:x}(cbo {} ea {} v0 {:x} v1 {:x} fmode {} fca {:08x} fcb {:08x}{})", h, field(h, kFieldOffset) & kOffsetMask, field(h, kFieldOffset) >> 24,
                                    (field(h, kFieldVolA) >> 4) & 0xFFFu, field(h, kFieldVolA) >> 20, (field(h, kFieldMisc) >> 16) & 3u, field(h, kFieldFcA), field(h, kFieldFcB),
                                    (field(h, kFieldState) & kStatePaused) ? " paused" : "");
        trace::log(trace::Cat::Apu, "vp: frame {} voices:{}", frames_, line.empty() ? " none" : line);
        if (const u32 d = dump_handle(); d < kMaxVoices && (field(d, kFieldState) & kStateActive)) dump_voice(d);
        // And the RMS level of each bin that carried sound over the half second, in dB of full scale.
        std::string bins;
        for (u32 b = 0; b < kBins; ++b) {
            const double ms = bin_energy_[b] / (750.0 * kSamplesPerFrame);
            if (ms > 1e-12) bins += std::format(" {}:{:.1f}", b, 10 * std::log10(ms));
        }
        bin_energy_.fill(0.0);
        trace::log(trace::Cat::Apu, "vp: frame {} bin rms dB:{}", frames_, bins.empty() ? " silent" : bins);
        if (stereo_samples_) trace::log(trace::Cat::Apu, "vp: frame {} stereo voices rendered {} samples, {} with L != R", frames_, stereo_samples_, stereo_differ_);
        stereo_samples_ = stereo_differ_ = 0;
    }
}

bool SetupEngine::retire(u32 handle, u32 list, u32 prev, u32 next, u32 state) {
    const bool host_unlinks = retrap_ && (host_.global_reg(kRegTrapForce1) & kTrapForceIdleVoice);
    if (!host_unlinks) {
        if (prev == kNoVoice) host_.set_global_reg(kRegListTop[list], next);
        else set_field(prev, kFieldPitchLink, (field(prev, kFieldPitchLink) & 0xFFFF0000u) | next);
    }
    set_field(handle, kFieldState, state & ~(kStateActive | kStateNew));
    book_[handle].off_pending = false;
    // Turning a voice off (VOICE_OFF, the end of its release, a stream run dry) reports it done through its notifier:
    // the first segment list's for a buffer voice, the list in play for a stream. DirectSound waits on it to know a
    // buffer has stopped.
    write_notifier(handle, (field(handle, kFieldFmt) & kFmtStream) ? book_[handle].ssl_index : 0u);
    trace::log(trace::Cat::Apu, "vp: voice {:x} off, notifier {} done", handle, (field(handle, kFieldFmt) & kFmtStream) ? book_[handle].ssl_index : 0u);
    ++retired_;
    // With the host unlinking, the walk raises the idle report from the next frame on, every frame it is still listed.
    if (!host_unlinks && (host_.global_reg(kRegTrapForce1) & kTrapForceIdleVoice)) host_.front_end_trap(kSe2FeIdleVoice, handle);
    return host_unlinks;
}

void SetupEngine::walk(u32 list) {
    u32 prev = kNoVoice;
    u32 handle = host_.global_reg(kRegListTop[list]) & kHandleMask;
    for (u32 guard = 0; handle < kMaxVoices && guard < kMaxVoices; ++guard) {
        const u32 link = field(handle, kFieldPitchLink);
        const u32 next = link & kHandleMask;
        u32 state = field(handle, kFieldState);
        bool retire_now = book_[handle].off_pending;
        if (retrap_ && !(state & kStateActive) && !retire_now && (host_.global_reg(kRegTrapForce1) & kTrapForceIdleVoice)) {
            host_.front_end_trap(kSe2FeIdleVoice, handle);  // an idle voice the host has not taken off yet
            prev = handle;
            handle = next;
            continue;
        }
        if (!retire_now && (state & kStateActive) && !(state & kStatePaused)) {
            bool ended = false;
            const float ef = step_envelope(handle, false, ended);
            ended = false;
            const float ea = step_envelope(handle, true, ended);
            if (ended) {
                retire_now = true;
            } else {
                // The pitch is octaves * 4096 (signed), modulated by the filter envelope through the pitch scale.
                const i32 pitch = static_cast<i16>(link >> 16);
                const i32 pitch_scale = static_cast<i8>(field(handle, kFieldEnv0) >> 24);
                const double step = std::exp2((pitch + pitch_scale * 32.0 * static_cast<double>(ef)) / 4096.0);
                const u32 fmt = field(handle, kFieldFmt);
                Frame samples{};
                const bool more = (fmt & kFmtMultipass) ? read_submix(fmt, samples)
                                  : (fmt & kFmtStream) ? render_stream(handle, fmt, step, samples) : render_buffer(handle, fmt, step, samples);
                filter(handle, fmt, samples);
                mix_voice(handle, samples, ea);
                if (direct_on_) direct_voice(handle, samples, ea);
                // A stream that runs dry and a one-shot buffer that reaches its end both turn off: the voice leaves its
                // list with its notifier written and the idle report, which is how DirectSound learns a buffer stopped
                // (a title waiting for a sound to finish otherwise waits for ever). The reference emulator turns it off too.
                if (!more) retire_now = true;
                // The first frame played: the voice is no longer new.
                state = field(handle, kFieldState) & ~kStateNew;
                set_field(handle, kFieldState, state);
            }
        }
        if (!retire_now || retire(handle, list, prev, next, state)) prev = handle;
        handle = next;
    }
}

// A multipass voice reads no memory: its input is one mix bin (CFG_FMT bits 20:16, where other voices keep their
// samples per block) as the voices walked before it this frame left it, the same signal on both channels, and with
// CLEAR_MIX (bit 26) the bin is emptied once read. The lists are walked 2D, 3D, multipass, so a submix's sources have
// all been mixed by then. It then runs its envelopes, filter and eight sends like any other voice. A submix never runs
// dry.
bool SetupEngine::read_submix(u32 fmt, Frame& out) {
    auto& bin = mix_[(fmt >> kFmtSamplesPerBlockShift) & 0x1Fu];
    for (u32 i = 0; i < kSamplesPerFrame; ++i) out[i] = {bin[i], bin[i]};
    if (fmt & kFmtClearMix) bin.fill(0.0f);
    return true;
}

// ── envelopes ────────────────────────────────────────────────────────────────────────────────────────────────────

float SetupEngine::step_envelope(u32 handle, bool amplitude, bool& ended) {
    const u32 shift = amplitude ? kStateEaShift : kStateEfShift;
    const u32 state = field(handle, kFieldState);
    u32 phase = (state >> shift) & 0xFu;
    const u32 times = field(handle, amplitude ? kFieldEnv0 : kFieldEnv1);  // attack rate 11:0, delay time 23:12
    const u32 shape = field(handle, amplitude ? kFieldEnvA : kFieldEnvF);  // decay rate 11:0, hold 23:12, sustain 31:24
    const u32 release = field(handle, amplitude ? kFieldLfoEnv : kFieldMisc) & 0xFFFu;
    const u32 ecnt = field(handle, kFieldEcnt);
    u32 count = amplitude ? (ecnt & 0xFFFFu) : (ecnt >> 16);
    const u32 level_field = amplitude ? kFieldOffset : kFieldNext;  // the level sits in bits 31:24
    const u32 level_word = field(handle, level_field);
    float level = static_cast<float>(level_word >> 24);
    const u32 attack = (times & 0xFFFu) * 16, hold = ((shape >> 12) & 0xFFFu) * 16, decay = (shape & 0xFFFu) * 16;
    const float sustain = static_cast<float>(shape >> 24);
    float value = 0.0f;
    switch (phase) {
        case kOff:
            count = 0;
            level = 255.0f;
            value = 255.0f;
            break;
        case kDelay:
            level = 0.0f;
            if (count == 0) phase = kAttack;
            else --count;
            break;
        case kAttack:
            value = (attack == 0 || count >= attack) ? 255.0f : static_cast<float>(count) * 255.0f / static_cast<float>(attack);
            level = value;
            if (count >= attack) {
                phase = kHold;
                count = hold;
            } else {
                ++count;
            }
            break;
        case kHold:
            level = value = 255.0f;
            if (count == 0) {
                phase = kDecay;
                count = decay;
            } else {
                --count;
            }
            break;
        case kDecay:
            // An exponential fall from full towards the sustain level over the decay time (the wiki's fit).
            value = decay == 0 ? 0.0f : 255.0f * std::pow(0.99988799f, static_cast<float>(decay - count) * 4096.0f * 16.0f / static_cast<float>(decay));
            if (value <= sustain + 0.2f || count == 0) {
                phase = kSustain;
                value = std::max(value, sustain);
            } else {
                --count;
                level = value;
            }
            break;
        case kSustain:
            count = 0;
            level = value = sustain;
            break;
        case kRelease:
            if (release == 0) {
                // A release rate of zero holds the level the release began at: the voice sounds on until VOICE_OFF
                // (a voice released from the off phase went straight to the forced release instead, above).
                // The boot sound releases its two bass drones this way and keeps re-pitching one of them for five
                // seconds; the recording of a real console still carries both fundamentals after the releases.
                value = level;
            } else if (count == 0) {
                phase = kForceRelease;
                value = 0.0f;
            } else {
                // From the level held at the release towards silence over the release time (-60 dB at its end).
                const float pos = std::clamp(1.0f - static_cast<float>(count) / static_cast<float>(release * 16), 0.0f, 1.0f);
                value = std::exp(-6.91f * pos) * level;
                --count;
            }
            break;
        default:  // kForceRelease
            if (amplitude) ended = true;
            value = 0.0f;
            break;
    }
    set_field(handle, kFieldState, (field(handle, kFieldState) & ~(0xFu << shift)) | (phase << shift));
    set_field(handle, kFieldEcnt, amplitude ? ((ecnt & 0xFFFF0000u) | (count & 0xFFFFu)) : ((ecnt & 0xFFFFu) | (count << 16)));
    set_field(handle, level_field, (level_word & kOffsetMask) | (static_cast<u32>(std::clamp(level, 0.0f, 255.0f)) << 24));
    return std::clamp(value / 255.0f, 0.0f, 1.0f);
}

// ── sample sources ───────────────────────────────────────────────────────────────────────────────────────────────

SetupEngine::Source SetupEngine::source_for(u32 fmt) const {
    Source s;
    s.stream = (fmt & kFmtStream) != 0;
    const u32 container = fmt >> kFmtContainerShift;
    s.adpcm = container == kContainerAdpcm;
    s.container = std::array<u32, 4>{1, 2, 0, 4}[container];
    s.sample_size = (fmt >> kFmtSampleSizeShift) & 3u;
    s.channels = (fmt & kFmtStereo) ? 2u : 1u;
    s.samples_per_block = 1 + ((fmt >> kFmtSamplesPerBlockShift) & 0x1Fu);
    s.sge_table = host_.global_reg(kRegSgeTable);
    return s;
}

// A byte of the voice-buffer space: 4 KiB pages named by the scatter-gather table at VPSGEADDR (8-byte entries,
// the page's physical address first).
bool SetupEngine::sge_byte(u32 sge_offset, PhysAddr& phys) const {
    u32 entry = 0;
    if (!bus_.read_ram(host_.global_reg(kRegSgeTable) + (sge_offset / 4096u) * 8u, &entry, 4)) return false;
    phys = (entry & ~0xFFFu) + (sge_offset & 0xFFFu);
    return true;
}

u32 SetupEngine::read_bytes_slow(const Source& src, u32 byte_offset, u32 bytes) const {
    u32 v = 0;
    // The common case: all the bytes in one 4 KiB page of the buffer (one scatter-gather lookup, one read; a sample
    // was a table lookup and a bus read per byte). Anything else, byte by byte below, with the same results.
    if (src.stream) {
        if (bus_.read_ram(src.segment + byte_offset, &v, bytes)) return v;
        v = 0;
    } else if (((src.base + byte_offset) & 0xFFFu) + bytes <= 0x1000u) {
        const u32 off = src.base + byte_offset;
        if ((off >> 12) != src.page) {
            src.page = off >> 12;
            u32 entry = 0;
            src.page_ptr = bus_.read_ram(src.sge_table + src.page * 8u, &entry, 4) ? bus_.ram_pointer(entry & ~0xFFFu, 0x1000) : nullptr;
        }
        if (src.page_ptr) {
            std::memcpy(&v, src.page_ptr + (off & 0xFFFu), bytes);
            return v;
        }
        PhysAddr at = 0;
        if (!sge_byte(src.base + byte_offset, at)) return 0;
        if (bus_.read_ram(at, &v, bytes)) return v;
        v = 0;
    }
    for (u32 i = 0; i < bytes; ++i) {
        PhysAddr at = 0;
        if (src.stream) at = src.segment + byte_offset + i;
        else if (!sge_byte(src.base + byte_offset + i, at)) return 0;
        u8 b = 0;
        bus_.read_ram(at, &b, 1);
        v |= static_cast<u32>(b) << (8 * i);
    }
    return v;
}

// Sample `index` of a source, both channels (a mono source repeats its one), full scale = 1.0.
void SetupEngine::read_sample(u32 handle, const Source& src, u32 index, float out[2]) {
    if (src.adpcm) {
        const u32 block = index / kAdpcmSamplesPlayed, within = index % kAdpcmSamplesPlayed;
        const u32 block_bytes = 36u * src.channels;
        Bookkeeping& b = book_[handle];
        const u64 key = (static_cast<u64>(src.stream ? src.segment : src.base) << 24) ^ block;
        if (b.adpcm_key != key) {
            std::array<u8, 72> raw{};
            for (u32 i = 0; i < block_bytes; i += 4) {
                const u32 w = read_bytes(src, block * block_bytes + i, 4);
                for (u32 k = 0; k < 4; ++k) raw[i + k] = static_cast<u8>(w >> (8 * k));
            }
            decode_adpcm_block(raw.data(), src.channels, b.adpcm);
            b.adpcm_key = key;
        }
        out[0] = b.adpcm[within * 2] / 32768.0f;
        out[1] = b.adpcm[within * 2 + 1] / 32768.0f;
        return;
    }
    const u32 block = src.container * std::max(src.samples_per_block, src.channels);
    for (u32 c = 0; c < 2; ++c) {
        const u32 ch = src.channels == 2 ? c : 0;
        const u32 raw = read_bytes(src, index * block + ch * src.container, src.container);
        float v = 0.0f;
        switch (src.sample_size) {
            case 0: v = (static_cast<float>(raw & 0xFFu) - 128.0f) / 128.0f; break;                       // U8
            case 1: v = static_cast<i16>(raw & 0xFFFFu) / 32768.0f; break;                                 // S16
            case 2: v = static_cast<float>(static_cast<i32>(raw << 8) >> 8) / 8388608.0f; break;           // S24
            default: v = static_cast<float>(static_cast<i32>(raw)) / 2147483648.0f; break;                 // S32
        }
        out[c] = v;
    }
}

// A buffer voice plays samples CBO..EBO (inclusive) at `step` source samples per output sample, interpolating
// linearly; past EBO a looping voice continues from LBO, another ends (CBO stays at EBO).
bool SetupEngine::render_buffer(u32 handle, u32 fmt, double step, Frame& out) {
    Bookkeeping& b = book_[handle];
    Source src = source_for(fmt);
    src.base = field(handle, kFieldPslStart) & kOffsetMask;
    const u32 ebo = field(handle, kFieldNext) & kOffsetMask, lbo = field(handle, kFieldPshSample) & kOffsetMask;
    const bool loop = (fmt & kFmtLoop) && ebo >= lbo;
    double pos = (field(handle, kFieldOffset) & kOffsetMask) + b.fraction;
    bool more = true;
    u32 n = 0;
    for (; n < kSamplesPerFrame; ++n) {
        if (pos >= static_cast<double>(ebo) + 1.0) {
            if (!loop) {
                more = false;
                break;
            }
            const double length = static_cast<double>(ebo) + 1 - lbo;
            pos = lbo + std::fmod(pos - (ebo + 1.0), length);
        }
        const u32 i0 = static_cast<u32>(pos);
        const u32 i1 = i0 < ebo ? i0 + 1 : (loop ? lbo : i0);
        float a[2], c[2];
        read_sample(handle, src, i0, a);
        read_sample(handle, src, i1, c);
        const float t = static_cast<float>(pos - i0);
        out[n] = {a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t};
        pos += step;
    }
    if (more && pos >= static_cast<double>(ebo) + 1.0) {
        if (loop) pos = lbo + std::fmod(pos - (ebo + 1.0), static_cast<double>(ebo) + 1 - lbo);
        else more = false;
    }
    if (!more) pos = ebo;
    const u32 whole = static_cast<u32>(pos);
    b.fraction = more ? pos - whole : 0.0;
    set_field(handle, kFieldOffset, (field(handle, kFieldOffset) & ~kOffsetMask) | (whole & kOffsetMask));
    return more;
}

// A stream voice plays segment after segment of its current SSL; a finished SSL is reported and the voice moves to
// the other one (A, B, A, ...), whose segments the driver refills meanwhile. An SSL with no segments starves the
// voice: a persistent voice waits (reporting once), any other ends. The position carries across segment ends.
bool SetupEngine::render_stream(u32 handle, u32 fmt, double step, Frame& out) {
    Bookkeeping& b = book_[handle];
    const PhysAddr table = host_.global_reg(kRegSslTable);
    double pos = (field(handle, kFieldOffset) & kOffsetMask) + b.fraction;
    // After the frame, a segment whose last sample has played is finished at once (and its list reported).
    for (u32 n = 0; n <= kSamplesPerFrame;) {
        if (b.ssl_count[b.ssl_index] == 0) {
            b.fraction = 0;
            b.ssl_segment = 0;
            set_field(handle, kFieldOffset, field(handle, kFieldOffset) & ~kOffsetMask);
            if (!(fmt & kFmtPersist)) return false;
            if (!b.starved_reported) {
                notify(handle, b.ssl_index);
                b.starved_reported = true;
            }
            return true;  // silence for the rest of the frame
        }
        b.starved_reported = false;
        u32 seg[2] = {0, 0};
        bus_.read_ram(table + (b.ssl_base[b.ssl_index] + b.ssl_segment) * 8, seg, 8);
        const double length = seg[1] & 0xFFFFu;  // samples in the segment
        if (pos >= length) {
            pos -= length;
            if (++b.ssl_segment >= b.ssl_count[b.ssl_index]) {
                notify(handle, b.ssl_index);
                b.ssl_segment = 0;
                b.ssl_index ^= 1;
            }
            if (length == 0) pos = 0;
            continue;
        }
        if (n == kSamplesPerFrame) break;
        // The segment's own format: container (17:16), samples per block - 1 (22:18), stereo (23).
        const u32 seg_fmt = (fmt & ~((3u << kFmtContainerShift) | (0x1Fu << kFmtSamplesPerBlockShift) | kFmtStereo)) |
                            (((seg[1] >> 16) & 3u) << kFmtContainerShift) | (((seg[1] >> 18) & 0x1Fu) << kFmtSamplesPerBlockShift) |
                            ((seg[1] & (1u << 23)) ? kFmtStereo : 0u);
        Source src = source_for(seg_fmt);
        if (src.adpcm) src.sample_size = 2;
        src.stream = true;
        src.segment = seg[0];
        const u32 i0 = static_cast<u32>(pos);
        const u32 i1 = static_cast<double>(i0) + 1 < length ? i0 + 1 : i0;  // no interpolation across segments
        float a[2], c[2];
        read_sample(handle, src, i0, a);
        read_sample(handle, src, i1, c);
        const float t = static_cast<float>(pos - i0);
        out[n] = {a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t};
        pos += step;
        ++n;
    }
    const u32 whole = static_cast<u32>(pos);
    b.fraction = pos - whole;
    set_field(handle, kFieldOffset, (field(handle, kFieldOffset) & ~kOffsetMask) | (whole & kOffsetMask));
    return true;
}

// The DLS2 low-pass: TAR_FCA (channel 0, and a mono voice) and TAR_FCB (channel 1) hold the cutoff coefficient in
// 15:0, 4096 * log2(2 sin(pi fc / 48000)) per the wiki's formula, and the resonance damping 10^(-dB/20) in 31:16
// (0x8000 = 1). Those are the two coefficients of a Chamberlin state-variable filter, run here on its low-pass
// output. Which voices filter (CFG_MISC FMODE, bits 17:16) follows the reference emulator: a 3D voice in mode 1, a
// 2D stereo voice in mode 1, a 2D mono voice in modes 1 and 3.
void SetupEngine::filter(u32 handle, u32 fmt, Frame& samples) {
    const u32 fmode = (field(handle, kFieldMisc) >> 16) & 3u;
    const bool stereo = (fmt & kFmtStereo) != 0;
    const bool lpf = (handle < k3dVoices || stereo) ? fmode == 1 : (fmode & 1u) != 0;
    if (!lpf) return;
    const u32 channels = stereo ? 2u : 1u;
    for (u32 ch = 0; ch < channels; ++ch) {
        const u32 word = field(handle, ch == 0 ? kFieldFcA : kFieldFcB);
        const float f = std::clamp(std::exp2(static_cast<float>(static_cast<i16>(word & 0xFFFFu)) / 4096.0f), 0.003906f, 1.0f);
        const float q = std::clamp(static_cast<float>(word >> 16) / 32768.0f, 0.079407f, 1.0f);
        auto& st = book_[handle].svf[ch];
        for (auto& s : samples) {
            st[0] += f * st[1];                    // low
            const float high = s[ch] - st[0] - q * st[1];
            st[1] += f * high;                     // band
            s[ch] = std::clamp(st[0], -1.0f, 1.0f);
        }
    }
    if (!stereo)
        for (auto& s : samples) s[1] = s[0];
}

// ── mixing ───────────────────────────────────────────────────────────────────────────────────────────────────────

// Eight bins per voice (V0-V5 in CFG_VBIN, V6-V7 in CFG_FMT), each with a 12-bit attenuation from TAR_VOLA-C
// (V6 and V7 split a nibble per word); even bins take the left channel, odd the right. The 3D voices' first four
// bins are the HRTF submixes. A bin's headroom divides what is added to it by 2^headroom.
std::array<SetupEngine::Send, 8> SetupEngine::sends(u32 handle) const {
    const u32 vbin = field(handle, kFieldVbin), fmt = field(handle, kFieldFmt);
    const u32 vola = field(handle, kFieldVolA), volb = field(handle, kFieldVolB), volc = field(handle, kFieldVolC);
    const std::array<u32, 8> bins{vbin & 0x1Fu, (vbin >> 5) & 0x1Fu, (vbin >> 10) & 0x1Fu, (vbin >> 16) & 0x1Fu, (vbin >> 21) & 0x1Fu,
                                  (vbin >> 26) & 0x1Fu, fmt & 0x1Fu, (fmt >> 5) & 0x1Fu};
    const std::array<u32, 8> vols{(vola >> 4) & 0xFFFu, vola >> 20, (volb >> 4) & 0xFFFu, volb >> 20, (volc >> 4) & 0xFFFu, volc >> 20,
                                  (vola & 0xFu) | ((volb & 0xFu) << 4) | ((volc & 0xFu) << 8),
                                  ((vola >> 16) & 0xFu) | (((volb >> 16) & 0xFu) << 4) | (((volc >> 16) & 0xFu) << 8)};
    std::array<Send, 8> out{};
    for (u32 k = 0; k < 8; ++k) {
        u32 bin = bins[k];
        u32 headroom;
        if (handle < k3dVoices && k < 4) {
            bin = hrtf_submix_[k];
            headroom = hrtf_headroom_;
        } else {
            headroom = submix_headroom_[bin];
        }
        const float level = attenuation(vols[k]);
        out[k] = {bin, level / static_cast<float>(1u << headroom), bins[k], level / static_cast<float>(1u << submix_headroom_[bins[k]])};
    }
    return out;
}

namespace {
// Instrument: MV_DEBUG_VOICES=41,4a mixes only the listed voices (hex handles); the rest still play silently.
bool soloed(u32 handle) {
    static const std::array<bool, SetupEngine::kMaxVoices> solo = [] {
        std::array<bool, SetupEngine::kMaxVoices> s{};
        const char* e = std::getenv("MV_DEBUG_VOICES");
        if (!e) s.fill(true);
        for (; e && *e; ) {
            char* end = nullptr;
            const unsigned long h = std::strtoul(e, &end, 16);
            if (end == e) break;
            if (h < SetupEngine::kMaxVoices) s[h] = true;
            e = *end ? end + 1 : end;
        }
        return s;
    }();
    return solo[handle];
}
}  // namespace

void SetupEngine::mix_voice(u32 handle, const Frame& samples, float gain) {
    if (gain <= 0.0f || !soloed(handle)) return;
    const bool stereo = (field(handle, kFieldFmt) & kFmtStereo) != 0;
    if (stereo)
        for (const auto& s : samples) {
            ++stereo_samples_;
            if (s[0] != s[1]) ++stereo_differ_;
        }
    const auto voice_sends = sends(handle);
    for (u32 k = 0; k < 8; ++k) {
        const u32 bin = voice_sends[k].bin;
        const float send_gain = voice_sends[k].gain;
        const float g = gain * send_gain;
        if (g == 0.0f) continue;
        const u32 channel = stereo ? (k & 1u) : 0u;
        for (u32 i = 0; i < kSamplesPerFrame; ++i) {
            mix_[bin][i] += g * samples[i][channel];
            peak_ = std::max(peak_, std::abs(mix_[bin][i]));
        }
    }
    ++frames_mixed_;
}

// The direct mix: the voice's two channels (a mono voice has its sample in both) straight into stereo at its loudest
// send, as the reference emulator plays it: the bin the voice names with that bin's headroom, a 3D voice's HRTF sends
// included (it would be routed to the HRTF submixes otherwise). Sends to a bin a multipass voice reads are left out, or
// a submix's sources would be heard twice.
void SetupEngine::direct_voice(u32 handle, const Frame& samples, float gain) {
    if (gain <= 0.0f || !soloed(handle)) return;
    float g = 0.0f;
    for (const Send& s : sends(handle))
        if (!(direct_skip_ & (1u << s.own_bin))) g = std::max(g, s.own_gain);
    g *= gain;
    if (g == 0.0f) return;
    for (u32 i = 0; i < kSamplesPerFrame; ++i) {
        direct_[i][0] += g * samples[i][0];
        direct_[i][1] += g * samples[i][1];
    }
}

u32 SetupEngine::multipass_bins() const {
    u32 bins = 0;
    for (u32 list = 0; list < kRegListTop.size(); ++list)
        for (u32 h = host_.global_reg(kRegListTop[list]) & kHandleMask, guard = 0; h < kMaxVoices && guard < kMaxVoices;
             h = field(h, kFieldPitchLink) & kHandleMask, ++guard) {
            const u32 fmt = field(h, kFieldFmt);
            if ((fmt & kFmtMultipass) && (field(h, kFieldState) & kStateActive)) bins |= 1u << ((fmt >> kFmtSamplesPerBlockShift) & 0x1Fu);
        }
    return bins;
}

u32 SetupEngine::dump_handle() {
    static const u32 h = [] {
        const char* d = std::getenv("MV_DEBUG_VOICE_DUMP");
        return d && *d ? static_cast<u32>(std::strtoul(d, nullptr, 16)) : kNoVoice;
    }();
    return h;
}

// scratch/voice-HEX-N.wav: the buffer as decoded (16-bit mono at the rate its pitch plays it); an ADPCM buffer's bytes as
// stored go to scratch/voice-HEX-N.adpcm too, for a decoder of our own choosing.
void SetupEngine::dump_voice(u32 handle) {
    static u32 dumps = 0;
    if (dumps >= 64) return;
    const u32 fmt = field(handle, kFieldFmt);
    if (fmt & kFmtStream) return;
    Source src = source_for(fmt);
    src.base = field(handle, kFieldPslStart) & kOffsetMask;
    const u32 n = std::min((field(handle, kFieldNext) & kOffsetMask) + 1, 1u << 20);
    const i32 pitch = static_cast<i16>(field(handle, kFieldPitchLink) >> 16);
    const u32 rate = static_cast<u32>(48000.0 * std::exp2(pitch / 4096.0));
    const u32 at = dumps++;
    if (std::FILE* w = std::fopen(std::format("scratch/voice-{:x}-{}.wav", handle, at).c_str(), "wb")) {
        const u32 bytes = n * 2;
        const u32 hdr[11] = {0x46464952u, 36 + bytes, 0x45564157u, 0x20746D66u, 16, 0x00010001u, rate, rate * 2, 0x00100002u, 0x61746164u, bytes};
        std::fwrite(hdr, 4, 11, w);
        for (u32 i = 0; i < n; ++i) {
            float s[2];
            read_sample(handle, src, i, s);
            const i16 v = static_cast<i16>(std::clamp(s[0] * 32767.0f, -32768.0f, 32767.0f));
            std::fwrite(&v, 2, 1, w);
        }
        std::fclose(w);
    }
    if (src.adpcm)
        if (std::FILE* w = std::fopen(std::format("scratch/voice-{:x}-{}.adpcm", handle, at).c_str(), "wb")) {
            const u32 bytes = (n + kAdpcmSamplesPlayed - 1) / kAdpcmSamplesPlayed * 36u * src.channels;
            for (u32 i = 0; i < bytes; i += 4) {
                const u32 v = read_bytes(src, i, 4);
                std::fwrite(&v, 4, 1, w);
            }
            std::fclose(w);
        }
    book_[handle].adpcm_key = ~0ull;  // the reads filled the ADPCM block cache
    trace::log(trace::Cat::Apu, "vp: voice {:x} dumped as {}: {} samples at {} Hz, cbo {}", handle, at, n, rate, field(handle, kFieldOffset) & kOffsetMask);
}

void SetupEngine::write_notifier(u32 handle, u32 index) {
    const PhysAddr at = host_.global_reg(kRegNotifierBase) + 16 * (kNotifierBase + handle * kNotifiersPerVoice + index);
    const u8 tail[2] = {1, kNotifierDone};  // bytes 14 and 15: the status last
    bus_.write_ram(at + 14, tail, 2);
    ++notifications_;
    host_.raise_interrupt(kIstsNotify);
}

void SetupEngine::notify(u32 handle, u32 ssl) {
    const PhysAddr at = host_.global_reg(kRegNotifierBase) + 16 * (kNotifierBase + handle * kNotifiersPerVoice + ssl);
    write_notifier(handle, ssl);
    trace::log(trace::Cat::Apu, "vp: voice {:x} SSL {} {} (notifier at {:08x})", handle, ssl == 0 ? "A" : "B",
               book_[handle].ssl_count[ssl] ? "done" : "starved", at);
}

}  // namespace mv::apu
