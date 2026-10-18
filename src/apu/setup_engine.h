#pragma once
#include <vector>
#include <array>
#include <cstddef>

#include "core/bus.h"
#include "core/types.h"

// The voice side of the APU as the kernel drives it. The front end (FE) decodes the method stream written to the
// voice processor's window into the voice structures in RAM; the setup engine (SE) walks the three voice lists
// (2D, 3D, multipass) once per frame. Each active voice produces the frame's 32 samples: it reads its buffer (PCM in
// 8/16/24/32-bit containers or Xbox ADPCM, through the voice-buffer scatter-gather table) or its stream segments, at
// its pitch with linear interpolation, under its amplitude envelope, and mixes them into eight of the 32 mix bins
// with its eight volumes and the bins' headroom. The bins are what the global processor reads from its MIXBUF.
// A voice that ends, is turned off or finishes its release is retired and reported to the front end, which traps to
// the host (method 0x8000 with the voice handle as parameter) when the kernel asked for idle-voice traps.
// Written from the xboxdevwiki APU page (voice lists, pitch, envelopes, MIXBUF), the xboxdevwiki Xbox ADPCM page,
// the public register listing (method numbers and voice-structure fields), traces of the kernel's method stream,
// and the reference emulator's voice processor read for behaviour (volume scale, headroom, envelope stepping).
namespace mv::apu {

class SetupEngineHost {
public:
    virtual ~SetupEngineHost() = default;
    virtual u32 global_reg(u32 offset) = 0;
    virtual void set_global_reg(u32 offset, u32 value) = 0;
    virtual void front_end_trap(u32 method, u32 parameter) = 0;
    // Sets interrupt status bits (ISTS) and re-evaluates the interrupt line.
    virtual void raise_interrupt(u32 ists_bits) = 0;
};

class SetupEngine {
public:
    // Methods, as offsets in the voice processor's window.
    static constexpr u32 kMethodSetAntecedentVoice = 0x120, kMethodVoiceOn = 0x124, kMethodVoiceOff = 0x128, kMethodVoiceRelease = 0x12C,
                         kMethodVoicePause = 0x140, kMethodSetCurrentVoice = 0x2F8, kMethodVoiceLock = 0x2FC;
    static constexpr u32 kMethodCfgFirst = 0x300, kMethodCfgLast = 0x37C;  // configuration and target fields
    static constexpr u32 kMethodBufBase = 0x3A0, kMethodBufLbo = 0x3A4, kMethodBufCbo = 0x3D8, kMethodBufEbo = 0x3DC;
    // Mixing: each bin's headroom (a right shift of what voices add to it, 0-7), the headroom of the 3D voices'
    // HRTF bins, and which four bins those are.
    // The voice-buffer scatter-gather table is written through methods: select an entry, then give it a page.
    static constexpr u32 kMethodSetCurrentInbufSge = 0x804, kMethodSetCurrentInbufSgeOffset = 0x808;
    static constexpr u32 kMethodSubmixHeadroomFirst = 0x200, kMethodSubmixHeadroomLast = 0x27C, kMethodHrtfHeadroom = 0x280,
                         kMethodHrtfSubmixes = 0x2C0;
    // Streams: a voice plays two lists of segments (SSL A and B) from a segment table in RAM; finishing one is
    // reported through the voice's notifier.
    static constexpr u32 kMethodSetNotifyDma = 0x180, kMethodSetSslDma = 0x18C, kMethodSetCurrentSsl = 0x190;
    static constexpr u32 kMethodVoiceSslA = 0x320, kMethodVoiceSslB = 0x35C;
    static constexpr u32 kMethodSslSegmentFirst = 0x600, kMethodSslSegmentLast = 0x7FC;  // offset word, length word
    static constexpr u32 kSslBasePageMask = 0x3FFFC0, kSslCountMask = 0xFF, kSslBaseShift = 8;
    static constexpr u32 kAntecedentHandle = 0xFFFFu, kAntecedentListShift = 16, kVoicePauseAction = 1u << 18;
    // The voice structure in RAM: 0x80 bytes per voice at the base the kernel programs.
    static constexpr u32 kVoiceSize = 0x80;
    static constexpr u32 kFieldVbin = 0x00, kFieldFmt = 0x04, kFieldEnv0 = 0x08, kFieldEnvA = 0x0C, kFieldEnv1 = 0x10, kFieldEnvF = 0x14,
                         kFieldMisc = 0x18, kFieldPslStart = 0x20, kFieldPshSample = 0x24, kFieldEcnt = 0x34, kFieldState = 0x54,
                         kFieldOffset = 0x58, kFieldNext = 0x5C, kFieldVolA = 0x60, kFieldVolB = 0x64, kFieldVolC = 0x68,
                         kFieldLfoEnv = 0x6C, kFieldFcA = 0x74, kFieldFcB = 0x78, kFieldPitchLink = 0x7C;
    static constexpr u32 kFmtSamplesPerBlockShift = 16, kFmtMultipass = 1u << 21, kFmtPersist = 1u << 23, kFmtStream = 1u << 24,
                         kFmtLoop = 1u << 25, kFmtStereo = 1u << 27, kFmtSampleSizeShift = 28, kFmtContainerShift = 30;
    static constexpr u32 kContainerAdpcm = 2;
    static constexpr u32 kStatePaused = 1u << 18, kStateNew = 1u << 20, kStateActive = 1u << 21;
    static constexpr u32 kStateEfShift = 24, kStateEaShift = 28;  // envelope phases, 4 bits each
    static constexpr u32 kOffsetMask = 0x00FFFFFFu, kHandleMask = 0xFFFFu, kNoVoice = 0xFFFFu, kMaxVoices = 256, k3dVoices = 64;
    // Envelope phases.
    enum Phase : u32 { kOff = 0, kDelay = 1, kAttack = 2, kHold = 3, kDecay = 4, kSustain = 5, kRelease = 6, kForceRelease = 7 };
    // Global registers the engine uses.
    static constexpr u32 kRegVoiceBase = 0x202C, kRegSgeTable = 0x2030, kRegSslTable = 0x2034, kRegNotifierBase = 0x115C;
    // The notifier of a voice's SSL: 16 bytes at FENADDR + 16 * (2 + voice * 4 + n), n = 0 for SSL A, 1 for B;
    // byte 15 the status (0x01 done), and the front end's notify interrupts (ISTS bits 5 and 6).
    static constexpr u32 kNotifierBase = 2, kNotifiersPerVoice = 4, kNotifierDone = 0x01, kIstsNotify = (1u << 5) | (1u << 6);
    static constexpr std::array<u32, 3> kRegListTop{0x2054, 0x2060, 0x206C};  // 2D, 3D, multipass
    static constexpr u32 kRegTrapForce1 = 0x1504, kTrapForceIdleVoice = 1u << 15;
    static constexpr u32 kSe2FeIdleVoice = 0x8000;
    static constexpr u32 kSamplesPerFrame = 32, kBins = 32;
    static constexpr u32 kAdpcmSamplesPerBlock = 65;  // the header sample and 64 nibbles per channel [verify]

    using Mix = std::array<std::array<float, kSamplesPerFrame>, kBins>;
    using Frame = std::array<std::array<float, 2>, kSamplesPerFrame>;  // left, right per sample

    SetupEngine(Bus& bus, SetupEngineHost& host) : bus_(bus), host_(host) {}

    // A method written to the voice processor's window; true when the engine consumed it.
    bool method(u32 offset, u32 value);
    // One audio frame: walk the lists, render and mix every active voice.
    void frame();
    // The frame's mix bins (full scale = 1.0), what the global processor's MIXBUF receives.
    const Mix& mix() const { return mix_; }

    u32 current_voice() const { return current_; }
    u64 voices_started() const { return started_; }
    // Instrument: every voice handle VOICE_ON has started since power-on, in order of first start.
    const std::vector<u32>& handles_started() const { return handles_started_; }
    u64 voices_retired() const { return retired_; }
    u64 notifications() const { return notifications_; }
    u64 frames_mixed() const { return frames_mixed_; }  // voice-frames that reached the bins
    float peak() const { return peak_; }  // the loudest bin sample so far (full scale = 1.0)
    u32 field(u32 handle, u32 offset) const;
    void set_field(u32 handle, u32 offset, u32 value);

private:
    struct Bookkeeping {
        double fraction{0};
        // Stream state, inside the voice processor (not in the voice structure): the two lists' base pages and
        // segment counts, the list playing and the segment within it.
        std::array<u32, 2> ssl_base{}, ssl_count{};
        u32 ssl_index{0}, ssl_segment{0};
        bool starved_reported{false};
        bool off_pending{false};
        // Played to the end of a one-shot buffer: silent but still on its list, until VOICE_OFF or a restart.
        bool finished{false};
        // The last ADPCM block decoded: its address and samples (left, right interleaved).
        u64 adpcm_key{~0ull};
        std::array<i16, kAdpcmSamplesPerBlock * 2> adpcm{};
        // The voice filter's state per channel: the state-variable filter's low-pass and band-pass outputs.
        std::array<std::array<float, 2>, 2> svf{};
    };
    // Where a voice reads its samples: a buffer (through the scatter-gather table) or a stream segment (physical).
    struct Source {
        bool stream{false};
        PhysAddr segment{0};  // a stream segment's first byte
        u32 base{0};          // a buffer's first byte in the scatter-gather space
        u32 container{2}, sample_size{1}, channels{1}, samples_per_block{1};
        bool adpcm{false};
    };
    PhysAddr voice_address(u32 handle) const;
    void link_on(u32 handle);
    bool on_a_list(u32 handle) const;  // whether the voice is on one of the three lists
    void walk(u32 list);
    void retire(u32 handle, u32 list, u32 prev, u32 next, u32 state);
    // A voice's frame of samples; false when the voice ends during it (the rest of the frame is silence).
    bool render_buffer(u32 handle, u32 fmt, double step, Frame& out);
    bool render_stream(u32 handle, u32 fmt, double step, Frame& out);
    void read_sample(u32 handle, const Source& src, u32 index, float out[2]);
    bool sge_byte(u32 sge_offset, PhysAddr& phys) const;
    u32 read_bytes(const Source& src, u32 byte_offset, u32 bytes) const;
    // One step of an envelope (amplitude or filter) at the frame rate; its level 0..1. The amplitude envelope's
    // FORCE_RELEASE (the end of a release) sets `ended`.
    float step_envelope(u32 handle, bool amplitude, bool& ended);
    // The voice's DLS2 low-pass filter where its filter mode asks for one (see the .cpp).
    void filter(u32 handle, u32 fmt, Frame& samples);
    void mix_voice(u32 handle, const Frame& samples, float gain);
    void notify(u32 handle, u32 ssl);
    Source source_for(u32 fmt) const;

    Bus& bus_;
    SetupEngineHost& host_;
    u32 current_{kNoVoice};
    u32 antecedent_{0};
    u32 current_ssl_page_{0};
    u32 inbuf_sge_{0};
    u64 notifications_{0};
    std::array<Bookkeeping, kMaxVoices> book_{};
    std::array<u8, kBins> submix_headroom_{};
    std::array<double, kBins> bin_energy_{};  // instrument: sum of squares per bin since the last listing
    u64 stereo_samples_{0}, stereo_differ_{0};  // instrument: samples stereo voices rendered since the last listing, and those with L != R
    u8 hrtf_headroom_{0};
    std::array<u8, 4> hrtf_submix_{0, 1, 2, 3};
    Mix mix_{};
    u64 started_{0}, retired_{0}, frames_mixed_{0};
    std::vector<u32> handles_started_;
    u64 frames_{0};
    float peak_{0.0f};
};

}  // namespace mv::apu
