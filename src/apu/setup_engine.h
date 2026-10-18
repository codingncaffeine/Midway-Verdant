#pragma once
#include <array>
#include <cstddef>

#include "core/bus.h"
#include "core/types.h"

// The voice side of the APU as the kernel drives it. The front end (FE) decodes the method stream written to the
// voice processor's window into the voice structures in RAM; the setup engine (SE) walks the three voice lists
// (2D, 3D, multipass) once per frame, advancing every active voice through its buffer at its pitch, wrapping the
// looping ones and retiring the ones that end or were turned off. A retired voice is reported to the front end,
// which traps to the host (method 0x8000 with the voice handle as parameter) when the kernel asked for idle-voice
// traps. No samples are produced yet: the mix buffer the global processor reads stays silent.
// Written from the xboxdevwiki APU page (voice lists, pitch, envelopes), the public register listing (method
// numbers and voice-structure fields), and traces of the kernel's method stream.
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
    // Streams: a voice plays two lists of segments (SSL A and B) from a segment table in RAM; finishing one is
    // reported through the voice's notifier.
    static constexpr u32 kMethodSetNotifyDma = 0x180, kMethodSetSslDma = 0x18C, kMethodSetCurrentSsl = 0x190;
    static constexpr u32 kMethodVoiceSslA = 0x320, kMethodVoiceSslB = 0x35C;
    static constexpr u32 kMethodSslSegmentFirst = 0x600, kMethodSslSegmentLast = 0x7FC;  // offset word, length word
    static constexpr u32 kSslBasePageMask = 0x3FFFC0, kSslCountMask = 0xFF, kSslBaseShift = 8;
    static constexpr u32 kAntecedentHandle = 0xFFFFu, kAntecedentListShift = 16, kVoicePauseAction = 1u << 18;
    // The voice structure in RAM: 0x80 bytes per voice at the base the kernel programs.
    static constexpr u32 kVoiceSize = 0x80;
    static constexpr u32 kFieldFmt = 0x04, kFieldPslStart = 0x20, kFieldPshSample = 0x24, kFieldState = 0x54, kFieldOffset = 0x58,
                         kFieldNext = 0x5C, kFieldPitchLink = 0x7C;
    static constexpr u32 kFmtPersist = 1u << 23, kFmtStream = 1u << 24, kFmtLoop = 1u << 25;
    static constexpr u32 kStatePaused = 1u << 18, kStateNew = 1u << 20, kStateActive = 1u << 21;
    static constexpr u32 kOffsetMask = 0x00FFFFFFu, kHandleMask = 0xFFFFu, kNoVoice = 0xFFFFu, kMaxVoices = 256;
    // Global registers the engine uses.
    static constexpr u32 kRegVoiceBase = 0x202C, kRegSslTable = 0x2034, kRegNotifierBase = 0x115C;
    // The notifier of a voice's SSL: 16 bytes at FENADDR + 16 * (2 + voice * 4 + n), n = 0 for SSL A, 1 for B;
    // byte 15 the status (0x01 done), and the front end's notify interrupts (ISTS bits 5 and 6).
    static constexpr u32 kNotifierBase = 2, kNotifiersPerVoice = 4, kNotifierDone = 0x01, kIstsNotify = (1u << 5) | (1u << 6);
    static constexpr std::array<u32, 3> kRegListTop{0x2054, 0x2060, 0x206C};  // 2D, 3D, multipass
    static constexpr u32 kRegTrapForce1 = 0x1504, kTrapForceIdleVoice = 1u << 15;
    static constexpr u32 kSe2FeIdleVoice = 0x8000;
    static constexpr u32 kSamplesPerFrame = 32;

    SetupEngine(Bus& bus, SetupEngineHost& host) : bus_(bus), host_(host) {}

    // A method written to the voice processor's window; true when the engine consumed it.
    bool method(u32 offset, u32 value);
    // One audio frame: walk the lists.
    void frame();

    u32 current_voice() const { return current_; }
    u64 voices_started() const { return started_; }
    u64 voices_retired() const { return retired_; }
    u64 notifications() const { return notifications_; }
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
    };
    PhysAddr voice_address(u32 handle) const;
    void link_on(u32 handle);
    void walk(u32 list);
    // A stream voice's turn: advance `step` samples through its segments; false when the voice ends.
    bool advance_stream(u32 handle, double step);
    void notify(u32 handle, u32 ssl);

    Bus& bus_;
    SetupEngineHost& host_;
    u32 current_{kNoVoice};
    u32 antecedent_{0};
    u32 current_ssl_page_{0};
    u64 notifications_{0};
    std::array<Bookkeeping, kMaxVoices> book_{};
    u64 started_{0}, retired_{0};
};

}  // namespace mv::apu
