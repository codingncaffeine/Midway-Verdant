#pragma once
#include <array>
#include <functional>
#include <string_view>

#include "core/bus.h"
#include "core/scheduler.h"
#include "pci/pci.h"

// The MCPX audio codec interface: an AC'97 controller in the Intel ICH register layout, reached through one 4 KiB
// memory window (the codec's mixer registers in the first 256 bytes, the bus master behind them) and the two I/O
// ranges. Each bus-master channel — PCM in, PCM out, microphone in, and the MCPX's S/PDIF out — walks a 32-entry
// buffer descriptor list in RAM at 48 kHz, 16-bit stereo, and raises the interrupt-on-completion and
// last-valid-buffer conditions as the Intel controller does. The samples a channel consumes go to a sink for the
// host's audio output. Written from the Intel ICH AC'97 controller register description and the AC'97 2.x codec
// register map (public), the xboxdevwiki ACI page, and traces of the kernel's programming.
namespace mv {

class Aci : public PciDevice, public MmioHandler, public PioHandler {
public:
    static constexpr u32 kMixerSize = 256, kBusMasterBase = 0x100, kBusMasterSize = 128, kMmioSize = 4096;
    enum Channel : u32 { PcmIn = 0, PcmOut = 1, MicIn = 2, Spdif = 3, kChannels = 4 };
    static constexpr std::array<u32, kChannels> kChannelOffset{0x00, 0x10, 0x20, 0x70};  // within the bus master
    // Channel registers, relative to the channel: descriptor list base, current index, last valid index, status,
    // position in current buffer, prefetched index, control.
    static constexpr u32 kBdbar = 0x0, kCiv = 0x4, kLvi = 0x5, kSr = 0x6, kPicb = 0x8, kPiv = 0xA, kCr = 0xB;
    static constexpr u16 kSrDch = 1u << 0, kSrCelv = 1u << 1, kSrLvbci = 1u << 2, kSrBcis = 1u << 3, kSrFifoe = 1u << 4;
    static constexpr u8 kCrRpbm = 1u << 0, kCrRr = 1u << 1, kCrLvbie = 1u << 2, kCrFeie = 1u << 3, kCrIoce = 1u << 4;
    static constexpr u32 kGlobCnt = 0x2C, kGlobSta = 0x30, kCas = 0x34;
    static constexpr u32 kGlobCntColdReset = 1u << 1, kGlobCntWarmReset = 1u << 2;
    static constexpr u32 kGlobStaPcr = 1u << 8;  // primary codec ready
    static constexpr std::array<u32, kChannels> kGlobStaChannelInt{1u << 5, 1u << 6, 1u << 7, 0};  // S/PDIF reports no interrupt
    static constexpr u32 kDescIoc = 1u << 31, kDescBup = 1u << 30, kDescLength = 0xFFFFu;
    static constexpr u32 kDescriptors = 32;
    // 48 kHz, two 16-bit channels: a descriptor's length counts 16-bit samples.
    static constexpr u64 kSamplesPerSecond = 96'000;
    // Mixer (codec) registers that read back a fixed value.
    static constexpr u32 kMixPowerdown = 0x26, kMixPowerdownReady = 0x000F;

    using SampleSink = std::function<void(Channel, const u8* data, u32 bytes)>;

    Aci(Bus& bus, Scheduler& sched, std::function<void(bool)> irq);
    std::string_view name() const override { return PciDevice::name(); }
    u32 mmio_read(PhysAddr addr, unsigned size) override;
    void mmio_write(PhysAddr addr, unsigned size, u32 value) override;
    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;

    // Samples consumed by a channel's buffers, in the order they were played; the host's audio output.
    void set_sample_sink(SampleSink sink) { sink_ = std::move(sink); }

    struct ChannelState {
        u32 bdbar{0};
        u8 civ{0}, lvi{0}, piv{0}, cr{0};
        u16 sr{kSrDch};
        // The buffer in flight.
        u32 buffer_address{0}, buffer_samples{0};
        bool buffer_ioc{false};
        Cycles buffer_started{0};
        TimerEvent done;
        u64 buffers_completed{0}, bytes_played{0};
    };
    const ChannelState& channel(Channel c) const { return ch_[c]; }
    bool irq_level() const { return irq_level_; }
    u64 reads() const { return reads_; }
    u64 writes() const { return writes_; }
    u32 glob_sta();

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    u32 read_byte(u32 offset);
    void write_byte(u32 offset, u8 value);
    u32 read(u32 offset, unsigned size);
    void write(u32 offset, unsigned size, u32 value);
    u32 mixer_read16(u32 reg);
    void mixer_write16(u32 reg, u16 value);
    u32 channel_read_byte(Channel c, u32 rel);
    void channel_write_byte(Channel c, u32 rel, u8 value);
    u16 picb(const ChannelState& s) const;
    void channel_reset(Channel c);
    void start_buffer(Channel c);
    void buffer_done(Channel c);
    bool channel_interrupting(const ChannelState& s) const;
    void update_irq();

    Bus& bus_;
    Scheduler& sched_;
    std::function<void(bool)> irq_;
    SampleSink sink_;
    u32 mmio_at_{0};
    u16 mixer_io_{0}, bus_master_io_{0};
    std::array<u16, kMixerSize / 2> mixer_{};
    std::array<ChannelState, kChannels> ch_{};
    u32 glob_cnt_{0};
    bool irq_level_{false};
    u64 reads_{0}, writes_{0};
};

}  // namespace mv
