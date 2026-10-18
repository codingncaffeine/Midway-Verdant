#pragma once
#include <array>
#include <functional>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"

namespace mv {

// The 8254 programmable interval timer at ports 0x40-0x43, three channels clocked at clk::kPitHz. Channel 0's
// output is IRQ0; channel 2 is gated by port 0x61 bit 0 and its output reads back on bit 5. Counts are derived
// from the master timeline when read, and the IRQ0 edge is an event on it.
class Pit : public PioHandler {
public:
    Pit(Bus& bus, Scheduler& sched, std::function<void(bool)> irq0);
    std::string_view name() const override { return "pit"; }

    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;
    // The reset line: no channel counting, no edge scheduled.
    void reset();

    // Port 0x61 bits: gate for channel 2 and the speaker data bit.
    void set_gate2(bool on);
    bool gate2() const { return channels_[2].gate; }
    bool out(int channel) const;

    struct Channel {
        u32 reload{0x10000};  // 1..65536 (0 written = 65536)
        u8 mode{0};
        u8 rw{3};  // 1 = LSB, 2 = MSB, 3 = LSB then MSB
        bool bcd{false};
        bool gate{true};
        bool counting{false};
        Cycles load_time{0};
        // access sequencing
        bool write_msb_next{false};
        u16 write_latch{0};
        bool read_msb_next{false};
        bool count_latched{false};
        u16 latched_count{0};
        bool status_latched{false};
        u8 latched_status{0};
        bool null_count{true};
    };
    // The fast mode's lost-tick policy (what KVM's in-kernel PIT and QEMU's lost_tick_policy=delay do): the vCPU
    // and the timeline are not in lockstep, so an IRQ0 edge can arrive while the previous one is still latched and
    // the 8259 would merge them; the guest's tick count (its clock) would then run slow. With a policy set, such an
    // edge is kept in a backlog and raised again each time the PIC acknowledges IRQ0. `still_pending` says whether
    // the previous edge is still latched. Not used by the accurate mode, whose timing is the hardware's.
    void set_lost_tick_policy(std::function<bool()> still_pending) { still_pending_ = std::move(still_pending); }
    void on_irq0_acknowledged();
    static constexpr u32 kMaxTickBacklog = 200;  // ticks; beyond this (a host stall) the oldest are dropped
    u64 ticks_delayed() const { return delayed_; }
    u64 ticks_dropped() const { return dropped_; }
    u32 tick_backlog() const { return backlog_; }
    const Channel& channel(int i) const { return channels_[static_cast<std::size_t>(i)]; }
    u16 current_count(int channel) const;

private:
    u64 ticks_since_load(const Channel& c) const;
    u16 count_at(const Channel& c, Cycles now) const;
    bool out_at(const Channel& c, Cycles now) const;
    Cycles next_irq_edge(const Channel& c) const;  // channel 0 only
    void schedule_irq();
    void pulse_irq0();
    void load_count(int index, u16 value);
    u8 status_byte(const Channel& c) const;

    Scheduler& sched_;
    std::function<void(bool)> irq0_;
    std::array<Channel, 3> channels_{};
    TimerEvent irq_event_;
    std::function<bool()> still_pending_;
    u32 backlog_{0};
    u64 delayed_{0}, dropped_{0};
};

}  // namespace mv
