#pragma once
#include <array>
#include <functional>

#include "core/bus.h"
#include "core/scheduler.h"

namespace mv {

// MC146818-compatible real-time clock and 128 bytes of CMOS at ports 0x70/0x71. Time advances with the master
// timeline from a base set at reset (the host's clock, or a fixed instant for deterministic runs). Periodic,
// alarm and update-ended interrupts drive IRQ8 as a level that register C's read clears.
class Rtc : public PioHandler {
public:
    Rtc(Bus& bus, Scheduler& sched, std::function<void(bool)> irq8);
    std::string_view name() const override { return "rtc"; }

    // Seconds since 1970-01-01 UTC at timeline zero.
    void set_base_time(i64 unix_seconds);
    i64 base_time() const { return base_seconds_; }
    i64 current_time() const;  // unix seconds at the current timeline position

    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;

    u8 cmos(u8 index) const { return cmos_[index & 0x7F]; }
    void set_cmos(u8 index, u8 v) { cmos_[index & 0x7F] = v; }

private:
    struct Civil {
        int year, month, day, weekday, hour, minute, second;
    };
    static Civil civil_from_unix(i64 t);
    static i64 unix_from_civil(const Civil& c);
    u8 to_reg(int v, bool hour = false) const;
    int from_reg(u8 v, bool hour = false) const;
    void refresh_time_registers();
    void apply_time_registers();
    void update_irq();
    void schedule_periodic();
    void schedule_update();

    Scheduler& sched_;
    std::function<void(bool)> irq8_;
    std::array<u8, 128> cmos_{};
    u8 index_{0};
    i64 base_seconds_{0};
    bool irq_level_{false};
    TimerEvent periodic_event_;
    TimerEvent update_event_;
};

}  // namespace mv
