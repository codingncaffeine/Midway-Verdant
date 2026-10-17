#include "mcpx/pit.h"

#include "debug/trace.h"

namespace mv {

Pit::Pit(Bus& bus, Scheduler& sched, std::function<void(bool)> irq0) : sched_(sched), irq0_(std::move(irq0)) {
    bus.map_pio(0x40, 4, this);
    irq_event_.name = "pit.irq0";
    irq_event_.fire = [this](Cycles) {
        // A rising edge on IRQ0: pulse the line (the PIC latches the edge).
        if (irq0_) {
            irq0_(true);
            irq0_(false);
        }
        schedule_irq();
    };
    for (auto& c : channels_) c.gate = true;
    channels_[2].gate = false;
}

void Pit::reset() {
    sched_.cancel(irq_event_);
    channels_ = {};
    for (auto& c : channels_) c.gate = true;
    channels_[2].gate = false;
}

u64 Pit::ticks_since_load(const Channel& c) const {
    const Cycles now = sched_.now();
    return now > c.load_time ? ticks_from_cycles(now - c.load_time, clk::kPitHz) : 0;
}

u16 Pit::count_at(const Channel& c, Cycles now) const {
    if (!c.counting) return static_cast<u16>(c.reload & 0xFFFF);
    const u64 elapsed = now > c.load_time ? ticks_from_cycles(now - c.load_time, clk::kPitHz) : 0;
    const u64 reload = c.reload;
    switch (c.mode) {
        case 0:
        case 1:
        case 4:
        case 5: {
            // Counts down from reload through zero and wraps at 65536.
            const u64 v = (reload - elapsed) & 0xFFFFu;
            return static_cast<u16>(v);
        }
        case 2:
        case 6: {
            const u64 v = reload - (elapsed % reload);
            return static_cast<u16>(v & 0xFFFF);
        }
        case 3:
        case 7:
        default: {
            // Square wave: decrements by two per clock; the second half starts at reload again.
            const u64 e = (elapsed * 2) % reload;
            const u64 v = reload - e;
            return static_cast<u16>(v & 0xFFFF);
        }
    }
}

bool Pit::out_at(const Channel& c, Cycles now) const {
    const u64 elapsed = now > c.load_time ? ticks_from_cycles(now - c.load_time, clk::kPitHz) : 0;
    const u64 reload = c.reload;
    switch (c.mode) {
        case 0: return c.counting && elapsed >= reload;
        case 1: return !(c.counting && elapsed < reload);
        case 2:
        case 6: return !c.counting || (elapsed % reload) != reload - 1;
        case 3:
        case 7: return !c.counting || (elapsed % reload) < (reload + 1) / 2;
        case 4:
        case 5: return !(c.counting && elapsed == reload);
        default: return true;
    }
}

bool Pit::out(int channel) const { return out_at(channels_[static_cast<std::size_t>(channel)], sched_.now()); }

u16 Pit::current_count(int channel) const { return count_at(channels_[static_cast<std::size_t>(channel)], sched_.now()); }

Cycles Pit::next_irq_edge(const Channel& c) const {
    if (!c.counting || !c.gate) return ~Cycles{0};
    const u64 elapsed = ticks_since_load(c);
    const u64 reload = c.reload;
    u64 next_ticks;
    switch (c.mode) {
        case 0:
            if (elapsed >= reload) return ~Cycles{0};
            next_ticks = reload;
            break;
        case 2:
        case 6:
        case 3:
        case 7:
            next_ticks = (elapsed / reload + 1) * reload;
            break;
        default:
            return ~Cycles{0};
    }
    return c.load_time + cycles_from_ticks_ceil(next_ticks, clk::kPitHz);
}

void Pit::schedule_irq() {
    const Cycles when = next_irq_edge(channels_[0]);
    if (when == ~Cycles{0}) {
        sched_.cancel(irq_event_);
    } else {
        sched_.schedule(irq_event_, when);
    }
}

void Pit::load_count(int index, u16 value) {
    Channel& c = channels_[static_cast<std::size_t>(index)];
    c.reload = value == 0 ? 0x10000u : value;
    c.load_time = sched_.now() + cycles_from_ticks(1, clk::kPitHz);  // the new count takes effect on the next clock
    c.counting = true;
    c.null_count = false;
    trace::log(trace::Cat::Timer, "pit ch{} mode {} reload {} ({:.3f} ms)", index, c.mode, c.reload,
               static_cast<double>(c.reload) * 1000.0 / static_cast<double>(clk::kPitHz));
    if (index == 0) schedule_irq();
}

u8 Pit::status_byte(const Channel& c) const {
    return static_cast<u8>((out_at(c, sched_.now()) ? 0x80 : 0) | (c.null_count ? 0x40 : 0) | (c.rw << 4) | (c.mode << 1) |
                           (c.bcd ? 1 : 0));
}

void Pit::set_gate2(bool on) {
    Channel& c = channels_[2];
    if (c.gate == on) return;
    c.gate = on;
    if (on && (c.mode == 2 || c.mode == 3)) c.load_time = sched_.now();  // gate rising reloads in modes 2/3
}

u32 Pit::pio_read(u16 port, unsigned size) {
    (void)size;
    const int index = port & 3;
    if (index == 3) return 0xFF;
    Channel& c = channels_[static_cast<std::size_t>(index)];
    if (c.status_latched) {
        c.status_latched = false;
        return c.latched_status;
    }
    u16 value = c.count_latched ? c.latched_count : count_at(c, sched_.now());
    switch (c.rw) {
        case 1:
            c.count_latched = false;
            return value & 0xFF;
        case 2:
            c.count_latched = false;
            return value >> 8;
        default:
            if (c.read_msb_next) {
                c.read_msb_next = false;
                c.count_latched = false;
                return value >> 8;
            }
            c.read_msb_next = true;
            return value & 0xFF;
    }
}

void Pit::pio_write(u16 port, unsigned size, u32 value) {
    (void)size;
    const u8 v = static_cast<u8>(value);
    const int index = port & 3;
    if (index == 3) {
        const int sel = (v >> 6) & 3;
        if (sel == 3) {
            // Read-back command: latch count and/or status of the selected channels.
            for (int ch = 0; ch < 3; ++ch) {
                if (!(v & (2 << ch))) continue;
                Channel& c = channels_[static_cast<std::size_t>(ch)];
                if (!(v & 0x20) && !c.count_latched) {
                    c.count_latched = true;
                    c.latched_count = count_at(c, sched_.now());
                }
                if (!(v & 0x10) && !c.status_latched) {
                    c.status_latched = true;
                    c.latched_status = status_byte(c);
                }
            }
            return;
        }
        Channel& c = channels_[static_cast<std::size_t>(sel)];
        const u8 rw = (v >> 4) & 3;
        if (rw == 0) {
            if (!c.count_latched) {
                c.count_latched = true;
                c.latched_count = count_at(c, sched_.now());
            }
            return;
        }
        c.rw = rw;
        c.mode = (v >> 1) & 7;
        c.bcd = (v & 1) != 0;
        c.write_msb_next = false;
        c.read_msb_next = false;
        c.count_latched = false;
        c.status_latched = false;
        c.counting = false;
        c.null_count = true;
        if (sel == 0) schedule_irq();
        return;
    }
    Channel& c = channels_[static_cast<std::size_t>(index)];
    switch (c.rw) {
        case 1: load_count(index, v); break;
        case 2: load_count(index, static_cast<u16>(v << 8)); break;
        default:
            if (c.write_msb_next) {
                c.write_msb_next = false;
                load_count(index, static_cast<u16>(c.write_latch | (v << 8)));
            } else {
                c.write_latch = v;
                c.write_msb_next = true;
                c.counting = false;  // writing the LSB stops the count until the MSB arrives
                if (index == 0) schedule_irq();
            }
            break;
    }
}

}  // namespace mv
