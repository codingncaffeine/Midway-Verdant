#include "mcpx/isa_misc.h"

#include "debug/trace.h"

namespace mv {

PostPort::PostPort(Bus& bus, Scheduler& sched) : sched_(sched) { bus.map_pio(0x80, 16, this); }

u32 PostPort::pio_read(u16 port, unsigned size) {
    (void)size;
    if (port == 0x80) return last_code() < 0 ? 0 : static_cast<u32>(last_code());
    return page_regs_[port & 0x0F];
}

void PostPort::pio_write(u16 port, unsigned size, u32 value) {
    (void)size;
    if (port == 0x80) {
        const u8 code = static_cast<u8>(value);
        history_.push_back({sched_.now(), code});
        trace::log(trace::Cat::Post, "POST {:02x}", code);
        return;
    }
    page_regs_[port & 0x0F] = static_cast<u8>(value);
}

IsaMisc::IsaMisc(Bus& bus, Scheduler& sched, Pit& pit) : sched_(sched), pit_(pit) {
    bus.map_pio(0x00, 16, this);
    bus.map_pio(0x61, 1, this);
    bus.map_pio(0x92, 1, this);
    bus.map_pio(0xC0, 32, this);
    bus.map_pio(0xF0, 2, this);
}

u32 IsaMisc::pio_read(u16 port, unsigned size) {
    (void)size;
    if (port == 0x61) {
        // Bit 4 toggles with the DRAM refresh cycle (~15 us); software spins on it as a crude timer.
        const bool refresh = (ns_from_cycles(sched_.now()) / 15'000ull) & 1u;
        return static_cast<u32>((port61_ & 0x0F) | (refresh ? 0x10 : 0) | (pit_.out(2) ? 0x20 : 0));
    }
    if (port == 0x92) return port92_;
    if (port < 0x10) return dma_lo_[port];
    if (port >= 0xC0 && port < 0xE0) return dma_hi_[port - 0xC0];
    return 0xFF;
}

void IsaMisc::pio_write(u16 port, unsigned size, u32 value) {
    (void)size;
    const u8 v = static_cast<u8>(value);
    if (port == 0x61) {
        port61_ = static_cast<u8>(v & 0x0F);
        pit_.set_gate2((v & 1) != 0);
        return;
    }
    if (port == 0x92) {
        port92_ = static_cast<u8>(v & 0x02);  // A20 bit; the fast-reset bit is not acted on
        return;
    }
    if (port < 0x10) {
        dma_lo_[port] = v;
        return;
    }
    if (port >= 0xC0 && port < 0xE0) {
        dma_hi_[port - 0xC0] = v;
        return;
    }
    // 0xF0/0xF1: coprocessor error clear — nothing to do without IRQ13 users.
}

PmBlock::PmBlock(Scheduler& sched, std::function<void(u16)> on_sleep) : sched_(sched), on_sleep_(std::move(on_sleep)) {}

u32 PmBlock::timer_value() const { return static_cast<u32>(ticks_from_cycles(sched_.now(), clk::kPmTimerHz)); }

u32 PmBlock::pio_read(u16 port, unsigned size) {
    const u16 off = static_cast<u16>(port - base_);
    if (off == 0x08 && size == 4) return timer_value();
    if (off >= kSize) return 0xFFFFFFFFu;
    u32 v = 0;
    for (unsigned i = 0; i < size && off + i < kSize; ++i) v |= static_cast<u32>(regs_[off + i]) << (8 * i);
    if (off != 0x00 && off != 0x02 && off != 0x04 && off != 0x08) {
        trace::log(trace::Cat::Pio, "pm read  +{:02x} size {} -> {:0{}x}", off, size, v, size * 2);
    }
    return v;
}

void PmBlock::pio_write(u16 port, unsigned size, u32 value) {
    const u16 off = static_cast<u16>(port - base_);
    if (off >= kSize) return;
    if (off == 0x00) {
        // PM1 status: write one to clear.
        for (unsigned i = 0; i < size && off + i < kSize; ++i)
            regs_[off + i] = static_cast<u8>(regs_[off + i] & ~static_cast<u8>(value >> (8 * i)));
        return;
    }
    for (unsigned i = 0; i < size && off + i < kSize; ++i) regs_[off + i] = static_cast<u8>(value >> (8 * i));
    if (off == 0x04 || (off < 0x04 && off + size > 0x04)) {
        const u16 cnt = static_cast<u16>(regs_[0x04] | (regs_[0x05] << 8));
        if (cnt & 0x2000) {  // SLP_EN
            const u16 type = static_cast<u16>((cnt >> 10) & 7);
            trace::log(trace::Cat::Boot, "ACPI sleep requested, SLP_TYP {}", type);
            regs_[0x05] = static_cast<u8>(regs_[0x05] & ~0x20u);
            if (on_sleep_) on_sleep_(type);
        }
        return;
    }
    if (off != 0x02) trace::log(trace::Cat::Pio, "pm write +{:02x} size {} <- {:0{}x}", off, size, value, size * 2);
}

}  // namespace mv
