#pragma once
#include <array>
#include <functional>
#include <vector>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"
#include "mcpx/pit.h"

namespace mv {

// Port 0x80: the POST code port, the boot chain's progress instrument. Also stores the DMA page registers that
// share the 0x80-0x8F range.
class PostPort : public PioHandler {
public:
    struct Entry {
        Cycles when;
        u8 code;
    };
    PostPort(Bus& bus, Scheduler& sched);
    std::string_view name() const override { return "post"; }
    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;
    const std::vector<Entry>& history() const { return history_; }
    int last_code() const { return history_.empty() ? -1 : history_.back().code; }

private:
    Scheduler& sched_;
    std::array<u8, 16> page_regs_{};
    std::vector<Entry> history_;
};

// The legacy odds and ends: port 0x61 (timer 2 gate, speaker, refresh toggle, timer 2 output), 0x92 (port A /
// A20), 0xF0/0xF1 (coprocessor error), the 8237 DMA registers (stored, no transfers: nothing on the console
// uses ISA DMA).
class IsaMisc : public PioHandler {
public:
    IsaMisc(Bus& bus, Scheduler& sched, Pit& pit);
    std::string_view name() const override { return "isa"; }
    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;

private:
    Scheduler& sched_;
    Pit& pit_;
    u8 port61_{0};
    u8 port92_{0};
    std::array<u8, 32> dma_lo_{};
    std::array<u8, 32> dma_hi_{};
};

// The ACPI power-management block the LPC bridge's BAR0 points at (0x8000 on a retail console): PM1 status,
// enable and control, the 3.579545 MHz PM timer, general-purpose events, and the MCPX's own registers in the
// same block (stored and logged until each is understood).
class PmBlock : public PioHandler {
public:
    static constexpr u32 kSize = 256;
    PmBlock(Scheduler& sched, std::function<void(u16 sleep_type)> on_sleep);
    std::string_view name() const override { return "pm"; }
    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;
    u32 timer_value() const;
    void set_base(u16 base) { base_ = base; }

private:
    Scheduler& sched_;
    std::function<void(u16)> on_sleep_;
    u16 base_{0x8000};
    std::array<u8, kSize> regs_{};
};

}  // namespace mv
