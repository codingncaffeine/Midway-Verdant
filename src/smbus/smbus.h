#pragma once
#include <functional>
#include <map>
#include <optional>
#include <string_view>

#include "core/bus.h"
#include "core/scheduler.h"
#include "pci/pci.h"

namespace mv {

// A slave on the SMBus. Returns false / nullopt to NACK (an absent register or an unsupported cycle).
class SmbusDevice {
public:
    virtual ~SmbusDevice() = default;
    virtual std::string_view name() const = 0;
    virtual bool quick(bool read) { (void)read; return true; }
    virtual std::optional<u8> receive_byte() { return std::nullopt; }
    virtual bool send_byte(u8 v) { (void)v; return false; }
    virtual std::optional<u8> read_byte(u8 command) = 0;
    virtual bool write_byte(u8 command, u8 value) = 0;
    virtual std::optional<u16> read_word(u8 command) {
        const auto lo = read_byte(command);
        const auto hi = read_byte(static_cast<u8>(command + 1));
        if (!lo || !hi) return std::nullopt;
        return static_cast<u16>(*lo | (*hi << 8));
    }
    virtual bool write_word(u8 command, u16 value) {
        return write_byte(command, static_cast<u8>(value)) && write_byte(static_cast<u8>(command + 1), static_cast<u8>(value >> 8));
    }
};

// PCI 0:1.1, the MCPX's SMBus host controller (AMD-756 lineage). Its I/O block (BAR1, 0xC000 on a retail
// console): +0 status, +2 control, +4 slave address, +6 data, +8 command. A cycle takes the time it takes on a
// 100 kHz bus: the completion is an event on the timeline, and software polls or takes the interrupt.
class SmbusController : public PciDevice, public PioHandler {
public:
    static constexpr u8 kStatusAbort = 0x01, kStatusCollision = 0x02, kStatusProtocolError = 0x04, kStatusBusy = 0x08,
                        kStatusComplete = 0x10, kStatusTimeout = 0x20;
    static constexpr u8 kControlStart = 0x08, kControlInterrupt = 0x10, kControlAbort = 0x20;

    SmbusController(Bus& bus, Scheduler& sched, std::function<void(bool)> irq);
    std::string_view name() const override { return "smbus"; }

    void attach(u8 address7, SmbusDevice& dev);
    SmbusDevice* device_at(u8 address7) const;

    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;

    struct Transaction {
        u8 address7;
        bool read;
        u8 cycle_type;
        u8 command;
        u16 data;
        bool acked;
    };
    const std::vector<Transaction>& log() const { return log_; }
    u64 transactions() const { return count_; }

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    void start_cycle();
    void complete_cycle();
    void update_irq();

    Bus& bus_;
    Scheduler& sched_;
    std::function<void(bool)> irq_;
    std::map<u8, SmbusDevice*> devices_;
    u32 mapped_at_{0};
    u8 status_{0};
    u8 control_{0};
    u8 address_{0};
    u16 data_{0};
    u8 command_{0};
    bool irq_level_{false};
    TimerEvent completion_;
    Transaction pending_{};
    u16 pending_result_{0};
    std::vector<Transaction> log_;
    u64 count_{0};
};

}  // namespace mv
