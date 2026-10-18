#pragma once
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/bus.h"
#include "core/scheduler.h"
#include "ide/disk_image.h"
#include "pci/pci.h"

namespace mv {

class IdeChannel;

// One device on the IDE cable (the console has a hard disk as master and a DVD drive as slave on one channel).
// The channel delivers task-file register writes and commands; the device fills the status and the data buffer
// and asks the channel to raise its interrupt. Timing is the channel's: completions are scheduled, not instant.
class IdeDevice {
public:
    virtual ~IdeDevice() = default;
    virtual std::string_view name() const = 0;
    virtual bool is_atapi() const = 0;
    // Software reset (SRST) state: signature registers, status; a device's security state survives it.
    virtual void reset() = 0;
    // Power-on or hardware reset: everything, including the security state (a locked drive locks again).
    virtual void hardware_reset() { reset(); }
    // A command byte was written; returns the time until the device's first response.
    virtual Cycles start_command(u8 command) = 0;
    // The scheduled continuation of the command (data ready, completion).
    virtual void continue_command() = 0;
    // 16-bit data port traffic while DRQ is set.
    virtual u16 read_data() = 0;
    virtual void write_data(u16 value) = 0;
    // Task file registers the channel keeps for both devices are passed in by reference.
    struct TaskFile {
        u8 features{0}, error{0}, sector_count{0}, lba_low{0}, lba_mid{0}, lba_high{0}, device{0}, status{0};
        u8 hob_features{0}, hob_sector_count{0}, hob_lba_low{0}, hob_lba_mid{0}, hob_lba_high{0};
    };
    TaskFile& tf() { return tf_; }
    const TaskFile& tf() const { return tf_; }
    void attach(IdeChannel* ch) { channel_ = ch; }

protected:
    IdeChannel* channel_{nullptr};
    TaskFile tf_;
};

// Status register bits.
namespace ata {
constexpr u8 kStatusErr = 0x01, kStatusDrq = 0x08, kStatusDsc = 0x10, kStatusDf = 0x20, kStatusDrdy = 0x40, kStatusBsy = 0x80;
constexpr u8 kErrorAbrt = 0x04, kErrorIdnf = 0x10, kErrorUnc = 0x40;
}  // namespace ata

// The MCPX IDE function (PCI 0:9.0) with its primary channel at 0x1F0-0x1F7 / 0x3F6 and the bus-master
// registers at BAR4 (0xFF60 on a retail console). Only the primary channel is wired on the console.
class IdeController : public PciDevice, public PioHandler {
public:
    IdeController(Bus& bus, Scheduler& sched, std::function<void(bool)> irq14);
    std::string_view name() const override { return "ide"; }

    void attach(int position, std::unique_ptr<IdeDevice> dev);  // 0 master, 1 slave
    IdeChannel& primary() { return *primary_; }
    // The reset line reaches the drives: a hardware reset, so a locked disk locks again.
    void reset();

    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    Bus& bus_;
    std::unique_ptr<IdeChannel> primary_;
    u32 busmaster_at_{0};
};

// One channel: the task file shared by its two devices, the device select, interrupt line, PIO data path and
// the bus-master DMA engine (PRD table, command, status).
class IdeChannel {
public:
    IdeChannel(Bus& bus, Scheduler& sched, std::function<void(bool)> irq);
    void attach(int position, std::unique_ptr<IdeDevice> dev);
    IdeDevice* selected() const;
    IdeDevice* device(int position) const { return devices_[static_cast<std::size_t>(position)].get(); }

    u32 read_task(u16 reg, unsigned size);       // 0x1F0 + reg
    void write_task(u16 reg, unsigned size, u32 value);
    static void latch_task(IdeDevice::TaskFile& tf, u16 reg, u8 v);
    u32 read_control(u16 reg);                   // 0x3F6 + reg
    void write_control(u16 reg, u32 value);
    u32 read_busmaster(u16 reg, unsigned size);  // BAR4 + reg
    void write_busmaster(u16 reg, unsigned size, u32 value);

    // Device-side services.
    void raise_interrupt();
    void schedule(IdeDevice& dev, Cycles delay);
    Bus& bus() { return bus_; }
    Scheduler& scheduler() { return sched_; }
    // DMA: move `len` bytes between the device buffer and the PRD list; true when the whole transfer went through.
    bool dma_transfer(const u8* to_memory, u8* from_memory, u32 len);
    bool dma_active() const { return (bm_command_ & 1u) != 0; }
    bool dma_is_write_to_memory() const { return (bm_command_ & 8u) != 0; }
    void dma_complete(bool ok);
    void software_reset();
    void hardware_reset();

    u64 commands() const { return commands_; }
    // Instruments: the last command byte each device received, and the bus master's registers.
    u8 last_command(int position) const { return last_command_[static_cast<std::size_t>(position)]; }
    u8 bm_command() const { return bm_command_; }
    u8 bm_status() const { return bm_status_; }
    u64 sectors_read() const { return sectors_read_; }
    u64 sectors_written() const { return sectors_written_; }
    void count_sectors(bool write, u64 n) { (write ? sectors_written_ : sectors_read_) += n; }

private:
    Bus& bus_;
    Scheduler& sched_;
    std::function<void(bool)> irq_;
    std::array<std::unique_ptr<IdeDevice>, 2> devices_{};
    int selected_{0};
    bool irq_level_{false};
    bool nien_{false};
    bool hob_{false};
    u8 bm_command_{0}, bm_status_{0};
    u32 bm_prdt_{0};
    TimerEvent events_[2];
    u64 commands_{0}, sectors_read_{0}, sectors_written_{0};
    std::array<u8, 2> last_command_{};
    IdeDevice::TaskFile empty_tf_;
};

}  // namespace mv
