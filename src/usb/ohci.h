#pragma once
#include <array>
#include <functional>
#include <string_view>

#include "core/bus.h"
#include "core/scheduler.h"
#include "pci/pci.h"
#include "usb/usb_device.h"

// An OpenHCI USB host controller (the MCPX's two, PCI 0:2.0 and 0:3.0) with its root hub. Every millisecond
// while operational it starts a frame: the frame number into the HCCA, the interrupt endpoints of this frame's
// periodic list, then the control and bulk lists; finished transfer descriptors go onto the done queue, written
// back to the HCCA as their delay interrupt asks. Isochronous endpoints are skipped (nothing on the console
// uses them). Transactions run whole within a frame, not bit-timed. Written from the OpenHCI 1.0a specification
// (chapters 4, 6, 7); see local-notes/hw/usb.md.
namespace mv {

class Ohci : public PciDevice, public MmioHandler {
public:
    static constexpr unsigned kPorts = 4;
    static constexpr u32 kMmioSize = 4096;

    Ohci(std::string_view name, Bus& bus, Scheduler& sched, u8 irq_line, std::function<void(bool)> irq);
    std::string_view name() const override { return PciDevice::name(); }
    u32 mmio_read(PhysAddr addr, unsigned size) override;
    void mmio_write(PhysAddr addr, unsigned size, u32 value) override;

    // Plugs a device into a root-hub port (1-based) before power-on; nullptr leaves it empty.
    void attach(unsigned port, usb::Device* device);
    // The reset line: registers at their hardware-reset values, USBRESET, the root hub's ports re-detected.
    void reset();

    struct Stats {
        u64 frames{0}, transfers{0}, naks{0}, stalls{0}, no_device{0}, writebacks{0};
    };
    const Stats& stats() const { return stats_; }
    u32 control() const { return control_; }

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    enum class Visit { Empty, Busy };  // whether an ED still holds transfer descriptors after its visit
    u32 read_reg(u32 offset);
    void write_reg(u32 offset, u32 value);
    void software_reset();
    void set_functional_state(u32 hcfs);
    void start_frame();
    bool walk_list(u32 head, bool periodic);
    Visit visit_ed(u32 ed_address, bool periodic);
    usb::Device* device_at(u8 address);
    u32 buffer_address(u32 cbp, u32 be, u32 offset) const;
    void port_write(unsigned port, u32 value);
    void port_connect(unsigned port);
    void update_irq();
    u32 rd32(u32 addr) const;
    void wr32(u32 addr, u32 v);

    Bus& bus_;
    Scheduler& sched_;
    std::function<void(bool)> irq_;
    u32 mmio_at_{0};
    TimerEvent frame_;
    Cycles frame_started_{0};

    u32 control_{0}, command_status_{0}, int_status_{0}, int_enable_{0}, hcca_{0};
    u32 period_current_{0}, control_head_{0}, control_current_{0}, bulk_head_{0}, bulk_current_{0}, done_head_{0};
    u32 fm_interval_{0x2EDF}, fm_number_{0}, periodic_start_{0}, ls_threshold_{0x628};
    u32 rh_a_{0}, rh_b_{0}, rh_status_{0};
    bool frame_toggle_{false};
    u32 done_counter_{7};
    std::array<u32, kPorts> port_status_{};
    std::array<usb::Device*, kPorts> ports_{};
    bool irq_level_{false};
    Stats stats_{};
};

}  // namespace mv
