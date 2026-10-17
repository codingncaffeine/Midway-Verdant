#pragma once
#include "core/bus.h"
#include "flash/flash.h"
#include "pci/pci.h"

namespace mv {

// PCI 0:1.0, the MCPX's LPC/ISA bridge. Configuration register 0x80 bit 1 hides the MCPX boot ROM; register
// 0x8C enables the IDE and network functions; BAR0 is the ACPI power-management I/O block (0x8000 on a
// retail console). The legacy devices behind it (PICs, PIT, RTC, DMA, POST port) are separate objects.
class Lpc : public PciDevice {
public:
    Lpc(Bus& bus, Flash& flash);

    // The PM block follows BAR0 wherever software puts it.
    void attach_pm_block(PioHandler* pm, u32 size);
    bool ide_nic_enabled() const { return (cfg32(0x8C) & 0x40000000u) != 0; }

protected:
    void on_config_write(u8 reg, unsigned size, u32 value) override;
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;
    void on_reset() override;

private:
    Bus& bus_;
    Flash& flash_;
    PioHandler* pm_{nullptr};
    u32 pm_size_{0};
    u32 pm_mapped_at_{0};
};

}  // namespace mv
