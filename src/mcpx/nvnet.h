#pragma once
#include <array>
#include <functional>
#include <string_view>

#include "core/bus.h"
#include "pci/pci.h"

// The MCPX's Ethernet controller with no cable plugged in. The kernel's network stack (linked into every title with
// system link or online play) brings it up at start-up: it resets the transmitter and receiver, waits for their
// status bits, reads the PHY's registers through the management interface and asks for the link state. A missing
// device answers every register with all ones, so the management interface never reports its transfer finished;
// this one completes each transfer at once and reports a PHY at address 1 whose link is down. Frames are never
// sent or received. Register layout from the public forcedeth driver's names for this family and xemu's model, read
// for behaviour; the PHY registers are the IEEE 802.3 clause 22 set.
namespace mv {

class NvNet : public PciDevice, public MmioHandler, public PioHandler {
public:
    static constexpr u32 kMmioSize = 1024, kIoSize = 8;
    static constexpr u32 kIrqStatus = 0x000, kIrqMask = 0x004, kSetupReg5 = 0x130, kTxRxControl = 0x144,
                         kMiiStatus = 0x180, kMdioAddr = 0x190, kMdioData = 0x194;
    static constexpr u32 kTxRxIdle = 1u << 3, kTxRxBit1 = 1u << 1;
    static constexpr u32 kSetupReg5Ready = 1u << 31;  // set once the transmitter and receiver are stopped
    static constexpr u32 kMdioBusy = 1u << 15, kMdioWrite = 1u << 10;
    static constexpr u32 kPhyAddress = 1;
    // PHY registers: control, status, identifiers, our advertisement, the link partner's.
    enum PhyReg : u32 { kBmcr = 0, kBmsr = 1, kPhyId1 = 2, kPhyId2 = 3, kAnar = 4, kAnlpar = 5, kPhyRegs = 6 };
    static constexpr u16 kBmsrLinkUp = 1u << 2, kBmsrAutonegDone = 1u << 5;

    NvNet(Bus& bus, std::function<void(bool)> irq);
    std::string_view name() const override { return PciDevice::name(); }
    u32 mmio_read(PhysAddr addr, unsigned size) override;
    void mmio_write(PhysAddr addr, unsigned size, u32 value) override;
    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;
    void reset();
    u16 phy(PhyReg r) const { return phy_[r]; }

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    u32& reg(u32 offset) { return regs_[(offset & (kMmioSize - 1)) / 4]; }
    void update_irq();
    void mdio_transfer();

    Bus& bus_;
    std::function<void(bool)> irq_;
    std::array<u32, kMmioSize / 4> regs_{};
    std::array<u16, kPhyRegs> phy_{};
    PhysAddr mmio_base_{0};
    u16 io_base_{0};
    bool irq_level_{false};
};

}  // namespace mv
