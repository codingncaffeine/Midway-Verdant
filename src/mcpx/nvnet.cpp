#include "mcpx/nvnet.h"

#include "debug/trace.h"

namespace mv {

NvNet::NvNet(Bus& bus, std::function<void(bool)> irq)
    : PciDevice("nvnet", 0x10DE, 0x01C3, 0x020000, 0xB1), bus_(bus), irq_(std::move(irq)) {
    define_bar(0, BarKind::Memory, kMmioSize);
    define_bar(1, BarKind::Io, kIoSize);
    set_cfg8(0x3C, 4);  // interrupt line 4, pin A
    set_cfg8(0x3D, 1);
    reset();
}

void NvNet::reset() {
    regs_.fill(0);
    reg(kTxRxControl) = kTxRxIdle;
    // Auto-negotiation enabled at full duplex; able to negotiate 10 and 100 Mbit/s at either duplex, but with no cable
    // the link is down and negotiation never completes, and there is no link partner to report.
    phy_ = {};
    phy_[kBmcr] = 0x1100;
    phy_[kBmsr] = 0x7809;
    phy_[kAnar] = 0x01E1;
    irq_level_ = false;
    if (irq_) irq_(false);
}

void NvNet::on_bar_changed(int index, u32 old_base, u32 new_base) {
    if (index == 0) {
        if (old_base) bus_.unmap_mmio(old_base, kMmioSize);
        mmio_base_ = new_base;
        if (new_base) bus_.map_mmio(new_base, kMmioSize, this);
    } else if (index == 1) {
        if (old_base) bus_.unmap_pio(static_cast<u16>(old_base), kIoSize);
        io_base_ = static_cast<u16>(new_base);
        if (new_base) bus_.map_pio(io_base_, kIoSize, this);
    }
}

void NvNet::update_irq() {
    const bool level = (reg(kIrqStatus) & reg(kIrqMask)) != 0;
    if (level == irq_level_) return;
    irq_level_ = level;
    if (irq_) irq_(level);
}

// A management transfer finishes in the same access that starts it: a read fills the data register, a write takes
// it. Only the PHY at address 1 answers; the bus floats high for every other address.
void NvNet::mdio_transfer() {
    u32& addr = reg(kMdioAddr);
    const u32 phy_address = (addr >> 5) & 0x1F, phy_reg = addr & 0x1F;
    if (addr & kMdioWrite) {
        if (phy_address == kPhyAddress && phy_reg < kPhyRegs && phy_reg != kBmsr)
            phy_[phy_reg] = static_cast<u16>(reg(kMdioData));
    } else {
        reg(kMdioData) = (phy_address == kPhyAddress) ? (phy_reg < kPhyRegs ? phy_[phy_reg] : 0u) : 0xFFFFu;
    }
    trace::log(trace::Cat::Pci, "nvnet: mdio {} phy {} reg {} = {:04x}", (addr & kMdioWrite) ? "write" : "read",
               phy_address, phy_reg, reg(kMdioData) & 0xFFFF);
    addr &= ~kMdioBusy;
}

u32 NvNet::mmio_read(PhysAddr addr, unsigned size) {
    const u32 offset = static_cast<u32>(addr - mmio_base_) & (kMmioSize - 1);
    const u32 word = reg(offset & ~3u);
    const unsigned shift = (offset & 3) * 8;
    const u32 v = size >= 4 ? word : (word >> shift) & ((1u << (size * 8)) - 1);
    return v;
}

void NvNet::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 offset = static_cast<u32>(addr - mmio_base_) & (kMmioSize - 1) & ~3u;
    if (size < 4) {
        const unsigned shift = ((static_cast<u32>(addr - mmio_base_)) & 3) * 8;
        const u32 mask = ((1u << (size * 8)) - 1) << shift;
        value = (reg(offset) & ~mask) | ((value << shift) & mask);
    }
    switch (offset) {
        case kIrqStatus:
        case kMiiStatus:  // write one to clear
            reg(offset) &= ~value;
            update_irq();
            break;
        case kIrqMask:
            reg(offset) = value;
            update_irq();
            break;
        case kMdioAddr:
            reg(offset) = value;
            mdio_transfer();
            break;
        case kTxRxControl:
            // The idle bit is the hardware's to report; with the link down nothing is ever transmitted, so it stays
            // set. Stopping both engines (a write of zero) reports them stopped through the setup register's top bit.
            reg(offset) = (value & ~kTxRxIdle) | kTxRxIdle;
            if (value & kTxRxBit1) {
                reg(kIrqStatus) = 0;
                update_irq();
            } else if (value == 0) {
                reg(kSetupReg5) |= kSetupReg5Ready;
            }
            break;
        default:
            reg(offset) = value;
            break;
    }
}

u32 NvNet::pio_read(u16 port, unsigned size) {
    (void)port;
    (void)size;
    return 0;
}

void NvNet::pio_write(u16 port, unsigned size, u32 value) {
    (void)port;
    (void)size;
    (void)value;
}

}  // namespace mv
