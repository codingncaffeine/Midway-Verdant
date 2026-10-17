#pragma once
#include <unordered_map>

#include "core/bus.h"
#include "pci/pci.h"

namespace mv {

// PCI 1:0.0, the NV2A, as a register store: enough identity for the boot chain's probes, every other register
// remembered and read back, every access logged. The real engines replace this block by block in Phase 4.
class Nv2aStub : public PciDevice, public MmioHandler {
public:
    static constexpr u32 kMmioSize = 16u * 1024 * 1024;
    static constexpr u32 kPmcBoot0 = 0x02A000A1;  // chip identity read by firmware [verify]

    explicit Nv2aStub(Bus& bus);
    std::string_view name() const override { return "nv2a"; }

    u32 mmio_read(PhysAddr addr, unsigned size) override;
    void mmio_write(PhysAddr addr, unsigned size, u32 value) override;

    u32 reg(u32 offset) const;
    u64 reads() const { return reads_; }
    u64 writes() const { return writes_; }

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    Bus& bus_;
    u32 mapped_at_{0};
    std::unordered_map<u32, u32> regs_;
    u64 reads_{0}, writes_{0};
};

// A PCI function that exists for the bus scan and stores its configuration, with its regions unmapped: the
// bus logs any access to them. Used for every MCPX function whose model is a later phase.
class PciPlaceholder : public PciDevice {
public:
    PciPlaceholder(std::string_view name, u16 device, u32 class_code, u8 revision, u8 header_type = 0)
        : PciDevice(name, 0x10DE, device, class_code, revision, header_type) {}
};

}  // namespace mv
