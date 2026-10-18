#include <gtest/gtest.h>

#include "core/bus.h"
#include "mcpx/nvnet.h"

using namespace mv;

namespace {
struct Rig {
    Bus bus{4096};
    bool irq = false;
    NvNet nic{bus, [this](bool level) { irq = level; }};
    static constexpr PhysAddr kBase = 0xFEF00000;
    Rig() {
        nic.config_write(0x10, 4, kBase);
        nic.config_write(0x04, 2, 0x0007);
    }
    u32 read(u32 off) { return bus.read(kBase + off, 4); }
    void write(u32 off, u32 v) { bus.write(kBase + off, 4, v); }
    u32 mdio_read(u32 phy, u32 reg) {
        write(NvNet::kMdioAddr, NvNet::kMdioBusy | (phy << 5) | reg);
        EXPECT_EQ(read(NvNet::kMdioAddr) & NvNet::kMdioBusy, 0u) << "a management transfer finishes at once";
        return read(NvNet::kMdioData);
    }
};
}  // namespace

TEST(NvNet, ThePhyAnswersAtAddressOneWithTheLinkDown) {
    Rig r;
    const u32 bmsr = r.mdio_read(1, NvNet::kBmsr);
    EXPECT_EQ(bmsr & NvNet::kBmsrLinkUp, 0u) << "no cable";
    EXPECT_EQ(bmsr & NvNet::kBmsrAutonegDone, 0u);
    EXPECT_NE(bmsr & 0x0008, 0u) << "able to auto-negotiate";
    EXPECT_EQ(r.mdio_read(2, NvNet::kBmsr), 0xFFFFu) << "no PHY at other addresses";
    // A write reaches the PHY's control register; the status register is read-only.
    r.write(NvNet::kMdioData, 0x3100);
    r.write(NvNet::kMdioAddr, NvNet::kMdioBusy | NvNet::kMdioWrite | (1u << 5) | NvNet::kBmcr);
    EXPECT_EQ(r.read(NvNet::kMdioAddr) & NvNet::kMdioBusy, 0u);
    EXPECT_EQ(r.mdio_read(1, NvNet::kBmcr), 0x3100u);
}

TEST(NvNet, InterruptStatusIsWriteOneToClearAndDrivesTheLine) {
    Rig r;
    EXPECT_EQ(r.read(NvNet::kIrqStatus), 0u) << "nothing pending after reset (a missing device reads all ones)";
    r.write(NvNet::kTxRxControl, 0);
    EXPECT_NE(r.read(NvNet::kSetupReg5) & NvNet::kSetupReg5Ready, 0u) << "stopped engines are reported";
    EXPECT_NE(r.read(NvNet::kTxRxControl) & NvNet::kTxRxIdle, 0u);
    EXPECT_FALSE(r.irq);
    r.write(NvNet::kIrqMask, 0xFF);
    EXPECT_FALSE(r.irq);
}
