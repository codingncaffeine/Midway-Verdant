#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "core/bus.h"
#if defined(__linux__)
#include "cpu/kvm/coalesced_ring.h"
#endif
#include "flash/flash.h"
#include "pci/pci.h"

using namespace mv;

namespace {
struct Recorder : MmioHandler, PioHandler {
    std::vector<std::tuple<PhysAddr, unsigned, u32, bool>> mmio;
    std::vector<std::tuple<u16, unsigned, u32, bool>> pio;
    std::string_view name() const override { return "rec"; }
    u32 mmio_read(PhysAddr a, unsigned s) override {
        mmio.emplace_back(a, s, 0, false);
        return 0x11223344u;
    }
    void mmio_write(PhysAddr a, unsigned s, u32 v) override { mmio.emplace_back(a, s, v, true); }
    u32 pio_read(u16 p, unsigned s) override {
        pio.emplace_back(p, s, 0, false);
        return 0xABu;
    }
    void pio_write(u16 p, unsigned s, u32 v) override { pio.emplace_back(p, s, v, true); }
};
}  // namespace

TEST(Bus, RamIsReachableDirectlyAndThroughTheGpuAlias) {
    Bus bus(1u << 20);
    bus.write(0x1000, 4, 0xDEADBEEF);
    EXPECT_EQ(bus.read(0x1000, 4), 0xDEADBEEFu);
    EXPECT_EQ(bus.read(0x1000, 1), 0xEFu);
    EXPECT_EQ(bus.read(0x1002, 2), 0xDEADu);
    EXPECT_EQ(bus.read(Bus::kRamAliasBase + 0x1000, 4), 0xDEADBEEFu);
    bus.write(Bus::kRamAliasBase + 0x1004, 2, 0x1234);
    EXPECT_EQ(bus.read(0x1004, 2), 0x1234u);
    // Beyond RAM the alias is open bus.
    EXPECT_EQ(bus.read(Bus::kRamAliasBase + (1u << 20), 4), 0xFFFFFFFFu);
}

TEST(Bus, MmioDispatchesByPageAndUnknownAccessesAreCounted) {
    Bus bus(1u << 20);
    Recorder r;
    bus.map_mmio(0xFD000000, 0x2000, &r);
    EXPECT_EQ(bus.read(0xFD001FFC, 4), 0x11223344u);
    bus.write(0xFD000010, 1, 0x7F);
    ASSERT_EQ(r.mmio.size(), 2u);
    EXPECT_EQ(std::get<0>(r.mmio[1]), 0xFD000010u);
    EXPECT_TRUE(std::get<3>(r.mmio[1]));
    EXPECT_EQ(bus.read(0xFD002000, 4), 0xFFFFFFFFu);  // page after the mapping
    bus.write(0xFD002004, 4, 1);
    const auto unknown = bus.unknown_mmio();
    ASSERT_EQ(unknown.size(), 2u);
    EXPECT_EQ(unknown[0].page, 0xFD002000u);
    bus.unmap_mmio(0xFD000000, 0x2000);
    EXPECT_EQ(bus.read(0xFD000000, 4), 0xFFFFFFFFu);
}

TEST(Bus, PortDispatchAndSizes) {
    Bus bus(1u << 16);
    Recorder r;
    bus.map_pio(0x40, 4, &r);
    EXPECT_EQ(bus.pio_read(0x43, 1), 0xABu);
    bus.pio_write(0x40, 2, 0x1234);
    EXPECT_EQ(bus.pio_read(0x44, 1), 0xFFu);  // unmapped
    ASSERT_EQ(r.pio.size(), 2u);
    EXPECT_EQ(std::get<0>(r.pio[1]), 0x40);
    EXPECT_EQ(std::get<1>(r.pio[1]), 2u);
    EXPECT_EQ(bus.unknown_pio().size(), 1u);
}

TEST(Pci, ConfigMechanismOneReachesDevicesAndBarsSizeThemselves) {
    Bus bus(1u << 16);
    PciBus pci(bus);
    PciDevice dev("dev", 0x10DE, 0x01B4, 0x0C0500, 0xB1);
    dev.define_bar(1, PciDevice::BarKind::Io, 16);
    dev.define_bar(2, PciDevice::BarKind::Memory, 0x1000, true);
    pci.add({0, 1, 1}, dev);
    // Vendor/device through the ports.
    bus.pio_write(0xCF8, 4, 0x80000900);  // bus 0 dev 1 fn 1 reg 0
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0x01B410DEu);
    EXPECT_EQ(bus.pio_read(0xCFE, 2), 0x01B4u);  // byte offset through the port
    // Class/revision at 0x08.
    bus.pio_write(0xCF8, 4, 0x80000908);
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0x0C0500B1u);
    // Identification is read-only.
    bus.pio_write(0xCF8, 4, 0x80000900);
    bus.pio_write(0xCFC, 4, 0x12345678);
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0x01B410DEu);
    // BAR1: I/O, 16 bytes. Probe with all ones, then program.
    bus.pio_write(0xCF8, 4, 0x80000914);
    bus.pio_write(0xCFC, 4, 0xFFFFFFFF);
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0x0000FFF1u);
    bus.pio_write(0xCFC, 4, 0xC001);
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0xC001u);
    EXPECT_EQ(dev.bar_base(1), 0xC000u);
    // BAR2: memory, 4 KiB, prefetchable.
    bus.pio_write(0xCF8, 4, 0x80000918);
    bus.pio_write(0xCFC, 4, 0xFFFFFFFF);
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0xFFFFF008u);
    bus.pio_write(0xCFC, 4, 0xFEC00000);
    EXPECT_EQ(dev.bar_base(2), 0xFEC00000u);
    // An unimplemented BAR reads zero whatever is written.
    bus.pio_write(0xCF8, 4, 0x80000910);
    bus.pio_write(0xCFC, 4, 0xFFFFFFFF);
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0u);
    // Absent devices and the two functions that freeze hardware answer all ones.
    bus.pio_write(0xCF8, 4, 0x80000100);  // 0:0.1
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0xFFFFFFFFu);
    bus.pio_write(0xCF8, 4, 0x80004800);  // 0:9.0 not registered here
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0xFFFFFFFFu);
    // The enable bit gates the data port.
    bus.pio_write(0xCF8, 4, 0x00000900);
    EXPECT_EQ(bus.pio_read(0xCFC, 4), 0xFFFFFFFFu);
}

TEST(Flash, ImageIsMirroredAndTheRomOverlaysUntilHidden) {
    Flash flash;
    std::vector<u8> image(256u * 1024, 0);
    for (std::size_t i = 0; i < image.size(); ++i) image[i] = static_cast<u8>(i * 7);
    std::string error;
    ASSERT_TRUE(flash.load_image(image, error)) << error;
    std::array<u8, 512> rom{};
    for (std::size_t i = 0; i < rom.size(); ++i) rom[i] = static_cast<u8>(0xA0 + (i & 0x0F));
    ASSERT_TRUE(flash.load_mcpx_rom(rom));
    // Mirrored 64 times through the window.
    EXPECT_EQ(flash.mmio_read(0xFF000000, 1), image[0]);
    EXPECT_EQ(flash.mmio_read(0xFF040000 + 5, 1), image[5]);
    EXPECT_EQ(flash.mmio_read(0xFFFC0000 + 100, 4), flash.mmio_read(0xFF000000 + 100, 4));
    // The overlay: the top 512 bytes are the ROM, the byte before them is flash.
    EXPECT_TRUE(flash.rom_visible());
    EXPECT_EQ(flash.mmio_read(0xFFFFFE00, 1), 0xA0u);
    EXPECT_EQ(flash.mmio_read(0xFFFFFFF0, 1), 0xA0u);
    EXPECT_EQ(flash.mmio_read(0xFFFFFDFF, 1), image[0x3FDFF]);
    EXPECT_EQ(flash.overlay_page()[0xE00], 0xA0u);
    EXPECT_EQ(flash.overlay_page()[0xDFF], image[0x3FDFF]);
    int hidden_calls = 0;
    flash.add_on_rom_hidden([&] { ++hidden_calls; });
    flash.hide_rom();
    flash.hide_rom();  // idempotent
    EXPECT_EQ(hidden_calls, 1);
    EXPECT_FALSE(flash.rom_visible());
    EXPECT_EQ(flash.mmio_read(0xFFFFFE00, 1), image[0x3FE00]);
    // A wrong size is refused with a message.
    std::vector<u8> bad(1000);
    EXPECT_FALSE(flash.load_image(bad, error));
    EXPECT_NE(error.find("1000"), std::string::npos);
}

// A device's coalescing zones reach the CPU backend's hook: those asked for before it was installed, then each change.
TEST(Bus, CoalescingZonesReachTheBackendsHook) {
    Bus bus(64 * 1024);
    bus.coalesce_writes(0xFE820000u, 0x1000, true);
    std::vector<std::tuple<PhysAddr, u32, bool>> seen;
    bus.set_coalesce_hook([&](PhysAddr base, u32 size, bool on) { seen.emplace_back(base, size, on); });
    bus.coalesce_writes(0xFE820000u, 0x1000, true);  // already on: nothing
    bus.coalesce_writes(0xFE820000u, 0x1000, false);
    bus.coalesce_writes(0xFD820000u, 0x1000, true);
    ASSERT_EQ(seen.size(), 3u);
    EXPECT_EQ(seen[0], std::make_tuple(PhysAddr{0xFE820000u}, 0x1000u, true));
    EXPECT_EQ(seen[1], std::make_tuple(PhysAddr{0xFE820000u}, 0x1000u, false));
    EXPECT_EQ(seen[2], std::make_tuple(PhysAddr{0xFD820000u}, 0x1000u, true));
}

#if defined(__linux__)  // the ring is KVM's
// The coalesced MMIO ring is taken in the guest's order from `first` to `last`, across the end of the ring, and every
// slot is handed back.
TEST(Bus, TheCoalescedRingIsTakenInOrderAcrossItsEnd) {
    alignas(4096) static u8 page[4096];
    std::memset(page, 0, sizeof page);
    auto* ring = reinterpret_cast<kvm_coalesced_mmio_ring*>(page);
    const u32 slots = static_cast<u32>((sizeof page - sizeof(kvm_coalesced_mmio_ring)) / sizeof(kvm_coalesced_mmio));
    auto put = [&](u32 value) {  // the kernel's side
        kvm_coalesced_mmio& e = *kvm::coalesced_entry(ring, ring->last);
        e.phys_addr = 0xFE8202F8u;
        e.len = 4;
        std::memcpy(e.data, &value, 4);
        ring->last = (ring->last + 1) % slots;
    };
    ring->first = ring->last = slots - 2;
    for (u32 v = 1; v <= 5; ++v) put(v);
    std::vector<u32> got;
    const u32 n = kvm::drain_coalesced(ring, slots, [&](u64 phys, u32 len, const u8* data) {
        EXPECT_EQ(phys, 0xFE8202F8u);
        EXPECT_EQ(len, 4u);
        u32 v;
        std::memcpy(&v, data, 4);
        got.push_back(v);
    });
    EXPECT_EQ(n, 5u);
    EXPECT_EQ(got, (std::vector<u32>{1, 2, 3, 4, 5}));
    EXPECT_EQ(ring->first, ring->last);
    EXPECT_EQ(ring->first, 3u);
    EXPECT_EQ(kvm::drain_coalesced(ring, slots, [](u64, u32, const u8*) { ADD_FAILURE() << "an empty ring gave a write"; }), 0u);
}
#endif
