// The video overlay: YUV 4:2:2 to RGB (BT.601 video range), placement, scaling and the colour key.
#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "core/bus.h"
#include "core/scheduler.h"
#include "nv2a/nv2a_stub.h"
#include "nv2a/overlay.h"

using namespace mv;

TEST(Overlay, ConvertsPlacesScalesAndKeys) {
    // Source: one row, UYVY, two pixel pairs: white then black (U = V = 128).
    std::vector<u8> ram(0x100, 0);
    const std::array<u8, 8> uyvy{128, 235, 128, 235, 128, 16, 128, 16};
    std::copy(uyvy.begin(), uyvy.end(), ram.begin() + 0x40);
    Overlay ov;
    ov.address = 0x40;
    ov.pitch = 8;
    ov.in_w = 4;
    ov.in_h = 1;
    ov.out_x = 2;
    ov.out_y = 1;
    ov.out_w = 8;  // twice the width: half a source pixel per screen pixel
    ov.out_h = 2;
    ov.dsdx = 1u << 19;
    ov.dtdy = 1u << 19;
    std::vector<u32> frame(16 * 4, 0x123456u);
    draw_overlay(ov, ram.data(), ram.size(), reinterpret_cast<u8*>(frame.data()), 16 * 4, 16, 4);
    EXPECT_EQ(frame[1 * 16 + 1], 0x123456u) << "left of the rectangle: untouched";
    EXPECT_EQ(frame[1 * 16 + 2], 0xFFFFFFu);  // white
    EXPECT_EQ(frame[1 * 16 + 5], 0xFFFFFFu);  // still the white pair, doubled
    EXPECT_EQ(frame[2 * 16 + 6], 0x000000u);  // black
    EXPECT_EQ(frame[3 * 16 + 6], 0x123456u) << "below the rectangle: untouched";

    // With the colour key only pixels of the key colour are replaced.
    std::vector<u32> keyed(16 * 4, 0x123456u);
    keyed[1 * 16 + 3] = 0x00FF00u;
    ov.color_key = true;
    ov.key = 0x00FF00u;
    draw_overlay(ov, ram.data(), ram.size(), reinterpret_cast<u8*>(keyed.data()), 16 * 4, 16, 4);
    EXPECT_EQ(keyed[1 * 16 + 2], 0x123456u);
    EXPECT_EQ(keyed[1 * 16 + 3], 0xFFFFFFu);
}

TEST(Ramdac, TheColourTableIsWrittenThroughTheDacPortsAndAppliedPerChannel) {
    Bus bus(4096);
    Scheduler sched;
    Nv2aStub gpu(bus, sched, nullptr);
    // Power-up: the identity, reported as no table so callers skip the work.
    EXPECT_FALSE(gpu.gamma_lut().has_value());
    // A ramp as software writes it: index 0, then red, green, blue for each of the 256 entries.
    // Red inverted, green halved, blue untouched: each channel must use its own column.
    gpu.mmio_write(Nv2aStub::kDacWriteIndex, 1, 0);
    for (u32 i = 0; i < 256; ++i) {
        gpu.mmio_write(Nv2aStub::kDacData, 1, 255 - i);
        gpu.mmio_write(Nv2aStub::kDacData, 1, i / 2);
        gpu.mmio_write(Nv2aStub::kDacData, 1, i);
    }
    const auto lut = gpu.gamma_lut();
    ASSERT_TRUE(lut.has_value());
    // Reading back from entry 10 returns its triple.
    gpu.mmio_write(Nv2aStub::kDacReadIndex, 1, 10);
    EXPECT_EQ(gpu.mmio_read(Nv2aStub::kDacData, 1), 245u);
    EXPECT_EQ(gpu.mmio_read(Nv2aStub::kDacData, 1), 5u);
    EXPECT_EQ(gpu.mmio_read(Nv2aStub::kDacData, 1), 10u);
    // An XRGB pixel (bytes B, G, R, X) of R=200 G=100 B=40 becomes R=55 G=50 B=40.
    std::array<u8, 8> px{40, 100, 200, 0, 0, 0, 0, 0};
    apply_gamma(*lut, px.data(), 8, 2, 1);
    EXPECT_EQ(px[0], 40);
    EXPECT_EQ(px[1], 50);
    EXPECT_EQ(px[2], 55);
    EXPECT_EQ(px[4], 0);
    EXPECT_EQ(px[5], 0);
    EXPECT_EQ(px[6], 255);
    // Writing the identity back makes the table disappear again.
    gpu.mmio_write(Nv2aStub::kDacWriteIndex, 1, 0);
    for (u32 i = 0; i < 256; ++i)
        for (int c = 0; c < 3; ++c) gpu.mmio_write(Nv2aStub::kDacData, 1, i);
    EXPECT_FALSE(gpu.gamma_lut().has_value());
}
