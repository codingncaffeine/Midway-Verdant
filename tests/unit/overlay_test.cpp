// The video overlay: YUV 4:2:2 to RGB (BT.601 video range), placement, scaling and the colour key.
#include <gtest/gtest.h>

#include <array>
#include <vector>

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
