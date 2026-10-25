// The video overlay: YUV 4:2:2 to RGB (BT.601 video range), placement, scaling and the colour key.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <format>
#include <vector>

#include "core/bus.h"
#include "core/scheduler.h"
#include "nv2a/nv2a_stub.h"
#include "nv2a/overlay.h"
#include "nv2a/yuv.h"

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

namespace {

// The overlay as it was first drawn, one screen pixel at a time (each converted from its texel): the reference the
// row-at-a-time drawing must match exactly.
void draw_overlay_per_pixel(const Overlay& ov, const u8* ram, u8* xrgb, u32 pitch, u32 w, u32 h) {
    const u32 x_end = std::min(w, ov.out_x + ov.out_w), y_end = std::min(h, ov.out_y + ov.out_h);
    for (u32 y = ov.out_y; y < y_end; ++y) {
        const u64 t = (static_cast<u64>(ov.t0) << 16) + static_cast<u64>(y - ov.out_y) * ov.dtdy;
        const u32 sy = std::min(static_cast<u32>(t >> 20), ov.in_h - 1);
        const u8* row = ram + ov.address + static_cast<u64>(sy) * ov.pitch;
        for (u32 x = ov.out_x; x < x_end; ++x) {
            u32* px = reinterpret_cast<u32*>(xrgb + static_cast<u64>(y) * pitch + 4u * x);
            if (ov.color_key && (*px & 0xFFFFFFu) != (ov.key & 0xFFFFFFu)) continue;
            const u64 s = (static_cast<u64>(ov.s0) << 16) + static_cast<u64>(x - ov.out_x) * ov.dsdx;
            const u32 sx = std::min(static_cast<u32>(s >> 20), ov.in_w - 1);
            const u8* pair = row + (sx & ~1u) * 2u;
            const int y0 = ov.yuy2 ? pair[0] : pair[1], u = ov.yuy2 ? pair[1] : pair[0], y1 = ov.yuy2 ? pair[2] : pair[3], v = ov.yuy2 ? pair[3] : pair[2];
            *px = yuv601_to_xrgb((sx & 1u) ? y1 : y0, u, v);
        }
    }
}

}  // namespace

// A movie over a picture at 6x and at an odd scale, keyed and not, in bands as the frontend draws it: the same pixels
// as one conversion per screen pixel.
TEST(Overlay, DrawingRowAtATimeMatchesDrawingEachPixel) {
    std::vector<u8> ram(0x40000);
    u32 seed = 12345;
    for (u8& b : ram) b = static_cast<u8>((seed = seed * 1103515245u + 12345u) >> 16);
    for (bool key : {false, true})
        for (bool yuy2 : {false, true})
            for (u32 sc : {1u, 6u}) {
                SCOPED_TRACE(std::format("key {} yuy2 {} scale {}", key, yuy2, sc));
                Overlay ov;
                ov.address = 0x1000;
                ov.pitch = 160;
                ov.in_w = 80;  // 80x60 source
                ov.in_h = 60;
                ov.s0 = 3u << 4;  // starting three texels in
                ov.out_x = 5 * sc;
                ov.out_y = 3 * sc;
                ov.out_w = 100 * sc;
                ov.out_h = 70 * sc;
                ov.dsdx = static_cast<u32>((70ull << 20) / (100 * sc));  // 70 texels over 100 pixels, then the scale
                ov.dtdy = static_cast<u32>((55ull << 20) / (70 * sc));
                ov.yuy2 = yuy2;
                ov.color_key = key;
                ov.key = 0x00FF00u;
                const u32 w = 120 * sc, h = 80 * sc;
                std::vector<u32> base(static_cast<std::size_t>(w) * h);
                for (std::size_t i = 0; i < base.size(); ++i) base[i] = (i / 7) % 3 == 0 ? 0x00FF00u : 0x123456u;
                std::vector<u32> want = base, got = base;
                draw_overlay_per_pixel(ov, ram.data(), reinterpret_cast<u8*>(want.data()), w * 4, w, h);
                for (u32 y0 = 0; y0 < h; y0 += 37 * sc) draw_overlay(ov, ram.data(), ram.size(), reinterpret_cast<u8*>(got.data()), w * 4, w, h, y0, y0 + 37 * sc);
                std::size_t differ = 0, first = got.size();
                for (std::size_t i = 0; i < got.size(); ++i)
                    if (got[i] != want[i] && differ++ == 0) first = i;
                EXPECT_EQ(differ, 0u) << "first at pixel " << first;
            }
}
