// The 2D engine: a SRCCOPY image blit between two surfaces of the 2D surfaces object.
#include <gtest/gtest.h>

#include <vector>

#include "nv2a/pgraph/twod.h"

using namespace mv;
using namespace mv::pgraph;

namespace {
// Two DMA objects: handle 1 a window at 0x1000, handle 2 a window at 0x8000 (both 16 KiB).
struct Windows : HandleResolver {
    DmaObject resolve(u32 handle) const override {
        if (handle == 1) return {0x1000, 0x3FFF, 0x3D, true};
        if (handle == 2) return {0x8000, 0x3FFF, 0x3D, true};
        return {};
    }
};
}  // namespace

TEST(TwoD, ImageBlitCopiesARectangleBetweenSurfacesWithTheirOwnPitches) {
    std::vector<u8> ram(0x10000, 0);
    for (u32 i = 0; i < 0x4000; ++i) ram[0x1000 + i] = static_cast<u8>(i * 7 + 1);
    Windows dma;
    TwoD e(ram.data(), ram.size(), dma);
    // The "must be in RAM first" hook hears the source, then the destination (a partial blit keeps the rest of it).
    std::vector<std::pair<u32, u32>> reads;
    u32 write_at = 0, write_len = 0;
    e.set_hooks([&](u32 p, u32 l) { reads.emplace_back(p, l); }, [&](u32 p, u32 l) { write_at = p; write_len = l; });
    // Surfaces: 32-bit pixels, source pitch 256, destination pitch 128; source offset 0x100, destination 0x40.
    e.method(TwoD::kClassSurfaces, TwoD::kSurfDmaSource, 1);
    e.method(TwoD::kClassSurfaces, TwoD::kSurfDmaDestination, 2);
    e.method(TwoD::kClassSurfaces, TwoD::kSurfFormat, 0x0A);
    e.method(TwoD::kClassSurfaces, TwoD::kSurfPitch, (128u << 16) | 256u);
    e.method(TwoD::kClassSurfaces, TwoD::kSurfOffsetSource, 0x100);
    e.method(TwoD::kClassSurfaces, TwoD::kSurfOffsetDestination, 0x40);
    // A 3x2 rectangle from (2, 1) to (5, 4).
    e.method(TwoD::kClassBlit, TwoD::kBlitOperation, TwoD::kOperationSrcCopy);
    e.method(TwoD::kClassBlit, TwoD::kBlitPointIn, (1u << 16) | 2u);
    e.method(TwoD::kClassBlit, TwoD::kBlitPointOut, (4u << 16) | 5u);
    EXPECT_TRUE(e.method(TwoD::kClassBlit, TwoD::kBlitSize, (2u << 16) | 3u));
    ASSERT_EQ(e.blits(), 1u) << e.last_error();
    for (u32 y = 0; y < 2; ++y)
        for (u32 b = 0; b < 12; ++b) {
            const u32 src = 0x1000 + 0x100 + (1 + y) * 256 + 2 * 4 + b;
            const u32 dst = 0x8000 + 0x40 + (4 + y) * 128 + 5 * 4 + b;
            EXPECT_EQ(ram[dst], ram[src]) << "row " << y << " byte " << b;
        }
    // Nothing outside the rectangle moved: the byte after each destination row and the row above are untouched.
    EXPECT_EQ(ram[0x8000 + 0x40 + 4 * 128 + 5 * 4 + 12], 0);
    EXPECT_EQ(ram[0x8000 + 0x40 + 3 * 128 + 5 * 4], 0);
    // The host heard the source range (a render target there reaches RAM first) and the destination range.
    ASSERT_EQ(reads.size(), 2u);
    EXPECT_EQ(reads[0], std::make_pair(0x1000u + 0x100 + 256 + 8, 256u + 12));
    EXPECT_EQ(reads[1], std::make_pair(0x8000u + 0x40 + 4 * 128 + 20, 128u + 12));
    EXPECT_EQ(write_at, 0x8000u + 0x40 + 4 * 128 + 20);
    EXPECT_EQ(write_len, 128u + 12);
}

TEST(TwoD, ImageBlitRefusesWhatItDoesNotModel) {
    std::vector<u8> ram(0x10000, 0x55);
    Windows dma;
    TwoD e(ram.data(), ram.size(), dma);
    e.method(TwoD::kClassSurfaces, TwoD::kSurfDmaSource, 1);
    e.method(TwoD::kClassSurfaces, TwoD::kSurfDmaDestination, 2);
    e.method(TwoD::kClassSurfaces, TwoD::kSurfFormat, 0x01);
    e.method(TwoD::kClassSurfaces, TwoD::kSurfPitch, (64u << 16) | 64u);
    e.method(TwoD::kClassBlit, TwoD::kBlitOperation, 5);  // not SRCCOPY
    e.method(TwoD::kClassBlit, TwoD::kBlitSize, (1u << 16) | 1u);
    EXPECT_EQ(e.blits(), 0u);
    EXPECT_NE(e.last_error().find("SRCCOPY"), std::string::npos) << e.last_error();
    // A rectangle past the end of its DMA window is refused, not clipped.
    e.method(TwoD::kClassBlit, TwoD::kBlitOperation, TwoD::kOperationSrcCopy);
    e.method(TwoD::kClassBlit, TwoD::kBlitSize, (300u << 16) | 64u);  // 300 rows of 64 bytes = 19200 > 16 KiB
    EXPECT_EQ(e.blits(), 0u);
    EXPECT_NE(e.last_error().find("DMA"), std::string::npos) << e.last_error();
}
