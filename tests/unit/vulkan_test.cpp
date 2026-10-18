#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "nv2a/pgraph/kelvin.h"
#include "nv2a/vulkan/vk_device.h"
#include "nv2a/vulkan/vk_renderer.h"

using namespace mv;
using namespace mv::pgraph;

namespace {

u32 bits(float f) {
    u32 v;
    std::memcpy(&v, &f, 4);
    return v;
}

struct RamResolver : HandleResolver {
    u32 size{0};
    DmaObject resolve(u32) const override { return {0, size - 1, 0x3D, true}; }
};

// The vertex program: oPos = v0 (already in screen space), oD0 = v3.
void load_passthrough_program(Kelvin& k) {
    // mov oPos.xyzw, v0 ; mov oD0.xyzw, v3 (final)
    auto enc = [](u32 v, u32 out, bool fin) {
        std::array<u32, 4> t{};
        t[1] |= 1u << 21;                       // MAC mov
        t[1] |= (v & 15) << 9;                  // input
        t[1] |= (0u << 6) | (1u << 4) | (2u << 2) | 3u;  // A swizzle xyzw
        t[2] |= 2u << 26;                       // A = input
        t[3] |= 1u << 11;                       // output register
        t[3] |= (out & 0xFF) << 3;
        t[3] |= 15u << 12;                      // mask xyzw
        if (fin) t[3] |= 1u;
        return t;
    };
    k.method(mthd::kTransformProgramLoad, 0);
    for (const auto& t : {enc(0, 0, false), enc(3, 3, true)})
        for (u32 w = 0; w < 4; ++w) k.method(mthd::kTransformProgram + 4 * w, t[w]);
    k.method(mthd::kTransformProgramStart, 0);
    k.method(mthd::kTransformExecutionMode, 2);
}

// Combiners: final colour = v0 (diffuse), alpha = v0.a.
void load_passthrough_combiners(Kelvin& k) {
    k.method(mthd::kCombinerControl, 1);
    k.method(mthd::kCombinerColorIcw, (4u << 24) | ((1u << 5) << 16));  // A = v0, B = one
    k.method(mthd::kCombinerColorOcw, 12u << 4);                        // AB -> r0
    k.method(mthd::kCombinerAlphaIcw, ((4u | 0x10u) << 24) | ((1u << 5) << 16));
    k.method(mthd::kCombinerAlphaOcw, 12u << 4);
    k.method(mthd::kCombinerSpecFogCw0, (12u << 24) | ((1u << 5) << 16));  // A = r0, B = one, C = D = zero
    k.method(mthd::kCombinerSpecFogCw1, (12u | 0x10u) << 8);              // G = r0.a
}

// A 64x64 linear target at 0x10000 (pitch 256) with a Z24S8 buffer at 0x20000, the pass-through program and
// combiners, no depth test, no culling.
void setup_target(Kelvin& k, u32 color_format = 8, u32 color_offset = 0x10000) {
    k.method(mthd::kDmaColor, 9);
    k.method(mthd::kDmaZeta, 10);
    k.method(mthd::kDmaVertexA, 3);
    k.method(mthd::kSurfaceFormat, 0x00000120 | color_format);
    k.method(mthd::kSurfaceClipH, 64u << 16);
    k.method(mthd::kSurfaceClipV, 64u << 16);
    k.method(mthd::kSurfacePitch, (256u << 16) | 256u);
    k.method(mthd::kSurfaceColorOffset, color_offset);
    k.method(mthd::kSurfaceZetaOffset, 0x20000);
    load_passthrough_program(k);
    load_passthrough_combiners(k);
    k.method(mthd::kColorMask, 0x01010101);
    k.method(mthd::kDepthTestEnable, 0);
    k.method(mthd::kCullFaceEnable, 0);
    k.method(mthd::kShadeModel, 0x1D01);
}

// One triangle over the whole 64x64 target, from a position array of four floats per vertex at 0x30000.
void draw_full_triangle(Kelvin& k, std::vector<u8>& ram) {
    const float verts[3][4] = {{0.0f, 0.0f, 0.0f, 1.0f}, {128.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 128.0f, 0.0f, 1.0f}};
    std::memcpy(&ram[0x30000], verts, sizeof verts);
    k.method(mthd::kVertexDataArrayFormat + 0, (16u << 8) | (4u << 4) | 2u);
    k.method(mthd::kVertexDataArrayOffset + 0, 0x30000);
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kDrawArrays, (2u << 24) | 0u);
    k.method(mthd::kBeginEnd, 0);
}

u32 pixel_at(const std::vector<u8>& ram, u32 x, u32 y) {
    u32 v;
    std::memcpy(&v, &ram[0x10000 + y * 256 + x * 4], 4);
    return v;
}

}  // namespace

TEST(Vulkan, RendersATriangleIntoGuestRam) {
    vk::Device device;
    std::string error;
    if (!device.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    std::vector<u8> ram(4u << 20, 0);
    RamResolver handles;
    handles.size = static_cast<u32>(ram.size());
    VulkanRenderer renderer(device, handles, ram.data(), static_cast<u32>(ram.size()));
    ASSERT_TRUE(renderer.init(error)) << error;
    Kelvin k;
    k.set_sink(&renderer);
    // A 64x64 linear A8R8G8B8 surface at 0x10000 with a Z24S8 buffer at 0x20000.
    k.method(mthd::kDmaColor, 9);
    k.method(mthd::kDmaZeta, 10);
    k.method(mthd::kDmaVertexA, 3);
    k.method(mthd::kSurfaceFormat, 0x00000128);
    k.method(mthd::kSurfaceClipH, 64u << 16);
    k.method(mthd::kSurfaceClipV, 64u << 16);
    k.method(mthd::kSurfacePitch, (256u << 16) | 256u);
    k.method(mthd::kSurfaceColorOffset, 0x10000);
    k.method(mthd::kSurfaceZetaOffset, 0x20000);
    k.method(mthd::kColorClear, 0xFF000080);  // dark blue
    k.method(mthd::kZstencilClear, 0xFFFFFF00);
    k.method(mthd::kClearRectH, 63u << 16);
    k.method(mthd::kClearRectV, 63u << 16);
    k.method(mthd::kClearSurface, 0xF3);
    load_passthrough_program(k);
    load_passthrough_combiners(k);
    k.method(mthd::kColorMask, 0x01010101);
    k.method(mthd::kDepthTestEnable, 0);
    k.method(mthd::kCullFaceEnable, 0);
    k.method(mthd::kShadeModel, 0x1D01);
    // A green triangle covering the lower-right half, as immediate vertices in screen space.
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kDiffuse4ub, 0xFF00FF00);  // A B G R bytes: alpha 1, green 1
    for (const float* p : {std::array<float, 3>{63.0f, 0.0f, 0.5f}.data(), std::array<float, 3>{63.0f, 63.0f, 0.5f}.data(),
                           std::array<float, 3>{0.0f, 63.0f, 0.5f}.data()}) {
        k.method(mthd::kVertexData4fM + 0, bits(p[0]));
        k.method(mthd::kVertexData4fM + 4, bits(p[1]));
        k.method(mthd::kVertexData4fM + 8, bits(p[2] * 16777215.0f));
        k.method(mthd::kVertexData4fM + 12, bits(1.0f));
    }
    k.method(mthd::kBeginEnd, 0);
    renderer.end_frame();
    const auto& st = renderer.stats();
    EXPECT_EQ(st.draws, 1u);
    EXPECT_EQ(st.clears, 1u);
    EXPECT_EQ(st.skipped, 0u) << (st.errors.empty() ? "" : st.errors[0]);
    for (const auto& e : st.errors) ADD_FAILURE() << "renderer: " << e;
    // Pixel (x, y) of the surface sits at 0x10000 + y * 256 + x * 4 as B G R A.
    auto pixel = [&](u32 x, u32 y) {
        u32 v;
        std::memcpy(&v, &ram[0x10000 + y * 256 + x * 4], 4);
        return v;
    };
    std::string map;
    for (u32 y = 0; y < 64; y += 8) {
        for (u32 x = 0; x < 64; x += 8) map += (pixel(x + 4, y + 4) & 0x00FF00u) ? 'G' : '.';
        map += '\n';
    }
    std::printf("%s", map.c_str());
    EXPECT_EQ(pixel(4, 4) & 0x00FFFFFFu, 0x000080u) << "top-left keeps the clear colour";
    EXPECT_EQ(pixel(60, 60) & 0x00FFFFFFu, 0x00FF00u) << "bottom-right is inside the green triangle";
    EXPECT_EQ(pixel(60, 60) >> 24, 0xFFu);
    EXPECT_EQ(pixel(2, 52) & 0x00FFFFFFu, 0x000080u) << "bottom-left is outside the triangle";
    EXPECT_EQ(pixel(52, 2) & 0x00FFFFFFu, 0x000080u) << "top-right is outside the triangle";
    EXPECT_EQ(pixel(40, 40) & 0x00FFFFFFu, 0x00FF00u) << "centre-right is inside";
    EXPECT_EQ(ram[0x10000 + 64 * 256], 0) << "nothing written past the surface";
    const auto surfaces = renderer.surfaces();
    ASSERT_EQ(surfaces.size(), 1u);
    EXPECT_EQ(surfaces[0].phys, 0x10000u);
    EXPECT_EQ(surfaces[0].width, 64u);
    std::printf("rendered on %s\n", device.name().c_str());
}

TEST(Vulkan, ADisabledAttributeReadsItsCurrentValue) {
    vk::Device device;
    std::string error;
    if (!device.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    std::vector<u8> ram(4u << 20, 0);
    RamResolver handles;
    handles.size = static_cast<u32>(ram.size());
    VulkanRenderer renderer(device, handles, ram.data(), static_cast<u32>(ram.size()));
    ASSERT_TRUE(renderer.init(error)) << error;
    Kelvin k;
    k.set_sink(&renderer);
    setup_target(k);
    // The diffuse array is disabled; its current value (set outside any batch) is what every vertex reads.
    k.method(mthd::kVertexDataArrayFormat + 12, 2u);
    k.method(mthd::kDiffuse4ub, 0xFF00FF00);  // A B G R from the top: opaque green
    draw_full_triangle(k, ram);
    renderer.end_frame();
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(pixel_at(ram, 8, 8), 0xFF00FF00u) << "not the (0, 0, 0, 1) of an attribute without data";
}

TEST(Vulkan, AG8B8TargetHasTwoBytesAPixel) {
    vk::Device device;
    std::string error;
    if (!device.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    std::vector<u8> ram(4u << 20, 0);
    RamResolver handles;
    handles.size = static_cast<u32>(ram.size());
    VulkanRenderer renderer(device, handles, ram.data(), static_cast<u32>(ram.size()));
    ASSERT_TRUE(renderer.init(error)) << error;
    Kelvin k;
    k.set_sink(&renderer);
    setup_target(k, 10);  // G8B8: 64 pixels are 128 bytes of each 256-byte line
    std::memset(&ram[0x10000], 0xEE, 64 * 256);
    k.method(mthd::kVertexDataArrayFormat + 12, 2u);
    k.method(mthd::kDiffuse4ub, 0xFF208040);  // A B G R from the top: green 0x80
    draw_full_triangle(k, ram);
    renderer.end_frame();
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(ram[0x10000 + 8 * 256 + 8 * 2 + 1], 0x80) << "pixel (8, 8)'s second byte is its green";
    EXPECT_EQ(ram[0x10000 + 8 * 256 + 63 * 2 + 1], 0x80) << "the line's last pixel ends at byte 128";
    EXPECT_EQ(ram[0x10000 + 8 * 256 + 128], 0xEE) << "nothing written past the line's 64 pixels";
    EXPECT_EQ(ram[0x10000 + 8 * 256 + 255], 0xEE);
}

TEST(Vulkan, ClearsDecodeTheValuePerFormatAndWriteOnlyTheChosenChannels) {
    vk::Device device;
    std::string error;
    if (!device.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    std::vector<u8> ram(4u << 20, 0);
    RamResolver handles;
    handles.size = static_cast<u32>(ram.size());
    VulkanRenderer renderer(device, handles, ram.data(), static_cast<u32>(ram.size()));
    ASSERT_TRUE(renderer.init(error)) << error;
    Kelvin k;
    k.set_sink(&renderer);
    auto clear = [&](u32 format, u32 offset, u32 value, u32 bits) {
        setup_target(k, format, offset);
        k.method(mthd::kColorClear, value);
        k.method(mthd::kClearRectH, 63u << 16);
        k.method(mthd::kClearRectV, 63u << 16);
        k.method(mthd::kClearSurface, bits);
    };
    auto word16 = [&](u32 offset) {
        u16 v;
        std::memcpy(&v, &ram[offset + 8 * 256 + 8 * 2], 2);
        return v;
    };
    auto word32 = [&](u32 offset) {
        u32 v;
        std::memcpy(&v, &ram[offset + 8 * 256 + 8 * 4], 4);
        return v;
    };
    clear(3, 0x100000, 0x0000F81F, 0xF0);  // R5G6B5: the 5-6-5 fields of the low half, magenta
    clear(1, 0x110000, 0xFFFF7C00, 0xF0);  // X1R5G5B5_Z1R5G5B5: the 5-5-5 fields, its top bit zero
    clear(2, 0x120000, 0x000003E0, 0xF0);  // X1R5G5B5_O1R5G5B5: its top bit one
    clear(4, 0x130000, 0xFF102030, 0xF0);  // X8R8G8B8_Z8R8G8B8: the unused byte zero
    clear(7, 0x140000, 0x00405060, 0xF0);  // X1A7R8G8B8_O1A7R8G8B8: the alpha's top bit one
    clear(8, 0x150000, 0x11223344, 0xF0);  // A8R8G8B8, then only red and alpha
    clear(8, 0x150000, 0xAABBCCDD, 0x90);
    renderer.end_frame();
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(word16(0x100000), 0xF81Fu);
    EXPECT_EQ(word16(0x110000), 0x7C00u);
    EXPECT_EQ(word16(0x120000), 0x83E0u);
    EXPECT_EQ(word32(0x130000), 0x00102030u);
    EXPECT_EQ(word32(0x140000), 0x80405060u);
    EXPECT_EQ(word32(0x150000), 0xAABB3344u) << "green and blue keep the first clear";
}
