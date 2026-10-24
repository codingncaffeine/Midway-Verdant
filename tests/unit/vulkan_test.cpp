#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pgraph/texture.h"
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

TEST(Vulkan, TheAlphaTestComparesEightBitValues) {
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
    k.method(mthd::kColorClear, 0xFF0000FF);  // blue where the test discards
    k.method(mthd::kClearSurface, 0xF0);
    // An alpha of 0.501 is 128 in 8 bits: EQUAL to a reference of 0x80, though not as a float (0x80 / 255 = 0.50196).
    k.method(mthd::kAlphaTestEnable, 1);
    k.method(mthd::kAlphaFunc, 0x202);
    k.method(mthd::kAlphaRef, 0x80);
    k.method(mthd::kVertexDataArrayFormat + 12, 2u);
    k.method(mthd::kDiffuse4f + 0, bits(0.0f));
    k.method(mthd::kDiffuse4f + 4, bits(1.0f));
    k.method(mthd::kDiffuse4f + 8, bits(0.0f));
    k.method(mthd::kDiffuse4f + 12, bits(0.501f));
    draw_full_triangle(k, ram);
    renderer.end_frame();
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(pixel_at(ram, 8, 8), 0x8000FF00u) << "the fragment passes the EQUAL test";
}

TEST(Vulkan, AZ24S8WriteBackKeepsTheStencil) {
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
    k.method(mthd::kZstencilClear, 0x80000042);  // depth 0x800000 in bits 31-8, stencil 0x42
    k.method(mthd::kClearSurface, 0x03);
    renderer.end_frame();
    renderer.write_back_all();  // depth reaches RAM when the GPU idles (or a texture samples it)
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
    u32 word;
    std::memcpy(&word, &ram[0x20000 + 8 * 256 + 8 * 4], 4);
    EXPECT_EQ(word, 0x80000042u) << "the zeta word holds the depth over the stencil byte";
}

namespace {

// The vertex program: oPos = v0, oD0 = v3, oT0-oT3 = v9-v12.
void load_texcoord_program(Kelvin& k) {
    auto enc = [](u32 v, u32 out, bool fin) {
        std::array<u32, 4> t{};
        t[1] |= 1u << 21;
        t[1] |= (v & 15) << 9;
        t[1] |= (0u << 6) | (1u << 4) | (2u << 2) | 3u;
        t[2] |= 2u << 26;
        t[3] |= 1u << 11;
        t[3] |= (out & 0xFF) << 3;
        t[3] |= 15u << 12;
        if (fin) t[3] |= 1u;
        return t;
    };
    k.method(mthd::kTransformProgramLoad, 0);
    for (const auto& t : {enc(0, 0, false), enc(3, 3, false), enc(9, 9, false), enc(10, 10, false), enc(11, 11, false), enc(12, 12, true)})
        for (u32 w = 0; w < 4; ++w) k.method(mthd::kTransformProgram + 4 * w, t[w]);
    k.method(mthd::kTransformProgramStart, 0);
    k.method(mthd::kTransformExecutionMode, 2);
}

// Combiners whose output is texture stage `stage`: colour = t (A = t, B = one), alpha = t.a.
void output_texture_stage(Kelvin& k, u32 stage) {
    k.method(mthd::kCombinerControl, 1);
    k.method(mthd::kCombinerColorIcw, ((8u + stage) << 24) | ((1u << 5) << 16));
    k.method(mthd::kCombinerColorOcw, 12u << 4);
    k.method(mthd::kCombinerAlphaIcw, (((8u + stage) | 0x10u) << 24) | ((1u << 5) << 16));
    k.method(mthd::kCombinerAlphaOcw, 12u << 4);
    k.method(mthd::kCombinerSpecFogCw0, (12u << 24) | ((1u << 5) << 16));
    k.method(mthd::kCombinerSpecFogCw1, (12u | 0x10u) << 8);
}

// A swizzled A8R8G8B8 texture of 2^log x 2^log texels at `offset`, one level, nearest filtering, clamped to the
// edge; texel (x, y) is texel(x, y).
template <typename F>
void load_texture(Kelvin& k, std::vector<u8>& ram, u32 stage, u32 offset, u32 log, F texel) {
    for (u32 y = 0; y < (1u << log); ++y)
        for (u32 x = 0; x < (1u << log); ++x) {
            const u32 v = texel(x, y);
            std::memcpy(&ram[offset + swizzled_offset(x, y, 0, log, log, 0) * 4], &v, 4);
        }
    const u32 b = mthd::kTextureBase + mthd::kTextureStride * stage;
    k.method(b + 0x0, offset);
    k.method(b + 0x4, (log << 24) | (log << 20) | (1u << 16) | (0x06u << 8) | (2u << 4) | 1u);
    k.method(b + 0x8, 0x00030303);              // clamp to edge in u, v and p
    k.method(b + 0xC, 0x40000000);              // enabled
    k.method(b + 0x14, (1u << 24) | (1u << 16));  // nearest
}

void texcoord(Kelvin& k, u32 set, float s, float t, float r, float q) {
    const u32 base = mthd::kTexcoord0_4f + 0x28 * set;
    k.method(base + 0, bits(s));
    k.method(base + 4, bits(t));
    k.method(base + 8, bits(r));
    k.method(base + 12, bits(q));
}

struct TextureShaderFixture {
    vk::Device device;
    std::vector<u8> ram = std::vector<u8>(4u << 20, 0);
    RamResolver handles;
    std::unique_ptr<VulkanRenderer> renderer;
    Kelvin k;
    bool init(std::string& error) {
        if (!device.init(error)) return false;
        handles.size = static_cast<u32>(ram.size());
        renderer = std::make_unique<VulkanRenderer>(device, handles, ram.data(), static_cast<u32>(ram.size()));
        if (!renderer->init(error)) return false;
        k.set_sink(renderer.get());
        setup_target(k);
        load_texcoord_program(k);
        k.method(mthd::kVertexDataArrayFormat + 12, 2u);  // diffuse, texcoords: their current values
        return true;
    }
};

}  // namespace

TEST(Vulkan, DotSTLooksUpTheTwoDotProducts) {
    TextureShaderFixture f;
    std::string error;
    if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    Kelvin& k = f.k;
    // Stage 0: a 1x1 normal map, R 0x7F G 0x81 B 0x00 = (1, -1, 0) as two's complement / 127 (dot mapping 3).
    load_texture(k, f.ram, 0, 0x200000, 0, [](u32, u32) { return 0xFF7F8100u; });
    // Stage 2: a 4x4 texture whose texel (x, y) is R = 0x40 x, G = 0x40 y, B = 0x11.
    load_texture(k, f.ram, 2, 0x210000, 2, [](u32 x, u32 y) { return 0xFF000011u | (0x40u * x) << 16 | (0x40u * y) << 8; });
    k.method(mthd::kShaderStageProgram, 1u | (0x11u << 5) | (9u << 10));  // 2D, dot product, dot ST
    k.method(mthd::kShaderOtherStageInput, 0);                              // stage 2 reads stage 0
    k.method(mthd::kDotRgbMapping, 0x033);
    output_texture_stage(k, 2);
    texcoord(k, 0, 0.5f, 0.5f, 0.0f, 1.0f);
    // dot1 = (0.875, 0.5, 7) . (1, -1, 0) = 0.375 and dot2 = (1.125, 0.5, 3) . (1, -1, 0) = 0.625: the centre of
    // texel (1, 2) of the 4x4 texture. (Read as 0..1 instead, the normal would give texel (2, 3).)
    texcoord(k, 1, 0.875f, 0.5f, 7.0f, 1.0f);
    texcoord(k, 2, 1.125f, 0.5f, 3.0f, 1.0f);
    draw_full_triangle(k, f.ram);
    f.renderer->end_frame();
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(f.renderer->stats().skipped, 0u);
    EXPECT_EQ(pixel_at(f.ram, 8, 8), 0xFF408011u);
    EXPECT_EQ(pixel_at(f.ram, 40, 20), 0xFF408011u);
}

TEST(Vulkan, DotZWWritesTheRatioOfTheTwoDotProductsAsTheDepth) {
    TextureShaderFixture f;
    std::string error;
    if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    Kelvin& k = f.k;
    k.method(mthd::kZstencilClear, 0xFFFFFF00);
    k.method(mthd::kClearRectH, 63u << 16);
    k.method(mthd::kClearRectV, 63u << 16);
    k.method(mthd::kClearSurface, 0x03);
    k.method(mthd::kDepthTestEnable, 1);
    k.method(mthd::kDepthFunc, 0x207);  // always
    k.method(mthd::kDepthMask, 1);
    load_texture(k, f.ram, 0, 0x200000, 0, [](u32, u32) { return 0xFF7F8100u; });  // (1, -1, 0) under mapping 3
    k.method(mthd::kShaderStageProgram, 1u | (0x11u << 5) | (0xAu << 10));
    k.method(mthd::kDotRgbMapping, 0x033);
    output_texture_stage(k, 0);
    texcoord(k, 0, 0.5f, 0.5f, 0.0f, 1.0f);
    texcoord(k, 1, 0.75f, 0.5f, 0.0f, 1.0f);  // dot1 = 0.25
    texcoord(k, 2, 1.5f, 0.5f, 0.0f, 1.0f);   // dot2 = 1.0: depth 0.25 of the 24-bit range
    draw_full_triangle(k, f.ram);
    f.renderer->end_frame();
    f.renderer->write_back_all();
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    u32 word;
    std::memcpy(&word, &f.ram[0x20000 + 8 * 256 + 8 * 4], 4);
    EXPECT_NEAR(static_cast<double>(word >> 8), 0.25 * 16777215.0, 1.0) << std::hex << word;
    EXPECT_EQ(pixel_at(f.ram, 8, 8), 0xFF7F8100u) << "the colour still draws";
}

TEST(Vulkan, DependentGBLooksUpTheSourcesGreenAndBlue) {
    TextureShaderFixture f;
    std::string error;
    if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    Kelvin& k = f.k;
    // Stage 0: G = 0x60 (0.376), B = 0xA0 (0.627); A = 0x20, R = 0xE0 would pick another texel for AR.
    load_texture(k, f.ram, 0, 0x200000, 0, [](u32, u32) { return 0x20E060A0u; });
    load_texture(k, f.ram, 1, 0x210000, 2, [](u32 x, u32 y) { return 0xFF000011u | (0x40u * x) << 16 | (0x40u * y) << 8; });
    k.method(mthd::kShaderStageProgram, 1u | (0x10u << 5));
    output_texture_stage(k, 1);
    texcoord(k, 0, 0.5f, 0.5f, 0.0f, 1.0f);
    draw_full_triangle(k, f.ram);
    // The same with alpha-red: (0.125, 0.878), texel (0, 3).
    k.method(mthd::kShaderStageProgram, 1u | (0xFu << 5));
    k.method(mthd::kSurfaceColorOffset, 0x100000);
    draw_full_triangle(k, f.ram);
    f.renderer->end_frame();
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(pixel_at(f.ram, 8, 8), 0xFF408011u) << "texel (1, 2)";
    u32 ar;
    std::memcpy(&ar, &f.ram[0x100000 + 8 * 256 + 8 * 4], 4);
    EXPECT_EQ(ar, 0xFF00C011u) << "texel (0, 3)";
}

TEST(Vulkan, BumpEnvOffsetsTheLookupByTheMatrixTimesDuDv) {
    TextureShaderFixture f;
    std::string error;
    if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    Kelvin& k = f.k;
    // Stage 0: B 0x7F, G 0x81 = (du, dv) = (1, -1); R 0xFF = luminance 1.
    load_texture(k, f.ram, 0, 0x200000, 0, [](u32, u32) { return 0xFFFF817Fu; });
    load_texture(k, f.ram, 1, 0x210000, 2, [](u32 x, u32 y) { return 0xFF000011u | (0x40u * x) << 16 | (0x40u * y) << 8; });
    // m00 0.5, m01 0.25, m10 0.25, m11 -0.25, written m00 m01 m11 m10: s' = s + 0.5 - 0.25, t' = t + 0.25 + 0.25.
    const u32 mat = mthd::kBumpEnvMat + mthd::kTextureStride;
    k.method(mat + 0, bits(0.5f));
    k.method(mat + 4, bits(0.25f));
    k.method(mat + 8, bits(-0.25f));
    k.method(mat + 12, bits(0.25f));
    k.method(mthd::kBumpEnvScale + mthd::kTextureStride, bits(0.5f));
    k.method(mthd::kBumpEnvOffset + mthd::kTextureStride, bits(0.25f));
    output_texture_stage(k, 1);
    texcoord(k, 0, 0.5f, 0.5f, 0.0f, 1.0f);
    texcoord(k, 1, 0.125f, 0.125f, 0.0f, 1.0f);  // (0.375, 0.625): texel (1, 2)
    auto draw_into = [&](u32 mode, u32 offset) {
        k.method(mthd::kShaderStageProgram, 1u | (mode << 5));
        k.method(mthd::kSurfaceColorOffset, offset);
        draw_full_triangle(k, f.ram);
    };
    draw_into(6, 0x100000);
    draw_into(7, 0x110000);  // the texel's colour times 0.5 x 1 + 0.25
    k.method(mthd::kBumpEnvScale + mthd::kTextureStride, bits(2.0f));
    k.method(mthd::kBumpEnvOffset + mthd::kTextureStride, bits(0.0f));
    draw_into(7, 0x120000);  // a factor of 2 clamps to 1
    f.renderer->end_frame();
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(f.renderer->stats().skipped, 0u);
    auto at = [&](u32 offset) {
        u32 v;
        std::memcpy(&v, &f.ram[offset + 8 * 256 + 8 * 4], 4);
        return v;
    };
    EXPECT_EQ(at(0x100000), 0xFF408011u) << "texel (1, 2)";
    EXPECT_EQ(at(0x110000), 0xFF30600Du) << "red, green and blue times 0.75; alpha kept";
    EXPECT_EQ(at(0x120000), 0xFF408011u) << "the factor clamped to 1";
}

// The polygon offset moves a triangle's depth by bias + factor x slope (depth-buffer units) when the enable of the front
// polygon mode is set: a decal drawn over the same plane passes a LESS test only with a negative bias. (Without it,
// coplanar decals z-fight: shadows that flicker or vanish.)
TEST(Vulkan, ThePolygonOffsetMovesACoplanarTriangleInFrontOnlyWhenEnabled) {
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
    k.method(mthd::kColorClear, 0xFF000080);
    k.method(mthd::kZstencilClear, 0xFFFFFF00);
    k.method(mthd::kClearRectH, 63u << 16);
    k.method(mthd::kClearRectV, 63u << 16);
    k.method(mthd::kClearSurface, 0xF3);
    k.method(mthd::kDepthTestEnable, 1);
    k.method(mthd::kDepthFunc, 0x201);  // LESS
    k.method(mthd::kDepthMask, 1);
    auto triangle = [&](u32 abgr, float depth) {  // over the whole target, at one depth (a fraction of the 24-bit range)
        k.method(mthd::kBeginEnd, 5);
        k.method(mthd::kDiffuse4ub, abgr);
        for (const auto& p : {std::array<float, 2>{0.0f, 0.0f}, std::array<float, 2>{128.0f, 0.0f}, std::array<float, 2>{0.0f, 128.0f}}) {
            k.method(mthd::kVertexData4fM + 0, bits(p[0]));
            k.method(mthd::kVertexData4fM + 4, bits(p[1]));
            k.method(mthd::kVertexData4fM + 8, bits(depth * 16777215.0f));
            k.method(mthd::kVertexData4fM + 12, bits(1.0f));
        }
        k.method(mthd::kBeginEnd, 0);
    };
    triangle(0xFF00FF00, 0.5f);  // green floor
    triangle(0xFF0000FF, 0.5f);  // red decal on the same plane, no offset: fails LESS
    renderer.end_frame();
    EXPECT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0x00FF00u) << "a coplanar triangle without an offset stays behind";
    k.method(mthd::kPolygonOffsetBias, bits(-16.0f));
    triangle(0xFF0000FF, 0.5f);  // the bias is set but the fill enable is not
    renderer.end_frame();
    EXPECT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0x00FF00u) << "the bias needs the fill enable";
    k.method(mthd::kPolyOffsetFillEnable, 1);
    triangle(0xFF0000FF, 0.5f);
    renderer.end_frame();
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0xFF0000u) << "pulled in front by 16 depth units";
}

TEST(Vulkan, ATargetWhoseMemoryAnotherEngineRewroteIsReloadedBeforeItIsDrawnIntoAgain) {
    // One title resets a glare buffer each frame with a 2D blit and then draws into it additively. The renderer keeps
    // the target on the GPU between frames, so drawing on over that copy ignored the blit and the buffer saturated.
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
    k.method(mthd::kVertexDataArrayFormat + 12, 2u);
    k.method(mthd::kDiffuse4ub, 0xFF000040);  // a little red, added each draw
    k.method(mthd::kBlendEnable, 1);
    k.method(mthd::kBlendSfactor, 1);  // ONE
    k.method(mthd::kBlendDfactor, 1);  // ONE
    k.method(mthd::kBlendEquation, 0x8006);
    draw_full_triangle(k, ram);
    renderer.end_frame();
    ASSERT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0x400000u);
    // Another engine writes the target's memory (the 2D blit's path: the bytes, then the renderer hears of it).
    for (u32 y = 0; y < 64; ++y)
        for (u32 x = 0; x < 64; ++x) {
            const u32 green = 0xFF003000u;
            std::memcpy(&ram[0x10000 + y * 256 + x * 4], &green, 4);
        }
    renderer.invalidate(0x10000, 64 * 256);
    draw_full_triangle(k, ram);
    renderer.end_frame();
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0x403000u) << "the draw adds to what RAM held, not to the GPU's earlier copy (0x800000)";
    EXPECT_EQ(renderer.stats().surface_reloads, 1u);
}

TEST(Vulkan, AWriteReportFromBeforeTheTargetExistedDoesNotUndoALaterClear) {
    // A title copies the last boot frame into its new buffers with the CPU, then clears them on the GPU. The watch on a
    // new target's pages is applied later, and the write log then reports pages written before the target existed:
    // taken as fresh CPU writes, the copy went back over the clear (the old frame showed around the game's picture).
    vk::Device device;
    std::string error;
    if (!device.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    std::vector<u8> ram(4u << 20, 0);
    RamResolver handles;
    handles.size = static_cast<u32>(ram.size());
    VulkanRenderer renderer(device, handles, ram.data(), static_cast<u32>(ram.size()));
    ASSERT_TRUE(renderer.init(error)) << error;
    for (u32 y = 0; y < 64; ++y)
        for (u32 x = 0; x < 64; ++x) {
            const u32 green = 0xFF00FF00u;  // the copy, in RAM before the target exists
            std::memcpy(&ram[0x10000 + y * 256 + x * 4], &green, 4);
        }
    Kelvin k;
    k.set_sink(&renderer);
    setup_target(k);
    k.method(mthd::kColorClear, 0xFF000000u);
    k.method(mthd::kClearRectH, 63u << 16);
    k.method(mthd::kClearRectV, 63u << 16);
    k.method(mthd::kClearSurface, 0xF0);
    renderer.cpu_wrote({0x10, 0x11, 0x12, 0x13});  // the late report: the bytes are what the target loaded
    renderer.end_frame();
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(pixel_at(ram, 8, 8), 0xFF000000u) << "the clear came after the copy";
    EXPECT_EQ(renderer.stats().cpu_written_targets, 0u);
}

TEST(Vulkan, ACpuWriteAfterTheTargetWasDrawnIsKeptAndDrawnOver) {
    // One title copies its last frame into both buffers with the CPU when it pauses, then draws its menu over them.
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
    k.method(mthd::kVertexDataArrayFormat + 12, 2u);
    k.method(mthd::kDiffuse4ub, 0xFF000040);  // a little red, added each draw
    k.method(mthd::kBlendEnable, 1);
    k.method(mthd::kBlendSfactor, 1);  // ONE
    k.method(mthd::kBlendDfactor, 1);  // ONE
    k.method(mthd::kBlendEquation, 0x8006);
    draw_full_triangle(k, ram);
    renderer.end_frame();
    ASSERT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0x400000u);
    for (u32 y = 0; y < 64; ++y)
        for (u32 x = 0; x < 64; ++x) {
            const u32 blue = 0xFF000030u;  // the CPU's copy
            std::memcpy(&ram[0x10000 + y * 256 + x * 4], &blue, 4);
        }
    renderer.cpu_wrote({0x10, 0x11, 0x12, 0x13});
    draw_full_triangle(k, ram);
    renderer.end_frame();
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0x400030u) << "the draw adds to the CPU's bytes";
    EXPECT_EQ(renderer.stats().cpu_written_targets, 1u);
}
