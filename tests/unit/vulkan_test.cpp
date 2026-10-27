#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "nv2a/overlay.h"
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

constexpr u32 kZminMaxControlForTest = 0x1D78;  // Kelvin SET_ZMIN_MAX_CONTROL: ZCLAMP_EN in bits 7:4 (0 cull, 1 clamp)

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
    // The depth clip range as the runtime sets it with the program (the pixel shader clamps or culls depth to it; the
    // registers are zero at power-on).
    k.method(mthd::kClipMin, std::bit_cast<u32>(0.0f));
    k.method(mthd::kClipMax, std::bit_cast<u32>(16777215.0f));
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

TEST(Vulkan, TheDepthWrittenIsTheConsolesClampedOrCulledToTheClipRangeAndOnItsStep) {
    // The console's depth, not the host's: culled outside the clip range (SET_ZMIN_MAX_CONTROL ZCLAMP_EN = cull) or
    // clamped to it (= clamp), and stored on the buffer's integer step, the nearest one. A shadow map's compare and an
    // occlusion test quad at a light's depth depend on the exact step.
    vk::Device device;
    std::string error;
    if (!device.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    std::vector<u8> ram(4u << 20, 0);
    RamResolver handles;
    handles.size = static_cast<u32>(ram.size());
    VulkanRenderer renderer(device, handles, ram.data(), static_cast<u32>(ram.size()));
    ASSERT_TRUE(renderer.init(error)) << error;
    if (!device.depth_clamp()) GTEST_SKIP() << "the device cannot clamp depth";
    Kelvin k;
    k.set_sink(&renderer);
    setup_target(k);
    k.method(mthd::kDepthTestEnable, 1);
    k.method(mthd::kDepthFunc, 0x207);  // ALWAYS: every fragment writes its depth
    k.method(mthd::kDepthMask, 1);
    k.method(mthd::kClipMax, bits(8000000.0f));
    auto clear = [&] {
        k.method(mthd::kColorClear, 0xFF000080);
        k.method(mthd::kZstencilClear, 0xFFFFFF00);
        k.method(mthd::kClearRectH, 63u << 16);
        k.method(mthd::kClearRectV, 63u << 16);
        k.method(mthd::kClearSurface, 0xF3);
    };
    auto triangle = [&](float depth_units) {  // green, over the whole target, at one depth in depth units
        k.method(mthd::kBeginEnd, 5);
        k.method(mthd::kDiffuse4ub, 0xFF00FF00);
        for (const auto& p : {std::array<float, 2>{0.0f, 0.0f}, std::array<float, 2>{128.0f, 0.0f}, std::array<float, 2>{0.0f, 128.0f}}) {
            k.method(mthd::kVertexData4fM + 0, bits(p[0]));
            k.method(mthd::kVertexData4fM + 4, bits(p[1]));
            k.method(mthd::kVertexData4fM + 8, bits(depth_units));
            k.method(mthd::kVertexData4fM + 12, bits(1.0f));
        }
        k.method(mthd::kBeginEnd, 0);
    };
    auto stored_depth = [&] {
        renderer.end_frame();
        renderer.write_back_all();
        u32 word;
        std::memcpy(&word, &ram[0x20000 + 8 * 256 + 8 * 4], 4);
        return word >> 8;
    };
    // Depth-only draws (a shadow map's pass) take the exact path: the pixel shader writes the console's depth.
    k.method(mthd::kColorMask, 0);
    k.method(kZminMaxControlForTest, 0x01);  // cull
    clear();
    triangle(12000000.0f);
    EXPECT_EQ(stored_depth(), 0xFFFFFFu) << "beyond the clip range and culled: the clear's depth stays";
    k.method(kZminMaxControlForTest, 0x11);  // clamp
    clear();
    triangle(12000000.0f);
    EXPECT_EQ(stored_depth(), 8000000u) << "clamped to the clip range's top";
    clear();
    triangle(4000000.75f);
    EXPECT_EQ(stored_depth(), 4000001u) << "on the nearest step (truncation would store 4000000)";
    clear();
    triangle(4000000.25f);
    EXPECT_EQ(stored_depth(), 4000000u) << "on the nearest step below";
    // A colour draw that writes depth takes the exact path too; one that only tests keeps the host's depth.
    k.method(mthd::kColorMask, 0x01010101);
    k.method(kZminMaxControlForTest, 0x01);
    k.method(mthd::kClipMax, bits(16777215.0f));
    clear();
    triangle(4000000.75f);
    EXPECT_EQ(stored_depth(), 4000001u) << "a colour draw lands on the same step";
    EXPECT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0x00FF00u);
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
}

// The w-buffer mode (SET_CONTROL0 Z_PERSPECTIVE_ENABLE, bit 16): the depth stored is the vertex w at the pixel, taken
// perspective correct (1/w is what varies linearly across the screen), not z/w. A racing title draws its whole scene
// so and reads the depth buffer back as a texture.
TEST(Vulkan, TheWBufferModeStoresThePerspectiveCorrectW) {
    vk::Device device;
    std::string error;
    if (!device.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    std::vector<u8> ram(4u << 20, 0);
    RamResolver handles;
    handles.size = static_cast<u32>(ram.size());
    VulkanRenderer renderer(device, handles, ram.data(), static_cast<u32>(ram.size()));
    ASSERT_TRUE(renderer.init(error)) << error;
    if (!device.depth_clamp()) GTEST_SKIP() << "the device cannot clamp depth";
    Kelvin k;
    k.set_sink(&renderer);
    setup_target(k);
    k.method(mthd::kDepthTestEnable, 1);
    k.method(mthd::kDepthFunc, 0x207);  // ALWAYS: every fragment writes its depth
    k.method(mthd::kDepthMask, 1);
    k.method(mthd::kClipMax, bits(8000000.0f));
    k.method(kZminMaxControlForTest, 0x11);  // clamp
    auto clear = [&] {
        k.method(mthd::kColorClear, 0xFF000080);
        k.method(mthd::kZstencilClear, 0xFFFFFF00);
        k.method(mthd::kClearRectH, 63u << 16);
        k.method(mthd::kClearRectV, 63u << 16);
        k.method(mthd::kClearSurface, 0xF3);
    };
    // Screen-space vertices (the passthrough program): x, y in pixels, z in depth units (already divided by w), w kept.
    auto triangle = [&](std::array<float, 3> w, float z) {
        k.method(mthd::kBeginEnd, 5);
        k.method(mthd::kDiffuse4ub, 0xFF00FF00);
        const std::array<std::array<float, 2>, 3> xy{{{0.0f, 0.0f}, {128.0f, 0.0f}, {0.0f, 128.0f}}};
        for (u32 i = 0; i < 3; ++i) {
            k.method(mthd::kVertexData4fM + 0, bits(xy[i][0]));
            k.method(mthd::kVertexData4fM + 4, bits(xy[i][1]));
            k.method(mthd::kVertexData4fM + 8, bits(z));
            k.method(mthd::kVertexData4fM + 12, bits(w[i]));
        }
        k.method(mthd::kBeginEnd, 0);
    };
    auto stored_depth = [&] {
        renderer.end_frame();
        renderer.write_back_all();
        u32 word;
        std::memcpy(&word, &ram[0x20000 + 8 * 256 + 8 * 4], 4);
        return word >> 8;
    };
    k.method(mthd::kControl0, 1u << 16);  // w-buffer
    clear();
    triangle({1000.0f, 1000.0f, 1000.0f}, 3000000.0f);
    EXPECT_EQ(stored_depth(), 1000u) << "w itself, not the z of 3000000";
    clear();
    // Pixel (8.5, 8.5): screen weights 0.867, 0.066, 0.066; 1/w interpolated gives w = 105.3 (a linear w would be 119.9).
    triangle({100.0f, 400.0f, 100.0f}, 3000000.0f);
    EXPECT_EQ(stored_depth(), 105u) << "perspective correct";
    k.method(mthd::kControl0, 0);  // back to z
    clear();
    triangle({100.0f, 400.0f, 100.0f}, 3000000.0f);
    EXPECT_EQ(stored_depth(), 3000000u) << "z without the w-buffer mode";
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
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

// A clamp-to-border stage shows its BORDER_COLOR (0x1B24, A8R8G8B8) outside the texture, as the hardware captures show
// (nxdk_pgraph_tests TextureBorderColor); it was always opaque black.
TEST(Vulkan, AClampToBorderStageShowsItsBorderColour) {
    TextureShaderFixture f;
    std::string error;
    if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    Kelvin& k = f.k;
    load_texture(k, f.ram, 0, 0x200000, 0, [](u32, u32) { return 0xFFFFFFFFu; });
    const u32 b = mthd::kTextureBase;
    k.method(b + 0x8, 0x00030404);   // border in u and v
    k.method(b + 0x24, 0x80FF2200);  // the border colour
    k.method(mthd::kShaderStageProgram, 1u);  // stage 0: 2D
    output_texture_stage(k, 0);
    texcoord(k, 0, 2.0f, 2.0f, 0.0f, 1.0f);  // outside the texture
    draw_full_triangle(k, f.ram);
    f.renderer->end_frame();
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    if (!f.device.custom_border()) {
        EXPECT_EQ(pixel_at(f.ram, 8, 8), 0xFF000000u) << "no custom border colours: the nearest built-in one (opaque black)";
        return;
    }
    EXPECT_EQ(pixel_at(f.ram, 8, 8), 0x80FF2200u);
    texcoord(k, 0, 0.5f, 0.5f, 0.0f, 1.0f);  // inside: the texel
    draw_full_triangle(k, f.ram);
    f.renderer->end_frame();
    EXPECT_EQ(pixel_at(f.ram, 8, 8), 0xFFFFFFFFu);
}

// Background compiles: the first draw that needs a new shader is skipped while a worker compiles it (nothing is drawn,
// no error), and the same draw renders once the shader and pipeline are ready.
TEST(Vulkan, WithBackgroundCompilesANewShadersDrawIsSkippedUntilItIsReady) {
    TextureShaderFixture f;
    std::string error;
    if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    f.renderer->set_async_compile(true);
    Kelvin& k = f.k;
    load_texture(k, f.ram, 0, 0x200000, 0, [](u32, u32) { return 0xFF336699u; });
    k.method(mthd::kShaderStageProgram, 1u);
    output_texture_stage(k, 0);
    texcoord(k, 0, 0.5f, 0.5f, 0.0f, 1.0f);
    draw_full_triangle(k, f.ram);
    f.renderer->end_frame();
    EXPECT_GE(f.renderer->stats().pending_skips, 1u);
    EXPECT_NE(pixel_at(f.ram, 8, 8), 0xFF336699u) << "skipped, not drawn";
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (pixel_at(f.ram, 8, 8) != 0xFF336699u && std::chrono::steady_clock::now() < until) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        draw_full_triangle(k, f.ram);
        f.renderer->end_frame();
    }
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_EQ(pixel_at(f.ram, 8, 8), 0xFF336699u) << "drawn once ready";
    EXPECT_EQ(f.renderer->stats().skipped, 0u) << "a pending draw is not an error";
}

// ── the internal resolution (PLAN D34) ──────────────────────────────────────────────────────────────────────────────

namespace {

// A renderer whose targets are drawn at `scale` times their size.
struct ScaledFixture {
    vk::Device device;
    std::vector<u8> ram = std::vector<u8>(16u << 20, 0);
    RamResolver handles;
    std::unique_ptr<VulkanRenderer> renderer;
    Kelvin k;
    bool init(u32 scale, std::string& error) {
        if (!device.init(error)) return false;
        handles.size = static_cast<u32>(ram.size());
        renderer = std::make_unique<VulkanRenderer>(device, handles, ram.data(), static_cast<u32>(ram.size()));
        if (!renderer->init(error)) return false;
        renderer->set_resolution_scale(scale);
        k.set_sink(renderer.get());
        return true;
    }
};

// The triangle of RendersATriangleIntoGuestRam (green over the lower-right half of the 64x64 target, on dark blue).
void draw_half_triangle(Kelvin& k) {
    k.method(mthd::kColorClear, 0xFF000080);
    k.method(mthd::kZstencilClear, 0xFFFFFF00);
    k.method(mthd::kClearRectH, 63u << 16);
    k.method(mthd::kClearRectV, 63u << 16);
    k.method(mthd::kClearSurface, 0xF3);
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kDiffuse4ub, 0xFF00FF00);
    for (const auto& p : {std::array<float, 2>{63.0f, 0.0f}, std::array<float, 2>{63.0f, 63.0f}, std::array<float, 2>{0.0f, 63.0f}}) {
        k.method(mthd::kVertexData4fM + 0, bits(p[0]));
        k.method(mthd::kVertexData4fM + 4, bits(p[1]));
        k.method(mthd::kVertexData4fM + 8, bits(0.5f * 16777215.0f));
        k.method(mthd::kVertexData4fM + 12, bits(1.0f));
    }
    k.method(mthd::kBeginEnd, 0);
}

}  // namespace

// What a scaled target leaves in RAM is what a 1x target would: clears in every format, a flat draw into a two-byte
// format, and depth with its stencil.
TEST(Vulkan, AScaledTargetReachesRamAsA1xTargetWould) {
    for (u32 scale : {2u, 3u, 4u, 8u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        ScaledFixture f;
        std::string error;
        if (!f.init(scale, error)) GTEST_SKIP() << "no Vulkan device: " << error;
        Kelvin& k = f.k;
        auto clear = [&](u32 format, u32 offset, u32 value, u32 bits) {
            setup_target(k, format, offset);
            k.method(mthd::kColorClear, value);
            k.method(mthd::kClearRectH, 63u << 16);
            k.method(mthd::kClearRectV, 63u << 16);
            k.method(mthd::kClearSurface, bits);
        };
        auto word16 = [&](u32 offset, u32 x, u32 y) {
            u16 v;
            std::memcpy(&v, &f.ram[offset + y * 256 + x * 2], 2);
            return v;
        };
        auto word32 = [&](u32 offset, u32 x, u32 y) {
            u32 v;
            std::memcpy(&v, &f.ram[offset + y * 256 + x * 4], 4);
            return v;
        };
        clear(3, 0x100000, 0x0000F81F, 0xF0);
        clear(1, 0x110000, 0xFFFF7C00, 0xF0);
        clear(4, 0x130000, 0xFF102030, 0xF0);
        clear(8, 0x150000, 0x11223344, 0xF0);
        clear(8, 0x150000, 0xAABBCCDD, 0x90);
        setup_target(k, 10, 0x160000);  // G8B8, drawn flat
        std::memset(&f.ram[0x160000], 0xEE, 64 * 256);
        k.method(mthd::kVertexDataArrayFormat + 12, 2u);
        k.method(mthd::kDiffuse4ub, 0xFF208040);
        k.method(mthd::kColorClear, 0);
        k.method(mthd::kClearSurface, 0xF0);
        draw_full_triangle(k, f.ram);
        setup_target(k);  // depth and stencil
        k.method(mthd::kZstencilClear, 0x80000042);
        k.method(mthd::kClearSurface, 0x03);
        f.renderer->end_frame();
        f.renderer->write_back_all();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        for (u32 at : {0u, 8u, 63u}) {
            EXPECT_EQ(word16(0x100000, at, at), 0xF81Fu);
            EXPECT_EQ(word16(0x110000, at, at), 0x7C00u);
            EXPECT_EQ(word32(0x130000, at, at), 0x00102030u);
            EXPECT_EQ(word32(0x150000, at, at), 0xAABB3344u);
            EXPECT_EQ(f.ram[0x160000 + at * 256 + at * 2 + 1], 0x80);
            EXPECT_EQ(word32(0x20000, at, at), 0x80000042u) << "the zeta word: depth over the stencil byte";
        }
        EXPECT_EQ(f.ram[0x160000 + 8 * 256 + 128], 0xEE) << "nothing written past the G8B8 line's 64 pixels";
        EXPECT_EQ(f.ram[0x150000 + 64 * 256], 0) << "nothing written past the surface";
    }
}

// A scaled target loads what RAM holds (each pixel widened to a block) and writes back exactly those bytes around
// what it drew: colour and depth.
TEST(Vulkan, AScaledTargetKeepsRamsPixelsAroundADraw) {
    for (u32 scale : {2u, 3u, 4u, 8u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        ScaledFixture f;
        std::string error;
        if (!f.init(scale, error)) GTEST_SKIP() << "no Vulkan device: " << error;
        Kelvin& k = f.k;
        auto pattern = [](u32 x, u32 y) { return 0xFF000000u | (x * 4) << 16 | (y * 4) << 8 | ((x ^ y) * 4); };
        auto zpattern = [](u32 x, u32 y) { return ((x * 977u + y * 131071u) & 0xFFFFFFu) << 8 | ((x + 3 * y) & 0xFFu); };
        for (u32 y = 0; y < 64; ++y)
            for (u32 x = 0; x < 64; ++x) {
                const u32 c = pattern(x, y), z = zpattern(x, y);
                std::memcpy(&f.ram[0x10000 + y * 256 + x * 4], &c, 4);
                std::memcpy(&f.ram[0x20000 + y * 256 + x * 4], &z, 4);
            }
        setup_target(k);
        k.method(mthd::kDepthTestEnable, 1);
        k.method(mthd::kDepthFunc, 0x207);  // ALWAYS: the triangle writes its depth, the rest keeps RAM's
        k.method(mthd::kDepthMask, 1);
        k.method(mthd::kBeginEnd, 5);
        k.method(mthd::kDiffuse4ub, 0xFF00FF00);
        for (const auto& p : {std::array<float, 2>{0.0f, 0.0f}, std::array<float, 2>{16.0f, 0.0f}, std::array<float, 2>{0.0f, 16.0f}}) {
            k.method(mthd::kVertexData4fM + 0, bits(p[0]));
            k.method(mthd::kVertexData4fM + 4, bits(p[1]));
            k.method(mthd::kVertexData4fM + 8, bits(0.25f * 16777215.0f));
            k.method(mthd::kVertexData4fM + 12, bits(1.0f));
        }
        k.method(mthd::kBeginEnd, 0);
        f.renderer->end_frame();
        f.renderer->write_back_all();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        u32 colour_misses = 0, depth_misses = 0;
        for (u32 y = 0; y < 64; ++y)
            for (u32 x = 0; x < 64; ++x) {
                if (x + y < 20) continue;  // the triangle and its edge
                u32 c, z;
                std::memcpy(&c, &f.ram[0x10000 + y * 256 + x * 4], 4);
                std::memcpy(&z, &f.ram[0x20000 + y * 256 + x * 4], 4);
                colour_misses += c != pattern(x, y);
                depth_misses += z != zpattern(x, y);
            }
        EXPECT_EQ(colour_misses, 0u);
        EXPECT_EQ(depth_misses, 0u);
        EXPECT_EQ(pixel_at(f.ram, 2, 2), 0xFF00FF00u) << "inside the triangle";
        u32 z;
        std::memcpy(&z, &f.ram[0x20000 + 2 * 256 + 2 * 4], 4);
        EXPECT_NEAR(static_cast<double>(z >> 8), 0x400000, 2.0) << "the triangle's depth (a quarter of the range)";
        EXPECT_EQ(z & 0xFFu, zpattern(2, 2) & 0xFFu) << "its stencil kept";
    }
}

// The picture of the scanned-out target is the scaled image: a diagonal edge steps on the image's pixels, not on
// blocks of them (a 1x picture widened has every edge on a block boundary).
TEST(Vulkan, TheScanoutPictureIsTheScaledImage) {
    ScaledFixture f;
    std::string error;
    if (!f.init(2, error)) GTEST_SKIP() << "no Vulkan device: " << error;
    setup_target(f.k);
    draw_half_triangle(f.k);
    f.renderer->flush_frame();
    ASSERT_TRUE(f.renderer->write_back_at(0x10000));
    f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    std::vector<u32> px;
    u32 scale = 0;
    ASSERT_TRUE(f.renderer->scaled_scanout(0x10000, 256, 4, false, 64, 64, px, scale));
    EXPECT_EQ(scale, 2u);
    ASSERT_EQ(px.size(), 128u * 128u);
    auto at = [&](u32 x, u32 y) {
        const u32 v = px[y * 128 + x];
        return v & 0x00FFFFFFu;
    };
    EXPECT_EQ(at(8, 8), 0x000080u);
    EXPECT_EQ(at(120, 120), 0x00FF00u);
    u32 odd_edges = 0, green = 0, green_1x = 0;
    for (u32 y = 0; y < 128; ++y) {
        u32 first = 128;
        for (u32 x = 0; x < 128; ++x) {
            if (at(x, y) == 0x00FF00u) {
                ++green;
                first = std::min(first, x);
            }
        }
        odd_edges += first < 128 && first % 2 == 1;
    }
    for (u32 y = 0; y < 64; ++y)
        for (u32 x = 0; x < 64; ++x) green_1x += (pixel_at(f.ram, x, y) & 0x00FFFFFFu) == 0x00FF00u;
    EXPECT_GT(odd_edges, 16u) << "the edge steps on the image's own pixels";
    EXPECT_NEAR(static_cast<double>(green), 4.0 * green_1x, 2.0 * 128) << "the same triangle, four times the pixels (within an edge)";
    EXPECT_FALSE(f.renderer->scaled_scanout(0x10100, 256, 4, false, 64, 64, px, scale)) << "a scanout one line down runs past the target";
    // Another engine writes the target's memory: RAM holds the picture until the target is drawn again.
    f.renderer->invalidate(0x10000, 64 * 256);
    ASSERT_TRUE(f.renderer->write_back_at(0x10000));
    f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
    EXPECT_FALSE(f.renderer->scaled_scanout(0x10000, 256, 4, false, 64, 64, px, scale));
}

// The scale changes while targets live: at the vblank they reach RAM and go, and come back from RAM at the new size.
TEST(Vulkan, TheInternalResolutionChangesAtTheVblank) {
    ScaledFixture f;
    std::string error;
    if (!f.init(1, error)) GTEST_SKIP() << "no Vulkan device: " << error;
    setup_target(f.k);
    draw_half_triangle(f.k);
    f.renderer->flush_frame();
    ASSERT_TRUE(f.renderer->write_back_at(0x10000));
    f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
    std::vector<u32> px;
    u32 scale = 0;
    EXPECT_FALSE(f.renderer->scaled_scanout(0x10000, 256, 4, false, 64, 64, px, scale)) << "1x: RAM holds the picture";
    const std::vector<u8> before(f.ram.begin() + 0x10000, f.ram.begin() + 0x10000 + 64 * 256);
    f.renderer->set_resolution_scale(3);
    EXPECT_FALSE(f.renderer->write_back_at(0x10000)) << "the targets went at the change";
    EXPECT_TRUE(std::equal(before.begin(), before.end(), f.ram.begin() + 0x10000)) << "RAM kept what they drew";
    f.k.method(mthd::kClearRectH, 15u << 16);  // a corner cleared at the new scale, the rest from RAM
    f.k.method(mthd::kClearRectV, 15u << 16);
    f.k.method(mthd::kColorClear, 0xFFFF0000);
    f.k.method(mthd::kClearSurface, 0xF0);
    f.renderer->flush_frame();
    ASSERT_TRUE(f.renderer->write_back_at(0x10000));
    f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    ASSERT_TRUE(f.renderer->scaled_scanout(0x10000, 256, 4, false, 64, 64, px, scale));
    EXPECT_EQ(scale, 3u);
    EXPECT_EQ(pixel_at(f.ram, 4, 4), 0xFFFF0000u);
    EXPECT_EQ(pixel_at(f.ram, 60, 60) & 0x00FFFFFFu, 0x00FF00u) << "the triangle came back from RAM";
}

// A scaled depth buffer reaches RAM through its stored-size copy, blitted down on the GPU: the same bytes as the rows
// read back one copy region each and thinned on the CPU (MV_RENDER_DEBUG=depthrows), for a sloped depth in 16 and 24
// bits with stencil. At an odd scale exactly; at an even one the device may take either texel beside the block's middle.
TEST(Vulkan, AScaledDepthBufferReachesRamThroughItsStoredSizeCopy) {
    for (u32 scale : {3u, 6u})
        for (u32 zfmt : {1u, 2u}) {
            SCOPED_TRACE(std::format("scale {} zeta format {}", scale, zfmt));
            const float zmax = zfmt == 1 ? 65535.0f : 16777215.0f;
            auto depth_in_ram = [&](bool rows, std::vector<u8>& out) {
                if (rows) ::setenv("MV_RENDER_DEBUG", "depthrows", 1);
                ScaledFixture f;
                std::string error;
                const bool ok = f.init(scale, error);
                ::unsetenv("MV_RENDER_DEBUG");
                if (!ok) return false;
                Kelvin& k = f.k;
                setup_target(k);
                k.method(mthd::kSurfaceFormat, (zfmt << 4) | 0x100 | 8);
                k.method(mthd::kZstencilClear, zfmt == 1 ? 0xFFFFu : 0xFFFFFF5Au);
                k.method(mthd::kClearRectH, 63u << 16);
                k.method(mthd::kClearRectV, 63u << 16);
                k.method(mthd::kClearSurface, 0x03);
                k.method(mthd::kDepthTestEnable, 1);
                k.method(mthd::kDepthFunc, 0x207);
                k.method(mthd::kDepthMask, 1);
                k.method(mthd::kBeginEnd, 5);
                k.method(mthd::kDiffuse4ub, 0xFF00FF00);
                for (const auto& p : {std::array<float, 3>{0.0f, 0.0f, 0.2f}, std::array<float, 3>{128.0f, 0.0f, 0.8f}, std::array<float, 3>{0.0f, 128.0f, 0.5f}}) {
                    k.method(mthd::kVertexData4fM + 0, bits(p[0]));
                    k.method(mthd::kVertexData4fM + 4, bits(p[1]));
                    k.method(mthd::kVertexData4fM + 8, bits(p[2] * zmax));
                    k.method(mthd::kVertexData4fM + 12, bits(1.0f));
                }
                k.method(mthd::kBeginEnd, 0);
                f.renderer->end_frame();
                f.renderer->write_back_all();
                for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
                out.assign(f.ram.begin() + 0x20000, f.ram.begin() + 0x20000 + 64 * 256);
                return true;
            };
            std::vector<u8> blit, rows;
            if (!depth_in_ram(false, blit) || !depth_in_ram(true, rows)) GTEST_SKIP() << "no Vulkan device";
            const u32 bpp = zfmt == 1 ? 2u : 4u;
            u32 misses = 0, far = 0;
            u64 worst = 0;
            for (u32 y = 0; y < 64; ++y)
                for (u32 x = 0; x < 64; ++x) {
                    u32 a = 0, b = 0;
                    std::memcpy(&a, &blit[y * 256 + x * bpp], bpp);
                    std::memcpy(&b, &rows[y * 256 + x * bpp], bpp);
                    if (a == b) continue;
                    ++misses;
                    if (zfmt == 2 && (a & 0xFF) != (b & 0xFF)) ++far;  // the stencil is flat: never differs
                    const u64 d = zfmt == 1 ? static_cast<u64>(a > b ? a - b : b - a) : static_cast<u64>((a >> 8) > (b >> 8) ? (a >> 8) - (b >> 8) : (b >> 8) - (a >> 8));
                    worst = std::max(worst, d);
                }
            EXPECT_EQ(far, 0u);
            u32 z;
            std::memcpy(&z, &blit[30 * 256 + 10 * bpp], bpp);
            if (zfmt == 1) z &= 0xFFFF;
            EXPECT_NE(z, zfmt == 1 ? 0xFFFFu : 0xFFFFFF5Au) << "the triangle's depth reached RAM";
            if (scale % 2) {
                EXPECT_EQ(misses, 0u);
            } else {
                // One texel of the slope at the internal resolution: (0.6 + 0.3) / 128 of the range over sc texels.
                EXPECT_LE(worst, static_cast<u64>(zmax * 0.9f / 128.0f / static_cast<float>(scale)) + 2u);
            }
        }
}

// A large target at a high scale: its depth rows do not fit the read-back buffer at once (1024 x 1024 at 4x is 21 MB of
// rows) and reach RAM in bands, exactly.
TEST(Vulkan, AScaledDepthBufferLargerThanTheReadBackReachesRamInBands) {
    ScaledFixture f;
    std::string error;
    if (!f.init(4, error)) GTEST_SKIP() << "no Vulkan device: " << error;
    Kelvin& k = f.k;
    constexpr u32 kColor = 0x100000, kZeta = 0x600000, kSize = 1024, kPitch = kSize * 4;
    auto zpattern = [](u32 x, u32 y) { return ((x * 977u + y * 131071u) & 0xFFFFFFu) << 8 | ((x + 3 * y) & 0xFFu); };
    for (u32 y = 0; y < kSize; ++y)
        for (u32 x = 0; x < kSize; ++x) {
            const u32 z = zpattern(x, y);
            std::memcpy(&f.ram[kZeta + y * kPitch + x * 4], &z, 4);
        }
    setup_target(k, 8, kColor);
    k.method(mthd::kSurfaceClipH, kSize << 16);
    k.method(mthd::kSurfaceClipV, kSize << 16);
    k.method(mthd::kSurfacePitch, (kPitch << 16) | kPitch);
    k.method(mthd::kSurfaceZetaOffset, kZeta);
    k.method(mthd::kDepthTestEnable, 1);
    k.method(mthd::kDepthFunc, 0x207);
    k.method(mthd::kDepthMask, 1);
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kDiffuse4ub, 0xFF00FF00);
    for (const auto& p : {std::array<float, 2>{0.0f, 0.0f}, std::array<float, 2>{16.0f, 0.0f}, std::array<float, 2>{0.0f, 16.0f}}) {
        k.method(mthd::kVertexData4fM + 0, bits(p[0]));
        k.method(mthd::kVertexData4fM + 4, bits(p[1]));
        k.method(mthd::kVertexData4fM + 8, bits(0.25f * 16777215.0f));
        k.method(mthd::kVertexData4fM + 12, bits(1.0f));
    }
    k.method(mthd::kBeginEnd, 0);
    f.renderer->end_frame();
    f.renderer->write_back_all();
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    u32 misses = 0;
    for (u32 y = 0; y < kSize; ++y)
        for (u32 x = 0; x < kSize; ++x) {
            if (x + y < 20) continue;
            u32 z;
            std::memcpy(&z, &f.ram[kZeta + y * kPitch + x * 4], 4);
            misses += z != zpattern(x, y);
        }
    EXPECT_EQ(misses, 0u);
    u32 z;
    std::memcpy(&z, &f.ram[kZeta + 2 * kPitch + 2 * 4], 4);
    EXPECT_NEAR(static_cast<double>(z >> 8), 0x400000, 2.0) << "the triangle's depth";
}

// An environment map drawn each frame: six swizzled targets, then a cube texture over them. The faces are copied into a
// cube image on the GPU (at any internal resolution); nothing goes through RAM.
TEST(Vulkan, ACubeMapOfSixRenderTargetsIsPutTogetherOnTheGpu) {
    for (u32 scale : {1u, 2u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        TextureShaderFixture f;
        std::string error;
        if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
        f.renderer->set_resolution_scale(scale);
        Kelvin& k = f.k;
        constexpr u32 kCube = 0x200000, kFace = 16 * 16 * 4;
        const u32 colours[6] = {0xFFFF0000u, 0xFF00FF00u, 0xFF0000FFu, 0xFFFFFF00u, 0xFF00FFFFu, 0xFFFF00FFu};
        for (u32 face = 0; face < 6; ++face) {
            k.method(mthd::kSurfaceFormat, (4u << 24) | (4u << 16) | 0x200u | 0x8u);  // swizzled 16x16 A8R8G8B8, no depth
            k.method(mthd::kSurfaceClipH, 16u << 16);
            k.method(mthd::kSurfaceClipV, 16u << 16);
            k.method(mthd::kSurfaceColorOffset, kCube + face * kFace);
            k.method(mthd::kColorClear, colours[face]);
            k.method(mthd::kClearRectH, 15u << 16);
            k.method(mthd::kClearRectV, 15u << 16);
            k.method(mthd::kClearSurface, 0xF0);
        }
        setup_target(k);
        load_texcoord_program(k);
        const u32 b = mthd::kTextureBase;
        k.method(b + 0x0, kCube);
        k.method(b + 0x4, (4u << 24) | (4u << 20) | (1u << 16) | (0x06u << 8) | (2u << 4) | (1u << 2) | 1u);  // a 16x16 cube map
        k.method(b + 0x8, 0x00030303);
        k.method(b + 0xC, 0x40000000);
        k.method(b + 0x14, (1u << 24) | (1u << 16));
        k.method(mthd::kShaderStageProgram, 3u);  // stage 0: cube map
        output_texture_stage(k, 0);
        const float dirs[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (u32 face = 0; face < 6; ++face) {
            texcoord(k, 0, dirs[face][0], dirs[face][1], dirs[face][2], 1.0f);
            draw_full_triangle(k, f.ram);
            f.renderer->end_frame();
            EXPECT_EQ(pixel_at(f.ram, 8, 8), colours[face]) << "face " << face;
        }
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        EXPECT_GE(f.renderer->stats().cube_assemblies, 1u);
        EXPECT_EQ(f.renderer->stats().writebacks_by[VulkanRenderer::Stats::kWbTexture], 0u) << "no face went through RAM";
    }
}

// The 2D engine copying a rectangle between two render targets: an image copy on the GPU (no write-back, no reload),
// and RAM shows the same result once the destination is written back.
TEST(Vulkan, A2DCopyBetweenTwoTargetsStaysOnTheGpu) {
    for (u32 scale : {1u, 2u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        ScaledFixture f;
        std::string error;
        if (!f.init(scale, error)) GTEST_SKIP() << "no Vulkan device: " << error;
        Kelvin& k = f.k;
        auto fill = [&](u32 offset, u32 colour) {
            setup_target(k, 8, offset);
            k.method(mthd::kColorClear, colour);
            k.method(mthd::kClearRectH, 63u << 16);
            k.method(mthd::kClearRectV, 63u << 16);
            k.method(mthd::kClearSurface, 0xF0);
        };
        fill(0x10000, 0xFFFF0000u);  // red
        fill(0x40000, 0xFF0000FFu);  // blue
        // 16x8 pixels from (4, 2) of the red target to (40, 30) of the blue one.
        ASSERT_TRUE(f.renderer->copy_rect(0x10000 + 2 * 256 + 4 * 4, 256, 0x40000 + 30 * 256 + 40 * 4, 256, 16, 8, 4));
        f.renderer->end_frame();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        auto px = [&](u32 x, u32 y) {
            u32 v;
            std::memcpy(&v, &f.ram[0x40000 + y * 256 + x * 4], 4);
            return v;
        };
        EXPECT_EQ(px(40, 30), 0xFFFF0000u);
        EXPECT_EQ(px(55, 37), 0xFFFF0000u) << "the rectangle's last pixel";
        EXPECT_EQ(px(56, 37), 0xFF0000FFu) << "right of it";
        EXPECT_EQ(px(40, 38), 0xFF0000FFu) << "below it";
        EXPECT_EQ(px(39, 30), 0xFF0000FFu) << "left of it";
        EXPECT_EQ(f.renderer->stats().gpu_copies, 1u);
        EXPECT_EQ(f.renderer->stats().writebacks_by[VulkanRenderer::Stats::kWbRange], 0u);
        // A source the CPU has written since: RAM has to carry the copy.
        f.renderer->invalidate(0x10000, 64 * 256);
        EXPECT_FALSE(f.renderer->copy_rect(0x10000, 256, 0x40000, 256, 16, 8, 4));
    }
}

// A screen-space quad whose left and top edges lie on pixel centres (a blur pass on the half-pixel convention) covers the
// first column and row whole at 1x; at 6x it must cover their whole blocks, not half of them (Fable's blur buffers kept
// three image columns of stale memory, shown as green squares along the picture's edges).
TEST(Vulkan, AScreenSpaceEdgeOnAPixelCentreCoversWholeBlocksAboveOneX) {
    ScaledFixture f;
    std::string error;
    if (!f.init(6, error)) GTEST_SKIP() << "no Vulkan device: " << error;
    Kelvin& k = f.k;
    setup_target(k);
    k.method(mthd::kColorClear, 0xFF0000FFu);  // blue underneath
    k.method(mthd::kClearRectH, 63u << 16);
    k.method(mthd::kClearRectV, 63u << 16);
    k.method(mthd::kClearSurface, 0xF0);
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kDiffuse4ub, 0xFF00FF00);
    // Two triangles from (0.49, 0.49) to (64.5, 64.5): a hair off the centres, as the vertex maths leaves them.
    for (const auto& p : {std::array<float, 2>{0.49f, 0.49f}, std::array<float, 2>{64.5f, 0.49f}, std::array<float, 2>{0.49f, 64.5f},
                          std::array<float, 2>{64.5f, 0.49f}, std::array<float, 2>{64.5f, 64.5f}, std::array<float, 2>{0.49f, 64.5f}}) {
        k.method(mthd::kVertexData4fM + 0, bits(p[0]));
        k.method(mthd::kVertexData4fM + 4, bits(p[1]));
        k.method(mthd::kVertexData4fM + 8, bits(0.0f));
        k.method(mthd::kVertexData4fM + 12, bits(1.0f));
    }
    k.method(mthd::kBeginEnd, 0);
    f.renderer->flush_frame();
    ASSERT_TRUE(f.renderer->write_back_at(0x10000));
    f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    std::vector<u32> px;
    u32 scale = 0;
    ASSERT_TRUE(f.renderer->scaled_scanout(0x10000, 256, 4, false, 64, 64, px, scale));
    ASSERT_EQ(scale, 6u);
    u32 blue = 0;
    for (u32 y = 0; y < 384; ++y)
        for (u32 x = 0; x < 384; ++x) blue += (px[y * 384 + x] & 0x00FFFFFFu) == 0x0000FFu;
    EXPECT_EQ(blue, 0u) << "every image pixel covered, the first column's and row's blocks whole";
    EXPECT_EQ(pixel_at(f.ram, 0, 0) & 0x00FFFFFFu, 0x00FF00u) << "and the write-back at the console's size";
}

// A 16-bit render target sampled right after it was drawn: read in place (R5G6B5, and X1R5G5B5 with alpha one), not
// written back and decoded (Fable samples its 16-bit picture every frame).
TEST(Vulkan, A16BitTargetIsSampledInPlace) {
    for (u32 scale : {1u, 4u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        TextureShaderFixture f;
        std::string error;
        if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
        f.renderer->set_resolution_scale(scale);
        Kelvin& k = f.k;
        struct Case {
            u32 surface_format, texture_format, clear, expect;
        };
        u32 n = 0;
        for (const Case c : {Case{3, 0x11, 0x0000F81Fu, 0xFFFF00FFu}, Case{1, 0x1C, 0xFFFF7C00u, 0xFFFF0000u}}) {
            const u32 at = 0x100000 + n++ * 0x10000;
            setup_target(k, c.surface_format, at);
            k.method(mthd::kColorClear, c.clear);
            k.method(mthd::kClearRectH, 63u << 16);
            k.method(mthd::kClearRectV, 63u << 16);
            k.method(mthd::kClearSurface, 0xF0);
            setup_target(k);
            load_texcoord_program(k);
            const u32 b = mthd::kTextureBase;
            k.method(b + 0x0, at);
            k.method(b + 0x4, (1u << 16) | (c.texture_format << 8) | (2u << 4) | 1u);  // a linear (rect) texture
            k.method(b + 0x8, 0x00030303);
            k.method(b + 0xC, 0x40000000);
            k.method(b + 0x10, 256u << 16);  // its pitch
            k.method(b + 0x14, (1u << 24) | (1u << 16));
            k.method(b + 0x1C, (64u << 16) | 64u);  // its size
            k.method(mthd::kShaderStageProgram, 1u);
            output_texture_stage(k, 0);
            texcoord(k, 0, 8.0f, 8.0f, 0.0f, 1.0f);  // texels
            draw_full_triangle(k, f.ram);
            f.renderer->end_frame();
            EXPECT_EQ(pixel_at(f.ram, 8, 8), c.expect) << "texture format " << c.texture_format;
        }
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        EXPECT_EQ(f.renderer->stats().writebacks_by[VulkanRenderer::Stats::kWbTexture], 0u) << "nothing went through RAM";
    }
}

// A DXT1 texture uploaded as its blocks (BC1) reads as the CPU decoder reads it: the four-colour blocks, and the
// three-colour ones with index 3 transparent black. Checked against the decoder with MV_NATIVE_DXT=0 as well.
TEST(Vulkan, ADxtTextureSampledAsBlocksMatchesTheDecoder) {
    // Four 4x4 blocks (an 8x8 texture): four-colour red/blue at index 0, 2 and 3, and a three-colour block at index 3.
    const auto block = [](u16 c0, u16 c1, u8 index) {
        std::array<u8, 8> b{static_cast<u8>(c0), static_cast<u8>(c0 >> 8), static_cast<u8>(c1), static_cast<u8>(c1 >> 8)};
        const u8 row = static_cast<u8>(index | (index << 2) | (index << 4) | (index << 6));
        for (u32 i = 4; i < 8; ++i) b[i] = row;
        return b;
    };
    const std::array<std::array<u8, 8>, 4> blocks{block(0xF800, 0x001F, 0), block(0xF800, 0x001F, 2), block(0xF800, 0x001F, 3), block(0x001F, 0xF800, 3)};
    std::vector<u8> expect_rgba(8 * 8 * 4);
    std::vector<u8> raw(blocks.size() * 8);  // filled by copies (GCC 14 misreads a growing insert as an overflow)
    for (std::size_t i = 0; i < blocks.size(); ++i) std::memcpy(raw.data() + i * 8, blocks[i].data(), 8);
    pgraph::decode_dxt1(raw.data(), 8, 8, expect_rgba.data());
    for (const bool native : {true, false}) {
        SCOPED_TRACE(native ? "as blocks" : "decoded on the CPU");
        if (native) unsetenv("MV_NATIVE_DXT");
        else setenv("MV_NATIVE_DXT", "0", 1);
        TextureShaderFixture f;
        std::string error;
        if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
        Kelvin& k = f.k;
        std::memcpy(&f.ram[0x200000], raw.data(), raw.size());
        load_texcoord_program(k);
        const u32 b = mthd::kTextureBase;
        k.method(b + 0x0, 0x200000);
        k.method(b + 0x4, (3u << 24) | (3u << 20) | (1u << 16) | (0x0Cu << 8) | (2u << 4) | 1u);  // 8x8 DXT1, one level
        k.method(b + 0x8, 0x00030303);
        k.method(b + 0xC, 0x40000000);
        k.method(b + 0x14, (1u << 24) | (1u << 16));  // nearest
        k.method(mthd::kShaderStageProgram, 1u);
        output_texture_stage(k, 0);
        for (u32 i = 0; i < 4; ++i) {
            const u32 bx = i % 2, by = i / 2;
            texcoord(k, 0, (static_cast<float>(bx) * 4.0f + 2.0f) / 8.0f, (static_cast<float>(by) * 4.0f + 2.0f) / 8.0f, 0.0f, 1.0f);
            draw_full_triangle(k, f.ram);
            f.renderer->end_frame();
            const u8* e = &expect_rgba[((by * 4 + 2) * 8 + bx * 4 + 2) * 4];
            const u32 got = pixel_at(f.ram, 10, 10);
            const int gr = static_cast<int>((got >> 16) & 0xFF), gg = static_cast<int>((got >> 8) & 0xFF), gb = static_cast<int>(got & 0xFF), ga = static_cast<int>(got >> 24);
            EXPECT_NEAR(gr, e[0], 3) << "block " << i << " red";
            EXPECT_NEAR(gg, e[1], 3) << "block " << i << " green";
            EXPECT_NEAR(gb, e[2], 3) << "block " << i << " blue";
            EXPECT_NEAR(ga, e[3], 3) << "block " << i << " alpha";
        }
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        if (native && f.device.bc_textures()) {
            EXPECT_EQ(f.renderer->stats().textures_native_dxt, 1u) << "uploaded as blocks";
        }
        if (!native) {
            EXPECT_EQ(f.renderer->stats().textures_native_dxt, 0u);
        }
    }
    unsetenv("MV_NATIVE_DXT");
}

// A mip chain drawn as one target per level, then sampled as one mip-mapped texture: put together on the GPU (at any
// internal resolution), nothing through RAM.
TEST(Vulkan, AMipChainOfRenderTargetsIsPutTogetherOnTheGpu) {
    for (u32 scale : {1u, 2u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        TextureShaderFixture f;
        std::string error;
        if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
        f.renderer->set_resolution_scale(scale);
        Kelvin& k = f.k;
        constexpr u32 kChain = 0x200000;
        const u32 colours[3] = {0xFFFF0000u, 0xFF00FF00u, 0xFF0000FFu};
        u32 at = kChain;
        for (u32 level = 0; level < 3; ++level) {
            const u32 log = 4 - level;
            k.method(mthd::kSurfaceFormat, (log << 24) | (log << 16) | 0x200u | 0x8u);  // swizzled A8R8G8B8, no depth
            k.method(mthd::kSurfaceClipH, (1u << log) << 16);
            k.method(mthd::kSurfaceClipV, (1u << log) << 16);
            k.method(mthd::kSurfaceColorOffset, at);
            k.method(mthd::kColorClear, colours[level]);
            k.method(mthd::kClearRectH, ((1u << log) - 1) << 16);
            k.method(mthd::kClearRectV, ((1u << log) - 1) << 16);
            k.method(mthd::kClearSurface, 0xF0);
            at += (1u << log) * (1u << log) * 4;
        }
        setup_target(k);
        load_texcoord_program(k);
        const u32 b = mthd::kTextureBase;
        k.method(b + 0x0, kChain);
        k.method(b + 0x4, (4u << 24) | (4u << 20) | (3u << 16) | (0x06u << 8) | (2u << 4) | 1u);  // 16x16, three levels
        k.method(b + 0x8, 0x00030303);
        k.method(b + 0xC, 0x40000000);
        k.method(b + 0x14, (1u << 24) | (1u << 16));
        k.method(mthd::kShaderStageProgram, 1u);
        output_texture_stage(k, 0);
        texcoord(k, 0, 0.5f, 0.5f, 0.0f, 1.0f);
        draw_full_triangle(k, f.ram);
        f.renderer->end_frame();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        EXPECT_EQ(pixel_at(f.ram, 8, 8), colours[0]) << "level 0, magnified";
        EXPECT_GE(f.renderer->stats().mip_assemblies, 1u);
        EXPECT_EQ(f.renderer->stats().writebacks_by[VulkanRenderer::Stats::kWbTexture], 0u) << "no level went through RAM";
    }
}

// Above 1x the scanout's copy in RAM lands in the background (the vblank no longer waits for the GPU): it reaches RAM
// when completed, a page the CPU wrote meanwhile keeps the CPU's bytes, and a target drawn into meanwhile stays dirty
// so the next write-back brings the newer picture.
TEST(Vulkan, AScanoutWrittenBackInTheBackgroundLandsUnderTheCpusPages) {
    ScaledFixture f;
    std::string error;
    if (!f.init(2, error)) GTEST_SKIP() << "no Vulkan device: " << error;
    Kelvin& k = f.k;
    setup_target(k);
    k.method(mthd::kColorClear, 0xFF102030u);
    k.method(mthd::kClearRectH, 63u << 16);
    k.method(mthd::kClearRectV, 63u << 16);
    k.method(mthd::kClearSurface, 0xF0);
    f.renderer->flush_frame();
    ASSERT_TRUE(f.renderer->write_back_at(0x10000));
    EXPECT_TRUE(f.renderer->write_backs_in_flight()) << "queued without a wait";
    EXPECT_EQ(pixel_at(f.ram, 5, 5), 0u) << "nothing in RAM before it lands";
    // The CPU writes one page of the frame buffer (rows 16-31 of a 256-byte pitch) before the copy lands.
    for (u32 i = 0; i < 4096; i += 4) {
        const u32 v = 0xFFABCDEFu;
        std::memcpy(&f.ram[0x11000 + i], &v, 4);
    }
    // And a draw goes into the target before it lands: the copy in flight is the older picture.
    k.method(mthd::kColorClear, 0xFF405060u);
    k.method(mthd::kClearSurface, 0xF0);
    const std::vector<u32> cpu_pages{0x11};
    ASSERT_TRUE(f.renderer->complete_write_backs(true, &cpu_pages));
    EXPECT_FALSE(f.renderer->write_backs_in_flight());
    EXPECT_EQ(pixel_at(f.ram, 5, 5), 0xFF102030u) << "the copy landed";
    EXPECT_EQ(pixel_at(f.ram, 5, 20), 0xFFABCDEFu) << "the CPU's page kept its bytes";
    f.renderer->write_back_all();
    EXPECT_EQ(pixel_at(f.ram, 5, 5), 0xFF405060u) << "drawn into meanwhile: still dirty, the newer picture follows";
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    EXPECT_GE(f.renderer->stats().background_write_back_batches, 1u);
}

// Above 1x the picture is converted on the GPU as it is read back: XRGB8888, the RAMDAC's colour table applied.
TEST(Vulkan, TheScaledPictureComesBackThroughTheColourTable) {
    ScaledFixture f;
    std::string error;
    if (!f.init(2, error)) GTEST_SKIP() << "no Vulkan device: " << error;
    setup_target(f.k);
    f.k.method(mthd::kColorClear, 0xFF102030u);
    f.k.method(mthd::kClearRectH, 63u << 16);
    f.k.method(mthd::kClearRectV, 63u << 16);
    f.k.method(mthd::kClearSurface, 0xF0);
    std::array<u8, 768> invert{};
    for (u32 i = 0; i < 768; ++i) invert[i] = static_cast<u8>(255 - i / 3);
    f.renderer->set_display_lut(invert);
    f.renderer->flush_frame();
    ASSERT_TRUE(f.renderer->write_back_at(0x10000));
    f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
    VulkanRenderer::ScaledPicture p;
    ASSERT_TRUE(f.renderer->scaled_picture(0x10000, 256, 4, 64, 64, p));
    EXPECT_TRUE(p.xrgb);
    f.renderer->picture_ready(p);
    u32 v;
    std::memcpy(&v, p.pixels + 10 * p.pitch + 10 * 4, 4);
    f.renderer->picture_done();
    EXPECT_EQ(v, 0x00EFDFCFu) << "each channel through the table, the unused byte clear";
    EXPECT_EQ(pixel_at(f.ram, 5, 5), 0xFF102030u) << "RAM keeps the frame buffer's own values";
}

namespace {

// A triangle over the whole 64x64 target at one depth (a fraction of the 24-bit range), tested LESS and written.
void depth_triangle(Kelvin& k, u32 abgr, float depth) {
    k.method(mthd::kDepthTestEnable, 1);
    k.method(mthd::kDepthFunc, 0x201);  // LESS
    k.method(mthd::kDepthMask, 1);
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kDiffuse4ub, abgr);
    for (const auto& p : {std::array<float, 2>{0.0f, 0.0f}, std::array<float, 2>{128.0f, 0.0f}, std::array<float, 2>{0.0f, 128.0f}}) {
        k.method(mthd::kVertexData4fM + 0, bits(p[0]));
        k.method(mthd::kVertexData4fM + 4, bits(p[1]));
        k.method(mthd::kVertexData4fM + 8, bits(depth * 16777215.0f));
        k.method(mthd::kVertexData4fM + 12, bits(1.0f));
    }
    k.method(mthd::kBeginEnd, 0);
}

// The depth (and stencil) of the bound zeta buffer cleared to `zstencil` over columns x0..x1 of every row.
void clear_depth(Kelvin& k, u32 zstencil, u32 x0 = 0, u32 x1 = 63) {
    k.method(mthd::kZstencilClear, zstencil);
    k.method(mthd::kClearRectH, (x1 << 16) | x0);
    k.method(mthd::kClearRectV, 63u << 16);
    k.method(mthd::kClearSurface, 0x03);
}

u32 zeta_word(const std::vector<u8>& ram, u32 zeta, u32 x, u32 y) {
    u32 v;
    std::memcpy(&v, &ram[zeta + y * 256 + x * 4], 4);
    return v;
}

}  // namespace

// One target alternating between two depth buffers (the boot animation's two shadow depths, every frame): each
// buffer's image stays on the GPU while the other is bound (D35), and each draw tests against its own buffer's
// contents; both reach RAM when everything is written back.
TEST(Vulkan, ATargetAlternatingBetweenTwoDepthBuffersKeepsEachOnTheGpu) {
    for (u32 scale : {1u, 3u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        ScaledFixture f;
        std::string error;
        if (!f.init(scale, error)) GTEST_SKIP() << "no Vulkan device: " << error;
        Kelvin& k = f.k;
        setup_target(k);  // zeta at 0x20000
        k.method(mthd::kColorClear, 0xFF000080);
        k.method(mthd::kClearRectH, 63u << 16);
        k.method(mthd::kClearRectV, 63u << 16);
        k.method(mthd::kClearSurface, 0xF0);
        clear_depth(k, 0xBFFFFF00);  // A: 0.75
        k.method(mthd::kSurfaceZetaOffset, 0x40000);
        clear_depth(k, 0x3FFFFF00);  // B: 0.25
        k.method(mthd::kSurfaceZetaOffset, 0x20000);
        depth_triangle(k, 0xFF00FF00, 0.5f);  // A holds 0.75: drawn, and A becomes 0.5
        k.method(mthd::kSurfaceZetaOffset, 0x40000);
        depth_triangle(k, 0xFF0000FF, 0.5f);  // B holds 0.25: not drawn
        f.renderer->end_frame();
        EXPECT_EQ(pixel_at(f.ram, 8, 8) & 0x00FFFFFFu, 0x00FF00u) << "each draw tested against its own depth buffer";
        f.renderer->write_back_all();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        for (u32 at : {0u, 30u, 63u}) {
            EXPECT_NEAR(static_cast<double>(zeta_word(f.ram, 0x20000, at, at) >> 8), 0x7FFFFF, 2) << "A: the green triangle's depth";
            EXPECT_EQ(zeta_word(f.ram, 0x40000, at, at), 0x3FFFFF00u) << "B: its clear";
        }
        const auto& st = f.renderer->stats();
        EXPECT_EQ(st.depth_uploads, 0u) << "nothing loaded from RAM";
        EXPECT_GE(st.depth_unparks, 2u) << "each buffer's image bound again from the GPU";
        EXPECT_EQ(st.writebacks_by[VulkanRenderer::Stats::kWbRange], 0u) << "no depth sent to RAM at the switches";
    }
}

// The CPU writes the memory of a depth buffer whose image is parked: the next draw tests against what the CPU wrote.
TEST(Vulkan, ACpuWriteToAParkedDepthBufferIsWhatTheNextDrawTests) {
    for (u32 scale : {1u, 3u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        ScaledFixture f;
        std::string error;
        if (!f.init(scale, error)) GTEST_SKIP() << "no Vulkan device: " << error;
        Kelvin& k = f.k;
        setup_target(k);
        k.method(mthd::kColorClear, 0xFF000080);
        k.method(mthd::kClearRectH, 63u << 16);
        k.method(mthd::kClearRectV, 63u << 16);
        k.method(mthd::kClearSurface, 0xF0);
        clear_depth(k, 0xBFFFFF00);  // A: 0.75
        k.method(mthd::kSurfaceZetaOffset, 0x40000);
        clear_depth(k, 0x3FFFFF00);  // B bound, A parked
        f.renderer->flush_frame();
        for (u32 y = 0; y < 64; ++y)
            for (u32 x = 0; x < 64; ++x) {
                const u32 v = 0x3FFFFF00;  // the CPU writes 0.25 over A
                std::memcpy(&f.ram[0x20000 + y * 256 + x * 4], &v, 4);
            }
        f.renderer->cpu_wrote({0x20, 0x21, 0x22, 0x23});
        k.method(mthd::kSurfaceZetaOffset, 0x20000);
        depth_triangle(k, 0xFF00FF00, 0.5f);  // against the CPU's 0.25: not drawn
        f.renderer->end_frame();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        EXPECT_EQ(pixel_at(f.ram, 8, 8) & 0x00FFFFFFu, 0x000080u) << "the CPU's depth, not the parked image's";
        EXPECT_EQ(pixel_at(f.ram, 60, 3) & 0x00FFFFFFu, 0x000080u);
    }
}

// A draw that samples the depth buffer of the target it draws into, as a shadow map: it compares against the depth the
// draws before it left (as the texture unit reading RAM would), taken from a GPU copy of the depth image, nothing
// through RAM (a flight title wrote its shadow map back and waited for it 46 times a frame).
TEST(Vulkan, ADrawSamplingItsOwnDepthBufferReadsACopyOnTheGpu) {
    for (u32 scale : {1u, 2u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        TextureShaderFixture f;
        std::string error;
        if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
        f.renderer->set_resolution_scale(scale);
        Kelvin& k = f.k;
        setup_target(k);  // A8R8G8B8 at 0x10000, Z24S8 at 0x20000, pitch 256
        clear_depth(k, 0x3FFFFF00, 0, 31);   // left half 0.25
        clear_depth(k, 0xBFFFFF00, 32, 63);  // right half 0.75
        load_texcoord_program(k);
        const u32 b = mthd::kTextureBase;
        k.method(b + 0x0, 0x20000);
        k.method(b + 0x4, (1u << 16) | (0x2Eu << 8) | (2u << 4) | 1u);  // linear D24S8, the depth buffer itself
        k.method(b + 0x8, 0x00030303);
        k.method(b + 0xC, 0x40000000);
        k.method(b + 0x10, 256u << 16);
        k.method(b + 0x14, (1u << 24) | (1u << 16));
        k.method(b + 0x1C, (64u << 16) | 64u);
        k.method(mthd::kShaderStageProgram, 2u);  // stage 0: projective 3D = the shadow compare on a depth texture
        k.method(0x1E6C, 1u);                     // lit where the stored depth < r
        output_texture_stage(k, 0);
        const float r = 0.5f * 16777215.0f;
        texcoord(k, 0, 8.0f, 8.0f, r, 1.0f);  // a texel of the left half: 0.25 < 0.5, lit
        draw_full_triangle(k, f.ram);
        f.renderer->end_frame();
        EXPECT_EQ(pixel_at(f.ram, 20, 20), 0xFFFFFFFFu) << "compared against the cleared 0.25";
        texcoord(k, 0, 56.0f, 8.0f, r, 1.0f);  // the right half: 0.75 < 0.5 fails
        draw_full_triangle(k, f.ram);
        f.renderer->end_frame();
        EXPECT_EQ(pixel_at(f.ram, 20, 20), 0x00000000u) << "compared against the cleared 0.75";
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        const auto& st = f.renderer->stats();
        EXPECT_EQ(st.writebacks_by[VulkanRenderer::Stats::kWbTexture], 0u) << "the depth did not go through RAM";
        EXPECT_GE(st.depth_snapshots, 1u);
    }
}

// A depth buffer read as a colour texture gets the Z24S8 words' bytes: as A8R8G8B8 (blue first) or A8B8G8R8 (red
// first, a racing title reads its scene depth so), converted on the GPU where the device holds D24S8, never through RAM.
TEST(Vulkan, ADepthBufferReadAsColourGivesItsBytesInEitherOrder) {
    TextureShaderFixture f;
    std::string error;
    if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    Kelvin& k = f.k;
    setup_target(k);  // A8R8G8B8 at 0x10000, Z24S8 at 0x20000
    clear_depth(k, 0x12345678);  // depth 0x123456, stencil 0x78
    for (const auto& [format, expect] : {std::pair{0x12u, 0x12345678u}, std::pair{0x3Fu, 0x12785634u}}) {
        SCOPED_TRACE(std::format("texture format {:02x}", format));
        setup_target(k, 8, 0x100000);  // another target draws, sampling the depth buffer
        load_texcoord_program(k);
        const u32 b = mthd::kTextureBase;
        k.method(b + 0x0, 0x20000);
        k.method(b + 0x4, (1u << 16) | (format << 8) | (2u << 4) | 1u);
        k.method(b + 0x8, 0x00030303);
        k.method(b + 0xC, 0x40000000);
        k.method(b + 0x10, 256u << 16);
        k.method(b + 0x14, (1u << 24) | (1u << 16));
        k.method(b + 0x1C, (64u << 16) | 64u);
        k.method(mthd::kShaderStageProgram, 1u);
        output_texture_stage(k, 0);
        texcoord(k, 0, 8.0f, 8.0f, 0.0f, 1.0f);
        const u64 forced = f.renderer->stats().writebacks_by[VulkanRenderer::Stats::kWbTexture];
        draw_full_triangle(k, f.ram);
        f.renderer->end_frame();
        u32 v;
        std::memcpy(&v, &f.ram[0x100000 + 8 * 256 + 8 * 4], 4);
        EXPECT_EQ(v, expect);
        if (f.renderer->stats().zeta_conversions) {
            EXPECT_EQ(f.renderer->stats().writebacks_by[VulkanRenderer::Stats::kWbTexture], forced) << "converted on the GPU, not through RAM";
        }
        setup_target(k);
    }
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
}

// Only a clear of the whole depth buffer may skip loading it: a partial one keeps what RAM holds elsewhere.
TEST(Vulkan, APartialDepthClearStillLoadsTheRestFromRam) {
    for (u32 scale : {1u, 3u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        ScaledFixture f;
        std::string error;
        if (!f.init(scale, error)) GTEST_SKIP() << "no Vulkan device: " << error;
        Kelvin& k = f.k;
        for (u32 y = 0; y < 64; ++y)
            for (u32 x = 0; x < 64; ++x) {
                const u32 v = 0x3FFFFF00;  // RAM holds 0.25
                std::memcpy(&f.ram[0x20000 + y * 256 + x * 4], &v, 4);
            }
        setup_target(k);
        k.method(mthd::kColorClear, 0xFF000080);
        k.method(mthd::kClearRectH, 63u << 16);
        k.method(mthd::kClearRectV, 63u << 16);
        k.method(mthd::kClearSurface, 0xF0);
        clear_depth(k, 0xBFFFFF00, 0, 31);  // the left half to 0.75
        depth_triangle(k, 0xFF00FF00, 0.5f);
        f.renderer->end_frame();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        EXPECT_EQ(pixel_at(f.ram, 8, 8) & 0x00FFFFFFu, 0x00FF00u) << "the cleared half passes";
        EXPECT_EQ(pixel_at(f.ram, 48, 8) & 0x00FFFFFFu, 0x000080u) << "the other half kept RAM's 0.25";
        EXPECT_EQ(f.renderer->stats().depth_loads_skipped, 0u);
    }
}

// A smaller target of the same zeta pitch binding a depth buffer a bigger target holds draws into the top-left
// rectangle of that target's depth image (D37), as on the hardware where the buffer is memory: nothing is copied or
// loaded, the bigger target's next draw tests what the smaller one wrote, and RAM gets both at the write-back
// (Gauntlet Seven Sorrows' 720x480 frame buffers on a 768x512 target's depth, each frame).
TEST(Vulkan, ASmallerTargetDrawsIntoTheDepthImageOfABiggerOneOfTheSamePitch) {
    for (u32 scale : {1u, 3u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        ScaledFixture f;
        std::string error;
        if (!f.init(scale, error)) GTEST_SKIP() << "no Vulkan device: " << error;
        Kelvin& k = f.k;
        setup_target(k);  // 64x64, zeta at 0x20000, pitch 256
        k.method(mthd::kColorClear, 0xFF000080);
        k.method(mthd::kClearRectH, 63u << 16);
        k.method(mthd::kClearRectV, 63u << 16);
        k.method(mthd::kClearSurface, 0xF0);
        clear_depth(k, 0xBFFFFF00);  // 0.75
        // The smaller target: 48x40 at 0x60000, the same depth buffer and pitch.
        k.method(mthd::kSurfaceClipH, 48u << 16);
        k.method(mthd::kSurfaceClipV, 40u << 16);
        k.method(mthd::kSurfaceColorOffset, 0x60000);
        depth_triangle(k, 0xFF00FF00, 0.5f);  // passes 0.75 and writes 0.5 over the top-left 48x40
        k.method(mthd::kSurfaceClipH, 64u << 16);
        k.method(mthd::kSurfaceClipV, 64u << 16);
        k.method(mthd::kSurfaceColorOffset, 0x10000);
        depth_triangle(k, 0xFFFF0000, 0.625f);  // blue: fails over the smaller target's 0.5, passes the 0.75 elsewhere
        f.renderer->end_frame();
        f.renderer->write_back_all();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        EXPECT_EQ(pixel_at(f.ram, 8, 8) & 0x00FFFFFFu, 0x000080u) << "tested against the smaller target's depth";
        EXPECT_EQ(pixel_at(f.ram, 56, 8) & 0x00FFFFFFu, 0x0000FFu) << "right of it: the clear's depth";
        EXPECT_EQ(pixel_at(f.ram, 8, 50) & 0x00FFFFFFu, 0x0000FFu) << "below it: the clear's depth";
        u32 small;
        std::memcpy(&small, &f.ram[0x60000 + 8 * 256 + 8 * 4], 4);
        EXPECT_EQ(small & 0x00FFFFFFu, 0x00FF00u) << "the smaller target drew";
        EXPECT_NEAR(static_cast<double>(zeta_word(f.ram, 0x20000, 8, 8) >> 8), 0x7FFFFF, 2) << "the smaller target's depth reached RAM";
        EXPECT_NEAR(static_cast<double>(zeta_word(f.ram, 0x20000, 56, 50) >> 8), 0x9FFFFF, 2) << "the bigger target's own";
        const auto& st = f.renderer->stats();
        EXPECT_GE(st.depth_borrows, 1u);
        EXPECT_EQ(st.depth_uploads, 0u) << "nothing loaded from RAM";
        EXPECT_EQ(st.writebacks_by[VulkanRenderer::Stats::kWbRange], 0u) << "no depth sent to RAM at the switches";
    }
}

// Above 1x the video overlay is drawn in by the pass that finishes the picture on the GPU (a movie over a rendered
// picture cost 20 ms a frame at 6x on the CPU): the result is what grab_frame draws on the CPU over the same picture,
// pixel for pixel, for both byte orders, with and without the colour key and the colour table.
TEST(Vulkan, TheVideoOverlayDrawnOnTheGpuIsTheOneTheCpuDraws) {
    for (u32 scale : {2u, 3u}) {
        for (u32 variant = 0; variant < 4; ++variant) {
            const bool yuy2 = variant & 1u, keyed = variant & 2u;
            SCOPED_TRACE(std::format("scale {} yuy2 {} key {}", scale, yuy2, keyed));
            ScaledFixture f;
            std::string error;
            if (!f.init(scale, error)) GTEST_SKIP() << "no Vulkan device: " << error;
            Kelvin& k = f.k;
            setup_target(k);
            k.method(mthd::kColorClear, 0xFF102030u);
            k.method(mthd::kClearRectH, 63u << 16);
            k.method(mthd::kClearRectV, 63u << 16);
            k.method(mthd::kClearSurface, 0xF0);
            k.method(mthd::kColorClear, 0xFF00FF00u);  // the key colour over part of the overlay's rectangle
            k.method(mthd::kClearRectH, (30u << 16) | 10u);
            k.method(mthd::kClearRectV, (40u << 16) | 12u);
            k.method(mthd::kClearSurface, 0xF0);
            // A 16x12 YUV 4:2:2 source of varied bytes at 0x80000.
            u32 seed = 12345;
            for (u32 i = 0; i < 32 * 12; ++i) {
                seed = seed * 1103515245u + 12345u;
                f.ram[0x80000 + i] = static_cast<u8>(seed >> 16);
            }
            Overlay ov;
            ov.address = 0x80000;
            ov.pitch = 32;
            ov.in_w = 16;
            ov.in_h = 12;
            ov.s0 = 0x13;  // 1.1875 texels in
            ov.t0 = 0x08;
            ov.dsdx = (15u << 20) / 40 + 77;
            ov.dtdy = (11u << 20) / 30 + 5;
            ov.out_x = 5;
            ov.out_y = 7;
            ov.out_w = 40;
            ov.out_h = 30;
            ov.yuy2 = yuy2;
            ov.color_key = keyed;
            ov.key = 0x0000FF00u;
            std::array<u8, 768> lut{};
            for (u32 i = 0; i < 768; ++i) lut[i] = static_cast<u8>(i / 3 ^ (variant * 17));
            // The key is compared with the finished pixel: the key colour through the table.
            ov.key = variant ? (static_cast<u32>(lut[0]) << 16) | (static_cast<u32>(lut[255 * 3 + 1]) << 8) | lut[2] : 0x0000FF00u;
            // The picture without the overlay, then the CPU's overlay over it as grab_frame draws it.
            f.renderer->set_display_lut(variant ? std::optional(lut) : std::nullopt);
            f.renderer->set_display_overlay(std::nullopt);
            f.renderer->flush_frame();
            ASSERT_TRUE(f.renderer->write_back_at(0x10000));
            f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
            const u32 fw = 64 * scale, fh = 64 * scale;
            std::vector<u32> want(static_cast<std::size_t>(fw) * fh);
            {
                VulkanRenderer::ScaledPicture p;
                ASSERT_TRUE(f.renderer->scaled_picture(0x10000, 256, 4, 64, 64, p));
                ASSERT_TRUE(p.xrgb);
                f.renderer->picture_ready(p);
                for (u32 y = 0; y < fh; ++y) std::memcpy(&want[static_cast<std::size_t>(y) * fw], p.pixels + static_cast<std::size_t>(y) * p.pitch, fw * 4);
                f.renderer->picture_done();
            }
            Overlay o = ov;
            o.out_x *= scale;
            o.out_y *= scale;
            o.out_w *= scale;
            o.out_h *= scale;
            o.dsdx /= scale;
            o.dtdy /= scale;
            draw_overlay(o, f.ram.data(), f.ram.size(), reinterpret_cast<u8*>(want.data()), fw * 4, fw, fh);
            // The same vblank with the overlay on: drawn in by the GPU.
            f.renderer->set_display_overlay(ov);
            ASSERT_TRUE(f.renderer->write_back_at(0x10000));
            f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
            for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
            VulkanRenderer::ScaledPicture p;
            ASSERT_TRUE(f.renderer->scaled_picture(0x10000, 256, 4, 64, 64, p));
            EXPECT_TRUE(p.overlaid);
            f.renderer->picture_ready(p);
            u32 differ = 0, first_x = 0, first_y = 0, got_px = 0, want_px = 0;
            for (u32 y = 0; y < fh; ++y)
                for (u32 x = 0; x < fw; ++x) {
                    u32 v;
                    std::memcpy(&v, p.pixels + static_cast<std::size_t>(y) * p.pitch + x * 4, 4);
                    if (v != want[static_cast<std::size_t>(y) * fw + x] && differ++ == 0) {
                        first_x = x;
                        first_y = y;
                        got_px = v;
                        want_px = want[static_cast<std::size_t>(y) * fw + x];
                    }
                }
            f.renderer->picture_done();
            EXPECT_EQ(differ, 0u) << std::format("first at {},{}: {:08x}, the CPU's {:08x}", first_x, first_y, got_px, want_px);
            // The overlay drew (inside the key's area when keyed), and a keyed one left the rest of its rectangle alone.
            const auto at = [&](u32 x, u32 y) { return want[static_cast<std::size_t>(y * scale) * fw + x * scale]; };
            EXPECT_NE(at(15, 20), at(60, 60)) << "drawn";
            if (keyed) EXPECT_EQ(at(6, 8), at(60, 60)) << "not the key colour: kept";
            else EXPECT_NE(at(6, 8), at(60, 60)) << "drawn";
        }
    }
}

namespace {

// The image a frontend was handed, read back (its bytes as they lie: B, G, R and the unused byte).
std::vector<u32> read_handed_over(vk::Device& device, const VulkanRenderer::HandedOver& h) {
    vk::Buffer buf;
    std::string error;
    std::vector<u32> out;
    if (!device.create_buffer(static_cast<VkDeviceSize>(h.width) * h.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk::BufferMemory::HostRead, buf, error)) return out;
    VkCommandBuffer cb = device.begin_once();
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    device.fn().vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {h.width, h.height, 1};
    device.fn().vkCmdCopyImageToBuffer(cb, h.info->image, VK_IMAGE_LAYOUT_GENERAL, buf.buffer, 1, &r);
    device.finish_once(cb);
    out.resize(static_cast<std::size_t>(h.width) * h.height);
    std::memcpy(out.data(), buf.mapped, out.size() * 4);
    device.destroy_buffer(buf);
    return out;
}

void handoff_scene(Kelvin& k) {
    setup_target(k);
    k.method(mthd::kColorClear, 0xFF102030u);
    k.method(mthd::kClearRectH, 63u << 16);
    k.method(mthd::kClearRectV, 63u << 16);
    k.method(mthd::kClearSurface, 0xF0);
    k.method(mthd::kColorClear, 0xFFC08040u);
    k.method(mthd::kClearRectH, (40u << 16) | 9u);
    k.method(mthd::kClearRectV, (50u << 16) | 3u);
    k.method(mthd::kClearSurface, 0xF0);
}

}  // namespace

// The picture handed to a libretro frontend on the GPU (D38): with set_handoff the capture stays in device memory and
// hand_over copies it into the frontend's image, which holds exactly what the host read-back holds (the colour table
// applied); a CPU-made picture goes in through staging unchanged. The image's view reads alpha as one.
TEST(Vulkan, AHandedOverPictureIsTheOneTheHostWouldHaveRead) {
    std::array<u8, 768> lut{};
    for (u32 i = 0; i < 768; ++i) lut[i] = static_cast<u8>(255 - i / 3);
    std::vector<u32> want;
    {
        ScaledFixture f;
        std::string error;
        if (!f.init(3, error)) GTEST_SKIP() << "no Vulkan device: " << error;
        handoff_scene(f.k);
        f.renderer->set_display_lut(lut);
        f.renderer->flush_frame();
        ASSERT_TRUE(f.renderer->write_back_at(0x10000));
        f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
        VulkanRenderer::ScaledPicture p;
        ASSERT_TRUE(f.renderer->scaled_picture(0x10000, 256, 4, 64, 64, p));
        f.renderer->picture_ready(p);
        want.resize(192 * 192);
        for (u32 y = 0; y < 192; ++y) std::memcpy(&want[static_cast<std::size_t>(y) * 192], p.pixels + static_cast<std::size_t>(y) * p.pitch, 192 * 4);
        f.renderer->picture_done();
    }
    ScaledFixture f;
    std::string error;
    ASSERT_TRUE(f.init(3, error)) << error;
    f.renderer->set_handoff(true);
    handoff_scene(f.k);
    f.renderer->set_display_lut(lut);
    f.renderer->flush_frame();
    ASSERT_TRUE(f.renderer->write_back_at(0x10000));
    f.renderer->complete_write_backs(true);  // above 1x the copy in RAM lands in the background
    VulkanRenderer::ScaledPicture p;
    ASSERT_TRUE(f.renderer->scaled_picture(0x10000, 256, 4, 64, 64, p));
    EXPECT_EQ(p.pixels, nullptr) << "nothing read back to the host";
    ASSERT_NE(p.buffer, VK_NULL_HANDLE);
    VulkanRenderer::HandedOver h;
    ASSERT_TRUE(f.renderer->hand_over(1, &p, 192, 192, nullptr, 0, h));
    f.renderer->picture_done();
    EXPECT_EQ(h.info->format, VK_FORMAT_B8G8R8A8_UNORM);
    EXPECT_EQ(h.info->components.a, VK_COMPONENT_SWIZZLE_ONE);
    const std::vector<u32> got = read_handed_over(f.device, h);
    ASSERT_EQ(got.size(), want.size());
    u32 differ = 0;
    for (std::size_t i = 0; i < got.size(); ++i) differ += got[i] != want[i];
    EXPECT_EQ(differ, 0u) << std::format("first pixels {:08x} vs {:08x}", got[0], want[0]);
    EXPECT_NE(got[static_cast<std::size_t>(30 * 3) * 192 + 20 * 3], got[5]) << "the scene has two colours";
    // A picture made on the CPU, into another sync index's image, at another size.
    std::vector<u32> cpu(100 * 70);
    for (std::size_t i = 0; i < cpu.size(); ++i) cpu[i] = static_cast<u32>(i * 2654435761u) & 0x00FFFFFFu;
    VulkanRenderer::HandedOver c;
    ASSERT_TRUE(f.renderer->hand_over(0, nullptr, 100, 70, cpu.data(), 100 * 4, c));
    EXPECT_EQ(read_handed_over(f.device, c), cpu);
    for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
}

namespace {

// A resolver that records CPU writes per 4 KiB page, as the machine's write tracking does.
struct TrackingResolver : HandleResolver {
    u32 size{0};
    u32 generation{1};
    std::vector<u32> pages = std::vector<u32>((16u << 20) / 4096, 0);
    DmaObject resolve(u32) const override { return {0, size - 1, 0x3D, true}; }
    u32 write_generation() const override { return generation; }
    u32 newest_write(u32 phys, u32 len) const override {
        u32 n = 0;
        for (u32 p = phys / 4096; p <= (phys + len - 1) / 4096; ++p) n = std::max(n, pages[p]);
        return n;
    }
    u32 page_write(u32 page) const override { return pages[page]; }
    void wrote(u32 page) { pages[page] = ++generation; }
};

}  // namespace

// A texture whose pages the CPU writes is checked page by page: a written page holding the same bytes keeps the decoded
// texture (no decode), a page with new texels decodes it again and the draw shows them.
TEST(Vulkan, ATextureWrittenWithTheSameBytesIsKeptAndWithNewOnesDecodedAgain) {
    vk::Device device;
    std::string error;
    if (!device.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
    std::vector<u8> ram(16u << 20, 0);
    TrackingResolver handles;
    handles.size = static_cast<u32>(ram.size());
    VulkanRenderer renderer(device, handles, ram.data(), static_cast<u32>(ram.size()));
    ASSERT_TRUE(renderer.init(error)) << error;
    Kelvin k;
    k.set_sink(&renderer);
    constexpr u32 kTex = 0x40000;  // 64x64 A8R8G8B8, swizzled: 16 KiB over four pages; texel (32,32) lies in the fourth
    auto fill = [&](u32 argb) {
        for (u32 i = 0; i < 64 * 64; ++i) std::memcpy(&ram[kTex + i * 4], &argb, 4);
    };
    fill(0xFF00FF00u);
    auto draw = [&] {
        setup_target(k);
        load_texcoord_program(k);
        const u32 b = mthd::kTextureBase;
        k.method(b + 0x0, kTex);
        k.method(b + 0x4, (6u << 24) | (6u << 20) | (1u << 16) | (0x06u << 8) | (2u << 4) | 1u);
        k.method(b + 0x8, 0x00030303);
        k.method(b + 0xC, 0x40000000);
        k.method(b + 0x14, (1u << 24) | (1u << 16));
        k.method(mthd::kShaderStageProgram, 1u);
        output_texture_stage(k, 0);
        texcoord(k, 0, 0.5f, 0.5f, 0.0f, 1.0f);
        draw_full_triangle(k, ram);
        renderer.end_frame();
    };
    draw();
    EXPECT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0x00FF00u);
    const u64 decoded = renderer.stats().textures_decoded;
    handles.wrote(kTex / 4096 + 1);  // the same bytes written again
    draw();
    EXPECT_EQ(renderer.stats().textures_decoded, decoded) << "kept";
    EXPECT_GE(renderer.stats().textures_unchanged, 1u);
    // New texels in the fourth page (where the sampled texel lies), only that page reported written.
    for (u32 i = 3 * 1024; i < 4 * 1024; ++i) {
        const u32 red = 0xFFFF0000u;
        std::memcpy(&ram[kTex + i * 4], &red, 4);
    }
    handles.wrote(kTex / 4096 + 3);
    draw();
    EXPECT_EQ(renderer.stats().textures_decoded, decoded + 1) << "decoded again";
    EXPECT_EQ(pixel_at(ram, 8, 8) & 0x00FFFFFFu, 0xFF0000u) << "the new texels drawn";
    for (const auto& e : renderer.stats().errors) ADD_FAILURE() << "renderer: " << e;
}

// A cube map of six R5G6B5 render targets (PGR2's reflections) is put together on the GPU like a 32-bit one: before,
// each face went through RAM every frame.
TEST(Vulkan, ACubeMapOfSix16BitTargetsIsPutTogetherOnTheGpu) {
    for (u32 scale : {1u, 2u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        TextureShaderFixture f;
        std::string error;
        if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
        f.renderer->set_resolution_scale(scale);
        Kelvin& k = f.k;
        constexpr u32 kCube = 0x200000, kFace = 16 * 16 * 2;
        const u32 clears[6] = {0xF800u, 0x07E0u, 0x001Fu, 0xFFE0u, 0x07FFu, 0xF81Fu};  // 5-6-5: red green blue yellow cyan magenta
        const u32 colours[6] = {0xFFFF0000u, 0xFF00FF00u, 0xFF0000FFu, 0xFFFFFF00u, 0xFF00FFFFu, 0xFFFF00FFu};
        for (u32 face = 0; face < 6; ++face) {
            k.method(mthd::kSurfaceFormat, (4u << 24) | (4u << 16) | 0x200u | 0x3u);  // swizzled 16x16 R5G6B5, no depth
            k.method(mthd::kSurfaceClipH, 16u << 16);
            k.method(mthd::kSurfaceClipV, 16u << 16);
            k.method(mthd::kSurfaceColorOffset, kCube + face * kFace);
            k.method(mthd::kColorClear, clears[face]);
            k.method(mthd::kClearRectH, 15u << 16);
            k.method(mthd::kClearRectV, 15u << 16);
            k.method(mthd::kClearSurface, 0xF0);
        }
        setup_target(k);
        load_texcoord_program(k);
        const u32 b = mthd::kTextureBase;
        k.method(b + 0x0, kCube);
        k.method(b + 0x4, (4u << 24) | (4u << 20) | (1u << 16) | (0x05u << 8) | (2u << 4) | (1u << 2) | 1u);  // a 16x16 R5G6B5 cube map
        k.method(b + 0x8, 0x00030303);
        k.method(b + 0xC, 0x40000000);
        k.method(b + 0x14, (1u << 24) | (1u << 16));
        k.method(mthd::kShaderStageProgram, 3u);
        output_texture_stage(k, 0);
        const float dirs[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (u32 face = 0; face < 6; ++face) {
            texcoord(k, 0, dirs[face][0], dirs[face][1], dirs[face][2], 1.0f);
            draw_full_triangle(k, f.ram);
            f.renderer->end_frame();
            EXPECT_EQ(pixel_at(f.ram, 8, 8), colours[face]) << "face " << face;
        }
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        EXPECT_GE(f.renderer->stats().cube_assemblies, 1u);
        EXPECT_EQ(f.renderer->stats().writebacks_by[VulkanRenderer::Stats::kWbTexture], 0u) << "no face went through RAM";
    }
}

// A B8 render target sampled as a Y8 texture (PGR2) is read in place through a (Y, Y, Y, 1) view.
TEST(Vulkan, AB8TargetSampledAsY8IsReadInPlace) {
    for (u32 scale : {1u, 2u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        TextureShaderFixture f;
        std::string error;
        if (!f.init(error)) GTEST_SKIP() << "no Vulkan device: " << error;
        f.renderer->set_resolution_scale(scale);
        Kelvin& k = f.k;
        constexpr u32 kB8 = 0x200000;
        k.method(mthd::kSurfaceFormat, 0x00000100u | 0x9u);  // pitch-linear B8, no depth
        k.method(mthd::kSurfaceClipH, 64u << 16);
        k.method(mthd::kSurfaceClipV, 64u << 16);
        k.method(mthd::kSurfacePitch, (64u << 16) | 64u);
        k.method(mthd::kSurfaceColorOffset, kB8);
        k.method(mthd::kColorClear, 0x00000080u);  // B8 takes the blue byte
        k.method(mthd::kClearRectH, 63u << 16);
        k.method(mthd::kClearRectV, 63u << 16);
        k.method(mthd::kClearSurface, 0xF0);
        setup_target(k);
        load_texcoord_program(k);
        const u32 b = mthd::kTextureBase;
        k.method(b + 0x0, kB8);
        k.method(b + 0x4, (1u << 16) | (0x13u << 8) | (2u << 4) | 1u);  // linear Y8, one level
        k.method(b + 0x8, 0x00030303);
        k.method(b + 0xC, 0x40000000);
        k.method(b + 0x10, 64u << 16);  // pitch
        k.method(b + 0x14, (1u << 24) | (1u << 16));
        k.method(b + 0x1C, (64u << 16) | 64u);  // 64x64
        k.method(mthd::kShaderStageProgram, 1u);
        output_texture_stage(k, 0);
        texcoord(k, 0, 8.0f, 8.0f, 0.0f, 1.0f);
        draw_full_triangle(k, f.ram);
        f.renderer->end_frame();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        EXPECT_EQ(pixel_at(f.ram, 8, 8), 0xFF808080u) << "(Y, Y, Y, 1)";
        EXPECT_EQ(f.renderer->stats().writebacks_by[VulkanRenderer::Stats::kWbTexture], 0u) << "not through RAM";
    }
}

// A B8 target stores the colour's blue byte (the hardware's single channel): a draw writing blue alone (colour mask B)
// lands it in RAM, and red is not what it keeps. Its R8 image swaps red and blue on the way in and out.
TEST(Vulkan, AB8TargetKeepsTheBlueByteOfADrawThatWritesBlueAlone) {
    for (u32 scale : {1u, 2u}) {
        SCOPED_TRACE(std::format("scale {}", scale));
        ScaledFixture f;
        std::string error;
        if (!f.init(scale, error)) GTEST_SKIP() << "no Vulkan device: " << error;
        Kelvin& k = f.k;
        constexpr u32 kB8 = 0x200000;
        setup_target(k);
        k.method(mthd::kSurfaceFormat, 0x00000100u | 0x9u);  // pitch-linear B8, no depth
        k.method(mthd::kSurfacePitch, (64u << 16) | 64u);
        k.method(mthd::kSurfaceColorOffset, kB8);
        k.method(mthd::kColorClear, 0x00000011u);
        k.method(mthd::kClearRectH, 63u << 16);
        k.method(mthd::kClearRectV, 63u << 16);
        k.method(mthd::kClearSurface, 0xF0);
        k.method(mthd::kColorMask, 0x00000001u);  // blue alone
        k.method(mthd::kBeginEnd, 5);
        k.method(mthd::kDiffuse4ub, 0xFF902010u);  // A B G R: blue 0x90, green 0x20, red 0x10
        for (const auto& p : {std::array<float, 2>{0.0f, 0.0f}, std::array<float, 2>{64.0f, 0.0f}, std::array<float, 2>{0.0f, 64.0f}}) {
            k.method(mthd::kVertexData4fM + 0, bits(p[0]));
            k.method(mthd::kVertexData4fM + 4, bits(p[1]));
            k.method(mthd::kVertexData4fM + 8, bits(0.0f));
            k.method(mthd::kVertexData4fM + 12, bits(1.0f));
        }
        k.method(mthd::kBeginEnd, 0);
        f.renderer->end_frame();
        f.renderer->write_back_all();
        for (const auto& e : f.renderer->stats().errors) ADD_FAILURE() << "renderer: " << e;
        EXPECT_EQ(f.ram[kB8 + 8 * 64 + 8], 0x90u) << "the blue byte drawn";
        EXPECT_EQ(f.ram[kB8 + 60 * 64 + 60], 0x11u) << "outside the triangle: the clear's blue byte";
    }
}
