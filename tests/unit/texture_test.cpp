#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "nv2a/pgraph/texture.h"

using namespace mv;
using namespace mv::pgraph;

namespace {
TextureStage stage_of(u32 format_word, u32 control1 = 0, u32 rect = 0) {
    TextureStage t;
    t.format = format_word;
    t.control1 = control1;
    t.image_rect = rect;
    t.control0 = 0x40000000;
    return t;
}
u32 fmt(u32 color, u32 log_w, u32 log_h, u32 levels = 1, bool cube = false, u32 dim = 2, u32 log_d = 0) {
    return 2u | (cube ? 4u : 0u) | (dim << 4) | (color << 8) | (levels << 16) | (log_w << 20) | (log_h << 24) | (log_d << 28);
}
}  // namespace

TEST(Texture, SwizzledOffsetsInterleaveXFirst) {
    EXPECT_EQ(swizzled_offset(0, 0, 0, 2, 2, 0), 0u);
    EXPECT_EQ(swizzled_offset(1, 0, 0, 2, 2, 0), 1u);
    EXPECT_EQ(swizzled_offset(0, 1, 0, 2, 2, 0), 2u);
    EXPECT_EQ(swizzled_offset(1, 1, 0, 2, 2, 0), 3u);
    EXPECT_EQ(swizzled_offset(2, 0, 0, 2, 2, 0), 4u);
    EXPECT_EQ(swizzled_offset(3, 3, 0, 2, 2, 0), 15u);
    // A wide image: once y's bits run out, the remaining x bits follow in order.
    EXPECT_EQ(swizzled_offset(2, 0, 0, 2, 1, 0), 4u);
    EXPECT_EQ(swizzled_offset(3, 1, 0, 2, 1, 0), 7u);
    EXPECT_EQ(swizzled_offset(1, 1, 1, 1, 1, 1), 7u);
}

TEST(Texture, DecodesASwizzledArgbImageWithMipLevels) {
    // 4x2 A8R8G8B8 with two levels: level 0 is 8 texels, level 1 is 2x1.
    std::vector<u8> ram(4096, 0);
    const TextureStage t = stage_of(fmt(0x06, 2, 1, 2));
    EXPECT_EQ(texture_size_in_memory(t), 8u * 4 + 2u * 4);
    for (u32 y = 0; y < 2; ++y)
        for (u32 x = 0; x < 4; ++x) {
            const u32 off = 0x100 + swizzled_offset(x, y, 0, 2, 1, 0) * 4;
            ram[off + 0] = static_cast<u8>(10 * x);       // blue
            ram[off + 1] = static_cast<u8>(100 + y);      // green
            ram[off + 2] = static_cast<u8>(200);          // red
            ram[off + 3] = 0x80;                          // alpha
        }
    const u8 level1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::memcpy(&ram[0x100 + 32], level1, 8);
    DecodedTexture d;
    ASSERT_TRUE(decode_texture(t, 0x100, MemoryView{ram.data(), 4096}, nullptr, d)) << d.error;
    EXPECT_EQ(d.pixel, HostPixel::Rgba8);
    EXPECT_EQ(d.width, 4u);
    EXPECT_EQ(d.height, 2u);
    EXPECT_EQ(d.levels, 2u);
    ASSERT_EQ(d.images.size(), 2u);
    const auto& img = d.image(0, 0);
    // texel (3, 1): R 200, G 101, B 30, A 0x80 at row-major position 7
    EXPECT_EQ(img[7 * 4 + 0], 200);
    EXPECT_EQ(img[7 * 4 + 1], 101);
    EXPECT_EQ(img[7 * 4 + 2], 30);
    EXPECT_EQ(img[7 * 4 + 3], 0x80);
    EXPECT_EQ(d.image(0, 1)[0 * 4 + 0], 3) << "level 1 texel 0 red = byte 2";
    EXPECT_EQ(d.image(0, 1)[1 * 4 + 3], 8);
}

// The table-driven swizzle reads every texel where swizzled_offset says, in non-square images, volumes and the three
// loops (A8R8G8B8, X8R8G8B8 and the generic one, here R5G6B5).
TEST(Texture, SwizzledDecodesReadEachTexelAtItsInterleavedOffset) {
    struct Shape { u32 lw, lh, ld; };
    for (const Shape s : {Shape{5, 3, 0}, Shape{2, 6, 0}, Shape{3, 2, 2}})
        for (const u32 format : {0x06u, 0x07u, 0x05u}) {
            SCOPED_TRACE(std::to_string(format) + " " + std::to_string(s.lw) + "x" + std::to_string(s.lh) + "x" + std::to_string(s.ld));
            const u32 bpp = format == 0x05 ? 2 : 4, w = 1u << s.lw, h = 1u << s.lh, d = 1u << s.ld;
            std::vector<u8> ram(w * h * d * bpp + 64);
            u32 seed = 12345;
            for (auto& b : ram) b = static_cast<u8>((seed = seed * 1103515245u + 12345u) >> 16);
            const TextureStage t = stage_of(fmt(format, s.lw, s.lh, 1, false, s.ld ? 3 : 2, s.ld));
            DecodedTexture dec;
            ASSERT_TRUE(decode_texture(t, 0, MemoryView{ram.data(), static_cast<u32>(ram.size())}, nullptr, dec)) << dec.error;
            const auto& img = dec.image(0, 0);
            for (u32 z = 0; z < d; ++z)
                for (u32 y = 0; y < h; ++y)
                    for (u32 x = 0; x < w; ++x) {
                        const u8* p = &ram[swizzled_offset(x, y, z, s.lw, s.lh, s.ld) * bpp];
                        const u8* q = &img[((static_cast<std::size_t>(z) * h + y) * w + x) * 4];
                        u32 r = p[2], g = p[1], b = p[0], a = format == 0x07 ? 255u : p[3];
                        if (format == 0x05) {
                            const u32 v = p[0] | (p[1] << 8);
                            r = ((v >> 11) << 3) | (v >> 13);
                            g = (((v >> 5) & 63) << 2) | (((v >> 5) & 63) >> 4);
                            b = ((v & 31) << 3) | ((v & 31) >> 2);
                            a = 255;
                        }
                        ASSERT_EQ((u32{q[0]} << 24) | (u32{q[1]} << 16) | (u32{q[2]} << 8) | q[3], (r << 24) | (g << 16) | (b << 8) | a) << x << "," << y << "," << z;
                    }
        }
}

TEST(Texture, DecodesLinearFormatsWithPitchAndPalettes) {
    std::vector<u8> ram(4096, 0);
    // 3x2 linear R5G6B5 with a pitch of 16 bytes: pure red, pure green, pure blue on row 1.
    const u16 row1[3] = {0xF800, 0x07E0, 0x001F};
    std::memcpy(&ram[0x200 + 16], row1, 6);
    const TextureStage lin = stage_of(fmt(0x11, 0, 0), 16u << 16, (3u << 16) | 2u);
    EXPECT_EQ(texture_size_in_memory(lin), 32u);
    DecodedTexture d;
    ASSERT_TRUE(decode_texture(lin, 0x200, MemoryView{ram.data(), 4096}, nullptr, d)) << d.error;
    EXPECT_EQ(d.width, 3u);
    EXPECT_EQ(d.height, 2u);
    const auto& img = d.image(0, 0);
    EXPECT_EQ(img[(3 + 0) * 4 + 0], 255);
    EXPECT_EQ(img[(3 + 1) * 4 + 1], 255);
    EXPECT_EQ(img[(3 + 2) * 4 + 2], 255);
    EXPECT_EQ(img[(3 + 2) * 4 + 0], 0);

    // 2x2 swizzled I8 through a palette of A8R8G8B8 entries.
    ram[0x300 + 0] = 1;
    ram[0x300 + 3] = 2;
    u8 palette[256 * 4]{};
    palette[1 * 4 + 2] = 0x11;  // entry 1: red 0x11
    palette[1 * 4 + 3] = 0xFF;
    palette[2 * 4 + 0] = 0x22;  // entry 2: blue 0x22
    const TextureStage pal = stage_of(fmt(0x0B, 1, 1));
    ASSERT_TRUE(decode_texture(pal, 0x300, MemoryView{ram.data(), 4096}, palette, d)) << d.error;
    EXPECT_EQ(d.image(0, 0)[0], 0x11);
    EXPECT_EQ(d.image(0, 0)[3], 0xFF);
    EXPECT_EQ(d.image(0, 0)[3 * 4 + 2], 0x22) << "texel (1,1) is swizzled index 3";

    // A 2x1 linear 16-bit depth image keeps its raw values.
    const u16 depths[2] = {0x1234, 0xFFFF};
    std::memcpy(&ram[0x400], depths, 4);
    const TextureStage dep = stage_of(fmt(0x30, 0, 0), 4u << 16, (2u << 16) | 1u);
    ASSERT_TRUE(decode_texture(dep, 0x400, MemoryView{ram.data(), 4096}, nullptr, d)) << d.error;
    EXPECT_EQ(d.pixel, HostPixel::R16);
    u16 v;
    std::memcpy(&v, &d.image(0, 0)[2], 2);
    EXPECT_EQ(v, 0xFFFF);
}

TEST(Texture, DecodesDxt1BlocksAndCubeFaces) {
    // One DXT1 block: colour 0 = red (0xF800), colour 1 = blue (0x001F); indices 0,1,2,3 on the first row.
    u8 block[8] = {0x00, 0xF8, 0x1F, 0x00, 0xE4, 0x00, 0x00, 0x00};  // 0xE4 = 11 10 01 00
    u8 out[4 * 4 * 4];
    decode_dxt1(block, 4, 4, out);
    EXPECT_EQ(out[0], 255);      // texel 0 red
    EXPECT_EQ(out[4 + 2], 255);  // texel 1 blue
    EXPECT_EQ(out[8 + 0], 170);  // texel 2 = 2/3 red
    EXPECT_EQ(out[12 + 2], 170); // texel 3 = 2/3 blue
    EXPECT_EQ(out[3], 255);

    // A 2x2 cube map of X8R8G8B8: six faces 128 bytes apart (the face alignment), each face's texel 0 = face index in red.
    std::vector<u8> ram(4096, 0);
    const TextureStage cube = stage_of(fmt(0x07, 1, 1, 1, true));
    EXPECT_EQ(texture_size_in_memory(cube), 128u * 6);
    for (u32 face = 0; face < 6; ++face) ram[0x800 + face * 128 + 2] = static_cast<u8>(face + 1);
    DecodedTexture d;
    ASSERT_TRUE(decode_texture(cube, 0x800, MemoryView{ram.data(), 4096}, nullptr, d)) << d.error;
    EXPECT_EQ(d.faces, 6u);
    ASSERT_EQ(d.images.size(), 6u);
    EXPECT_EQ(d.image(5, 0)[0], 6);
    EXPECT_EQ(d.image(5, 0)[3], 255);

    // Out of RAM: refused with a message.
    EXPECT_FALSE(decode_texture(cube, 4096 - 64, MemoryView{ram.data(), 4096}, nullptr, d));
    EXPECT_NE(d.error.find("outside RAM"), std::string::npos) << d.error;
}

TEST(Texture, DecodesLinearYuvPairsWithTheirOwnChroma) {
    // 4x2 linear YUV, pitch 16: each pair of pixels shares one U and V. Pair 0 is BT.601 red (U 90, V 240) with lumas
    // 82 and 235; pair 1 is grey (U = V = 128) with lumas 16 (black) and 235 (white).
    std::vector<u8> ram(4096, 0);
    const u8 yuy2[8] = {82, 90, 235, 240, 16, 128, 235, 128};    // Y0 U Y1 V
    const u8 uyvy[8] = {90, 82, 240, 235, 128, 16, 128, 235};    // U Y0 V Y1
    std::memcpy(&ram[0x100 + 16], yuy2, 8);
    std::memcpy(&ram[0x200 + 16], uyvy, 8);
    for (u32 format : {0x24u, 0x25u}) {
        const TextureStage t = stage_of(fmt(format, 0, 0), 16u << 16, (4u << 16) | 2u);
        EXPECT_EQ(format_info(format).bytes_per_pixel, 2u);
        EXPECT_TRUE(format_info(format).linear);
        EXPECT_EQ(texture_size_in_memory(t), 32u);
        DecodedTexture d;
        ASSERT_TRUE(decode_texture(t, format == 0x24 ? 0x100 : 0x200, MemoryView{ram.data(), 4096}, nullptr, d)) << d.error;
        EXPECT_EQ(d.pixel, HostPixel::Rgba8);
        const auto& img = d.image(0, 0);
        auto px = [&](u32 x, u32 y) {
            const u8* p = &img[(y * 4 + x) * 4];
            return (static_cast<u32>(p[0]) << 24) | (static_cast<u32>(p[1]) << 16) | (static_cast<u32>(p[2]) << 8) | p[3];
        };
        EXPECT_EQ(px(0, 1), 0xFF0100FFu) << "format " << format << ": luma 82 with the red chroma";
        EXPECT_EQ(px(1, 1), 0xFFB3B2FFu) << "format " << format << ": luma 235, the same chroma";
        EXPECT_EQ(px(2, 1), 0x000000FFu) << "format " << format << ": video black";
        EXPECT_EQ(px(3, 1), 0xFFFFFFFFu) << "format " << format << ": video white";
        EXPECT_EQ(px(0, 0), 0x008700FFu) << "format " << format << ": all-zero bytes are a dark green, not a placeholder";
    }
}

TEST(Texture, X8Y24DepthIsTheTopThreeBytes) {
    // A Z24S8 word holds the depth in bits 31-8 above the stencil byte, as the depth surface writes it: a far depth
    // with stencil 0x5A is 0xFFFFFF5A and reads 1.0; the middle depth with no stencil reads half.
    std::vector<u8> ram(4096, 0);
    const u32 words[2] = {0xFFFFFF5Au, 0x80000000u};
    std::memcpy(&ram[0x100], words, 8);
    for (u32 format : {0x2Eu, 0x2Fu}) {
        const TextureStage t = stage_of(fmt(format, 0, 0), 8u << 16, (2u << 16) | 1u);
        EXPECT_TRUE(format_info(format).depth) << "format " << format;
        DecodedTexture d;
        ASSERT_TRUE(decode_texture(t, 0x100, MemoryView{ram.data(), 4096}, nullptr, d)) << d.error;
        EXPECT_EQ(d.pixel, HostPixel::R32f);
        float f[2];
        std::memcpy(f, d.image(0, 0).data(), 8);
        EXPECT_FLOAT_EQ(f[0], 1.0f) << "format " << format;
        EXPECT_FLOAT_EQ(f[1], static_cast<float>(0x800000) / 16777215.0f) << "format " << format;
    }
}

TEST(Texture, A8IsWhiteWithAlpha) {
    std::vector<u8> ram(4096, 0);
    ram[0x100] = 0x40;
    ram[0x101] = 0xC0;
    for (const TextureStage& t : {stage_of(fmt(0x19, 1, 0)), stage_of(fmt(0x1F, 0, 0), 4u << 16, (2u << 16) | 1u)}) {
        DecodedTexture d;
        ASSERT_TRUE(decode_texture(t, 0x100, MemoryView{ram.data(), 4096}, nullptr, d)) << d.error;
        const auto& img = d.image(0, 0);
        EXPECT_EQ((std::vector<u8>(img.begin(), img.end())), (std::vector<u8>{255, 255, 255, 0x40, 255, 255, 255, 0xC0}));
    }
}

TEST(Texture, DecodesCompressedVolumesWithTheSlicesBlocksTogether) {
    // An 8x4x2 DXT1 volume: two block positions, two slices. In memory each position's slices come together:
    // (x0 z0) red, (x0 z1) green, (x1 z0) blue, (x1 z1) white, every texel index 0 (colour 0).
    const u16 colours[4] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    u8 blocks[4 * 8] = {};
    for (u32 i = 0; i < 4; ++i) std::memcpy(&blocks[i * 8], &colours[i], 2);
    std::vector<u8> out(8 * 4 * 2 * 4);
    decode_dxt_volume(0x0C, blocks, 8, 4, 2, out.data());
    const auto texel = [&](u32 x, u32 y, u32 z) { return &out[((z * 4 + y) * 8 + x) * 4]; };
    EXPECT_EQ(texel(0, 0, 0)[0], 255);  // slice 0, left block: red
    EXPECT_EQ(texel(3, 3, 1)[1], 255);  // slice 1, left block: green
    EXPECT_EQ(texel(1, 0, 1)[0], 0);
    EXPECT_EQ(texel(4, 0, 0)[2], 255);  // slice 0, right block: blue
    EXPECT_EQ(texel(4, 0, 0)[0], 0);
    EXPECT_EQ(texel(7, 2, 1)[0], 255);  // slice 1, right block: white
    EXPECT_EQ(texel(7, 2, 1)[1], 255);

    // Through the stage decoder: the same bytes as a 3D DXT1 texture give the same slices (a flat reading of
    // the blocks would put green beside red in slice 0).
    std::vector<u8> ram(4096, 0);
    std::memcpy(&ram[0x100], blocks, sizeof blocks);
    const TextureStage vol = stage_of(fmt(0x0C, 3, 2, 1, false, 3, 1));
    DecodedTexture d;
    ASSERT_TRUE(decode_texture(vol, 0x100, MemoryView{ram.data(), 4096}, nullptr, d)) << d.error;
    ASSERT_EQ(d.depth, 2u);
    EXPECT_EQ(d.image(0, 0)[4 * 4 + 2], 255);  // slice 0, x 4: blue
    EXPECT_EQ(d.image(0, 0)[(8 * 4 + 4) * 4 + 1], 255);  // slice 1, x 4: white
}
