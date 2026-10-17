#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "nv2a/pgraph/kelvin.h"

using namespace mv;
using namespace mv::pgraph;

namespace {

u32 bits(float f) {
    u32 v;
    std::memcpy(&v, &f, 4);
    return v;
}

struct RecordingSink : RenderSink {
    std::vector<DrawBatch> draws;
    std::vector<u32> clears;
    std::vector<u32> semaphores;
    void clear(const KelvinState&, u32 clear_bits) override { clears.push_back(clear_bits); }
    void draw(const KelvinState&, const DrawBatch& batch) override { draws.push_back(batch); }
    void semaphore_release(const KelvinState&, u32 value) override { semaphores.push_back(value); }
};

struct TableResolver : HandleResolver {
    DmaObject obj;
    DmaObject resolve(u32 handle) const override { return handle == 3 ? obj : DmaObject{}; }
};

}  // namespace

TEST(Kelvin, StoresRegistersAndDecodesSurfaceAndTextureFields) {
    Kelvin k;
    k.method(mthd::kSurfaceFormat, 0x08080228);
    k.method(mthd::kSurfaceClipH, 0x01000000);
    k.method(mthd::kSurfaceClipV, 0x01000000);
    k.method(mthd::kSurfacePitch, 0x04000400);
    const SurfaceState s = k.state().surface();
    EXPECT_EQ(s.color_format(), 8u);
    EXPECT_EQ(s.zeta_format(), 2u);
    EXPECT_TRUE(s.swizzled());
    EXPECT_EQ(s.log_width(), 8u);
    EXPECT_EQ(s.clip_width(), 256u);
    EXPECT_EQ(s.color_pitch(), 1024u);
    EXPECT_EQ(s.color_bytes(), 4u);
    EXPECT_EQ(s.zeta_bytes(), 4u);

    k.method(mthd::kTextureBase + 3 * mthd::kTextureStride + 4, 0x0001302a);
    k.method(mthd::kTextureBase + 3 * mthd::kTextureStride + 0x1C, 0x02000200);
    k.method(mthd::kTextureBase + 3 * mthd::kTextureStride + 0x10, 0x04000000);
    k.method(mthd::kTextureBase + 3 * mthd::kTextureStride + 0xC, 0x4003ffc0);
    const TextureStage t = k.state().texture(3);
    EXPECT_EQ(t.color_format(), 0x30u);
    EXPECT_EQ(t.dimensionality(), 2u);
    EXPECT_EQ(t.mip_levels(), 1u);
    EXPECT_EQ(t.rect_width(), 512u);
    EXPECT_EQ(t.pitch(), 1024u);
    EXPECT_TRUE(t.enabled());
    EXPECT_EQ(k.state().texture(0).color_format(), 0u);
}

TEST(Kelvin, ProgramAndConstantLoadsAutoIncrement) {
    Kelvin k;
    k.method(mthd::kTransformProgramLoad, 2);
    for (u32 i = 0; i < 12; ++i) k.method(mthd::kTransformProgram + 4 * (i % 32), 0x100 + i);  // 3 instructions
    EXPECT_EQ(k.state().program[2][0], 0x100u);
    EXPECT_EQ(k.state().program[2][3], 0x103u);
    EXPECT_EQ(k.state().program[4][3], 0x10Bu);
    EXPECT_EQ(k.state().program_load, 5u);
    k.method(mthd::kTransformConstantLoad, 0x60);
    for (u32 i = 0; i < 8; ++i) k.method(mthd::kTransformConstant + 4 * i, bits(static_cast<float>(i)));
    EXPECT_EQ(k.state().constants[0x60][1], bits(1.0f));
    EXPECT_EQ(k.state().constants[0x61][3], bits(7.0f));
    EXPECT_EQ(k.state().constant_load, 0x62u);
}

TEST(Kelvin, IndexedDrawFetchesArraysFromMemory) {
    Kelvin k;
    RecordingSink sink;
    k.set_sink(&sink);
    k.method(mthd::kDmaVertexA, 3);
    // v0: 3 floats stride 16 at 0x1000; v3: 4 D3D bytes stride 16 at offset 12 of the same vertex.
    k.method(mthd::kVertexDataArrayFormat + 0, (16u << 8) | (3u << 4) | 2u);
    k.method(mthd::kVertexDataArrayOffset + 0, 0x1000);
    k.method(mthd::kVertexDataArrayFormat + 12, (16u << 8) | (4u << 4) | 0u);
    k.method(mthd::kVertexDataArrayOffset + 12, 0x100C);
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kArrayElement16, (2u << 16) | 1u);
    k.method(mthd::kArrayElement32, 3);
    k.method(mthd::kBeginEnd, 0);
    ASSERT_EQ(sink.draws.size(), 1u);
    EXPECT_EQ(sink.draws[0].primitive, Primitive::Triangles);
    EXPECT_EQ(sink.draws[0].indices, (std::vector<u32>{1, 2, 3}));

    std::vector<u8> ram(0x2000, 0);
    for (u32 v = 0; v < 4; ++v) {
        const float xyz[3] = {static_cast<float>(v), 10.0f + static_cast<float>(v), -1.0f};
        std::memcpy(&ram[0x1000 + v * 16], xyz, 12);
        const u8 bgra[4] = {0x10, 0x20, 0x30, 0xFF};  // blue 0x10, green 0x20, red 0x30
        std::memcpy(&ram[0x1000 + v * 16 + 12], bgra, 4);
    }
    TableResolver handles;
    handles.obj = {0, 0x1FFF, 0x3D, true};
    FetchedVertices out;
    ASSERT_TRUE(fetch_vertices(k.state(), handles, MemoryView{ram.data(), static_cast<u32>(ram.size())}, sink.draws[0], out)) << out.error;
    EXPECT_EQ(out.count, 3u);  // indices 1..3 rebased to 0..2
    EXPECT_EQ(out.indices, (std::vector<u32>{0, 1, 2}));
    EXPECT_EQ(out.enabled, static_cast<u16>((1u << 0) | (1u << 3)));
    EXPECT_FLOAT_EQ(out.attribute[0][0], 1.0f);
    EXPECT_FLOAT_EQ(out.attribute[0][4 + 1], 12.0f);
    EXPECT_FLOAT_EQ(out.attribute[0][3], 1.0f) << "w defaults to 1";
    EXPECT_FLOAT_EQ(out.attribute[3][0], static_cast<float>(0x30) / 255.0f) << "D3D bytes read as R G B A";
    EXPECT_FLOAT_EQ(out.attribute[3][2], static_cast<float>(0x10) / 255.0f);
    EXPECT_FLOAT_EQ(out.attribute[3][3], 1.0f);
}

TEST(Kelvin, DrawArraysAndOutOfRangeArraysReportAnError) {
    Kelvin k;
    RecordingSink sink;
    k.set_sink(&sink);
    k.method(mthd::kDmaVertexA, 3);
    k.method(mthd::kVertexDataArrayFormat + 0, (12u << 8) | (3u << 4) | 2u);
    k.method(mthd::kVertexDataArrayOffset + 0, 0x1FE0);
    k.method(mthd::kBeginEnd, 6);
    k.method(mthd::kDrawArrays, (3u << 24) | 10u);  // 4 vertices from index 10
    k.method(mthd::kBeginEnd, 0);
    ASSERT_EQ(sink.draws.size(), 1u);
    EXPECT_EQ(sink.draws[0].ranges, (std::vector<std::pair<u32, u32>>{{10, 4}}));
    std::vector<u8> ram(0x2000, 0);
    TableResolver handles;
    handles.obj = {0, 0xFFFFFFFFu, 0x3D, true};
    FetchedVertices out;
    EXPECT_FALSE(fetch_vertices(k.state(), handles, MemoryView{ram.data(), static_cast<u32>(ram.size())}, sink.draws[0], out));
    EXPECT_NE(out.error.find("outside RAM"), std::string::npos) << out.error;
    EXPECT_FALSE(fetch_vertices(k.state(), handles, MemoryView{nullptr, 0}, sink.draws[0], out));
    // A DMA object shorter than the array: refused before RAM is consulted.
    handles.obj = {0, 0x1FFF, 0x3D, true};
    EXPECT_FALSE(fetch_vertices(k.state(), handles, MemoryView{ram.data(), static_cast<u32>(ram.size())}, sink.draws[0], out));
    EXPECT_NE(out.error.find("DMA object"), std::string::npos) << out.error;
}

TEST(Kelvin, InlineArrayInterleavesEnabledAttributes) {
    Kelvin k;
    RecordingSink sink;
    k.set_sink(&sink);
    k.method(mthd::kVertexDataArrayFormat + 0, (3u << 4) | 2u);   // position: 3 floats
    k.method(mthd::kVertexDataArrayFormat + 12, (4u << 4) | 4u);  // diffuse: 4 OpenGL bytes
    k.method(mthd::kVertexDataArrayFormat + 36, (2u << 4) | 2u);  // texcoord0: 2 floats
    k.method(mthd::kBeginEnd, 8);
    for (u32 v = 0; v < 2; ++v) {
        k.method(mthd::kInlineArray, bits(1.0f + static_cast<float>(v)));
        k.method(mthd::kInlineArray, bits(2.0f));
        k.method(mthd::kInlineArray, bits(3.0f));
        k.method(mthd::kInlineArray, 0xFF00FF00);
        k.method(mthd::kInlineArray, bits(0.5f));
        k.method(mthd::kInlineArray, bits(0.25f));
    }
    k.method(mthd::kBeginEnd, 0);
    ASSERT_EQ(sink.draws.size(), 1u);
    TableResolver handles;
    FetchedVertices out;
    ASSERT_TRUE(fetch_vertices(k.state(), handles, MemoryView{nullptr, 0}, sink.draws[0], out)) << out.error;
    EXPECT_EQ(out.count, 2u);
    EXPECT_EQ(out.primitive, Primitive::Quads);
    EXPECT_FLOAT_EQ(out.attribute[0][4], 2.0f);
    EXPECT_FLOAT_EQ(out.attribute[3][1], 1.0f);
    EXPECT_FLOAT_EQ(out.attribute[3][0], 0.0f);
    EXPECT_FLOAT_EQ(out.attribute[9][1], 0.25f);
    EXPECT_TRUE(out.indices.empty());
}

TEST(Kelvin, ImmediateVerticesCompleteOnPosition) {
    Kelvin k;
    RecordingSink sink;
    k.set_sink(&sink);
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kDiffuse4ub, 0xFF0000FF);  // sticky colour: red, alpha 1
    k.method(mthd::kTexcoord0_2f, bits(0.5f));
    k.method(mthd::kTexcoord0_2f + 4, bits(0.75f));
    k.method(mthd::kVertex3f, bits(1.0f));
    k.method(mthd::kVertex3f + 4, bits(2.0f));
    k.method(mthd::kVertex3f + 8, bits(3.0f));
    k.method(mthd::kVertexData4fM + 0, bits(4.0f));  // second vertex through the generic method
    k.method(mthd::kVertexData4fM + 4, bits(5.0f));
    k.method(mthd::kVertexData4fM + 8, bits(6.0f));
    k.method(mthd::kVertexData4fM + 12, bits(1.0f));
    k.method(mthd::kVertexData2s + 0, (0xFFFEu << 16) | 7u);  // third: x 7, y -2
    k.method(mthd::kBeginEnd, 0);
    ASSERT_EQ(sink.draws.size(), 1u);
    EXPECT_EQ(sink.draws[0].immediate_count, 3u);
    TableResolver handles;
    FetchedVertices out;
    ASSERT_TRUE(fetch_vertices(k.state(), handles, MemoryView{nullptr, 0}, sink.draws[0], out)) << out.error;
    EXPECT_EQ(out.count, 3u);
    EXPECT_TRUE(out.enabled & (1u << 3));
    EXPECT_TRUE(out.enabled & (1u << 9));
    EXPECT_FLOAT_EQ(out.attribute[0][2], 3.0f);
    EXPECT_FLOAT_EQ(out.attribute[0][3], 1.0f);
    EXPECT_FLOAT_EQ(out.attribute[0][4], 4.0f);
    EXPECT_FLOAT_EQ(out.attribute[0][8], 7.0f);
    EXPECT_FLOAT_EQ(out.attribute[0][9], -2.0f);
    EXPECT_FLOAT_EQ(out.attribute[3][0], 1.0f) << "colour stays for every vertex";
    EXPECT_FLOAT_EQ(out.attribute[3][8], 1.0f);
    EXPECT_FLOAT_EQ(out.attribute[9][5], 0.75f);
}

TEST(Kelvin, ClearsAndSemaphoresReachTheSink) {
    Kelvin k;
    RecordingSink sink;
    k.set_sink(&sink);
    k.method(mthd::kColorClear, 0xFF102030);
    k.method(mthd::kClearSurface, 0xF3);
    k.method(mthd::kSemaphoreRelease, 5);
    EXPECT_EQ(sink.clears, (std::vector<u32>{0xF3}));
    EXPECT_EQ(sink.semaphores, (std::vector<u32>{5}));
    EXPECT_EQ(k.state().clear_color(), 0xFF102030u);
    EXPECT_EQ(k.methods_seen(), 3u);
    EXPECT_EQ(k.histogram()[mthd::kClearSurface >> 2], 1u);
}
