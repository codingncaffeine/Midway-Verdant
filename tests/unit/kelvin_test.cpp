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
    // The viewport registers land in the constant file where the XDK's programs read them.
    k.method(mthd::kViewportScale + 4, bits(-128.0f));
    k.method(mthd::kViewportOffset, bits(128.5f));
    k.method(0x0440 + 4 * 5, bits(7.0f));  // projection matrix row 1, column 1
    EXPECT_EQ(k.state().constants[58][1], bits(-128.0f));
    EXPECT_EQ(k.state().constants[59][0], bits(128.5f));
    EXPECT_EQ(k.state().constants[5][1], bits(7.0f));
    EXPECT_EQ(k.state().reg(mthd::kViewportOffset), bits(128.5f)) << "and stay readable as registers";
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

TEST(Kelvin, ShortNamedSettersFillTheComponentsTheyLeaveOut) {
    Kelvin k;
    const u32 fours[4] = {mthd::kTexcoord0_4f, mthd::kTexcoord1_4f, mthd::kTexcoord2_4f, mthd::kTexcoord3_4f};
    const u32 twos[4] = {mthd::kTexcoord0_2f, mthd::kTexcoord1_2f, mthd::kTexcoord2_2f, mthd::kTexcoord3_2f};
    for (u32 t = 0; t < 4; ++t) {
        const u32 four = fours[t], two = twos[t];
        for (u32 c = 0; c < 4; ++c) k.method(four + 4 * c, bits(5.0f + static_cast<float>(c)));
        EXPECT_FLOAT_EQ(k.state().immediate[9 + t][3], 8.0f) << "texcoord " << t;
        k.method(two, bits(0.25f));
        k.method(two + 4, bits(0.5f));
        const auto& v = k.state().immediate[9 + t];
        EXPECT_FLOAT_EQ(v[0], 0.25f) << "texcoord " << t;
        EXPECT_FLOAT_EQ(v[1], 0.5f) << "texcoord " << t;
        EXPECT_FLOAT_EQ(v[2], 0.0f) << "texcoord " << t << ": a 2F coordinate after a 4F one has z = 0";
        EXPECT_FLOAT_EQ(v[3], 1.0f) << "texcoord " << t << ": and w = 1";
    }
    for (u32 c = 0; c < 4; ++c) k.method(mthd::kDiffuse4f + 4 * c, bits(0.5f));
    for (u32 c = 0; c < 3; ++c) k.method(mthd::kDiffuse3f + 4 * c, bits(0.25f));
    EXPECT_FLOAT_EQ(k.state().immediate[3][3], 1.0f) << "a 3F colour is opaque";
    for (u32 c = 0; c < 4; ++c) k.method(mthd::kWeight4f + 4 * c, bits(0.75f));
    k.method(mthd::kWeight1f, bits(0.125f));
    EXPECT_EQ(k.state().immediate[1], (std::array<float, 4>{0.125f, 0.0f, 0.0f, 1.0f})) << "attribute 1 is the weight";
    k.method(mthd::kWeight3f + 4, bits(0.5f));
    EXPECT_FLOAT_EQ(k.state().immediate[1][1], 0.5f);
    k.method(mthd::kFogCoord, bits(3.0f));
    EXPECT_EQ(k.state().immediate[5], (std::array<float, 4>{3.0f, 3.0f, 3.0f, 3.0f})) << "attribute 5 is the fog coordinate, in every lane";
}

TEST(Kelvin, ConstantAttributesReadTheirOneValue) {
    Kelvin k;
    RecordingSink sink;
    k.set_sink(&sink);
    std::vector<u8> ram(0x2000, 0);
    TableResolver handles;
    handles.obj = {0, 0x1FFF, 0x3D, true};
    const MemoryView memory{ram.data(), static_cast<u32>(ram.size())};
    k.set_vertex_memory(&handles, memory);
    for (u32 v = 0; v < 4; ++v) {
        const float xyz[3] = {static_cast<float>(v), 0.0f, 0.0f};
        std::memcpy(&ram[0x1000 + v * 12], xyz, 12);
    }
    const u8 colours[8] = {10, 20, 30, 40, 50, 60, 70, 80};  // the second element must never be read
    std::memcpy(&ram[0x1800], colours, 8);
    k.method(mthd::kDmaVertexA, 3);
    k.method(mthd::kVertexDataArrayFormat + 0, (12u << 8) | (3u << 4) | 2u);
    k.method(mthd::kVertexDataArrayOffset + 0, 0x1000);
    k.method(mthd::kVertexDataArrayFormat + 12, (0u << 8) | (4u << 4) | 4u);  // diffuse: 4 OpenGL bytes, stride 0
    k.method(mthd::kVertexDataArrayOffset + 12, 0x1800);
    k.method(mthd::kVertexDataArrayFormat + 16, 2u);                           // specular: disabled
    k.method(mthd::kSpecular4ub, 0xFF000080);  // its current value (A B G R from the top): red 0x80, alpha 1
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kDrawArrays, (2u << 24) | 1u);  // vertices 1..3
    k.method(mthd::kBeginEnd, 0);
    ASSERT_EQ(sink.draws.size(), 1u);
    FetchedVertices out;
    ASSERT_TRUE(fetch_vertices(k.state(), handles, memory, sink.draws[0], out)) << out.error;
    EXPECT_EQ(out.count, 3u);
    EXPECT_EQ(out.enabled, 1u) << "the stride-0 and the disabled attribute carry no per-vertex data";
    EXPECT_FLOAT_EQ(out.constant[3][0], 10.0f / 255.0f) << "a stride of 0 reads element 0 for every vertex";
    EXPECT_FLOAT_EQ(out.constant[3][3], 40.0f / 255.0f);
    EXPECT_FLOAT_EQ(out.constant[4][0], 128.0f / 255.0f) << "a disabled attribute reads its current value";
    EXPECT_FLOAT_EQ(out.constant[4][3], 1.0f);
    EXPECT_FLOAT_EQ(out.constant[5][3], 1.0f) << "an attribute never set reads (0, 0, 0, 1)";
    EXPECT_FLOAT_EQ(k.state().immediate[0][0], 3.0f) << "the draw left its last vertex as the current position";
    EXPECT_FLOAT_EQ(k.state().immediate[3][1], 20.0f / 255.0f) << "and the stride-0 element as the current diffuse";

    // The next draw disables the diffuse array: it reads what the last draw left. An immediate batch that sets
    // only positions reads the same.
    k.method(mthd::kVertexDataArrayFormat + 12, 2u);
    k.method(mthd::kVertexDataArrayFormat + 16, (12u << 8) | (3u << 4) | 2u);  // specular from the position array
    k.method(mthd::kVertexDataArrayOffset + 16, 0x1000);
    k.method(mthd::kBeginEnd, 5);
    k.method(mthd::kArrayElement16, (2u << 16) | 0u);
    k.method(mthd::kArrayElement32, 1);
    k.method(mthd::kBeginEnd, 0);
    ASSERT_EQ(sink.draws.size(), 2u);
    ASSERT_TRUE(fetch_vertices(k.state(), handles, memory, sink.draws[1], out)) << out.error;
    EXPECT_FLOAT_EQ(out.constant[3][2], 30.0f / 255.0f);
    EXPECT_FLOAT_EQ(k.state().immediate[4][0], 1.0f) << "the last index (1), not the largest (2), is the last vertex";
    k.method(mthd::kBeginEnd, 5);
    for (u32 v = 0; v < 3; ++v)
        for (u32 c = 0; c < 3; ++c) k.method(mthd::kVertex3f + 4 * c, bits(static_cast<float>(v)));
    k.method(mthd::kBeginEnd, 0);
    ASSERT_EQ(sink.draws.size(), 3u);
    ASSERT_TRUE(fetch_vertices(k.state(), handles, memory, sink.draws[2], out)) << out.error;
    EXPECT_EQ(out.enabled, 1u);
    EXPECT_FLOAT_EQ(out.constant[3][0], 10.0f / 255.0f) << "an immediate batch keeps the colour set before it";
    EXPECT_FLOAT_EQ(out.constant[4][0], 1.0f);

    // Inline data latches too: its last vertex.
    k.method(mthd::kVertexDataArrayFormat + 16, 2u);
    k.method(mthd::kBeginEnd, 2);
    for (float x : {7.0f, 8.0f})
        for (float c : {x, 0.0f, 0.0f}) k.method(mthd::kInlineArray, bits(c));
    k.method(mthd::kBeginEnd, 0);
    EXPECT_FLOAT_EQ(k.state().immediate[0][0], 8.0f);
}
