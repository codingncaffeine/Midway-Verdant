#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pgraph/replay.h"
#include "nv2a/shaders/vsh_interpreter.h"
#include "nv2a/shaders/vsh_translate.h"

using namespace mv;
using namespace mv::pgraph;
using namespace mv::pgraph::vsh;

namespace {
u32 bits(float f) {
    u32 v;
    std::memcpy(&v, &f, 4);
    return v;
}
// Encoder as in vsh_test: enough for dp4 / mul / rcc / mad.
struct Enc {
    std::array<u32, 4> t{};
    Enc& mac(Mac m) { t[1] |= static_cast<u32>(m) << 21; return *this; }
    Enc& ilu(Ilu i) { t[1] |= static_cast<u32>(i) << 25; return *this; }
    Enc& input(u32 v) { t[1] |= (v & 15) << 9; return *this; }
    Enc& constant(u32 c) { t[1] |= (c & 0xFF) << 13; return *this; }
    static u32 swz(u32 x, u32 y, u32 z, u32 w) { return (x << 6) | (y << 4) | (z << 2) | w; }
    Enc& a(Source s, u32 reg, u32 sw = swz(0, 1, 2, 3)) { t[2] |= static_cast<u32>(s) << 26; t[2] |= (reg & 15) << 28; t[1] |= sw & 0xFF; return *this; }
    Enc& b(Source s, u32 reg, u32 sw = swz(0, 1, 2, 3)) { t[2] |= static_cast<u32>(s) << 11; t[2] |= (reg & 15) << 13; t[2] |= (sw & 0xFF) << 17; return *this; }
    Enc& c(Source s, u32 reg, u32 sw = swz(0, 1, 2, 3)) { t[3] |= static_cast<u32>(s) << 28; t[2] |= (reg >> 2) & 3; t[3] |= (reg & 3) << 30; t[2] |= (sw & 0xFF) << 2; return *this; }
    Enc& temp(u32 r, u32 mac_mask, u32 ilu_mask = 0) { t[3] |= (r & 15) << 20; t[3] |= (mac_mask & 15) << 24; t[3] |= (ilu_mask & 15) << 16; return *this; }
    Enc& out(u32 index, u32 mask) { t[3] |= 1u << 11; t[3] |= (index & 0xFF) << 3; t[3] |= (mask & 15) << 12; return *this; }
    Enc& fin() { t[3] |= 1u; return *this; }
};
[[maybe_unused]] constexpr u32 kX = 8, kY = 4, kZ = 2, kW = 1;
}  // namespace

TEST(VshInterpreter, RunsTheXdkScreenSpaceEpilogue) {
    KelvinState s;
    u32 slot = 0;
    auto put = [&](const std::array<u32, 4>& t) { s.program[slot++] = t; };
    for (u32 row = 0; row < 4; ++row) put(Enc().mac(Mac::Dp4).input(0).constant(96 + row).a(Source::Input, 0).b(Source::Constant, 0).temp(12, kX >> row).t);
    put(Enc().mac(Mac::Mul).constant(58).a(Source::Temp, 12).b(Source::Constant, 0).temp(12, kX | kY | kZ).ilu(Ilu::Rcc).c(Source::Temp, 12, Enc::swz(3, 3, 3, 3)).temp(12, kX | kY | kZ, kX).t);
    put(Enc().mac(Mac::Mad).constant(59).a(Source::Temp, 12).b(Source::Temp, 1, Enc::swz(0, 0, 0, 0)).c(Source::Constant, 0).out(0, kX | kY | kZ).fin().t);
    // Identity transform except w = 2 * z + 4; viewport scale (128, -128, 1000), offset (128, 128, 0).
    const float m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 2, 4}};
    for (u32 r = 0; r < 4; ++r)
        for (u32 c = 0; c < 4; ++c) s.constants[96 + r][c] = bits(m[r][c]);
    const float scale[4] = {128, -128, 1000, 0}, offset[4] = {128, 128, 0, 0};
    for (u32 c = 0; c < 4; ++c) {
        s.constants[58][c] = bits(scale[c]);
        s.constants[59][c] = bits(offset[c]);
    }
    const Program p = extract_program(s);
    ASSERT_EQ(p.instructions.size(), 6u);
    std::array<Vec4, kAttributes> in{};
    in[0] = Vec4{{1.0f, -0.5f, 2.0f, 1.0f}};  // w = 2*2 + 4 = 8
    const Outputs o = run(p, in, s.constants);
    EXPECT_FLOAT_EQ(o.pos[3], 8.0f);
    EXPECT_FLOAT_EQ(o.pos[0], 1.0f * 128.0f / 8.0f + 128.0f);
    EXPECT_FLOAT_EQ(o.pos[1], -0.5f * -128.0f / 8.0f + 128.0f);
    EXPECT_FLOAT_EQ(o.pos[2], 2.0f * 1000.0f / 8.0f);
}

namespace {
struct DrawSink : RenderSink {
    const Replay* replay{nullptr};
    u32 frame{0}, n{0}, want_frame{2}, want_draw{21};
    KelvinState state;
    FetchedVertices vertices;
    bool have{false};
    void clear(const KelvinState&, u32) override {}
    void semaphore_release(const KelvinState&, u32) override { ++frame; n = 0; }
    void draw(const KelvinState& s, const DrawBatch& b) override {
        if (have || frame != want_frame || n++ != want_draw) return;
        state = s;
        have = fetch_vertices(s, *replay, replay->memory(), b, vertices);
    }
};
}  // namespace

TEST(VshInterpreter, PlacesTheBootAnimationsGeometryOnItsTarget) {
    std::filesystem::path capture = std::filesystem::path(MV_SOURCE_DIR) / "scratch" / "boot-anim.mvcap";
    if (const char* env = std::getenv("MV_BOOT_CAPTURE")) capture = env;
    if (!std::filesystem::exists(capture)) GTEST_SKIP() << "no boot-animation capture at " << capture;
    Replay r;
    std::string error;
    ASSERT_TRUE(r.open(capture, error)) << error;
    Kelvin k;
    DrawSink sink;
    sink.replay = &r;
    k.set_sink(&sink);
    ASSERT_TRUE(r.run(k, nullptr, error, 3)) << error;
    ASSERT_TRUE(sink.have);
    const Program p = extract_program(sink.state);
    // The viewport scale and offset reach the program through the constant file (slots 58 and 59).
    float scale_x;
    std::memcpy(&scale_x, &sink.state.constants[58][0], 4);
    EXPECT_FLOAT_EQ(scale_x, 128.0f);
    u32 inside = 0;
    for (u32 v = 0; v < sink.vertices.count; ++v) {
        std::array<Vec4, kAttributes> in{};
        for (u32 a = 0; a < kAttributes; ++a)
            if (sink.vertices.enabled & (1u << a)) std::memcpy(in[a].v, &sink.vertices.attribute[a][v * 4], 16);
        const Outputs o = run(p, in, sink.state.constants);
        if (o.pos[0] >= 0.0f && o.pos[0] <= 256.0f && o.pos[1] >= -64.0f && o.pos[1] <= 320.0f) ++inside;
    }
    EXPECT_GT(inside, 0u) << "a draw known to overlap the target has no vertex near it";
}
