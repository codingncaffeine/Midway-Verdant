#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pgraph/replay.h"
#include "nv2a/shaders/glsl_compiler.h"
#include "nv2a/shaders/vsh_translate.h"

using namespace mv;
using namespace mv::pgraph;
using namespace mv::pgraph::vsh;

namespace {

// Our own encoder from the documented field layout; the decoder must invert it.
struct Enc {
    std::array<u32, 4> t{};
    Enc& mac(Mac m) { t[1] |= static_cast<u32>(m) << 21; return *this; }
    Enc& ilu(Ilu i) { t[1] |= static_cast<u32>(i) << 25; return *this; }
    Enc& input(u32 v) { t[1] |= (v & 15) << 9; return *this; }
    Enc& constant(u32 c) { t[1] |= (c & 0xFF) << 13; return *this; }
    static u32 swz(u32 x, u32 y, u32 z, u32 w) { return (x << 6) | (y << 4) | (z << 2) | w; }
    Enc& a(Source s, u32 reg, u32 swizzle = swz(0, 1, 2, 3), bool neg = false) {
        t[2] |= static_cast<u32>(s) << 26;
        t[2] |= (reg & 15) << 28;
        t[1] |= swizzle & 0xFF;
        if (neg) t[1] |= 1u << 8;
        return *this;
    }
    Enc& b(Source s, u32 reg, u32 swizzle = swz(0, 1, 2, 3), bool neg = false) {
        t[2] |= static_cast<u32>(s) << 11;
        t[2] |= (reg & 15) << 13;
        t[2] |= (swizzle & 0xFF) << 17;
        if (neg) t[2] |= 1u << 25;
        return *this;
    }
    Enc& c(Source s, u32 reg, u32 swizzle = swz(0, 1, 2, 3), bool neg = false) {
        t[3] |= static_cast<u32>(s) << 28;
        t[2] |= (reg >> 2) & 3;
        t[3] |= (reg & 3) << 30;
        t[2] |= (swizzle & 0xFF) << 2;
        if (neg) t[2] |= 1u << 10;
        return *this;
    }
    Enc& temp(u32 r, u32 mac_mask, u32 ilu_mask = 0) {
        t[3] |= (r & 15) << 20;
        t[3] |= (mac_mask & 15) << 24;
        t[3] |= (ilu_mask & 15) << 16;
        return *this;
    }
    Enc& out(u32 index, u32 mask, bool from_ilu = false) {
        t[3] |= 1u << 11;  // an output register, not a constant
        t[3] |= (index & 0xFF) << 3;
        t[3] |= (mask & 15) << 12;
        if (from_ilu) t[3] |= 1u << 2;
        return *this;
    }
    Enc& relative() { t[3] |= 1u << 1; return *this; }
    Enc& fin() { t[3] |= 1u; return *this; }
};

constexpr u32 kX = 8, kY = 4, kZ = 2, kW = 1;

}  // namespace

TEST(Vsh, DecodesWhatTheEncoderWrote) {
    // dp4 oPos.x, v0, c[96]  (a typical first instruction of an XDK shader)
    const auto t = Enc().mac(Mac::Dp4).input(0).constant(96).a(Source::Input, 0).b(Source::Constant, 0).out(0, kX).t;
    const Instruction i = decode(t);
    EXPECT_EQ(i.mac, Mac::Dp4);
    EXPECT_EQ(i.ilu, Ilu::Nop);
    EXPECT_EQ(i.a.source, Source::Input);
    EXPECT_EQ(i.a.index, 0u);
    EXPECT_EQ(i.b.source, Source::Constant);
    EXPECT_EQ(i.b.index, 96u);
    EXPECT_EQ(i.output_mask, kX);
    EXPECT_EQ(i.output_index, 0u);
    EXPECT_FALSE(i.final);
    EXPECT_EQ(disassemble(i), "dp4 oPos.x, v0, c[96]");

    // mad R2.xyz, -R1.wzyx, c[A0+5], v3.x | rcc R1.w, R0.w ; final
    const auto p = Enc()
                       .mac(Mac::Mad)
                       .ilu(Ilu::Rcc)
                       .input(3)
                       .constant(5)
                       .relative()
                       .a(Source::Temp, 1, Enc::swz(3, 2, 1, 0), true)
                       .b(Source::Constant, 0)
                       .c(Source::Input, 0, Enc::swz(0, 0, 0, 0))
                       .temp(2, kX | kY | kZ, kW)
                       .fin()
                       .t;
    const Instruction j = decode(p);
    EXPECT_EQ(j.mac, Mac::Mad);
    EXPECT_EQ(j.ilu, Ilu::Rcc);
    EXPECT_TRUE(j.a.negate);
    EXPECT_EQ(j.a.swizzle, (std::array<u8, 4>{3, 2, 1, 0}));
    EXPECT_TRUE(j.b.relative);
    EXPECT_EQ(j.b.index, 5u);
    EXPECT_EQ(j.c.source, Source::Input);
    EXPECT_EQ(j.c.index, 3u);
    EXPECT_EQ(j.temp, 2u);
    EXPECT_TRUE(j.final);
    EXPECT_EQ(disassemble(j), "mad R2.xyz, -R1.wzyx, c[A0+5], v3.x | rcc R1.w, v3.x  ; final");
}

TEST(Vsh, TranslatesAndCompilesAHandWrittenProgram) {
    KelvinState s;
    s.r[mthd::kTransformProgramStart >> 2] = 0;
    u32 slot = 0;
    auto put = [&](const std::array<u32, 4>& t) { s.program[slot++] = t; };
    // Transform by the matrix in c[96..99], divide by w with rcc, scale into screen space with c[58]/c[59].
    for (u32 row = 0; row < 4; ++row) put(Enc().mac(Mac::Dp4).input(0).constant(96 + row).a(Source::Input, 0).b(Source::Constant, 0).temp(12, kX >> row).t);
    put(Enc().ilu(Ilu::Rcc).c(Source::Temp, 12, Enc::swz(3, 3, 3, 3)).temp(1, 0, kX).t);
    put(Enc().mac(Mac::Mul).constant(58).a(Source::Temp, 12).b(Source::Constant, 0).temp(12, kX | kY | kZ).t);
    put(Enc().mac(Mac::Mad).constant(59).a(Source::Temp, 12).b(Source::Temp, 1, Enc::swz(0, 0, 0, 0)).c(Source::Constant, 0).out(0, kX | kY | kZ).t);
    put(Enc().mac(Mac::Mov).input(3).a(Source::Input, 0).out(3, 15).t);
    put(Enc().mac(Mac::Mov).input(9).a(Source::Input, 0).out(9, 15).t);
    put(Enc().mac(Mac::Arl).a(Source::Input, 0, Enc::swz(0, 0, 0, 0)).fin().t);
    const Program p = extract_program(s);
    ASSERT_EQ(p.instructions.size(), 10u);
    EXPECT_FALSE(p.truncated);
    const Translation tr = translate(p, Options{});
    EXPECT_TRUE(tr.error.empty()) << tr.error;
    EXPECT_EQ(tr.inputs_read, static_cast<u16>((1u << 0) | (1u << 3) | (1u << 9)));
    EXPECT_NE(tr.glsl.find("vp_rcc((oPos).wwww)"), std::string::npos) << tr.glsl;
    // The host position is multiplied back by the SIGNED w, so geometry behind the eye clips at w = 0 instead of
    // mirroring onto the screen (|w| drew the boot animation's black wedges).
    EXPECT_NE(tr.glsl.find("gl_Position = vec4(ndc * wa, zndc * wa, wa);"), std::string::npos) << tr.glsl;
    EXPECT_EQ(tr.glsl.find("abs(oPos.w)"), std::string::npos) << tr.glsl;
    const CompiledShader c = compile_glsl(ShaderStage::Vertex, tr.glsl, "hand-written");
    EXPECT_TRUE(c.ok()) << c.log << "\n" << tr.glsl;
}

TEST(Vsh, RefusesNonsenseGlslAsAControl) {
    const CompiledShader c = compile_glsl(ShaderStage::Vertex, "#version 450\nvoid main() { gl_Position = nonsense; }\n", "control");
    EXPECT_FALSE(c.ok());
    EXPECT_NE(c.log.find("nonsense"), std::string::npos) << c.log;
}

namespace {
struct FirstDrawSink : RenderSink {
    KelvinState captured;
    bool have{false};
    u32 target_frame{2};
    u32 frame{0};
    void clear(const KelvinState&, u32) override {}
    void draw(const KelvinState& s, const DrawBatch&) override {
        if (!have && frame == target_frame) {
            captured = s;
            have = true;
        }
    }
    void semaphore_release(const KelvinState&, u32) override { ++frame; }
};
}  // namespace

TEST(Vsh, TranslatesTheBootAnimationsOwnProgram) {
    std::filesystem::path capture = std::filesystem::path(MV_SOURCE_DIR) / "scratch" / "boot-anim.mvcap";
    if (const char* env = std::getenv("MV_BOOT_CAPTURE")) capture = env;
    if (!std::filesystem::exists(capture)) GTEST_SKIP() << "no boot-animation capture at " << capture;
    Replay r;
    std::string error;
    ASSERT_TRUE(r.open(capture, error)) << error;
    Kelvin k;
    FirstDrawSink sink;
    k.set_sink(&sink);
    ASSERT_TRUE(r.run(k, nullptr, error, 3)) << error;
    ASSERT_TRUE(sink.have);
    EXPECT_TRUE(sink.captured.vertex_program_mode());
    const Program p = extract_program(sink.captured);
    EXPECT_FALSE(p.truncated);
    EXPECT_GT(p.instructions.size(), 3u);
    const Translation tr = translate(p, Options{});
    EXPECT_TRUE(tr.error.empty()) << tr.error;
    EXPECT_TRUE(tr.inputs_read & 1u) << "a program that never reads the position";
    const CompiledShader c = compile_glsl(ShaderStage::Vertex, tr.glsl, "boot-animation");
    EXPECT_TRUE(c.ok()) << c.log << "\n" << tr.glsl;
    std::string listing;
    for (const auto& line : tr.disassembly) listing += line + "\n";
    EXPECT_NE(listing.find("dp4"), std::string::npos) << listing;
    std::printf("boot animation program (%zu instructions):\n%s", p.instructions.size(), listing.c_str());
}

TEST(Vsh, TheFixedFunctionPipelineIsGeneratedForEveryValidState) {
    // The firmware's state (mode 0, unlit): the position through the composite matrix and the viewport offset,
    // the colours passed through, and with specular off the specular colour is black.
    KelvinState s;
    std::string why;
    ASSERT_TRUE(fixed_function_supported(s, why)) << why;
    Translation tr = fixed_function_translate(s, Options{});
    ASSERT_TRUE(tr.error.empty()) << tr.error;
    EXPECT_NE(tr.glsl.find("vec4 p = xf_mul(v0, 0);"), std::string::npos) << tr.glsl;
    EXPECT_NE(tr.glsl.find("c[59].xyz"), std::string::npos);
    EXPECT_NE(tr.glsl.find("oD0 = v3;"), std::string::npos);
    EXPECT_NE(tr.glsl.find("oD1 = vec4(0.0, 0.0, 0.0, 1.0);"), std::string::npos);
    CompiledShader c = compile_glsl(ShaderStage::Vertex, tr.glsl, "fixed-function");
    ASSERT_TRUE(c.ok()) << c.log << "\n" << tr.glsl;

    // A sweep over the features: every light type in every slot, the colour sources, local eye and separate
    // specular, skinning, normalisation, every texgen mode, texture matrices and fog generation. Each compiles.
    const u32 texgens[] = {0, 0x2400, 0x2401, 0x2402, 0x8512, 0x8511};
    const u32 fog_modes[] = {0, 1, 2, 3, 6};
    for (u32 n = 0; n < 24; ++n) {
        KelvinState f;
        f.r[mthd::kLightingEnable >> 2] = n % 3 != 2;
        f.r[mthd::kLightEnableMask >> 2] = (0x1B1Bu * (n + 1)) & 0xFFFFu;  // a mix of infinite, local and spot
        f.r[mthd::kColorMaterial >> 2] = (n * 0x25u) & 0x55u;               // sources 0 or 1 in each field...
        if (n % 4 == 3) f.r[mthd::kColorMaterial >> 2] = 0xAA;              // ...or all from the vertex specular
        f.r[0x0294 >> 2] = (n & 1u) | ((n & 2u) ? (1u << 16) : 0u);           // separate specular, local eye
        f.r[0x03B8 >> 2] = n % 5 != 0;                                         // specular enable
        f.r[0x0328 >> 2] = n % 7;                                              // skin mode
        f.r[0x03A4 >> 2] = n & 1u;                                             // normalisation
        f.r[0x09CC >> 2] = (n >> 1) & 1u;                                      // infinite viewer
        for (u32 t = 0; t < 4; ++t) {
            f.r[(0x0420 >> 2) + t] = (n + t) & 1u;
            for (u32 k = 0; k < 4; ++k) {
                u32 g = texgens[(n + t + k) % 6];
                if ((g == 0x2402 && k >= 2) || ((g == 0x8512 || g == 0x8511) && k == 3)) g = 0x2400;
                f.r[(0x03C0 >> 2) + 4 * t + k] = g;
            }
        }
        f.r[0x02A4 >> 2] = 1;
        f.r[0x02A0 >> 2] = fog_modes[n % 5];
        ASSERT_TRUE(fixed_function_supported(f, why)) << n << ": " << why;
        tr = fixed_function_translate(f, Options{});
        ASSERT_TRUE(tr.error.empty()) << n << ": " << tr.error;
        c = compile_glsl(ShaderStage::Vertex, tr.glsl, "fixed-function-sweep");
        ASSERT_TRUE(c.ok()) << n << ": " << c.log << "\n" << tr.glsl;
        if (f.r[mthd::kLightingEnable >> 2] && f.r[mthd::kLightEnableMask >> 2]) {
            EXPECT_NE(tr.glsl.find("// light "), std::string::npos) << n;
        }
    }

    // Invalid enumerants are refused with a reason.
    KelvinState bad;
    bad.r[0x0328 >> 2] = 7;
    EXPECT_FALSE(fixed_function_supported(bad, why));
    EXPECT_NE(why.find("skin"), std::string::npos);
    bad = KelvinState{};
    bad.r[(0x03C0 >> 2) + 3] = 0x2402;  // a sphere map on Q
    EXPECT_FALSE(fixed_function_supported(bad, why));
    EXPECT_NE(why.find("texgen"), std::string::npos);
    bad = KelvinState{};
    bad.r[mthd::kLightingEnable >> 2] = 1;
    bad.r[mthd::kColorMaterial >> 2] = 3;
    EXPECT_FALSE(fixed_function_supported(bad, why));
}

TEST(Vsh, TheFixedFunctionUniformsCarryTheLightsWhereTheShaderReadsThem) {
    KelvinState s;
    auto setf = [&](u32 method, float v) { std::memcpy(&s.r[method >> 2], &v, 4); };
    setf(mthd::kMaterialEmission + 4, 0.25f);
    setf(mthd::kMaterialAlpha, 0.5f);
    setf(mthd::kSceneAmbientColor + 8, 0.125f);
    const u32 light3 = 0x1000 + 3 * 0x80;
    setf(light3 + 0x0C, 0.75f);          // diffuse red
    setf(light3 + 0x24, 100.0f);         // range
    setf(light3 + 0x5C + 8, -4.0f);      // local position z
    setf(light3 + 0x68 + 4, 0.5f);       // attenuation, linear term
    setf(light3 + 0x4C + 12, 0.875f);    // spot direction w
    std::vector<float> u(kFixedFunctionVectors * 4, -1.0f);
    fixed_function_uniforms(s, u.data());
    EXPECT_EQ(u[1], 0.25f) << "emission green";
    EXPECT_EQ(u[3], 0.5f) << "material alpha";
    EXPECT_EQ(u[6], 0.125f) << "scene ambient blue";
    EXPECT_FLOAT_EQ(u[9], -0.494592f) << "an unset specular curve takes the reset values";
    const std::size_t l = 16 + 3 * kLightVectors * 4;
    EXPECT_EQ(u[l + 4], 0.75f) << "ff_light[b + 1] = diffuse";
    EXPECT_EQ(u[l + 12 + 2], -4.0f) << "ff_light[b + 3].xyz = local position";
    EXPECT_EQ(u[l + 15], 100.0f) << "ff_light[b + 3].w = range";
    EXPECT_EQ(u[l + 16 + 1], 0.5f) << "ff_light[b + 4] = attenuation";
    EXPECT_EQ(u[l + 32 + 3], 0.875f) << "ff_light[b + 8].w = spot direction w";
    EXPECT_EQ(u[l - 1], 0.0f) << "light 2's spot direction w is clear";
    // The shader reads them at those indices: light 3's diffuse is ff_light[28], its position ff_light[30].
    s.r[mthd::kLightingEnable >> 2] = 1;
    s.r[mthd::kLightEnableMask >> 2] = 3u << 6;  // light 3 a spot
    const Translation tr = fixed_function_translate(s, Options{});
    EXPECT_NE(tr.glsl.find("ff_light[28].xyz * (attenuation * nDotVP)"), std::string::npos) << tr.glsl;
    EXPECT_NE(tr.glsl.find("VP = ff_light[30].xyz"), std::string::npos) << tr.glsl;
    EXPECT_NE(tr.glsl.find("dot(ff_light[35].xyz, VP) + ff_light[35].w"), std::string::npos) << tr.glsl;
    EXPECT_EQ(vsh::kUniformBytes, (kConstantSlots + 2 + 4 + 8 * 9) * 16);
}
