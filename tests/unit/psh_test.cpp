#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>

#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pgraph/replay.h"
#include "nv2a/shaders/glsl_compiler.h"
#include "nv2a/shaders/psh_translate.h"

using namespace mv;
using namespace mv::pgraph;
using namespace mv::pgraph::psh;

namespace {
// One combiner input: register, mapping, alpha-channel flag, packed as the hardware does.
u32 in(u32 reg, u32 mapping = 0, bool alpha = false) { return (mapping << 5) | (alpha ? 0x10u : 0u) | reg; }
u32 icw(u32 a, u32 b, u32 c, u32 d) { return (a << 24) | (b << 16) | (c << 8) | d; }
}  // namespace

TEST(Psh, ModulateTextureByDiffuseCompiles) {
    KelvinState s;
    s.r[mthd::kCombinerControl >> 2] = 1;  // one stage
    // rgb: AB = t0 * v0 into r0; alpha: t0.a * v0.a into r0.
    s.r[mthd::kCombinerColorIcw >> 2] = icw(in(8), in(4), in(0), in(0));
    s.r[mthd::kCombinerColorOcw >> 2] = 12u << 4;
    s.r[mthd::kCombinerAlphaIcw >> 2] = icw(in(8, 0, true), in(4, 0, true), in(0), in(0));
    s.r[mthd::kCombinerAlphaOcw >> 2] = 12u << 4;
    // final: A = r0 (unsigned), B = one (zero inverted), C = zero, D = zero; G = r0.a.
    s.r[mthd::kCombinerSpecFogCw0 >> 2] = icw(in(12), in(0, 1), in(0), in(0));
    s.r[mthd::kCombinerSpecFogCw1 >> 2] = icw(in(0), in(0), in(12, 0, true), 0);
    s.r[mthd::kShaderStageProgram >> 2] = 1;  // stage 0: 2D projective
    s.r[(mthd::kTextureBase + 4) >> 2] = 0x0001002a;           // 2D, A8R8G8B8 swizzled... colour 0 (Y8) is fine too
    s.r[(mthd::kTextureBase + 0xC) >> 2] = 0x40000000;         // enabled
    s.r[mthd::kAlphaTestEnable >> 2] = 1;
    s.r[mthd::kAlphaFunc >> 2] = 0x204;  // greater
    const Key k = key_from_state(s);
    EXPECT_TRUE(k.texture_enabled[0]);
    const Translation t = translate(k);
    EXPECT_TRUE(t.error.empty()) << t.error;
    EXPECT_EQ(t.samplers[0], SamplerKind::Sampler2D);
    EXPECT_EQ(t.samplers[1], SamplerKind::None);
    EXPECT_NE(t.glsl.find("r0.rgb = ab;"), std::string::npos) << t.glsl;
    EXPECT_NE(t.glsl.find("if (!(round(clamp(final_color.a, 0.0, 1.0) * 255.0) > round(alpha_ref.x * 255.0))) discard;"), std::string::npos) << t.glsl;
    const CompiledShader c = compile_glsl(ShaderStage::Fragment, t.glsl, "modulate");
    EXPECT_TRUE(c.ok()) << c.log << "\n" << t.glsl;
}

TEST(Psh, UniformsDecodeFactorsFogAndAlphaReference) {
    KelvinState s;
    s.r[mthd::kCombinerFactor0 >> 2] = 0x80FF4020;
    s.r[(mthd::kCombinerFactor1 + 28) >> 2] = 0x01020304;
    s.r[0x02A8 >> 2] = 0xFF0000C0;  // fog colour register: A B G R from the top
    s.r[mthd::kAlphaRef >> 2] = 0x7F;
    s.r[(mthd::kTextureBase + 4) >> 2] = 0x00003000;  // depth Y16 format on stage 0
    Uniforms u{};
    uniforms_from_state(s, {{{1.0f, 1.0f}, {0.5f, 0.25f}, {1.0f, 1.0f}, {1.0f, 1.0f}}}, u);
    EXPECT_FLOAT_EQ(u.c0[0][0], 1.0f);               // red
    EXPECT_FLOAT_EQ(u.c0[0][1], 64.0f / 255.0f);     // green
    EXPECT_FLOAT_EQ(u.c0[0][2], 32.0f / 255.0f);     // blue
    EXPECT_FLOAT_EQ(u.c0[0][3], 128.0f / 255.0f);    // alpha
    EXPECT_FLOAT_EQ(u.c1[7][0], 2.0f / 255.0f);
    EXPECT_FLOAT_EQ(u.fog_color[0], 192.0f / 255.0f);
    EXPECT_FLOAT_EQ(u.fog_color[3], 1.0f);
    EXPECT_FLOAT_EQ(u.alpha_ref[0], 127.0f / 255.0f);
    EXPECT_FLOAT_EQ(u.tex_scale[1][0], 0.5f);
    EXPECT_FLOAT_EQ(u.tex_scale[0][2], 65535.0f);
    // The shadow compare's range per depth format: 24-bit (fixed, or float read as fixed) and the 16-bit float's
    // largest value.
    s.r[(mthd::kTextureBase + 0x40 + 4) >> 2] = 0x00002F00;
    s.r[(mthd::kTextureBase + 0x80 + 4) >> 2] = 0x00003100;
    s.r[(mthd::kTextureBase + 0xC0 + 4) >> 2] = 0x00002A00;
    uniforms_from_state(s, {{{1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}}}, u);
    EXPECT_FLOAT_EQ(u.tex_scale[1][2], 16777215.0f);
    EXPECT_FLOAT_EQ(u.tex_scale[2][2], 511.9375f);
    EXPECT_FLOAT_EQ(u.tex_scale[3][2], 16777215.0f);
}

namespace {
struct FirstDrawSink : RenderSink {
    KelvinState captured;
    bool have{false};
    u32 frame{0};
    void clear(const KelvinState&, u32) override {}
    void draw(const KelvinState& s, const DrawBatch&) override {
        if (!have && frame == 2) {
            captured = s;
            have = true;
        }
    }
    void semaphore_release(const KelvinState&, u32) override { ++frame; }
};
}  // namespace

TEST(Psh, TranslatesTheBootAnimationsCombiners) {
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
    const Key key = key_from_state(sink.captured);
    EXPECT_GE(key.combiner_control & 0xFFu, 1u);
    const Translation t = translate(key);
    EXPECT_TRUE(t.error.empty()) << t.error << "\n" << t.glsl;
    const CompiledShader c = compile_glsl(ShaderStage::Fragment, t.glsl, "boot-animation");
    EXPECT_TRUE(c.ok()) << c.log << "\n" << t.glsl;
    std::printf("boot animation combiners: %u stages, stage programs %05x, samplers %d %d %d %d\n", key.combiner_control & 0xFFu,
                key.shader_stage_program, static_cast<int>(t.samplers[0]), static_cast<int>(t.samplers[1]), static_cast<int>(t.samplers[2]),
                static_cast<int>(t.samplers[3]));
}

TEST(Psh, ShadowCompareTakesTheStoredDepthAsTheLeftOperand) {
    // A depth texture in a 3D-projective stage is a shadow lookup: t = (stored depth FUNC r). The function numbering
    // and the operand order are two sources' conventions that only agree this way round (nxdk: 4 = LESS on r OP
    // texel; the behaviour reference: 4 = GREATER on texel OP r); the reverse lit every shadowed face and darkened
    // every lit one.
    KelvinState s;
    s.r[mthd::kCombinerControl >> 2] = 1;
    s.r[mthd::kCombinerColorIcw >> 2] = icw(in(8), in(0, 1), in(0), in(0));
    s.r[mthd::kCombinerColorOcw >> 2] = 12u << 4;
    s.r[mthd::kShaderStageProgram >> 2] = 2;               // stage 0: 3D projective
    s.r[(mthd::kTextureBase + 4) >> 2] = 0x0001302a;       // 2D, colour 0x30 (16-bit depth)
    s.r[(mthd::kTextureBase + 0xC) >> 2] = 0x40000000;     // enabled
    s.r[0x1E6C >> 2] = 4;                                  // SET_SHADOW_COMPARE_FUNC
    const Translation t = translate(key_from_state(s));
    ASSERT_TRUE(t.error.empty()) << t.error;
    EXPECT_EQ(t.samplers[0], SamplerKind::Shadow2D);
    EXPECT_NE(t.glsl.find("texel > clamp(p.z, 0.0, tex_scale[0].z)"), std::string::npos) << t.glsl;
    const CompiledShader c = compile_glsl(ShaderStage::Fragment, t.glsl, "shadow");
    EXPECT_TRUE(c.ok()) << c.log << "\n" << t.glsl;
}

namespace {
// One stage of combiners whose final colour is texture stage `stage` (A = t, B = one; G = t.a).
KelvinState output_texture(u32 stage) {
    KelvinState s;
    s.r[mthd::kCombinerControl >> 2] = 1;
    s.r[mthd::kCombinerColorIcw >> 2] = icw(in(8 + stage), in(0, 1), in(0), in(0));
    s.r[mthd::kCombinerColorOcw >> 2] = 12u << 4;
    s.r[mthd::kCombinerSpecFogCw0 >> 2] = icw(in(12), in(0, 1), in(0), in(0));
    s.r[mthd::kCombinerSpecFogCw1 >> 2] = icw(in(0), in(0), in(8 + stage, 0, true), 0);
    return s;
}

void expect_compiles(const Translation& t, const char* name) {
    ASSERT_TRUE(t.error.empty()) << t.error;
    const CompiledShader c = compile_glsl(ShaderStage::Fragment, t.glsl, name);
    EXPECT_TRUE(c.ok()) << c.log << "\n" << t.glsl;
}
}  // namespace

TEST(Psh, PassThroughClampsTheCoordinatesToZeroOne) {
    KelvinState s = output_texture(0);
    s.r[mthd::kShaderStageProgram >> 2] = 4;
    const Translation t = translate(key_from_state(s));
    EXPECT_EQ(t.samplers[0], SamplerKind::None);
    EXPECT_NE(t.glsl.find("t0 = clamp(vtx_t0, vec4(0.0), vec4(1.0));"), std::string::npos) << t.glsl;
    expect_compiles(t, "pass-through");
}

TEST(Psh, ClipPlaneTakesEachComponentsSenseFromTheModeRegister) {
    KelvinState s = output_texture(1);
    s.r[mthd::kShaderStageProgram >> 2] = 5u << 5;           // stage 1: clip plane
    s.r[mthd::kShaderClipPlaneMode >> 2] = 0x5u << 4;         // stage 1: s and r discard at or above zero
    const Key k = key_from_state(s);
    EXPECT_EQ(k.clip_plane_mode, 0x50u);
    const Translation t = translate(k);
    EXPECT_NE(t.glsl.find("if (any(notEqual(lessThan(vtx_t1, vec4(0.0)), bvec4(true, false, true, false)))) discard;"), std::string::npos)
        << t.glsl;
    expect_compiles(t, "clip-plane");
    // All four senses zero: the plain test, discarding below zero.
    s.r[mthd::kShaderClipPlaneMode >> 2] = 0x5u;  // stage 0's bits only
    const Translation t0 = translate(key_from_state(s));
    EXPECT_NE(t0.glsl.find("if (any(lessThan(vtx_t1, vec4(0.0)))) discard;"), std::string::npos) << t0.glsl;
}

namespace {
u32 fbits(float f) {
    u32 v;
    std::memcpy(&v, &f, 4);
    return v;
}
// Texture stage `stage` enabled with the given FORMAT word (only its shader-shaping fields matter here).
void enable_texture(KelvinState& s, u32 stage, u32 format) {
    s.r[(mthd::kTextureBase + mthd::kTextureStride * stage + 4) >> 2] = format;
    s.r[(mthd::kTextureBase + mthd::kTextureStride * stage + 0xC) >> 2] = 0x40000000;
}
constexpr u32 kFormat2D = 0x00010620, kFormatCube = 0x00010624, kFormat3D = 0x00010630;  // A8R8G8B8, one level
}  // namespace

TEST(Psh, EveryDotModeAndMappingCompiles) {
    struct Case {
        const char* name;
        u32 program;  // SET_SHADER_STAGE_PROGRAM
        u32 out;      // the stage the combiners output
        std::array<u32, 4> formats;
        std::array<SamplerKind, 4> samplers;
    };
    using enum SamplerKind;
    const Case cases[] = {
        {"dot ST", 1u | (0x11u << 5) | (9u << 10), 2, {kFormat2D, 0, kFormat2D, 0}, {Sampler2D, None, Sampler2D, None}},
        {"dot ST on a volume", 1u | (0x11u << 5) | (9u << 10), 2, {kFormat2D, 0, kFormat3D, 0}, {Sampler2D, None, Sampler3D, None}},
        {"dot ZW", 1u | (0x11u << 5) | (0xAu << 10), 0, {kFormat2D, 0, 0, 0}, {Sampler2D, None, None, None}},
        {"dot STR 3D", 1u | (0x11u << 5) | (0x11u << 10) | (0xDu << 15), 3, {kFormat2D, 0, 0, kFormat3D}, {Sampler2D, None, None, Sampler3D}},
        {"dot STR cube", 1u | (0x11u << 5) | (0x11u << 10) | (0xEu << 15), 3, {kFormat2D, 0, 0, kFormatCube}, {Sampler2D, None, None, SamplerCube}},
        {"dot reflect", 1u | (0x11u << 5) | (0xBu << 10) | (0xCu << 15), 3, {kFormat2D, 0, kFormatCube, kFormatCube},
         {Sampler2D, None, SamplerCube, SamplerCube}},
        {"dot reflect const", 1u | (0x11u << 5) | (0x11u << 10) | (0x12u << 15), 3, {kFormat2D, 0, 0, kFormatCube},
         {Sampler2D, None, None, SamplerCube}},
        {"dot STR cube on a 2D texture", 1u | (0x11u << 5) | (0x11u << 10) | (0xEu << 15), 3, {kFormat2D, 0, 0, kFormat2D},
         {Sampler2D, None, None, None}},
    };
    for (const Case& c : cases) {
        for (u32 mapping = 0; mapping < 8; ++mapping) {
            KelvinState s = output_texture(c.out);
            s.r[mthd::kShaderStageProgram >> 2] = c.program;
            s.r[mthd::kDotRgbMapping >> 2] = mapping * 0x111u;
            for (u32 i = 0; i < 4; ++i)
                if (c.formats[i]) enable_texture(s, i, c.formats[i]);
            const Key k = key_from_state(s);
            EXPECT_EQ(k.dot_mapping, mapping * 0x111u);
            const Translation t = translate(k);
            SCOPED_TRACE(std::format("{}, mapping {}", c.name, mapping));
            EXPECT_EQ(t.samplers, c.samplers);
            expect_compiles(t, c.name);
            EXPECT_NE(t.glsl.find(std::format("dot1 = dot(vtx_t1.xyz, psh_dotmap{}(t0));", mapping)), std::string::npos) << t.glsl;
        }
    }
}

TEST(Psh, DotModesReadTheirSourceStageAndTheEyeVector) {
    KelvinState s = output_texture(3);
    s.r[mthd::kShaderStageProgram >> 2] = 1u | (0x11u << 5) | (0x11u << 10) | (0x12u << 15);
    s.r[mthd::kShaderOtherStageInput >> 2] = (1u << 16) | (2u << 20);  // stage 2 reads stage 1, stage 3 stage 2
    s.r[mthd::kDotRgbMapping >> 2] = 0x321;
    enable_texture(s, 0, kFormat2D);
    enable_texture(s, 3, kFormatCube);
    s.r[mthd::kEyeVector >> 2] = fbits(0.25f);
    s.r[(mthd::kEyeVector + 4) >> 2] = fbits(-0.5f);
    s.r[(mthd::kEyeVector + 8) >> 2] = fbits(2.0f);
    const Translation t = translate(key_from_state(s));
    EXPECT_NE(t.glsl.find("dot1 = dot(vtx_t1.xyz, psh_dotmap1(t0));"), std::string::npos) << t.glsl;
    EXPECT_NE(t.glsl.find("dot2 = dot(vtx_t2.xyz, psh_dotmap2(t1));"), std::string::npos) << t.glsl;
    EXPECT_NE(t.glsl.find("dot3 = dot(vtx_t3.xyz, psh_dotmap3(t2));"), std::string::npos) << t.glsl;
    EXPECT_NE(t.glsl.find("vec3 e = eye.xyz;"), std::string::npos) << t.glsl;
    expect_compiles(t, "dot sources");
    // A source that is not an earlier stage has no result yet: zero.
    s.r[mthd::kShaderOtherStageInput >> 2] = 3u << 20;
    EXPECT_NE(translate(key_from_state(s)).glsl.find("dot3 = dot(vtx_t3.xyz, psh_dotmap3(vec4(0.0)));"), std::string::npos);
    Uniforms u{};
    uniforms_from_state(s, {{{1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}}}, u);
    EXPECT_FLOAT_EQ(u.eye[0], 0.25f);
    EXPECT_FLOAT_EQ(u.eye[1], -0.5f);
    EXPECT_FLOAT_EQ(u.eye[2], 2.0f);
}

TEST(Psh, DotZWReplacesTheDepth) {
    KelvinState s = output_texture(0);
    s.r[mthd::kShaderStageProgram >> 2] = 1u | (0x11u << 5) | (0xAu << 10);
    enable_texture(s, 0, kFormat2D);
    const Translation t = translate(key_from_state(s));
    EXPECT_NE(t.glsl.find("float depth = dot1 / dot2;"), std::string::npos) << t.glsl;
    EXPECT_NE(t.glsl.find("gl_FragDepth = depth;"), std::string::npos) << t.glsl;
    expect_compiles(t, "dot ZW");
}

TEST(Psh, ADotModeOutsideItsStagesReadsAsNone) {
    // Dot ST in stage 1 has no earlier dot product to pair with; dot reflect specular exists only in stage 3.
    KelvinState s = output_texture(1);
    s.r[mthd::kShaderStageProgram >> 2] = 1u | (9u << 5) | (0xCu << 10);
    enable_texture(s, 0, kFormat2D);
    enable_texture(s, 1, kFormat2D);
    enable_texture(s, 2, kFormatCube);
    const Translation t = translate(key_from_state(s));
    EXPECT_EQ(t.samplers[1], SamplerKind::None);
    EXPECT_EQ(t.samplers[2], SamplerKind::None);
    EXPECT_EQ(t.glsl.find("dot1 ="), std::string::npos) << t.glsl;
    expect_compiles(t, "misplaced dot modes");
}
