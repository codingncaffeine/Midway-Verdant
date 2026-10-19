#include "nv2a/shaders/psh_translate.h"

#include <cstdlib>
#include <string_view>
#include <format>

#include "nv2a/shaders/fsh_interface.h"

namespace mv::pgraph::psh {

namespace {

constexpr u32 kShadowCompareFunc = 0x1E6C, kFogEnable = 0x02A4, kFogColor = 0x02A8, kSpecularFogFactor = 0x1E20;

bool depth_format(u32 color_format) {
    switch (color_format) {
        case 0x2A: case 0x2B: case 0x2C: case 0x2D: case 0x2E: case 0x2F: case 0x30: case 0x31: return true;
        default: return false;
    }
}

// The texture stage modes (SET_SHADER_STAGE_PROGRAM, five bits per stage). Their behaviour is the NV20 texture
// shader's, as the public NV_texture_shader specification describes it (each mode below names its operation there),
// and the Direct3D 8 ps.1.x texture instruction it implements.
enum : u32 {
    kModeNone = 0, kModeProject2D = 1, kModeProject3D = 2, kModeCube = 3, kModePassThrough = 4, kModeClipPlane = 5,
    kModeBumpEnv = 6,                 // OFFSET_TEXTURE_2D: texbem
    kModeBumpEnvLuminance = 7,        // OFFSET_TEXTURE_2D_SCALE: texbeml
    kModeBrdf = 8,                    // no NV_texture_shader counterpart: a placeholder
    kModeDotST = 9,                   // DOT_PRODUCT_TEXTURE_2D: texm3x2tex
    kModeDotZW = 0xA,                 // DOT_PRODUCT_DEPTH_REPLACE: texm3x2depth
    kModeDotReflectDiffuse = 0xB,     // DOT_PRODUCT_DIFFUSE_CUBE_MAP (the middle row of a 3x3 reflect)
    kModeDotReflectSpecular = 0xC,    // DOT_PRODUCT_REFLECT_CUBE_MAP: texm3x3vspec, the eye from the three q
    kModeDotSTR3D = 0xD,              // DOT_PRODUCT_TEXTURE_3D (NV_texture_shader2): texm3x3tex on a volume
    kModeDotSTRCube = 0xE,            // DOT_PRODUCT_TEXTURE_CUBE_MAP: texm3x3tex
    kModeDependentAR = 0xF,           // DEPENDENT_AR_TEXTURE_2D: texreg2ar
    kModeDependentGB = 0x10,          // DEPENDENT_GB_TEXTURE_2D: texreg2gb
    kModeDotProduct = 0x11,           // DOT_PRODUCT: texm3x2pad, texm3x3pad
    kModeDotReflectSpecularConst = 0x12  // DOT_PRODUCT_CONST_EYE_REFLECT_CUBE_MAP: texm3x3spec, a constant eye
};

u32 stage_mode(const Key& key, u32 stage) { return (key.shader_stage_program >> (5 * stage)) & 0x1Fu; }

// The stages each mode exists in (the public nxdk register header lists the values per stage): a mode that reads
// earlier stages' dot products needs that many stages before it. Elsewhere it has no meaning; it is read as none,
// as an inconsistent NV_texture_shader stage is.
bool mode_valid_at(u32 mode, u32 stage) {
    switch (mode) {
        case kModeBumpEnv: case kModeBumpEnvLuminance: case kModeDependentAR: case kModeDependentGB: return stage >= 1;
        case kModeBrdf: return stage >= 2;
        case kModeDotProduct: return stage == 1 || stage == 2;
        case kModeDotST: case kModeDotZW: return stage >= 2;
        case kModeDotReflectDiffuse: return stage == 2;
        case kModeDotReflectSpecular: case kModeDotSTR3D: case kModeDotSTRCube: case kModeDotReflectSpecularConst: return stage == 3;
        default: return true;
    }
}

bool dot_mode(u32 mode) {
    switch (mode) {
        case kModeDotST: case kModeDotZW: case kModeDotReflectDiffuse: case kModeDotReflectSpecular: case kModeDotSTR3D:
        case kModeDotSTRCube: case kModeDotProduct: case kModeDotReflectSpecularConst: return true;
        default: return false;
    }
}

// The earlier stage whose result a stage reads (SET_SHADER_OTHER_STAGE_INPUT): stage 1 always reads stage 0, stage 2
// the stage in bits 19-16, stage 3 the one in bits 23-20.
u32 source_stage(const Key& key, u32 stage) {
    if (stage == 2) return (key.other_stage_input >> 16) & 0xFu;
    if (stage == 3) return (key.other_stage_input >> 20) & 0xFu;
    return 0;
}

// The source's colour as GLSL: a stage that is not earlier than this one has no result yet, so it reads zero.
std::string source_name(const Key& key, u32 stage) {
    const u32 src = source_stage(key, stage);
    return src < stage ? std::format("t{}", src) : std::string("vec4(0.0)");
}

// This stage's dot product: its (s, t, r) with the source's colour as SET_DOT_RGBMAPPING maps it (bits 2-0 for
// stage 1, 6-4 for stage 2, 10-8 for stage 3).
std::string dot_expr(const Key& key, u32 stage) {
    const u32 mapping = stage == 0 ? 0 : (key.dot_mapping >> (4 * (stage - 1))) & 7u;
    return std::format("dot(vtx_t{}.xyz, psh_dotmap{}({}))", stage, mapping, source_name(key, stage));
}

// The dot mappings, written from the mapping names in the public register headers (zero to one; minus one to one in
// the Microsoft, OpenGL and NVIDIA conventions; HILO; HILO hemisphere in the same three conventions). The colour
// arrives as unsigned 0..1 channels in our R G B A order (the guest's ARGB names), so a signed byte is the channel's
// byte read as two's complement. HILO takes the texel's 32 bits as two 16-bit halves, A R high and G B low; the
// hemisphere forms add z = sqrt(1 - hi^2 - lo^2) as NV_texture_shader's signed HILO does. The reference emulator
// passes mappings 5-7 through unchanged; ours follow the NV form, unconfirmed on hardware [verify]. The NVIDIA signed
// 8-bit convention is byte / 127 here, as both behaviour references read it; NVIDIA's own table (NV_texture_shader,
// signed fixed point) has 127 -> 0.992, which would be byte / 128 [verify].
constexpr const char* kDotMapFunctions =
    "float psh_s8(float x) {\n"
    "    float b = x * 255.0;\n"
    "    return b >= 127.5 ? b - 256.0 : b;\n"
    "}\n"
    "vec3 psh_s8v(vec3 c) { return vec3(psh_s8(c.r), psh_s8(c.g), psh_s8(c.b)); }\n"
    "float psh_s16(float v) { return v >= 32767.5 ? v - 65536.0 : v; }\n"
    "vec2 psh_hilo(vec4 c) { return vec2(c.a * 65280.0 + c.r * 255.0, c.g * 65280.0 + c.b * 255.0); }\n"
    "vec3 psh_hemisphere(vec2 hl) { return vec3(hl, sqrt(max(0.0, 1.0 - dot(hl, hl)))); }\n"
    "vec3 psh_dotmap0(vec4 c) { return c.rgb; }\n"
    "vec3 psh_dotmap1(vec4 c) { return (c.rgb * 255.0 - 128.0) / 127.0; }\n"
    "vec3 psh_dotmap2(vec4 c) { return (2.0 * psh_s8v(c.rgb) + 1.0) / 255.0; }\n"
    "vec3 psh_dotmap3(vec4 c) { return psh_s8v(c.rgb) / 127.0; }\n"
    "vec3 psh_dotmap4(vec4 c) { return vec3(psh_hilo(c) / 65535.0, 1.0); }\n"
    "vec3 psh_dotmap5(vec4 c) { return psh_hemisphere((psh_hilo(c) - 32768.0) / 32767.0); }\n"
    "vec3 psh_dotmap6(vec4 c) {\n"
    "    vec2 h = psh_hilo(c);\n"
    "    return psh_hemisphere((2.0 * vec2(psh_s16(h.x), psh_s16(h.y)) + 1.0) / 65535.0);\n"
    "}\n"
    "vec3 psh_dotmap7(vec4 c) {\n"
    "    vec2 h = psh_hilo(c);\n"
    "    return psh_hemisphere(vec2(psh_s16(h.x), psh_s16(h.y)) / 32767.0);\n"
    "}\n";

// A two-coordinate lookup into stage i's texture: a linear texture's coordinates are in texels (tex_scale holds
// 1 / size), and a volume texture in a 2D mode takes `r` as its third coordinate [verify].
std::string lookup_2d(u32 i, SamplerKind kind, const std::string& st, const std::string& r) {
    if (kind == SamplerKind::Sampler3D) return std::format("texture(tex{}, vec3({}, {}))", i, st, r);
    return std::format("texture(tex{0}, ({1}) * tex_scale[{0}].xy)", i, st);
}

// What each stage's texture binding must receive.
SamplerKind sampler_kind(const Key& key, u32 stage) {
    const u32 mode = stage_mode(key, stage);
    const u32 fmt = key.texture_format[stage];
    const bool cube = (fmt >> 2) & 1u;
    const u32 dim = (fmt >> 4) & 0xFu;
    const u32 color = (fmt >> 8) & 0xFFu;
    if (!key.texture_enabled[stage] || !mode_valid_at(mode, stage)) return SamplerKind::None;
    switch (mode) {
        case kModeProject2D:
        case kModeProject3D:
        case kModeCube:
            if (mode == kModeCube || cube) return SamplerKind::SamplerCube;
            if (mode == kModeProject3D && depth_format(color)) return SamplerKind::Shadow2D;
            return dim == 3 ? SamplerKind::Sampler3D : SamplerKind::Sampler2D;
        // The dependent modes read the kind of texture they are made for; another kind makes the stage inconsistent
        // (NV_texture_shader), which reads as none. A 2D texture in the 3D mode is a volume one texel deep.
        case kModeBumpEnv:
        case kModeBumpEnvLuminance:
        case kModeDotST:
        case kModeDependentAR:
        case kModeDependentGB:
            if (cube) return SamplerKind::None;
            return dim == 3 ? SamplerKind::Sampler3D : SamplerKind::Sampler2D;
        case kModeDotSTR3D:
            if (cube) return SamplerKind::None;
            return dim == 3 ? SamplerKind::Sampler3D : SamplerKind::Sampler2D;
        case kModeDotReflectDiffuse:
        case kModeDotReflectSpecular:
        case kModeDotSTRCube:
        case kModeDotReflectSpecularConst: return cube ? SamplerKind::SamplerCube : SamplerKind::None;
        default: return SamplerKind::None;
    }
}

struct Input {
    u32 reg, mapping;
    bool alpha;
};

Input input_at(u32 word, u32 shift) {
    const u32 f = (word >> shift) & 0xFFu;
    return {f & 0xFu, f >> 5, ((f >> 4) & 1u) != 0};
}

std::string map(const std::string& x, u32 mapping) {
    switch (mapping) {
        case 0: return std::format("max({}, 0.0)", x);
        case 1: return std::format("(1.0 - clamp({}, 0.0, 1.0))", x);
        case 2: return std::format("(2.0 * max({}, 0.0) - 1.0)", x);
        case 3: return std::format("(1.0 - 2.0 * max({}, 0.0))", x);
        case 4: return std::format("(max({}, 0.0) - 0.5)", x);
        case 5: return std::format("(0.5 - max({}, 0.0))", x);
        case 6: return x;
        default: return std::format("(-{})", x);
    }
}

std::string op(const std::string& e, u32 o) {
    switch (o) {
        case 1: return std::format("({} - 0.5)", e);
        case 2: return std::format("({} * 2.0)", e);
        case 3: return std::format("(({} - 0.5) * 2.0)", e);
        case 4: return std::format("({} * 4.0)", e);
        case 6: return std::format("({} * 0.5)", e);
        default: return e;
    }
}

// The register a combiner input names, as a vec4 expression; stage 8 is the final combiner.
std::string reg_name(u32 reg, u32 stage, const Key& key, std::string& error) {
    const bool per_stage0 = (key.combiner_control >> 12) & 1u, per_stage1 = (key.combiner_control >> 16) & 1u;
    switch (reg) {
        case 0: return "vec4(0.0)";
        case 1: return std::format("c0[{}]", stage == 8 ? 8 : (per_stage0 ? stage : 0));
        case 2: return std::format("c1[{}]", stage == 8 ? 8 : (per_stage1 ? stage : 0));
        case 3: return "fog";
        case 4: return "v0";
        case 5: return "v1";
        case 8: case 9: case 10: case 11: return std::format("t{}", reg - 8);
        case 12: return "r0";
        case 13: return "r1";
        case 14:
            if (stage == 8) return "v1r0";
            break;
        case 15:
            if (stage == 8) return "efprod";
            break;
        default: break;
    }
    if (error.empty()) error = std::format("combiner stage {} reads register {}", stage, reg);
    return "vec4(0.0)";
}

const char* dest_name(u32 reg) {
    switch (reg) {
        case 4: return "v0";
        case 5: return "v1";
        case 8: return "t0";
        case 9: return "t1";
        case 10: return "t2";
        case 11: return "t3";
        case 12: return "r0";
        case 13: return "r1";
        default: return nullptr;  // 0 discards; the rest are not writable
    }
}

std::string rgb_input(const Input& in, u32 stage, const Key& key, std::string& error) {
    return map(reg_name(in.reg, stage, key, error) + (in.alpha ? ".aaa" : ".rgb"), in.mapping);
}

std::string alpha_input(const Input& in, u32 stage, const Key& key, std::string& error) {
    return map(reg_name(in.reg, stage, key, error) + (in.alpha ? ".a" : ".b"), in.mapping);
}

const char* compare_op(u32 func) {
    switch (func & 7u) {
        case 0: return "false";
        case 1: return "<";
        case 2: return "==";
        case 3: return "<=";
        case 4: return ">";
        case 5: return "!=";
        case 6: return ">=";
        default: return "true";
    }
}

}  // namespace

Key key_from_state(const KelvinState& s) {
    Key k;
    k.combiner_control = s.reg(mthd::kCombinerControl);
    for (u32 i = 0; i < 8; ++i) {
        k.color_icw[i] = s.reg(mthd::kCombinerColorIcw + 4 * i);
        k.color_ocw[i] = s.reg(mthd::kCombinerColorOcw + 4 * i);
        k.alpha_icw[i] = s.reg(mthd::kCombinerAlphaIcw + 4 * i);
        k.alpha_ocw[i] = s.reg(mthd::kCombinerAlphaOcw + 4 * i);
    }
    k.final_cw0 = s.reg(mthd::kCombinerSpecFogCw0);
    k.final_cw1 = s.reg(mthd::kCombinerSpecFogCw1);
    k.shader_stage_program = s.reg(mthd::kShaderStageProgram);
    k.other_stage_input = s.reg(mthd::kShaderOtherStageInput);
    k.clip_plane_mode = s.reg(mthd::kShaderClipPlaneMode) & 0xFFFFu;
    k.dot_mapping = s.reg(mthd::kDotRgbMapping) & 0x777u;
    k.shadow_compare_func = s.reg(kShadowCompareFunc) & 7u;
    // MV_RENDER_DEBUG=noshadow: every shadow-map comparison passes (the control for a shadow-map artefact).
    static const bool no_shadow = [] {
        const char* d = std::getenv("MV_RENDER_DEBUG");
        return d && std::string_view(d).find("noshadow") != std::string_view::npos;
    }();
    if (no_shadow) k.shadow_compare_func = 7;
    for (u32 t = 0; t < 4; ++t) {
        const TextureStage ts = s.texture(t);
        k.texture_format[t] = ts.format & 0xFFFFu;
        k.texture_enabled[t] = ts.enabled();
    }
    k.alpha_test = s.alpha_test();
    k.alpha_func = s.reg(mthd::kAlphaFunc);
    k.smooth_shading = s.reg(mthd::kShadeModel) != 0x1D00;
    // Instrument: MV_RENDER_DEBUG=nofog draws as if fog were off (the vertex side follows in the renderer).
    static const bool no_fog = std::getenv("MV_RENDER_DEBUG") && std::string_view(std::getenv("MV_RENDER_DEBUG")).find("nofog") != std::string_view::npos;
    k.fog_enable = !no_fog && (s.reg(kFogEnable) & 1u);
    return k;
}

void uniforms_from_state(const KelvinState& s, const std::array<std::array<float, 2>, 4>& texture_scale, Uniforms& out) {
    auto argb = [](u32 v, float* dst) {
        dst[0] = static_cast<float>((v >> 16) & 0xFFu) / 255.0f;
        dst[1] = static_cast<float>((v >> 8) & 0xFFu) / 255.0f;
        dst[2] = static_cast<float>(v & 0xFFu) / 255.0f;
        dst[3] = static_cast<float>(v >> 24) / 255.0f;
    };
    for (u32 i = 0; i < 8; ++i) {
        argb(s.reg(mthd::kCombinerFactor0 + 4 * i), out.c0[i]);
        argb(s.reg(mthd::kCombinerFactor1 + 4 * i), out.c1[i]);
    }
    argb(s.reg(kSpecularFogFactor), out.c0[8]);
    argb(s.reg(kSpecularFogFactor + 4), out.c1[8]);
    const u32 fog = s.reg(kFogColor);  // this register is A B G R from the top
    out.fog_color[0] = static_cast<float>(fog & 0xFFu) / 255.0f;
    out.fog_color[1] = static_cast<float>((fog >> 8) & 0xFFu) / 255.0f;
    out.fog_color[2] = static_cast<float>((fog >> 16) & 0xFFu) / 255.0f;
    out.fog_color[3] = static_cast<float>(fog >> 24) / 255.0f;
    out.alpha_ref[0] = static_cast<float>(s.reg(mthd::kAlphaRef) & 0xFFu) / 255.0f;
    out.alpha_ref[1] = out.alpha_ref[2] = out.alpha_ref[3] = 0.0f;
    for (u32 t = 0; t < 4; ++t) {
        out.tex_scale[t][0] = texture_scale[t][0];
        out.tex_scale[t][1] = texture_scale[t][1];
        const u32 fmt = s.texture(t).color_format();
        // The depth range a shadow compare works in: 24-bit fixed point, 16-bit fixed point, or the 16-bit float
        // format's largest value (the 24-bit float forms are read as fixed point).
        if (fmt == 0x2A || fmt == 0x2B || fmt == 0x2E || fmt == 0x2F) out.tex_scale[t][2] = 16777215.0f;
        else if (fmt == 0x2D || fmt == 0x31) out.tex_scale[t][2] = 511.9375f;
        else out.tex_scale[t][2] = 65535.0f;
        out.tex_scale[t][3] = 0.0f;
    }
    for (u32 c = 0; c < 3; ++c) out.eye[c] = s.regf(mthd::kEyeVector + 4 * c);
    out.eye[3] = 0.0f;
    // SET_TEXTURE_SET_BUMP_ENV_MAT's four words are m00, m01, m11, m10: the reference emulator stores the last two
    // crosswise, and we follow it [verify: NV_texture_shader's a1..a4 order would be m00 m01 m10 m11].
    for (u32 t = 0; t < 4; ++t) {
        const u32 b = mthd::kTextureStride * t;
        out.bump_mat[t][0] = s.regf(mthd::kBumpEnvMat + b);
        out.bump_mat[t][1] = s.regf(mthd::kBumpEnvMat + b + 4);
        out.bump_mat[t][2] = s.regf(mthd::kBumpEnvMat + b + 12);
        out.bump_mat[t][3] = s.regf(mthd::kBumpEnvMat + b + 8);
        out.bump_lum[t][0] = s.regf(mthd::kBumpEnvScale + b);
        out.bump_lum[t][1] = s.regf(mthd::kBumpEnvOffset + b);
        out.bump_lum[t][2] = out.bump_lum[t][3] = 0.0f;
    }
}

Translation translate(const Key& key) {
    Translation t;
    std::string g;
    g += "#version 450\n\n";
    const char* flat = key.smooth_shading ? "" : "flat ";
    g += std::format("layout(location = {}) {}in vec4 vtx_d0;\n", kVaryingD0, flat);
    g += std::format("layout(location = {}) {}in vec4 vtx_d1;\n", kVaryingD1, flat);
    g += std::format("layout(location = {}) {}in vec4 vtx_b0;\n", kVaryingB0, flat);
    g += std::format("layout(location = {}) {}in vec4 vtx_b1;\n", kVaryingB1, flat);
    for (u32 i = 0; i < 4; ++i) g += std::format("layout(location = {}) in vec4 vtx_t{};\n", kVaryingT0 + i, i);
    g += std::format("layout(location = {}) in float vtx_fog;\n", kVaryingFog);
    g += std::format("layout(location = {}) in vec4 vtx_pos;\n\n", kVaryingPos);
    g +=
        "layout(std140, set = 0, binding = 1) uniform PshUniforms {\n"
        "    vec4 c0[9];\n"
        "    vec4 c1[9];\n"
        "    vec4 fog_color;\n"
        "    vec4 alpha_ref;\n"
        "    vec4 tex_scale[4];\n"
        "    vec4 eye;\n"
        "    vec4 bump_mat[4];\n"
        "    vec4 bump_lum[4];\n"
        "};\n";
    // Texture stages: which sampler each needs.
    for (u32 i = 0; i < 4; ++i) {
        const SamplerKind kind = sampler_kind(key, i);
        t.samplers[i] = kind;
        const char* type = kind == SamplerKind::SamplerCube ? "samplerCube" : (kind == SamplerKind::Sampler3D ? "sampler3D" : "sampler2D");
        if (kind != SamplerKind::None) g += std::format("layout(set = 0, binding = {}) uniform {} tex{};\n", kSamplerBindingBase + i, type, i);
    }
    bool dots = false, bumps = false;
    for (u32 i = 0; i < 4; ++i) {
        const u32 mode = stage_mode(key, i);
        if (!mode_valid_at(mode, i)) continue;
        dots |= dot_mode(mode);
        bumps |= mode == kModeBumpEnv || mode == kModeBumpEnvLuminance;
    }
    if (dots || bumps) g += kDotMapFunctions;
    g += "\nlayout(location = 0) out vec4 frag_color;\n\n";
    g += "void main() {\n";
    g += "    vec4 v0 = vtx_d0;\n    vec4 v1 = vtx_d1;\n";
    g += std::format("    vec4 fog = vec4(fog_color.rgb, {});\n", key.fog_enable ? "clamp(vtx_fog, 0.0, 1.0)" : "1.0");
    // A stage without a result (mode none, a disabled texture) reads (0, 0, 0, 0), as NV_texture_shader's NONE does.
    for (u32 i = 0; i < 4; ++i) g += std::format("    vec4 t{} = vec4(0.0);\n", i);
    if (dots) g += "    float dot0 = 0.0, dot1 = 0.0, dot2 = 0.0, dot3 = 0.0;\n";
    for (u32 i = 0; i < 4; ++i) {
        const u32 mode = mode_valid_at(stage_mode(key, i), i) ? stage_mode(key, i) : kModeNone;
        const SamplerKind kind = t.samplers[i];
        // Every dot mode first takes this stage's dot product (NV_texture_shader: dotC); the lookups then read it
        // with the earlier stages' ones.
        if (dot_mode(mode)) g += std::format("    dot{} = {};\n", i, dot_expr(key, i));
        switch (mode) {
            case kModeNone: break;
            case kModeProject2D:
            case kModeProject3D:
                if (kind == SamplerKind::None) break;
                if (kind == SamplerKind::Shadow2D) {
                    // The stored depth is the left operand: t = (texel FUNC r), with r = z/w clamped to the
                    // format's depth range (raw units, like the texel).
                    g += std::format(
                        "    {{\n"
                        "        vec3 p = vtx_t{0}.xyz / vtx_t{0}.w;\n"
                        "        float texel = texture(tex{0}, p.xy * tex_scale[{0}].xy).r * tex_scale[{0}].z;\n"
                        "        t{0} = vec4({1} ? 1.0 : 0.0);\n"
                        "    }}\n",
                        i,
                        key.shadow_compare_func == 0 ? std::string("false")
                        : key.shadow_compare_func == 7 ? std::string("true")
                                                       : std::format("texel {} clamp(p.z, 0.0, tex_scale[{}].z)", compare_op(key.shadow_compare_func), i));
                } else if (kind == SamplerKind::Sampler3D) {
                    g += std::format("    t{0} = textureProj(tex{0}, vtx_t{0});\n", i);
                } else if (kind == SamplerKind::SamplerCube) {
                    g += std::format("    t{0} = texture(tex{0}, vtx_t{0}.xyz);\n", i);
                } else {
                    g += std::format("    t{0} = textureProj(tex{0}, vec3(vtx_t{0}.xy * tex_scale[{0}].xy, vtx_t{0}.w));\n", i);
                }
                break;
            case kModeCube:
                if (kind == SamplerKind::None) break;
                if (kind == SamplerKind::SamplerCube) g += std::format("    t{0} = texture(tex{0}, vtx_t{0}.xyz);\n", i);
                else g += std::format("    t{0} = textureProj(tex{0}, vec3(vtx_t{0}.xy * tex_scale[{0}].xy, vtx_t{0}.w));\n", i);
                break;
            case kModePassThrough:
                // The texture coordinates become the colour, each clamped to [0, 1] (NV_texture_shader PASS_THROUGH;
                // D3D's texcoord clamps the same way).
                g += std::format("    t{0} = clamp(vtx_t{0}, vec4(0.0), vec4(1.0));\n", i);
                break;
            case kModeClipPlane: {
                // Each of s, t, r, q is tested against zero and any failing test discards the fragment (NV_texture_shader
                // CULL_FRAGMENT). SET_SHADER_CLIP_PLANE_MODE bit 4 * stage + component picks the test: 0 passes
                // when the component is >= 0 (discards below zero), 1 passes when it is < 0 (discards at or above).
                const u32 sense = (key.clip_plane_mode >> (4 * i)) & 0xFu;
                if (sense == 0) {
                    g += std::format("    if (any(lessThan(vtx_t{0}, vec4(0.0)))) discard;\n", i);
                } else {
                    g += std::format("    if (any(notEqual(lessThan(vtx_t{0}, vec4(0.0)), bvec4({1}, {2}, {3}, {4})))) discard;\n", i,
                                     (sense & 1u) ? "true" : "false", (sense & 2u) ? "true" : "false", (sense & 4u) ? "true" : "false",
                                     (sense & 8u) ? "true" : "false");
                }
                break;
            }
            case kModeBumpEnv:
            case kModeBumpEnvLuminance:
                // The source's blue and green bytes, as two's complement / 127, are (du, dv) [verify -128: NV's own
                // signed table would read byte / 128]; the stage's 2x2 matrix turns them into an offset of its (s, t),
                // without a divide by q: s' = s + m00 du + m10 dv, t' = t + m01 du + m11 dv (NV_texture_shader
                // OFFSET_TEXTURE_2D, D3D texbem). Blue and green are the bump texel's first and second bytes (a V8U8
                // texel is the hardware's G8B8). Mode 7 then scales red, green and blue (not alpha) by the source's
                // red, unsigned, times the luminance scale plus the offset, clamped to [0, 1]: NV_texture_shader
                // OFFSET_TEXTURE_2D_SCALE. The reference emulator scales all four channels unclamped, and D3D's
                // texbeml text has RGBA [verify]. A volume texture takes r as its third coordinate [verify].
                if (kind != SamplerKind::None) {
                    const std::string src = source_name(key, i);
                    g += std::format(
                        "    {{\n"
                        "        vec2 dudv = vec2(psh_s8({1}.b), psh_s8({1}.g)) / 127.0;\n"
                        "        vec2 st = vtx_t{0}.xy + vec2(dot(bump_mat[{0}].xz, dudv), dot(bump_mat[{0}].yw, dudv));\n"
                        "        t{0} = {2};\n",
                        i, src, lookup_2d(i, kind, "st", std::format("vtx_t{}.z", i)));
                    if (mode == kModeBumpEnvLuminance)
                        g += std::format("        t{0}.rgb *= clamp(bump_lum[{0}].x * {1}.r + bump_lum[{0}].y, 0.0, 1.0);\n", i, src);
                    g += "    }\n";
                }
                break;
            case kModeBrdf:
                // BRDF has no public description beyond its name and the mode is left at (0, 0, 0, 0), as the
                // reference emulator leaves it: a documented placeholder [verify].
                break;
            case kModeDotProduct:
                // A dot product alone: its result feeds the stages after it and its colour is (0, 0, 0, 0)
                // (NV_texture_shader DOT_PRODUCT; D3D's texm3x2pad / texm3x3pad results are not readable).
                break;
            case kModeDotST:
                // (s, t) = (the previous stage's dot, this stage's dot): texm3x2tex.
                if (kind != SamplerKind::None) g += std::format("    t{0} = {1};\n", i, lookup_2d(i, kind, std::format("vec2(dot{}, dot{})", i - 1, i), "0.0"));
                break;
            case kModeDotZW:
                // The fragment's depth becomes the previous stage's dot over this stage's one, in the [0, 1] window
                // range; a depth outside it rejects the fragment (NV_texture_shader DOT_PRODUCT_DEPTH_REPLACE; D3D's
                // texm3x2depth text instead clamps it and makes a zero divisor 1.0 [verify]). The colour is zero.
                g += std::format(
                    "    {{\n"
                    "        float depth = dot{0} / dot{1};\n"
                    "        if (!(depth >= 0.0 && depth <= 1.0)) discard;\n"
                    "        gl_FragDepth = depth;\n"
                    "    }}\n",
                    i - 1, i);
                break;
            case kModeDotReflectDiffuse:
                // Stage 2 of a 3x3 reflection looks the normal itself up in its cube map: (dot1, dot2, dot3), with
                // stage 3's dot taken here from stage 3's coordinates, source and mapping (NV_texture_shader
                // DOT_PRODUCT_DIFFUSE_CUBE_MAP).
                g += std::format("    dot3 = {};\n", dot_expr(key, 3));
                if (kind != SamplerKind::None) g += "    t2 = texture(tex2, vec3(dot1, dot2, dot3));\n";
                break;
            case kModeDotReflectSpecular:
            case kModeDotReflectSpecularConst:
                // The normal N = (dot1, dot2, dot3) reflects the eye vector E: R = 2 N (N.E) / (N.N) - E, looked up in
                // the cube map. E is the q coordinates of stages 1-3 in mode 0xC (texm3x3vspec) and SET_EYE_VECTOR
                // in mode 0x12 (texm3x3spec's constant eye; NV_texture_shader's CONST_EYE) [verify: the register].
                if (kind != SamplerKind::None) {
                    g += std::format(
                        "    {{\n"
                        "        vec3 n = vec3(dot1, dot2, dot3);\n"
                        "        vec3 e = {};\n"
                        "        t3 = texture(tex3, 2.0 * dot(n, e) / dot(n, n) * n - e);\n"
                        "    }}\n",
                        mode == kModeDotReflectSpecular ? "vec3(vtx_t1.w, vtx_t2.w, vtx_t3.w)" : "eye.xyz");
                }
                break;
            case kModeDotSTR3D:
                // (s, t, r) = (dot1, dot2, dot3) into a volume texture: texm3x3tex.
                if (kind == SamplerKind::Sampler3D) g += "    t3 = texture(tex3, vec3(dot1, dot2, dot3));\n";
                else if (kind != SamplerKind::None) g += std::format("    t3 = {};\n", lookup_2d(3, kind, "vec2(dot1, dot2)", "dot3"));
                break;
            case kModeDotSTRCube:
                // (dot1, dot2, dot3) as a cube-map direction: texm3x3tex.
                if (kind != SamplerKind::None) g += "    t3 = texture(tex3, vec3(dot1, dot2, dot3));\n";
                break;
            case kModeDependentAR:
            case kModeDependentGB:
                // The source's colour is the coordinate: (alpha, red) for texreg2ar, (green, blue) for texreg2gb, in
                // the guest's ARGB channel names, which our channels carry.
                if (kind != SamplerKind::None) {
                    const std::string src = source_name(key, i);
                    g += std::format("    t{} = {};\n", i,
                                     lookup_2d(i, kind, mode == kModeDependentAR ? std::format("{0}.ar", src) : std::format("{0}.gb", src), "0.0"));
                }
                break;
            default:
                if (t.error.empty()) t.error = std::format("texture stage {} mode {} is reserved", i, mode);
                break;
        }
    }
    // MV_RENDER_DEBUG=white:N / grey:N (a bit per stage, 1-15): those stages' results replaced by white (mid grey),
    // to find which stage carries an artefact.
    {
        const auto mask = [](std::string_view name) {
            const char* d = std::getenv("MV_RENDER_DEBUG");
            const std::string_view v = d ? d : "";
            const auto at = v.find(name);
            return at == std::string_view::npos ? 0u : static_cast<u32>(std::strtoul(v.data() + at + name.size(), nullptr, 10));
        };
        static const u32 white = mask("white:"), grey = mask("grey:");
        for (u32 i = 0; i < 4; ++i) {
            if (white & (1u << i)) g += std::format("    t{} = vec4(1.0);\n", i);
            else if (grey & (1u << i)) g += std::format("    t{} = vec4(0.5);\n", i);
        }
        // showa:N / showc:N (a stage 0-3): the output is that stage's sampled alpha (colour), opaque.
        const auto stage = [](std::string_view name) {
            const char* d = std::getenv("MV_RENDER_DEBUG");
            const std::string_view v = d ? d : "";
            const auto at = v.find(name);
            return at == std::string_view::npos ? -1 : static_cast<int>(std::strtoul(v.data() + at + name.size(), nullptr, 10) & 3u);
        };
        static const int show_a = stage("showa:"), show_c = stage("showc:");
        if (show_a >= 0) g += std::format("    vec4 debug_show = vec4(t{}.aaa, 1.0);\n", show_a);
        else if (show_c >= 0) g += std::format("    vec4 debug_show = vec4(t{}.rgb, 1.0);\n", show_c);
    }
    // The combiners' r0 starts with t0's alpha (NV_register_combiners: spare0 alpha = texture 0 alpha). With stage 0
    // set to none that is 0 here; the reference emulator uses 1 there, which no public source settles [verify].
    g += "    vec4 r0 = vec4(0.0, 0.0, 0.0, t0.a);\n    vec4 r1 = vec4(0.0);\n";

    const u32 stages = key.combiner_control & 0xFFu;
    const bool mux_msb = (key.combiner_control >> 8) & 1u;
    const std::string mux_sel = mux_msb ? "(r0.a >= 0.5)" : "((int(r0.a * 255.0 + 0.5) & 1) == 1)";
    for (u32 s = 0; s < stages && s < 8; ++s) {
        const u32 icw = key.color_icw[s], ocw = key.color_ocw[s], aicw = key.alpha_icw[s], aocw = key.alpha_ocw[s];
        g += std::format("    {{  // combiner stage {}\n", s);
        g += std::format("        vec3 a = {};\n", rgb_input(input_at(icw, 24), s, key, t.error));
        g += std::format("        vec3 b = {};\n", rgb_input(input_at(icw, 16), s, key, t.error));
        g += std::format("        vec3 c = {};\n", rgb_input(input_at(icw, 8), s, key, t.error));
        g += std::format("        vec3 d = {};\n", rgb_input(input_at(icw, 0), s, key, t.error));
        g += std::format("        vec3 ab = {};\n", ((ocw >> 13) & 1u) ? "vec3(dot(a, b))" : "a * b");
        g += std::format("        vec3 cd = {};\n", ((ocw >> 12) & 1u) ? "vec3(dot(c, d))" : "c * d");
        g += std::format("        vec3 sum = {};\n", ((ocw >> 14) & 1u) ? std::format("({} ? cd : ab)", mux_sel) : "ab + cd");
        const u32 cop = (ocw >> 15) & 7u;
        g += std::format("        ab = clamp({}, -1.0, 1.0);\n", op("ab", cop));
        g += std::format("        cd = clamp({}, -1.0, 1.0);\n", op("cd", cop));
        g += std::format("        sum = clamp({}, -1.0, 1.0);\n", op("sum", cop));
        g += std::format("        float aa = {};\n", alpha_input(input_at(aicw, 24), s, key, t.error));
        g += std::format("        float ba = {};\n", alpha_input(input_at(aicw, 16), s, key, t.error));
        g += std::format("        float ca = {};\n", alpha_input(input_at(aicw, 8), s, key, t.error));
        g += std::format("        float da = {};\n", alpha_input(input_at(aicw, 0), s, key, t.error));
        g += "        float ab_a = aa * ba;\n        float cd_a = ca * da;\n";
        g += std::format("        float sum_a = {};\n", ((aocw >> 14) & 1u) ? std::format("({} ? cd_a : ab_a)", mux_sel) : "ab_a + cd_a");
        const u32 aop = (aocw >> 15) & 7u;
        g += std::format("        ab_a = clamp({}, -1.0, 1.0);\n", op("ab_a", aop));
        g += std::format("        cd_a = clamp({}, -1.0, 1.0);\n", op("cd_a", aop));
        g += std::format("        sum_a = clamp({}, -1.0, 1.0);\n", op("sum_a", aop));
        // Every read is done; now the writes.
        if (const char* n = dest_name((ocw >> 4) & 0xFu)) {
            g += std::format("        {}.rgb = ab;\n", n);
            if ((ocw >> 19) & 1u) g += std::format("        {}.a = ab.b;\n", n);
        }
        if (const char* n = dest_name(ocw & 0xFu)) {
            g += std::format("        {}.rgb = cd;\n", n);
            if ((ocw >> 18) & 1u) g += std::format("        {}.a = cd.b;\n", n);
        }
        if (const char* n = dest_name((ocw >> 8) & 0xFu)) g += std::format("        {}.rgb = sum;\n", n);
        if (const char* n = dest_name((aocw >> 4) & 0xFu)) g += std::format("        {}.a = ab_a;\n", n);
        if (const char* n = dest_name(aocw & 0xFu)) g += std::format("        {}.a = cd_a;\n", n);
        if (const char* n = dest_name((aocw >> 8) & 0xFu)) g += std::format("        {}.a = sum_a;\n", n);
        g += "    }\n";
    }

    // The final combiner: out.rgb = A * B + (1 - A) * C + D, out.a = G; E * F and v1 + r0 are extra sources.
    {
        const u32 cw0 = key.final_cw0, cw1 = key.final_cw1;
        const bool clamp_sum = (cw1 >> 7) & 1u, inv_v1 = (cw1 >> 6) & 1u, inv_r0 = (cw1 >> 5) & 1u;
        g += "    vec4 final_color;\n    {\n";
        g += std::format("        vec4 v1r0 = vec4({}{} + {}{}, 0.0);\n", inv_v1 ? "1.0 - " : "", "v1.rgb", inv_r0 ? "1.0 - " : "", "r0.rgb");
        if (clamp_sum) g += "        v1r0 = clamp(v1r0, 0.0, 1.0);\n";
        g += std::format("        vec3 e = {};\n", rgb_input(input_at(cw1, 24), 8, key, t.error));
        g += std::format("        vec3 f = {};\n", rgb_input(input_at(cw1, 16), 8, key, t.error));
        g += "        vec4 efprod = vec4(e * f, 0.0);\n";
        g += std::format("        vec3 fa = {};\n", rgb_input(input_at(cw0, 24), 8, key, t.error));
        g += std::format("        vec3 fb = {};\n", rgb_input(input_at(cw0, 16), 8, key, t.error));
        g += std::format("        vec3 fc = {};\n", rgb_input(input_at(cw0, 8), 8, key, t.error));
        g += std::format("        vec3 fd = {};\n", rgb_input(input_at(cw0, 0), 8, key, t.error));
        g += std::format("        float fg = {};\n", alpha_input(input_at(cw1, 8), 8, key, t.error));
        g += "        final_color = vec4(fa * fb + (1.0 - fa) * fc + fd, fg);\n    }\n";
    }
    if (key.alpha_test) {
        const u32 func = key.alpha_func & 7u;
        if (func == 0) g += "    discard;\n";
        // The test compares the alpha rounded to 8 bits with the 8-bit reference, so EQUAL and NOTEQUAL work on
        // the values the guest wrote, not on floats that differ below an eighth bit.
        else if (func != 7)
            g += std::format("    if (!(round(clamp(final_color.a, 0.0, 1.0) * 255.0) {} round(alpha_ref.x * 255.0))) discard;\n", compare_op(func));
    }
    g += "    frag_color = clamp(final_color, 0.0, 1.0);\n";
    if (g.find("vec4 debug_show") != std::string::npos) g += "    frag_color = debug_show;\n";
    g += "}\n";
    t.glsl = std::move(g);
    return t;
}

}  // namespace mv::pgraph::psh
