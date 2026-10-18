#include "nv2a/shaders/psh_translate.h"

#include <format>

#include "nv2a/shaders/fsh_interface.h"

namespace mv::pgraph::psh {

namespace {

constexpr u32 kShaderClipPlaneMode = 0x17F8, kShadowCompareFunc = 0x1E6C, kFogEnable = 0x02A4, kFogColor = 0x02A8,
              kSpecularFogFactor = 0x1E20;

bool depth_format(u32 color_format) {
    switch (color_format) {
        case 0x2A: case 0x2B: case 0x2C: case 0x2D: case 0x2E: case 0x30: case 0x31: return true;
        default: return false;
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
    k.shadow_compare_func = s.reg(kShadowCompareFunc) & 7u;
    for (u32 t = 0; t < 4; ++t) {
        const TextureStage ts = s.texture(t);
        k.texture_format[t] = ts.format & 0xFFFFu;
        k.texture_enabled[t] = ts.enabled();
    }
    k.alpha_test = s.alpha_test();
    k.alpha_func = s.reg(mthd::kAlphaFunc);
    k.smooth_shading = s.reg(mthd::kShadeModel) != 0x1D00;
    k.fog_enable = s.reg(kFogEnable) & 1u;
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
        out.tex_scale[t][2] = (fmt == 0x2A || fmt == 0x2B || fmt == 0x2E) ? 16777215.0f : 65535.0f;
        out.tex_scale[t][3] = 0.0f;
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
        "};\n";
    // Texture stages: which sampler each needs.
    for (u32 i = 0; i < 4; ++i) {
        const u32 mode = (key.shader_stage_program >> (5 * i)) & 0x1Fu;
        const u32 fmt = key.texture_format[i];
        const bool cube = (fmt >> 2) & 1u;
        const u32 dim = (fmt >> 4) & 0xFu;
        const u32 color = (fmt >> 8) & 0xFFu;
        SamplerKind kind = SamplerKind::None;
        if (mode == 0 || mode == 4 || mode == 5 || !key.texture_enabled[i]) {
            kind = SamplerKind::None;
        } else if (mode == 3 || cube) {
            kind = SamplerKind::SamplerCube;
        } else if (mode == 2 && depth_format(color)) {
            kind = SamplerKind::Shadow2D;
        } else if (dim == 3) {
            kind = SamplerKind::Sampler3D;
        } else {
            kind = SamplerKind::Sampler2D;
        }
        t.samplers[i] = kind;
        const char* type = kind == SamplerKind::SamplerCube ? "samplerCube" : (kind == SamplerKind::Sampler3D ? "sampler3D" : "sampler2D");
        if (kind != SamplerKind::None) g += std::format("layout(set = 0, binding = {}) uniform {} tex{};\n", kSamplerBindingBase + i, type, i);
    }
    g += "\nlayout(location = 0) out vec4 frag_color;\n\n";
    g += "void main() {\n";
    g += "    vec4 v0 = vtx_d0;\n    vec4 v1 = vtx_d1;\n";
    g += std::format("    vec4 fog = vec4(fog_color.rgb, {});\n", key.fog_enable ? "clamp(vtx_fog, 0.0, 1.0)" : "1.0");
    for (u32 i = 0; i < 4; ++i) {
        const u32 mode = (key.shader_stage_program >> (5 * i)) & 0x1Fu;
        const SamplerKind kind = t.samplers[i];
        g += std::format("    vec4 t{} = vec4(0.0);\n", i);
        if (kind == SamplerKind::None && mode != 4 && mode != 5) continue;
        switch (mode) {
            case 1:  // 2D projective
            case 2:  // 3D projective
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
            case 3:  // cube map
                if (kind == SamplerKind::SamplerCube) g += std::format("    t{0} = texture(tex{0}, vtx_t{0}.xyz);\n", i);
                else g += std::format("    t{0} = textureProj(tex{0}, vec3(vtx_t{0}.xy * tex_scale[{0}].xy, vtx_t{0}.w));\n", i);
                break;
            case 4:  // pass through
                g += std::format("    t{0} = vtx_t{0};\n", i);
                break;
            case 5:  // clip plane
                g += std::format("    if (any(lessThan(vtx_t{0}, vec4(0.0)))) discard;\n", i);
                break;
            default:
                if (t.error.empty()) t.error = std::format("texture stage {} mode {} is not supported yet", i, mode);
                break;
        }
    }
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
        else if (func != 7) g += std::format("    if (!(final_color.a {} alpha_ref.x)) discard;\n", compare_op(func));
    }
    g += "    frag_color = clamp(final_color, 0.0, 1.0);\n";
    g += "}\n";
    t.glsl = std::move(g);
    return t;
}

}  // namespace mv::pgraph::psh
