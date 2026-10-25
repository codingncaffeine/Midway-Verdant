#include "nv2a/shaders/vsh_translate.h"

#include <format>

#include "nv2a/shaders/fsh_interface.h"

namespace mv::pgraph::vsh {

namespace {

u32 field(const std::array<u32, 4>& t, u32 word, u32 start, u32 bits) { return (t[word] >> start) & ((1u << bits) - 1); }

Operand operand_a(const std::array<u32, 4>& t) {
    Operand o;
    o.source = static_cast<Source>(field(t, 2, 26, 2));
    o.negate = field(t, 1, 8, 1);
    o.swizzle = {static_cast<u8>(field(t, 1, 6, 2)), static_cast<u8>(field(t, 1, 4, 2)), static_cast<u8>(field(t, 1, 2, 2)),
                 static_cast<u8>(field(t, 1, 0, 2))};
    o.index = field(t, 2, 28, 4);
    return o;
}

Operand operand_b(const std::array<u32, 4>& t) {
    Operand o;
    o.source = static_cast<Source>(field(t, 2, 11, 2));
    o.negate = field(t, 2, 25, 1);
    o.swizzle = {static_cast<u8>(field(t, 2, 23, 2)), static_cast<u8>(field(t, 2, 21, 2)), static_cast<u8>(field(t, 2, 19, 2)),
                 static_cast<u8>(field(t, 2, 17, 2))};
    o.index = field(t, 2, 13, 4);
    return o;
}

Operand operand_c(const std::array<u32, 4>& t) {
    Operand o;
    o.source = static_cast<Source>(field(t, 3, 28, 2));
    o.negate = field(t, 2, 10, 1);
    o.swizzle = {static_cast<u8>(field(t, 2, 8, 2)), static_cast<u8>(field(t, 2, 6, 2)), static_cast<u8>(field(t, 2, 4, 2)),
                 static_cast<u8>(field(t, 2, 2, 2))};
    o.index = (field(t, 2, 0, 2) << 2) | field(t, 3, 30, 2);
    return o;
}

// The shared index fields: one input attribute number and one constant slot serve all three operands.
void resolve_shared(Operand& o, const std::array<u32, 4>& t, bool relative) {
    if (o.source == Source::Input) o.index = field(t, 1, 9, 4);
    if (o.source == Source::Constant) {
        o.index = field(t, 1, 13, 8);
        o.relative = relative;
    }
}

const char* mac_name(Mac m) {
    switch (m) {
        case Mac::Nop: return "nop";
        case Mac::Mov: return "mov";
        case Mac::Mul: return "mul";
        case Mac::Add: return "add";
        case Mac::Mad: return "mad";
        case Mac::Dp3: return "dp3";
        case Mac::Dph: return "dph";
        case Mac::Dp4: return "dp4";
        case Mac::Dst: return "dst";
        case Mac::Min: return "min";
        case Mac::Max: return "max";
        case Mac::Slt: return "slt";
        case Mac::Sge: return "sge";
        case Mac::Arl: return "arl";
    }
    return "?";
}

const char* ilu_name(Ilu i) {
    switch (i) {
        case Ilu::Nop: return "nop";
        case Ilu::Mov: return "mov";
        case Ilu::Rcp: return "rcp";
        case Ilu::Rcc: return "rcc";
        case Ilu::Rsq: return "rsq";
        case Ilu::Exp: return "exp";
        case Ilu::Log: return "log";
        case Ilu::Lit: return "lit";
    }
    return "?";
}

unsigned mac_operands(Mac m) {
    switch (m) {
        case Mac::Nop: return 0;
        case Mac::Mov: case Mac::Arl: return 1;
        case Mac::Add: return 2;  // A and C
        case Mac::Mad: return 3;
        default: return 2;        // A and B
    }
}

const char* output_name(u32 index) {
    switch (index) {
        case 0: return "oPos";
        case 3: return "oD0";
        case 4: return "oD1";
        case 5: return "oFog";
        case 6: return "oPts";
        case 7: return "oB0";
        case 8: return "oB1";
        case 9: return "oT0";
        case 10: return "oT1";
        case 11: return "oT2";
        case 12: return "oT3";
        default: return nullptr;
    }
}

std::string swizzle_string(const Operand& o) {
    static constexpr char kNames[4] = {'x', 'y', 'z', 'w'};
    if (o.swizzle == std::array<u8, 4>{0, 1, 2, 3}) return "";
    std::string s = ".";
    if (o.swizzle[0] == o.swizzle[1] && o.swizzle[1] == o.swizzle[2] && o.swizzle[2] == o.swizzle[3]) return s + kNames[o.swizzle[0] & 3];
    for (u8 c : o.swizzle) s += kNames[c & 3];
    return s;
}

std::string operand_string(const Operand& o) {
    std::string s = o.negate ? "-" : "";
    switch (o.source) {
        case Source::Temp: s += std::format("R{}", o.index); break;
        case Source::Input: s += std::format("v{}", o.index); break;
        case Source::Constant: s += o.relative ? std::format("c[A0+{}]", o.index) : std::format("c[{}]", o.index); break;
        case Source::None: s += "?"; break;
    }
    return s + swizzle_string(o);
}

// GLSL expression for an operand, always a vec4.
std::string operand_glsl(const Operand& o) {
    std::string base;
    switch (o.source) {
        case Source::Temp: base = o.index == 12 ? "oPos" : std::format("R{}", o.index); break;
        case Source::Input: base = std::format("v{}", o.index); break;
        case Source::Constant:
            base = o.relative ? std::format("c[clamp(A0 + {}, 0, {})]", o.index, kConstantSlots - 1) : std::format("c[{}]", o.index);
            break;
        case Source::None: base = "vec4(0.0)"; break;
    }
    // GLSL needs all four components here: a one-letter swizzle would make a scalar.
    std::string sw;
    if (o.swizzle != std::array<u8, 4>{0, 1, 2, 3}) {
        static constexpr char kNames[4] = {'x', 'y', 'z', 'w'};
        sw = ".";
        for (u8 c : o.swizzle) sw += kNames[c & 3];
    }
    std::string s = sw.empty() ? base : "(" + base + ")" + sw;
    return o.negate ? "(-" + s + ")" : s;
}

}  // namespace

std::string mask_string(u32 mask) {
    std::string s;
    if (mask & 8) s += 'x';
    if (mask & 4) s += 'y';
    if (mask & 2) s += 'z';
    if (mask & 1) s += 'w';
    return s;
}

Instruction decode(const std::array<u32, 4>& t) {
    Instruction i;
    i.ilu = static_cast<Ilu>(field(t, 1, 25, 3));
    i.mac = static_cast<Mac>(field(t, 1, 21, 4));
    const bool relative = field(t, 3, 1, 1);
    i.a = operand_a(t);
    i.b = operand_b(t);
    i.c = operand_c(t);
    resolve_shared(i.a, t, relative);
    resolve_shared(i.b, t, relative);
    resolve_shared(i.c, t, relative);
    i.mac_mask = field(t, 3, 24, 4);
    i.temp = field(t, 3, 20, 4);
    i.ilu_mask = field(t, 3, 16, 4);
    i.output_mask = field(t, 3, 12, 4);
    i.output_is_constant = field(t, 3, 11, 1) == 0;  // bit set = an output register, clear = a constant slot
    i.output_index = field(t, 3, 3, 8);
    i.output_from_ilu = field(t, 3, 2, 1);
    i.final = field(t, 3, 0, 1);
    return i;
}

std::string disassemble(const Instruction& i) {
    std::string s;
    auto dest = [&](u32 mask, u32 temp) { return std::format("R{}.{}", temp, mask_string(mask)); };
    auto output = [&]() {
        const char* n = i.output_is_constant ? nullptr : output_name(i.output_index & 0xF);
        return std::format("{}.{}", i.output_is_constant ? std::format("c[{}]", i.output_index) : (n ? n : std::format("o{}", i.output_index)),
                           mask_string(i.output_mask));
    };
    const bool paired = i.mac != Mac::Nop && i.ilu != Ilu::Nop;
    if (i.mac != Mac::Nop) {
        s += mac_name(i.mac);
        std::string dests;
        if (i.mac == Mac::Arl) {
            dests = " A0.x";
        } else {
            if (i.mac_mask) dests += " " + dest(i.mac_mask, i.temp);
            if (!i.output_from_ilu && i.output_mask) dests += (dests.empty() ? " " : " + ") + output();
        }
        s += dests;
        const unsigned n = mac_operands(i.mac);
        if (n >= 1) s += ", " + operand_string(i.a);
        if (i.mac == Mac::Add) s += ", " + operand_string(i.c);
        else if (n >= 2) s += ", " + operand_string(i.b);
        if (n >= 3) s += ", " + operand_string(i.c);
    }
    if (i.ilu != Ilu::Nop) {
        if (!s.empty()) s += " | ";
        s += ilu_name(i.ilu);
        std::string dests;
        if (i.ilu_mask) dests += " " + dest(i.ilu_mask, paired ? 1 : i.temp);
        if (i.output_from_ilu && i.output_mask) dests += (dests.empty() ? " " : " + ") + output();
        s += dests + ", " + operand_string(i.c);
    }
    if (i.final) s += "  ; final";
    return s;
}

Program extract_program(const KelvinState& state) {
    Program p;
    p.start = state.program_start();
    for (u32 slot = p.start; slot < kProgramSlots; ++slot) {
        p.instructions.push_back(decode(state.program[slot]));
        if (p.instructions.back().final) return p;
    }
    p.truncated = true;
    return p;
}

namespace {

// The vertex shader around the body: inputs, the uniform block, outputs, the hardware's arithmetic helpers and
// main()'s registers (head); the mapping of the screen-space oPos back to clip space and the varyings (tail).
void append_head(std::string& g) {
    g += "#version 450\n\n";
    for (u32 i = 0; i < kAttributes; ++i) g += std::format("layout(location = {}) in vec4 v{};\n", i, i);
    g += std::format(
        "\nlayout(std140, set = 0, binding = 0) uniform VshUniforms {{\n"
        "    vec4 c[{}];\n"
        "    vec4 surface_size;       // width, height of the render target in pixels; z: image pixels per pixel (the internal resolution)\n"
        "    vec4 clip_range;         // z min, z max (depth units)\n"
        "    vec4 ff_emission_alpha;  // fixed function: material emission rgb, material alpha\n"
        "    vec4 ff_scene_ambient;   // fixed function: scene ambient rgb\n"
        "    vec4 ff_specular;        // fixed function: the specular power curve (L, M, N) of SET_SPECULAR_PARAMS\n"
        "    vec4 fog_params;         // SET_FOG_PARAMS (the factor = f(distance) coefficients)\n"
        "    vec4 ff_light[72];       // fixed function: nine per light (see vsh_translate.h)\n"
        "}};\n\n",
        kConstantSlots);
    g += std::format("layout(location = {}) out vec4 vtx_d0;\n", kVaryingD0);
    g += std::format("layout(location = {}) out vec4 vtx_d1;\n", kVaryingD1);
    g += std::format("layout(location = {}) out vec4 vtx_b0;\n", kVaryingB0);
    g += std::format("layout(location = {}) out vec4 vtx_b1;\n", kVaryingB1);
    for (u32 i = 0; i < 4; ++i) g += std::format("layout(location = {}) out vec4 vtx_t{};\n", kVaryingT0 + i, i);
    g += std::format("layout(location = {}) out float vtx_fog;\n", kVaryingFog);
    g += std::format("layout(location = {}) out vec4 vtx_pos;\n\n", kVaryingPos);
    g +=
        "// The hardware's rules: a product with a zero factor is zero whatever the other factor is; the reciprocal\n"
        "// ops take a scalar and replicate it; RCC clamps the magnitude to [2^-64, 2^64]; LIT clamps the exponent.\n"
        "vec4 vp_mul(vec4 a, vec4 b) {\n"
        "    vec4 r = a * b;\n"
        "    bvec4 z = bvec4(equal(a, vec4(0.0)).x || equal(b, vec4(0.0)).x, equal(a, vec4(0.0)).y || equal(b, vec4(0.0)).y,\n"
        "                    equal(a, vec4(0.0)).z || equal(b, vec4(0.0)).z, equal(a, vec4(0.0)).w || equal(b, vec4(0.0)).w);\n"
        "    return mix(r, vec4(0.0), z);\n"
        "}\n"
        "vec4 vp_dp3(vec4 a, vec4 b) { return vec4(dot(a.xyz, b.xyz)); }\n"
        "vec4 vp_dph(vec4 a, vec4 b) { return vec4(dot(vec4(a.xyz, 1.0), b)); }\n"
        "vec4 vp_dp4(vec4 a, vec4 b) { return vec4(dot(a, b)); }\n"
        "vec4 vp_dst(vec4 a, vec4 b) { return vec4(1.0, a.y * b.y, a.z, b.w); }\n"
        "vec4 vp_slt(vec4 a, vec4 b) { return vec4(lessThan(a, b)); }\n"
        "vec4 vp_sge(vec4 a, vec4 b) { return vec4(greaterThanEqual(a, b)); }\n"
        "int vp_arl(vec4 a) { return int(floor(a.x + 0.001)); }\n"
        "vec4 vp_rcp(vec4 a) { return vec4(1.0 / a.x); }\n"
        "float vp_clamp_away(float t) {\n"
        "    if (t > 0.0 || floatBitsToUint(t) == 0u) return clamp(t, uintBitsToFloat(0x1F800000u), uintBitsToFloat(0x5F800000u));\n"
        "    return clamp(t, uintBitsToFloat(0xDF800000u), uintBitsToFloat(0x9F800000u));\n"
        "}\n"
        "vec4 vp_rcc(vec4 a) { return vec4(vp_clamp_away(1.0 / a.x)); }\n"
        "vec4 vp_rsq(vec4 a) {\n"
        "    if (a.x == 0.0) return vec4(uintBitsToFloat(0x7F800000u));\n"
        "    if (isinf(a.x)) return vec4(0.0);\n"
        "    return vec4(inversesqrt(abs(a.x)));\n"
        "}\n"
        "vec4 vp_exp(vec4 a) { float f = floor(a.x); return vec4(exp2(f), a.x - f, exp2(a.x), 1.0); }\n"
        "vec4 vp_log(vec4 a) {\n"
        "    float t = abs(a.x);\n"
        "    if (t == 0.0) return vec4(uintBitsToFloat(0xFF800000u), 1.0, uintBitsToFloat(0xFF800000u), 1.0);\n"
        "    float f = floor(log2(t));\n"
        "    return vec4(f, t / exp2(f), log2(t), 1.0);\n"
        "}\n"
        "vec4 vp_lit(vec4 s) {\n"
        "    float e = 1.0 / 256.0;\n"
        "    s.w = clamp(s.w, -(128.0 - e), 128.0 - e);\n"
        "    s.x = max(s.x, 0.0);\n"
        "    s.y = max(s.y, 0.0);\n"
        "    return vec4(1.0, s.x, (s.x > 0.0) ? exp2(s.w * log2(s.y)) : 0.0, 1.0);\n"
        "}\n"
        "// A vector through the 4x4 matrix in constant slots s..s+3 (one dot product per row: dp4 x4).\n"
        "vec4 xf_mul(vec4 v, int s) { return vec4(dot(v, c[s]), dot(v, c[s + 1]), dot(v, c[s + 2]), dot(v, c[s + 3])); }\n\n";
    g += "void main() {\n";
    for (u32 r = 0; r < 12; ++r) g += std::format("    vec4 R{} = vec4(0.0);\n", r);
    g += "    int A0 = 0;\n";
    g += "    vec4 oPos = vec4(0.0, 0.0, 0.0, 1.0), oD0 = vec4(0.0, 0.0, 0.0, 1.0), oD1 = vec4(0.0, 0.0, 0.0, 1.0);\n";
    g += "    vec4 oB0 = vec4(0.0, 0.0, 0.0, 1.0), oB1 = vec4(0.0, 0.0, 0.0, 1.0), oFog = vec4(0.0, 0.0, 0.0, 1.0);\n";
    g += "    vec4 oPts = vec4(0.0, 0.0, 0.0, 1.0), oT0 = vec4(0.0, 0.0, 0.0, 1.0), oT1 = vec4(0.0, 0.0, 0.0, 1.0);\n";
    g += "    vec4 oT2 = vec4(0.0, 0.0, 0.0, 1.0), oT3 = vec4(0.0, 0.0, 0.0, 1.0);\n";
    g += "    vec4 mac, ilu;\n\n";
}

void append_tail(std::string& g, const Options& options) {
    if (options.fog) {
        // The transform turns the fog distance it left in oFog.x into the fog factor (1 clear, 0 fogged) per
        // SET_FOG_MODE, with SET_FOG_PARAMS holding the curve's coefficients (the reference emulator's reading).
        const u32 m = options.fog_mode;
        const bool linear = m == 0x2601 || m == 0x804, exp1 = m == 0x800 || m == 0x802, abs_mode = m == 0x802 || m == 0x803 || m == 0x804;
        g += "\n    float fogDistance = oFog.x;\n";
        if (linear) g += "    float fogFactor = fog_params.x + fogDistance * fog_params.y - 1.0;\n";
        else if (exp1) g += "    float fogFactor = fog_params.x + exp2(fogDistance * fog_params.y * 16.0) - 1.5;\n";
        else g += "    float fogFactor = fog_params.x + exp2(-fogDistance * fogDistance * fog_params.y * fog_params.y * 32.0) - 1.5;\n";
        if (abs_mode) g += "    fogFactor = abs(fogFactor);\n";
        const char* at_infinity = (linear || m == 0x800) ? "1.0" : "0.0";
        g += std::format("    oFog = isinf(fogDistance) ? vec4({0}) : (isnan(fogFactor) ? vec4({1}) : vec4(clamp(fogFactor, -3.4e38, 3.4e38)));\n", at_infinity,
                         (linear || m == 0x800) ? "1.0" : "0.0");
    }
    g +=
        "\n    // The program leaves oPos in screen space (x, y in pixels and z in depth units, all already divided by w,\n"
        "    // w kept for perspective-correct interpolation); the host rasteriser wants clip space, so undo that.\n"
        "    // Snapped to the hardware's 1/16 pixel, in the image's pixels (finer at an internal resolution above 1x).\n"
        "    oPos.xy = trunc(oPos.xy * (16.0 * surface_size.z)) / (16.0 * surface_size.z);\n"
        "    oPos.w = vp_clamp_away(oPos.w);\n"
        "    vtx_pos = oPos;\n"
        "    vec2 ndc = (2.0 * oPos.xy - surface_size.xy) / surface_size.xy;\n"
        "    float zndc = oPos.z / clip_range.y;\n"
        "    // Multiplied back by the SIGNED w: a triangle with vertices behind the eye is then clipped at w = 0 as the\n"
        "    // hardware does; with |w| their mirrored positions drew large wedges over the picture.\n"
        "    float wa = oPos.w;\n"
        "    gl_Position = vec4(ndc * wa, zndc * wa, wa);\n"
        "    gl_PointSize = max(oPts.x, 1.0) * surface_size.z;\n"
        "    vtx_d0 = clamp(oD0, 0.0, 1.0);\n"
        "    vtx_d1 = clamp(oD1, 0.0, 1.0);\n"
        "    vtx_b0 = clamp(oB0, 0.0, 1.0);\n"
        "    vtx_b1 = clamp(oB1, 0.0, 1.0);\n"
        "    vtx_t0 = oT0;\n"
        "    vtx_t1 = oT1;\n"
        "    vtx_t2 = oT2;\n"
        "    vtx_t3 = oT3;\n"
        "    vtx_fog = oFog.x;\n"
        "}\n";
}

}  // namespace

Translation translate(const Program& program, const Options& options) {
    Translation t;
    std::string g;
    append_head(g);

    for (std::size_t n = 0; n < program.instructions.size(); ++n) {
        const Instruction& i = program.instructions[n];
        t.disassembly.push_back(disassemble(i));
        for (const Operand* o : {&i.a, &i.b, &i.c})
            if (o->source == Source::Input) t.inputs_read = static_cast<u16>(t.inputs_read | (1u << (o->index & 15)));
        g += std::format("    // {}: {}\n", program.start + n, t.disassembly.back());
        const bool paired = i.mac != Mac::Nop && i.ilu != Ilu::Nop;
        // Both units read their operands before any write lands.
        if (i.mac != Mac::Nop) {
            const std::string a = operand_glsl(i.a), b = operand_glsl(i.b), c = operand_glsl(i.c);
            switch (i.mac) {
                case Mac::Mov: g += std::format("    mac = {};\n", a); break;
                case Mac::Mul: g += std::format("    mac = vp_mul({}, {});\n", a, b); break;
                case Mac::Add: g += std::format("    mac = {} + {};\n", a, c); break;
                case Mac::Mad: g += std::format("    mac = vp_mul({}, {}) + {};\n", a, b, c); break;
                case Mac::Dp3: g += std::format("    mac = vp_dp3({}, {});\n", a, b); break;
                case Mac::Dph: g += std::format("    mac = vp_dph({}, {});\n", a, b); break;
                case Mac::Dp4: g += std::format("    mac = vp_dp4({}, {});\n", a, b); break;
                case Mac::Dst: g += std::format("    mac = vp_dst({}, {});\n", a, b); break;
                case Mac::Min: g += std::format("    mac = min({}, {});\n", a, b); break;
                case Mac::Max: g += std::format("    mac = max({}, {});\n", a, b); break;
                case Mac::Slt: g += std::format("    mac = vp_slt({}, {});\n", a, b); break;
                case Mac::Sge: g += std::format("    mac = vp_sge({}, {});\n", a, b); break;
                case Mac::Arl: g += std::format("    mac = {};\n", a); break;
                case Mac::Nop: break;
            }
        }
        if (i.ilu != Ilu::Nop) {
            const std::string c = operand_glsl(i.c);
            switch (i.ilu) {
                case Ilu::Mov: g += std::format("    ilu = {};\n", c); break;
                case Ilu::Rcp: g += std::format("    ilu = vp_rcp({});\n", c); break;
                case Ilu::Rcc: g += std::format("    ilu = vp_rcc({});\n", c); break;
                case Ilu::Rsq: g += std::format("    ilu = vp_rsq({});\n", c); break;
                case Ilu::Exp: g += std::format("    ilu = vp_exp({});\n", c); break;
                case Ilu::Log: g += std::format("    ilu = vp_log({});\n", c); break;
                case Ilu::Lit: g += std::format("    ilu = vp_lit({});\n", c); break;
                case Ilu::Nop: break;
            }
        }
        // Writes.
        auto write = [&](const std::string& dst, u32 mask, const char* src) {
            const std::string m = mask_string(mask);
            if (m.empty()) return;
            g += std::format("    {}.{} = {}.{};\n", dst, m, src, m);
        };
        if (i.mac == Mac::Arl) {
            g += "    A0 = vp_arl(mac);\n";
        } else if (i.mac != Mac::Nop) {
            write(i.temp == 12 ? "oPos" : std::format("R{}", i.temp), i.mac_mask, "mac");
            if (!i.output_from_ilu && i.output_mask && !i.output_is_constant) {
                if (const char* name = output_name(i.output_index & 0xF)) write(name, i.output_mask, "mac");
            }
        }
        if (i.ilu != Ilu::Nop) {
            const u32 temp = paired ? 1 : i.temp;
            write(temp == 12 ? "oPos" : std::format("R{}", temp), i.ilu_mask, "ilu");
            if (i.output_from_ilu && i.output_mask && !i.output_is_constant) {
                if (const char* name = output_name(i.output_index & 0xF)) write(name, i.output_mask, "ilu");
            }
        }
        if (i.output_is_constant && i.output_mask) {
            if (t.error.empty()) t.error = std::format("instruction {} writes constant c[{}] (vertex state shaders are not supported yet)", n, i.output_index);
        }
    }

    append_tail(g, options);
    t.glsl = std::move(g);
    return t;
}


// ── the fixed-function vertex pipeline ──────────────────────────────────────────────────────────────────────────

namespace {

namespace ff {
constexpr u32 kLightControl = 0x0294, kColorMaterial = 0x0298, kFogGenMode = 0x02A0, kFogEnable = 0x02A4, kSkinMode = 0x0328,
              kNormalizationEnable = 0x03A4, kSpecularEnable = 0x03B8, kTexgen = 0x03C0, kTextureMatrixEnable = 0x0420,
              kTexgenViewModel = 0x09CC, kFogParams = 0x09C0, kSpecularParams = 0x09E0, kLight = 0x1000, kLightStride = 0x80;
// Within a light's block: colours (three floats each), range, infinite half vector and direction, spot falloff
// (three) and direction (four), local position and attenuation (three).
constexpr u32 kLightAmbient = 0x00, kLightDiffuse = 0x0C, kLightSpecular = 0x18, kLightRange = 0x24, kLightHalfVector = 0x28,
              kLightDirection = 0x34, kLightSpotFalloff = 0x40, kLightSpotDirection = 0x4C, kLightPosition = 0x5C, kLightAttenuation = 0x68;
constexpr u32 kSeparateSpecular = 1u << 0, kLocalEye = 1u << 16;
// SET_TEXGEN_* values.
constexpr u32 kTexgenOff = 0, kEyeLinear = 0x2400, kObjectLinear = 0x2401, kSphereMap = 0x2402, kNormalMap = 0x8511, kReflectionMap = 0x8512;
// XF context slots.
constexpr u32 kComposite = 0x00, kModelView = 0x08, kInverseModelView = 0x0C, kEye = 0x38, kViewportOffset = 0x3B, kTexgenPlanes = 0x40,
              kTextureMatrix = 0x44;
// The power curve (x + L) / (M x + N) the hardware evaluates for specular and spot falloff, before software sets
// one: the reference emulator's reset values [verify].
constexpr float kCurveL = 0.0f, kCurveM = -0.494592f, kCurveN = 1.494592f;
}  // namespace ff

// SET_SKIN_MODE: the number of model-view matrices blended, and whether the last weight is 1 - the others.
struct Skin {
    u32 matrices{0};
    bool generated_last{false};
};
bool skin_mode(u32 v, Skin& s) {
    switch (v) {
        case 0: s = {0, false}; return true;
        case 1: s = {2, true}; return true;
        case 2: s = {2, false}; return true;
        case 3: s = {3, true}; return true;
        case 4: s = {3, false}; return true;
        case 5: s = {4, true}; return true;
        case 6: s = {4, false}; return true;
        default: return false;
    }
}

// A colour source of SET_COLOR_MATERIAL (0 material, 1 the vertex diffuse, 2 the vertex specular).
const char* colour_source(u32 src) { return src == 1 ? "v3" : "v4"; }

}  // namespace

std::vector<u32> fixed_function_key(const KelvinState& s) {
    std::vector<u32> k{s.reg(ff::kLightControl), s.reg(ff::kColorMaterial), s.reg(ff::kFogGenMode), s.reg(ff::kFogEnable) & 1u,
                       s.reg(mthd::kLightingEnable) & 1u, s.reg(ff::kSkinMode), s.reg(ff::kNormalizationEnable) & 1u,
                       s.reg(ff::kSpecularEnable) & 1u, s.reg(mthd::kLightEnableMask), s.reg(ff::kTexgenViewModel)};
    for (u32 i = 0; i < 16; ++i) k.push_back(s.reg(ff::kTexgen + 4 * i));
    for (u32 i = 0; i < 4; ++i) k.push_back(s.reg(ff::kTextureMatrixEnable + 4 * i) & 1u);
    return k;
}

bool fixed_function_supported(const KelvinState& s, std::string& why) {
    Skin skin;
    if (!skin_mode(s.reg(ff::kSkinMode), skin)) {
        why = std::format("fixed-function skin mode {} is not a valid mode", s.reg(ff::kSkinMode));
        return false;
    }
    if (s.reg(mthd::kLightingEnable) & 1u) {
        const u32 cm = s.reg(ff::kColorMaterial);
        for (u32 shift = 0; shift < 8; shift += 2)
            if (((cm >> shift) & 3u) == 3u) {
                why = std::format("fixed-function colour material {:x} names an invalid source", cm);
                return false;
            }
    }
    for (u32 t = 0; t < 4; ++t)
        for (u32 c = 0; c < 4; ++c) {
            const u32 g = s.reg(ff::kTexgen + 16 * t + 4 * c);
            const bool ok = g == ff::kTexgenOff || g == ff::kEyeLinear || g == ff::kObjectLinear || (g == ff::kSphereMap && c < 2) ||
                            ((g == ff::kReflectionMap || g == ff::kNormalMap) && c < 3);
            if (!ok) {
                why = std::format("fixed-function texgen {:x} on coordinate {} of texture {} is not valid", g, "STRQ"[c], t);
                return false;
            }
        }
    if (s.reg(ff::kFogEnable) & 1u) {
        const u32 m = s.reg(ff::kFogGenMode);
        if (m > 3 && m != 6) {
            why = std::format("fixed-function fog generation mode {} is not valid", m);
            return false;
        }
    }
    return true;
}

void fixed_function_uniforms(const KelvinState& s, float* out) {
    auto put3 = [&](u32 method, float* to) {
        for (u32 i = 0; i < 3; ++i) to[i] = s.regf(method + 4 * i);
    };
    auto curve = [&](u32 method, float* to) {
        put3(method, to);
        if (to[0] == 0.0f && to[1] == 0.0f && to[2] == 0.0f) {
            to[0] = ff::kCurveL;
            to[1] = ff::kCurveM;
            to[2] = ff::kCurveN;
        }
    };
    std::fill(out, out + kFixedFunctionVectors * 4, 0.0f);
    put3(mthd::kMaterialEmission, out + 0);
    out[3] = s.regf(mthd::kMaterialAlpha);
    put3(mthd::kSceneAmbientColor, out + 4);
    curve(ff::kSpecularParams, out + 8);
    for (u32 i = 0; i < 3; ++i) out[12 + i] = s.regf(ff::kFogParams + 4 * i);
    for (u32 l = 0; l < kLights; ++l) {
        float* v = out + 16 + l * kLightVectors * 4;
        const u32 base = ff::kLight + ff::kLightStride * l;
        put3(base + ff::kLightAmbient, v + 0);
        put3(base + ff::kLightDiffuse, v + 4);
        put3(base + ff::kLightSpecular, v + 8);
        put3(base + ff::kLightPosition, v + 12);
        v[15] = s.regf(base + ff::kLightRange);
        put3(base + ff::kLightAttenuation, v + 16);
        put3(base + ff::kLightDirection, v + 20);
        put3(base + ff::kLightHalfVector, v + 24);
        curve(base + ff::kLightSpotFalloff, v + 28);
        for (u32 i = 0; i < 4; ++i) v[32 + i] = s.regf(base + ff::kLightSpotDirection + 4 * i);
    }
}

Translation fixed_function_translate(const KelvinState& s, const Options& options) {
    Translation t;
    if (!fixed_function_supported(s, t.error)) return t;
    std::string g;
    append_head(g);
    auto xf = [](const std::string& v, u32 slot) { return std::format("xf_mul({}, {})", v, slot); };
    Skin skin;
    skin_mode(s.reg(ff::kSkinMode), skin);
    const bool lit = (s.reg(mthd::kLightingEnable) & 1u) != 0;
    u16 inputs = (1u << 0) | (1u << 3) | (1u << 4) | (1u << 7) | (1u << 8) | (0xFu << 9);

    // Eye space: the position and the normal through the model-view matrices, blended when skinning.
    g += "    // The fixed-function pipeline\n";
    if (skin.matrices == 0) {
        g += std::format("    vec4 tPosition = {};\n", xf("v0", ff::kModelView));
        g += std::format("    vec3 tNormal = {}.xyz;\n", xf("vec4(v2.xyz, 0.0)", ff::kInverseModelView));
    } else {
        inputs |= 1u << 1;
        g += "    vec4 tPosition = vec4(0.0);\n    vec3 tNormal = vec3(0.0);\n    float weight_left = 1.0;\n";
        for (u32 m = 0; m < skin.matrices; ++m) {
            const bool last_generated = skin.generated_last && m + 1 == skin.matrices;
            g += std::format("    {{ float w = {};", last_generated ? "weight_left" : std::format("v1.{}", "xyzw"[m]));
            if (!last_generated) g += " weight_left -= w;";
            g += std::format(" tPosition += {} * w; tNormal += {}.xyz * w; }}\n", xf("v0", ff::kModelView + 8 * m),
                             xf("vec4(v2.xyz, 0.0)", ff::kInverseModelView + 8 * m));
        }
    }
    if (s.reg(ff::kNormalizationEnable) & 1u) g += "    tNormal = normalize(tNormal);\n";

    // Texture coordinates: the vertex's, or generated per coordinate, then the texture matrix.
    const bool infinite_viewer = s.reg(ff::kTexgenViewModel) == 1;
    for (u32 tex = 0; tex < 4; ++tex) {
        g += std::format("    oT{} = v{};\n", tex, 9 + tex);
        for (u32 c = 0; c < 4; ++c) {
            const u32 mode = s.reg(ff::kTexgen + 16 * tex + 4 * c);
            const char comp = "xyzw"[c];
            const u32 plane = ff::kTexgenPlanes + 8 * tex + c;
            const std::string eye = infinite_viewer ? "vec3(0.0, 0.0, 1.0)" : "normalize(tPosition.xyz)";
            switch (mode) {
                case ff::kEyeLinear: g += std::format("    oT{}.{} = dot(c[{}], tPosition);\n", tex, comp, plane); break;
                case ff::kObjectLinear: g += std::format("    oT{}.{} = dot(c[{}], v0);\n", tex, comp, plane); break;
                case ff::kSphereMap:
                    g += std::format("    {{ vec3 r = reflect({}, tNormal); oT{}.{} = r.{} / (2.0 * length(r + vec3(0.0, 0.0, 1.0))) + 0.5; }}\n", eye, tex,
                                     comp, comp);
                    break;
                case ff::kReflectionMap: g += std::format("    oT{}.{} = reflect({}, tNormal).{};\n", tex, comp, eye, comp); break;
                case ff::kNormalMap: g += std::format("    oT{}.{} = tNormal.{};\n", tex, comp, comp); break;
                default: break;
            }
        }
        if (s.reg(ff::kTextureMatrixEnable + 4 * tex) & 1u) g += std::format("    oT{} = {};\n", tex, xf(std::format("oT{}", tex), ff::kTextureMatrix + 8 * tex));
    }

    // Colours.
    if (!lit) {
        g += "    oD0 = v3;\n    oD1 = v4;\n    oB0 = v7;\n    oB1 = v8;\n";
    } else {
        inputs |= 1u << 2;
        const u32 cm = s.reg(ff::kColorMaterial);
        const u32 emission = cm & 3u, ambient = (cm >> 2) & 3u, diffuse = (cm >> 4) & 3u, specular = (cm >> 6) & 3u;
        const std::string alpha = diffuse == 0 ? "ff_emission_alpha.w" : std::format("{}.w", colour_source(diffuse));
        // The XDK folds the material's emission and ambient into the scene ambient; a vertex colour named as the
        // emission or ambient source is scaled by the emission register instead [reference emulator's reading].
        if (emission == 0 || ambient == 0) {
            g += std::format("    oD0 = vec4(ff_scene_ambient.xyz, {});\n", alpha);
            const u32 added = emission != 0 ? emission : ambient;
            if (added != 0) g += std::format("    oD0.xyz += {}.xyz * ff_emission_alpha.xyz;\n", colour_source(added));
        } else {
            g += std::format("    oD0 = vec4({}.xyz + {}.xyz * ff_emission_alpha.xyz, {});\n", colour_source(emission), colour_source(ambient), alpha);
        }
        g += "    oD1 = vec4(0.0, 0.0, 0.0, v4.w);\n";
        const bool local_eye = (s.reg(ff::kLightControl) & ff::kLocalEye) != 0;
        if (local_eye) g += std::format("    vec3 VPeye = normalize(c[{0}].xyz / c[{0}].w - tPosition.xyz / tPosition.w);\n", ff::kEye);
        const u32 mask = s.reg(mthd::kLightEnableMask);
        auto scaled = [&](u32 src, const std::string& light) {
            return src == 0 ? light : std::format("{}.xyz * {}", colour_source(src), light);
        };
        for (u32 l = 0; l < kLights; ++l) {
            const u32 type = (mask >> (2 * l)) & 3u;  // 1 infinite, 2 local, 3 spot
            if (type == 0) continue;
            const u32 b = l * kLightVectors;
            g += std::format("    {{  // light {} ({})\n", l, type == 1 ? "infinite" : type == 2 ? "local" : "spot");
            g += "        float attenuation = 0.0, nDotVP = 0.0, nDotHV = 0.0;\n        vec3 VP = vec3(0.0);\n";
            if (type == 1) {
                g += std::format("        attenuation = 1.0;\n        vec3 direction = normalize(ff_light[{}].xyz);\n", b + 5);
                g += "        nDotVP = max(0.0, dot(tNormal, direction));\n";
                if (local_eye) g += "        nDotHV = max(0.0, dot(tNormal, normalize(direction + VPeye)));\n";
                else g += std::format("        nDotHV = max(0.0, dot(tNormal, ff_light[{}].xyz));\n", b + 6);
            } else {
                g += std::format("        VP = ff_light[{}].xyz - tPosition.xyz / tPosition.w;\n        float d = length(VP);\n", b + 3);
                g += std::format("        if (d <= ff_light[{}].w) {{\n", b + 3);
                g += "            VP = normalize(VP);\n";
                g += std::format("            attenuation = 1.0 / (ff_light[{0}].x + ff_light[{0}].y * d + ff_light[{0}].z * d * d);\n", b + 4);
                // Without a local viewer the reference takes the half vector as VP itself [verify].
                g += std::format("            vec3 halfVector = normalize(VP{});\n", local_eye ? " + VPeye" : "");
                g += "            nDotVP = max(0.0, dot(tNormal, VP));\n            nDotHV = max(0.0, dot(tNormal, halfVector));\n        }\n";
                if (type == 3) {
                    // The spot factor: s = dot(direction, VP) + w through the falloff curve (x + L) / (M x + N);
                    // 1 inside the inner cone (s >= 1), 0 outside (s <= 0).
                    g += std::format("        float sp = dot(ff_light[{0}].xyz, VP) + ff_light[{0}].w;\n", b + 8);
                    g += "        if (sp <= 0.0) attenuation = 0.0;\n";
                    g += std::format("        else if (sp < 1.0) {{\n            vec3 k = ff_light[{}].xyz;\n            float f = 0.0;\n", b + 7);
                    g += "            if (sp > -k.x) {\n                float den = k.y * sp + k.z;\n";
                    g += "                if (den == 0.0) f = 1.0;\n                else if (den > 0.0) f = clamp((sp + k.x) / den, 0.0, 1.0);\n            }\n";
                    g += "            attenuation *= f;\n        }\n";
                }
            }
            // The specular power through the same curve with SET_SPECULAR_PARAMS.
            g += "        float pf = 0.0;\n";
            g += "        if (nDotVP > 0.0 && nDotHV > -ff_specular.x) {\n            float den = ff_specular.y * nDotHV + ff_specular.z;\n";
            g += "            if (den == 0.0) pf = 10000.0;\n            else if (den > 0.0) pf = clamp((nDotHV + ff_specular.x) / den, 0.0, 10000.0);\n        }\n";
            g += std::format("        oD0.xyz += {};\n", scaled(ambient, std::format("ff_light[{}].xyz * attenuation", b + 0)));
            g += std::format("        oD0.xyz += {};\n", scaled(diffuse, std::format("ff_light[{}].xyz * (attenuation * nDotVP)", b + 1)));
            g += std::format("        oD1.xyz += {};\n", scaled(specular, std::format("ff_light[{}].xyz * (attenuation * pf)", b + 2)));
            g += "    }\n";
        }
        // Two-sided lighting is not modelled: the back colours are the vertex's [verify].
        g += "    oB0 = v7;\n    oB1 = v8;\n";
    }
    if (!(s.reg(ff::kSpecularEnable) & 1u)) {
        g += "    oD1 = vec4(0.0, 0.0, 0.0, 1.0);\n    oB1 = vec4(0.0, 0.0, 0.0, 1.0);\n";
    } else if (!(s.reg(ff::kLightControl) & ff::kSeparateSpecular)) {
        // Specular summed into the diffuse colour by the transform; the specular register then carries the vertex's.
        if (lit) g += "    oD0.xyz += oD1.xyz;\n    oB0.xyz += oB1.xyz;\n";
        g += "    oD1 = v4;\n    oB1 = v8;\n";
    }

    // The fog coordinate.
    if (s.reg(ff::kFogEnable) & 1u) {
        switch (s.reg(ff::kFogGenMode)) {
            case 0: g += "    oFog = vec4(clamp(v4.w, 0.0, 1.0));\n"; break;
            case 1: g += "    oFog = vec4(length(tPosition.xyz));\n"; break;
            case 2: g += "    oFog = vec4(dot(c[57].xyz, tPosition.xyz) + c[57].w);\n"; break;
            case 3: g += "    oFog = vec4(abs(dot(c[57].xyz, tPosition.xyz) + c[57].w));\n"; break;
            default:
                inputs |= 1u << 5;
                g += "    oFog = vec4(v5.x);\n";
                break;
        }
    }

    // The position: the composite matrix (without the model-view when skinning blended it already), the divide by w
    // and the viewport offset, as the XDK's program epilogue leaves it.
    g += std::format("    vec4 p = {};\n", xf(skin.matrices == 0 ? "v0" : "tPosition", ff::kComposite));
    g += std::format("    oPos = vec4(p.xyz * vp_rcc(vec4(p.w)).x + c[{}].xyz, p.w);\n", ff::kViewportOffset);
    append_tail(g, options);
    t.inputs_read = inputs;
    t.glsl = std::move(g);
    return t;
}

}  // namespace mv::pgraph::vsh
