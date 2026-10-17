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

Translation translate(const Program& program, const Options& options) {
    Translation t;
    (void)options;
    std::string g;
    g += "#version 450\n\n";
    for (u32 i = 0; i < kAttributes; ++i) g += std::format("layout(location = {}) in vec4 v{};\n", i, i);
    g += std::format(
        "\nlayout(std140, set = 0, binding = 0) uniform VshUniforms {{\n"
        "    vec4 c[{}];\n"
        "    vec4 surface_size;  // width, height of the render target in pixels\n"
        "    vec4 clip_range;    // z min, z max (depth units)\n"
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
        "}\n\n";
    g += "void main() {\n";
    for (u32 r = 0; r < 12; ++r) g += std::format("    vec4 R{} = vec4(0.0);\n", r);
    g += "    int A0 = 0;\n";
    g += "    vec4 oPos = vec4(0.0, 0.0, 0.0, 1.0), oD0 = vec4(0.0, 0.0, 0.0, 1.0), oD1 = vec4(0.0, 0.0, 0.0, 1.0);\n";
    g += "    vec4 oB0 = vec4(0.0, 0.0, 0.0, 1.0), oB1 = vec4(0.0, 0.0, 0.0, 1.0), oFog = vec4(0.0, 0.0, 0.0, 1.0);\n";
    g += "    vec4 oPts = vec4(0.0, 0.0, 0.0, 1.0), oT0 = vec4(0.0, 0.0, 0.0, 1.0), oT1 = vec4(0.0, 0.0, 0.0, 1.0);\n";
    g += "    vec4 oT2 = vec4(0.0, 0.0, 0.0, 1.0), oT3 = vec4(0.0, 0.0, 0.0, 1.0);\n";
    g += "    vec4 mac, ilu;\n\n";

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

    g +=
        "\n    // The program leaves oPos in screen space (x, y in pixels and z in depth units, all already divided by w,\n"
        "    // w kept for perspective-correct interpolation); the host rasteriser wants clip space, so undo that.\n"
        "    oPos.xy = trunc(oPos.xy * 16.0) / 16.0;\n"
        "    oPos.w = vp_clamp_away(oPos.w);\n"
        "    vtx_pos = oPos;\n"
        "    vec2 ndc = (2.0 * oPos.xy - surface_size.xy) / surface_size.xy;\n"
        "    float zndc = oPos.z / clip_range.y;\n"
        "    gl_Position = vec4(ndc * oPos.w, zndc * oPos.w, oPos.w);\n"
        "    gl_PointSize = max(oPts.x, 1.0);\n"
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
    t.glsl = std::move(g);
    return t;
}

}  // namespace mv::pgraph::vsh
