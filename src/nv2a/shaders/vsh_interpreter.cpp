#include "nv2a/shaders/vsh_interpreter.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mv::pgraph::vsh {

namespace {

float as_float(u32 v) {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

float clamp_away(float t) {
    const float lo = as_float(0x1F800000u), hi = as_float(0x5F800000u);
    if (t > 0.0f || (t == 0.0f && !std::signbit(t))) return std::clamp(t, lo, hi);
    return std::clamp(t, -hi, -lo);
}

struct Machine {
    std::array<Vec4, 12> r{};
    int a0{0};
    Outputs out;
    const std::array<Vec4, kAttributes>* in{nullptr};
    const std::array<std::array<u32, 4>, kConstants>* consts{nullptr};

    Vec4 fetch(const Operand& o) const {
        Vec4 base;
        switch (o.source) {
            case Source::Temp: base = o.index == 12 ? out.pos : r[o.index < 12 ? o.index : 0]; break;
            case Source::Input: base = (*in)[o.index & 15]; break;
            case Source::Constant: {
                int idx = static_cast<int>(o.index);
                if (o.relative) idx = std::clamp(a0 + idx, 0, static_cast<int>(kConstants) - 1);
                for (u32 i = 0; i < 4; ++i) base[i] = as_float((*consts)[static_cast<std::size_t>(idx)][i]);
                break;
            }
            case Source::None: base = Vec4{{0.0f, 0.0f, 0.0f, 0.0f}}; break;
        }
        Vec4 s;
        for (u32 i = 0; i < 4; ++i) s[i] = base[o.swizzle[i] & 3];
        if (o.negate)
            for (u32 i = 0; i < 4; ++i) s[i] = -s[i];
        return s;
    }

    Vec4* output(u32 index) {
        switch (index & 0xF) {
            case 0: return &out.pos;
            case 3: return &out.d0;
            case 4: return &out.d1;
            case 5: return &out.fog;
            case 6: return &out.pts;
            case 7: return &out.b0;
            case 8: return &out.b1;
            case 9: return &out.t0;
            case 10: return &out.t1;
            case 11: return &out.t2;
            case 12: return &out.t3;
            default: return nullptr;
        }
    }

    static void write(Vec4& dst, u32 mask, const Vec4& src) {
        if (mask & 8) dst[0] = src[0];
        if (mask & 4) dst[1] = src[1];
        if (mask & 2) dst[2] = src[2];
        if (mask & 1) dst[3] = src[3];
    }

    static Vec4 mul(const Vec4& a, const Vec4& b) {
        Vec4 o;
        for (u32 i = 0; i < 4; ++i) o[i] = (a[i] == 0.0f || b[i] == 0.0f) ? 0.0f : a[i] * b[i];
        return o;
    }
    static Vec4 splat(float f) { return Vec4{{f, f, f, f}}; }

    void step(const Instruction& i) {
        Vec4 mac, ilu;
        const bool paired = i.mac != Mac::Nop && i.ilu != Ilu::Nop;
        if (i.mac != Mac::Nop) {
            const Vec4 a = fetch(i.a), b = fetch(i.b), c = fetch(i.c);
            switch (i.mac) {
                case Mac::Mov: case Mac::Arl: mac = a; break;
                case Mac::Mul: mac = mul(a, b); break;
                case Mac::Add: for (u32 k = 0; k < 4; ++k) mac[k] = a[k] + c[k]; break;
                case Mac::Mad: { Vec4 p = mul(a, b); for (u32 k = 0; k < 4; ++k) mac[k] = p[k] + c[k]; break; }
                case Mac::Dp3: mac = splat(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]); break;
                case Mac::Dph: mac = splat(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + b[3]); break;
                case Mac::Dp4: mac = splat(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]); break;
                case Mac::Dst: mac = Vec4{{1.0f, a[1] * b[1], a[2], b[3]}}; break;
                case Mac::Min: for (u32 k = 0; k < 4; ++k) mac[k] = std::min(a[k], b[k]); break;
                case Mac::Max: for (u32 k = 0; k < 4; ++k) mac[k] = std::max(a[k], b[k]); break;
                case Mac::Slt: for (u32 k = 0; k < 4; ++k) mac[k] = a[k] < b[k] ? 1.0f : 0.0f; break;
                case Mac::Sge: for (u32 k = 0; k < 4; ++k) mac[k] = a[k] >= b[k] ? 1.0f : 0.0f; break;
                case Mac::Nop: break;
            }
        }
        if (i.ilu != Ilu::Nop) {
            const Vec4 c = fetch(i.c);
            const float x = c[0];
            switch (i.ilu) {
                case Ilu::Mov: ilu = c; break;
                case Ilu::Rcp: ilu = splat(1.0f / x); break;
                case Ilu::Rcc: ilu = splat(clamp_away(1.0f / x)); break;
                case Ilu::Rsq: ilu = splat(x == 0.0f ? INFINITY : (std::isinf(x) ? 0.0f : 1.0f / std::sqrt(std::fabs(x)))); break;
                case Ilu::Exp: { const float f = std::floor(x); ilu = Vec4{{std::exp2(f), x - f, std::exp2(x), 1.0f}}; break; }
                case Ilu::Log: {
                    const float t = std::fabs(x);
                    if (t == 0.0f) ilu = Vec4{{-INFINITY, 1.0f, -INFINITY, 1.0f}};
                    else { const float f = std::floor(std::log2(t)); ilu = Vec4{{f, t / std::exp2(f), std::log2(t), 1.0f}}; }
                    break;
                }
                case Ilu::Lit: {
                    const float e = 1.0f / 256.0f;
                    const float w = std::clamp(c[3], -(128.0f - e), 128.0f - e), sx = std::max(c[0], 0.0f), sy = std::max(c[1], 0.0f);
                    ilu = Vec4{{1.0f, sx, sx > 0.0f ? std::exp2(w * std::log2(sy)) : 0.0f, 1.0f}};
                    break;
                }
                case Ilu::Nop: break;
            }
        }
        if (i.mac == Mac::Arl) {
            a0 = static_cast<int>(std::floor(mac[0] + 0.001f));
        } else if (i.mac != Mac::Nop) {
            write(i.temp == 12 ? out.pos : r[i.temp < 12 ? i.temp : 0], i.mac_mask, mac);
            if (!i.output_from_ilu && !i.output_is_constant)
                if (Vec4* o = output(i.output_index)) write(*o, i.output_mask, mac);
        }
        if (i.ilu != Ilu::Nop) {
            const u32 temp = paired ? 1 : i.temp;
            write(temp == 12 ? out.pos : r[temp < 12 ? temp : 0], i.ilu_mask, ilu);
            if (i.output_from_ilu && !i.output_is_constant)
                if (Vec4* o = output(i.output_index)) write(*o, i.output_mask, ilu);
        }
    }
};

}  // namespace

Outputs run(const Program& program, const std::array<Vec4, kAttributes>& inputs, const std::array<std::array<u32, 4>, kConstants>& constants) {
    Machine m;
    m.in = &inputs;
    m.consts = &constants;
    for (auto& t : m.r) t = Vec4{{0.0f, 0.0f, 0.0f, 0.0f}};
    for (const Instruction& i : program.instructions) m.step(i);
    return m.out;
}

}  // namespace mv::pgraph::vsh
