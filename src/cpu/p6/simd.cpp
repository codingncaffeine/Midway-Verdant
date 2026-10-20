// The MMX and SSE units of the Pentium III (no SSE2). MMX registers alias the x87 mantissas with the tag and
// TOP side effects the architecture defines; SSE arithmetic runs on the host's SSE unit under the guest's
// MXCSR (rounding and flush-to-zero), its exception flags read back. Written from SDM Vol. 1 chapters 9-10
// and the Vol. 2 instruction pages.
#include <cmath>
#include <cstring>

#include "cpu/p6/cpu.h"

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define MV_HOST_SSE 1
#else
#define MV_HOST_SSE 0
#endif

namespace mv::p6 {

namespace {

constexpr u32 kMxIE = 1u << 0, kMxDE = 1u << 1, kMxZE = 1u << 2, kMxOE = 1u << 3, kMxUE = 1u << 4, kMxPE = 1u << 5, kMxFlags = 0x3Fu;

// The host SSE control under the guest's MXCSR: rounding and flush-to-zero from the guest, everything masked.
class HostSse {
public:
    explicit HostSse(u32 guest_mxcsr) {
#if MV_HOST_SSE
        saved_ = _mm_getcsr();
        _mm_setcsr((guest_mxcsr & 0xE000u) | 0x1F80u);
#else
        (void)guest_mxcsr;
#endif
    }
    u32 flags() const {
#if MV_HOST_SSE
        return _mm_getcsr() & kMxFlags;
#else
        return 0;
#endif
    }
    ~HostSse() {
#if MV_HOST_SSE
        _mm_setcsr(saved_);
#endif
    }

private:
    u32 saved_{0x1F80};
};

inline u32 f2u(float f) {
    u32 u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline float u2f(u32 u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline bool is_nan32(u32 u) { return (u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0; }
inline bool is_snan32(u32 u) { return is_nan32(u) && (u & 0x00400000u) == 0; }
inline bool is_denormal32(u32 u) { return (u & 0x7F800000u) == 0 && (u & 0x007FFFFFu) != 0; }
constexpr u32 kQNanIndefinite = 0xFFC00000u;

struct V128 {
    u32 w[4];
};
struct V64 {
    u64 q;
};

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

struct Simd {
    Cpu& cpu;
    CpuState& st;
    FpuState& f;
    Memory& mem;
    const Insn& in;

    Simd(Cpu& c, const Insn& i) : cpu(c), st(c.state()), f(c.state().fpu), mem(c.memory()), in(i) {}

    u32 mem_addr(unsigned size, Access access, bool aligned = false) {
        const u32 ea = cpu.effective_address(in);
        if (aligned && (ea & 15)) raise_gp(0);
        return cpu.linear(in.mem_seg, ea, size, access);
    }

    // ---- MMX register file (the x87 mantissas) ----
    void mmx_touch() {
        // Any MMX instruction: TOP = 0 and every tag valid.
        f.set_top(0);
        f.ftw = 0;
    }
    u64 mmx_get(unsigned r) const {
        u64 v;
        std::memcpy(&v, &f.st[r], 8);
        return v;
    }
    void mmx_set(unsigned r, u64 v) {
        u8 raw[10];
        std::memcpy(raw, &v, 8);
        raw[8] = raw[9] = 0xFF;  // the exponent field reads as all ones after an MMX write
        long double ld = 0;
        std::memcpy(&ld, raw, 10);
        f.st[r] = ld;
    }
    u64 mmx_src() {
        if (in.mem) {
            const u32 lin = mem_addr(8, Access::Read);
            return mem.read(lin, 4) | (static_cast<u64>(mem.read(lin + 4, 4)) << 32);
        }
        return mmx_get(in.rm);
    }
    void mmx_store64(u64 v) {
        const u32 lin = mem_addr(8, Access::Write);
        mem.probe_write(lin, 8);
        mem.write(lin, 4, static_cast<u32>(v));
        mem.write(lin + 4, 4, static_cast<u32>(v >> 32));
    }

    // ---- XMM ----
    V128 xmm_get(unsigned r) const {
        V128 v;
        std::memcpy(v.w, f.xmm[r].data(), 16);
        return v;
    }
    void xmm_set(unsigned r, const V128& v) { std::memcpy(f.xmm[r].data(), v.w, 16); }
    V128 xmm_src128(bool aligned) {
        if (in.mem) {
            const u32 lin = mem_addr(16, Access::Read, aligned);
            V128 v;
            for (unsigned i = 0; i < 4; ++i) v.w[i] = mem.read(lin + 4 * i, 4);
            return v;
        }
        return xmm_get(in.rm);
    }
    u32 xmm_src32() {
        if (in.mem) return mem.read(mem_addr(4, Access::Read), 4);
        return xmm_get(in.rm).w[0];
    }
    u64 xmm_src64_low() {
        if (in.mem) {
            const u32 lin = mem_addr(8, Access::Read);
            return mem.read(lin, 4) | (static_cast<u64>(mem.read(lin + 4, 4)) << 32);
        }
        const V128 v = xmm_get(in.rm);
        return v.w[0] | (static_cast<u64>(v.w[1]) << 32);
    }
    void xmm_store128(const V128& v, bool aligned) {
        const u32 lin = mem_addr(16, Access::Write, aligned);
        mem.probe_write(lin, 16);
        for (unsigned i = 0; i < 4; ++i) mem.write(lin + 4 * i, 4, v.w[i]);
    }

    // SSE exception reporting: masked flags accumulate in MXCSR; an unmasked one raises #XM (or #UD without
    // CR4.OSXMMEXCPT) and leaves the destination untouched.
    bool report(u32 flags) {
        f.mxcsr |= flags & kMxFlags;
        const u32 unmasked = flags & kMxFlags & ~((f.mxcsr >> 7) & kMxFlags);
        if (unmasked) {
            if (st.cr4 & cr4::OSXMMEXCPT) raise(kXM);
            raise_ud();
        }
        return true;
    }
    // Pre-operation checks the host does not report the same way: SNaN operands and denormals.
    u32 operand_flags(u32 a, u32 b, bool with_b) const {
        u32 fl = 0;
        if (is_snan32(a) || (with_b && is_snan32(b))) fl |= kMxIE;
        if (is_denormal32(a) || (with_b && is_denormal32(b))) fl |= kMxDE;
        return fl;
    }
    // NaN propagation of SSE: the first operand's NaN (quietened) wins, else the second's.
    bool nan_result(u32 a, u32 b, bool with_b, u32& out) const {
        if (is_nan32(a)) {
            out = a | 0x00400000u;
            return true;
        }
        if (with_b && is_nan32(b)) {
            out = b | 0x00400000u;
            return true;
        }
        return false;
    }

    enum class FOp { Add, Sub, Mul, Div, Min, Max, Sqrt, Rcp, Rsqrt };

    // One lane of an SSE arithmetic instruction with the host under the guest's MXCSR.
    u32 lane(FOp op, u32 a, u32 b, u32& flags) {
        const bool unary = op == FOp::Sqrt || op == FOp::Rcp || op == FOp::Rsqrt;
        flags |= operand_flags(unary ? b : a, b, !unary);
        u32 r;
        if (op == FOp::Min || op == FOp::Max) {
            // MIN/MAX return the second operand when either is a NaN or both are zero.
            if (is_nan32(a) || is_nan32(b)) return b;
            const float fa = u2f(a), fb = u2f(b);
            if (fa == fb) return b;
            return (op == FOp::Min ? fa < fb : fa > fb) ? a : b;
        }
        if (nan_result(unary ? b : a, b, !unary, r)) return r;
        HostSse host(f.mxcsr);
        volatile float va = u2f(a), vb = u2f(b);
        volatile float vr;
        switch (op) {
            case FOp::Add: vr = va + vb; break;
            case FOp::Sub: vr = va - vb; break;
            case FOp::Mul: vr = va * vb; break;
            case FOp::Div: vr = va / vb; break;
            case FOp::Sqrt: vr = std::sqrt(static_cast<float>(vb)); break;
            case FOp::Rcp: {
                // The hardware's 12-bit approximation; here the exact reciprocal rounded to 12 significant bits.
                const float exact = 1.0f / vb;
                vr = round_to_bits(exact, 12);
                break;
            }
            case FOp::Rsqrt: {
                const float exact = 1.0f / std::sqrt(static_cast<float>(vb));
                vr = round_to_bits(exact, 12);
                break;
            }
            default: vr = 0; break;
        }
        flags |= host.flags();
        if (op == FOp::Rcp || op == FOp::Rsqrt) flags &= ~(kMxPE | kMxIE | kMxDE);  // approximations report nothing
        return f2u(vr);
    }
    static float round_to_bits(float v, unsigned bits) {
        u32 u = f2u(v);
        if ((u & 0x7F800000u) == 0x7F800000u) return v;
        const u32 drop = 23 - bits + 1;
        const u32 half = 1u << (drop - 1);
        u = (u + half) & ~((1u << drop) - 1);
        return u2f(u);
    }

    // ---- compares ----
    bool compare_pred(u32 a, u32 b, unsigned pred, u32& flags) {
        const bool unordered = is_nan32(a) || is_nan32(b);
        // EQ, LT, LE, UNORD, NEQ, NLT, NLE, ORD; the ordered predicates signal on quiet NaNs (LT/LE/NLT/NLE),
        // the others only on signalling NaNs.
        const bool signals_quiet = pred == 1 || pred == 2 || pred == 5 || pred == 6;
        if ((unordered && signals_quiet) || is_snan32(a) || is_snan32(b)) flags |= kMxIE;
        if (is_denormal32(a) || is_denormal32(b)) flags |= kMxDE;
        const float fa = u2f(a), fb = u2f(b);
        switch (pred) {
            case 0: return !unordered && fa == fb;
            case 1: return !unordered && fa < fb;
            case 2: return !unordered && fa <= fb;
            case 3: return unordered;
            case 4: return unordered || fa != fb;
            case 5: return unordered || !(fa < fb);
            case 6: return unordered || !(fa <= fb);
            default: return !unordered;
        }
    }

    void run();
    void run_mmx();
    void run_sse();
};

// ---------------------------------------------------------------------------------------------------------------
// MMX

namespace {
template <class T, unsigned N>
void unpack(u64 a, u64 b, bool high, u64& out) {
    T av[N], bv[N], r[N];
    std::memcpy(av, &a, 8);
    std::memcpy(bv, &b, 8);
    const unsigned base = high ? N / 2 : 0;
    for (unsigned i = 0; i < N / 2; ++i) {
        r[2 * i] = av[base + i];
        r[2 * i + 1] = bv[base + i];
    }
    std::memcpy(&out, r, 8);
}
i16 sat16(i32 v) { return static_cast<i16>(v > 32767 ? 32767 : v < -32768 ? -32768 : v); }
i8 sat8(i32 v) { return static_cast<i8>(v > 127 ? 127 : v < -128 ? -128 : v); }
u8 satu8(i32 v) { return static_cast<u8>(v > 255 ? 255 : v < 0 ? 0 : v); }
u16 satu16(i32 v) { return static_cast<u16>(v > 65535 ? 65535 : v < 0 ? 0 : v); }
}  // namespace

void Simd::run_mmx() {
    if (st.cr0 & cr0::EM) raise_ud();
    if (st.cr0 & cr0::TS) raise(kNM);
    if (f.fsw & 0x80) raise(kMF);  // a pending x87 exception is taken first
    const u8 op = in.opcode;
    if (op == 0x77) {  // EMMS
        f.ftw = 0xFFFF;
        return;
    }
    // Reads of the source and the destination (every other MMX instruction) touch the x87 state.
    const unsigned d = in.reg;
    if (op == 0x6E) {  // MOVD mm, r/m32
        mmx_touch();
        const u32 v = in.mem ? mem.read(mem_addr(4, Access::Read), 4) : st.gpr[in.rm];
        mmx_set(d, v);
        return;
    }
    if (op == 0x7E) {  // MOVD r/m32, mm
        mmx_touch();
        const u32 v = static_cast<u32>(mmx_get(d));
        if (in.mem) mem.write(mem_addr(4, Access::Write), 4, v);
        else st.gpr[in.rm] = v;
        return;
    }
    if (op == 0x6F) {  // MOVQ mm, mm/m64
        mmx_touch();
        mmx_set(d, mmx_src());
        return;
    }
    if (op == 0x7F || op == 0xE7) {  // MOVQ mm/m64, mm; MOVNTQ m64, mm
        mmx_touch();
        if (in.mem) mmx_store64(mmx_get(d));
        else if (op == 0x7F) mmx_set(in.rm, mmx_get(d));
        else raise_ud();
        return;
    }
    if (op == 0xC5) {  // PEXTRW r32, mm, imm8
        mmx_touch();
        if (in.mem) raise_ud();
        const u64 v = mmx_get(in.rm);
        st.gpr[in.reg] = static_cast<u32>((v >> (16 * (in.imm & 3))) & 0xFFFF);
        return;
    }
    if (op == 0xC4) {  // PINSRW mm, r32/m16, imm8
        mmx_touch();
        const u32 w = in.mem ? mem.read(mem_addr(2, Access::Read), 2) : (st.gpr[in.rm] & 0xFFFF);
        const unsigned sh = 16 * (in.imm & 3);
        u64 v = mmx_get(d);
        v = (v & ~(u64{0xFFFF} << sh)) | (static_cast<u64>(w) << sh);
        mmx_set(d, v);
        return;
    }
    if (op == 0xD7) {  // PMOVMSKB r32, mm
        mmx_touch();
        if (in.mem) raise_ud();
        const u64 v = mmx_get(in.rm);
        u32 m = 0;
        for (unsigned i = 0; i < 8; ++i)
            if ((v >> (8 * i + 7)) & 1) m |= 1u << i;
        st.gpr[in.reg] = m;
        return;
    }
    if (op == 0xF7) {  // MASKMOVQ mm, mm: bytes of mm(reg) with the mask's high bit set go to DS:(E)DI
        mmx_touch();
        if (in.mem) raise_ud();
        const u64 data = mmx_get(in.reg), mask = mmx_get(in.rm);
        const u8 seg = in.seg != kSegNone ? in.seg : static_cast<u8>(kDs);
        const u32 di = in.adsize == 4 ? st.gpr[kEdi] : st.reg16(kEdi);
        for (unsigned i = 0; i < 8; ++i)
            if ((mask >> (8 * i + 7)) & 1) mem.write(cpu.linear(seg, di + i, 1, Access::Write), 1, static_cast<u32>((data >> (8 * i)) & 0xFF));
        return;
    }
    if (op == 0x70) {  // PSHUFW mm, mm/m64, imm8
        mmx_touch();
        const u64 s = mmx_src();
        u64 r = 0;
        for (unsigned i = 0; i < 4; ++i) r |= ((s >> (16 * ((in.imm >> (2 * i)) & 3))) & 0xFFFF) << (16 * i);
        mmx_set(d, r);
        return;
    }
    if (op == 0x71 || op == 0x72 || op == 0x73) {  // shifts by immediate: the register is in rm
        mmx_touch();
        if (in.mem) raise_ud();
        const unsigned r = in.rm;
        const u64 v = mmx_get(r);
        const unsigned n = in.imm & 0xFF;
        u64 out = 0;
        const unsigned width = op == 0x71 ? 16 : op == 0x72 ? 32 : 64;
        const unsigned count = 64 / width;
        for (unsigned i = 0; i < count; ++i) {
            const u64 lane_mask = width == 64 ? ~u64{0} : ((u64{1} << width) - 1);
            u64 lane = (v >> (width * i)) & lane_mask;
            u64 res;
            if (in.reg == 2) res = n >= width ? 0 : lane >> n;                                   // PSRL
            else if (in.reg == 6) res = n >= width ? 0 : (lane << n) & lane_mask;                // PSLL
            else if (in.reg == 4 && width != 64) {                                               // PSRA
                const i64 sv = width == 16 ? static_cast<i16>(lane) : static_cast<i32>(lane);
                res = static_cast<u64>(sv >> (n >= width ? width - 1 : n)) & lane_mask;
            } else raise_ud();
            out |= res << (width * i);
        }
        mmx_set(r, out);
        return;
    }
    mmx_touch();
    const u64 a = mmx_get(d);
    const u64 b = mmx_src();
    u64 r = 0;
    auto lanes16 = [&](auto fn) {
        for (unsigned i = 0; i < 4; ++i) {
            const i32 x = static_cast<i16>((a >> (16 * i)) & 0xFFFF), y = static_cast<i16>((b >> (16 * i)) & 0xFFFF);
            r |= (static_cast<u64>(static_cast<u16>(fn(x, y)))) << (16 * i);
        }
    };
    auto lanes8 = [&](auto fn) {
        for (unsigned i = 0; i < 8; ++i) {
            const i32 x = static_cast<i8>((a >> (8 * i)) & 0xFF), y = static_cast<i8>((b >> (8 * i)) & 0xFF);
            r |= (static_cast<u64>(static_cast<u8>(fn(x, y)))) << (8 * i);
        }
    };
    auto lanes8u = [&](auto fn) {
        for (unsigned i = 0; i < 8; ++i) {
            const i32 x = static_cast<i32>((a >> (8 * i)) & 0xFF), y = static_cast<i32>((b >> (8 * i)) & 0xFF);
            r |= (static_cast<u64>(static_cast<u8>(fn(x, y)))) << (8 * i);
        }
    };
    auto lanes16u = [&](auto fn) {
        for (unsigned i = 0; i < 4; ++i) {
            const i32 x = static_cast<i32>((a >> (16 * i)) & 0xFFFF), y = static_cast<i32>((b >> (16 * i)) & 0xFFFF);
            r |= (static_cast<u64>(static_cast<u16>(fn(x, y)))) << (16 * i);
        }
    };
    auto lanes32 = [&](auto fn) {
        for (unsigned i = 0; i < 2; ++i) {
            const u32 x = static_cast<u32>(a >> (32 * i)), y = static_cast<u32>(b >> (32 * i));
            r |= static_cast<u64>(fn(x, y)) << (32 * i);
        }
    };
    auto shift_by = [&](unsigned width, bool left, bool arith) {
        const unsigned n = b > 255 ? 255 : static_cast<unsigned>(b);
        const unsigned count = 64 / width;
        for (unsigned i = 0; i < count; ++i) {
            const u64 lane_mask = width == 64 ? ~u64{0} : ((u64{1} << width) - 1);
            const u64 lane = (a >> (width * i)) & lane_mask;
            u64 res;
            if (arith) {
                const i64 sv = width == 16 ? static_cast<i16>(lane) : static_cast<i32>(lane);
                res = static_cast<u64>(sv >> (n >= width ? width - 1 : n)) & lane_mask;
            } else if (left) res = n >= width ? 0 : (lane << n) & lane_mask;
            else res = n >= width ? 0 : lane >> n;
            r |= res << (width * i);
        }
    };
    switch (op) {
        case 0x60: unpack<u8, 8>(a, b, false, r); break;
        case 0x61: unpack<u16, 4>(a, b, false, r); break;
        case 0x62: unpack<u32, 2>(a, b, false, r); break;
        case 0x68: unpack<u8, 8>(a, b, true, r); break;
        case 0x69: unpack<u16, 4>(a, b, true, r); break;
        case 0x6A: unpack<u32, 2>(a, b, true, r); break;
        case 0x63:  // PACKSSWB
            for (unsigned i = 0; i < 4; ++i) {
                r |= static_cast<u64>(static_cast<u8>(sat8(static_cast<i16>((a >> (16 * i)) & 0xFFFF)))) << (8 * i);
                r |= static_cast<u64>(static_cast<u8>(sat8(static_cast<i16>((b >> (16 * i)) & 0xFFFF)))) << (8 * (i + 4));
            }
            break;
        case 0x67:  // PACKUSWB
            for (unsigned i = 0; i < 4; ++i) {
                r |= static_cast<u64>(satu8(static_cast<i16>((a >> (16 * i)) & 0xFFFF))) << (8 * i);
                r |= static_cast<u64>(satu8(static_cast<i16>((b >> (16 * i)) & 0xFFFF))) << (8 * (i + 4));
            }
            break;
        case 0x6B:  // PACKSSDW
            for (unsigned i = 0; i < 2; ++i) {
                r |= static_cast<u64>(static_cast<u16>(sat16(static_cast<i32>(a >> (32 * i))))) << (16 * i);
                r |= static_cast<u64>(static_cast<u16>(sat16(static_cast<i32>(b >> (32 * i))))) << (16 * (i + 2));
            }
            break;
        case 0x64: lanes8([](i32 x, i32 y) { return x > y ? 0xFF : 0; }); break;
        case 0x65: lanes16([](i32 x, i32 y) { return x > y ? 0xFFFF : 0; }); break;
        case 0x66: lanes32([](u32 x, u32 y) { return static_cast<i32>(x) > static_cast<i32>(y) ? 0xFFFFFFFFu : 0u; }); break;
        case 0x74: lanes8([](i32 x, i32 y) { return x == y ? 0xFF : 0; }); break;
        case 0x75: lanes16([](i32 x, i32 y) { return x == y ? 0xFFFF : 0; }); break;
        case 0x76: lanes32([](u32 x, u32 y) { return x == y ? 0xFFFFFFFFu : 0u; }); break;
        case 0xD1: shift_by(16, false, false); break;
        case 0xD2: shift_by(32, false, false); break;
        case 0xD3: shift_by(64, false, false); break;
        case 0xE1: shift_by(16, false, true); break;
        case 0xE2: shift_by(32, false, true); break;
        case 0xF1: shift_by(16, true, false); break;
        case 0xF2: shift_by(32, true, false); break;
        case 0xF3: shift_by(64, true, false); break;
        case 0xD5: lanes16([](i32 x, i32 y) { return (x * y) & 0xFFFF; }); break;                           // PMULLW
        case 0xE5: lanes16([](i32 x, i32 y) { return (x * y) >> 16; }); break;                              // PMULHW
        case 0xE4:  // PMULHUW: the product in 32 unsigned bits (0xFFFF * 0xFFFF overflows an int: undefined in C++)
            lanes16u([](i32 x, i32 y) { return static_cast<i32>((static_cast<u32>(x) * static_cast<u32>(y)) >> 16); });
            break;
        case 0xF5:                                                                                           // PMADDWD
            for (unsigned i = 0; i < 2; ++i) {
                const i32 x0 = static_cast<i16>((a >> (32 * i)) & 0xFFFF), x1 = static_cast<i16>((a >> (32 * i + 16)) & 0xFFFF);
                const i32 y0 = static_cast<i16>((b >> (32 * i)) & 0xFFFF), y1 = static_cast<i16>((b >> (32 * i + 16)) & 0xFFFF);
                // In 64 bits: 0x8000 * 0x8000 twice is 2^31, past an int (the hardware wraps it to 0x80000000).
                r |= static_cast<u64>(static_cast<u32>(static_cast<i64>(x0) * y0 + static_cast<i64>(x1) * y1)) << (32 * i);
            }
            break;
        case 0xF6: {  // PSADBW
            u32 sum = 0;
            for (unsigned i = 0; i < 8; ++i) {
                const i32 x = static_cast<i32>((a >> (8 * i)) & 0xFF), y = static_cast<i32>((b >> (8 * i)) & 0xFF);
                sum += static_cast<u32>(x > y ? x - y : y - x);
            }
            r = sum;
            break;
        }
        case 0xD8: lanes8u([](i32 x, i32 y) { return satu8(x - y); }); break;    // PSUBUSB
        case 0xD9: lanes16u([](i32 x, i32 y) { return satu16(x - y); }); break;  // PSUBUSW
        case 0xDC: lanes8u([](i32 x, i32 y) { return satu8(x + y); }); break;    // PADDUSB
        case 0xDD: lanes16u([](i32 x, i32 y) { return satu16(x + y); }); break;  // PADDUSW
        case 0xE8: lanes8([](i32 x, i32 y) { return sat8(x - y); }); break;      // PSUBSB
        case 0xE9: lanes16([](i32 x, i32 y) { return sat16(x - y); }); break;    // PSUBSW
        case 0xEC: lanes8([](i32 x, i32 y) { return sat8(x + y); }); break;      // PADDSB
        case 0xED: lanes16([](i32 x, i32 y) { return sat16(x + y); }); break;    // PADDSW
        case 0xF8: lanes8u([](i32 x, i32 y) { return (x - y) & 0xFF; }); break;
        case 0xF9: lanes16u([](i32 x, i32 y) { return (x - y) & 0xFFFF; }); break;
        case 0xFA: lanes32([](u32 x, u32 y) { return x - y; }); break;
        case 0xFC: lanes8u([](i32 x, i32 y) { return (x + y) & 0xFF; }); break;
        case 0xFD: lanes16u([](i32 x, i32 y) { return (x + y) & 0xFFFF; }); break;
        case 0xFE: lanes32([](u32 x, u32 y) { return x + y; }); break;
        case 0xDA: lanes8u([](i32 x, i32 y) { return x < y ? x : y; }); break;   // PMINUB
        case 0xDE: lanes8u([](i32 x, i32 y) { return x > y ? x : y; }); break;   // PMAXUB
        case 0xEA: lanes16([](i32 x, i32 y) { return x < y ? x : y; }); break;   // PMINSW
        case 0xEE: lanes16([](i32 x, i32 y) { return x > y ? x : y; }); break;   // PMAXSW
        case 0xE0: lanes8u([](i32 x, i32 y) { return (x + y + 1) >> 1; }); break;   // PAVGB
        case 0xE3: lanes16u([](i32 x, i32 y) { return (x + y + 1) >> 1; }); break;  // PAVGW
        case 0xDB: r = a & b; break;
        case 0xDF: r = ~a & b; break;
        case 0xEB: r = a | b; break;
        case 0xEF: r = a ^ b; break;
        default: raise_ud();
    }
    mmx_set(d, r);
}

// ---------------------------------------------------------------------------------------------------------------
// SSE

void Simd::run_sse() {
    if (!(st.cr4 & cr4::OSFXSR)) raise_ud();
    if (st.cr0 & cr0::EM) raise_ud();
    if (st.cr0 & cr0::TS) raise(kNM);
    const u8 op = in.opcode;
    const bool scalar = in.rep == 1;  // the F3 prefix
    const unsigned d = in.reg;
    u32 flags = 0;
    switch (op) {
        case 0x10: {  // MOVUPS xmm, xmm/m128 | MOVSS xmm, xmm/m32
            if (scalar) {
                V128 v = xmm_get(d);
                if (in.mem) v = V128{{xmm_src32(), 0, 0, 0}};
                else v.w[0] = xmm_get(in.rm).w[0];
                xmm_set(d, v);
            } else {
                xmm_set(d, xmm_src128(false));
            }
            return;
        }
        case 0x11: {  // MOVUPS xmm/m128, xmm | MOVSS xmm/m32, xmm
            const V128 v = xmm_get(d);
            if (scalar) {
                if (in.mem) mem.write(mem_addr(4, Access::Write), 4, v.w[0]);
                else {
                    V128 t = xmm_get(in.rm);
                    t.w[0] = v.w[0];
                    xmm_set(in.rm, t);
                }
            } else {
                if (in.mem) xmm_store128(v, false);
                else xmm_set(in.rm, v);
            }
            return;
        }
        case 0x12: {  // MOVLPS xmm, m64 | MOVHLPS xmm, xmm
            V128 v = xmm_get(d);
            if (in.mem) {
                const u64 q = xmm_src64_low();
                v.w[0] = static_cast<u32>(q);
                v.w[1] = static_cast<u32>(q >> 32);
            } else {
                const V128 s = xmm_get(in.rm);
                v.w[0] = s.w[2];
                v.w[1] = s.w[3];
            }
            xmm_set(d, v);
            return;
        }
        case 0x13: {  // MOVLPS m64, xmm
            if (!in.mem) raise_ud();
            const V128 v = xmm_get(d);
            const u32 lin = mem_addr(8, Access::Write);
            mem.probe_write(lin, 8);
            mem.write(lin, 4, v.w[0]);
            mem.write(lin + 4, 4, v.w[1]);
            return;
        }
        case 0x16: {  // MOVHPS xmm, m64 | MOVLHPS xmm, xmm
            V128 v = xmm_get(d);
            if (in.mem) {
                const u64 q = xmm_src64_low();
                v.w[2] = static_cast<u32>(q);
                v.w[3] = static_cast<u32>(q >> 32);
            } else {
                const V128 s = xmm_get(in.rm);
                v.w[2] = s.w[0];
                v.w[3] = s.w[1];
            }
            xmm_set(d, v);
            return;
        }
        case 0x17: {  // MOVHPS m64, xmm
            if (!in.mem) raise_ud();
            const V128 v = xmm_get(d);
            const u32 lin = mem_addr(8, Access::Write);
            mem.probe_write(lin, 8);
            mem.write(lin, 4, v.w[2]);
            mem.write(lin + 4, 4, v.w[3]);
            return;
        }
        case 0x14: case 0x15: {  // UNPCKLPS / UNPCKHPS
            const V128 a = xmm_get(d), b = xmm_src128(true);
            const unsigned base = op == 0x15 ? 2 : 0;
            xmm_set(d, V128{{a.w[base], b.w[base], a.w[base + 1], b.w[base + 1]}});
            return;
        }
        case 0x28: xmm_set(d, xmm_src128(true)); return;  // MOVAPS xmm, xmm/m128
        case 0x29: case 0x2B: {                            // MOVAPS xmm/m128, xmm; MOVNTPS m128, xmm
            const V128 v = xmm_get(d);
            if (in.mem) xmm_store128(v, true);
            else if (op == 0x29) xmm_set(in.rm, v);
            else raise_ud();
            return;
        }
        case 0x2A: {  // CVTPI2PS xmm, mm/m64 | CVTSI2SS xmm, r/m32
            V128 v = xmm_get(d);
            HostSse host(f.mxcsr);
            if (scalar) {
                const i32 s = static_cast<i32>(in.mem ? mem.read(mem_addr(4, Access::Read), 4) : st.gpr[in.rm]);
                volatile float r = static_cast<float>(s);
                v.w[0] = f2u(r);
            } else {
                if (!in.mem) mmx_touch();
                const u64 q = mmx_src();
                volatile float r0 = static_cast<float>(static_cast<i32>(static_cast<u32>(q)));
                volatile float r1 = static_cast<float>(static_cast<i32>(static_cast<u32>(q >> 32)));
                v.w[0] = f2u(r0);
                v.w[1] = f2u(r1);
            }
            flags = host.flags();
            report(flags);
            xmm_set(d, v);
            return;
        }
        case 0x2C: case 0x2D: {  // CVT(T)PS2PI mm, xmm/m64 | CVT(T)SS2SI r32, xmm/m32
            const bool truncate = op == 0x2C;
            auto convert = [&](u32 bits, u32& out) {
                const float x = u2f(bits);
                if (is_nan32(bits) || !(x >= -2147483648.0f && x < 2147483648.0f)) {
                    flags |= kMxIE;
                    out = 0x80000000u;
                    return;
                }
                HostSse host(truncate ? ((f.mxcsr & ~0x6000u) | 0x6000u) : f.mxcsr);
                volatile float vx = x;
#if MV_HOST_SSE
                const i32 r = _mm_cvtss_si32(_mm_set_ss(vx));
#else
                const i32 r = static_cast<i32>(truncate ? std::trunc(vx) : std::nearbyint(vx));
#endif
                flags |= host.flags();
                out = static_cast<u32>(r);
            };
            if (scalar) {
                u32 out;
                convert(xmm_src32(), out);
                report(flags);
                st.gpr[in.reg] = out;
            } else {
                const u64 q = xmm_src64_low();
                u32 lo, hi;
                convert(static_cast<u32>(q), lo);
                convert(static_cast<u32>(q >> 32), hi);
                report(flags);
                mmx_touch();
                mmx_set(in.reg, lo | (static_cast<u64>(hi) << 32));
            }
            return;
        }
        case 0x2E: case 0x2F: {  // UCOMISS / COMISS xmm, xmm/m32
            const u32 a = xmm_get(d).w[0], b = xmm_src32();
            const bool unordered = is_nan32(a) || is_nan32(b);
            if (is_snan32(a) || is_snan32(b) || (op == 0x2F && unordered)) flags |= kMxIE;
            if (is_denormal32(a) || is_denormal32(b)) flags |= kMxDE;
            report(flags);
            st.materialise();
            u32 fl = st.flags & ~(fl::ZF | fl::PF | fl::CF | fl::OF | fl::SF | fl::AF);
            if (unordered) fl |= fl::ZF | fl::PF | fl::CF;
            else {
                const float fa = u2f(a), fb = u2f(b);
                if (fa < fb) fl |= fl::CF;
                else if (fa == fb) fl |= fl::ZF;
            }
            st.flags = fl;
            return;
        }
        case 0x50: {  // MOVMSKPS r32, xmm
            if (in.mem) raise_ud();
            const V128 v = xmm_get(in.rm);
            u32 m = 0;
            for (unsigned i = 0; i < 4; ++i)
                if (v.w[i] & 0x80000000u) m |= 1u << i;
            st.gpr[in.reg] = m;
            return;
        }
        case 0x51: case 0x52: case 0x53: case 0x58: case 0x59: case 0x5C: case 0x5D: case 0x5E: case 0x5F: {
            FOp fop;
            switch (op) {
                case 0x51: fop = FOp::Sqrt; break;
                case 0x52: fop = FOp::Rsqrt; break;
                case 0x53: fop = FOp::Rcp; break;
                case 0x58: fop = FOp::Add; break;
                case 0x59: fop = FOp::Mul; break;
                case 0x5C: fop = FOp::Sub; break;
                case 0x5D: fop = FOp::Min; break;
                case 0x5E: fop = FOp::Div; break;
                default: fop = FOp::Max; break;
            }
            V128 a = xmm_get(d);
            if (scalar) {
                const u32 b = xmm_src32();
                a.w[0] = lane(fop, a.w[0], b, flags);
            } else {
                const V128 b = xmm_src128(true);
                for (unsigned i = 0; i < 4; ++i) a.w[i] = lane(fop, a.w[i], b.w[i], flags);
            }
            report(flags);
            xmm_set(d, a);
            return;
        }
        case 0x54: case 0x55: case 0x56: case 0x57: {  // ANDPS ANDNPS ORPS XORPS
            V128 a = xmm_get(d);
            const V128 b = xmm_src128(true);
            for (unsigned i = 0; i < 4; ++i)
                a.w[i] = op == 0x54 ? (a.w[i] & b.w[i]) : op == 0x55 ? (~a.w[i] & b.w[i]) : op == 0x56 ? (a.w[i] | b.w[i]) : (a.w[i] ^ b.w[i]);
            xmm_set(d, a);
            return;
        }
        case 0xC2: {  // CMPPS / CMPSS xmm, xmm/m, imm8
            const unsigned pred = in.imm & 7;
            if ((in.imm & 0xFF) > 7) raise_ud();
            V128 a = xmm_get(d);
            if (scalar) {
                const u32 b = xmm_src32();
                a.w[0] = compare_pred(a.w[0], b, pred, flags) ? 0xFFFFFFFFu : 0u;
            } else {
                const V128 b = xmm_src128(true);
                for (unsigned i = 0; i < 4; ++i) a.w[i] = compare_pred(a.w[i], b.w[i], pred, flags) ? 0xFFFFFFFFu : 0u;
            }
            report(flags);
            xmm_set(d, a);
            return;
        }
        case 0xC6: {  // SHUFPS xmm, xmm/m128, imm8
            const V128 a = xmm_get(d), b = xmm_src128(true);
            V128 r;
            r.w[0] = a.w[in.imm & 3];
            r.w[1] = a.w[(in.imm >> 2) & 3];
            r.w[2] = b.w[(in.imm >> 4) & 3];
            r.w[3] = b.w[(in.imm >> 6) & 3];
            xmm_set(d, r);
            return;
        }
        default: break;
    }
    raise_ud();
}

void Simd::run() {
    if (in.prefix_66 || in.rep == 2) raise_ud();  // SSE2 encodings: not on a Pentium III
    const u8 op = in.opcode;
    const bool mmx_map = (op >= 0x60 && op <= 0x7F && op != 0x7A && op != 0x7B && op != 0x7C && op != 0x7D) || op == 0xC4 || op == 0xC5 ||
                         (op >= 0xD1 && op <= 0xFE && op != 0xD4 && op != 0xD6 && op != 0xE6 && op != 0xF4 && op != 0xFB);
    // CVTPI2PS/CVTPS2PI and the MMX extensions with F3 are SSE forms; the rest of the MMX map ignores F3.
    if (mmx_map) {
        run_mmx();
        return;
    }
    run_sse();
}

void Cpu::exec_simd(const Insn& in) {
    Simd unit(*this, in);
    unit.run();
}

}  // namespace mv::p6
