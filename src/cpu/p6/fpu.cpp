// The x87 unit: the register stack, tags, control and status words, every D8-DF encoding, and the save-area
// formats (FXSAVE/FXRSTOR, FNSAVE/FRSTOR, FNSTENV/FLDENV). The arithmetic runs on the host's x87 under the
// guest's precision and rounding control (bit-exact for the basic operations on an x86 host; the transcendental
// instructions are the host's own, which differ from a Pentium III's in the last bits). Written from SDM Vol. 1
// chapter 8 and the Vol. 2 instruction pages.
#include <cmath>
#include <cstring>

#include "cpu/p6/cpu.h"

namespace mv::p6 {

namespace {

// Status word bits.
// Status word: exceptions in bits 0-5, SF 6, ES 7, C0 8, C1 9, C2 10, TOP 11-13, C3 14, B 15.
constexpr u16 kSwIE = 1u << 0, kSwDE = 1u << 1, kSwZE = 1u << 2, kSwOE = 1u << 3, kSwUE = 1u << 4, kSwPE = 1u << 5, kSwSF = 1u << 6,
              kSwES = 1u << 7, kSwC0 = 1u << 8, kSwC1 = 1u << 9, kSwC2 = 1u << 10, kSwC3 = 1u << 14, kSwB = 1u << 15;
constexpr u16 kSwExceptions = 0x3F;
constexpr u16 kSwCond = kSwC0 | kSwC1 | kSwC2 | kSwC3;

enum Tag : u8 { kTagValid = 0, kTagZero = 1, kTagSpecial = 2, kTagEmpty = 3 };

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define MV_HOST_X87 1
#else
#define MV_HOST_X87 0
#endif

// The host x87 under the guest's control word: precision and rounding from the guest, every exception masked
// on the host; the host's exception flags are read back into the guest's status word.
class HostFpu {
public:
    explicit HostFpu(u16 guest_cw) {
#if MV_HOST_X87
        asm volatile("fnstcw %0" : "=m"(saved_));
        const u16 cw = static_cast<u16>(0x3F | (guest_cw & 0x0F00));
        asm volatile("fnclex\n\tfldcw %0" : : "m"(cw) : "memory");
#else
        (void)guest_cw;
#endif
    }
    // The host's exception flags since construction (IE DE ZE OE UE PE in the low six bits).
    u16 flags() {
#if MV_HOST_X87
        u16 sw;
        asm volatile("fnstsw %0" : "=m"(sw) : : "memory");
        return static_cast<u16>(sw & kSwExceptions);
#else
        return 0;
#endif
    }
    // The host's condition codes (for FPREM, FXAM-like instructions run on the host).
    u16 status() {
#if MV_HOST_X87
        u16 sw;
        asm volatile("fnstsw %0" : "=m"(sw) : : "memory");
        return sw;
#else
        return 0;
#endif
    }
    ~HostFpu() {
#if MV_HOST_X87
        asm volatile("fnclex\n\tfldcw %0" : : "m"(saved_) : "memory");
#endif
    }

private:
    u16 saved_{0x37F};
};

struct F80 {
    u8 bytes[10];
};

inline long double from_f80(const F80& f) {
    long double v = 0;
    std::memcpy(&v, f.bytes, 10);
    return v;
}
inline F80 to_f80(long double v) {
    F80 f{};
    std::memcpy(f.bytes, &v, 10);
    return f;
}

// Classification by the bits of the extended format (the sign is separate).
struct Class {
    bool nan, inf, zero, denormal, unsupported, negative, snan;
};
Class classify(long double v) {
    F80 f = to_f80(v);
    const u16 se = static_cast<u16>(f.bytes[8] | (f.bytes[9] << 8));
    const u16 exp = se & 0x7FFF;
    u64 mant;
    std::memcpy(&mant, f.bytes, 8);
    Class c{};
    c.negative = (se & 0x8000) != 0;
    const bool integer_bit = (mant >> 63) != 0;
    if (exp == 0x7FFF) {
        if (!integer_bit) c.unsupported = true;
        else if ((mant & 0x7FFFFFFFFFFFFFFFull) == 0) c.inf = true;
        else {
            c.nan = true;
            c.snan = ((mant >> 62) & 1) == 0;
        }
    } else if (exp == 0) {
        if (mant == 0) c.zero = true;
        else c.denormal = true;  // pseudo-denormals (integer bit set) count as denormal operands too
    } else if (!integer_bit) {
        c.unsupported = true;
    }
    return c;
}

Tag tag_of(long double v) {
    const Class c = classify(v);
    if (c.zero) return kTagZero;
    if (c.nan || c.inf || c.denormal || c.unsupported) return kTagSpecial;
    return kTagValid;
}

long double indefinite() {
    F80 f{};
    f.bytes[7] = 0xC0;
    f.bytes[9] = 0xFF;
    f.bytes[8] = 0xFF;
    return from_f80(f);
}

// A quiet NaN made from a signalling one.
long double quieten(long double v) {
    F80 f = to_f80(v);
    f.bytes[7] |= 0x40;
    return from_f80(f);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

struct X87 {
    Cpu& cpu;
    CpuState& st;
    FpuState& f;
    Memory& mem;
    const Insn& in;
    u16 cw_mask;  // the guest's exception mask bits

    X87(Cpu& c, const Insn& i) : cpu(c), st(c.state()), f(c.state().fpu), mem(c.memory()), in(i), cw_mask(static_cast<u16>(f.fcw & 0x3F)) {}

    unsigned top() const { return f.top(); }
    unsigned phys(unsigned i) const { return (f.top() + i) & 7; }
    Tag tag(unsigned i) const { return static_cast<Tag>((f.ftw >> (2 * phys(i))) & 3); }
    void set_tag(unsigned i, Tag t) {
        const unsigned p = phys(i);
        f.ftw = static_cast<u16>((f.ftw & ~(3u << (2 * p))) | (static_cast<unsigned>(t) << (2 * p)));
    }
    bool empty(unsigned i) const { return tag(i) == kTagEmpty; }
    long double get(unsigned i) const { return f.st[phys(i)]; }
    void set(unsigned i, long double v) {
        f.st[phys(i)] = v;
        set_tag(i, tag_of(v));
    }
    void set_cc(u16 bits) { f.fsw = static_cast<u16>((f.fsw & ~kSwCond) | (bits & kSwCond)); }
    void clear_c1() { f.fsw = static_cast<u16>(f.fsw & ~kSwC1); }

    // Raise exception flags; an unmasked one arms ES/B (the #MF comes at the next x87 instruction).
    void raise_flags(u16 bits) {
        f.fsw |= bits & kSwExceptions;
        if (bits & kSwExceptions & static_cast<u16>(~cw_mask)) f.fsw |= kSwES | kSwB;
    }
    bool masked(u16 bit) const { return (cw_mask & bit) != 0; }

    // Stack underflow on an empty source: IE + SF, C1 = 0; masked gives the indefinite.
    bool underflow() {
        raise_flags(kSwIE | kSwSF);
        clear_c1();
        return masked(kSwIE);
    }
    void push(long double v) {
        clear_c1();  // C1 reports only a stack overflow here
        const unsigned t = (top() - 1) & 7;
        if (((f.ftw >> (2 * t)) & 3) != kTagEmpty) {
            // Stack overflow: IE + SF, C1 = 1; masked pushes the indefinite.
            raise_flags(kSwIE | kSwSF);
            f.fsw |= kSwC1;
            if (!masked(kSwIE)) return;
            v = indefinite();
        }
        f.set_top(t);
        f.st[t] = v;
        f.ftw = static_cast<u16>((f.ftw & ~(3u << (2 * t))) | (static_cast<unsigned>(tag_of(v)) << (2 * t)));
    }
    void pop() {
        set_tag(0, kTagEmpty);
        f.set_top((top() + 1) & 7);
    }

    // Operand checks: a signalling NaN or an unsupported operand raises IE; a denormal raises DE.
    bool check_operands(long double a, long double b, bool with_b, bool ordered_compare = false) {
        const Class ca = classify(a), cb = classify(b);
        u16 flags = 0;
        if (ca.snan || ca.unsupported || (with_b && (cb.snan || cb.unsupported))) flags |= kSwIE;
        if (ordered_compare && (ca.nan || (with_b && cb.nan))) flags |= kSwIE;
        if (ca.denormal || (with_b && cb.denormal)) flags |= kSwDE;
        if (flags) {
            raise_flags(flags);
            if ((flags & kSwIE) && !masked(kSwIE)) return false;
            if ((flags & kSwDE) && !masked(kSwDE)) return false;
        }
        return true;
    }

    // The result of an operation with a NaN among the operands (masked invalid): the NaN with the larger
    // significand, quietened; the indefinite for an unsupported operand.
    bool nan_result(long double a, long double b, bool with_b, long double& out) {
        const Class ca = classify(a), cb = classify(b);
        if (ca.unsupported || (with_b && cb.unsupported)) {
            out = indefinite();
            return true;
        }
        if (ca.nan && with_b && cb.nan) {
            F80 fa = to_f80(a), fb = to_f80(b);
            u64 ma, mb;
            std::memcpy(&ma, fa.bytes, 8);
            std::memcpy(&mb, fb.bytes, 8);
            out = quieten((ma & 0x7FFFFFFFFFFFFFFFull) >= (mb & 0x7FFFFFFFFFFFFFFFull) ? a : b);
            return true;
        }
        if (ca.nan) {
            out = quieten(a);
            return true;
        }
        if (with_b && cb.nan) {
            out = quieten(b);
            return true;
        }
        return false;
    }

    enum class Arith { Add, Sub, SubR, Mul, Div, DivR };

    // dst = dst op src with the host under the guest's control word; the host's flags join the status word.
    void arith(Arith op, unsigned dst, long double a, long double b) {
        if (!check_operands(a, b, true)) return;
        long double r = 0;
        if (!nan_result(a, b, true, r)) {
            HostFpu host(f.fcw);
            volatile long double va = a, vb = b;
            switch (op) {
                case Arith::Add: r = va + vb; break;
                case Arith::Sub: r = va - vb; break;
                case Arith::SubR: r = vb - va; break;
                case Arith::Mul: r = va * vb; break;
                case Arith::Div: r = va / vb; break;
                case Arith::DivR: r = vb / va; break;
            }
            volatile long double vr = r;
            const u16 status = host.status();
            const u16 flags = static_cast<u16>(status & kSwExceptions);
            r = vr;
            raise_flags(flags);
            // A masked zero divide gives the infinity, a masked overflow the infinity or max per rounding (the
            // host did that); an unmasked one leaves the destination alone.
            if (flags & static_cast<u16>(~cw_mask) & (kSwZE | kSwOE | kSwUE | kSwPE | kSwIE)) return;
            set(dst, r);
            set_c1_from(status);
            return;
        }
        set(dst, r);
        clear_c1();
    }
    // C1 reports a round-up on an inexact result: the host's own C1 after the operation.
    void set_c1_from(u16 host_status) { f.fsw = static_cast<u16>((f.fsw & ~kSwC1) | (host_status & kSwC1)); }

    // ---- memory operands ----
    u32 mem_addr(unsigned size, Access access) { return cpu.linear(in.mem_seg, cpu.effective_address(in), size, access); }
    long double load_m32() {
        const u32 bits = mem.read(mem_addr(4, Access::Read), 4);
        float v;
        std::memcpy(&v, &bits, 4);
        return static_cast<long double>(v);
    }
    long double load_m64() {
        const u32 lin = mem_addr(8, Access::Read);
        u64 bits = mem.read(lin, 4) | (static_cast<u64>(mem.read(lin + 4, 4)) << 32);
        double v;
        std::memcpy(&v, &bits, 8);
        return static_cast<long double>(v);
    }
    long double load_m80() {
        F80 raw{};
        mem.read_bytes(mem_addr(10, Access::Read), raw.bytes, 10);
        return from_f80(raw);
    }
    i64 load_int(unsigned size) {
        const u32 lin = mem_addr(size, Access::Read);
        if (size == 2) return static_cast<i16>(mem.read(lin, 2));
        if (size == 4) return static_cast<i32>(mem.read(lin, 4));
        return static_cast<i64>(mem.read(lin, 4) | (static_cast<u64>(mem.read(lin + 4, 4)) << 32));
    }
    void store_m32(long double v) {
        const u32 lin = mem_addr(4, Access::Write);
        mem.probe_write(lin, 4);
        float out;
        {
            HostFpu host(f.fcw);
            volatile long double vv = v;
            volatile float vo = static_cast<float>(vv);
            const u16 status = host.status();
            raise_flags(static_cast<u16>(status & kSwExceptions));
            set_c1_from(status);
            out = vo;
        }
        u32 bits;
        std::memcpy(&bits, &out, 4);
        mem.write(lin, 4, bits);
    }
    void store_m64(long double v) {
        const u32 lin = mem_addr(8, Access::Write);
        mem.probe_write(lin, 8);
        double out;
        {
            HostFpu host(f.fcw);
            volatile long double vv = v;
            volatile double vo = static_cast<double>(vv);
            const u16 status = host.status();
            raise_flags(static_cast<u16>(status & kSwExceptions));
            set_c1_from(status);
            out = vo;
        }
        u64 bits;
        std::memcpy(&bits, &out, 8);
        mem.write(lin, 4, static_cast<u32>(bits));
        mem.write(lin + 4, 4, static_cast<u32>(bits >> 32));
    }
    void store_m80(long double v) {
        const u32 lin = mem_addr(10, Access::Write);
        mem.probe_write(lin, 10);
        F80 raw = to_f80(v);
        mem.write_bytes(lin, raw.bytes, 10);
        clear_c1();
    }
    // Integer store with the guest's rounding; out of range gives the integer indefinite and IE.
    void store_int(long double v, unsigned size, bool truncate) {
        const u32 lin = mem_addr(size, Access::Write);
        mem.probe_write(lin, size);
        i64 out;
        bool invalid = false;
        const Class c = classify(v);
        if (c.nan || c.inf || c.unsupported) {
            invalid = true;
        } else {
            HostFpu host(static_cast<u16>(truncate ? ((f.fcw & ~0x0C00u) | 0x0C00u) : f.fcw));
            volatile long double vv = v;
#if MV_HOST_X87
            long double x = vv;
            if (size == 2) {
                i16 o;
                asm volatile("fistps %0" : "=m"(o) : "t"(x) : "st");
                out = o;
            } else if (size == 4) {
                i32 o;
                asm volatile("fistpl %0" : "=m"(o) : "t"(x) : "st");
                out = o;
            } else {
                i64 o;
                asm volatile("fistpll %0" : "=m"(o) : "t"(x) : "st");
                out = o;
            }
            const u16 status = host.status();
            const u16 flags = static_cast<u16>(status & kSwExceptions);
            raise_flags(flags);
            set_c1_from(status);
            if (flags & kSwIE) invalid = true;
#else
            const long double r = truncate ? std::trunc(static_cast<long double>(vv)) : std::nearbyint(static_cast<long double>(vv));
            const long double lo = size == 2 ? -32768.0L : size == 4 ? -2147483648.0L : -9223372036854775808.0L;
            const long double hi = size == 2 ? 32767.0L : size == 4 ? 2147483647.0L : 9223372036854775807.0L;
            if (r < lo || r > hi) invalid = true;
            else {
                out = static_cast<i64>(r);
                if (r != static_cast<long double>(vv)) raise_flags(kSwPE);
            }
#endif
        }
        if (invalid) {
            raise_flags(kSwIE);
            if (!masked(kSwIE)) return;
            out = size == 2 ? -32768 : size == 4 ? static_cast<i64>(-2147483648LL) : static_cast<i64>(0x8000000000000000ull);
        }
        if (size == 2) mem.write(lin, 2, static_cast<u32>(out) & 0xFFFFu);
        else if (size == 4) mem.write(lin, 4, static_cast<u32>(out));
        else {
            mem.write(lin, 4, static_cast<u32>(static_cast<u64>(out)));
            mem.write(lin + 4, 4, static_cast<u32>(static_cast<u64>(out) >> 32));
        }
    }

    // ---- comparisons ----
    // C3 C2 C0 from a compare: less 001, greater 000, equal 100, unordered 111.
    void compare_cc(long double a, long double b, bool unordered_quiet) {
        const Class ca = classify(a), cb = classify(b);
        if (ca.nan || cb.nan || ca.unsupported || cb.unsupported) {
            // FCOM raises IE on any NaN; FUCOM only on a signalling one or an unsupported operand.
            const bool signal = !unordered_quiet || ca.snan || cb.snan || ca.unsupported || cb.unsupported;
            if (signal) raise_flags(kSwIE);
            if (signal && !masked(kSwIE)) return;
            set_cc(kSwC3 | kSwC2 | kSwC0);
            return;
        }
        if (ca.denormal || cb.denormal) {
            raise_flags(kSwDE);
            if (!masked(kSwDE)) return;
        }
        if (a < b) set_cc(kSwC0);
        else if (a > b) set_cc(0);
        else set_cc(kSwC3);
    }
    // ZF PF CF from a compare (FCOMI family).
    void compare_flags(long double a, long double b, bool unordered_quiet) {
        const Class ca = classify(a), cb = classify(b);
        st.materialise();
        u32 fl = st.flags & ~(fl::ZF | fl::PF | fl::CF);
        if (ca.nan || cb.nan || ca.unsupported || cb.unsupported) {
            const bool signal = !unordered_quiet || ca.snan || cb.snan || ca.unsupported || cb.unsupported;
            if (signal) raise_flags(kSwIE);
            if (signal && !masked(kSwIE)) return;
            st.flags = fl | fl::ZF | fl::PF | fl::CF;
            return;
        }
        if (a < b) fl |= fl::CF;
        else if (a == b) fl |= fl::ZF;
        st.flags = fl;
        clear_c1();
    }

    // ---- the instruction ----
    void run();
    void run_d8_dc_de(bool pop_after, bool reversed_regs);
    void run_d9();
    void run_da();
    void run_db();
    void run_dd();
    void run_df();
    void store_env(bool);
    void load_env(bool);
    void store_state();
    void load_state();
};

// ---------------------------------------------------------------------------------------------------------------

void X87::run_d8_dc_de(bool pop_after, bool dc_form) {
    const unsigned reg = in.reg;
    if (in.mem) {
        // D8: m32fp; DC: m64fp; DE: m16int; DA: m32int (handled by the caller's conversion).
        if (empty(0)) {
            if (!underflow()) return;
            set(0, indefinite());
            return;
        }
        long double b;
        if (in.opcode == 0xD8) b = load_m32();
        else if (in.opcode == 0xDC) b = load_m64();
        else if (in.opcode == 0xDE) b = static_cast<long double>(load_int(2));
        else b = static_cast<long double>(load_int(4));
        const long double a = get(0);
        switch (reg) {
            case 0: arith(Arith::Add, 0, a, b); break;
            case 1: arith(Arith::Mul, 0, a, b); break;
            case 2: compare_cc(a, b, false); break;
            case 3: compare_cc(a, b, false); pop(); break;
            case 4: arith(Arith::Sub, 0, a, b); break;
            case 5: arith(Arith::SubR, 0, a, b); break;
            case 6: arith(Arith::Div, 0, a, b); break;
            case 7: arith(Arith::DivR, 0, a, b); break;
        }
        return;
    }
    const unsigned i = in.rm;
    if (reg == 2 || reg == 3) {
        // FCOM/FCOMP ST(i) (D8), aliases on DC/DE (DE D9 = FCOMPP).
        if (in.opcode == 0xDE) {
            if (reg == 3 && i == 1) {  // FCOMPP
                if (empty(0) || empty(1)) {
                    if (!underflow()) return;
                    set_cc(kSwC3 | kSwC2 | kSwC0);
                } else {
                    compare_cc(get(0), get(1), false);
                }
                pop();
                pop();
                return;
            }
            raise_ud();
        }
        if (empty(0) || empty(i)) {
            if (!underflow()) return;
            set_cc(kSwC3 | kSwC2 | kSwC0);
        } else {
            compare_cc(get(0), get(i), false);
        }
        if (reg == 3) pop();
        return;
    }
    if (empty(0) || empty(i)) {
        if (!underflow()) return;
        set(dc_form ? i : 0, indefinite());
        if (pop_after) pop();
        return;
    }
    // D8: ST(0) = ST(0) op ST(i). DC/DE: ST(i) = ST(i) op ST(0), with SUB/DIV's reversed encodings.
    const long double s0 = get(0), si = get(i);
    if (!dc_form) {
        switch (reg) {
            case 0: arith(Arith::Add, 0, s0, si); break;
            case 1: arith(Arith::Mul, 0, s0, si); break;
            case 4: arith(Arith::Sub, 0, s0, si); break;
            case 5: arith(Arith::SubR, 0, s0, si); break;
            case 6: arith(Arith::Div, 0, s0, si); break;
            case 7: arith(Arith::DivR, 0, s0, si); break;
            default: raise_ud();
        }
    } else {
        switch (reg) {
            case 0: arith(Arith::Add, i, si, s0); break;
            case 1: arith(Arith::Mul, i, si, s0); break;
            case 4: arith(Arith::SubR, i, si, s0); break;  // DC E0+i / DE E0+i: ST(i) = ST(0) - ST(i)
            case 5: arith(Arith::Sub, i, si, s0); break;   // DC E8+i / DE E8+i: ST(i) = ST(i) - ST(0)
            case 6: arith(Arith::DivR, i, si, s0); break;  // DC F0+i / DE F0+i: ST(i) = ST(0) / ST(i)
            case 7: arith(Arith::Div, i, si, s0); break;   // DC F8+i / DE F8+i: ST(i) = ST(i) / ST(0)
            default: raise_ud();
        }
    }
    if (pop_after) pop();
}

void X87::run_d9() {
    if (in.mem) {
        switch (in.reg) {
            case 0: {  // FLD m32fp
                const long double v = load_m32();
                const Class c = classify(v);
                if (c.snan) {
                    raise_flags(kSwIE);
                    if (!masked(kSwIE)) return;
                    push(quieten(v));
                    return;
                }
                if (c.denormal) {
                    raise_flags(kSwDE);
                    if (!masked(kSwDE)) return;
                }
                push(v);
                return;
            }
            case 2: case 3: {  // FST/FSTP m32fp
                if (empty(0)) {
                    if (!underflow()) return;
                    store_m32(indefinite());
                } else {
                    store_m32(get(0));
                }
                if (in.reg == 3) pop();
                return;
            }
            case 4: {  // FLDENV
                load_env(false);
                return;
            }
            case 5: {  // FLDCW
                const u32 v = mem.read(mem_addr(2, Access::Read), 2);
                f.fcw = static_cast<u16>((v & 0x1F7Fu) | 0x40u);
                cw_mask = static_cast<u16>(f.fcw & 0x3F);
                // Unmasking a pending exception arms it.
                if (f.fsw & kSwExceptions & static_cast<u16>(~cw_mask)) f.fsw |= kSwES | kSwB;
                else f.fsw = static_cast<u16>(f.fsw & ~(kSwES | kSwB));
                return;
            }
            case 6: {  // FNSTENV
                store_env(false);
                return;
            }
            case 7: {  // FNSTCW
                const u32 lin = mem_addr(2, Access::Write);
                mem.write(lin, 2, f.fcw);
                return;
            }
            default: raise_ud();
        }
    }
    const unsigned i = in.rm;
    switch (in.modrm) {
        case 0xC0: case 0xC1: case 0xC2: case 0xC3: case 0xC4: case 0xC5: case 0xC6: case 0xC7: {  // FLD ST(i)
            if (empty(i)) {
                if (!underflow()) return;
                push(indefinite());
                return;
            }
            push(get(i));
            return;
        }
        case 0xC8: case 0xC9: case 0xCA: case 0xCB: case 0xCC: case 0xCD: case 0xCE: case 0xCF: {  // FXCH ST(i)
            if (empty(0) || empty(i)) {
                if (!underflow()) return;
                if (empty(0)) set(0, indefinite());
                if (empty(i)) set(i, indefinite());
            }
            const long double a = get(0), b = get(i);
            set(0, b);
            set(i, a);
            clear_c1();
            return;
        }
        case 0xD0: return;  // FNOP
        case 0xE0: {        // FCHS
            if (empty(0)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            set(0, -get(0));
            clear_c1();
            return;
        }
        case 0xE1: {  // FABS
            if (empty(0)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            set(0, std::fabs(get(0)));
            clear_c1();
            return;
        }
        case 0xE4: {  // FTST
            if (empty(0)) {
                if (!underflow()) return;
                set_cc(kSwC3 | kSwC2 | kSwC0);
                return;
            }
            compare_cc(get(0), 0.0L, false);
            return;
        }
        case 0xE5: {  // FXAM
            u16 cc;
            if (empty(0)) {
                cc = kSwC3 | kSwC0;
                const Class c = classify(f.st[phys(0)]);
                if (c.negative) cc |= kSwC1;
            } else {
                const Class c = classify(get(0));
                if (c.unsupported) cc = 0;
                else if (c.nan) cc = kSwC0;
                else if (c.inf) cc = kSwC2 | kSwC0;
                else if (c.zero) cc = kSwC3;
                else if (c.denormal) cc = kSwC3 | kSwC2;
                else cc = kSwC2;
                if (c.negative) cc |= kSwC1;
            }
            set_cc(cc);
            return;
        }
        case 0xE8: push(1.0L); return;                       // FLD1
        case 0xE9: push(3.32192809488736234787L); return;    // FLDL2T
        case 0xEA: push(1.44269504088896340736L); return;    // FLDL2E
        case 0xEB: push(3.14159265358979323851L); return;    // FLDPI
        case 0xEC: push(0.301029995663981195214L); return;   // FLDLG2
        case 0xED: push(0.693147180559945309417L); return;   // FLDLN2
        case 0xEE: push(0.0L); return;                       // FLDZ
        case 0xF0: {  // F2XM1
            if (empty(0)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            long double x = get(0);
            if (!check_operands(x, 0, false)) return;
            long double r;
            if (!nan_result(x, 0, false, r)) {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                asm volatile("f2xm1" : "+t"(x));
                r = x;
#else
                r = std::exp2(x) - 1.0L;
#endif
                raise_flags(host.flags());
            }
            set(0, r);
            return;
        }
        case 0xF1: case 0xF9: {  // FYL2X, FYL2XP1: ST(1) = ST(1) * log2(ST(0) [+1]); pop
            if (empty(0) || empty(1)) {
                if (!underflow()) return;
                set(1, indefinite());
                pop();
                return;
            }
            long double x = get(0), y = get(1);
            if (!check_operands(x, y, true)) return;
            long double r;
            if (!nan_result(x, y, true, r)) {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                if (in.modrm == 0xF1) asm volatile("fyl2x" : "=t"(r) : "0"(x), "u"(y) : "st(1)");
                else asm volatile("fyl2xp1" : "=t"(r) : "0"(x), "u"(y) : "st(1)");
#else
                r = in.modrm == 0xF1 ? y * std::log2(x) : y * std::log2(x + 1.0L);
#endif
                raise_flags(host.flags());
            }
            set(1, r);
            pop();
            return;
        }
        case 0xF2: {  // FPTAN: ST(0) = tan(ST(0)); push 1.0
            if (empty(0)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            long double x = get(0);
            if (!check_operands(x, 0, false)) return;
            long double r;
            bool in_range = true;
            if (!nan_result(x, 0, false, r)) {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                long double one;
                asm volatile("fptan" : "=t"(one), "=u"(r) : "0"(x));
                (void)one;
                in_range = (host.status() & kSwC2) == 0;
#else
                r = std::tan(x);
#endif
                raise_flags(host.flags());
            }
            if (!in_range) {
                f.fsw |= kSwC2;
                return;
            }
            f.fsw = static_cast<u16>(f.fsw & ~kSwC2);
            set(0, r);
            push(1.0L);
            return;
        }
        case 0xF3: {  // FPATAN: ST(1) = atan(ST(1) / ST(0)); pop
            if (empty(0) || empty(1)) {
                if (!underflow()) return;
                set(1, indefinite());
                pop();
                return;
            }
            long double x = get(0), y = get(1);
            if (!check_operands(x, y, true)) return;
            long double r;
            if (!nan_result(x, y, true, r)) {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                asm volatile("fpatan" : "=t"(r) : "0"(x), "u"(y) : "st(1)");
#else
                r = std::atan2(y, x);
#endif
                raise_flags(host.flags());
            }
            set(1, r);
            pop();
            return;
        }
        case 0xF4: {  // FXTRACT: ST(0) = significand, push exponent... (exponent in ST(1), significand in ST(0))
            if (empty(0)) {
                if (!underflow()) return;
                set(0, indefinite());
                push(indefinite());
                return;
            }
            long double x = get(0);
            if (!check_operands(x, 0, false)) return;
            long double sig, exp;
            {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                asm volatile("fxtract" : "=t"(sig), "=u"(exp) : "0"(x));
#else
                int e;
                sig = std::frexp(x, &e) * 2.0L;
                exp = static_cast<long double>(e - 1);
#endif
                raise_flags(host.flags());
            }
            set(0, exp);
            push(sig);
            return;
        }
        case 0xF5: case 0xF8: {  // FPREM1 / FPREM: ST(0) = ST(0) rem ST(1)
            if (empty(0) || empty(1)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            long double a = get(0), b = get(1);
            if (!check_operands(a, b, true)) return;
            long double r;
            u16 cc = 0;
            if (!nan_result(a, b, true, r)) {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                if (in.modrm == 0xF5) asm volatile("fprem1" : "+t"(a) : "u"(b));
                else asm volatile("fprem" : "+t"(a) : "u"(b));
                r = a;
                cc = static_cast<u16>(host.status() & kSwCond);
#else
                r = in.modrm == 0xF5 ? std::remainder(a, b) : std::fmod(a, b);
                const long double q = std::trunc(a / b);
                const u64 qi = static_cast<u64>(std::fabs(q));
                if (qi & 1) cc |= kSwC1;
                if (qi & 2) cc |= kSwC3;
                if (qi & 4) cc |= kSwC0;
#endif
                raise_flags(host.flags());
            }
            set(0, r);
            set_cc(cc);
            return;
        }
        case 0xF6: f.set_top((top() - 1) & 7); clear_c1(); return;  // FDECSTP
        case 0xF7: f.set_top((top() + 1) & 7); clear_c1(); return;  // FINCSTP
        case 0xFA: {  // FSQRT
            if (empty(0)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            long double x = get(0);
            if (!check_operands(x, 0, false)) return;
            long double r;
            if (!nan_result(x, 0, false, r)) {
                HostFpu host(f.fcw);
                volatile long double vx = x;
                volatile long double vr = std::sqrt(static_cast<long double>(vx));
                raise_flags(host.flags());
                r = vr;
            }
            set(0, r);
            return;
        }
        case 0xFB: {  // FSINCOS: ST(0) = sin, push cos
            if (empty(0)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            long double x = get(0);
            if (!check_operands(x, 0, false)) return;
            long double s, c;
            bool in_range = true;
            if (nan_result(x, 0, false, s)) {
                c = s;
            } else {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                asm volatile("fsincos" : "=t"(c), "=u"(s) : "0"(x));
                in_range = (host.status() & kSwC2) == 0;
#else
                s = std::sin(x);
                c = std::cos(x);
#endif
                raise_flags(host.flags());
            }
            if (!in_range) {
                f.fsw |= kSwC2;
                return;
            }
            f.fsw = static_cast<u16>(f.fsw & ~kSwC2);
            set(0, s);
            push(c);
            return;
        }
        case 0xFC: {  // FRNDINT
            if (empty(0)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            long double x = get(0);
            if (!check_operands(x, 0, false)) return;
            long double r;
            if (!nan_result(x, 0, false, r)) {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                asm volatile("frndint" : "+t"(x));
                r = x;
#else
                r = std::nearbyint(x);
#endif
                raise_flags(host.flags());
            }
            set(0, r);
            return;
        }
        case 0xFD: {  // FSCALE: ST(0) = ST(0) * 2^trunc(ST(1))
            if (empty(0) || empty(1)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            long double a = get(0), b = get(1);
            if (!check_operands(a, b, true)) return;
            long double r;
            if (!nan_result(a, b, true, r)) {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                asm volatile("fscale" : "+t"(a) : "u"(b));
                r = a;
#else
                r = std::ldexp(a, static_cast<int>(std::trunc(b)));
#endif
                raise_flags(host.flags());
            }
            set(0, r);
            return;
        }
        case 0xFE: case 0xFF: {  // FSIN / FCOS
            if (empty(0)) {
                if (!underflow()) return;
                set(0, indefinite());
                return;
            }
            long double x = get(0);
            if (!check_operands(x, 0, false)) return;
            long double r;
            bool in_range = true;
            if (!nan_result(x, 0, false, r)) {
                HostFpu host(f.fcw);
#if MV_HOST_X87
                if (in.modrm == 0xFE) asm volatile("fsin" : "+t"(x));
                else asm volatile("fcos" : "+t"(x));
                r = x;
                in_range = (host.status() & kSwC2) == 0;
#else
                r = in.modrm == 0xFE ? std::sin(x) : std::cos(x);
#endif
                raise_flags(host.flags());
            }
            if (!in_range) {
                f.fsw |= kSwC2;
                return;
            }
            f.fsw = static_cast<u16>(f.fsw & ~kSwC2);
            set(0, r);
            return;
        }
        default: raise_ud();
    }
}

void X87::run_da() {
    if (in.mem) {
        // FIADD/FIMUL/FICOM/FICOMP/FISUB/FISUBR/FIDIV/FIDIVR m32int
        run_d8_dc_de(false, false);
        return;
    }
    const unsigned i = in.rm;
    if (in.modrm == 0xE9) {  // FUCOMPP
        if (empty(0) || empty(1)) {
            if (!underflow()) return;
            set_cc(kSwC3 | kSwC2 | kSwC0);
        } else {
            compare_cc(get(0), get(1), true);
        }
        pop();
        pop();
        return;
    }
    if (in.modrm < 0xE0) {  // FCMOVB/E/BE/U ST(0), ST(i)
        if (empty(0) || empty(i)) {
            if (!underflow()) return;
            set(0, indefinite());
            return;
        }
        bool take = false;
        const u32 fl = st.eflags();
        switch (in.reg) {
            case 0: take = (fl & fl::CF) != 0; break;
            case 1: take = (fl & fl::ZF) != 0; break;
            case 2: take = (fl & (fl::CF | fl::ZF)) != 0; break;
            case 3: take = (fl & fl::PF) != 0; break;
        }
        if (take) set(0, get(i));
        clear_c1();
        return;
    }
    raise_ud();
}

void X87::run_db() {
    if (in.mem) {
        switch (in.reg) {
            case 0: push(static_cast<long double>(load_int(4))); clear_c1(); return;  // FILD m32int
            case 2: case 3: {  // FIST/FISTP m32int
                if (empty(0)) {
                    if (!underflow()) return;
                    store_int(indefinite(), 4, false);
                } else {
                    store_int(get(0), 4, false);
                }
                if (in.reg == 3) pop();
                return;
            }
            case 5: {  // FLD m80fp
                const long double v = load_m80();
                const Class c = classify(v);
                if (c.snan || c.unsupported) {
                    raise_flags(kSwIE);
                    if (!masked(kSwIE)) return;
                    push(c.unsupported ? indefinite() : quieten(v));
                    return;
                }
                push(v);
                return;
            }
            case 7: {  // FSTP m80fp
                if (empty(0)) {
                    if (!underflow()) return;
                    store_m80(indefinite());
                } else {
                    store_m80(get(0));
                }
                pop();
                return;
            }
            default: raise_ud();
        }
    }
    const unsigned i = in.rm;
    switch (in.modrm) {
        case 0xE2:  // FNCLEX
            f.fsw = static_cast<u16>(f.fsw & ~(kSwExceptions | kSwES | kSwSF | kSwB));
            return;
        case 0xE3:  // FNINIT
            f.fcw = 0x37F;
            f.fsw = 0;
            f.ftw = 0xFFFF;
            f.fip = f.fdp = 0;
            f.fcs = f.fds = 0;
            f.fop = 0;
            return;
        default: break;
    }
    if (in.modrm < 0xE0) {  // FCMOVNB/NE/NBE/NU
        if (empty(0) || empty(i)) {
            if (!underflow()) return;
            set(0, indefinite());
            return;
        }
        bool take = false;
        const u32 fl = st.eflags();
        switch (in.reg) {
            case 0: take = (fl & fl::CF) == 0; break;
            case 1: take = (fl & fl::ZF) == 0; break;
            case 2: take = (fl & (fl::CF | fl::ZF)) == 0; break;
            case 3: take = (fl & fl::PF) == 0; break;
        }
        if (take) set(0, get(i));
        clear_c1();
        return;
    }
    if (in.reg == 5 || in.reg == 6) {  // FUCOMI / FCOMI ST(0), ST(i)
        if (empty(0) || empty(i)) {
            if (!underflow()) return;
            st.materialise();
            st.flags |= fl::ZF | fl::PF | fl::CF;
            return;
        }
        compare_flags(get(0), get(i), in.reg == 5);
        return;
    }
    raise_ud();
}

void X87::run_dd() {
    if (in.mem) {
        switch (in.reg) {
            case 0: {  // FLD m64fp
                const long double v = load_m64();
                const Class c = classify(v);
                if (c.snan) {
                    raise_flags(kSwIE);
                    if (!masked(kSwIE)) return;
                    push(quieten(v));
                    return;
                }
                if (c.denormal) {
                    raise_flags(kSwDE);
                    if (!masked(kSwDE)) return;
                }
                push(v);
                return;
            }
            case 2: case 3: {  // FST/FSTP m64fp
                if (empty(0)) {
                    if (!underflow()) return;
                    store_m64(indefinite());
                } else {
                    store_m64(get(0));
                }
                if (in.reg == 3) pop();
                return;
            }
            case 4: load_state(); return;   // FRSTOR
            case 6: store_state(); return;  // FNSAVE
            case 7: {                       // FNSTSW m16
                mem.write(mem_addr(2, Access::Write), 2, f.fsw);
                return;
            }
            default: raise_ud();
        }
    }
    const unsigned i = in.rm;
    switch (in.reg) {
        case 0: set_tag(i, kTagEmpty); return;  // FFREE ST(i)
        case 2: case 3: {                       // FST/FSTP ST(i)
            if (empty(0)) {
                if (!underflow()) return;
                set(i, indefinite());
            } else {
                set(i, get(0));
            }
            if (in.reg == 3) pop();
            return;
        }
        case 4: case 5: {  // FUCOM/FUCOMP ST(i)
            if (empty(0) || empty(i)) {
                if (!underflow()) return;
                set_cc(kSwC3 | kSwC2 | kSwC0);
            } else {
                compare_cc(get(0), get(i), true);
            }
            if (in.reg == 5) pop();
            return;
        }
        default: raise_ud();
    }
}

void X87::run_df() {
    if (in.mem) {
        switch (in.reg) {
            case 0: push(static_cast<long double>(load_int(2))); clear_c1(); return;  // FILD m16int
            case 2: case 3: {  // FIST/FISTP m16int
                if (empty(0)) {
                    if (!underflow()) return;
                    store_int(indefinite(), 2, false);
                } else {
                    store_int(get(0), 2, false);
                }
                if (in.reg == 3) pop();
                return;
            }
            case 4: {  // FBLD m80bcd
                u8 raw[10];
                mem.read_bytes(mem_addr(10, Access::Read), raw, 10);
                long double v = 0;
                for (int d = 8; d >= 0; --d) v = v * 100.0L + static_cast<long double>((raw[d] >> 4) * 10 + (raw[d] & 0xF));
                if (raw[9] & 0x80) v = -v;
                push(v);
                return;
            }
            case 5: push(static_cast<long double>(load_int(8))); clear_c1(); return;  // FILD m64int
            case 6: {  // FBSTP m80bcd
                const u32 lin = mem_addr(10, Access::Write);
                mem.probe_write(lin, 10);
                u8 raw[10]{};
                if (empty(0)) {
                    if (!underflow()) return;
                    raw[7] = 0xC0;
                    raw[8] = raw[9] = 0xFF;
                } else {
                    long double v = get(0);
                    long double r;
                    {
                        HostFpu host(f.fcw);
#if MV_HOST_X87
                        asm volatile("frndint" : "+t"(v));
                        r = v;
#else
                        r = std::nearbyint(v);
#endif
                        raise_flags(host.flags());
                    }
                    const bool neg = r < 0;
                    if (neg) r = -r;
                    if (!(r < 1e18L)) {
                        raise_flags(kSwIE);
                        if (!masked(kSwIE)) return;
                        raw[7] = 0xC0;
                        raw[8] = raw[9] = 0xFF;
                    } else {
                        for (int d = 0; d < 9; ++d) {
                            const long double q = std::floor(r / 100.0L);
                            const unsigned rem = static_cast<unsigned>(r - q * 100.0L);
                            raw[d] = static_cast<u8>(((rem / 10) << 4) | (rem % 10));
                            r = q;
                        }
                        if (neg) raw[9] = 0x80;
                    }
                }
                mem.write_bytes(lin, raw, 10);
                pop();
                return;
            }
            case 7: {  // FISTP m64int
                if (empty(0)) {
                    if (!underflow()) return;
                    store_int(indefinite(), 8, false);
                } else {
                    store_int(get(0), 8, false);
                }
                pop();
                return;
            }
            default: raise_ud();
        }
    }
    const unsigned i = in.rm;
    if (in.modrm == 0xE0) {  // FNSTSW AX
        st.set_reg16(kEax, f.fsw);
        return;
    }
    switch (in.reg) {
        case 0: set_tag(i, kTagEmpty); pop(); return;  // FFREEP ST(i)
        case 5: case 6: {                              // FUCOMIP / FCOMIP
            if (empty(0) || empty(i)) {
                if (!underflow()) return;
                st.materialise();
                st.flags |= fl::ZF | fl::PF | fl::CF;
                pop();
                return;
            }
            compare_flags(get(0), get(i), in.reg == 5);
            pop();
            return;
        }
        default: raise_ud();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Environment and state images (FNSTENV/FLDENV, FNSAVE/FRSTOR)

void X87::store_env(bool) {
    const bool wide = in.size == 4;
    const bool pm = st.protected_mode();
    const unsigned n = wide ? 28 : 14;
    const u32 lin = mem_addr(n, Access::Write);
    mem.probe_write(lin, n);
    u8 img[28]{};
    auto put16 = [&](unsigned off, u32 v) { img[off] = static_cast<u8>(v); img[off + 1] = static_cast<u8>(v >> 8); };
    auto put32 = [&](unsigned off, u32 v) { put16(off, v & 0xFFFF); put16(off + 2, v >> 16); };
    if (wide) {
        put32(0, f.fcw);
        put32(4, f.fsw);
        put32(8, f.ftw);
        if (pm) {
            put32(12, f.fip);
            put32(16, f.fcs | (static_cast<u32>(f.fop & 0x7FF) << 16));
            put32(20, f.fdp);
            put32(24, f.fds);
        } else {
            const u32 ip = f.fcs * 16u + f.fip, dp = f.fds * 16u + f.fdp;
            put32(12, ip & 0xFFFF);
            put32(16, (f.fop & 0x7FF) | ((ip >> 16) << 12));
            put32(20, dp & 0xFFFF);
            put32(24, (dp >> 16) << 12);
        }
    } else {
        put16(0, f.fcw);
        put16(2, f.fsw);
        put16(4, f.ftw);
        if (pm) {
            put16(6, f.fip);
            put16(8, f.fcs);
            put16(10, f.fdp);
            put16(12, f.fds);
        } else {
            const u32 ip = f.fcs * 16u + f.fip, dp = f.fds * 16u + f.fdp;
            put16(6, ip & 0xFFFF);
            put16(8, (f.fop & 0x7FF) | (((ip >> 16) & 0xF) << 12));
            put16(10, dp & 0xFFFF);
            put16(12, ((dp >> 16) & 0xF) << 12);
        }
    }
    mem.write_bytes(lin, img, n);
    // FNSTENV masks every exception afterwards.
    f.fcw |= 0x3F;
    cw_mask = 0x3F;
    f.fsw = static_cast<u16>(f.fsw & ~(kSwES | kSwB));
}

void X87::load_env(bool) {
    const bool wide = in.size == 4;
    const bool pm = st.protected_mode();
    const unsigned n = wide ? 28 : 14;
    const u32 lin = mem_addr(n, Access::Read);
    u8 img[28]{};
    mem.read_bytes(lin, img, n);
    auto get16 = [&](unsigned off) { return static_cast<u32>(img[off] | (img[off + 1] << 8)); };
    auto get32 = [&](unsigned off) { return get16(off) | (get16(off + 2) << 16); };
    if (wide) {
        f.fcw = static_cast<u16>((get32(0) & 0x1F7Fu) | 0x40u);
        f.fsw = static_cast<u16>(get32(4));
        f.ftw = static_cast<u16>(get32(8));
        if (pm) {
            f.fip = get32(12);
            f.fcs = static_cast<u16>(get32(16));
            f.fop = static_cast<u16>((get32(16) >> 16) & 0x7FF);
            f.fdp = get32(20);
            f.fds = static_cast<u16>(get32(24));
        } else {
            f.fip = get32(12) & 0xFFFF;
            f.fop = static_cast<u16>(get32(16) & 0x7FF);
            f.fcs = static_cast<u16>((get32(16) >> 12) & 0xFFFF);  // the high part lands in the selector slot
            f.fdp = get32(20) & 0xFFFF;
            f.fds = static_cast<u16>((get32(24) >> 12) & 0xFFFF);
        }
    } else {
        f.fcw = static_cast<u16>((get16(0) & 0x1F7Fu) | 0x40u);
        f.fsw = static_cast<u16>(get16(2));
        f.ftw = static_cast<u16>(get16(4));
        if (pm) {
            f.fip = get16(6);
            f.fcs = static_cast<u16>(get16(8));
            f.fdp = get16(10);
            f.fds = static_cast<u16>(get16(12));
        } else {
            f.fip = get16(6);
            f.fop = static_cast<u16>(get16(8) & 0x7FF);
            f.fcs = static_cast<u16>((get16(8) >> 12) & 0xF);
            f.fdp = get16(10);
            f.fds = static_cast<u16>((get16(12) >> 12) & 0xF);
        }
    }
    cw_mask = static_cast<u16>(f.fcw & 0x3F);
    if (f.fsw & kSwExceptions & static_cast<u16>(~cw_mask)) f.fsw |= kSwES | kSwB;
    else f.fsw = static_cast<u16>(f.fsw & ~(kSwES | kSwB));
}

void X87::store_state() {
    // FNSAVE: the environment then ST(0)..ST(7) as 80-bit values; then FNINIT.
    const bool wide = in.size == 4;
    const unsigned env = wide ? 28 : 14;
    const u32 lin = mem_addr(env + 80, Access::Write);
    mem.probe_write(lin, env + 80);
    store_env(false);
    // store_env masked the exceptions; FNSAVE saves the state as it was, so restore the word for the image.
    for (unsigned i = 0; i < 8; ++i) {
        F80 raw = to_f80(f.st[phys(i)]);
        mem.write_bytes(lin + env + i * 10, raw.bytes, 10);
    }
    f.fcw = 0x37F;
    f.fsw = 0;
    f.ftw = 0xFFFF;
    f.fip = f.fdp = 0;
    f.fcs = f.fds = 0;
    f.fop = 0;
    cw_mask = 0x3F;
}

void X87::load_state() {
    const bool wide = in.size == 4;
    const unsigned env = wide ? 28 : 14;
    const u32 lin = mem_addr(env + 80, Access::Read);
    load_env(false);
    for (unsigned i = 0; i < 8; ++i) {
        F80 raw{};
        mem.read_bytes(lin + env + i * 10, raw.bytes, 10);
        f.st[phys(i)] = from_f80(raw);
    }
}

void X87::run() {
    // An armed exception is taken before the next waiting x87 instruction; the control instructions
    // (FNINIT, FNCLEX, FNSTCW, FNSTSW, FNSTENV, FNSAVE, FLDCW? no: FLDCW waits) run regardless.
    const bool control = (in.opcode == 0xDB && !in.mem && (in.modrm == 0xE2 || in.modrm == 0xE3)) ||
                         (in.opcode == 0xD9 && in.mem && (in.reg == 6 || in.reg == 7)) ||
                         (in.opcode == 0xDD && in.mem && (in.reg == 6 || in.reg == 7)) || (in.opcode == 0xDF && !in.mem && in.modrm == 0xE0);
    if (!control && (f.fsw & kSwES)) raise(kMF);
    if (!control) {
        // The last-instruction pointers: FIP/FCS, the opcode, and FDP/FDS for a memory operand.
        f.fip = cpu.last_eip();
        f.fcs = st.seg[kCs].selector;
        f.fop = static_cast<u16>(((in.opcode & 7) << 8) | in.modrm);
        if (in.mem) {
            f.fdp = cpu.effective_address(in);
            f.fds = st.seg[in.mem_seg].selector;
        }
    }
    switch (in.opcode) {
        case 0xD8: run_d8_dc_de(false, false); break;
        case 0xD9: run_d9(); break;
        case 0xDA: run_da(); break;
        case 0xDB: run_db(); break;
        case 0xDC: run_d8_dc_de(false, true); break;
        case 0xDD: run_dd(); break;
        case 0xDE: run_d8_dc_de(true, true); break;
        case 0xDF: run_df(); break;
        default: raise_ud();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The Cpu entry points

void Cpu::exec_x87(const Insn& in) {
    if (st_.cr0 & cr0::EM) raise(kNM);
    if (st_.cr0 & cr0::TS) raise(kNM);
    X87 unit(*this, in);
    unit.run();
}

namespace {
// Tags from the register contents for FXRSTOR's abridged tag word.
u16 full_tags(const FpuState& f, u8 abridged) {
    u16 ftw = 0;
    for (unsigned p = 0; p < 8; ++p) {
        const unsigned t = (abridged >> p) & 1 ? static_cast<unsigned>(tag_of(f.st[p])) : static_cast<unsigned>(kTagEmpty);
        ftw = static_cast<u16>(ftw | (t << (2 * p)));
    }
    return ftw;
}
u8 abridged_tags(u16 ftw) {
    u8 a = 0;
    for (unsigned p = 0; p < 8; ++p)
        if (((ftw >> (2 * p)) & 3) != kTagEmpty) a = static_cast<u8>(a | (1u << p));
    return a;
}
}  // namespace

void Cpu::fxsave(const Insn& in) {
    if (!(st_.cr4 & cr4::OSFXSR)) raise_ud();
    if (st_.cr0 & cr0::EM) raise_ud();
    if (st_.cr0 & cr0::TS) raise(kNM);
    const u32 ea = effective_address(in);
    if (ea & 15) raise_gp(0);
    const u32 lin = linear(in.mem_seg, ea, 512, Access::Write);
    mem_.probe_write(lin, 512);
    const FpuState& f = st_.fpu;
    u8 img[512]{};
    auto put16 = [&](unsigned off, u32 v) { img[off] = static_cast<u8>(v); img[off + 1] = static_cast<u8>(v >> 8); };
    auto put32 = [&](unsigned off, u32 v) { put16(off, v & 0xFFFF); put16(off + 2, v >> 16); };
    put16(0, f.fcw);
    put16(2, f.fsw);
    img[4] = abridged_tags(f.ftw);
    put16(6, f.fop);
    put32(8, f.fip);
    put16(12, f.fcs);
    put32(16, f.fdp);
    put16(20, f.fds);
    put32(24, f.mxcsr);
    put32(28, 0);  // MXCSR_MASK: zero on processors before the Pentium 4 (software uses FFBF)
    for (unsigned i = 0; i < 8; ++i) {
        F80 raw = to_f80(f.st[(f.top() + i) & 7]);
        std::memcpy(img + 32 + i * 16, raw.bytes, 10);
    }
    for (unsigned i = 0; i < 8; ++i) std::memcpy(img + 160 + i * 16, f.xmm[i].data(), 16);
    mem_.write_bytes(lin, img, 512);
}

void Cpu::fxrstor(const Insn& in) {
    if (!(st_.cr4 & cr4::OSFXSR)) raise_ud();
    if (st_.cr0 & cr0::EM) raise_ud();
    if (st_.cr0 & cr0::TS) raise(kNM);
    const u32 ea = effective_address(in);
    if (ea & 15) raise_gp(0);
    const u32 lin = linear(in.mem_seg, ea, 512, Access::Read);
    u8 img[512];
    mem_.read_bytes(lin, img, 512);
    auto get16 = [&](unsigned off) { return static_cast<u32>(img[off] | (img[off + 1] << 8)); };
    auto get32 = [&](unsigned off) { return get16(off) | (get16(off + 2) << 16); };
    const u32 mxcsr = get32(24);
    if (mxcsr & ~0xFFBFu) raise_gp(0);
    FpuState& f = st_.fpu;
    f.fcw = static_cast<u16>((get16(0) & 0x1F7Fu) | 0x40u);
    f.fsw = static_cast<u16>(get16(2));
    f.fop = static_cast<u16>(get16(6) & 0x7FF);
    f.fip = get32(8);
    f.fcs = static_cast<u16>(get16(12));
    f.fdp = get32(16);
    f.fds = static_cast<u16>(get16(20));
    f.mxcsr = mxcsr;
    for (unsigned i = 0; i < 8; ++i) {
        F80 raw{};
        std::memcpy(raw.bytes, img + 32 + i * 16, 10);
        f.st[(f.top() + i) & 7] = from_f80(raw);
    }
    // The abridged tags name physical registers; the full tags come from the contents.
    f.ftw = full_tags(f, img[4]);
    for (unsigned i = 0; i < 8; ++i) std::memcpy(f.xmm[i].data(), img + 160 + i * 16, 16);
}

void Cpu::ldmxcsr(const Insn& in) {
    if (!(st_.cr4 & cr4::OSFXSR)) raise_ud();
    if (st_.cr0 & cr0::EM) raise_ud();
    if (st_.cr0 & cr0::TS) raise(kNM);
    const u32 v = mem_.read(linear(in.mem_seg, effective_address(in), 4, Access::Read), 4);
    if (v & ~0xFFBFu) raise_gp(0);
    st_.fpu.mxcsr = v;
}

void Cpu::stmxcsr(const Insn& in) {
    if (!(st_.cr4 & cr4::OSFXSR)) raise_ud();
    if (st_.cr0 & cr0::EM) raise_ud();
    if (st_.cr0 & cr0::TS) raise(kNM);
    mem_.write(linear(in.mem_seg, effective_address(in), 4, Access::Write), 4, st_.fpu.mxcsr);
}

}  // namespace mv::p6
