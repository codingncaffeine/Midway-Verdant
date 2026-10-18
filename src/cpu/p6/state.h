#pragma once
// The architectural state of the interpreted Pentium III: general, segment, control, debug and system registers,
// the flags with their lazy evaluation, the x87/MMX and SSE state. Everything the KVM oracle compares lives here.
#include <array>
#include <cstring>

#include "core/types.h"

namespace mv::p6 {

enum Gpr : u8 { kEax, kEcx, kEdx, kEbx, kEsp, kEbp, kEsi, kEdi };
enum SegReg : u8 { kEs, kCs, kSs, kDs, kFs, kGs };
inline constexpr u8 kSegCount = 6;
inline constexpr u8 kSegNone = 0xFF;

namespace fl {
inline constexpr u32 CF = 1u << 0, PF = 1u << 2, AF = 1u << 4, ZF = 1u << 6, SF = 1u << 7, TF = 1u << 8, IF = 1u << 9,
                     DF = 1u << 10, OF = 1u << 11, IOPL = 3u << 12, NT = 1u << 14, RF = 1u << 16, VM = 1u << 17,
                     AC = 1u << 18, VIF = 1u << 19, VIP = 1u << 20, ID = 1u << 21;
inline constexpr u32 kAlways = 1u << 1;
inline constexpr u32 kArith = CF | PF | AF | ZF | SF | OF;
// The bits software can change on a Pentium III through POPF/IRET (VM and RF are handled separately).
inline constexpr u32 kWritable = CF | PF | AF | ZF | SF | TF | IF | DF | OF | IOPL | NT | AC | ID;
}  // namespace fl

namespace cr0 {
inline constexpr u32 PE = 1u << 0, MP = 1u << 1, EM = 1u << 2, TS = 1u << 3, ET = 1u << 4, NE = 1u << 5, WP = 1u << 16,
                     AM = 1u << 18, NW = 1u << 29, CD = 1u << 30, PG = 1u << 31;
inline constexpr u32 kReset = 0x60000010u;
}  // namespace cr0

namespace cr4 {
inline constexpr u32 VME = 1u << 0, PVI = 1u << 1, TSD = 1u << 2, DE = 1u << 3, PSE = 1u << 4, PAE = 1u << 5, MCE = 1u << 6,
                     PGE = 1u << 7, PCE = 1u << 8, OSFXSR = 1u << 9, OSXMMEXCPT = 1u << 10;
inline constexpr u32 kKnown = 0x7FFu;
}  // namespace cr4

// A segment register with its cached descriptor. `ar` holds the access rights in the layout of a descriptor's
// second dword shifted down by 8 (bits 0-3 type, 4 S, 5-6 DPL, 7 P, 12 AVL, 13 L, 14 D/B, 15 G) plus bit 16
// "unusable" (a null selector was loaded, or the register was never loaded in protected mode).
struct Segment {
    u16 selector{0};
    u32 base{0};
    u32 limit{0xFFFF};  // the last valid offset, after granularity scaling
    u32 ar{0};

    static constexpr u32 kUnusable = 1u << 16;
    u8 type() const { return static_cast<u8>(ar & 0xF); }
    bool system() const { return (ar & 0x10) == 0; }
    unsigned dpl() const { return (ar >> 5) & 3; }
    bool present() const { return (ar & 0x80) != 0; }
    bool db() const { return (ar & (1u << 14)) != 0; }
    bool granularity() const { return (ar & (1u << 15)) != 0; }
    bool unusable() const { return (ar & kUnusable) != 0; }
    bool code() const { return (ar & 0x18) == 0x18; }
    bool data() const { return (ar & 0x18) == 0x10; }
    bool expand_down() const { return data() && (ar & 0x4) != 0; }
    bool writable() const { return data() && (ar & 0x2) != 0; }
    bool readable() const { return data() || (ar & 0x2) != 0; }  // data segments are always readable
    bool conforming() const { return code() && (ar & 0x4) != 0; }
    bool accessed() const { return (ar & 0x1) != 0; }
};

struct DescTable {
    u32 base{0};
    u32 limit{0xFFFF};
};

// How the arithmetic flags of the last instruction are computed when asked for.
enum class LazyOp : u8 {
    None,    // flags_ holds every bit
    Add,     // r = a + b
    Adc,     // r = a + b + c
    Sub,     // r = a - b
    Sbb,     // r = a - b - c
    Logic,   // CF = OF = AF = 0; SF/ZF/PF from r
    Result,  // SF/ZF/PF from r; CF, OF, AF given (shifts, INC/DEC, NEG, multiplies, ...)
};

// x87 and SSE state. The x87 registers are kept in physical order (TOP indexes them); MMX aliases their
// mantissas. `long double` is the 80-bit extended format on x86-64 GCC and Clang; a software float lane is the
// portability item for hosts without it.
struct FpuState {
    std::array<long double, 8> st{};
    u16 fcw{0x37F};
    u16 fsw{0};
    u16 ftw{0xFFFF};  // the full two-bit tag word (11 = empty)
    u16 fop{0};
    u32 fip{0}, fdp{0};
    u16 fcs{0}, fds{0};
    u32 mxcsr{0x1F80};
    std::array<std::array<u8, 16>, 8> xmm{};
    unsigned top() const { return (fsw >> 11) & 7; }
    void set_top(unsigned t) { fsw = static_cast<u16>((fsw & ~0x3800u) | ((t & 7u) << 11)); }
};

struct CpuState {
    std::array<u32, 8> gpr{};
    u32 eip{0xFFF0};
    // EFLAGS: every bit when lazy_op == None; otherwise the arithmetic bits come from the lazy fields.
    u32 flags{fl::kAlways};
    LazyOp lazy_op{LazyOp::None};
    u8 lazy_size{4};  // operand size in bytes of the lazy operation
    u8 lazy_c{0};     // carry in (Adc/Sbb) or the CF to report (Result)
    u8 lazy_of{0}, lazy_af{0};  // OF and AF to report (Result)
    u32 lazy_a{0}, lazy_b{0}, lazy_r{0};
    // Which arithmetic flag bits the last instruction left architecturally undefined (the oracle masks them).
    u32 undefined_flags{0};

    std::array<Segment, kSegCount> seg{};
    Segment ldtr{}, tr{};
    DescTable gdtr{}, idtr{};
    u32 cr0{cr0::kReset}, cr2{0}, cr3{0}, cr4{0};
    std::array<u32, 8> dr{};
    u64 tsc{0};
    u32 sysenter_cs{0}, sysenter_esp{0}, sysenter_eip{0};
    u64 apic_base{0};
    u64 pat{0x0007040600070406ull};
    u64 mtrr_def_type{0};
    std::array<u64, 16> mtrr_var{};    // base/mask pairs 0x200..0x20F
    std::array<u64, 11> mtrr_fixed{};  // 0x250, 0x258, 0x259, 0x268..0x26F
    u64 misc_enable{0};
    u64 bios_sign_id{0};
    FpuState fpu{};

    bool halted{false};
    // Interrupts are not taken after STI (when it set IF) and after a load of SS, for one instruction.
    bool inhibit_interrupts{false};

    bool protected_mode() const { return (cr0 & cr0::PE) != 0; }
    bool paging() const { return (cr0 & cr0::PG) != 0; }
    bool v8086() const { return (flags & fl::VM) != 0; }
    unsigned cpl() const { return protected_mode() ? (v8086() ? 3u : static_cast<unsigned>(seg[kCs].selector & 3u)) : 0u; }
    // The default operand/address size and the stack width follow the cached descriptors' D/B bits in every
    // mode (a protected-mode load survives into real mode: "big real mode"); virtual-8086 mode is 16-bit.
    bool code32() const { return !v8086() && seg[kCs].db(); }
    bool stack32() const { return !v8086() && seg[kSs].db(); }
    unsigned iopl() const { return (flags >> 12) & 3; }

    u8 reg8(unsigned r) const { return r < 4 ? static_cast<u8>(gpr[r]) : static_cast<u8>(gpr[r - 4] >> 8); }
    u16 reg16(unsigned r) const { return static_cast<u16>(gpr[r]); }
    void set_reg8(unsigned r, u8 v) {
        if (r < 4) gpr[r] = (gpr[r] & 0xFFFFFF00u) | v;
        else gpr[r - 4] = (gpr[r - 4] & 0xFFFF00FFu) | (static_cast<u32>(v) << 8);
    }
    void set_reg16(unsigned r, u16 v) { gpr[r] = (gpr[r] & 0xFFFF0000u) | v; }
    u32 reg(unsigned r, unsigned size) const { return size == 4 ? gpr[r] : size == 2 ? reg16(r) : reg8(r); }
    void set_reg(unsigned r, unsigned size, u32 v) {
        if (size == 4) gpr[r] = v;
        else if (size == 2) set_reg16(r, static_cast<u16>(v));
        else set_reg8(r, static_cast<u8>(v));
    }

    // The flags with the lazy bits materialised.
    u32 eflags() const;
    void set_eflags(u32 v) {
        lazy_op = LazyOp::None;
        flags = (v & ~fl::kAlways) | fl::kAlways;
    }
    bool flag(u32 bit) const { return (eflags() & bit) != 0; }
    void materialise() {
        if (lazy_op != LazyOp::None) set_eflags(eflags());
    }
    // Set the arithmetic flags from the last operation.
    void lazy(LazyOp op, unsigned size, u32 a, u32 b, u32 r, u8 c = 0) {
        lazy_op = op;
        lazy_size = static_cast<u8>(size);
        lazy_a = a;
        lazy_b = b;
        lazy_r = r;
        lazy_c = c;
    }
    void lazy_result(unsigned size, u32 r, bool cf, bool of, bool af) {
        lazy_op = LazyOp::Result;
        lazy_size = static_cast<u8>(size);
        lazy_r = r;
        lazy_c = cf ? 1 : 0;
        lazy_of = of ? 1 : 0;
        lazy_af = af ? 1 : 0;
    }
    bool cf() const;
};

inline u32 size_mask(unsigned size) { return size == 4 ? 0xFFFFFFFFu : size == 2 ? 0xFFFFu : 0xFFu; }
inline u32 sign_bit(unsigned size) { return size == 4 ? 0x80000000u : size == 2 ? 0x8000u : 0x80u; }
inline u32 sign_extend(u32 v, unsigned from_size) {
    if (from_size == 1) return static_cast<u32>(static_cast<i32>(static_cast<i8>(v)));
    if (from_size == 2) return static_cast<u32>(static_cast<i32>(static_cast<i16>(v)));
    return v;
}

extern const std::array<u8, 256> kParityEven;  // 1 when the byte has an even number of set bits

inline bool CpuState::cf() const {
    switch (lazy_op) {
        case LazyOp::None: return (flags & fl::CF) != 0;
        case LazyOp::Add: return (lazy_r & size_mask(lazy_size)) < (lazy_a & size_mask(lazy_size));
        case LazyOp::Adc: return lazy_c ? (lazy_r & size_mask(lazy_size)) <= (lazy_a & size_mask(lazy_size))
                                        : (lazy_r & size_mask(lazy_size)) < (lazy_a & size_mask(lazy_size));
        case LazyOp::Sub: return (lazy_a & size_mask(lazy_size)) < (lazy_b & size_mask(lazy_size));
        case LazyOp::Sbb: return lazy_c ? (lazy_a & size_mask(lazy_size)) <= (lazy_b & size_mask(lazy_size))
                                        : (lazy_a & size_mask(lazy_size)) < (lazy_b & size_mask(lazy_size));
        case LazyOp::Logic: return false;
        case LazyOp::Result: return lazy_c != 0;
    }
    return false;
}

inline u32 CpuState::eflags() const {
    if (lazy_op == LazyOp::None) return flags;
    const u32 mask = size_mask(lazy_size);
    const u32 sign = sign_bit(lazy_size);
    const u32 r = lazy_r & mask;
    u32 f = flags & ~fl::kArith;
    if (r == 0) f |= fl::ZF;
    if (r & sign) f |= fl::SF;
    if (kParityEven[r & 0xFF]) f |= fl::PF;
    if (cf()) f |= fl::CF;
    switch (lazy_op) {
        case LazyOp::Add:
        case LazyOp::Adc:
            if (((lazy_a ^ r) & (lazy_b ^ r)) & sign) f |= fl::OF;
            if ((lazy_a ^ lazy_b ^ r) & 0x10) f |= fl::AF;
            break;
        case LazyOp::Sub:
        case LazyOp::Sbb:
            if (((lazy_a ^ lazy_b) & (lazy_a ^ r)) & sign) f |= fl::OF;
            if ((lazy_a ^ lazy_b ^ r) & 0x10) f |= fl::AF;
            break;
        case LazyOp::Result:
            if (lazy_of) f |= fl::OF;
            if (lazy_af) f |= fl::AF;
            break;
        case LazyOp::Logic:
        case LazyOp::None: break;
    }
    return f;
}

// A fault raised while an instruction executes; delivered by the step loop through the IDT. `error_code` is
// pushed when `has_error` is set. Vector numbers as the architecture defines them.
struct Fault {
    u8 vector;
    bool has_error;
    u32 error_code;
};
inline constexpr u8 kDE = 0, kDB = 1, kNMI = 2, kBP = 3, kOF = 4, kBR = 5, kUD = 6, kNM = 7, kDF = 8, kTS = 10, kNP = 11, kSS = 12,
                    kGP = 13, kPF = 14, kMF = 16, kAC = 17, kMC = 18, kXM = 19;
[[noreturn]] inline void raise(u8 vector) { throw Fault{vector, false, 0}; }
[[noreturn]] inline void raise(u8 vector, u32 error_code) { throw Fault{vector, true, error_code}; }
[[noreturn]] inline void raise_ud() { throw Fault{kUD, false, 0}; }
[[noreturn]] inline void raise_gp(u32 error_code = 0) { throw Fault{kGP, true, error_code}; }

// Something the interpreter does not implement: the backend stops the run and reports it (never silently wrong).
struct Unimplemented {
    const char* what;
};

}  // namespace mv::p6
