#include "cpu/p6/decode.h"

namespace mv::p6 {

namespace {

// Operand forms, named as the SDM's opcode map abbreviations: E = ModRM r/m, G = ModRM reg, b = byte,
// v = operand size, w = word, z = word or dword, I = immediate, J = relative, Z = register in the opcode,
// O = memory offset, M = memory only, S = segment register, C/D = control/debug register, A = far pointer.
enum class F : u8 {
    None,
    EbGb, EvGv, GbEb, GvEv, ALIb, eAXIz,
    Eb, Ev, Ew,
    EbIb, EvIz, EvIbs, EvIb,
    Eb1, Ev1, EbCL, EvCL,
    Zb, Zv, ZbIb, ZvIv,
    Jb, Jz, Ap, Ib, Ibs, Iw, Iz, IwIb,
    ALOb, eAXOv, ObAL, OveAX,
    GvM, GvMp, EwSw, SwEw, GvEb, GvEw, EvGvIb, EvGvCL, GvEvIbs, GvEvIz, EwGw, GvMa,
    Mb, Mw, Md, Mq, Ms,
    RdCd, CdRd, RdDd, DdRd,
    SegOp,
    InALIb, IneAXIb, InALDX, IneAXDX, OutIbAL, OutIbeAX, OutDXAL, OutDXeAX,
    StrB, StrV,
    X87, Simd, SimdIb,
    Grp,
    Prefix,
    Escape,
};

struct Entry {
    Op op;
    F form;
    u8 arg;
};

constexpr Entry E(Op op, F form = F::None, u8 arg = 0) { return Entry{op, form, arg}; }
constexpr Entry INV = E(Op::Invalid);

enum Group : u8 { G1, G1A, G2, G3b, G3v, G4, G5, G6, G7, G8, G9, G11b, G11v, G15, G16, G12, G13, G14 };

#define ALU6(OP, base) /* 00-05 pattern */ \
    E(OP, F::EbGb), E(OP, F::EvGv), E(OP, F::GbEb), E(OP, F::GvEv), E(OP, F::ALIb), E(OP, F::eAXIz)

constexpr Entry kMap1[256] = {
    // 00
    ALU6(Op::Add, 0), E(Op::PushSeg, F::SegOp, kEs), E(Op::PopSeg, F::SegOp, kEs),
    ALU6(Op::Or, 8), E(Op::PushSeg, F::SegOp, kCs), E(Op::Invalid, F::Escape),
    // 10
    ALU6(Op::Adc, 0), E(Op::PushSeg, F::SegOp, kSs), E(Op::PopSeg, F::SegOp, kSs),
    ALU6(Op::Sbb, 0), E(Op::PushSeg, F::SegOp, kDs), E(Op::PopSeg, F::SegOp, kDs),
    // 20
    ALU6(Op::And, 0), E(Op::Invalid, F::Prefix), E(Op::Daa),
    ALU6(Op::Sub, 0), E(Op::Invalid, F::Prefix), E(Op::Das),
    // 30
    ALU6(Op::Xor, 0), E(Op::Invalid, F::Prefix), E(Op::Aaa),
    ALU6(Op::Cmp, 0), E(Op::Invalid, F::Prefix), E(Op::Aas),
    // 40
    E(Op::Inc, F::Zv), E(Op::Inc, F::Zv), E(Op::Inc, F::Zv), E(Op::Inc, F::Zv), E(Op::Inc, F::Zv), E(Op::Inc, F::Zv), E(Op::Inc, F::Zv), E(Op::Inc, F::Zv),
    E(Op::Dec, F::Zv), E(Op::Dec, F::Zv), E(Op::Dec, F::Zv), E(Op::Dec, F::Zv), E(Op::Dec, F::Zv), E(Op::Dec, F::Zv), E(Op::Dec, F::Zv), E(Op::Dec, F::Zv),
    // 50
    E(Op::Push, F::Zv), E(Op::Push, F::Zv), E(Op::Push, F::Zv), E(Op::Push, F::Zv), E(Op::Push, F::Zv), E(Op::Push, F::Zv), E(Op::Push, F::Zv), E(Op::Push, F::Zv),
    E(Op::Pop, F::Zv), E(Op::Pop, F::Zv), E(Op::Pop, F::Zv), E(Op::Pop, F::Zv), E(Op::Pop, F::Zv), E(Op::Pop, F::Zv), E(Op::Pop, F::Zv), E(Op::Pop, F::Zv),
    // 60
    E(Op::Pusha), E(Op::Popa), E(Op::Bound, F::GvMa), E(Op::Arpl, F::EwGw), E(Op::Invalid, F::Prefix), E(Op::Invalid, F::Prefix),
    E(Op::Invalid, F::Prefix), E(Op::Invalid, F::Prefix),
    E(Op::Push, F::Iz), E(Op::Imul3, F::GvEvIz), E(Op::Push, F::Ibs), E(Op::Imul3, F::GvEvIbs), E(Op::Ins, F::StrB), E(Op::Ins, F::StrV),
    E(Op::Outs, F::StrB), E(Op::Outs, F::StrV),
    // 70
    E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb),
    E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb), E(Op::Jcc, F::Jb),
    // 80
    E(Op::Invalid, F::Grp, G1), E(Op::Invalid, F::Grp, G1), E(Op::Invalid, F::Grp, G1), E(Op::Invalid, F::Grp, G1),
    E(Op::Test, F::EbGb), E(Op::Test, F::EvGv), E(Op::Xchg, F::EbGb), E(Op::Xchg, F::EvGv),
    E(Op::Mov, F::EbGb), E(Op::Mov, F::EvGv), E(Op::Mov, F::GbEb), E(Op::Mov, F::GvEv), E(Op::MovFromSeg, F::EwSw), E(Op::Lea, F::GvM),
    E(Op::MovSeg, F::SwEw), E(Op::Invalid, F::Grp, G1A),
    // 90
    E(Op::Nop), E(Op::Xchg, F::Zv), E(Op::Xchg, F::Zv), E(Op::Xchg, F::Zv), E(Op::Xchg, F::Zv), E(Op::Xchg, F::Zv), E(Op::Xchg, F::Zv), E(Op::Xchg, F::Zv),
    E(Op::Cbw), E(Op::Cwd), E(Op::CallFar, F::Ap), E(Op::Wait), E(Op::Pushf), E(Op::Popf), E(Op::Sahf), E(Op::Lahf),
    // A0
    E(Op::Mov, F::ALOb), E(Op::Mov, F::eAXOv), E(Op::Mov, F::ObAL), E(Op::Mov, F::OveAX), E(Op::Movs, F::StrB), E(Op::Movs, F::StrV),
    E(Op::Cmps, F::StrB), E(Op::Cmps, F::StrV),
    E(Op::Test, F::ALIb), E(Op::Test, F::eAXIz), E(Op::Stos, F::StrB), E(Op::Stos, F::StrV), E(Op::Lods, F::StrB), E(Op::Lods, F::StrV),
    E(Op::Scas, F::StrB), E(Op::Scas, F::StrV),
    // B0
    E(Op::Mov, F::ZbIb), E(Op::Mov, F::ZbIb), E(Op::Mov, F::ZbIb), E(Op::Mov, F::ZbIb), E(Op::Mov, F::ZbIb), E(Op::Mov, F::ZbIb), E(Op::Mov, F::ZbIb), E(Op::Mov, F::ZbIb),
    E(Op::Mov, F::ZvIv), E(Op::Mov, F::ZvIv), E(Op::Mov, F::ZvIv), E(Op::Mov, F::ZvIv), E(Op::Mov, F::ZvIv), E(Op::Mov, F::ZvIv), E(Op::Mov, F::ZvIv), E(Op::Mov, F::ZvIv),
    // C0
    E(Op::Invalid, F::Grp, G2), E(Op::Invalid, F::Grp, G2), E(Op::Ret, F::Iw), E(Op::Ret), E(Op::Les, F::GvMp), E(Op::Lds, F::GvMp),
    E(Op::Invalid, F::Grp, G11b), E(Op::Invalid, F::Grp, G11v),
    E(Op::Enter, F::IwIb), E(Op::Leave), E(Op::Retf, F::Iw), E(Op::Retf), E(Op::Int3), E(Op::Int, F::Ib), E(Op::Into), E(Op::Iret),
    // D0
    E(Op::Invalid, F::Grp, G2), E(Op::Invalid, F::Grp, G2), E(Op::Invalid, F::Grp, G2), E(Op::Invalid, F::Grp, G2),
    E(Op::Aam, F::Ib), E(Op::Aad, F::Ib), E(Op::Salc), E(Op::Xlat),
    E(Op::X87, F::X87), E(Op::X87, F::X87), E(Op::X87, F::X87), E(Op::X87, F::X87), E(Op::X87, F::X87), E(Op::X87, F::X87), E(Op::X87, F::X87), E(Op::X87, F::X87),
    // E0
    E(Op::Loopne, F::Jb), E(Op::Loope, F::Jb), E(Op::Loop, F::Jb), E(Op::Jcxz, F::Jb), E(Op::In, F::InALIb), E(Op::In, F::IneAXIb),
    E(Op::Out, F::OutIbAL), E(Op::Out, F::OutIbeAX),
    E(Op::Call, F::Jz), E(Op::Jmp, F::Jz), E(Op::JmpFar, F::Ap), E(Op::Jmp, F::Jb), E(Op::In, F::InALDX), E(Op::In, F::IneAXDX),
    E(Op::Out, F::OutDXAL), E(Op::Out, F::OutDXeAX),
    // F0
    E(Op::Invalid, F::Prefix), E(Op::Int1), E(Op::Invalid, F::Prefix), E(Op::Invalid, F::Prefix), E(Op::Hlt), E(Op::Cmc),
    E(Op::Invalid, F::Grp, G3b), E(Op::Invalid, F::Grp, G3v),
    E(Op::Clc), E(Op::Stc), E(Op::Cli), E(Op::Sti), E(Op::Cld), E(Op::Std), E(Op::Invalid, F::Grp, G4), E(Op::Invalid, F::Grp, G5),
};
#undef ALU6

#define JCC16 \
    E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), \
    E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz), E(Op::Jcc, F::Jz)
#define SETCC16 \
    E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), \
    E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb), E(Op::Setcc, F::Eb)
#define CMOV16 \
    E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), \
    E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), \
    E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv), E(Op::Cmovcc, F::GvEv)
#define SIMD8 E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd)
#define INV8 INV, INV, INV, INV, INV, INV, INV, INV

constexpr Entry kMap0F[256] = {
    // 00
    E(Op::Invalid, F::Grp, G6), E(Op::Invalid, F::Grp, G7), E(Op::Lar, F::GvEw), E(Op::Lsl, F::GvEw), INV, INV, E(Op::Clts), INV,
    E(Op::Invd), E(Op::Wbinvd), INV, E(Op::Ud2), INV, E(Op::NopEv, F::Ev), INV, INV,
    // 10
    SIMD8,
    E(Op::Invalid, F::Grp, G16), E(Op::NopEv, F::Ev), E(Op::NopEv, F::Ev), E(Op::NopEv, F::Ev), E(Op::NopEv, F::Ev), E(Op::NopEv, F::Ev), E(Op::NopEv, F::Ev), E(Op::NopEv, F::Ev),
    // 20
    E(Op::MovFromCr, F::RdCd), E(Op::MovFromDr, F::RdDd), E(Op::MovToCr, F::CdRd), E(Op::MovToDr, F::DdRd), INV, INV, INV, INV,
    SIMD8,
    // 30
    E(Op::Wrmsr), E(Op::Rdtsc), E(Op::Rdmsr), E(Op::Rdpmc), E(Op::Sysenter), E(Op::Sysexit), INV, INV,
    INV8,
    // 40
    CMOV16,
    // 50
    SIMD8, SIMD8,
    // 60
    SIMD8, SIMD8,
    // 70
    E(Op::Simd, F::SimdIb), E(Op::Invalid, F::Grp, G12), E(Op::Invalid, F::Grp, G13), E(Op::Invalid, F::Grp, G14), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd),
    E(Op::Simd, F::Simd), E(Op::Simd, F::None),
    INV, INV, INV, INV, INV, INV, E(Op::Simd, F::Simd), E(Op::Simd, F::Simd),
    // 80
    JCC16,
    // 90
    SETCC16,
    // A0
    E(Op::PushSeg, F::SegOp, kFs), E(Op::PopSeg, F::SegOp, kFs), E(Op::Cpuid), E(Op::Bt, F::EvGv), E(Op::Shld, F::EvGvIb), E(Op::Shld, F::EvGvCL), INV, INV,
    E(Op::PushSeg, F::SegOp, kGs), E(Op::PopSeg, F::SegOp, kGs), E(Op::Rsm), E(Op::Bts, F::EvGv), E(Op::Shrd, F::EvGvIb), E(Op::Shrd, F::EvGvCL),
    E(Op::Invalid, F::Grp, G15), E(Op::Imul2, F::GvEv),
    // B0
    E(Op::Cmpxchg, F::EbGb), E(Op::Cmpxchg, F::EvGv), E(Op::Lss, F::GvMp), E(Op::Btr, F::EvGv), E(Op::Lfs, F::GvMp), E(Op::Lgs, F::GvMp),
    E(Op::Movzx, F::GvEb), E(Op::Movzx, F::GvEw),
    INV, INV, E(Op::Invalid, F::Grp, G8), E(Op::Btc, F::EvGv), E(Op::Bsf, F::GvEv), E(Op::Bsr, F::GvEv), E(Op::Movsx, F::GvEb), E(Op::Movsx, F::GvEw),
    // C0
    E(Op::Xadd, F::EbGb), E(Op::Xadd, F::EvGv), E(Op::Simd, F::SimdIb), INV, E(Op::Simd, F::SimdIb), E(Op::Simd, F::SimdIb), E(Op::Simd, F::SimdIb),
    E(Op::Invalid, F::Grp, G9),
    E(Op::Bswap, F::Zv), E(Op::Bswap, F::Zv), E(Op::Bswap, F::Zv), E(Op::Bswap, F::Zv), E(Op::Bswap, F::Zv), E(Op::Bswap, F::Zv), E(Op::Bswap, F::Zv), E(Op::Bswap, F::Zv),
    // D0
    INV, E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd),
    SIMD8,
    // E0
    SIMD8, SIMD8,
    // F0
    INV, E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd), E(Op::Simd, F::Simd),
    SIMD8,
};
#undef JCC16
#undef SETCC16
#undef CMOV16
#undef SIMD8
#undef INV8

// Group members: the op, and a form that overrides the outer entry's when it is not None.
struct GroupEntry {
    Op op;
    F form;
};
constexpr GroupEntry kGroups[][8] = {
    /* G1 */ {{Op::Add, F::None}, {Op::Or, F::None}, {Op::Adc, F::None}, {Op::Sbb, F::None}, {Op::And, F::None}, {Op::Sub, F::None}, {Op::Xor, F::None}, {Op::Cmp, F::None}},
    /* G1A */ {{Op::Pop, F::Ev}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}},
    /* G2 */ {{Op::Rol, F::None}, {Op::Ror, F::None}, {Op::Rcl, F::None}, {Op::Rcr, F::None}, {Op::Shl, F::None}, {Op::Shr, F::None}, {Op::Shl, F::None}, {Op::Sar, F::None}},
    /* G3b */ {{Op::Test, F::EbIb}, {Op::Test, F::EbIb}, {Op::Not, F::Eb}, {Op::Neg, F::Eb}, {Op::Mul, F::Eb}, {Op::Imul1, F::Eb}, {Op::Div, F::Eb}, {Op::Idiv, F::Eb}},
    /* G3v */ {{Op::Test, F::EvIz}, {Op::Test, F::EvIz}, {Op::Not, F::Ev}, {Op::Neg, F::Ev}, {Op::Mul, F::Ev}, {Op::Imul1, F::Ev}, {Op::Div, F::Ev}, {Op::Idiv, F::Ev}},
    /* G4 */ {{Op::Inc, F::Eb}, {Op::Dec, F::Eb}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}},
    /* G5 */ {{Op::Inc, F::Ev}, {Op::Dec, F::Ev}, {Op::CallRm, F::Ev}, {Op::CallFarRm, F::GvMp}, {Op::JmpRm, F::Ev}, {Op::JmpFarRm, F::GvMp}, {Op::Push, F::Ev}, {Op::Invalid, F::None}},
    /* G6 */ {{Op::Sldt, F::Ew}, {Op::Str, F::Ew}, {Op::Lldt, F::Ew}, {Op::Ltr, F::Ew}, {Op::Verr, F::Ew}, {Op::Verw, F::Ew}, {Op::Invalid, F::None}, {Op::Invalid, F::None}},
    /* G7 */ {{Op::Sgdt, F::Ms}, {Op::Sidt, F::Ms}, {Op::Lgdt, F::Ms}, {Op::Lidt, F::Ms}, {Op::Smsw, F::Ew}, {Op::Invalid, F::None}, {Op::Lmsw, F::Ew}, {Op::Invlpg, F::Mb}},
    /* G8 */ {{Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Bt, F::EvIb}, {Op::Bts, F::EvIb}, {Op::Btr, F::EvIb}, {Op::Btc, F::EvIb}},
    /* G9 */ {{Op::Invalid, F::None}, {Op::Cmpxchg8b, F::Mq}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}},
    /* G11b */ {{Op::Mov, F::EbIb}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}},
    /* G11v */ {{Op::Mov, F::EvIz}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}},
    /* G15 */ {{Op::Fxsave, F::Mb}, {Op::Fxrstor, F::Mb}, {Op::Ldmxcsr, F::Md}, {Op::Stmxcsr, F::Md}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Sfence, F::None}},
    /* G16 */ {{Op::Prefetch, F::Mb}, {Op::Prefetch, F::Mb}, {Op::Prefetch, F::Mb}, {Op::Prefetch, F::Mb}, {Op::NopEv, F::Ev}, {Op::NopEv, F::Ev}, {Op::NopEv, F::Ev}, {Op::NopEv, F::Ev}},
    /* G12 */ {{Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Simd, F::SimdIb}, {Op::Invalid, F::None}, {Op::Simd, F::SimdIb}, {Op::Invalid, F::None}, {Op::Simd, F::SimdIb}, {Op::Invalid, F::None}},
    /* G13 */ {{Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Simd, F::SimdIb}, {Op::Invalid, F::None}, {Op::Simd, F::SimdIb}, {Op::Invalid, F::None}, {Op::Simd, F::SimdIb}, {Op::Invalid, F::None}},
    /* G14 */ {{Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Simd, F::SimdIb}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Invalid, F::None}, {Op::Simd, F::SimdIb}, {Op::Invalid, F::None}},
};

struct Decoder {
    CodeSource& code;
    Insn& in;
    unsigned pos{0};
    bool code32;

    u8 next() { return code.byte(pos++); }
    u16 next16() {
        const u16 lo = next();
        return static_cast<u16>(lo | (static_cast<u16>(next()) << 8));
    }
    u32 next32() {
        const u32 lo = next16();
        return lo | (static_cast<u32>(next16()) << 16);
    }
    u32 imm_z() { return in.size == 2 ? next16() : next32(); }
    u32 imm_v() { return imm_z(); }
    u32 imm_bs() { return sign_extend(next(), 1) & size_mask(in.size); }

    void modrm() {
        const u8 m = next();
        in.has_modrm = true;
        in.modrm = m;
        in.mod = static_cast<u8>(m >> 6);
        in.reg = static_cast<u8>((m >> 3) & 7);
        in.rm = static_cast<u8>(m & 7);
        if (in.mod == 3) return;
        in.mem = true;
        if (in.adsize == 4) {
            u8 base = in.rm;
            if (in.rm == 4) {
                const u8 sib = next();
                in.scale = static_cast<u8>(sib >> 6);
                const u8 index = static_cast<u8>((sib >> 3) & 7);
                in.index = index == 4 ? Insn::kNoReg : index;
                base = static_cast<u8>(sib & 7);
            }
            if (in.mod == 0 && base == 5) {
                in.base = Insn::kNoReg;
                in.disp = static_cast<i32>(next32());
            } else {
                in.base = base;
                if (in.mod == 1) in.disp = static_cast<i8>(next());
                else if (in.mod == 2) in.disp = static_cast<i32>(next32());
            }
            in.mem_seg = (in.base == kEbp || in.base == kEsp) ? kSs : kDs;
        } else {
            static constexpr u8 kBase16[8] = {kEbx, kEbx, kEbp, kEbp, kEsi, kEdi, kEbp, kEbx};
            static constexpr u8 kIndex16[8] = {kEsi, kEdi, kEsi, kEdi, Insn::kNoReg, Insn::kNoReg, Insn::kNoReg, Insn::kNoReg};
            if (in.mod == 0 && in.rm == 6) {
                in.base = Insn::kNoReg;
                in.disp = static_cast<i16>(next16());
                in.mem_seg = kDs;
            } else {
                in.base = kBase16[in.rm];
                in.index = kIndex16[in.rm];
                if (in.mod == 1) in.disp = static_cast<i8>(next());
                else if (in.mod == 2) in.disp = static_cast<i16>(next16());
                in.mem_seg = in.base == kEbp ? kSs : kDs;
            }
        }
    }
    // The r/m operand as a register or the memory operand.
    Opnd rm_opnd() const { return in.mem ? Opnd{OpndKind::Mem, 0} : Opnd{OpndKind::Reg, in.rm}; }
    Opnd reg_opnd() const { return Opnd{OpndKind::Reg, in.reg}; }
    void need_mem() const {
        if (!in.mem) raise_ud();
    }
    void moffs() {
        in.mem = true;
        in.base = in.index = Insn::kNoReg;
        in.disp = in.adsize == 2 ? static_cast<i32>(next16()) : static_cast<i32>(next32());
        in.mem_seg = kDs;
    }

    void apply(const Entry& e) {
        in.op = e.op;
        switch (e.form) {
            case F::None: break;
            case F::EbGb: in.size = 1; modrm(); in.dst = rm_opnd(); in.src = reg_opnd(); break;
            case F::EvGv: modrm(); in.dst = rm_opnd(); in.src = reg_opnd(); break;
            case F::GbEb: in.size = 1; modrm(); in.dst = reg_opnd(); in.src = rm_opnd(); break;
            case F::GvEv: modrm(); in.dst = reg_opnd(); in.src = rm_opnd(); break;
            case F::ALIb: in.size = 1; in.dst = {OpndKind::Reg, kEax}; in.src = {OpndKind::Imm, 0}; in.imm = next(); break;
            case F::eAXIz: in.dst = {OpndKind::Reg, kEax}; in.src = {OpndKind::Imm, 0}; in.imm = imm_z(); break;
            case F::Eb: in.size = 1; modrm(); in.dst = rm_opnd(); break;
            case F::Ev: modrm(); in.dst = rm_opnd(); break;
            case F::Ew: in.size = 2; modrm(); in.dst = rm_opnd(); break;
            case F::EbIb: in.size = 1; modrm(); in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = next(); break;
            case F::EvIz: modrm(); in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = imm_z(); break;
            case F::EvIbs: modrm(); in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = imm_bs(); break;
            case F::EvIb: modrm(); in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = next(); break;
            case F::Eb1: in.size = 1; modrm(); in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = 1; break;
            case F::Ev1: modrm(); in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = 1; break;
            case F::EbCL: in.size = 1; modrm(); in.dst = rm_opnd(); in.src = {OpndKind::Reg, kEcx}; break;
            case F::EvCL: modrm(); in.dst = rm_opnd(); in.src = {OpndKind::Reg, kEcx}; break;
            case F::Zb: in.size = 1; in.dst = {OpndKind::Reg, static_cast<u8>(in.opcode & 7)}; break;
            case F::Zv: in.dst = {OpndKind::Reg, static_cast<u8>(in.opcode & 7)}; break;
            case F::ZbIb: in.size = 1; in.dst = {OpndKind::Reg, static_cast<u8>(in.opcode & 7)}; in.src = {OpndKind::Imm, 0}; in.imm = next(); break;
            case F::ZvIv: in.dst = {OpndKind::Reg, static_cast<u8>(in.opcode & 7)}; in.src = {OpndKind::Imm, 0}; in.imm = imm_v(); break;
            case F::Jb: in.imm = sign_extend(next(), 1); break;
            case F::Jz: in.imm = in.size == 2 ? sign_extend(next16(), 2) : next32(); break;
            case F::Ap: in.imm = imm_z(); in.imm2 = next16(); break;
            case F::Ib: in.imm = next(); break;
            case F::Ibs: in.src = {OpndKind::Imm, 0}; in.imm = imm_bs(); break;
            case F::Iw: in.imm = next16(); break;
            case F::Iz: in.src = {OpndKind::Imm, 0}; in.imm = imm_z(); break;
            case F::IwIb: in.imm = next16(); in.imm2 = next(); break;
            case F::ALOb: in.size = 1; moffs(); in.dst = {OpndKind::Reg, kEax}; in.src = {OpndKind::Mem, 0}; break;
            case F::eAXOv: moffs(); in.dst = {OpndKind::Reg, kEax}; in.src = {OpndKind::Mem, 0}; break;
            case F::ObAL: in.size = 1; moffs(); in.dst = {OpndKind::Mem, 0}; in.src = {OpndKind::Reg, kEax}; break;
            case F::OveAX: moffs(); in.dst = {OpndKind::Mem, 0}; in.src = {OpndKind::Reg, kEax}; break;
            case F::GvM: modrm(); need_mem(); in.dst = reg_opnd(); in.src = {OpndKind::Mem, 0}; break;
            case F::GvMp: modrm(); need_mem(); in.dst = reg_opnd(); in.src = {OpndKind::Mem, 0}; in.src_size = static_cast<u8>(in.size + 2); break;
            case F::GvMa: modrm(); need_mem(); in.dst = reg_opnd(); in.src = {OpndKind::Mem, 0}; in.src_size = static_cast<u8>(in.size * 2); break;
            case F::EwSw: modrm(); if (in.reg > 5) raise_ud(); in.dst = rm_opnd(); in.src = {OpndKind::Seg, in.reg}; in.src_size = 2; break;
            case F::SwEw: modrm(); if (in.reg > 5 || in.reg == kCs) raise_ud(); in.size = 2; in.dst = {OpndKind::Seg, in.reg}; in.src = rm_opnd(); break;
            case F::GvEb: modrm(); in.dst = reg_opnd(); in.src = rm_opnd(); in.src_size = 1; break;
            case F::GvEw: modrm(); in.dst = reg_opnd(); in.src = rm_opnd(); in.src_size = 2; break;
            case F::EvGvIb: modrm(); in.dst = rm_opnd(); in.src = reg_opnd(); in.src2 = {OpndKind::Imm, 0}; in.imm = next(); break;
            case F::EvGvCL: modrm(); in.dst = rm_opnd(); in.src = reg_opnd(); in.src2 = {OpndKind::Reg, kEcx}; break;
            case F::GvEvIbs: modrm(); in.dst = reg_opnd(); in.src = rm_opnd(); in.src2 = {OpndKind::Imm, 0}; in.imm = imm_bs(); break;
            case F::GvEvIz: modrm(); in.dst = reg_opnd(); in.src = rm_opnd(); in.src2 = {OpndKind::Imm, 0}; in.imm = imm_z(); break;
            case F::EwGw: in.size = 2; modrm(); in.dst = rm_opnd(); in.src = reg_opnd(); break;
            case F::Mb: in.size = 1; modrm(); need_mem(); in.dst = {OpndKind::Mem, 0}; break;
            case F::Mw: in.size = 2; modrm(); need_mem(); in.dst = {OpndKind::Mem, 0}; break;
            case F::Md: in.size = 4; modrm(); need_mem(); in.dst = {OpndKind::Mem, 0}; break;
            case F::Mq: modrm(); need_mem(); in.dst = {OpndKind::Mem, 0}; break;
            case F::Ms: modrm(); need_mem(); in.dst = {OpndKind::Mem, 0}; break;
            case F::RdCd: modrm(); in.size = 4; in.dst = {OpndKind::Reg, in.rm}; in.src = {OpndKind::Cr, in.reg}; break;
            case F::CdRd: modrm(); in.size = 4; in.dst = {OpndKind::Cr, in.reg}; in.src = {OpndKind::Reg, in.rm}; break;
            case F::RdDd: modrm(); in.size = 4; in.dst = {OpndKind::Reg, in.rm}; in.src = {OpndKind::Dr, in.reg}; break;
            case F::DdRd: modrm(); in.size = 4; in.dst = {OpndKind::Dr, in.reg}; in.src = {OpndKind::Reg, in.rm}; break;
            case F::SegOp: in.dst = {OpndKind::Seg, e.arg}; break;
            case F::InALIb: in.size = 1; in.dst = {OpndKind::Reg, kEax}; in.imm = next(); break;
            case F::IneAXIb: in.dst = {OpndKind::Reg, kEax}; in.imm = next(); break;
            case F::InALDX: in.size = 1; in.dst = {OpndKind::Reg, kEax}; in.port_dx = true; break;
            case F::IneAXDX: in.dst = {OpndKind::Reg, kEax}; in.port_dx = true; break;
            case F::OutIbAL: in.size = 1; in.src = {OpndKind::Reg, kEax}; in.imm = next(); break;
            case F::OutIbeAX: in.src = {OpndKind::Reg, kEax}; in.imm = next(); break;
            case F::OutDXAL: in.size = 1; in.src = {OpndKind::Reg, kEax}; in.port_dx = true; break;
            case F::OutDXeAX: in.src = {OpndKind::Reg, kEax}; in.port_dx = true; break;
            case F::StrB: in.size = 1; break;
            case F::StrV: break;
            case F::X87: modrm(); break;
            case F::Simd: modrm(); in.dst = {OpndKind::Mmx, in.reg}; in.src = in.mem ? Opnd{OpndKind::Mem, 0} : Opnd{OpndKind::Mmx, in.rm}; break;
            case F::SimdIb: modrm(); in.dst = {OpndKind::Mmx, in.reg}; in.src = in.mem ? Opnd{OpndKind::Mem, 0} : Opnd{OpndKind::Mmx, in.rm}; in.imm = next(); break;
            case F::Grp: {
                modrm();
                const GroupEntry& g = kGroups[e.arg][in.reg];
                if (g.op == Op::Invalid) raise_ud();
                // The group's own form, or the outer opcode's form for the groups whose members share it.
                F form = g.form;
                if (form == F::None) {
                    switch (in.opcode) {
                        case 0x80: case 0x82: form = F::EbIb; break;
                        case 0x81: form = F::EvIz; break;
                        case 0x83: form = F::EvIbs; break;
                        case 0xC0: form = F::EbIb; break;
                        case 0xC1: form = F::EvIb; break;
                        case 0xD0: form = F::Eb1; break;
                        case 0xD1: form = F::Ev1; break;
                        case 0xD2: form = F::EbCL; break;
                        case 0xD3: form = F::EvCL; break;
                        default: form = F::None; break;
                    }
                }
                // The ModRM byte is already consumed: re-run the form's operand setup without reading it again.
                in.op = g.op;
                apply_after_modrm(form);
                break;
            }
            case F::Prefix:
            case F::Escape: raise_ud();
        }
    }

    // The same operand setup as `apply`, for group members, with the ModRM byte already read.
    void apply_after_modrm(F form) {
        switch (form) {
            case F::None: break;
            case F::Eb: in.size = 1; in.dst = rm_opnd(); break;
            case F::Ev: in.dst = rm_opnd(); break;
            case F::Ew: in.size = 2; in.dst = rm_opnd(); break;
            case F::EbIb: in.size = 1; in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = next(); break;
            case F::EvIz: in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = imm_z(); break;
            case F::EvIbs: in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = imm_bs(); break;
            case F::EvIb: in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = next(); break;
            case F::Eb1: in.size = 1; in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = 1; break;
            case F::Ev1: in.dst = rm_opnd(); in.src = {OpndKind::Imm, 0}; in.imm = 1; break;
            case F::EbCL: in.size = 1; in.dst = rm_opnd(); in.src = {OpndKind::Reg, kEcx}; break;
            case F::EvCL: in.dst = rm_opnd(); in.src = {OpndKind::Reg, kEcx}; break;
            case F::GvMp: need_mem(); in.src = {OpndKind::Mem, 0}; in.src_size = static_cast<u8>(in.size + 2); break;
            case F::Mb: in.size = 1; need_mem(); in.dst = {OpndKind::Mem, 0}; break;
            case F::Md: in.size = 4; need_mem(); in.dst = {OpndKind::Mem, 0}; break;
            case F::Mq: need_mem(); in.dst = {OpndKind::Mem, 0}; break;
            case F::Ms: need_mem(); in.dst = {OpndKind::Mem, 0}; break;
            case F::SimdIb: in.dst = {OpndKind::Mmx, in.reg}; in.src = in.mem ? Opnd{OpndKind::Mem, 0} : Opnd{OpndKind::Mmx, in.rm}; in.imm = next(); break;
            default: raise_ud();
        }
    }

    void run() {
        in.size = code32 ? 4 : 2;
        in.adsize = code32 ? 4 : 2;
        bool opsize = false, adsize = false;
        for (;;) {
            if (pos >= 15) raise_gp(0);
            const u8 b = code.byte(pos);
            bool prefix = true;
            switch (b) {
                case 0x66: opsize = true; break;
                case 0x67: adsize = true; break;
                case 0x26: in.seg = kEs; break;
                case 0x2E: in.seg = kCs; break;
                case 0x36: in.seg = kSs; break;
                case 0x3E: in.seg = kDs; break;
                case 0x64: in.seg = kFs; break;
                case 0x65: in.seg = kGs; break;
                case 0xF0: in.lock = true; break;
                case 0xF2: in.rep = 2; break;
                case 0xF3: in.rep = 1; break;
                default: prefix = false; break;
            }
            if (!prefix) break;
            ++pos;
        }
        if (opsize) in.size = code32 ? 2 : 4;
        if (adsize) in.adsize = code32 ? 2 : 4;
        in.prefix_66 = opsize;
        u8 op = next();
        if (op == 0x0F) {
            in.two_byte = true;
            op = next();
            in.opcode = op;
            const Entry& e = kMap0F[op];
            if (e.op == Op::Invalid && e.form != F::Grp) raise_ud();
            in.cc = static_cast<u8>(op & 0xF);
            apply(e);
        } else {
            in.opcode = op;
            const Entry& e = kMap1[op];
            if (e.op == Op::Invalid && e.form != F::Grp) raise_ud();
            in.cc = static_cast<u8>(op & 0xF);
            apply(e);
        }
        if (in.mem && in.seg != kSegNone) in.mem_seg = in.seg;
        if (pos > 15) raise_gp(0);
        in.len = static_cast<u8>(pos);
    }
};

}  // namespace

void decode(CodeSource& code, bool code32, Insn& out) {
    out = Insn{};
    Decoder d{code, out, 0, code32};
    d.run();
}

const char* op_name(Op op) {
    switch (op) {
#define N(x) case Op::x: return #x
        N(Invalid); N(Add); N(Or); N(Adc); N(Sbb); N(And); N(Sub); N(Xor); N(Cmp); N(Test);
        N(Mov); N(Xchg); N(Lea); N(MovSeg); N(MovFromSeg); N(Movzx); N(Movsx); N(Lds); N(Les); N(Lfs); N(Lgs); N(Lss); N(Xlat); N(Cbw); N(Cwd); N(Bswap);
        N(Inc); N(Dec); N(Not); N(Neg); N(Mul); N(Imul1); N(Div); N(Idiv); N(Imul2); N(Imul3);
        N(Rol); N(Ror); N(Rcl); N(Rcr); N(Shl); N(Shr); N(Sar); N(Shld); N(Shrd);
        N(Push); N(Pop); N(PushSeg); N(PopSeg); N(Pusha); N(Popa); N(Pushf); N(Popf); N(Enter); N(Leave);
        N(Jmp); N(Jcc); N(JmpFar); N(JmpRm); N(JmpFarRm); N(Call); N(CallFar); N(CallRm); N(CallFarRm); N(Ret); N(Retf); N(Iret); N(Loop); N(Loope); N(Loopne); N(Jcxz);
        N(Int); N(Int3); N(Into); N(Int1);
        N(Cli); N(Sti); N(Cld); N(Std); N(Clc); N(Stc); N(Cmc); N(Sahf); N(Lahf); N(Setcc); N(Cmovcc); N(Salc);
        N(Daa); N(Das); N(Aaa); N(Aas); N(Aam); N(Aad);
        N(Bt); N(Bts); N(Btr); N(Btc); N(Bsf); N(Bsr); N(Xadd); N(Cmpxchg); N(Cmpxchg8b);
        N(Movs); N(Cmps); N(Stos); N(Lods); N(Scas); N(Ins); N(Outs);
        N(In); N(Out);
        N(Hlt); N(Nop); N(Wait); N(Cpuid); N(Rdtsc); N(Rdmsr); N(Wrmsr); N(Rdpmc); N(Sysenter); N(Sysexit); N(Wbinvd); N(Invd); N(Ud2); N(Rsm); N(Bound); N(Arpl); N(Lar); N(Lsl);
        N(MovToCr); N(MovFromCr); N(MovToDr); N(MovFromDr); N(Sldt); N(Str); N(Lldt); N(Ltr); N(Verr); N(Verw); N(Sgdt); N(Sidt); N(Lgdt); N(Lidt); N(Smsw); N(Lmsw); N(Invlpg);
        N(Clts); N(Fxsave); N(Fxrstor); N(Ldmxcsr); N(Stmxcsr); N(Sfence); N(Prefetch); N(NopEv);
        N(X87); N(Simd);
#undef N
    }
    return "?";
}

}  // namespace mv::p6
