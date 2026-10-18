#pragma once
// The IA-32 instruction decoder for the Pentium III: prefixes, both opcode maps, the ModRM groups, 16- and
// 32-bit addressing. The output is an operation and operand descriptors; the executor resolves registers and
// the memory operand's address from them. Written from the opcode maps of the SDM Vol. 2 appendix A.
#include <cstddef>

#include "core/types.h"
#include "cpu/p6/state.h"

namespace mv::p6 {

enum class Op : u16 {
    Invalid,
    // Two-operand arithmetic and logic (lazy flags).
    Add, Or, Adc, Sbb, And, Sub, Xor, Cmp, Test,
    // Moves and exchanges.
    Mov, Xchg, Lea, MovSeg, MovFromSeg, Movzx, Movsx, Lds, Les, Lfs, Lgs, Lss, Xlat, Cbw, Cwd, Bswap,
    // One-operand arithmetic.
    Inc, Dec, Not, Neg, Mul, Imul1, Div, Idiv, Imul2, Imul3,
    // Shifts and rotates (the count is `src`: an immediate, CL or 1).
    Rol, Ror, Rcl, Rcr, Shl, Shr, Sar, Shld, Shrd,
    // Stack.
    Push, Pop, PushSeg, PopSeg, Pusha, Popa, Pushf, Popf, Enter, Leave,
    // Control transfers.
    Jmp, Jcc, JmpFar, JmpRm, JmpFarRm, Call, CallFar, CallRm, CallFarRm, Ret, Retf, Iret, Loop, Loope, Loopne, Jcxz,
    Int, Int3, Into, Int1,
    // Flags and conditions.
    Cli, Sti, Cld, Std, Clc, Stc, Cmc, Sahf, Lahf, Setcc, Cmovcc, Salc,
    // Decimal.
    Daa, Das, Aaa, Aas, Aam, Aad,
    // Bits.
    Bt, Bts, Btr, Btc, Bsf, Bsr, Xadd, Cmpxchg, Cmpxchg8b,
    // Strings (the size is in `size`; REP in `rep`).
    Movs, Cmps, Stos, Lods, Scas, Ins, Outs,
    // I/O: `port_dx` says the port is in DX, else the immediate.
    In, Out,
    // System.
    Hlt, Nop, Wait, Cpuid, Rdtsc, Rdmsr, Wrmsr, Rdpmc, Sysenter, Sysexit, Wbinvd, Invd, Ud2, Rsm, Bound, Arpl, Lar, Lsl,
    MovToCr, MovFromCr, MovToDr, MovFromDr, Sldt, Str, Lldt, Ltr, Verr, Verw, Sgdt, Sidt, Lgdt, Lidt, Smsw, Lmsw, Invlpg,
    Clts, Fxsave, Fxrstor, Ldmxcsr, Stmxcsr, Sfence, Prefetch, NopEv,
    // The x87 escape (opcode D8-DF, decoded by the FPU unit from `opcode` and `modrm`).
    X87,
    // MMX and SSE (decoded by the SIMD unit from `opcode` and the prefix).
    Simd,
};

enum class OpndKind : u8 { None, Reg, Mem, Imm, Seg, Cr, Dr, Mmx, Xmm };

struct Opnd {
    OpndKind kind{OpndKind::None};
    u8 reg{0};  // register number (GPR by `size`, segment, control, debug, MMX or XMM register)
};

struct Insn {
    Op op{Op::Invalid};
    u8 len{0};       // instruction length in bytes
    u8 size{4};      // operand size in bytes: 1, 2 or 4 (for the main operands)
    u8 src_size{4};  // the source's size when it differs (MOVZX/MOVSX, LAR/LSL's Ew, the far pointer forms)
    u8 adsize{4};    // address size: 2 or 4
    u8 seg{kSegNone};  // segment override, or none
    u8 rep{0};         // 0 none, 1 = F3 (REP/REPE), 2 = F2 (REPNE)
    bool lock{false};
    bool port_dx{false};
    bool prefix_66{false};  // the operand-size prefix was present (an SSE2 form on the SIMD maps)
    u8 cc{0};        // condition code for Jcc/Setcc/Cmovcc (the low nibble of the opcode)
    u8 opcode{0};    // the last opcode byte (for X87, Simd, the groups)
    bool two_byte{false};
    bool has_modrm{false};
    u8 modrm{0}, mod{0}, reg{0}, rm{0};
    // The memory operand, when any operand is memory: effective address = base + index * scale + disp,
    // masked to the address size, in segment `mem_seg` (the override applied already).
    bool mem{false};
    u8 base{kNoReg}, index{kNoReg}, scale{0};
    u8 mem_seg{kDs};
    i32 disp{0};
    u32 imm{0};   // the immediate (already sign- or zero-extended as the form requires)
    u32 imm2{0};  // a second immediate: ENTER's level, a far pointer's selector
    Opnd dst{}, src{}, src2{};

    static constexpr u8 kNoReg = 0xFF;
};

// Code bytes come through this interface so the decoder never reads past what it needs (the byte after an
// instruction may be on an unmapped page). Throws the fault the fetch raised.
class CodeSource {
public:
    virtual ~CodeSource() = default;
    virtual u8 byte(unsigned offset) = 0;
};

// Decode one instruction. `code32` gives the default operand and address size. Throws Fault (#UD on an invalid
// encoding, #GP(0) past 15 bytes) or whatever the code source throws.
void decode(CodeSource& code, bool code32, Insn& out);

// For the disassembler and the oracle's reports.
const char* op_name(Op op);

}  // namespace mv::p6
