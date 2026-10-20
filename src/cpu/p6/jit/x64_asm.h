#pragma once
// A small x86-64 assembler for the recompiler (D26): the integer forms it emits, with 32-bit displacements off a
// base register, labels with rel32 fixups. Written from the SDM Vol. 2 encoding tables (ModRM, SIB, REX).
#include <cstring>
#include <vector>

#include "core/types.h"

namespace mv::p6::x64 {

enum Reg : u8 { rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15 };

// A memory operand: [base + index * 2^scale + disp]; index none = 0xFF.
struct Mem {
    Reg base;
    i32 disp{0};
    u8 index{0xFF};
    u8 scale{0};
};
inline Mem mem(Reg base, i32 disp = 0) { return Mem{base, disp}; }
inline Mem mem(Reg base, Reg index, u8 scale, i32 disp = 0) { return Mem{base, disp, static_cast<u8>(index), scale}; }

// Condition codes (the low nibble of Jcc/SETcc/CMOVcc).
enum Cond : u8 { O, NO, B, AE, E, NE, BE, A, S, NS, P, NP, L, GE, LE, G };

struct Label {
    int id{-1};
};

class Asm {
public:
    std::vector<u8> code;

    std::size_t size() const { return code.size(); }
    void byte(u8 b) { code.push_back(b); }
    void dword(u32 v) {
        for (int i = 0; i < 4; ++i) byte(static_cast<u8>(v >> (8 * i)));
    }
    void qword(u64 v) {
        for (int i = 0; i < 8; ++i) byte(static_cast<u8>(v >> (8 * i)));
    }

    // ---- labels ----
    Label label() {
        labels_.push_back(-1);
        return Label{static_cast<int>(labels_.size() - 1)};
    }
    void bind(Label l) { labels_[static_cast<std::size_t>(l.id)] = static_cast<long>(code.size()); }
    bool bound(Label l) const { return labels_[static_cast<std::size_t>(l.id)] >= 0; }
    // Patch every rel32 now that the labels are bound; false if one was never bound.
    bool finish() {
        for (const Fixup& f : fixups_) {
            const long target = labels_[static_cast<std::size_t>(f.label)];
            if (target < 0) return false;
            const i32 rel = static_cast<i32>(target - static_cast<long>(f.at + 4));
            std::memcpy(&code[f.at], &rel, 4);
        }
        return true;
    }

    // ---- encoding core ----
    // REX: W (64-bit operand), R (ModRM.reg high), X (SIB.index high), B (ModRM.rm / SIB.base high).
    void rex(bool w, unsigned reg, unsigned index, unsigned base, bool force = false) {
        const u8 r = static_cast<u8>(0x40 | (w ? 8 : 0) | ((reg & 8) ? 4 : 0) | ((index & 8) ? 2 : 0) | ((base & 8) ? 1 : 0));
        if (r != 0x40 || force) byte(r);
    }
    void modrm_mem(unsigned reg, const Mem& m) {
        const unsigned base = m.base & 7;
        const bool sib = m.index != 0xFF || base == 4;
        unsigned mod;
        if (m.disp == 0 && base != 5) mod = 0;
        else if (m.disp >= -128 && m.disp <= 127) mod = 1;
        else mod = 2;
        byte(static_cast<u8>(mod << 6 | (reg & 7) << 3 | (sib ? 4 : base)));
        if (sib) byte(static_cast<u8>(m.scale << 6 | ((m.index == 0xFF ? 4 : m.index) & 7) << 3 | base));
        if (mod == 1) byte(static_cast<u8>(m.disp));
        else if (mod == 2) dword(static_cast<u32>(m.disp));
    }
    void modrm_reg(unsigned reg, unsigned rm) { byte(static_cast<u8>(0xC0 | (reg & 7) << 3 | (rm & 7))); }
    // An instruction with a register and a memory operand; size 1, 2, 4 or 8.
    void op_rm(unsigned size, std::initializer_list<u8> opcode, unsigned reg, const Mem& m) {
        if (size == 2) byte(0x66);
        rex(size == 8, reg, m.index == 0xFF ? 0 : m.index, m.base, size == 1 && reg >= 4 && reg < 8);
        for (u8 o : opcode) byte(o);
        modrm_mem(reg, m);
    }
    void op_rr(unsigned size, std::initializer_list<u8> opcode, unsigned reg, unsigned rm) {
        if (size == 2) byte(0x66);
        rex(size == 8, reg, 0, rm, size == 1 && ((reg >= 4 && reg < 8) || (rm >= 4 && rm < 8)));
        for (u8 o : opcode) byte(o);
        modrm_reg(reg, rm);
    }

    // ---- moves ----
    void mov(unsigned size, Reg dst, const Mem& src) { op_rm(size, {static_cast<u8>(size == 1 ? 0x8A : 0x8B)}, dst, src); }
    void mov(unsigned size, const Mem& dst, Reg src) { op_rm(size, {static_cast<u8>(size == 1 ? 0x88 : 0x89)}, src, dst); }
    void mov(unsigned size, Reg dst, Reg src) { op_rr(size, {static_cast<u8>(size == 1 ? 0x88 : 0x89)}, src, dst); }
    void mov_imm(unsigned size, const Mem& dst, u32 imm) {
        op_rm(size, {static_cast<u8>(size == 1 ? 0xC6 : 0xC7)}, 0, dst);
        if (size == 1) byte(static_cast<u8>(imm));
        else if (size == 2) {
            byte(static_cast<u8>(imm));
            byte(static_cast<u8>(imm >> 8));
        } else {
            dword(imm);
        }
    }
    void mov32(Reg dst, u32 imm) {
        rex(false, 0, 0, dst);
        byte(static_cast<u8>(0xB8 + (dst & 7)));
        dword(imm);
    }
    void mov64(Reg dst, u64 imm) {
        rex(true, 0, 0, dst);
        byte(static_cast<u8>(0xB8 + (dst & 7)));
        qword(imm);
    }
    // movzx/movsx r32, r/m8 or r/m16.
    void movzx(Reg dst, unsigned from, const Mem& src) { op_rm(4, {0x0F, static_cast<u8>(from == 1 ? 0xB6 : 0xB7)}, dst, src); }
    void movsx(Reg dst, unsigned from, const Mem& src) { op_rm(4, {0x0F, static_cast<u8>(from == 1 ? 0xBE : 0xBF)}, dst, src); }
    void movzx(Reg dst, unsigned from, Reg src) {
        rex(false, dst, 0, src, from == 1 && src >= 4 && src < 8);
        byte(0x0F);
        byte(static_cast<u8>(from == 1 ? 0xB6 : 0xB7));
        modrm_reg(dst, src);
    }
    void movsx(Reg dst, unsigned from, Reg src) {
        rex(false, dst, 0, src, from == 1 && src >= 4 && src < 8);
        byte(0x0F);
        byte(static_cast<u8>(from == 1 ? 0xBE : 0xBF));
        modrm_reg(dst, src);
    }
    void lea(unsigned size, Reg dst, const Mem& m) { op_rm(size, {0x8D}, dst, m); }

    // ---- arithmetic: /digit = 0 add, 1 or, 2 adc, 3 sbb, 4 and, 5 sub, 6 xor, 7 cmp ----
    void alu(unsigned size, unsigned digit, Reg dst, Reg src) { op_rr(size, {static_cast<u8>(digit * 8 + (size == 1 ? 0 : 1))}, src, dst); }
    void alu(unsigned size, unsigned digit, Reg dst, const Mem& src) { op_rm(size, {static_cast<u8>(digit * 8 + (size == 1 ? 2 : 3))}, dst, src); }
    void alu(unsigned size, unsigned digit, const Mem& dst, Reg src) { op_rm(size, {static_cast<u8>(digit * 8 + (size == 1 ? 0 : 1))}, src, dst); }
    void alu_imm(unsigned size, unsigned digit, Reg dst, u32 imm) {
        const i32 s = static_cast<i32>(imm);
        if (size != 1 && s >= -128 && s <= 127) {
            op_rr(size, {0x83}, digit, dst);
            byte(static_cast<u8>(imm));
            return;
        }
        op_rr(size, {static_cast<u8>(size == 1 ? 0x80 : 0x81)}, digit, dst);
        imm_sized(size, imm);
    }
    void alu_imm(unsigned size, unsigned digit, const Mem& dst, u32 imm) {
        const i32 s = static_cast<i32>(imm);
        if (size != 1 && s >= -128 && s <= 127) {
            op_rm(size, {0x83}, digit, dst);
            byte(static_cast<u8>(imm));
            return;
        }
        op_rm(size, {static_cast<u8>(size == 1 ? 0x80 : 0x81)}, digit, dst);
        imm_sized(size, imm);
    }
    void test(unsigned size, Reg a, Reg b) { op_rr(size, {static_cast<u8>(size == 1 ? 0x84 : 0x85)}, b, a); }
    void test_imm(unsigned size, Reg a, u32 imm) {
        op_rr(size, {static_cast<u8>(size == 1 ? 0xF6 : 0xF7)}, 0, a);
        imm_sized(size, imm);
    }
    void test_imm(unsigned size, const Mem& m, u32 imm) {
        op_rm(size, {static_cast<u8>(size == 1 ? 0xF6 : 0xF7)}, 0, m);
        imm_sized(size, imm);
    }
    // /2 not, /3 neg, /4 mul, /5 imul (one operand).
    void unary(unsigned size, unsigned digit, Reg r) { op_rr(size, {static_cast<u8>(size == 1 ? 0xF6 : 0xF7)}, digit, r); }
    // INC (/0) and DEC (/1): FE/FF.
    void incdec(unsigned size, bool inc, Reg r) { op_rr(size, {static_cast<u8>(size == 1 ? 0xFE : 0xFF)}, inc ? 0 : 1, r); }
    void imul(unsigned size, Reg dst, Reg src) { op_rr(size, {0x0F, 0xAF}, dst, src); }
    // Shifts: /4 shl, /5 shr, /7 sar, /0 rol, /1 ror.
    void shift_imm(unsigned size, unsigned digit, Reg r, u8 count) {
        op_rr(size, {static_cast<u8>(size == 1 ? 0xC0 : 0xC1)}, digit, r);
        byte(count);
    }
    void shift_cl(unsigned size, unsigned digit, Reg r) { op_rr(size, {static_cast<u8>(size == 1 ? 0xD2 : 0xD3)}, digit, r); }
    void setcc(Cond c, Reg r) {
        rex(false, 0, 0, r, r >= 4 && r < 8);
        byte(0x0F);
        byte(static_cast<u8>(0x90 + c));
        modrm_reg(0, r);
    }
    void cmov(unsigned size, Cond c, Reg dst, Reg src) { op_rr(size, {0x0F, static_cast<u8>(0x40 + c)}, dst, src); }
    void bt(unsigned size, const Mem& m, Reg bit) { op_rm(size, {0x0F, 0xA3}, bit, m); }
    void xchg(unsigned size, Reg a, Reg b) { op_rr(size, {static_cast<u8>(size == 1 ? 0x86 : 0x87)}, a, b); }
    // ---- SSE (xmm0-xmm5 only: the scratch registers of both ABIs) ----
    // An SSE instruction with an xmm register and a memory operand: optional prefix (0x66/0xF3/0xF2, 0 = none), 0F op.
    void sse(u8 prefix, u8 op, unsigned xreg, const Mem& m) {
        if (prefix) byte(prefix);
        rex(false, xreg, m.index == 0xFF ? 0 : m.index, m.base);
        byte(0x0F);
        byte(op);
        modrm_mem(xreg, m);
    }
    void sse_rr(u8 prefix, u8 op, unsigned xreg, unsigned xrm) {
        if (prefix) byte(prefix);
        rex(false, xreg, 0, xrm);
        byte(0x0F);
        byte(op);
        modrm_reg(xreg, xrm);
    }
    // MOVMSKPS r32, xmm.
    void movmskps(Reg dst, unsigned xrm) { sse_rr(0, 0x50, dst, xrm); }
    void ldmxcsr(const Mem& m) {
        rex(false, 0, m.index == 0xFF ? 0 : m.index, m.base);
        byte(0x0F);
        byte(0xAE);
        modrm_mem(2, m);
    }
    void stmxcsr(const Mem& m) {
        rex(false, 0, m.index == 0xFF ? 0 : m.index, m.base);
        byte(0x0F);
        byte(0xAE);
        modrm_mem(3, m);
    }

    // ---- x87 ----
    // An x87 instruction with a memory operand: escape byte and ModRM reg digit.
    void x87_mem(u8 escape, unsigned digit, const Mem& m) { op_rm(4, {escape}, digit, m); }
    void x87_raw(u8 escape, u8 modrm) {
        byte(escape);
        byte(modrm);
    }
    void fnstsw_ax() { x87_raw(0xDF, 0xE0); }
    void fnclex() { x87_raw(0xDB, 0xE2); }

    void bswap(Reg r) {
        rex(false, 0, 0, r);
        byte(0x0F);
        byte(static_cast<u8>(0xC8 + (r & 7)));
    }

    // ---- stack and control ----
    void push(Reg r) {
        rex(false, 0, 0, r);
        byte(static_cast<u8>(0x50 + (r & 7)));
    }
    void pop(Reg r) {
        rex(false, 0, 0, r);
        byte(static_cast<u8>(0x58 + (r & 7)));
    }
    void ret() { byte(0xC3); }
    void call(Reg r) {
        rex(false, 0, 0, r);
        byte(0xFF);
        modrm_reg(2, r);
    }
    // Call an absolute address through RAX.
    void call_abs(const void* fn) {
        mov64(rax, reinterpret_cast<u64>(fn));
        call(rax);
    }
    void jmp(Reg r) {
        rex(false, 0, 0, r);
        byte(0xFF);
        modrm_reg(4, r);
    }
    void jmp(Label l) {
        byte(0xE9);
        fixup(l);
    }
    void jcc(Cond c, Label l) {
        byte(0x0F);
        byte(static_cast<u8>(0x80 + c));
        fixup(l);
    }
    // lea r64, [rip + label]
    void lea_rip(Reg dst, Label l) {
        rex(true, dst, 0, 0);
        byte(0x8D);
        byte(static_cast<u8>(0x05 | (dst & 7) << 3));
        fixup(l);
    }

private:
    struct Fixup {
        std::size_t at;
        int label;
    };
    void fixup(Label l) {
        fixups_.push_back({code.size(), l.id});
        dword(0);
    }
    void imm_sized(unsigned size, u32 imm) {
        if (size == 1) byte(static_cast<u8>(imm));
        else if (size == 2) {
            byte(static_cast<u8>(imm));
            byte(static_cast<u8>(imm >> 8));
        } else {
            dword(imm);
        }
    }
    std::vector<long> labels_;
    std::vector<Fixup> fixups_;
};

}  // namespace mv::p6::x64
