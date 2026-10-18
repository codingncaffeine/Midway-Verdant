// The interpreted Pentium III: decoder forms, the flag rules on asymmetric cases, strings, real-mode
// interrupts, the switch to protected mode, paging with its accessed/dirty bits and a page fault through the
// IDT. The KVM oracle checks the core against the hardware on a whole boot; these keep the rules pinned
// where no hypervisor runs.
#include <gtest/gtest.h>

#include <cstring>
#include <initializer_list>

#include "core/bus.h"
#include "cpu/p6/cpu.h"
#include "cpu/p6/decode.h"

using namespace mv;
using namespace mv::p6;

namespace {

struct Rig {
    Bus bus{4u << 20};
    Cpu cpu{bus};
    Rig() {
        cpu.reset(0x68A);
        // Real mode with CS at zero and 32-bit code and stack descriptors (big real mode): code at 0x1000, a
        // stack at 0x8000. Tests that want 16-bit code clear the D bit themselves.
        cpu.state().seg[kCs] = Segment{0, 0, 0xFFFFFFFFu, 0xC09B};
        cpu.state().seg[kSs] = Segment{0, 0, 0xFFFFFFFFu, 0xC093};
        cpu.state().seg[kDs] = cpu.state().seg[kEs] = Segment{0, 0, 0xFFFFFFFFu, 0xC093};
        cpu.state().eip = 0x1000;
        cpu.state().gpr[kEsp] = 0x8000;
    }
    void code(std::initializer_list<u8> bytes, u32 at = 0x1000) {
        u32 a = at;
        for (u8 b : bytes) bus.ram()[a++] = b;
    }
    void run(unsigned n) {
        for (unsigned i = 0; i < n; ++i) cpu.step();
    }
    CpuState& s() { return cpu.state(); }
    u32 flags() { return cpu.state().eflags() & fl::kArith; }
};

struct Bytes : CodeSource {
    std::initializer_list<u8> b;
    explicit Bytes(std::initializer_list<u8> bytes) : b(bytes) {}
    u8 byte(unsigned offset) override { return *(b.begin() + offset); }
};

Insn decode_bytes(std::initializer_list<u8> bytes, bool code32 = true) {
    Bytes src(bytes);
    Insn in;
    decode(src, code32, in);
    return in;
}

}  // namespace

// ---- the decoder ----

TEST(P6Decode, AluFormsAndLengths) {
    Insn in = decode_bytes({0x01, 0xD8});  // add eax, ebx
    EXPECT_EQ(in.op, Op::Add);
    EXPECT_EQ(in.len, 2);
    EXPECT_EQ(in.size, 4);
    EXPECT_EQ(in.dst.kind, OpndKind::Reg);
    EXPECT_EQ(in.dst.reg, kEax);
    EXPECT_EQ(in.src.reg, kEbx);

    in = decode_bytes({0x66, 0x83, 0xC0, 0xFF});  // add ax, -1 (sign-extended imm8)
    EXPECT_EQ(in.op, Op::Add);
    EXPECT_EQ(in.size, 2);
    EXPECT_EQ(in.imm, 0xFFFFu);
    EXPECT_EQ(in.len, 4);

    in = decode_bytes({0x8B, 0x44, 0x91, 0x10});  // mov eax, [ecx + edx*4 + 0x10]
    EXPECT_EQ(in.op, Op::Mov);
    EXPECT_TRUE(in.mem);
    EXPECT_EQ(in.base, kEcx);
    EXPECT_EQ(in.index, kEdx);
    EXPECT_EQ(in.scale, 2);
    EXPECT_EQ(in.disp, 0x10);
    EXPECT_EQ(in.mem_seg, kDs);

    in = decode_bytes({0x8B, 0x45, 0xFC});  // mov eax, [ebp - 4]: SS by default
    EXPECT_EQ(in.mem_seg, kSs);
    EXPECT_EQ(in.disp, -4);

    in = decode_bytes({0x26, 0x8B, 0x05, 0x78, 0x56, 0x34, 0x12});  // mov eax, es:[0x12345678]
    EXPECT_EQ(in.mem_seg, kEs);
    EXPECT_EQ(in.base, Insn::kNoReg);
    EXPECT_EQ(in.disp, 0x12345678);
    EXPECT_EQ(in.len, 7);
}

TEST(P6Decode, SixteenBitAddressing) {
    Insn in = decode_bytes({0x8B, 0x46, 0x02}, false);  // mov ax, [bp + 2]
    EXPECT_EQ(in.size, 2);
    EXPECT_EQ(in.adsize, 2);
    EXPECT_EQ(in.base, kEbp);
    EXPECT_EQ(in.index, Insn::kNoReg);
    EXPECT_EQ(in.mem_seg, kSs);
    EXPECT_EQ(in.disp, 2);
    in = decode_bytes({0x8B, 0x06, 0x34, 0x12}, false);  // mov ax, [0x1234]
    EXPECT_EQ(in.base, Insn::kNoReg);
    EXPECT_EQ(in.disp, 0x1234);
    EXPECT_EQ(in.mem_seg, kDs);
    in = decode_bytes({0x66, 0x8B, 0x00}, false);  // mov eax, [bx + si] with the operand-size prefix
    EXPECT_EQ(in.size, 4);
    EXPECT_EQ(in.base, kEbx);
    EXPECT_EQ(in.index, kEsi);
}

TEST(P6Decode, GroupsAndTwoByteMap) {
    Insn in = decode_bytes({0xF7, 0xF1});  // div ecx
    EXPECT_EQ(in.op, Op::Div);
    in = decode_bytes({0xF7, 0xC1, 0x01, 0x00, 0x00, 0x00});  // test ecx, 1
    EXPECT_EQ(in.op, Op::Test);
    EXPECT_EQ(in.len, 6);
    in = decode_bytes({0xC1, 0xE0, 0x03});  // shl eax, 3
    EXPECT_EQ(in.op, Op::Shl);
    EXPECT_EQ(in.imm, 3u);
    in = decode_bytes({0x0F, 0xAE, 0xF8});  // sfence
    EXPECT_EQ(in.op, Op::Sfence);
    EXPECT_FALSE(in.mem);
    in = decode_bytes({0x0F, 0xAE, 0x09});  // fxrstor [ecx]
    EXPECT_EQ(in.op, Op::Fxrstor);
    in = decode_bytes({0x0F, 0x01, 0x16, 0x00, 0x20, 0x00, 0x00});  // lgdt [0x2000]
    EXPECT_EQ(in.op, Op::Lgdt);
    in = decode_bytes({0x0F, 0x22, 0xC0});  // mov cr0, eax
    EXPECT_EQ(in.op, Op::MovToCr);
    EXPECT_EQ(in.dst.reg, 0);
    in = decode_bytes({0x0F, 0xB6, 0xC1});  // movzx eax, cl
    EXPECT_EQ(in.op, Op::Movzx);
    EXPECT_EQ(in.src_size, 1);
    in = decode_bytes({0xF3, 0x0F, 0x10, 0x01});  // movss xmm0, [ecx]
    EXPECT_EQ(in.op, Op::Simd);
    EXPECT_EQ(in.rep, 1);
    in = decode_bytes({0xD9, 0xEE});  // fldz
    EXPECT_EQ(in.op, Op::X87);
    EXPECT_EQ(in.modrm, 0xEE);
}

TEST(P6Decode, TooLongRaisesGp) {
    // Sixteen prefixes before the opcode: more than 15 bytes.
    EXPECT_THROW(decode_bytes({0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x90}), Fault);
}

// ---- flags on asymmetric cases ----

TEST(P6Flags, AddSubCarryOverflow) {
    Rig r;
    // mov eax, 0x7FFFFFFF; add eax, 1
    r.code({0xB8, 0xFF, 0xFF, 0xFF, 0x7F, 0x83, 0xC0, 0x01});
    r.run(2);
    EXPECT_EQ(r.s().gpr[kEax], 0x80000000u);
    EXPECT_EQ(r.flags(), fl::OF | fl::SF | fl::AF | fl::PF);
    // mov al, 0; sub al, 1 -> 0xFF: CF, SF, AF, PF
    r.s().eip = 0x1000;
    r.code({0xB0, 0x00, 0x2C, 0x01});
    r.run(2);
    EXPECT_EQ(r.s().reg8(kEax), 0xFF);
    EXPECT_EQ(r.flags(), fl::CF | fl::SF | fl::AF | fl::PF);
    // 16-bit: mov ax, 0xFFFF; add ax, 1 -> 0: CF ZF AF PF, no OF
    r.s().eip = 0x1000;
    r.code({0x66, 0xB8, 0xFF, 0xFF, 0x66, 0x83, 0xC0, 0x01});
    r.run(2);
    EXPECT_EQ(r.s().reg16(kEax), 0u);
    EXPECT_EQ(r.flags(), fl::CF | fl::ZF | fl::AF | fl::PF);
}

TEST(P6Flags, AdcSbbWithCarryIn) {
    Rig r;
    // stc; mov eax, 0xFFFFFFFF; adc eax, 0 -> 0 with CF (carry from the carry-in alone)
    r.code({0xF9, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0x83, 0xD0, 0x00});
    r.run(3);
    EXPECT_EQ(r.s().gpr[kEax], 0u);
    EXPECT_TRUE(r.flags() & fl::CF);
    EXPECT_TRUE(r.flags() & fl::ZF);
    // stc; mov eax, 5; sbb eax, 5 -> -1: CF set (5 <= 5 with borrow), SF
    r.s().eip = 0x1000;
    r.code({0xF9, 0xB8, 0x05, 0x00, 0x00, 0x00, 0x83, 0xD8, 0x05});
    r.run(3);
    EXPECT_EQ(r.s().gpr[kEax], 0xFFFFFFFFu);
    EXPECT_TRUE(r.flags() & fl::CF);
    EXPECT_TRUE(r.flags() & fl::SF);
    EXPECT_FALSE(r.flags() & fl::OF);
}

TEST(P6Flags, IncDecKeepCarry) {
    Rig r;
    // stc; mov eax, 0xFFFFFFFF; inc eax -> 0, ZF, CF still set, no OF
    r.code({0xF9, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0x40});
    r.run(3);
    EXPECT_EQ(r.s().gpr[kEax], 0u);
    EXPECT_EQ(r.flags(), fl::CF | fl::ZF | fl::AF | fl::PF);
    // clc; mov eax, 0x80000000; dec eax -> 0x7FFFFFFF: OF, no CF
    r.s().eip = 0x1000;
    r.code({0xF8, 0xB8, 0x00, 0x00, 0x00, 0x80, 0x48});
    r.run(3);
    EXPECT_EQ(r.s().gpr[kEax], 0x7FFFFFFFu);
    EXPECT_EQ(r.flags(), fl::OF | fl::AF | fl::PF);
}

TEST(P6Flags, ShiftsAndRotates) {
    Rig r;
    // mov eax, 0x80000001; shl eax, 1 -> 2, CF=1, OF = msb(result) ^ CF = 1
    r.code({0xB8, 0x01, 0x00, 0x00, 0x80, 0xD1, 0xE0});
    r.run(2);
    EXPECT_EQ(r.s().gpr[kEax], 2u);
    EXPECT_TRUE(r.flags() & fl::CF);
    EXPECT_TRUE(r.flags() & fl::OF);
    // mov al, 0x81; sar al, 1 -> 0xC0, CF=1, OF=0
    r.s().eip = 0x1000;
    r.code({0xB0, 0x81, 0xD0, 0xF8});
    r.run(2);
    EXPECT_EQ(r.s().reg8(kEax), 0xC0);
    EXPECT_TRUE(r.flags() & fl::CF);
    EXPECT_FALSE(r.flags() & fl::OF);
    EXPECT_TRUE(r.flags() & fl::SF);
    // mov ax, 0x8001; ror ax, 1 -> 0xC000: CF = new msb = 1, OF = msb ^ msb-1 = 0
    r.s().eip = 0x1000;
    r.code({0x66, 0xB8, 0x01, 0x80, 0x66, 0xD1, 0xC8});
    r.run(2);
    EXPECT_EQ(r.s().reg16(kEax), 0xC000u);
    EXPECT_TRUE(r.flags() & fl::CF);
    EXPECT_FALSE(r.flags() & fl::OF);
    // stc; mov eax, 0x40000000; rcl eax, 2 -> (0x40000000 with CF=1) rotated twice through carry = 0x00000003? no:
    // bits: 0100...0 | c=1 -> rotate left by 2 over 33 bits: result 0x00000002 with CF = 1? compute: 33-bit value
    // c:a = 1_0100..0; rotl 1 -> 0_1000..01 (CF=0, a=0x80000001); rotl 1 -> 1_0000..010 (CF=1, a=0x00000002).
    r.s().eip = 0x1000;
    r.code({0xF9, 0xB8, 0x00, 0x00, 0x00, 0x40, 0xC1, 0xD0, 0x02});
    r.run(3);
    EXPECT_EQ(r.s().gpr[kEax], 0x00000002u);
    EXPECT_TRUE(r.flags() & fl::CF);
    // shift by zero leaves the flags alone: stc; mov eax, 1; shl eax, 0 (via cl = 0)
    r.s().eip = 0x1000;
    r.code({0xF9, 0xB8, 0x01, 0x00, 0x00, 0x00, 0xB1, 0x00, 0xD3, 0xE0});
    r.run(4);
    EXPECT_EQ(r.s().gpr[kEax], 1u);
    EXPECT_TRUE(r.flags() & fl::CF);
    // shld eax, ebx, 4: eax = 0x12345678, ebx = 0x9ABCDEF0 -> 0x23456789, CF = bit 28 of eax = 1
    r.s().eip = 0x1000;
    r.code({0xB8, 0x78, 0x56, 0x34, 0x12, 0xBB, 0xF0, 0xDE, 0xBC, 0x9A, 0x0F, 0xA4, 0xD8, 0x04});
    r.run(3);
    EXPECT_EQ(r.s().gpr[kEax], 0x23456789u);
    EXPECT_TRUE(r.flags() & fl::CF);
}

TEST(P6Flags, MultiplyDivide) {
    Rig r;
    // mov eax, 0x10000; mul eax -> edx:eax = 1:0, CF = OF = 1
    r.code({0xB8, 0x00, 0x00, 0x01, 0x00, 0xF7, 0xE0});
    r.run(2);
    EXPECT_EQ(r.s().gpr[kEax], 0u);
    EXPECT_EQ(r.s().gpr[kEdx], 1u);
    EXPECT_TRUE(r.flags() & fl::CF);
    EXPECT_TRUE(r.flags() & fl::OF);
    // imul eax, eax, -2 with eax = 3 -> -6, no overflow
    r.s().eip = 0x1000;
    r.code({0xB8, 0x03, 0x00, 0x00, 0x00, 0x6B, 0xC0, 0xFE});
    r.run(2);
    EXPECT_EQ(r.s().gpr[kEax], 0xFFFFFFFAu);
    EXPECT_FALSE(r.flags() & fl::CF);
    // mov edx, 0; mov eax, 100; mov ecx, 7; div ecx -> 14 rem 2
    r.s().eip = 0x1000;
    r.code({0xBA, 0x00, 0x00, 0x00, 0x00, 0xB8, 0x64, 0x00, 0x00, 0x00, 0xB9, 0x07, 0x00, 0x00, 0x00, 0xF7, 0xF1});
    r.run(4);
    EXPECT_EQ(r.s().gpr[kEax], 14u);
    EXPECT_EQ(r.s().gpr[kEdx], 2u);
    // idiv by -1 of INT_MIN -> #DE delivered through the IVT: vector 0 at 0:0 -> CS:IP = IVT[0]
    r.bus.write_ram(0, "\x00\x20\x00\x00", 4);  // vector 0 -> 0000:2000
    r.s().eip = 0x1000;
    r.code({0xBA, 0xFF, 0xFF, 0xFF, 0xFF, 0xB8, 0x00, 0x00, 0x00, 0x80, 0xB9, 0xFF, 0xFF, 0xFF, 0xFF, 0xF7, 0xF9});
    r.run(4);
    EXPECT_EQ(r.s().eip, 0x2000u);
    EXPECT_EQ(r.s().gpr[kEax], 0x80000000u);  // untouched by the faulting divide
}

TEST(P6Flags, LogicClearsCarryAndBsf) {
    Rig r;
    // stc; mov eax, 0xF0; and eax, 0x0F -> 0: ZF PF, CF cleared
    r.code({0xF9, 0xB8, 0xF0, 0x00, 0x00, 0x00, 0x83, 0xE0, 0x0F});
    r.run(3);
    EXPECT_EQ(r.s().gpr[kEax], 0u);
    EXPECT_EQ(r.flags(), fl::ZF | fl::PF);
    // bsf eax, ebx with ebx = 0x40 -> 6; with ebx = 0 -> ZF, eax unchanged
    r.s().eip = 0x1000;
    r.code({0xBB, 0x40, 0x00, 0x00, 0x00, 0x0F, 0xBC, 0xC3, 0x31, 0xDB, 0x0F, 0xBC, 0xC3});
    r.run(2);
    EXPECT_EQ(r.s().gpr[kEax], 6u);
    EXPECT_FALSE(r.flags() & fl::ZF);
    r.run(2);
    EXPECT_EQ(r.s().gpr[kEax], 6u);
    EXPECT_TRUE(r.flags() & fl::ZF);
}

// ---- strings, stack, control flow ----

TEST(P6Exec, RepMovsOverlapAndDirection) {
    Rig r;
    for (u32 i = 0; i < 8; ++i) r.bus.ram()[0x3000 + i] = static_cast<u8>(i);
    // Forward copy one byte up (overlapping): the hardware's byte-by-byte semantics smear the first byte.
    // mov esi, 0x3000; mov edi, 0x3001; mov ecx, 7; cld; rep movsb
    r.code({0xBE, 0x00, 0x30, 0x00, 0x00, 0xBF, 0x01, 0x30, 0x00, 0x00, 0xB9, 0x07, 0x00, 0x00, 0x00, 0xFC, 0xF3, 0xA4});
    r.run(5);
    for (u32 i = 0; i < 8; ++i) EXPECT_EQ(r.bus.ram()[0x3000 + i], 0) << i;
    EXPECT_EQ(r.s().gpr[kEcx], 0u);
    EXPECT_EQ(r.s().gpr[kEsi], 0x3007u);
    // Backward with DF: std; mov esi, 0x3107; mov edi, 0x3207; mov ecx, 4; rep movsw (16-bit elements, 4 of them)
    for (u32 i = 0; i < 8; ++i) r.bus.ram()[0x3100 + i] = static_cast<u8>(0x10 + i);
    r.s().eip = 0x1000;
    r.code({0xFD, 0xBE, 0x06, 0x31, 0x00, 0x00, 0xBF, 0x06, 0x32, 0x00, 0x00, 0xB9, 0x04, 0x00, 0x00, 0x00, 0xF3, 0x66, 0xA5});
    r.run(5);
    for (u32 i = 0; i < 8; ++i) EXPECT_EQ(r.bus.ram()[0x3200 + i], 0x10 + i) << i;
    EXPECT_EQ(r.s().gpr[kEsi], 0x30FEu);
}

TEST(P6Exec, RepeScasStopsOnMismatch) {
    Rig r;
    std::memcpy(r.bus.ram() + 0x3000, "aaab", 4);
    // mov edi, 0x3000; mov al, 'a'; mov ecx, 4; cld; repe scasb -> stops at 'b': ecx = 0, edi = 0x3004, ZF = 0
    r.code({0xBF, 0x00, 0x30, 0x00, 0x00, 0xB0, 'a', 0xB9, 0x04, 0x00, 0x00, 0x00, 0xFC, 0xF3, 0xAE});
    r.run(5);
    EXPECT_EQ(r.s().gpr[kEdi], 0x3004u);
    EXPECT_EQ(r.s().gpr[kEcx], 0u);
    EXPECT_FALSE(r.flags() & fl::ZF);
}

TEST(P6Exec, PushPopCallRetAndPushfPopf) {
    Rig r;
    // In real mode with a 16-bit stack: push eax (32-bit operand in 16-bit code needs 66); here 16-bit pushes.
    // mov ax, 0x1234; push ax; pop bx; call 0x1020; (at 0x1020: ret)
    r.code({0xB8, 0x34, 0x12, 0x50, 0x5B, 0xE8, 0x18, 0x00, 0x90}, 0x1000);
    r.code({0xC3}, 0x1020);
    r.s().seg[kCs].ar = 0x9B;  // 16-bit code segment (D = 0)
    r.run(3);
    EXPECT_EQ(r.s().reg16(kEbx), 0x1234u);
    EXPECT_EQ(r.s().reg16(kEsp), 0x8000u);
    r.run(1);  // call
    EXPECT_EQ(r.s().eip, 0x1020u);
    EXPECT_EQ(r.s().reg16(kEsp), 0x7FFEu);
    u16 ret = 0;
    r.bus.read_ram(0x7FFE, &ret, 2);
    EXPECT_EQ(ret, 0x1008u);
    r.run(1);  // ret
    EXPECT_EQ(r.s().eip, 0x1008u);
    // stc; pushf; pop ax -> bit 0 set and bit 1 always set; and ax, ~1; push ax; popf -> CF clear
    r.s().eip = 0x1000;
    r.code({0xF9, 0x9C, 0x58, 0x83, 0xE0, 0xFE, 0x50, 0x9D});
    r.run(3);
    EXPECT_EQ(r.s().reg16(kEax) & 3u, 3u);
    r.run(3);
    EXPECT_FALSE(r.flags() & fl::CF);
}

TEST(P6Exec, RealModeInterruptAndIret) {
    Rig r;
    // IVT[0x21] -> 0000:3000; the handler: inc bx; iret
    r.bus.write_ram(0x21 * 4, "\x00\x30\x00\x00", 4);
    r.code({0x66, 0x43, 0xCF}, 0x3000);
    // sti; int 0x21; nop
    r.code({0xFB, 0xCD, 0x21, 0x90});
    r.s().seg[kCs].ar = 0x9B;
    r.run(2);
    EXPECT_EQ(r.s().eip, 0x3000u);
    EXPECT_FALSE(r.s().flags & fl::IF);  // INT clears IF in real mode
    EXPECT_EQ(r.s().reg16(kEsp), 0x7FFAu);
    r.run(2);
    EXPECT_EQ(r.s().eip, 0x1003u);
    EXPECT_TRUE(r.s().flags & fl::IF);
    EXPECT_EQ(r.s().reg16(kEbx), 1u);
    EXPECT_EQ(r.s().reg16(kEsp), 0x8000u);
}

TEST(P6Exec, ExternalInterruptHonoursIfAndTheStiShadow) {
    Rig r;
    int pending = 0x30;
    r.cpu.set_interrupt_source([&] { return pending; }, [&] {
        const int v = pending;
        pending = -1;
        return v;
    });
    r.bus.write_ram(0x30 * 4, "\x00\x30\x00\x00", 4);
    r.code({0xCF}, 0x3000);  // iret
    r.code({0xFB, 0x90, 0x90});  // sti; nop; nop
    r.s().seg[kCs].ar = 0x9B;
    EXPECT_FALSE(r.cpu.take_interrupt());  // IF clear
    r.run(1);                               // sti
    EXPECT_FALSE(r.cpu.take_interrupt());  // the shadow after STI
    r.run(1);                               // nop
    EXPECT_TRUE(r.cpu.take_interrupt());
    EXPECT_EQ(r.s().eip, 0x3000u);
    EXPECT_EQ(pending, -1);
}

// ---- protected mode and paging ----

namespace {
void write_descriptor(Bus& bus, u32 at, u32 base, u32 limit, u8 access, u8 flags) {
    const u32 lo = (limit & 0xFFFF) | (base << 16);
    const u32 hi = ((base >> 16) & 0xFF) | (static_cast<u32>(access) << 8) | (limit & 0xF0000) | (static_cast<u32>(flags) << 20) | (base & 0xFF000000);
    bus.write_ram(at, &lo, 4);
    bus.write_ram(at + 4, &hi, 4);
}
void write_gate(Bus& bus, u32 at, u16 selector, u32 offset, u8 type) {
    const u32 lo = (offset & 0xFFFF) | (static_cast<u32>(selector) << 16);
    const u32 hi = (offset & 0xFFFF0000) | (static_cast<u32>(type | 0x80) << 8);
    bus.write_ram(at, &lo, 4);
    bus.write_ram(at + 4, &hi, 4);
}
}  // namespace

TEST(P6System, EnterProtectedModeThroughGdtAndFarJump) {
    Rig r;
    // GDT at 0x2000: null, code 0x08 (flat 32-bit), data 0x10 (flat)
    write_descriptor(r.bus, 0x2000, 0, 0, 0, 0);
    write_descriptor(r.bus, 0x2008, 0, 0xFFFFF, 0x9A, 0xC);
    write_descriptor(r.bus, 0x2010, 0, 0xFFFFF, 0x92, 0xC);
    r.bus.write_ram(0x2100, "\x17\x00\x00\x20\x00\x00", 6);  // limit 0x17, base 0x2000
    // 16-bit real-mode code: lgdt [0x2100]; mov eax, cr0; or al, 1; mov cr0, eax; jmp 0x08:0x1100
    r.code({0x0F, 0x01, 0x16, 0x00, 0x21, 0x0F, 0x20, 0xC0, 0x0C, 0x01, 0x0F, 0x22, 0xC0, 0x66, 0xEA, 0x00, 0x11, 0x00, 0x00, 0x08, 0x00});
    // 32-bit code at 0x1100: mov ax, 0x10; mov ds, ax; mov dword [0x3000], 0xCAFEBABE
    r.code({0x66, 0xB8, 0x10, 0x00, 0x8E, 0xD8, 0xC7, 0x05, 0x00, 0x30, 0x00, 0x00, 0xBE, 0xBA, 0xFE, 0xCA}, 0x1100);
    r.s().seg[kCs].ar = 0x9B;
    r.run(5);
    EXPECT_TRUE(r.s().protected_mode());
    EXPECT_EQ(r.s().seg[kCs].selector, 0x08);
    EXPECT_TRUE(r.s().seg[kCs].db());
    EXPECT_EQ(r.s().seg[kCs].limit, 0xFFFFFFFFu);
    EXPECT_EQ(r.s().eip, 0x1100u);
    r.run(3);
    u32 v = 0;
    r.bus.read_ram(0x3000, &v, 4);
    EXPECT_EQ(v, 0xCAFEBABEu);
    EXPECT_EQ(r.s().seg[kDs].selector, 0x10);
    EXPECT_TRUE(r.s().seg[kDs].writable());
    // The descriptor's accessed bit was set in memory.
    u32 hi = 0;
    r.bus.read_ram(0x2014, &hi, 4);
    EXPECT_TRUE(hi & 0x100);
}

TEST(P6System, PagingTranslatesSetsBitsAndFaults) {
    Rig r;
    // Flat protected mode directly.
    r.s().cr0 |= cr0::PE;
    r.s().seg[kCs] = Segment{0x08, 0, 0xFFFFFFFFu, 0xC09B};
    r.s().seg[kDs] = r.s().seg[kEs] = r.s().seg[kSs] = Segment{0x10, 0, 0xFFFFFFFFu, 0xC093};
    r.s().gpr[kEsp] = 0x9000;
    // Page directory at 0x100000, page table 0 at 0x101000: identity-map the first 4 MiB except page 0x7000
    // (not present); map linear 0x00400000 (the second table slot) as a 4 MiB page onto physical 0.
    const u32 pd = 0x100000, pt = 0x101000;
    u32 e = pt | 3;
    r.bus.write_ram(pd, &e, 4);
    e = 0 | 3 | 0x80;  // PS, present, writable
    r.bus.write_ram(pd + 4, &e, 4);
    for (u32 i = 0; i < 1024; ++i) {
        e = (i << 12) | 3;
        if (i == 7) e = 0;
        r.bus.write_ram(pt + 4 * i, &e, 4);
    }
    // IDT at 0x200000 with a #PF gate to 0x5000.
    r.s().idtr = DescTable{0x200000, 0xFF};
    write_gate(r.bus, 0x200000 + 14 * 8, 0x08, 0x5000, 0x0E);
    write_descriptor(r.bus, 0x2000, 0, 0, 0, 0);
    write_descriptor(r.bus, 0x2008, 0, 0xFFFFF, 0x9A, 0xC);
    write_descriptor(r.bus, 0x2010, 0, 0xFFFFF, 0x92, 0xC);
    r.s().gdtr = DescTable{0x2000, 0x17};
    r.s().cr4 |= cr4::PSE;
    r.s().cr3 = pd;
    // mov eax, cr0; or eax, 0x80000000; mov cr0, eax; mov dword [0x6000], 1; mov ebx, [0x00400000 + 0x6000]; mov [0x7000], eax
    r.code({0x0F, 0x20, 0xC0, 0x0D, 0x00, 0x00, 0x00, 0x80, 0x0F, 0x22, 0xC0, 0xC7, 0x05, 0x00, 0x60, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x8B,
            0x1D, 0x00, 0x60, 0x40, 0x00, 0x89, 0x05, 0x00, 0x70, 0x00, 0x00});
    r.run(3);
    EXPECT_TRUE(r.s().paging());
    r.run(2);
    u32 v = 0;
    r.bus.read_ram(0x6000, &v, 4);
    EXPECT_EQ(v, 1u);
    EXPECT_EQ(r.s().gpr[kEbx], 1u);  // through the 4 MiB page alias
    // Accessed and dirty on the 4 KiB entry for page 6; accessed on the PDE.
    u32 pte = 0;
    r.bus.read_ram(pt + 4 * 6, &pte, 4);
    EXPECT_TRUE(pte & 0x20);
    EXPECT_TRUE(pte & 0x40);
    u32 pde = 0;
    r.bus.read_ram(pd, &pde, 4);
    EXPECT_TRUE(pde & 0x20);
    // The write to the missing page: #PF with error code 2 (write, not present), CR2 = 0x7000, EIP at the gate.
    r.run(1);
    EXPECT_EQ(r.s().eip, 0x5000u);
    EXPECT_EQ(r.s().cr2, 0x7000u);
    u32 err = 0;
    r.bus.read_ram(r.s().gpr[kEsp], &err, 4);
    EXPECT_EQ(err, 2u);
    u32 ret_eip = 0;
    r.bus.read_ram(r.s().gpr[kEsp] + 4, &ret_eip, 4);
    EXPECT_EQ(ret_eip, 0x101Bu);  // the faulting instruction itself
    EXPECT_FALSE(r.s().flags & fl::IF);
}

TEST(P6System, CpuidAndRdtscAndMsr) {
    Rig r;
    // xor eax, eax; cpuid; -> "GenuineIntel" in ebx:edx:ecx
    r.code({0x31, 0xC0, 0x0F, 0xA2, 0x0F, 0x31});
    r.run(2);
    EXPECT_EQ(r.s().gpr[kEbx], 0x756E6547u);
    EXPECT_EQ(r.s().gpr[kEdx], 0x49656E69u);
    EXPECT_EQ(r.s().gpr[kEcx], 0x6C65746Eu);
    r.run(1);  // rdtsc
    // The counter as the third instruction reads it: the two retired instructions' cycles under the pipeline
    // model (at least one each), well under the 2^32 roll-over.
    EXPECT_GE(r.s().gpr[kEax], 2u);
    EXPECT_LT(r.s().gpr[kEax], 1000u);
    EXPECT_EQ(r.s().gpr[kEdx], 0u);
}

TEST(P6Fpu, ArithmeticC1IsTheRoundUpBit) {
    if (!kHostX87) GTEST_SKIP() << "no 80-bit host x87: extended-precision rounding is approximated";
    Rig r;
    // fninit; fld1; fld1; fadd st0,st1; fadd st0,st1 (3); fld1; fadd st0,st0 (2); fdiv st0,st1 (2/3)
    r.code({0xDB, 0xE3, 0xD9, 0xE8, 0xD9, 0xE8, 0xD8, 0xC1, 0xD8, 0xC1, 0xD9, 0xE8, 0xD8, 0xC0, 0xD8, 0xF1,
            // fldcw [0x2000] (round down); fld1; fadd st0,st0 (2); fdiv st0,st2 (2/3 again)
            0xD9, 0x2D, 0x00, 0x20, 0x00, 0x00, 0xD9, 0xE8, 0xD8, 0xC0, 0xD8, 0xF2});
    r.bus.ram()[0x2000] = 0x7F;
    r.bus.ram()[0x2001] = 0x07;
    constexpr u16 kC1 = 1u << 9;
    r.run(8);
    // 2/3's 64-bit significand is 1010...10 followed by 1010...: round to nearest goes up, C1 = 1.
    EXPECT_EQ(r.s().fpu.fsw & kC1, kC1);
    r.run(2);
    EXPECT_EQ(r.s().fpu.fsw & kC1, 0u);  // a push clears it
    r.run(2);
    // The same quotient rounded down: inexact, but not rounded up, C1 = 0.
    EXPECT_EQ(r.s().fpu.fsw & kC1, 0u);
    EXPECT_NE(r.s().fpu.fsw & 0x20u, 0u);  // PE
}

TEST(P6Simd, UnaryScalarOpsReadTheSourceLane) {
    Rig r;
    r.s().cr4 |= cr4::OSFXSR;
    // sqrtss xmm0, xmm1; rcpss xmm2, xmm3
    r.code({0xF3, 0x0F, 0x51, 0xC1, 0xF3, 0x0F, 0x53, 0xD3});
    const float nine = 9.0f, four = 4.0f, two = 2.0f, eight = 8.0f;
    std::memcpy(r.s().fpu.xmm[0].data(), &nine, 4);
    std::memcpy(r.s().fpu.xmm[1].data(), &four, 4);
    std::memcpy(r.s().fpu.xmm[2].data(), &eight, 4);
    std::memcpy(r.s().fpu.xmm[3].data(), &two, 4);
    r.run(2);
    float root = 0, recip = 0;
    std::memcpy(&root, r.s().fpu.xmm[0].data(), 4);
    std::memcpy(&recip, r.s().fpu.xmm[2].data(), 4);
    EXPECT_EQ(root, 2.0f);   // sqrt of the source (4), not of the destination (9)
    EXPECT_EQ(recip, 0.5f);  // 1/2 is exact in 12 bits
}

TEST(P6Fpu, RegisterStoreClearsC1) {
    if (!kHostX87) GTEST_SKIP() << "no 80-bit host x87: extended-precision rounding is approximated";
    Rig r;
    // fninit; 2/3 rounded up as above (C1 = 1); fstp st(1)
    r.code({0xDB, 0xE3, 0xD9, 0xE8, 0xD9, 0xE8, 0xD8, 0xC1, 0xD8, 0xC1, 0xD9, 0xE8, 0xD8, 0xC0, 0xD8, 0xF1, 0xDD, 0xD9});
    constexpr u16 kC1 = 1u << 9;
    r.run(8);
    ASSERT_EQ(r.s().fpu.fsw & kC1, kC1);
    r.run(1);
    EXPECT_EQ(r.s().fpu.fsw & kC1, 0u);
}
