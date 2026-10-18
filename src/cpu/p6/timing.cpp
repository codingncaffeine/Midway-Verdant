#include "cpu/p6/timing.h"

#include <algorithm>

namespace mv::p6 {

namespace {

// The register-form cost of an instruction: µops per port (p01 goes to whichever ALU port is free first),
// the result latency, and the unit a non-pipelined operation keeps busy (0 none, 1 the integer divider,
// 2 the floating-point divider/square root). Memory forms add the load and store µops outside this table.
// From Agner Fog's "Instruction tables", Pentium II/III section.
struct Cost {
    u8 p0, p1, p01, p2, p34;
    u8 lat;
    u16 busy;
    u8 busy_unit;
};

constexpr Cost C(u8 p0, u8 p1, u8 p01, u8 lat = 1, u8 p2 = 0, u8 p34 = 0, u16 busy = 0, u8 unit = 0) {
    return Cost{p0, p1, p01, p2, p34, lat, busy, unit};
}

Cost cost_of(const Insn& in) {
    const bool mem = in.mem;
    switch (in.op) {
        case Op::Mov: return C(0, 0, mem ? 0 : 1, 1);
        case Op::MovFromSeg: return C(0, 0, 1, 1);
        case Op::MovSeg: return mem ? C(0, 0, 7, 8) : C(0, 0, 8, 5);
        case Op::Movzx: case Op::Movsx: return C(0, 0, mem ? 0 : 1, 1);
        case Op::Cmovcc: return C(0, 1, 1, 1);
        case Op::Xchg: return mem ? C(0, 0, 4, 20) : C(0, 0, 3, 1);
        case Op::Xlat: return C(0, 0, 1, 1, 1);
        case Op::Lea: return C(1, 0, 0, 1);
        case Op::Lds: case Op::Les: case Op::Lfs: case Op::Lgs: case Op::Lss: return C(0, 0, 8, 8, 2);
        case Op::Push: case Op::PushSeg: return C(0, 0, in.op == Op::PushSeg ? 2 : 1, 1, 0, 1);
        case Op::Pop: return mem ? C(0, 0, 5, 3, 1, 1) : C(0, 0, 1, 3, 1);
        case Op::PopSeg: return C(0, 0, 8, 8, 1);
        case Op::Pushf: return C(3, 0, 11, 2, 0, 1);
        case Op::Popf: return C(10, 0, 6, 10, 1);
        case Op::Pusha: return C(0, 0, 2, 2, 0, 8);
        case Op::Popa: return C(0, 0, 2, 3, 8);
        case Op::Lahf: case Op::Sahf: return C(0, 0, 1, 1);
        case Op::Add: case Op::Sub: case Op::And: case Op::Or: case Op::Xor: case Op::Cmp: case Op::Test: return C(0, 0, 1, 1);
        case Op::Adc: case Op::Sbb: return C(0, 0, mem && in.dst.kind == OpndKind::Mem ? 3 : 2, 2);
        case Op::Inc: case Op::Dec: case Op::Neg: case Op::Not: return C(0, 0, 1, 1);
        case Op::Daa: case Op::Das: case Op::Aaa: case Op::Aas: return C(0, 0, 1, 1);
        case Op::Aad: return C(1, 0, 2, 4);
        case Op::Aam: return C(1, 1, 2, 15);
        case Op::Imul1: case Op::Imul2: case Op::Imul3: return C(1, 0, 0, 4);
        case Op::Mul: return C(1, 0, 1, 4);
        case Op::Div: case Op::Idiv:
            if (in.size == 1) return C(2, 0, 1, 19, 0, 0, 12, 1);
            if (in.size == 2) return C(3, 0, 1, 23, 0, 0, 21, 1);
            return C(3, 0, 1, 39, 0, 0, 37, 1);
        case Op::Cbw: return C(0, 0, 1, 1);
        case Op::Cwd: return C(1, 0, 0, 1);
        case Op::Shl: case Op::Shr: case Op::Sar: case Op::Rol: case Op::Ror: return C(1, 0, 0, 1);
        case Op::Rcl: case Op::Rcr:
            if (in.src.kind == OpndKind::Imm && in.imm == 1) return C(1, 1, 0, 1);
            return in.size == 1 ? C(4, 0, 4, 4) : C(3, 0, 3, 3);
        case Op::Shld: case Op::Shrd: return C(2, 0, mem ? 1 : 0, 2);
        case Op::Bt: return mem ? C(1, 0, 6, 2) : C(0, 0, 1, 1);
        case Op::Bts: case Op::Btr: case Op::Btc: return mem ? C(1, 0, 6, 2) : C(0, 0, 1, 1);
        case Op::Bsf: case Op::Bsr: return C(0, 1, 1, 2);
        case Op::Setcc: return C(0, 0, 1, 1);
        case Op::Jmp: case Op::Jcc: case Op::JmpRm: return C(0, 1, 0, 1);
        case Op::JmpFar: case Op::JmpFarRm: return C(21, 0, 0, 21);
        case Op::Call: return C(1, 1, 0, 1, 0, 1);
        case Op::CallRm: return mem ? C(1, 0, 4, 2, 0, 1) : C(1, 0, 2, 1, 0, 1);
        case Op::CallFar: case Op::CallFarRm: return C(28, 0, 0, 28, 0, 2);
        case Op::Ret: return C(1, 0, in.imm ? 3 : 2, 2, 1);
        case Op::Retf: return C(23, 0, 0, 23, 3);
        case Op::Iret: return C(25, 0, 0, 25, 3);
        case Op::Jcxz: return C(0, 1, 1, 1);
        case Op::Loop: case Op::Loope: case Op::Loopne: return C(2, 1, 8, 8);
        case Op::Enter: return C(0, 0, 12, 12, 0, 1);
        case Op::Leave: return C(0, 0, 2, 3, 1);
        case Op::Bound: return C(7, 0, 6, 7, 2);
        case Op::Clc: case Op::Stc: case Op::Cmc: return C(0, 0, 1, 1);
        case Op::Cld: case Op::Std: return C(0, 0, 4, 4);
        case Op::Cli: return C(9, 0, 0, 9);
        case Op::Sti: return C(17, 0, 0, 17);
        case Op::Int: case Op::Int3: case Op::Into: case Op::Int1: return C(30, 0, 0, 30, 0, 3);
        case Op::Lods: return C(0, 0, 0, 3, 2);
        case Op::Stos: return C(0, 0, 1, 1, 0, 1);
        case Op::Movs: return C(0, 0, 1, 3, 3, 1);
        case Op::Scas: return C(0, 0, 1, 3, 2);
        case Op::Cmps: return C(0, 0, 4, 4, 2);
        case Op::Ins: return C(18, 0, 1, 20, 0, 1);
        case Op::Outs: return C(18, 0, 1, 20, 1);
        case Op::Bswap: return C(1, 0, 1, 1);
        case Op::Nop: case Op::NopEv: case Op::Prefetch: return C(0, 0, 1, 1);
        case Op::Cpuid: return C(35, 0, 0, 35);
        case Op::Rdtsc: return C(31, 0, 0, 31);
        case Op::Rdpmc: return C(31, 0, 0, 31);
        case Op::In: case Op::Out: return C(18, 0, 0, 18);
        case Op::Hlt: return C(1, 0, 0, 1);
        case Op::Wait: return C(0, 0, 2, 2);
        case Op::Rdmsr: case Op::Wrmsr: return C(40, 0, 0, 40);
        case Op::Sysenter: case Op::Sysexit: return C(30, 0, 0, 30);
        case Op::Wbinvd: case Op::Invd: return C(200, 0, 0, 200);
        case Op::MovToCr: return in.dst.reg == 3 ? C(40, 0, 0, 40) : C(20, 0, 0, 20);
        case Op::MovFromCr: case Op::MovFromDr: case Op::MovToDr: return C(10, 0, 0, 10);
        case Op::Lgdt: case Op::Lidt: return C(20, 0, 0, 20, 2);
        case Op::Sgdt: case Op::Sidt: return C(10, 0, 0, 10, 0, 2);
        case Op::Lldt: case Op::Ltr: return C(60, 0, 0, 60, 2);
        case Op::Sldt: case Op::Str: case Op::Smsw: return C(5, 0, 0, 5);
        case Op::Lmsw: return C(20, 0, 0, 20);
        case Op::Invlpg: return C(30, 0, 0, 30);
        case Op::Clts: return C(10, 0, 0, 10);
        case Op::Verr: case Op::Verw: case Op::Lar: case Op::Lsl: return C(10, 0, 3, 12, 2);
        case Op::Arpl: return C(0, 0, 3, 3);
        case Op::Xadd: return mem ? C(0, 0, 4, 20) : C(0, 0, 3, 2);
        case Op::Cmpxchg: return mem ? C(0, 0, 5, 20) : C(0, 0, 4, 2);
        case Op::Cmpxchg8b: return C(0, 0, 8, 20, 2, 2);
        case Op::Salc: return C(0, 0, 1, 1);
        case Op::Fxsave: return C(116, 0, 0, 62, 0, 32);
        case Op::Fxrstor: return C(89, 0, 0, 68, 32);
        case Op::Ldmxcsr: return C(11, 0, 0, 15, 1);
        case Op::Stmxcsr: return C(6, 0, 0, 7, 0, 1);
        case Op::Sfence: return C(0, 0, 0, 6, 0, 1);
        case Op::Ud2: case Op::Rsm: case Op::Invalid: return C(1, 0, 0, 1);
        case Op::X87: {
            // By the escape and ModRM: the arithmetic group, loads/stores, the transcendentals.
            const u8 op = in.opcode;
            const u8 reg = in.reg;
            if (mem) {
                switch (op) {
                    case 0xD8: case 0xDC:  // arithmetic with m32/m64
                        if (reg == 1) return C(1, 0, 0, 6);
                        if (reg == 6 || reg == 7) return C(1, 0, 0, 38, 0, 0, 37, 2);
                        if (reg == 2 || reg == 3) return C(1, 0, 0, 1);
                        return C(1, 0, 0, 4);
                    case 0xDA: case 0xDE: return C(6, 0, 0, 6);  // integer-operand arithmetic
                    case 0xD9:
                        if (reg == 0) return C(0, 0, 0, 1);         // FLD m32
                        if (reg == 2 || reg == 3) return C(0, 0, 0, 1, 0, 1);  // FST(P) m32
                        if (reg == 5) return C(1, 1, 0, 10);        // FLDCW
                        if (reg == 7) return C(1, 0, 0, 1, 0, 1);   // FNSTCW
                        return C(30, 0, 0, 30, 7, 7);               // FLDENV/FNSTENV
                    case 0xDB:
                        if (reg == 5) return C(2, 0, 0, 2, 1);      // FLD m80
                        if (reg == 7) return C(2, 0, 0, 2, 1, 2);   // FSTP m80
                        if (reg == 0) return C(3, 0, 0, 5);         // FILD
                        return C(2, 0, 0, 5, 0, 1);                 // FIST(P)
                    case 0xDD:
                        if (reg == 0) return C(0, 0, 0, 1);         // FLD m64
                        if (reg == 2 || reg == 3) return C(0, 0, 0, 1, 0, 1);
                        if (reg == 4) return C(72, 0, 0, 72, 27);   // FRSTOR
                        if (reg == 6) return C(141, 0, 0, 141, 0, 27);  // FNSAVE
                        return C(1, 0, 0, 1, 0, 1);                 // FNSTSW m16
                    case 0xDF:
                        if (reg == 0 || reg == 5) return C(3, 0, 0, 5);  // FILD
                        if (reg == 4) return C(38, 0, 0, 38, 2);         // FBLD
                        if (reg == 6) return C(165, 0, 0, 165, 2, 2);    // FBSTP
                        return C(2, 0, 0, 5, 0, 1);                      // FIST(P)
                    default: return C(1, 0, 0, 1);
                }
            }
            switch (op) {
                case 0xD8: case 0xDC: case 0xDE:
                    if (reg == 1) return C(1, 0, 0, 5, 0, 0, 2, 0);
                    if (reg == 6 || reg == 7) return C(1, 0, 0, 38, 0, 0, 37, 2);
                    if (reg == 2 || reg == 3) return C(1, 0, 0, 1);
                    return C(1, 0, 0, 3);
                case 0xD9:
                    switch (in.modrm) {
                        case 0xC8: case 0xC9: case 0xCA: case 0xCB: case 0xCC: case 0xCD: case 0xCE: case 0xCF: return C(0, 0, 1, 0);  // FXCH
                        case 0xD0: return C(1, 0, 0, 1);
                        case 0xE0: return C(3, 0, 0, 2);
                        case 0xE1: return C(1, 0, 0, 1);
                        case 0xE4: case 0xE5: return C(1, 0, 0, 2);
                        case 0xEE: return C(1, 0, 0, 1);
                        case 0xE8: case 0xE9: case 0xEA: case 0xEB: case 0xEC: case 0xED: return C(2, 0, 0, 2);
                        case 0xF0: return C(30, 0, 0, 66);
                        case 0xF1: return C(45, 0, 0, 103);
                        case 0xF2: return C(60, 0, 0, 80);
                        case 0xF3: return C(55, 0, 0, 90);
                        case 0xF4: return C(15, 0, 0, 15);
                        case 0xF5: return C(33, 0, 0, 33);
                        case 0xF8: return C(23, 0, 0, 23);
                        case 0xF9: return C(42, 0, 0, 100);
                        case 0xFA: return C(1, 0, 0, 69, 0, 0, 69, 2);
                        case 0xFB: return C(60, 0, 0, 80);
                        case 0xFC: return C(30, 0, 0, 30);
                        case 0xFD: return C(56, 0, 0, 56);
                        case 0xFE: case 0xFF: return C(50, 0, 0, 60);
                        case 0xF6: case 0xF7: return C(1, 0, 0, 1);
                        default: return C(1, 0, 0, 1);  // FLD ST(i)
                    }
                case 0xDA: return in.modrm == 0xE9 ? C(1, 1, 0, 1) : C(2, 0, 0, 2);
                case 0xDB:
                    if (in.modrm == 0xE2) return C(0, 0, 3, 3);
                    if (in.modrm == 0xE3) return C(13, 0, 0, 13);
                    if (in.modrm < 0xE0) return C(2, 0, 0, 2);
                    return C(1, 0, 0, 1);
                case 0xDD: return C(1, 0, 0, 1);
                case 0xDF: return in.modrm == 0xE0 ? C(3, 0, 0, 7) : C(1, 0, 0, 1);
                default: return C(1, 0, 0, 1);
            }
        }
        case Op::Simd: {
            const u8 op = in.opcode;
            const bool scalar = in.rep == 1;
            // MMX
            if ((op >= 0x60 && op <= 0x7F) || op == 0xC4 || op == 0xC5 || (op >= 0xD1 && op <= 0xFE)) {
                if (op == 0x77) return C(11, 0, 0, 6);
                if (op == 0x6E || op == 0x7E) return C(0, 0, 1, 1);
                if (op == 0x6F || op == 0x7F || op == 0xE7) return C(0, 0, 1, 1);
                if (op == 0xD5 || op == 0xE5 || op == 0xE4 || op == 0xF5) return C(1, 0, 0, 3);
                if (op == 0xF6) return C(2, 0, 1, 5);
                if ((op >= 0x60 && op <= 0x6B) || op == 0x70 || (op >= 0x71 && op <= 0x73) || op == 0xD1 || op == 0xD2 || op == 0xD3 ||
                    op == 0xE1 || op == 0xE2 || op == 0xF1 || op == 0xF2 || op == 0xF3 || op == 0xC4 || op == 0xD7)
                    return C(0, 1, 0, 1);
                if (op == 0xC5) return C(0, 1, 1, 2);
                return C(0, 0, 1, 1);
            }
            switch (op) {
                case 0x10: case 0x11: return scalar ? C(0, 0, 1, 1) : (mem ? C(mem && in.dst.kind == OpndKind::Mem ? 1 : 0, 0, 0, 2, 2, 2) : C(0, 0, 2, 1));
                case 0x28: case 0x29: case 0x2B: return mem ? C(0, 0, 0, 2, 1, 1) : C(0, 0, 2, 1);
                case 0x12: case 0x13: case 0x16: case 0x17: return C(0, 0, 1, 1);
                case 0x14: case 0x15: return C(2, 2, 0, 3);
                case 0x2A: return C(0, 0, 2, scalar ? 4 : 3);
                case 0x2C: case 0x2D: return C(0, 0, scalar ? 1 : 2, 3);
                case 0x2E: case 0x2F: return C(0, 0, 1, 1);
                case 0x50: return C(1, 0, 0, 1);
                case 0x51: return scalar ? C(2, 0, 0, 30, 0, 0, 28, 2) : C(2, 0, 0, 56, 0, 0, 56, 2);
                case 0x52: case 0x53: return scalar ? C(1, 0, 0, 1) : C(2, 0, 0, 2);
                case 0x54: case 0x55: case 0x56: case 0x57: return C(0, 0, 2, 2);
                case 0x58: case 0x5C: return scalar ? C(0, 0, 1, 3) : C(0, 0, 2, 3);
                case 0x59: return scalar ? C(1, 0, 0, 4) : C(2, 0, 0, 4);
                case 0x5E: return scalar ? C(1, 0, 0, 18, 0, 0, 17, 2) : C(2, 0, 0, 48, 0, 0, 34, 2);
                case 0x5D: case 0x5F: return scalar ? C(0, 0, 1, 3) : C(0, 0, 2, 3);
                case 0xC2: return scalar ? C(0, 0, 1, 3) : C(0, 0, 2, 3);
                case 0xC6: return C(2, 1, 0, 2);
                default: return C(0, 0, 2, 2);
            }
        }
    }
    return C(0, 0, 1, 1);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

Timing::Cache::Cache(unsigned sets, unsigned ways) : sets_(sets), ways_(ways) {}

bool Timing::Cache::access(PhysAddr phys, u64 now) {
    const u32 line = phys >> 5;
    const unsigned set = line & (sets_ - 1);
    const unsigned base = set * ways_;
    unsigned victim = base;
    u64 oldest = ~u64{0};
    for (unsigned w = 0; w < ways_; ++w) {
        const unsigned i = base + w;
        if (valid_[i] && tags_[i] == line) {
            last_use_[i] = now;
            return true;
        }
        if (!valid_[i]) {
            victim = i;
            oldest = 0;
        } else if (last_use_[i] < oldest) {
            oldest = last_use_[i];
            victim = i;
        }
    }
    valid_[victim] = 1;
    tags_[victim] = line;
    last_use_[victim] = now;
    return false;
}

Timing::Timing() { reset(); }

void Timing::reset() {
    decode_cycle_ = 0;
    decode_slot_ = decode_bytes_ = 0;
    front_end_restart_ = 0;
    port_free_.fill(0);
    reg_ready_.fill(0);
    retire_cycle_ = 0;
    retire_slot_ = 0;
    rob_.fill(0);
    rob_pos_ = 0;
    int_div_free_ = fp_div_free_ = 0;
    l1i_ = Cache(128, 4);
    l1d_ = Cache(128, 4);
    l2_ = Cache(512, 8);
    btb_.fill(Btb{});
    rsb_.fill(0);
    rsb_top_ = 0;
    stats_ = TimingStats{};
}

void Timing::skip_to(u64 cycle) {
    if (cycle <= retire_cycle_) return;
    decode_cycle_ = cycle;
    decode_slot_ = decode_bytes_ = 0;
    front_end_restart_ = cycle;
    for (auto& p : port_free_) p = std::max(p, cycle);
    for (auto& r : reg_ready_) r = std::min(r, cycle);
    retire_cycle_ = cycle;
    retire_slot_ = 0;
}

u64 Timing::alloc_port(unsigned port, u64 earliest) {
    const u64 t = std::max(port_free_[port], earliest);
    port_free_[port] = t + 1;
    return t;
}

// The extra cycles a data access costs beyond the L1 hit latency.
unsigned Timing::memory_penalty(PhysAddr phys, u64 now, bool store) {
    if (l1d_.access(phys, now)) return 0;
    ++stats_.l1d_misses;
    unsigned penalty = tp::kL2Latency;
    if (!l2_.access(phys, now)) {
        ++stats_.l2_misses;
        penalty += tp::kDramLatency;
    }
    // A store miss is absorbed by the store buffer: the line is allocated, the instruction does not wait.
    return store ? 0 : penalty;
}

u64 Timing::fetch_ready(PhysAddr phys, unsigned len, u64 now) {
    u64 ready = now;
    for (PhysAddr line = phys & ~31u; line <= ((phys + len - 1) & ~31u); line += 32) {
        if (l1i_.access(line, now)) continue;
        ++stats_.l1i_misses;
        unsigned penalty = tp::kL2Latency;
        if (!l2_.access(line, now)) {
            ++stats_.l2_misses;
            penalty += tp::kDramLatency;
        }
        ready = std::max(ready, now + penalty);
        if (line == 0xFFFFFFE0u) break;
    }
    return ready;
}

u64 Timing::retire(const InsnTiming& t) {
    ++stats_.instructions;
    if (t.interrupt_delivery) {
        // The microcode of the delivery: the front end restarts at the handler.
        const u64 start = std::max(retire_cycle_, decode_cycle_);
        const u64 done = start + tp::kInterruptDelivery;
        retire_cycle_ = done;
        retire_slot_ = 0;
        decode_cycle_ = done;
        decode_slot_ = decode_bytes_ = 0;
        front_end_restart_ = done;
        stats_.cycles = retire_cycle_;
        return retire_cycle_;
    }
    const Insn& in = *t.in;
    Cost c = cost_of(in);
    // The memory operand adds its load and store µops; a string step scales with its elements.
    unsigned loads = c.p2, stores = c.p34;
    if (in.mem) {
        const bool is_dst = in.dst.kind == OpndKind::Mem;
        const bool is_src = in.src.kind == OpndKind::Mem || in.src2.kind == OpndKind::Mem;
        const bool read_modify = is_dst && (in.op != Op::Mov && in.op != Op::Setcc && in.op != Op::Pop && in.op != Op::MovFromSeg && in.op != Op::Sgdt &&
                                            in.op != Op::Sidt && in.op != Op::Stmxcsr && in.op != Op::Fxsave && in.op != Op::Str && in.op != Op::Sldt &&
                                            in.op != Op::Smsw);
        if (is_src || read_modify || (in.op == Op::X87 && in.dst.kind != OpndKind::Mem)) loads += 1;
        if (is_dst) stores += 1;
    }
    unsigned alu = static_cast<unsigned>(c.p0) + c.p1 + c.p01;
    if (t.rep_iterations > 1) {
        const unsigned n = t.rep_iterations;
        loads *= n;
        stores *= n;
        alu = alu * n;
        c.p01 = static_cast<u8>(std::min<unsigned>(255, c.p01 * n));
        c.p0 = static_cast<u8>(std::min<unsigned>(255, c.p0 * n));
        c.p1 = static_cast<u8>(std::min<unsigned>(255, c.p1 * n));
    }
    const unsigned uops = std::max(1u, alu + loads + stores * 2);
    stats_.uops += uops;

    // ---- front end: fetch and decode ----
    u64 fetch_at = std::max(decode_cycle_, front_end_restart_);
    const u64 fetched = fetch_ready(t.fetch_phys, t.len, fetch_at);
    if (fetched > fetch_at) {
        stats_.stall_fetch += fetched - fetch_at;
        fetch_at = fetched;
        decode_slot_ = decode_bytes_ = 0;
    }
    u64 decode = std::max(decode_cycle_, fetch_at);
    if (decode > decode_cycle_) {
        decode_slot_ = decode_bytes_ = 0;
        decode_cycle_ = decode;
    }
    // The 4-1-1 rule: a multi-µop instruction takes the first decoder, the second and third take one µop each;
    // sixteen bytes a cycle; a long microcoded instruction holds the decoders for its µops at three a cycle.
    const bool fits = (uops == 1 && decode_slot_ < 3) || (decode_slot_ == 0 && uops <= 4);
    if (!fits || decode_bytes_ + t.len > 16) {
        decode_cycle_ += 1;
        decode = decode_cycle_;
        decode_slot_ = decode_bytes_ = 0;
    }
    decode_slot_ += 1;
    decode_bytes_ += t.len;
    if (uops > 4) {
        decode_cycle_ += (uops + 2) / 3;
        decode_slot_ = 3;
    }
    // The reorder buffer: the µop this one displaces must have retired.
    {
        const unsigned oldest = (rob_pos_ + uops - 1) % tp::kRobEntries;  // the entry this instruction's last µop displaces
        const u64 must_retire = rob_[oldest];
        if (must_retire > decode) {
            stats_.stall_rob += must_retire - decode;
            decode = must_retire;
            decode_cycle_ = std::max(decode_cycle_, decode);
        }
    }

    // ---- issue ----
    u64 inputs = 0;
    for (unsigned r = 0; r < kRegCount; ++r)
        if (t.reads & (1u << r)) inputs = std::max(inputs, reg_ready_[r]);
    const u64 issue_base = decode + tp::kDecodeToIssue;
    if (inputs > issue_base) stats_.stall_dependency += inputs - issue_base;
    u64 operands_ready = std::max(issue_base, inputs);
    // Loads: the address first, then the data after the cache answers; device reads wait on the chipset.
    u64 data_ready = operands_ready;
    unsigned access_index = 0;
    for (unsigned i = 0; i < loads; ++i) {
        const u64 at = alloc_port(2, operands_ready);
        unsigned latency = tp::kLoadLatency;
        bool device = false;
        while (access_index < t.naccesses && (t.accesses[access_index].kind == 1 || t.accesses[access_index].kind == 3 || t.accesses[access_index].kind == 5))
            ++access_index;
        if (access_index < t.naccesses) {
            const MemAccess& a = t.accesses[access_index++];
            if (a.kind == 0) latency += memory_penalty(a.phys, at, false);
            else if (a.kind == 2) {
                latency = tp::kDeviceLoad;
                device = true;
                ++stats_.device_loads;
            } else if (a.kind == 4) {
                latency = tp::kPortIo;
                ++stats_.port_ios;
            }
        }
        (void)device;
        data_ready = std::max(data_ready, at + latency);
    }
    // The ALU µops: port 0/1 as the table says, p01 to whichever is free first; a non-pipelined unit blocks.
    u64 done = data_ready;
    u64 last_alu = data_ready;
    auto issue_alu = [&](unsigned port, unsigned count) {
        for (unsigned i = 0; i < count; ++i) last_alu = std::max(last_alu, alloc_port(port, data_ready));
    };
    issue_alu(0, c.p0);
    issue_alu(1, c.p1);
    for (unsigned i = 0; i < c.p01; ++i) {
        const unsigned port = port_free_[0] <= port_free_[1] ? 0 : 1;
        last_alu = std::max(last_alu, alloc_port(port, data_ready));
    }
    if (c.busy_unit == 1) {
        const u64 at = std::max(last_alu, int_div_free_);
        int_div_free_ = at + c.busy;
        last_alu = at;
    } else if (c.busy_unit == 2) {
        const u64 at = std::max(last_alu, fp_div_free_);
        fp_div_free_ = at + c.busy;
        last_alu = at;
    }
    if (alu > 0) done = last_alu + c.lat;
    else if (loads > 0) done = data_ready;
    else done = operands_ready + c.lat;
    // Stores: address on port 3, data on port 4 once the data exists; device and port writes are posted.
    for (unsigned i = 0; i < stores; ++i) {
        const u64 addr_at = alloc_port(3, operands_ready);
        const u64 data_at = alloc_port(4, std::max(addr_at, alu > 0 ? done : operands_ready));
        unsigned extra = 0;
        for (unsigned k = 0; k < t.naccesses; ++k) {
            const MemAccess& a = t.accesses[k];
            if (a.kind == 1) memory_penalty(a.phys, data_at, true);
            else if (a.kind == 3) {
                extra = tp::kDeviceStore;
                ++stats_.device_stores;
            } else if (a.kind == 5) {
                extra = tp::kPortIo;
                ++stats_.port_ios;
            }
        }
        done = std::max(done, data_at + 1 + extra);
    }
    for (unsigned r = 0; r < kRegCount; ++r)
        if (t.writes & (1u << r)) reg_ready_[r] = done;

    // ---- branches ----
    if (t.is_branch) {
        ++stats_.branches;
        bool predicted_taken;
        u32 predicted_target = 0;
        Btb& b = btb_[(t.branch_eip >> 2) & 511];
        const bool btb_hit = b.tag == t.branch_eip;
        if (t.is_ret) {
            predicted_taken = true;
            predicted_target = rsb_[(rsb_top_ + 15) & 15];
            rsb_top_ = (rsb_top_ + 15) & 15;
        } else {
            predicted_taken = btb_hit && b.counter >= 2;
            predicted_target = b.target;
        }
        const bool mispredict = predicted_taken != t.taken || (t.taken && predicted_target != t.target);
        if (t.is_call) {
            rsb_[rsb_top_] = t.branch_eip + t.len;
            rsb_top_ = (rsb_top_ + 1) & 15;
        }
        if (!t.is_ret) {
            if (!btb_hit) {
                b.tag = t.branch_eip;
                b.counter = t.taken ? 2 : 1;
            } else if (t.taken) {
                if (b.counter < 3) ++b.counter;
            } else if (b.counter > 0) {
                --b.counter;
            }
            if (t.taken) b.target = t.target;
        }
        if (mispredict) {
            ++stats_.mispredicts;
            const u64 resolved = alu > 0 ? last_alu + 1 : done;
            front_end_restart_ = std::max(front_end_restart_, resolved + tp::kMispredictBubble);
        }
    }

    // ---- retirement: in order, three µops a cycle ----
    u64 r = std::max(done, retire_cycle_);
    unsigned remaining = uops;
    if (r > retire_cycle_) retire_slot_ = 0;
    while (remaining > 0) {
        const unsigned room = tp::kRetireWidth - retire_slot_;
        if (room == 0) {
            ++r;
            retire_slot_ = 0;
            continue;
        }
        const unsigned take = std::min(room, remaining);
        retire_slot_ += take;
        remaining -= take;
    }
    retire_cycle_ = r;
    for (unsigned i = 0; i < uops; ++i) {
        rob_[rob_pos_] = r;
        rob_pos_ = (rob_pos_ + 1) % tp::kRobEntries;
    }
    stats_.cycles = retire_cycle_;
    return retire_cycle_;
}

}  // namespace mv::p6
