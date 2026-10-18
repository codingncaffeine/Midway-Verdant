#include "cpu/p6/cpu.h"

#include <bit>
#include <cstring>
#include <format>

#include "debug/trace.h"

namespace mv::p6 {

const std::array<u8, 256> kParityEven = [] {
    std::array<u8, 256> t{};
    for (unsigned i = 0; i < 256; ++i) t[i] = (std::popcount(i) & 1) ? 0 : 1;
    return t;
}();

namespace {
constexpr unsigned kRepIterationsPerStep = 4096;

u32 rotl(u32 v, unsigned n, unsigned bits) {
    n %= bits;
    if (n == 0) return v;
    return ((v << n) | (v >> (bits - n))) & size_mask(bits / 8);
}
u32 rotr(u32 v, unsigned n, unsigned bits) {
    n %= bits;
    if (n == 0) return v;
    return ((v >> n) | (v << (bits - n))) & size_mask(bits / 8);
}
}  // namespace

// Code bytes for the decoder: a page's worth copied from RAM when the code is there, byte reads otherwise.
class Cpu::Fetcher : public CodeSource {
public:
    explicit Fetcher(Cpu& cpu) : cpu_(cpu) {}
    u8 byte(unsigned offset) override {
        if (offset < have_) return buf_[offset];
        while (have_ <= offset) {
            const u32 off = cpu_.st_.eip + have_;
            const u32 lin = cpu_.linear(kCs, off, 1, Access::Fetch);
            if (have_ == 0) {
                cpu_.fetch_phys_ = cpu_.mem_.translate(lin, Access::Fetch);
                // Fast path: the rest of the page from RAM in one copy.
                const unsigned to_page_end = 4096 - (lin & 0xFFFu);
                if (const u8* host = cpu_.mem_.host_pointer(lin, Access::Fetch)) {
                    const unsigned n = to_page_end < 16 ? to_page_end : 16;
                    std::memcpy(buf_, host, n);
                    have_ = n;
                    continue;
                }
            }
            buf_[have_++] = static_cast<u8>(cpu_.mem_.read(lin, 1));
        }
        return buf_[offset];
    }
    const u8* bytes() const { return buf_; }

private:
    Cpu& cpu_;
    u8 buf_[16]{};
    unsigned have_{0};
};

Cpu::Cpu(Bus& bus) : bus_(bus), mem_(bus, st_) {}

void Cpu::reset(u32 cpuid_signature) {
    cpuid_signature_ = cpuid_signature;
    st_ = CpuState{};
    st_.gpr[kEdx] = cpuid_signature;
    st_.eip = 0xFFF0;
    st_.flags = fl::kAlways;
    for (auto& s : st_.seg) {
        s.selector = 0;
        s.base = 0;
        s.limit = 0xFFFF;
        s.ar = 0x93;  // present, data, read/write, accessed
    }
    st_.seg[kCs].selector = 0xF000;
    st_.seg[kCs].base = 0xFFFF0000u;
    st_.seg[kCs].ar = 0x9B;  // present, code, execute/read, accessed
    st_.ldtr = Segment{0, 0, 0xFFFF, 0x82};
    st_.tr = Segment{0, 0, 0xFFFF, 0x8B};
    st_.gdtr = DescTable{};
    st_.idtr = DescTable{};
    st_.cr0 = cr0::kReset;
    st_.dr[6] = 0xFFFF0FF0u;
    st_.dr[7] = 0x400u;
    st_.fpu.fcw = 0x40;
    st_.fpu.ftw = 0xFFFF;
    st_.fpu.mxcsr = 0x1F80;
    st_.halted = false;
    triple_fault_ = false;
    mem_.flush_tlb(false);
    timing_.reset();
    timing_.skip_to(cycles_);
}

// ---------------------------------------------------------------------------------------------------------------
// The cycle model's view of an instruction: the scoreboard registers it reads and writes.

namespace {
void regs_of(const Insn& in, unsigned top, u32& reads, u32& writes) {
    reads = writes = 0;
    auto gpr = [](unsigned r) { return 1u << (r & 7); };
    auto opnd = [&](const Opnd& o, bool is_read, bool is_write) {
        u32 bit = 0;
        switch (o.kind) {
            case OpndKind::Reg: bit = gpr(o.reg); break;
            case OpndKind::Seg: bit = 1u << (kRegSeg + o.reg); break;
            case OpndKind::Mmx: {
                // SSE forms carry XMM numbers in the same operand slot.
                const u8 op = in.opcode;
                const bool sse = (op >= 0x10 && op <= 0x17) || (op >= 0x28 && op <= 0x2F) || (op >= 0x50 && op <= 0x5F) || op == 0xC2 || op == 0xC6;
                bit = 1u << ((sse ? kRegXmm : kRegX87) + (o.reg & 7));
                break;
            }
            case OpndKind::Xmm: bit = 1u << (kRegXmm + (o.reg & 7)); break;
            default: break;
        }
        if (is_read) reads |= bit;
        if (is_write) writes |= bit;
    };
    if (in.mem) {
        if (in.base != Insn::kNoReg) reads |= gpr(in.base);
        if (in.index != Insn::kNoReg) reads |= gpr(in.index);
    }
    const u32 flags = 1u << kRegFlags;
    const u32 esp = gpr(kEsp), eax = gpr(kEax), ecx = gpr(kEcx), edx = gpr(kEdx), ebx = gpr(kEbx), esi = gpr(kEsi), edi = gpr(kEdi), ebp = gpr(kEbp);
    switch (in.op) {
        case Op::Mov: case Op::Movzx: case Op::Movsx: case Op::Lea: case Op::MovFromSeg: case Op::MovFromCr: case Op::MovFromDr:
            opnd(in.dst, false, true);
            opnd(in.src, true, false);
            break;
        case Op::MovSeg: case Op::MovToCr: case Op::MovToDr:
            opnd(in.dst, false, true);
            opnd(in.src, true, false);
            break;
        case Op::Cmovcc:
            opnd(in.dst, true, true);
            opnd(in.src, true, false);
            reads |= flags;
            break;
        case Op::Setcc: opnd(in.dst, false, true); reads |= flags; break;
        case Op::Lds: case Op::Les: case Op::Lfs: case Op::Lgs: case Op::Lss: opnd(in.dst, false, true); writes |= 1u << (kRegSeg + kDs); break;
        case Op::Xchg: opnd(in.dst, true, true); opnd(in.src.kind == OpndKind::None ? Opnd{OpndKind::Reg, kEax} : in.src, true, true); break;
        case Op::Xlat: reads |= eax | ebx; writes |= eax; break;
        case Op::Cbw: reads |= eax; writes |= eax; break;
        case Op::Cwd: reads |= eax; writes |= edx; break;
        case Op::Bswap: opnd(in.dst, true, true); break;
        case Op::Cmp: case Op::Test: opnd(in.dst, true, false); opnd(in.src, true, false); writes |= flags; break;
        case Op::Adc: case Op::Sbb: opnd(in.dst, true, true); opnd(in.src, true, false); reads |= flags; writes |= flags; break;
        case Op::Add: case Op::Or: case Op::And: case Op::Sub: case Op::Xor: case Op::Xadd:
            opnd(in.dst, true, true);
            opnd(in.src, true, in.op == Op::Xadd);
            writes |= flags;
            break;
        case Op::Inc: case Op::Dec: opnd(in.dst, true, true); reads |= flags; writes |= flags; break;
        case Op::Not: opnd(in.dst, true, true); break;
        case Op::Neg: opnd(in.dst, true, true); writes |= flags; break;
        case Op::Mul: case Op::Imul1: opnd(in.dst, true, false); reads |= eax; writes |= eax | edx | flags; break;
        case Op::Div: case Op::Idiv: opnd(in.dst, true, false); reads |= eax | edx; writes |= eax | edx | flags; break;
        case Op::Imul2: opnd(in.dst, true, true); opnd(in.src, true, false); writes |= flags; break;
        case Op::Imul3: opnd(in.dst, false, true); opnd(in.src, true, false); writes |= flags; break;
        case Op::Rol: case Op::Ror: case Op::Shl: case Op::Shr: case Op::Sar:
            opnd(in.dst, true, true);
            opnd(in.src, true, false);
            writes |= flags;
            break;
        case Op::Rcl: case Op::Rcr: opnd(in.dst, true, true); opnd(in.src, true, false); reads |= flags; writes |= flags; break;
        case Op::Shld: case Op::Shrd: opnd(in.dst, true, true); opnd(in.src, true, false); opnd(in.src2, true, false); writes |= flags; break;
        case Op::Push: opnd(in.src.kind != OpndKind::None ? in.src : in.dst, true, false); reads |= esp; writes |= esp; break;
        case Op::Pop: opnd(in.dst, false, true); reads |= esp; writes |= esp; break;
        case Op::PushSeg: opnd(in.dst, true, false); reads |= esp; writes |= esp; break;
        case Op::PopSeg: opnd(in.dst, false, true); reads |= esp; writes |= esp; break;
        case Op::Pusha: reads |= 0xFF; writes |= esp; break;
        case Op::Popa: reads |= esp; writes |= 0xFF; break;
        case Op::Pushf: reads |= esp | flags; writes |= esp; break;
        case Op::Popf: reads |= esp; writes |= esp | flags; break;
        case Op::Enter: reads |= esp | ebp; writes |= esp | ebp; break;
        case Op::Leave: reads |= ebp; writes |= esp | ebp; break;
        case Op::Jcc: reads |= flags; break;
        case Op::Jmp: case Op::JmpFar: break;
        case Op::JmpRm: case Op::JmpFarRm: opnd(in.dst, true, false); break;
        case Op::Call: case Op::CallFar: reads |= esp; writes |= esp; break;
        case Op::CallRm: case Op::CallFarRm: opnd(in.dst, true, false); reads |= esp; writes |= esp; break;
        case Op::Ret: case Op::Retf: reads |= esp; writes |= esp; break;
        case Op::Iret: reads |= esp; writes |= esp | flags; break;
        case Op::Loop: reads |= ecx; writes |= ecx; break;
        case Op::Loope: case Op::Loopne: reads |= ecx | flags; writes |= ecx; break;
        case Op::Jcxz: reads |= ecx; break;
        case Op::Int: case Op::Int3: case Op::Int1: reads |= esp; writes |= esp | flags; break;
        case Op::Into: reads |= esp | flags; writes |= esp | flags; break;
        case Op::Cli: case Op::Sti: case Op::Cld: case Op::Std: case Op::Clc: case Op::Stc: case Op::Cmc: reads |= flags; writes |= flags; break;
        case Op::Sahf: reads |= eax | flags; writes |= flags; break;
        case Op::Lahf: reads |= flags; writes |= eax; break;
        case Op::Salc: reads |= flags; writes |= eax; break;
        case Op::Daa: case Op::Das: case Op::Aaa: case Op::Aas: case Op::Aam: case Op::Aad: reads |= eax | flags; writes |= eax | flags; break;
        case Op::Bt: opnd(in.dst, true, false); opnd(in.src, true, false); writes |= flags; break;
        case Op::Bts: case Op::Btr: case Op::Btc: opnd(in.dst, true, true); opnd(in.src, true, false); writes |= flags; break;
        case Op::Bsf: case Op::Bsr: opnd(in.dst, false, true); opnd(in.src, true, false); writes |= flags; break;
        case Op::Cmpxchg: opnd(in.dst, true, true); opnd(in.src, true, false); reads |= eax; writes |= eax | flags; break;
        case Op::Cmpxchg8b: reads |= eax | ebx | ecx | edx; writes |= eax | edx | flags; break;
        case Op::Movs: reads |= esi | edi | ecx; writes |= esi | edi | ecx; break;
        case Op::Cmps: case Op::Scas: reads |= esi | edi | ecx | eax; writes |= esi | edi | ecx | flags; break;
        case Op::Stos: reads |= edi | ecx | eax; writes |= edi | ecx; break;
        case Op::Lods: reads |= esi | ecx; writes |= esi | ecx | eax; break;
        case Op::Ins: reads |= edi | ecx | edx; writes |= edi | ecx; break;
        case Op::Outs: reads |= esi | ecx | edx; writes |= esi | ecx; break;
        case Op::In: reads |= edx; writes |= eax; break;
        case Op::Out: reads |= edx | eax; break;
        case Op::Cpuid: reads |= eax; writes |= eax | ebx | ecx | edx; break;
        case Op::Rdtsc: case Op::Rdpmc: writes |= eax | edx; break;
        case Op::Rdmsr: reads |= ecx; writes |= eax | edx; break;
        case Op::Wrmsr: reads |= ecx | eax | edx; break;
        case Op::Sysenter: writes |= esp; break;
        case Op::Sysexit: reads |= ecx | edx; writes |= esp; break;
        case Op::Arpl: opnd(in.dst, true, true); opnd(in.src, true, false); writes |= flags; break;
        case Op::Lar: case Op::Lsl: opnd(in.dst, false, true); opnd(in.src, true, false); writes |= flags; break;
        case Op::Verr: case Op::Verw: opnd(in.dst, true, false); writes |= flags; break;
        case Op::Bound: opnd(in.dst, true, false); break;
        case Op::X87: {
            const u32 st0 = 1u << (kRegX87 + (top & 7));
            reads |= st0;
            writes |= st0;
            if (!in.mem) {
                const u32 sti = 1u << (kRegX87 + ((top + in.rm) & 7));
                reads |= sti;
                writes |= sti;
            }
            if (in.opcode == 0xDA || in.opcode == 0xDB) reads |= flags;  // FCMOVcc
            if ((in.opcode == 0xDB || in.opcode == 0xDF) && !in.mem && in.reg >= 5) writes |= flags;  // FCOMI family
            if (in.opcode == 0xDF && in.modrm == 0xE0) writes |= eax;
            break;
        }
        case Op::Simd:
            opnd(in.dst, true, true);
            opnd(in.src, true, false);
            if (in.opcode == 0x2E || in.opcode == 0x2F) writes |= flags;
            if (in.opcode == 0x6E || in.opcode == 0xC4) opnd(Opnd{OpndKind::Reg, in.rm}, true, false);
            if (in.opcode == 0x7E || in.opcode == 0xC5 || in.opcode == 0xD7 || in.opcode == 0x50) {
                if (!in.mem) writes |= gpr(in.opcode == 0x7E ? in.rm : in.reg);
            }
            break;
        default:
            opnd(in.dst, true, true);
            opnd(in.src, true, false);
            break;
    }
}
}  // namespace

void Cpu::account(bool interrupt_delivery) {
    InsnTiming t{};
    t.in = &insn_;
    t.interrupt_delivery = interrupt_delivery;
    if (!interrupt_delivery) {
        t.fetch_phys = fetch_phys_;
        t.len = insn_.len;
        regs_of(insn_, top_before_, t.reads, t.writes);
        switch (insn_.op) {
            case Op::Jmp: case Op::Jcc: case Op::JmpRm: case Op::JmpFar: case Op::JmpFarRm: case Op::Loop: case Op::Loope: case Op::Loopne: case Op::Jcxz:
                t.is_branch = true;
                break;
            case Op::Call: case Op::CallRm: case Op::CallFar: case Op::CallFarRm:
                t.is_branch = true;
                t.is_call = true;
                break;
            case Op::Ret: case Op::Retf: case Op::Iret:
                t.is_branch = true;
                t.is_ret = true;
                break;
            default: break;
        }
        if (t.is_branch) {
            const u32 fallthrough = (insn_eip_ + insn_.len) & (st_.code32() ? 0xFFFFFFFFu : 0xFFFFu);
            t.taken = st_.eip != fallthrough || last_faulted_;
            t.branch_eip = insn_eip_;
            t.target = st_.eip;
            t.is_indirect = insn_.op == Op::JmpRm || insn_.op == Op::CallRm || insn_.op == Op::JmpFarRm || insn_.op == Op::CallFarRm;
        }
        static_assert(sizeof(Memory::AccessRecord) == sizeof(MemAccess));
        t.accesses = reinterpret_cast<const MemAccess*>(mem_.accesses());
        t.naccesses = mem_.access_count();
        t.rep_iterations = rep_iterations_;
    }
    cycles_ = timing_.retire(t);
}

// ---------------------------------------------------------------------------------------------------------------
// Operands

u32 Cpu::effective_address(const Insn& in) const {
    u32 ea = static_cast<u32>(in.disp);
    if (in.base != Insn::kNoReg) ea += st_.gpr[in.base];
    if (in.index != Insn::kNoReg) ea += st_.gpr[in.index] << in.scale;
    if (in.adsize == 2) ea &= 0xFFFFu;
    return ea;
}

u32 Cpu::linear(u8 seg, u32 offset, unsigned size, Access access) {
    const Segment& s = st_.seg[seg];
    const u8 vec = seg == kSs ? kSS : kGP;
    if (st_.protected_mode() && !st_.v8086()) {
        if (s.unusable()) raise(vec, 0);
        if (access == Access::Write && !s.writable()) raise(vec, 0);
        if (access == Access::Read && !s.readable()) raise(vec, 0);
        if (access == Access::Fetch && !s.code()) raise(kGP, 0);
        const u32 last = offset + size - 1;
        if (last < offset) raise(vec, 0);
        if (s.expand_down()) {
            const u32 upper = s.db() ? 0xFFFFFFFFu : 0xFFFFu;
            if (offset <= s.limit || last > upper) raise(vec, 0);
        } else if (last > s.limit) {
            raise(vec, 0);
        }
        return s.base + offset;
    }
    const u32 last = offset + size - 1;
    if (last < offset || last > s.limit) raise(vec, 0);
    return s.base + offset;
}

u32 Cpu::read_op(const Insn& in, const Opnd& o, unsigned size) {
    switch (o.kind) {
        case OpndKind::Reg: return st_.reg(o.reg, size);
        case OpndKind::Mem: return mem_.read(linear(in.mem_seg, effective_address(in), size, Access::Read), size);
        case OpndKind::Imm: return in.imm & size_mask(size);
        case OpndKind::Seg: return st_.seg[o.reg].selector;
        case OpndKind::Cr: return read_cr(o.reg);
        case OpndKind::Dr: {
            if (st_.cpl() != 0) raise_gp(0);
            unsigned n = o.reg;
            if ((n == 4 || n == 5)) {
                if (st_.cr4 & cr4::DE) raise_ud();
                n += 2;
            }
            return st_.dr[n];
        }
        case OpndKind::None:
        case OpndKind::Mmx:
        case OpndKind::Xmm: break;
    }
    raise_ud();
}

void Cpu::write_op(const Insn& in, const Opnd& o, unsigned size, u32 value) {
    switch (o.kind) {
        case OpndKind::Reg: st_.set_reg(o.reg, size, value); return;
        case OpndKind::Mem: mem_.write(linear(in.mem_seg, effective_address(in), size, Access::Write), size, value); return;
        case OpndKind::Seg: load_segment(o.reg, static_cast<u16>(value)); return;
        case OpndKind::Cr: write_cr(o.reg, value); return;
        case OpndKind::Dr: {
            if (st_.cpl() != 0) raise_gp(0);
            unsigned n = o.reg;
            if ((n == 4 || n == 5)) {
                if (st_.cr4 & cr4::DE) raise_ud();
                n += 2;
            }
            if (n == 6) value = (value & 0xE00Fu) | 0xFFFF0FF0u;
            if (n == 7) value = (value & 0xFFFF23FFu) | 0x400u;
            st_.dr[n] = value;
            return;
        }
        case OpndKind::Imm:
        case OpndKind::None:
        case OpndKind::Mmx:
        case OpndKind::Xmm: break;
    }
    raise_ud();
}

void Cpu::push(unsigned size, u32 value) {
    const u32 sp = (stack_pointer() - size) & (st_.stack32() ? 0xFFFFFFFFu : 0xFFFFu);
    mem_.write(linear(kSs, sp, size, Access::Write), size, value);
    set_stack_pointer(sp);
}

u32 Cpu::pop(unsigned size) {
    const u32 sp = stack_pointer();
    const u32 v = mem_.read(linear(kSs, sp, size, Access::Read), size);
    set_stack_pointer((sp + size) & (st_.stack32() ? 0xFFFFFFFFu : 0xFFFFu));
    return v;
}

bool Cpu::condition(u8 cc) const {
    const u32 f = st_.eflags();
    const bool cf = f & fl::CF, zf = f & fl::ZF, sf = f & fl::SF, of = f & fl::OF, pf = f & fl::PF;
    bool r;
    switch (cc >> 1) {
        case 0: r = of; break;
        case 1: r = cf; break;
        case 2: r = zf; break;
        case 3: r = cf || zf; break;
        case 4: r = sf; break;
        case 5: r = pf; break;
        case 6: r = sf != of; break;
        default: r = zf || (sf != of); break;
    }
    return (cc & 1) ? !r : r;
}

// ---------------------------------------------------------------------------------------------------------------
// The step

void Cpu::step() {
    last_rdtsc_ = last_cpuid_ = last_faulted_ = last_rep_step_ = last_delivered_ = false;
    st_.undefined_flags = 0;
    st_.inhibit_interrupts = false;
    insn_eip_ = st_.eip;
    insn_cs_ = st_.seg[kCs].selector;
    mem_.begin_instruction();
    rep_iterations_ = 0;
    top_before_ = st_.fpu.top();
    try {
        Fetcher f(*this);
        decode(f, st_.code32(), insn_);
        // The whole instruction must lie inside the code segment.
        if (insn_.len > 1) linear(kCs, st_.eip + insn_.len - 1, 1, Access::Fetch);
        st_.eip += insn_.len;
        if (!st_.code32()) st_.eip &= 0xFFFFu;
        execute(insn_);
        ++instructions_;
        account(false);
    } catch (const Fault& fault) {
        st_.eip = insn_eip_;
        last_faulted_ = true;
        ++instructions_;
        account(false);
        fault_entry(fault);
        account(true);
    }
}

void Cpu::fault_entry(const Fault& f) {
    trace::log(trace::Cat::Cpu, "fault {:02x} error {:08x} at {:04x}:{:08x} ({})", f.vector, f.error_code, insn_cs_, insn_eip_, op_name(insn_.op));
    try {
        deliver(f.vector, false, false, f.has_error, f.error_code);
    } catch (const Fault&) {
        try {
            deliver(kDF, false, false, true, 0);
        } catch (const Fault&) {
            triple_fault_ = true;
        }
    }
}

bool Cpu::take_interrupt() {
    if (st_.inhibit_interrupts) return false;
    if (!(st_.flags & fl::IF)) return false;
    if (!irq_pending_ || irq_pending_() < 0) return false;
    const int v = irq_ack_();
    if (v < 0) return false;
    deliver_external(static_cast<u8>(v));
    return true;
}

void Cpu::deliver_external(u8 vector) {
    st_.halted = false;
    last_delivered_ = true;
    last_vector_ = vector;
    insn_eip_ = st_.eip;
    insn_cs_ = st_.seg[kCs].selector;
    mem_.begin_instruction();
    try {
        deliver(vector, true, false, false, 0);
    } catch (const Fault& f) {
        st_.eip = insn_eip_;
        fault_entry(f);
    }
    account(true);
}

std::string Cpu::last_bytes() {
    std::string s;
    // Re-read the bytes from memory without faulting (the page is mapped: it was just fetched).
    try {
        for (unsigned i = 0; i < insn_.len; ++i) {
            const u32 lin = st_.seg[kCs].base + insn_eip_ + i;
            s += std::format("{:02x}", mem_.read(lin, 1));
        }
    } catch (const Fault&) {
        s += "?";
    }
    return s;
}

std::string Cpu::describe() const {
    const CpuState& s = st_;
    return std::format("cs {:04x} eip {:08x} eflags {:08x} eax {:08x} ebx {:08x} ecx {:08x} edx {:08x} esi {:08x} edi {:08x} ebp {:08x} esp {:08x} cr0 {:08x} cr3 {:08x}",
                       s.seg[kCs].selector, s.eip, s.eflags(), s.gpr[kEax], s.gpr[kEbx], s.gpr[kEcx], s.gpr[kEdx], s.gpr[kEsi], s.gpr[kEdi],
                       s.gpr[kEbp], s.gpr[kEsp], s.cr0, s.cr3);
}

// ---------------------------------------------------------------------------------------------------------------
// Execution

namespace {
// A resolved destination: a register, or a memory location whose write permission is already checked.
struct Dst {
    bool mem;
    u32 lin;
    u8 reg;
};
}  // namespace

static Dst resolve_rmw(Cpu& cpu, const Insn& in, unsigned size) {
    if (in.dst.kind == OpndKind::Mem) {
        const u32 lin = cpu.linear(in.mem_seg, cpu.effective_address(in), size, Access::Write);
        cpu.memory().probe_write(lin, size);
        return Dst{true, lin, 0};
    }
    return Dst{false, 0, in.dst.reg};
}

static u32 load_dst(Cpu& cpu, const Dst& d, unsigned size) {
    return d.mem ? cpu.memory().read(d.lin, size) : cpu.state().reg(d.reg, size);
}

static void store_dst(Cpu& cpu, const Dst& d, unsigned size, u32 v) {
    if (d.mem) cpu.memory().write(d.lin, size, v & size_mask(size));
    else cpu.state().set_reg(d.reg, size, v & size_mask(size));
}

void Cpu::exec_alu(const Insn& in) {
    const unsigned size = in.size;
    const u32 mask = size_mask(size);
    if (in.op == Op::Cmp || in.op == Op::Test) {
        const u32 a = read_op(in, in.dst, size) & mask;
        const u32 b = read_op(in, in.src, size) & mask;
        if (in.op == Op::Cmp) st_.lazy(LazyOp::Sub, size, a, b, (a - b) & mask);
        else st_.lazy(LazyOp::Logic, size, a, b, a & b);
        return;
    }
    const Dst d = resolve_rmw(*this, in, size);
    const u32 b = read_op(in, in.src, size) & mask;
    const u32 a = load_dst(*this, d, size) & mask;
    u32 r;
    LazyOp op;
    u8 c = 0;
    switch (in.op) {
        case Op::Add: r = a + b; op = LazyOp::Add; break;
        case Op::Adc: c = st_.cf() ? 1 : 0; r = a + b + c; op = LazyOp::Adc; break;
        case Op::Sub: r = a - b; op = LazyOp::Sub; break;
        case Op::Sbb: c = st_.cf() ? 1 : 0; r = a - b - c; op = LazyOp::Sbb; break;
        case Op::And: r = a & b; op = LazyOp::Logic; break;
        case Op::Or: r = a | b; op = LazyOp::Logic; break;
        case Op::Xor: r = a ^ b; op = LazyOp::Logic; break;
        default: raise_ud();
    }
    r &= mask;
    store_dst(*this, d, size, r);
    st_.lazy(op, size, a, b, r, c);
}

void Cpu::exec_shift(const Insn& in) {
    const unsigned size = in.size;
    const unsigned bits = size * 8;
    const u32 mask = size_mask(size);
    const u32 msb = sign_bit(size);
    const Dst d = resolve_rmw(*this, in, size);
    const unsigned count = read_op(in, in.op == Op::Shld || in.op == Op::Shrd ? in.src2 : in.src, 1) & 0x1F;
    const u32 a = load_dst(*this, d, size) & mask;
    if (count == 0) return;  // flags and the operand are untouched
    u32 r;
    bool cf, of, af = false;
    const bool cf_in = st_.cf();
    switch (in.op) {
        case Op::Shl: {
            const u64 wide = static_cast<u64>(a) << count;
            r = static_cast<u32>(wide) & mask;
            cf = ((wide >> bits) & 1) != 0;
            of = ((r & msb) != 0) != cf;
            if (count != 1) st_.undefined_flags |= fl::OF;
            break;
        }
        case Op::Shr: {
            r = count >= 32 ? 0 : (a >> count) & mask;
            cf = count <= bits && count >= 1 ? ((a >> (count - 1)) & 1) != 0 : false;
            of = ((a & msb) != 0) != ((r & msb) != 0);
            if (count != 1) st_.undefined_flags |= fl::OF;
            break;
        }
        case Op::Sar: {
            const i32 sa = static_cast<i32>(sign_extend(a, size));
            if (count >= bits) {
                r = sa < 0 ? mask : 0;
                cf = sa < 0;
            } else {
                r = static_cast<u32>(sa >> count) & mask;
                cf = ((static_cast<u32>(sa) >> (count - 1)) & 1) != 0;
            }
            of = false;
            if (count != 1) st_.undefined_flags |= fl::OF;
            break;
        }
        case Op::Rol: {
            r = rotl(a, count, bits);
            cf = (r & 1) != 0;
            of = ((r & msb) != 0) != cf;
            if (count != 1) st_.undefined_flags |= fl::OF;
            st_.materialise();
            st_.flags = (st_.flags & ~(fl::CF | fl::OF)) | (cf ? fl::CF : 0) | (of ? fl::OF : 0);
            store_dst(*this, d, size, r);
            return;
        }
        case Op::Ror: {
            r = rotr(a, count, bits);
            cf = (r & msb) != 0;
            of = ((r & msb) != 0) != ((r & (msb >> 1)) != 0);
            if (count != 1) st_.undefined_flags |= fl::OF;
            st_.materialise();
            st_.flags = (st_.flags & ~(fl::CF | fl::OF)) | (cf ? fl::CF : 0) | (of ? fl::OF : 0);
            store_dst(*this, d, size, r);
            return;
        }
        case Op::Rcl: {
            const unsigned n = count % (bits + 1);
            if (n == 0) {
                r = a;
                cf = cf_in;
            } else {
                const u64 wide = (static_cast<u64>(a) | (static_cast<u64>(cf_in ? 1 : 0) << bits));  // bits+1 wide
                const u64 rot = ((wide << n) | (wide >> (bits + 1 - n))) & ((u64{1} << (bits + 1)) - 1);
                r = static_cast<u32>(rot) & mask;
                cf = ((rot >> bits) & 1) != 0;
            }
            of = ((r & msb) != 0) != cf;
            if (count != 1) st_.undefined_flags |= fl::OF;
            st_.materialise();
            st_.flags = (st_.flags & ~(fl::CF | fl::OF)) | (cf ? fl::CF : 0) | (of ? fl::OF : 0);
            store_dst(*this, d, size, r);
            return;
        }
        case Op::Rcr: {
            const unsigned n = count % (bits + 1);
            if (n == 0) {
                r = a;
                cf = cf_in;
            } else {
                const u64 wide = (static_cast<u64>(a) | (static_cast<u64>(cf_in ? 1 : 0) << bits));
                const u64 rot = ((wide >> n) | (wide << (bits + 1 - n))) & ((u64{1} << (bits + 1)) - 1);
                r = static_cast<u32>(rot) & mask;
                cf = ((rot >> bits) & 1) != 0;
            }
            of = ((a & msb) != 0) != cf_in;
            if (count != 1) st_.undefined_flags |= fl::OF;
            st_.materialise();
            st_.flags = (st_.flags & ~(fl::CF | fl::OF)) | (cf ? fl::CF : 0) | (of ? fl::OF : 0);
            store_dst(*this, d, size, r);
            return;
        }
        case Op::Shld: {
            const u32 b = read_op(in, in.src, size) & mask;
            if (count > bits) {
                // Undefined for 16-bit operands; the hardware shifts the 32-bit concatenation.
                st_.undefined_flags |= fl::kArith;
            }
            const unsigned n = count;
            if (n >= bits) {
                const u64 wide = (static_cast<u64>(a) << bits) | b;
                r = static_cast<u32>((wide << (n - bits)) >> 0) & mask;
                r = static_cast<u32>(((static_cast<u64>(a) << bits | b) << n) >> bits) & mask;
                cf = ((static_cast<u64>(b) >> (2 * bits - n)) & 1) != 0;
                (void)wide;
            } else {
                r = ((a << n) | (b >> (bits - n))) & mask;
                cf = ((a >> (bits - n)) & 1) != 0;
            }
            of = ((a & msb) != 0) != ((r & msb) != 0);
            if (count != 1) st_.undefined_flags |= fl::OF;
            break;
        }
        case Op::Shrd: {
            const u32 b = read_op(in, in.src, size) & mask;
            if (count > bits) st_.undefined_flags |= fl::kArith;
            const unsigned n = count;
            if (n >= bits) {
                r = static_cast<u32>(((static_cast<u64>(b) << bits) | a) >> n) & mask;
                cf = (((static_cast<u64>(b) << bits) | a) >> (n - 1) & 1) != 0;
            } else {
                r = ((a >> n) | (b << (bits - n))) & mask;
                cf = ((a >> (n - 1)) & 1) != 0;
            }
            of = ((a & msb) != 0) != ((r & msb) != 0);
            if (count != 1) st_.undefined_flags |= fl::OF;
            break;
        }
        default: raise_ud();
    }
    st_.undefined_flags |= fl::AF;
    store_dst(*this, d, size, r);
    st_.lazy_result(size, r, cf, of, af);
}

void Cpu::exec_string(const Insn& in) {
    const unsigned size = in.size;
    const bool a32 = in.adsize == 4;
    const u32 amask = a32 ? 0xFFFFFFFFu : 0xFFFFu;
    const u8 src_seg = in.seg != kSegNone ? in.seg : static_cast<u8>(kDs);
    const i32 delta = (st_.flags & fl::DF) ? -static_cast<i32>(size) : static_cast<i32>(size);
    auto si = [&] { return st_.gpr[kEsi] & amask; };
    auto di = [&] { return st_.gpr[kEdi] & amask; };
    auto cx = [&] { return st_.gpr[kEcx] & amask; };
    auto advance = [&](unsigned r) { st_.gpr[r] = (st_.gpr[r] & ~amask) | ((st_.gpr[r] + static_cast<u32>(delta)) & amask); };
    auto dec_cx = [&] { st_.gpr[kEcx] = (st_.gpr[kEcx] & ~amask) | ((st_.gpr[kEcx] - 1) & amask); };
    // Under the oracle a REP runs one element per step, like a trap-flag single step does; string I/O is the
    // exception: a hypervisor batches a whole REP INS/OUTS into one exit, so it runs whole here too.
    const bool string_io = in.op == Op::Ins || in.op == Op::Outs;
    const unsigned limit = single_iteration_ && !string_io ? 1 : kRepIterationsPerStep;
    unsigned done = 0;
    for (;;) {
        if (in.rep && cx() == 0) return;
        switch (in.op) {
            case Op::Movs: {
                const u32 dl = linear(kEs, di(), size, Access::Write);
                mem_.probe_write(dl, size);
                const u32 v = mem_.read(linear(src_seg, si(), size, Access::Read), size);
                mem_.write(dl, size, v);
                advance(kEsi);
                advance(kEdi);
                break;
            }
            case Op::Cmps: {
                const u32 a = mem_.read(linear(src_seg, si(), size, Access::Read), size);
                const u32 b = mem_.read(linear(kEs, di(), size, Access::Read), size);
                st_.lazy(LazyOp::Sub, size, a, b, (a - b) & size_mask(size));
                advance(kEsi);
                advance(kEdi);
                break;
            }
            case Op::Stos:
                mem_.write(linear(kEs, di(), size, Access::Write), size, st_.reg(kEax, size));
                advance(kEdi);
                break;
            case Op::Lods:
                st_.set_reg(kEax, size, mem_.read(linear(src_seg, si(), size, Access::Read), size));
                advance(kEsi);
                break;
            case Op::Scas: {
                const u32 a = st_.reg(kEax, size);
                const u32 b = mem_.read(linear(kEs, di(), size, Access::Read), size);
                st_.lazy(LazyOp::Sub, size, a, b, (a - b) & size_mask(size));
                advance(kEdi);
                break;
            }
            case Op::Ins: {
                const u32 dl = linear(kEs, di(), size, Access::Write);
                mem_.probe_write(dl, size);
                const u32 v = mem_.io_read(st_.reg16(kEdx), size);
                mem_.write(dl, size, v);
                advance(kEdi);
                break;
            }
            case Op::Outs: {
                const u32 v = mem_.read(linear(src_seg, si(), size, Access::Read), size);
                mem_.io_write(st_.reg16(kEdx), size, v);
                advance(kEsi);
                break;
            }
            default: raise_ud();
        }
        ++rep_iterations_;
        if (!in.rep) return;
        dec_cx();
        if (in.op == Op::Cmps || in.op == Op::Scas) {
            const bool zf = st_.flag(fl::ZF);
            if (in.rep == 1 && !zf) return;
            if (in.rep == 2 && zf) return;
        }
        if (++done >= limit && cx() != 0) {
            // Not finished: the instruction resumes at the next step (interrupts are taken in between).
            st_.eip = insn_eip_;
            last_rep_step_ = true;
            return;
        }
    }
}

void Cpu::execute(const Insn& in) {
    const unsigned size = in.size;
    const u32 mask = size_mask(size);
    switch (in.op) {
        case Op::Add: case Op::Or: case Op::Adc: case Op::Sbb: case Op::And: case Op::Sub: case Op::Xor: case Op::Cmp: case Op::Test:
            exec_alu(in);
            return;
        case Op::Rol: case Op::Ror: case Op::Rcl: case Op::Rcr: case Op::Shl: case Op::Shr: case Op::Sar: case Op::Shld: case Op::Shrd:
            exec_shift(in);
            return;
        case Op::Movs: case Op::Cmps: case Op::Stos: case Op::Lods: case Op::Scas: case Op::Ins: case Op::Outs:
            if ((in.op == Op::Ins || in.op == Op::Outs) && st_.protected_mode() && st_.cpl() > st_.iopl()) raise_gp(0);
            exec_string(in);
            return;

        case Op::Mov: {
            const u32 v = read_op(in, in.src, size);
            write_op(in, in.dst, size, v);
            return;
        }
        case Op::MovFromSeg: {
            const u32 v = st_.seg[in.src.reg].selector;
            if (in.dst.kind == OpndKind::Mem) write_op(in, in.dst, 2, v);
            else write_op(in, in.dst, size, v);
            return;
        }
        case Op::MovSeg: {
            const u32 v = read_op(in, in.src, 2);
            load_segment(in.dst.reg, static_cast<u16>(v));
            if (in.dst.reg == kSs) st_.inhibit_interrupts = true;
            return;
        }
        case Op::Xchg: {
            Opnd src = in.src;
            if (src.kind == OpndKind::None) src = Opnd{OpndKind::Reg, kEax};
            if (in.dst.kind == OpndKind::Mem) {
                const Dst d = resolve_rmw(*this, in, size);
                const u32 a = load_dst(*this, d, size);
                const u32 b = read_op(in, src, size);
                store_dst(*this, d, size, b);
                write_op(in, src, size, a);
            } else {
                const u32 a = read_op(in, in.dst, size);
                const u32 b = read_op(in, src, size);
                write_op(in, in.dst, size, b);
                write_op(in, src, size, a);
            }
            return;
        }
        case Op::Lea: write_op(in, in.dst, size, effective_address(in) & mask); return;
        case Op::Movzx: write_op(in, in.dst, size, read_op(in, in.src, in.src_size)); return;
        case Op::Movsx: write_op(in, in.dst, size, sign_extend(read_op(in, in.src, in.src_size), in.src_size) & mask); return;
        case Op::Lds: case Op::Les: case Op::Lfs: case Op::Lgs: case Op::Lss: {
            const u32 lin = linear(in.mem_seg, effective_address(in), in.src_size, Access::Read);
            const u32 off = mem_.read(lin, size);
            const u16 sel = static_cast<u16>(mem_.read(lin + size, 2));
            const u8 seg = in.op == Op::Lds ? kDs : in.op == Op::Les ? kEs : in.op == Op::Lfs ? kFs : in.op == Op::Lgs ? kGs : kSs;
            load_segment(seg, sel);
            st_.set_reg(in.dst.reg, size, off);
            if (seg == kSs) st_.inhibit_interrupts = true;
            return;
        }
        case Op::Xlat: {
            const u32 base = in.adsize == 4 ? st_.gpr[kEbx] + st_.reg8(kEax) : ((st_.reg16(kEbx) + st_.reg8(kEax)) & 0xFFFFu);
            const u8 seg = in.seg != kSegNone ? in.seg : static_cast<u8>(kDs);
            st_.set_reg8(kEax, static_cast<u8>(mem_.read(linear(seg, base, 1, Access::Read), 1)));
            return;
        }
        case Op::Cbw:
            if (size == 2) st_.set_reg16(kEax, static_cast<u16>(sign_extend(st_.reg8(kEax), 1)));
            else st_.gpr[kEax] = sign_extend(st_.reg16(kEax), 2);
            return;
        case Op::Cwd:
            if (size == 2) st_.set_reg16(kEdx, (st_.reg16(kEax) & 0x8000) ? 0xFFFFu : 0u);
            else st_.gpr[kEdx] = (st_.gpr[kEax] & 0x80000000u) ? 0xFFFFFFFFu : 0u;
            return;
        case Op::Bswap: {
            const u32 v = st_.gpr[in.dst.reg];
            if (size == 4) st_.gpr[in.dst.reg] = std::byteswap(v);
            else st_.set_reg16(in.dst.reg, 0);  // undefined; the hardware zeroes it
            return;
        }

        case Op::Inc: case Op::Dec: {
            const Dst d = resolve_rmw(*this, in, size);
            const u32 a = load_dst(*this, d, size) & mask;
            const bool cf = st_.cf();
            const u32 r = (in.op == Op::Inc ? a + 1 : a - 1) & mask;
            const bool of = in.op == Op::Inc ? r == sign_bit(size) : a == sign_bit(size);
            const bool af = in.op == Op::Inc ? (a & 0xF) == 0xF : (a & 0xF) == 0;
            store_dst(*this, d, size, r);
            st_.lazy_result(size, r, cf, of, af);
            return;
        }
        case Op::Not: {
            const Dst d = resolve_rmw(*this, in, size);
            store_dst(*this, d, size, ~load_dst(*this, d, size));
            return;
        }
        case Op::Neg: {
            const Dst d = resolve_rmw(*this, in, size);
            const u32 a = load_dst(*this, d, size) & mask;
            const u32 r = (0 - a) & mask;
            store_dst(*this, d, size, r);
            st_.lazy(LazyOp::Sub, size, 0, a, r);
            return;
        }
        case Op::Mul: {
            const u32 b = read_op(in, in.dst, size) & mask;
            st_.undefined_flags |= fl::SF | fl::ZF | fl::PF | fl::AF;
            if (size == 1) {
                const u32 r = st_.reg8(kEax) * b;
                st_.set_reg16(kEax, static_cast<u16>(r));
                st_.lazy_result(1, r & 0xFF, (r >> 8) != 0, (r >> 8) != 0, false);
            } else if (size == 2) {
                const u32 r = st_.reg16(kEax) * b;
                st_.set_reg16(kEax, static_cast<u16>(r));
                st_.set_reg16(kEdx, static_cast<u16>(r >> 16));
                st_.lazy_result(2, r & 0xFFFF, (r >> 16) != 0, (r >> 16) != 0, false);
            } else {
                const u64 r = static_cast<u64>(st_.gpr[kEax]) * b;
                st_.gpr[kEax] = static_cast<u32>(r);
                st_.gpr[kEdx] = static_cast<u32>(r >> 32);
                st_.lazy_result(4, static_cast<u32>(r), (r >> 32) != 0, (r >> 32) != 0, false);
            }
            return;
        }
        case Op::Imul1: {
            const i32 b = static_cast<i32>(sign_extend(read_op(in, in.dst, size) & mask, size));
            st_.undefined_flags |= fl::SF | fl::ZF | fl::PF | fl::AF;
            if (size == 1) {
                const i32 r = static_cast<i32>(static_cast<i8>(st_.reg8(kEax))) * b;
                st_.set_reg16(kEax, static_cast<u16>(r));
                const bool ov = r != static_cast<i8>(r);
                st_.lazy_result(1, static_cast<u32>(r) & 0xFF, ov, ov, false);
            } else if (size == 2) {
                const i32 r = static_cast<i32>(static_cast<i16>(st_.reg16(kEax))) * b;
                st_.set_reg16(kEax, static_cast<u16>(r));
                st_.set_reg16(kEdx, static_cast<u16>(static_cast<u32>(r) >> 16));
                const bool ov = r != static_cast<i16>(r);
                st_.lazy_result(2, static_cast<u32>(r) & 0xFFFF, ov, ov, false);
            } else {
                const i64 r = static_cast<i64>(static_cast<i32>(st_.gpr[kEax])) * b;
                st_.gpr[kEax] = static_cast<u32>(static_cast<u64>(r));
                st_.gpr[kEdx] = static_cast<u32>(static_cast<u64>(r) >> 32);
                const bool ov = r != static_cast<i32>(r);
                st_.lazy_result(4, static_cast<u32>(static_cast<u64>(r)), ov, ov, false);
            }
            return;
        }
        case Op::Imul2: case Op::Imul3: {
            const i64 a = static_cast<i32>(sign_extend(read_op(in, in.src, size) & mask, size));
            const i64 b = in.op == Op::Imul2 ? static_cast<i32>(sign_extend(st_.reg(in.dst.reg, size), size))
                                             : static_cast<i32>(sign_extend(in.imm & mask, size));
            const i64 r = a * b;
            const u32 low = static_cast<u32>(static_cast<u64>(r)) & mask;
            const bool ov = r != static_cast<i64>(static_cast<i32>(sign_extend(low, size)));
            st_.set_reg(in.dst.reg, size, low);
            st_.undefined_flags |= fl::SF | fl::ZF | fl::PF | fl::AF;
            st_.lazy_result(size, low, ov, ov, false);
            return;
        }
        case Op::Div: {
            const u32 b = read_op(in, in.dst, size) & mask;
            if (b == 0) raise(kDE);
            st_.undefined_flags |= fl::kArith;
            if (size == 1) {
                const u32 a = st_.reg16(kEax);
                const u32 q = a / b;
                if (q > 0xFF) raise(kDE);
                st_.set_reg8(kEax, static_cast<u8>(q));
                st_.set_reg8(kEax + 4, static_cast<u8>(a % b));
            } else if (size == 2) {
                const u32 a = (static_cast<u32>(st_.reg16(kEdx)) << 16) | st_.reg16(kEax);
                const u32 q = a / b;
                if (q > 0xFFFF) raise(kDE);
                st_.set_reg16(kEax, static_cast<u16>(q));
                st_.set_reg16(kEdx, static_cast<u16>(a % b));
            } else {
                const u64 a = (static_cast<u64>(st_.gpr[kEdx]) << 32) | st_.gpr[kEax];
                const u64 q = a / b;
                if (q > 0xFFFFFFFFu) raise(kDE);
                st_.gpr[kEax] = static_cast<u32>(q);
                st_.gpr[kEdx] = static_cast<u32>(a % b);
            }
            return;
        }
        case Op::Idiv: {
            const u32 bu = read_op(in, in.dst, size) & mask;
            if (bu == 0) raise(kDE);
            st_.undefined_flags |= fl::kArith;
            if (size == 1) {
                const i32 a = static_cast<i16>(st_.reg16(kEax));
                const i32 b = static_cast<i8>(bu);
                const i32 q = a / b;
                if (q > 127 || q < -128) raise(kDE);
                st_.set_reg8(kEax, static_cast<u8>(q));
                st_.set_reg8(kEax + 4, static_cast<u8>(a % b));
            } else if (size == 2) {
                const i32 a = static_cast<i32>((static_cast<u32>(st_.reg16(kEdx)) << 16) | st_.reg16(kEax));
                const i32 b = static_cast<i16>(bu);
                const i32 q = a / b;
                if (q > 32767 || q < -32768) raise(kDE);
                st_.set_reg16(kEax, static_cast<u16>(q));
                st_.set_reg16(kEdx, static_cast<u16>(a % b));
            } else {
                const i64 a = static_cast<i64>((static_cast<u64>(st_.gpr[kEdx]) << 32) | st_.gpr[kEax]);
                const i64 b = static_cast<i32>(bu);
                if (a == INT64_MIN && b == -1) raise(kDE);
                const i64 q = a / b;
                if (q > 0x7FFFFFFFll || q < -0x80000000ll) raise(kDE);
                st_.gpr[kEax] = static_cast<u32>(static_cast<u64>(q));
                st_.gpr[kEdx] = static_cast<u32>(static_cast<u64>(a % b));
            }
            return;
        }

        case Op::Push: {
            const Opnd o = in.src.kind != OpndKind::None ? in.src : in.dst;
            push(size, read_op(in, o, size));
            return;
        }
        case Op::Pop: {
            if (in.dst.kind == OpndKind::Mem) {
                // The address uses the stack pointer after the pop; the pop happens first.
                const u32 v = pop(size);
                try {
                    write_op(in, in.dst, size, v);
                } catch (const Fault&) {
                    set_stack_pointer(stack_pointer() - size);
                    throw;
                }
            } else {
                st_.set_reg(in.dst.reg, size, pop(size));
            }
            return;
        }
        case Op::PushSeg: push(size, st_.seg[in.dst.reg].selector); return;
        case Op::PopSeg: {
            const u32 sp = stack_pointer();
            const u16 sel = static_cast<u16>(mem_.read(linear(kSs, sp, size, Access::Read), size));
            load_segment(in.dst.reg, sel);
            set_stack_pointer((sp + size) & (st_.stack32() ? 0xFFFFFFFFu : 0xFFFFu));
            if (in.dst.reg == kSs) st_.inhibit_interrupts = true;
            return;
        }
        case Op::Pusha: {
            const u32 sp = st_.reg(kEsp, size);
            const std::array<u32, 8> saved = st_.gpr;
            try {
                push(size, saved[kEax]); push(size, saved[kEcx]); push(size, saved[kEdx]); push(size, saved[kEbx]);
                push(size, sp); push(size, saved[kEbp]); push(size, saved[kEsi]); push(size, saved[kEdi]);
            } catch (const Fault&) {
                st_.gpr = saved;
                throw;
            }
            return;
        }
        case Op::Popa: {
            const std::array<u32, 8> saved = st_.gpr;
            try {
                st_.set_reg(kEdi, size, pop(size)); st_.set_reg(kEsi, size, pop(size)); st_.set_reg(kEbp, size, pop(size));
                set_stack_pointer(stack_pointer() + size);
                st_.set_reg(kEbx, size, pop(size)); st_.set_reg(kEdx, size, pop(size)); st_.set_reg(kEcx, size, pop(size));
                st_.set_reg(kEax, size, pop(size));
            } catch (const Fault&) {
                st_.gpr = saved;
                throw;
            }
            return;
        }
        case Op::Pushf: {
            if (st_.v8086() && st_.iopl() < 3) raise_gp(0);
            push(size, st_.eflags() & ~(fl::VM | fl::RF));
            return;
        }
        case Op::Popf: {
            if (st_.v8086() && st_.iopl() < 3) raise_gp(0);
            const u32 v = pop(size);
            const u32 old = st_.eflags();
            u32 writable = fl::kWritable;
            if (size == 2) writable &= 0xFFFFu;
            if (st_.protected_mode()) {
                if (st_.cpl() > 0) writable &= ~fl::IOPL;
                if (st_.cpl() > st_.iopl()) writable &= ~fl::IF;
            }
            st_.set_eflags((old & ~writable) | (v & writable));
            return;
        }
        case Op::Enter: {
            const u32 alloc = in.imm;
            const unsigned level = in.imm2 & 0x1F;
            const bool s32 = st_.stack32();
            const u32 smask = s32 ? 0xFFFFFFFFu : 0xFFFFu;
            const u32 saved_sp = st_.gpr[kEsp], saved_bp = st_.gpr[kEbp];
            try {
                push(size, st_.reg(kEbp, size));
                const u32 frame = stack_pointer();
                if (level > 0) {
                    for (unsigned i = 1; i < level; ++i) {
                        const u32 bp = (st_.reg(kEbp, s32 ? 4 : 2) - size) & smask;
                        st_.gpr[kEbp] = s32 ? bp : ((st_.gpr[kEbp] & 0xFFFF0000u) | bp);
                        push(size, mem_.read(linear(kSs, bp, size, Access::Read), size));
                    }
                    push(size, frame);
                }
                if (size == 4) st_.gpr[kEbp] = frame;
                else st_.set_reg16(kEbp, static_cast<u16>(frame));
                set_stack_pointer((stack_pointer() - alloc) & smask);
            } catch (const Fault&) {
                st_.gpr[kEsp] = saved_sp;
                st_.gpr[kEbp] = saved_bp;
                throw;
            }
            return;
        }
        case Op::Leave: {
            const u32 bp = st_.stack32() ? st_.gpr[kEbp] : st_.reg16(kEbp);
            const u32 v = mem_.read(linear(kSs, bp, size, Access::Read), size);
            set_stack_pointer(bp + size);
            st_.set_reg(kEbp, size, v);
            return;
        }

        case Op::Jmp: st_.eip = (st_.eip + in.imm) & (size == 2 ? 0xFFFFu : 0xFFFFFFFFu); return;
        case Op::Jcc:
            if (condition(in.cc)) st_.eip = (st_.eip + in.imm) & (size == 2 ? 0xFFFFu : 0xFFFFFFFFu);
            return;
        case Op::Call: {
            const u32 target = (st_.eip + in.imm) & (size == 2 ? 0xFFFFu : 0xFFFFFFFFu);
            push(size, st_.eip);
            st_.eip = target;
            return;
        }
        case Op::JmpRm: st_.eip = read_op(in, in.dst, size) & mask; return;
        case Op::CallRm: {
            const u32 target = read_op(in, in.dst, size) & mask;
            push(size, st_.eip);
            st_.eip = target;
            return;
        }
        case Op::JmpFar: jump_far(static_cast<u16>(in.imm2), in.imm & mask, in); return;
        case Op::CallFar: call_far(static_cast<u16>(in.imm2), in.imm & mask, in); return;
        case Op::JmpFarRm: case Op::CallFarRm: {
            const u32 lin = linear(in.mem_seg, effective_address(in), size + 2, Access::Read);
            const u32 off = mem_.read(lin, size);
            const u16 sel = static_cast<u16>(mem_.read(lin + size, 2));
            if (in.op == Op::JmpFarRm) jump_far(sel, off, in);
            else call_far(sel, off, in);
            return;
        }
        case Op::Ret: {
            const u32 target = pop(size) & mask;
            set_stack_pointer(stack_pointer() + in.imm);
            st_.eip = target;
            return;
        }
        case Op::Retf: ret_far(size, in.imm); return;
        case Op::Iret: iret(size); return;
        case Op::Loop: case Op::Loope: case Op::Loopne: case Op::Jcxz: {
            const u32 amask = in.adsize == 4 ? 0xFFFFFFFFu : 0xFFFFu;
            u32 count = st_.gpr[kEcx] & amask;
            bool taken;
            if (in.op == Op::Jcxz) {
                taken = count == 0;
            } else {
                count = (count - 1) & amask;
                st_.gpr[kEcx] = (st_.gpr[kEcx] & ~amask) | count;
                taken = count != 0;
                if (in.op == Op::Loope) taken = taken && st_.flag(fl::ZF);
                if (in.op == Op::Loopne) taken = taken && !st_.flag(fl::ZF);
            }
            if (taken) st_.eip = (st_.eip + in.imm) & (size == 2 ? 0xFFFFu : 0xFFFFFFFFu);
            return;
        }
        case Op::Int: deliver(static_cast<u8>(in.imm), false, true, false, 0); return;
        case Op::Int3: deliver(kBP, false, true, false, 0); return;
        case Op::Into:
            if (st_.flag(fl::OF)) deliver(kOF, false, true, false, 0);
            return;
        case Op::Int1: deliver(kDB, false, true, false, 0); return;

        case Op::Cli:
            if (st_.protected_mode() && st_.cpl() > st_.iopl()) raise_gp(0);
            st_.materialise();
            st_.flags &= ~fl::IF;
            return;
        case Op::Sti: {
            if (st_.protected_mode() && st_.cpl() > st_.iopl()) raise_gp(0);
            st_.materialise();
            if (!(st_.flags & fl::IF)) st_.inhibit_interrupts = true;
            st_.flags |= fl::IF;
            return;
        }
        case Op::Cld: st_.materialise(); st_.flags &= ~fl::DF; return;
        case Op::Std: st_.materialise(); st_.flags |= fl::DF; return;
        case Op::Clc: st_.materialise(); st_.flags &= ~fl::CF; return;
        case Op::Stc: st_.materialise(); st_.flags |= fl::CF; return;
        case Op::Cmc: st_.materialise(); st_.flags ^= fl::CF; return;
        case Op::Sahf: {
            st_.materialise();
            const u32 ah = st_.reg8(kEax + 4);
            st_.flags = (st_.flags & ~(fl::SF | fl::ZF | fl::AF | fl::PF | fl::CF)) | (ah & (fl::SF | fl::ZF | fl::AF | fl::PF | fl::CF));
            return;
        }
        case Op::Lahf: st_.set_reg8(kEax + 4, static_cast<u8>(st_.eflags() & 0xFF)); return;
        case Op::Setcc: write_op(in, in.dst, 1, condition(in.cc) ? 1 : 0); return;
        case Op::Cmovcc: {
            const u32 v = read_op(in, in.src, size);
            if (condition(in.cc)) st_.set_reg(in.dst.reg, size, v);
            return;
        }
        case Op::Salc: st_.set_reg8(kEax, st_.cf() ? 0xFF : 0); return;

        case Op::Daa: case Op::Das: {
            const u32 f = st_.eflags();
            u32 al = st_.reg8(kEax);
            const u32 old_al = al;
            const bool old_cf = f & fl::CF;
            bool cf = false, af = false;
            if ((al & 0xF) > 9 || (f & fl::AF)) {
                if (in.op == Op::Daa) {
                    al += 6;
                    cf = old_cf || al > 0xFF;
                } else {
                    cf = old_cf || old_al < 6;
                    al -= 6;
                }
                af = true;
            }
            if (old_al > 0x99 || old_cf) {
                al = in.op == Op::Daa ? al + 0x60 : al - 0x60;
                cf = true;
            }
            al &= 0xFF;
            st_.set_reg8(kEax, static_cast<u8>(al));
            st_.undefined_flags |= fl::OF;
            st_.lazy_result(1, al, cf, false, af);
            return;
        }
        case Op::Aaa: case Op::Aas: {
            const u32 f = st_.eflags();
            u32 al = st_.reg8(kEax), ah = st_.reg8(kEax + 4);
            bool cf = false, af = false;
            if ((al & 0xF) > 9 || (f & fl::AF)) {
                if (in.op == Op::Aaa) {
                    al += 6;
                    ah += 1;
                } else {
                    al -= 6;
                    ah -= 1;
                }
                cf = af = true;
            }
            al &= 0xF;
            st_.set_reg8(kEax, static_cast<u8>(al));
            st_.set_reg8(kEax + 4, static_cast<u8>(ah));
            st_.undefined_flags |= fl::OF | fl::SF | fl::ZF | fl::PF;
            st_.lazy_result(1, al, cf, false, af);
            return;
        }
        case Op::Aam: {
            const u32 base = in.imm & 0xFF;
            if (base == 0) raise(kDE);
            const u32 al = st_.reg8(kEax);
            st_.set_reg8(kEax + 4, static_cast<u8>(al / base));
            st_.set_reg8(kEax, static_cast<u8>(al % base));
            st_.undefined_flags |= fl::OF | fl::AF | fl::CF;
            st_.lazy_result(1, al % base, false, false, false);
            return;
        }
        case Op::Aad: {
            const u32 base = in.imm & 0xFF;
            const u32 al = (st_.reg8(kEax) + st_.reg8(kEax + 4) * base) & 0xFF;
            st_.set_reg8(kEax, static_cast<u8>(al));
            st_.set_reg8(kEax + 4, 0);
            st_.undefined_flags |= fl::OF | fl::AF | fl::CF;
            st_.lazy_result(1, al, false, false, false);
            return;
        }

        case Op::Bt: case Op::Bts: case Op::Btr: case Op::Btc: {
            const unsigned bits = size * 8;
            u32 bitno;
            Dst d;
            if (in.dst.kind == OpndKind::Mem && in.src.kind == OpndKind::Reg) {
                // A register bit offset addresses memory beyond the operand: signed, in operand-size units.
                const i32 off = static_cast<i32>(sign_extend(st_.reg(in.src.reg, size), size));
                const i32 unit = static_cast<i32>(bits);
                i32 index = off / unit;
                i32 rem = off % unit;
                if (rem < 0) {
                    rem += unit;
                    index -= 1;
                }
                const u32 ea = (effective_address(in) + static_cast<u32>(index * static_cast<i32>(size))) & (in.adsize == 2 ? 0xFFFFu : 0xFFFFFFFFu);
                const u32 lin = linear(in.mem_seg, ea, size, in.op == Op::Bt ? Access::Read : Access::Write);
                if (in.op != Op::Bt) mem_.probe_write(lin, size);
                d = Dst{true, lin, 0};
                bitno = static_cast<u32>(rem);
            } else {
                bitno = read_op(in, in.src, in.src.kind == OpndKind::Imm ? 1 : size) & (bits - 1);
                if (in.op == Op::Bt) {
                    d = in.dst.kind == OpndKind::Mem ? Dst{true, linear(in.mem_seg, effective_address(in), size, Access::Read), 0} : Dst{false, 0, in.dst.reg};
                } else {
                    d = resolve_rmw(*this, in, size);
                }
            }
            const u32 a = load_dst(*this, d, size) & mask;
            const bool cf = ((a >> bitno) & 1) != 0;
            if (in.op != Op::Bt) {
                const u32 r = in.op == Op::Bts ? a | (1u << bitno) : in.op == Op::Btr ? a & ~(1u << bitno) : a ^ (1u << bitno);
                store_dst(*this, d, size, r);
            }
            st_.materialise();
            st_.flags = (st_.flags & ~fl::CF) | (cf ? fl::CF : 0);
            st_.undefined_flags |= fl::OF | fl::SF | fl::ZF | fl::AF | fl::PF;
            return;
        }
        case Op::Bsf: case Op::Bsr: {
            const u32 v = read_op(in, in.src, size) & mask;
            st_.materialise();
            st_.undefined_flags |= fl::OF | fl::SF | fl::AF | fl::PF | fl::CF;
            if (v == 0) {
                st_.flags |= fl::ZF;
            } else {
                st_.flags &= ~fl::ZF;
                const u32 index = in.op == Op::Bsf ? static_cast<u32>(std::countr_zero(v)) : 31u - static_cast<u32>(std::countl_zero(v));
                st_.set_reg(in.dst.reg, size, index);
            }
            return;
        }
        case Op::Xadd: {
            const Dst d = resolve_rmw(*this, in, size);
            const u32 a = load_dst(*this, d, size) & mask;
            const u32 b = read_op(in, in.src, size) & mask;
            const u32 r = (a + b) & mask;
            store_dst(*this, d, size, r);
            write_op(in, in.src, size, a);
            st_.lazy(LazyOp::Add, size, a, b, r);
            return;
        }
        case Op::Cmpxchg: {
            const Dst d = resolve_rmw(*this, in, size);
            const u32 a = load_dst(*this, d, size) & mask;
            const u32 acc = st_.reg(kEax, size);
            const u32 b = read_op(in, in.src, size) & mask;
            if (acc == a) store_dst(*this, d, size, b);
            else {
                store_dst(*this, d, size, a);
                st_.set_reg(kEax, size, a);
            }
            st_.lazy(LazyOp::Sub, size, acc, a, (acc - a) & mask);
            return;
        }
        case Op::Cmpxchg8b: {
            const u32 lin = linear(in.mem_seg, effective_address(in), 8, Access::Write);
            mem_.probe_write(lin, 8);
            const u32 lo = mem_.read(lin, 4), hi = mem_.read(lin + 4, 4);
            st_.materialise();
            if (lo == st_.gpr[kEax] && hi == st_.gpr[kEdx]) {
                mem_.write(lin, 4, st_.gpr[kEbx]);
                mem_.write(lin + 4, 4, st_.gpr[kEcx]);
                st_.flags |= fl::ZF;
            } else {
                mem_.write(lin, 4, lo);
                mem_.write(lin + 4, 4, hi);
                st_.gpr[kEax] = lo;
                st_.gpr[kEdx] = hi;
                st_.flags &= ~fl::ZF;
            }
            return;
        }

        case Op::In: {
            const u16 port = in.port_dx ? st_.reg16(kEdx) : static_cast<u16>(in.imm);
            if (st_.protected_mode() && st_.cpl() > st_.iopl()) raise_gp(0);
            st_.set_reg(kEax, size, mem_.io_read(port, size));
            return;
        }
        case Op::Out: {
            const u16 port = in.port_dx ? st_.reg16(kEdx) : static_cast<u16>(in.imm);
            if (st_.protected_mode() && st_.cpl() > st_.iopl()) raise_gp(0);
            mem_.io_write(port, size, st_.reg(kEax, size));
            return;
        }

        case Op::Nop: case Op::NopEv: case Op::Prefetch: return;
        case Op::X87: exec_x87(in); return;
        case Op::Simd: exec_simd(in); return;
        case Op::Fxsave: fxsave(in); return;
        case Op::Fxrstor: fxrstor(in); return;
        case Op::Ldmxcsr: ldmxcsr(in); return;
        case Op::Stmxcsr: stmxcsr(in); return;
        case Op::Sfence:
            if (in.mem) raise_ud();  // the memory form is CLFLUSH, which the Pentium III lacks
            return;
        default: exec_system(in); return;
    }
}

}  // namespace mv::p6
