#include "cpu/p6/jit/jit.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define MV_JIT_X64 1
#include "cpu/p6/jit/x64_asm.h"
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace mv::p6 {

namespace {
constexpr std::size_t kBufferBytes = 64u << 20;
}

Jit::Jit() {
#if MV_JIT_X64
#if defined(_WIN32)
    void* p = VirtualAlloc(nullptr, kBufferBytes, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    void* p = mmap(nullptr, kBufferBytes, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) p = nullptr;
#endif
    if (p) {
        buffer_ = static_cast<u8*>(p);
        capacity_ = kBufferBytes;
    }
#endif
}

Jit::~Jit() {
    if (!buffer_) return;
#if defined(_WIN32)
    VirtualFree(buffer_, 0, MEM_RELEASE);
#else
    munmap(buffer_, capacity_);
#endif
}

void Jit::flush(BlockEngine& engine) {
    engine.drop_native();
    used_ = 0;
    ++flushes_;
}

bool Jit::eligible(const Block& b, Cpu& cpu) {
    const CpuState& s = cpu.state();
    if (!b.native || s.eip != b.jit_eip || s.inhibit_interrupts) return false;
    if ((s.cr0 & (cr0::PE | cr0::PG)) != (cr0::PE | cr0::PG) || (s.flags & fl::VM)) return false;
    if (cpu.memory().dirty_tracking() != b.jit_dirty) return false;
    for (unsigned g = 0; g < kSegCount; ++g) {
        if (!(b.jit_segs & (1u << g))) continue;
        const Segment& now = s.seg[g];
        const Segment& then = b.jit_seg[g];
        if (now.base != then.base || now.limit != then.limit || now.ar != then.ar) return false;
    }
    return true;
}

#if MV_JIT_X64
namespace {
using namespace x64;

static_assert(std::is_standard_layout_v<CpuState>);
static_assert(std::is_standard_layout_v<JitCtx>);
static_assert(sizeof(LazyOp) == 1);
constexpr i32 kGpr = static_cast<i32>(offsetof(CpuState, gpr));
constexpr i32 kEipOff = static_cast<i32>(offsetof(CpuState, eip));
constexpr i32 kFlagsOff = static_cast<i32>(offsetof(CpuState, flags));
constexpr i32 kLazyOp = static_cast<i32>(offsetof(CpuState, lazy_op));
constexpr i32 kLazyC = static_cast<i32>(offsetof(CpuState, lazy_c));
constexpr i32 kLazyOf = static_cast<i32>(offsetof(CpuState, lazy_of));
constexpr i32 kLazyAf = static_cast<i32>(offsetof(CpuState, lazy_af));
constexpr i32 kLazyA = static_cast<i32>(offsetof(CpuState, lazy_a));
constexpr i32 kLazyB = static_cast<i32>(offsetof(CpuState, lazy_b));
constexpr i32 kLazyR = static_cast<i32>(offsetof(CpuState, lazy_r));
static_assert(offsetof(CpuState, lazy_size) == offsetof(CpuState, lazy_op) + 1 && offsetof(CpuState, lazy_c) == offsetof(CpuState, lazy_op) + 2);
constexpr i32 kCtxSt = static_cast<i32>(offsetof(JitCtx, st));
constexpr i32 kCtxExit = static_cast<i32>(offsetof(JitCtx, exit));
constexpr i32 kCtxScratch = static_cast<i32>(offsetof(JitCtx, scratch));
static_assert(offsetof(Memory::TlbEntry, tag) == 0 && offsetof(Memory::TlbEntry, phys_page) == 4 && offsetof(Memory::TlbEntry, host) == 8 &&
              offsetof(Memory::TlbEntry, writable) == 16 && sizeof(Memory::TlbEntry) == 32);

// Host registers: the context, the guest state, the TLB, the RAM base, the data cache's MRU lines, the bitmap of
// pages holding translated code. All callee-saved; RAX RCX RDX RSI RDI R8-R11 are scratch.
constexpr Reg kCtx = rbx, kSt = r12, kTlb = r13, kRam = r14, kMru = r15, kCodeMap = rbp;
#if defined(_WIN32)
constexpr Reg kArg0 = rcx, kArg1 = rdx, kArg2 = r8;
#else
constexpr Reg kArg0 = rdi, kArg1 = rsi, kArg2 = rdx;
#endif

Mem greg(u8 r, unsigned size) {
    if (size == 1) return r < 4 ? mem(kSt, kGpr + 4 * r) : mem(kSt, kGpr + 4 * (r - 4) + 1);
    return mem(kSt, kGpr + 4 * (r & 7));
}
Mem st_at(i32 off) { return mem(kSt, off); }
Mem ctx_at(i32 off) { return mem(kCtx, off); }

// ---- helpers the generated code calls ----

// Execute instruction `i` with the interpreter, as the block interpreter does: the clock at its start, its
// penalties added, the run left when it faulted, reached a device, rewrote its own page or passed the limit.
u32 h_exec(JitCtx* c, u32 i) {
    Block& b = *c->block;
    Cpu& cpu = *c->cpu;
    Memory& mem = cpu.memory();
    cpu.state().eip = b.jit_eip + b.off[i];
    cpu.set_cycles_only(c->base + (i ? b.after[i - 1] : 0));
    ++c->helpers;
    bool ok;
    try {
        ok = cpu.exec_decoded(b.insns[i]);
    } catch (...) {
        *c->error = std::current_exception();
        return (i << 2) | kJitFault;
    }
    c->base += mem.take_penalty();
    if (!ok) return (i << 2) | kJitFault;
    if (i + 1 == b.n) return (i << 2) | kJitEnd;
    if (mem.device_events() != c->devices || c->engine->running_dead() || c->base + b.after[i] >= c->limit) return (i << 2) | kJitExit;
    if (c->base + b.cycles >= c->limit) {
        c->exit = 2;  // as in h_cache: the interpreter finishes the block
        return (i << 2) | kJitExit;
    }
    return 0;
}

// A data access the inline check could not settle (not the set's most recent line): the full cache model.
void h_cache(JitCtx* c, u32 phys, u32 arg) {
    const unsigned i = arg >> 8;
    const u64 p = c->cpu->cache_timing().data_access(phys, (arg & 1) != 0);
    c->base += p;
    // The limit at the next boundary: leave there. Past the block's end only: a later boundary may reach it, which
    // the interpreter finds instruction by instruction — hand the rest of the block over to it.
    if (c->base + c->block->after[i] >= c->limit) c->exit = 1;
    else if (c->base + c->block->cycles >= c->limit) c->exit = 2;
}

u32 h_cf(const CpuState* s) { return s->cf() ? 1u : 0u; }
u32 h_cond(const Cpu* cpu, u32 cc) { return cpu->condition(static_cast<u8>(cc)) ? 1u : 0u; }

// What the translator knows about the lazy flags at a point of the block: the kind the last native flag writer
// left (the interpreter leaves the same kind for the same instruction, so a fallback does not change it), and
// whether the host's own flags still hold that instruction's result.
enum class Known : u8 { Unknown, Add, Sub, Logic, Result };

u32 mask_of(unsigned size) { return size == 4 ? 0xFFFFFFFFu : size == 2 ? 0xFFFFu : 0xFFu; }
bool uses_cf(u8 cc) { return (cc >> 1) == 1 || (cc >> 1) == 3; }  // B/AE, BE/A

class Translator {
public:
    Translator(Block& b, Cpu& cpu) : b_(b), cpu_(cpu), st_(cpu.state()) {}
    Asm a;

    void run() {
        epilogue_ = a.label();
        prologue();
        for (unsigned i = 0; i < b_.n; ++i) {
            Label after = a.label();
            if (!native(i, after)) inline_helper(i);
            a.bind(after);
        }
        // The last instruction always leaves (native or helper); nothing falls through to here.
        a.bind(epilogue_);
        epilogue();
        for (const Stub& s : stubs_) {
            a.bind(s.entry);
            call_helper(s.index);
            if (s.index + 1 == b_.n) a.jmp(epilogue_);
            else {
                a.test(4, rax, rax);
                a.jcc(NE, epilogue_);
                a.jmp(s.resume);
            }
        }
    }

private:
    struct Stub {
        Label entry, resume;
        unsigned index;
    };

    Block& b_;
    Cpu& cpu_;
    const CpuState& st_;
    Label epilogue_;
    std::vector<Stub> stubs_;
    Known known_{Known::Unknown};
    unsigned known_size_{4};
    bool host_flags_{false};   // the host flags hold the last flag writer's result
    bool host_cf_{false};      // ... including CF (not after INC/DEC)

    u32 eip_of(unsigned i) const { return b_.jit_eip + b_.off[i]; }
    u32 next_eip(unsigned i) const { return eip_of(i) + b_.insns[i].len; }
    bool last(unsigned i) const { return i + 1 == b_.n; }

    void prologue() {
        a.push(rbp);
        a.push(rbx);
        a.push(r12);
        a.push(r13);
        a.push(r14);
        a.push(r15);
#if defined(_WIN32)
        a.push(rsi);
        a.push(rdi);
        a.alu_imm(8, 5, rsp, 40);  // shadow space + alignment
#else
        a.alu_imm(8, 5, rsp, 8);  // 16-byte alignment at calls
#endif
        a.mov(8, kCtx, kArg0);
        a.mov(8, kSt, ctx_at(kCtxSt));
        Memory& m = cpu_.memory();
        a.mov64(kTlb, reinterpret_cast<u64>(m.tlb()));
        a.mov64(kRam, reinterpret_cast<u64>(m.bus().ram()));
        a.mov64(kMru, reinterpret_cast<u64>(cpu_.cache_timing().l1d_mru()));
        a.mov64(kCodeMap, reinterpret_cast<u64>(m.bus().code_map()));
    }
    void epilogue() {
#if defined(_WIN32)
        a.alu_imm(8, 0, rsp, 40);
        a.pop(rdi);
        a.pop(rsi);
#else
        a.alu_imm(8, 0, rsp, 8);
#endif
        a.pop(r15);
        a.pop(r14);
        a.pop(r13);
        a.pop(r12);
        a.pop(rbx);
        a.pop(rbp);
        a.ret();
    }

    void call_helper(unsigned i) {
        a.mov(8, kArg0, kCtx);
        a.mov32(kArg1, i);
        a.call_abs(reinterpret_cast<const void*>(&h_exec));
    }
    void inline_helper(unsigned i) {
        call_helper(i);
        if (last(i)) a.jmp(epilogue_);
        else {
            a.test(4, rax, rax);
            a.jcc(NE, epilogue_);
        }
        known_ = Known::Unknown;
        host_flags_ = host_cf_ = false;
    }
    // A fallback for instruction i: its native checks jump here; the interpreter executes it from the start.
    Label fallback(unsigned i, Label resume) {
        Stub s{a.label(), resume, i};
        stubs_.push_back(s);
        return s.entry;
    }
    void leave(unsigned i, u32 eip, u32 kind) {
        a.mov_imm(4, st_at(kEipOff), eip);
        a.mov32(rax, (i << 2) | kind);
        a.jmp(epilogue_);
    }
    // After an instruction whose cache lookups may have reached the limit: leave if the context says so. Leaves the
    // host flags alone (JRCXZ).
    void exit_check(unsigned i) {
        if (last(i)) return;
        a.movzx(rcx, 1, ctx_at(kCtxExit));
        a.byte(0xE3);  // jrcxz over the leave sequence
        const std::size_t at = a.size();
        a.byte(0);
        leave(i, next_eip(i), kJitExit);
        a.code[at] = static_cast<u8>(a.size() - at - 1);
    }

    // ---- memory ----
    // EAX = the effective address of the memory operand (32-bit addressing).
    void effective_address(const Insn& in) {
        if (in.base != Insn::kNoReg) a.mov(4, rax, greg(in.base, 4));
        else a.mov32(rax, 0);
        if (in.index != Insn::kNoReg) {
            a.mov(4, rcx, greg(in.index, 4));
            a.lea(4, rax, mem(rax, rcx, in.scale, in.disp));
        } else if (in.disp != 0) {
            a.lea(4, rax, mem(rax, in.disp));
        }
    }
    // Whether an access through segment `seg` can be native here; records the segment the code assumes.
    bool segment_ok(u8 seg, unsigned size, bool write) {
        const Segment& s = st_.seg[seg];
        if (s.unusable() || !s.data() || s.expand_down()) return false;
        if (write && !s.writable()) return false;
        if (s.limit != 0xFFFFFFFFu && s.limit < size - 1) return false;
        b_.jit_segs |= static_cast<u8>(1u << seg);
        b_.jit_seg[seg] = s;
        return true;
    }
    // EAX holds the offset in `seg`: limit check, linear address, page and TLB checks. Leaves RSI = the host byte,
    // EDI = the physical address. Jumps to `fail` for anything the interpreter must do.
    void access_checks(u8 seg, unsigned size, bool write, Label fail) {
        const Segment& s = st_.seg[seg];
        if (s.limit != 0xFFFFFFFFu) {
            a.alu_imm(4, 7, rax, s.limit - (size - 1));
            a.jcc(A, fail);
        }
        if (s.base) a.lea(4, rax, mem(rax, static_cast<i32>(s.base)));
        if (size > 1) {
            a.mov(4, rdx, rax);
            a.alu_imm(4, 4, rdx, 0xFFF);
            a.alu_imm(4, 7, rdx, 0x1000 - size);
            a.jcc(A, fail);
        }
        a.mov(4, rdx, rax);
        a.shift_imm(4, 5, rdx, 12);  // the linear page
        a.mov(4, rcx, rdx);
        a.alu_imm(4, 4, rcx, Memory::kTlbSize - 1);
        a.shift_imm(4, 4, rcx, 5);
        a.alu(4, 7, rdx, mem(kTlb, rcx, 0, 0));  // tag
        a.jcc(NE, fail);
        if (write) {
            a.op_rm(1, {0x80}, 7, mem(kTlb, rcx, 0, 16));  // cmp byte [writable], 0
            a.byte(0);
            a.jcc(E, fail);
        }
        a.mov(8, rsi, mem(kTlb, rcx, 0, 8));  // host page
        a.test(8, rsi, rsi);
        a.jcc(E, fail);
        if (write) {
            // Never into a page holding translated code (the interpreter drops its blocks).
            a.mov(8, rdx, rsi);
            a.alu(8, 5, rdx, kRam);
            a.shift_imm(8, 5, rdx, 12);
            a.bt(8, mem(kCodeMap), rdx);
            a.jcc(B, fail);
        }
        a.mov(4, rdi, mem(kTlb, rcx, 0, 4));  // physical page
        a.mov(4, rdx, rax);
        a.alu_imm(4, 4, rdx, 0xFFF);
        a.alu(4, 1, rdi, rdx);  // or: the physical address
        a.alu(8, 0, rsi, rdx);  // the host byte
    }
    // The cache model for an access at EDI (RSI, RAX and RDI are kept).
    void cache(unsigned i, bool store) {
        const unsigned sets = cpu_.cache_timing().l1d_sets();
        Label hit = a.label();
        a.mov(4, rcx, rdi);
        a.shift_imm(4, 5, rcx, 5);  // the line
        a.mov(4, rdx, rcx);
        a.alu_imm(4, 4, rdx, sets - 1);
        a.lea(4, rcx, mem(rcx, 1));
        a.alu(4, 7, rcx, mem(kMru, rdx, 2, 0));
        // MV_JIT_NO_MRU=1 sends every access through the full model (a control for the inline check).
        static const bool every_access = std::getenv("MV_JIT_NO_MRU") != nullptr;
        if (!every_access) a.jcc(E, hit);
        a.mov(8, ctx_at(kCtxScratch), rsi);
        a.mov(8, ctx_at(kCtxScratch + 8), rax);
        a.mov(8, ctx_at(kCtxScratch + 32), rdi);
        a.mov(4, kArg1, rdi);
        a.mov(8, kArg0, kCtx);
        a.mov32(kArg2, (i << 8) | (store ? 1u : 0u));
        a.call_abs(reinterpret_cast<const void*>(&h_cache));
        a.mov(8, rsi, ctx_at(kCtxScratch));
        a.mov(8, rax, ctx_at(kCtxScratch + 8));
        a.mov(8, rdi, ctx_at(kCtxScratch + 32));
        a.bind(hit);
    }
    void load_host(Reg dst, unsigned size) {
        if (size == 4) a.mov(4, dst, mem(rsi));
        else a.movzx(dst, size, mem(rsi));
    }
    void load_greg(Reg dst, u8 r, unsigned size) {
        if (size == 4) a.mov(4, dst, greg(r, 4));
        else a.movzx(dst, size, greg(r, size));
    }

    // ---- flags ----
    void set_lazy(LazyOp op, unsigned size) {
        a.mov_imm(2, st_at(kLazyOp), static_cast<u32>(op) | (size << 8));
        a.mov_imm(1, st_at(kLazyC), 0);
    }
    // EDX = 0/1: the condition `cc` of the current lazy flags. Clobbers the host flags.
    void condition_to_edx(u8 cc) {
        if (host_flags_ && (host_cf_ || !uses_cf(cc))) {
            a.setcc(static_cast<Cond>(cc), rdx);
            a.movzx(rdx, 1, rdx);
            return;
        }
        if (known_ == Known::Add || known_ == Known::Sub || known_ == Known::Logic) {
            // Redo the operation on the operands the lazy state holds: the host flags are the architecture's.
            const unsigned s = known_size_;
            if (known_ == Known::Logic) {
                a.mov(4, rdx, st_at(kLazyR));
                a.test(s, rdx, rdx);
            } else {
                a.mov(4, rdx, st_at(kLazyA));
                a.alu(s, known_ == Known::Add ? 0 : 7, rdx, st_at(kLazyB));
            }
            a.setcc(static_cast<Cond>(cc), rdx);
            a.movzx(rdx, 1, rdx);
            return;
        }
        a.mov64(kArg0, reinterpret_cast<u64>(&cpu_));
        a.mov32(kArg1, cc);
        a.call_abs(reinterpret_cast<const void*>(&h_cond));
        a.mov(4, rdx, rax);
    }
    // EDX = the current CF (for INC/DEC's lazy record). Clobbers the host flags and the scratch registers.
    void cf_to_edx() {
        switch (known_) {
            case Known::Add:
                a.mov(4, rdx, st_at(kLazyR));
                a.alu(4, 7, rdx, st_at(kLazyA));
                a.setcc(B, rdx);
                a.movzx(rdx, 1, rdx);
                return;
            case Known::Sub:
                a.mov(4, rdx, st_at(kLazyA));
                a.alu(4, 7, rdx, st_at(kLazyB));
                a.setcc(B, rdx);
                a.movzx(rdx, 1, rdx);
                return;
            case Known::Logic: a.mov32(rdx, 0); return;
            case Known::Result: a.movzx(rdx, 1, st_at(kLazyC)); return;
            case Known::Unknown:
                a.mov(8, kArg0, kSt);
                a.call_abs(reinterpret_cast<const void*>(&h_cf));
                a.mov(4, rdx, rax);
                return;
        }
    }

    // ---- instructions ----
    bool native(unsigned i, Label after) {
        const Insn& in = b_.insns[i];
        if (in.lock && in.op != Op::Nop) return false;
        if (in.mem && in.adsize != 4) return false;
        switch (in.op) {
            case Op::Nop: case Op::NopEv: case Op::Prefetch:
                if (in.mem) return false;
                if (last(i)) leave(i, next_eip(i), kJitEnd);
                return true;
            case Op::Mov: return mov(i, in, after);
            case Op::Movzx: case Op::Movsx: return extend(i, in, after);
            case Op::Lea: return lea(i, in);
            case Op::Add: case Op::Or: case Op::And: case Op::Sub: case Op::Xor: case Op::Cmp: case Op::Test: return alu(i, in, after);
            case Op::Inc: case Op::Dec: return incdec(i, in, after);
            case Op::Neg: case Op::Not: return negnot(i, in, after);
            case Op::Push: return push(i, in, after);
            case Op::Pop: return pop(i, in, after);
            case Op::Jcc: return jcc(i, in);
            case Op::Jmp: return jmp(i, in);
            case Op::Call: return call(i, in, after);
            case Op::Ret: return ret(i, in, after);
            case Op::Setcc: return setcc(i, in);
            case Op::Cmovcc: return cmov(i, in);
            case Op::Xchg: return xchg(i, in);
            default: return false;
        }
    }
    // The end of a native non-branch instruction: the last one leaves the block; others check the exit flag.
    void finish(unsigned i, bool touched_memory) {
        if (last(i)) leave(i, next_eip(i), kJitEnd);
        else if (touched_memory) exit_check(i);
    }

    bool mov(unsigned i, const Insn& in, Label after) {
        const unsigned size = in.size;
        if (in.dst.kind == OpndKind::Mem) {
            if (in.src.kind != OpndKind::Reg && in.src.kind != OpndKind::Imm) return false;
            if (!segment_ok(in.mem_seg, size, true)) return false;
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, size, true, f);
            cache(i, true);
            if (in.src.kind == OpndKind::Reg) {
                load_greg(rax, in.src.reg, size);
                a.mov(size, mem(rsi), rax);
            } else {
                a.mov_imm(size, mem(rsi), in.imm & mask_of(size));
            }
            host_flags_ = host_cf_ = false;
            finish(i, true);
            return true;
        }
        if (in.dst.kind != OpndKind::Reg) return false;
        if (in.src.kind == OpndKind::Mem) {
            if (!segment_ok(in.mem_seg, size, false)) return false;
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, size, false, f);
            cache(i, false);
            load_host(rax, size);
            a.mov(size, greg(in.dst.reg, size), rax);
            host_flags_ = host_cf_ = false;
            finish(i, true);
            return true;
        }
        if (in.src.kind == OpndKind::Reg) {
            load_greg(rax, in.src.reg, size);
            a.mov(size, greg(in.dst.reg, size), rax);
        } else if (in.src.kind == OpndKind::Imm) {
            a.mov_imm(size, greg(in.dst.reg, size), in.imm & mask_of(size));
        } else {
            return false;
        }
        finish(i, false);
        return true;
    }

    bool extend(unsigned i, const Insn& in, Label after) {
        if (in.dst.kind != OpndKind::Reg || (in.size != 2 && in.size != 4)) return false;
        const unsigned from = in.src_size;
        if (from != 1 && from != 2) return false;
        const bool sx = in.op == Op::Movsx;
        bool touched = false;
        if (in.src.kind == OpndKind::Mem) {
            if (!segment_ok(in.mem_seg, from, false)) return false;
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, from, false, f);
            cache(i, false);
            if (sx) a.movsx(rax, from, mem(rsi));
            else a.movzx(rax, from, mem(rsi));
            touched = true;
            host_flags_ = host_cf_ = false;
        } else if (in.src.kind == OpndKind::Reg) {
            if (sx) a.movsx(rax, from, greg(in.src.reg, from));
            else a.movzx(rax, from, greg(in.src.reg, from));
        } else {
            return false;
        }
        a.mov(in.size, greg(in.dst.reg, in.size), rax);
        finish(i, touched);
        return true;
    }

    bool lea(unsigned i, const Insn& in) {
        if (in.dst.kind != OpndKind::Reg || (in.size != 2 && in.size != 4) || !in.mem) return false;
        effective_address(in);
        a.mov(in.size, greg(in.dst.reg, in.size), rax);
        finish(i, false);
        return true;
    }

    bool alu(unsigned i, const Insn& in, Label after) {
        const unsigned size = in.size;
        const bool compare = in.op == Op::Cmp || in.op == Op::Test;
        const bool dst_mem = in.dst.kind == OpndKind::Mem;
        const bool src_mem = in.src.kind == OpndKind::Mem;
        if (!dst_mem && in.dst.kind != OpndKind::Reg) return false;
        if (!src_mem && in.src.kind != OpndKind::Reg && in.src.kind != OpndKind::Imm) return false;
        if (dst_mem && src_mem) return false;
        unsigned digit = 0;
        LazyOp kind = LazyOp::Add;
        Known known = Known::Add;
        switch (in.op) {
            case Op::Add: digit = 0; kind = LazyOp::Add; known = Known::Add; break;
            case Op::Or: digit = 1; kind = LazyOp::Logic; known = Known::Logic; break;
            case Op::And: case Op::Test: digit = 4; kind = LazyOp::Logic; known = Known::Logic; break;
            case Op::Sub: case Op::Cmp: digit = 5; kind = LazyOp::Sub; known = Known::Sub; break;
            case Op::Xor: digit = 6; kind = LazyOp::Logic; known = Known::Logic; break;
            default: return false;
        }
        const bool touched = dst_mem || src_mem;
        if (touched) {
            if (!segment_ok(in.mem_seg, size, dst_mem && !compare)) return false;
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, size, dst_mem && !compare, f);
            // The interpreter's accesses in its order: the destination's read, then its write.
            cache(i, false);
            if (dst_mem && !compare) cache(i, true);
        }
        // EAX = a (the destination), ECX = b (the source), both zero-extended.
        if (dst_mem) load_host(rax, size);
        else load_greg(rax, in.dst.reg, size);
        if (src_mem) load_host(rcx, size);
        else if (in.src.kind == OpndKind::Imm) a.mov32(rcx, in.imm & mask_of(size));
        else load_greg(rcx, in.src.reg, size);
        a.mov(4, st_at(kLazyA), rax);
        a.mov(4, st_at(kLazyB), rcx);
        if (compare) a.mov(4, rdx, rax);
        const Reg r = compare ? rdx : rax;
        a.alu(size, digit, r, rcx);  // the host flags are the architecture's from here
        a.mov(4, st_at(kLazyR), r);
        if (!compare) {
            if (dst_mem) a.mov(size, mem(rsi), rax);
            else a.mov(size, greg(in.dst.reg, size), rax);
        }
        set_lazy(kind, size);
        known_ = known;
        known_size_ = size;
        host_flags_ = host_cf_ = !touched;  // a fallback path joins here without them
        // The flags must survive the exit check: it uses JRCXZ.
        finish(i, touched);
        return true;
    }

    bool incdec(unsigned i, const Insn& in, Label after) {
        const unsigned size = in.size;
        const bool m = in.dst.kind == OpndKind::Mem;
        if (!m && in.dst.kind != OpndKind::Reg) return false;
        if (m && !segment_ok(in.mem_seg, size, true)) return false;
        // The CF the record keeps, from the flags before the instruction.
        cf_to_edx();
        a.mov(1, ctx_at(kCtxScratch + 16), rdx);
        if (m) {
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, size, true, f);
            cache(i, false);
            cache(i, true);
            load_host(rax, size);
        } else {
            load_greg(rax, in.dst.reg, size);
        }
        const bool inc = in.op == Op::Inc;
        // AF from a: INC carries out of the low nibble when it is 0xF; DEC borrows when it is 0.
        a.mov(4, rcx, rax);
        a.alu_imm(4, 4, rcx, 0xF);
        a.alu_imm(4, 7, rcx, inc ? 0xF : 0);
        a.setcc(E, rcx);
        a.mov(1, st_at(kLazyAf), rcx);
        a.incdec(size, inc, rax);  // the host's OF is the architecture's
        a.setcc(O, rcx);
        a.mov(1, st_at(kLazyOf), rcx);
        a.mov(4, st_at(kLazyR), rax);
        if (m) a.mov(size, mem(rsi), rax);
        else a.mov(size, greg(in.dst.reg, size), rax);
        a.movzx(rcx, 1, ctx_at(kCtxScratch + 16));
        a.mov(1, st_at(kLazyC), rcx);
        a.mov_imm(2, st_at(kLazyOp), static_cast<u32>(LazyOp::Result) | (size << 8));
        known_ = Known::Result;
        known_size_ = size;
        host_flags_ = !m;
        host_cf_ = false;
        finish(i, m);
        return true;
    }

    bool negnot(unsigned i, const Insn& in, Label after) {
        const unsigned size = in.size;
        const bool m = in.dst.kind == OpndKind::Mem;
        if (!m && in.dst.kind != OpndKind::Reg) return false;
        if (m) {
            if (!segment_ok(in.mem_seg, size, true)) return false;
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, size, true, f);
            cache(i, false);
            cache(i, true);
            load_host(rax, size);
        } else {
            load_greg(rax, in.dst.reg, size);
        }
        if (in.op == Op::Not) {
            a.unary(size, 2, rax);
            if (m) a.mov(size, mem(rsi), rax);
            else a.mov(size, greg(in.dst.reg, size), rax);
            if (m) host_flags_ = host_cf_ = false;
            finish(i, m);
            return true;
        }
        a.mov_imm(4, st_at(kLazyA), 0);
        a.mov(4, st_at(kLazyB), rax);
        a.unary(size, 3, rax);
        a.mov(4, st_at(kLazyR), rax);
        if (m) a.mov(size, mem(rsi), rax);
        else a.mov(size, greg(in.dst.reg, size), rax);
        set_lazy(LazyOp::Sub, size);
        known_ = Known::Sub;
        known_size_ = size;
        host_flags_ = host_cf_ = !m;
        finish(i, m);
        return true;
    }

    // PUSH of a register or an immediate (32-bit operand and stack).
    bool push(unsigned i, const Insn& in, Label after) {
        const Opnd o = in.src.kind != OpndKind::None ? in.src : in.dst;
        if (in.size != 4 || (o.kind != OpndKind::Reg && o.kind != OpndKind::Imm)) return false;
        if (!segment_ok(kSs, 4, true)) return false;
        const Label f = fallback(i, after);
        a.mov(4, rax, greg(kEsp, 4));
        a.lea(4, rax, mem(rax, -4));
        a.mov(4, ctx_at(kCtxScratch + 24), rax);  // the new ESP
        access_checks(kSs, 4, true, f);
        if (o.kind == OpndKind::Reg) load_greg(rax, o.reg, 4);
        else a.mov32(rax, in.imm);
        cache(i, true);
        a.mov(4, mem(rsi), rax);
        a.mov(4, rcx, ctx_at(kCtxScratch + 24));
        a.mov(4, greg(kEsp, 4), rcx);
        host_flags_ = host_cf_ = false;
        finish(i, true);
        return true;
    }

    bool pop(unsigned i, const Insn& in, Label after) {
        if (in.size != 4 || in.dst.kind != OpndKind::Reg) return false;
        if (!segment_ok(kSs, 4, false)) return false;
        const Label f = fallback(i, after);
        a.mov(4, rax, greg(kEsp, 4));
        a.mov(4, ctx_at(kCtxScratch + 24), rax);
        access_checks(kSs, 4, false, f);
        cache(i, false);
        a.mov(4, rax, mem(rsi));
        a.mov(4, rcx, ctx_at(kCtxScratch + 24));
        a.lea(4, rcx, mem(rcx, 4));
        a.mov(4, greg(kEsp, 4), rcx);
        a.mov(4, greg(in.dst.reg, 4), rax);  // POP ESP: the popped value wins
        host_flags_ = host_cf_ = false;
        finish(i, true);
        return true;
    }

    bool jcc(unsigned i, const Insn& in) {
        if (!last(i) || in.size != 4) return false;
        condition_to_edx(in.cc);
        Label taken = a.label();
        a.test(4, rdx, rdx);
        a.jcc(NE, taken);
        leave(i, next_eip(i), kJitEnd);
        a.bind(taken);
        leave(i, next_eip(i) + in.imm, kJitEnd);
        return true;
    }
    bool jmp(unsigned i, const Insn& in) {
        if (!last(i) || in.size != 4) return false;
        leave(i, next_eip(i) + in.imm, kJitEnd);
        return true;
    }
    bool call(unsigned i, const Insn& in, Label after) {
        if (!last(i) || in.size != 4 || !segment_ok(kSs, 4, true)) return false;
        const Label f = fallback(i, after);
        a.mov(4, rax, greg(kEsp, 4));
        a.lea(4, rax, mem(rax, -4));
        a.mov(4, ctx_at(kCtxScratch + 24), rax);
        access_checks(kSs, 4, true, f);
        cache(i, true);
        a.mov_imm(4, mem(rsi), next_eip(i));
        a.mov(4, rcx, ctx_at(kCtxScratch + 24));
        a.mov(4, greg(kEsp, 4), rcx);
        leave(i, next_eip(i) + in.imm, kJitEnd);
        return true;
    }
    bool ret(unsigned i, const Insn& in, Label after) {
        if (!last(i) || in.size != 4 || !segment_ok(kSs, 4, false)) return false;
        const Label f = fallback(i, after);
        a.mov(4, rax, greg(kEsp, 4));
        a.mov(4, ctx_at(kCtxScratch + 24), rax);
        access_checks(kSs, 4, false, f);
        cache(i, false);
        a.mov(4, rax, mem(rsi));
        a.mov(4, st_at(kEipOff), rax);
        a.mov(4, rcx, ctx_at(kCtxScratch + 24));
        a.lea(4, rcx, mem(rcx, static_cast<i32>(4 + in.imm)));
        a.mov(4, greg(kEsp, 4), rcx);
        a.mov32(rax, (i << 2) | kJitEnd);
        a.jmp(epilogue_);
        return true;
    }
    bool setcc(unsigned i, const Insn& in) {
        if (in.dst.kind != OpndKind::Reg) return false;
        condition_to_edx(in.cc);
        a.mov(1, greg(in.dst.reg, 1), rdx);
        host_flags_ = host_cf_ = false;
        finish(i, false);
        return true;
    }
    bool cmov(unsigned i, const Insn& in) {
        if (in.dst.kind != OpndKind::Reg || in.src.kind != OpndKind::Reg || (in.size != 2 && in.size != 4)) return false;
        condition_to_edx(in.cc);
        Label skip = a.label();
        a.test(4, rdx, rdx);
        a.jcc(E, skip);
        load_greg(rax, in.src.reg, in.size);
        a.mov(in.size, greg(in.dst.reg, in.size), rax);
        a.bind(skip);
        host_flags_ = host_cf_ = false;
        finish(i, false);
        return true;
    }
    bool xchg(unsigned i, const Insn& in) {
        const Opnd src = in.src.kind == OpndKind::None ? Opnd{OpndKind::Reg, kEax} : in.src;
        if (in.dst.kind != OpndKind::Reg || src.kind != OpndKind::Reg) return false;
        const unsigned size = in.size;
        load_greg(rax, in.dst.reg, size);
        load_greg(rcx, src.reg, size);
        a.mov(size, greg(in.dst.reg, size), rcx);
        a.mov(size, greg(src.reg, size), rax);
        finish(i, false);
        return true;
    }
};

}  // namespace

bool Jit::compile(Block& b, Cpu& cpu) {
    if (!buffer_ || b.rep || b.mode != 3) return false;
    b.jit_eip = cpu.state().eip;
    b.jit_segs = 0;
    b.jit_dirty = cpu.memory().dirty_tracking();
    Translator t(b, cpu);
    t.run();
    if (!t.a.finish()) return false;
    const std::size_t n = t.a.code.size();
    if (used_ + n > capacity_) return false;
    std::memcpy(buffer_ + used_, t.a.code.data(), n);
    b.native = buffer_ + used_;
    used_ += (n + 15) & ~std::size_t{15};
    ++compiled_;
    return true;
}

#else  // no x86-64 host: blocks stay interpreted

bool Jit::compile(Block&, Cpu&) { return false; }

#endif

}  // namespace mv::p6
