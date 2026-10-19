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
constexpr i32 kCtxBase = static_cast<i32>(offsetof(JitCtx, base)), kCtxLimit = static_cast<i32>(offsetof(JitCtx, limit));
constexpr i32 kCtxBlock = static_cast<i32>(offsetof(JitCtx, block)), kCtxBudget = static_cast<i32>(offsetof(JitCtx, chain_budget));
constexpr i32 kCtxInsnsDone = static_cast<i32>(offsetof(JitCtx, insns_done)), kCtxEntryEip = static_cast<i32>(offsetof(JitCtx, entry_eip));
constexpr i32 kCtxEnded = static_cast<i32>(offsetof(JitCtx, ended));
constexpr i32 kCtxStores = static_cast<i32>(offsetof(JitCtx, stores)), kCtxMispredicts = static_cast<i32>(offsetof(JitCtx, mispredicts));
constexpr i32 kCtxInlineChains = static_cast<i32>(offsetof(JitCtx, inline_chains)), kCtxInlineMispredicts = static_cast<i32>(offsetof(JitCtx, inline_mispredicts));
constexpr i32 kCtxInlineSched = static_cast<i32>(offsetof(JitCtx, inline_sched)), kCtxInlineInsns = static_cast<i32>(offsetof(JitCtx, inline_insns));
constexpr i32 kCtxLogging = static_cast<i32>(offsetof(JitCtx, logging)), kCtxLogCount = static_cast<i32>(offsetof(JitCtx, log_count));
constexpr i32 kCtxLog = static_cast<i32>(offsetof(JitCtx, log));
// The scratch slots, one owner each: the cache call's saved RSI RAX RDI; INC/DEC's CF; a new ESP; a computed
// address or target; the x87 control words (the guest's for the host, C++'s); two pairs of saved RSI/RDI (an
// instruction with two accesses) or DIV's operands.
constexpr i32 slot(unsigned n) { return kCtxScratch + static_cast<i32>(8 * n); }
constexpr i32 kSaveRsi = slot(0), kSaveRax = slot(1), kSaveRdi = slot(2), kIncCf = slot(3), kNewEsp = slot(4), kCtxEa = slot(5);
constexpr i32 kCtxHostCw = slot(6), kCtxDefCw = slot(6) + 2;
constexpr i32 kTmp0 = slot(7), kTmp1 = slot(8), kTmp2 = slot(9), kTmp3 = slot(10);
constexpr i32 kCtxSseHost = slot(6) + 4, kCtxSseDef = slot(11);  // u32 each: the guest's MXCSR for the host, C++'s
constexpr i32 kCr0Off = static_cast<i32>(offsetof(CpuState, cr0));
constexpr i32 kFpuOff = static_cast<i32>(offsetof(CpuState, fpu));
constexpr i32 kSt0 = kFpuOff + static_cast<i32>(offsetof(FpuState, st));
constexpr i32 kFcw = kFpuOff + static_cast<i32>(offsetof(FpuState, fcw));
constexpr i32 kFsw = kFpuOff + static_cast<i32>(offsetof(FpuState, fsw));
constexpr i32 kFtw = kFpuOff + static_cast<i32>(offsetof(FpuState, ftw));
constexpr i32 kFop = kFpuOff + static_cast<i32>(offsetof(FpuState, fop));
constexpr i32 kFip = kFpuOff + static_cast<i32>(offsetof(FpuState, fip));
constexpr i32 kFdp = kFpuOff + static_cast<i32>(offsetof(FpuState, fdp));
constexpr i32 kFcs = kFpuOff + static_cast<i32>(offsetof(FpuState, fcs));
constexpr i32 kFds = kFpuOff + static_cast<i32>(offsetof(FpuState, fds));
constexpr i32 kMxcsr = kFpuOff + static_cast<i32>(offsetof(FpuState, mxcsr));
constexpr i32 kXmm = kFpuOff + static_cast<i32>(offsetof(FpuState, xmm));
constexpr i32 kCr4Off = static_cast<i32>(offsetof(CpuState, cr4));
constexpr i32 seg_selector(unsigned s) { return static_cast<i32>(offsetof(CpuState, seg) + s * sizeof(Segment) + offsetof(Segment, selector)); }
// Native x87 needs the interpreter's 80-bit registers in 16-byte slots (GCC/Clang on x86-64).
constexpr bool kNativeX87 = kHostX87 && sizeof(long double) == 16;
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
    ++c->helper_ops[static_cast<unsigned>(b.insns[i].op) & 255];
    if (b.insns[i].op == Op::X87) {
        const Insn& x = b.insns[i];
        // Form: escape (0-7), then the ModRM reg field, memory or register; register forms of D9/DB keep the low bits.
        const unsigned form = (x.opcode - 0xD8u) * 32u + (x.mem ? 16u : 0u) + (x.mem ? x.reg : ((x.opcode == 0xD9 || x.opcode == 0xDB) ? (x.modrm & 0x3F) >> 2 : x.reg)) ;
        ++c->helper_ops[256 + (form & 255)];
    }
    if (b.insns[i].op == Op::Simd) {
        const Insn& x = b.insns[i];
        // Form: the opcode, the prefix (none, 66, F3, F2), memory or register.
        const unsigned pfx = x.prefix_66 ? 1u : x.rep == 1 ? 2u : x.rep == 2 ? 3u : 0u;
        ++c->helper_ops[512 + (x.opcode * 4u + pfx) * 2u + (x.mem ? 1u : 0u)];
    }
    bool ok;
    try {
        ok = cpu.exec_decoded(b.insns[i]);
    } catch (...) {
        *c->error = std::current_exception();
        return (i << 2) | kJitFault;
    }
    const u64 penalty = mem.take_penalty();
    c->base += penalty;
    c->penalties += penalty;
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
    c->penalties += p;
    // The limit at the next boundary: leave there. Past the block's end only: a later boundary may reach it, which
    // the interpreter finds instruction by instruction — hand the rest of the block over to it.
    if (c->base + c->block->after[i] >= c->limit) c->exit = 1;
    else if (c->base + c->block->cycles >= c->limit) c->exit = 2;
}

// The end of a block whose last instruction left normally: the engine costs the exit and returns where the run goes
// on (the next block's native code, or null: back to the dispatcher).
void* h_chain(JitCtx* c, Block::Link* slot) { return c->engine->chain(*c, slot); }

void h_materialise(CpuState* s) { s->materialise(); }
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
    Translator(Block& b, Cpu& cpu, BlockEngine::NativeView ev) : b_(b), cpu_(cpu), st_(cpu.state()), ev_(ev) {}
    Asm a;
    std::size_t body{0};  // where the block's own code starts, past the prologue (chains enter there)

    void run() {
        epilogue_ = a.label();
        prologue();
        body = a.size();
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
            if (s.cw_before) default_cw();
            if (s.sse_before) sse_flush();
            call_helper(s.index);
            if (s.index + 1 == b_.n) a.jmp(epilogue_);
            else {
                a.test(4, rax, rax);
                a.jcc(NE, epilogue_);
                if (s.cw_after) guest_cw();
                if (s.sse_after) sse_load();
                a.jmp(s.resume);
            }
        }
    }

private:
    struct Stub {
        Label entry, resume;
        unsigned index;
        bool cw_before, cw_after;  // the host x87 control word was the guest's at the jump / must be at the resume
        bool sse_before, sse_after;  // the same for the host MXCSR
    };

    Block& b_;
    Cpu& cpu_;
    const CpuState& st_;
    BlockEngine::NativeView ev_;
    Label epilogue_;
    std::vector<Stub> stubs_;
    Known known_{Known::Unknown};
    unsigned known_size_{4};
    bool host_flags_{false};   // the host flags hold the last flag writer's result
    bool host_cf_{false};      // ... including CF (not after INC/DEC)
    bool cw_active_{false};    // the host x87 runs under the guest's control word (see x87_enter)
    bool sse_active_{false};   // the host SSE runs under the guest's MXCSR (see sse_enter)

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
        a.x87_mem(0xD9, 7, ctx_at(kCtxDefCw));  // fnstcw: the control word C++ code runs under
        a.stmxcsr(ctx_at(kCtxSseDef));  // and C++'s MXCSR, its sticky flags cleared
        a.alu_imm(4, 4, ctx_at(kCtxSseDef), ~0x3Fu);
        a.ldmxcsr(ctx_at(kCtxSseDef));
        Memory& m = cpu_.memory();
        a.mov64(kTlb, reinterpret_cast<u64>(m.tlb()));
        a.mov64(kRam, reinterpret_cast<u64>(m.bus().ram()));
        a.mov64(kMru, reinterpret_cast<u64>(cpu_.cache_timing().l1d_mru()));
        a.mov64(kCodeMap, reinterpret_cast<u64>(m.bus().code_map()));
    }
    void epilogue() {
        a.x87_mem(0xD9, 5, ctx_at(kCtxDefCw));  // fldcw: back to it
        a.ldmxcsr(ctx_at(kCtxSseDef));
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
        x87_leave();
        sse_leave();
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
    Label fallback(unsigned i, Label resume) { return fallback(i, resume, cw_active_, cw_active_); }
    Label fallback(unsigned i, Label resume, bool cw_before, bool cw_after) { return fallback(i, resume, cw_before, cw_after, sse_active_, sse_active_); }
    Label fallback(unsigned i, Label resume, bool cw_before, bool cw_after, bool sse_before, bool sse_after) {
        Stub s{a.label(), resume, i, cw_before, cw_after, sse_before, sse_after};
        stubs_.push_back(s);
        return s.entry;
    }
    // The block's normal end at EIP `eip` (kNoEip: already stored): the engine costs the exit through link slot `slot`
    // and may chain into the next block's native code.
    static constexpr u64 kNoEip = ~u64{0};
    void end(unsigned i, u64 eip, unsigned slot) {
        x87_leave();
        sse_leave();
        if (eip != kNoEip) a.mov_imm(4, st_at(kEipOff), static_cast<u32>(eip));
        Label slow = a.label();
        const BranchKind kind = b_.branch;
        const bool inline_chain = eip != kNoEip && (kind == BranchKind::None || kind == BranchKind::Conditional || kind == BranchKind::Direct ||
                                                     kind == BranchKind::Call);
        // Taken as the engine counts it: a jump to the next instruction is not taken.
        if (inline_chain) chain_inline(static_cast<u32>(eip), slot, static_cast<u32>(eip) != next_eip(b_.n - 1u), slow);
        a.bind(slow);
        a.mov(8, kArg0, kCtx);
        a.mov64(kArg1, reinterpret_cast<u64>(&b_.links[slot]));
        a.call_abs(reinterpret_cast<const void*>(&h_chain));
        Label back = a.label();
        a.test(8, rax, rax);
        a.jcc(E, back);
        a.jmp(rax);
        a.bind(back);
        a.mov32(rax, (i << 2) | kJitEnd);
        a.jmp(epilogue_);
    }
    // BlockEngine::chain's common case in the block's own code: the exit's prediction worked out without committing,
    // every condition of a chain checked (the link, the next block's translation, the budget, the limit with the
    // slowest fetch, no device write into code, no penalty pending, the fetch translation, the fetch's lines being
    // the most recent of their sets: no cache state to change); then the predictor, the statistics and the run
    // move on and the next block's code is entered. Anything else goes to `slow`, where the engine does it all.
    void chain_inline(u32 target, unsigned slot, bool taken, Label slow) {
        static const i32 kEntry = [] {
            static const Block probe;
            return static_cast<i32>(reinterpret_cast<const u8*>(&probe.entry) - reinterpret_cast<const u8*>(&probe));
        }();
        const i32 eBody = kEntry + static_cast<i32>(offsetof(Block::Entry, body));
        const i32 eHost = kEntry + static_cast<i32>(offsetof(Block::Entry, host_page));
        const i32 eRuns = kEntry + static_cast<i32>(offsetof(Block::Entry, runs));
        const i32 eCycles = kEntry + static_cast<i32>(offsetof(Block::Entry, cycles));
        const i32 eLine = kEntry + static_cast<i32>(offsetof(Block::Entry, fetch_line));
        const i32 eLines = kEntry + static_cast<i32>(offsetof(Block::Entry, fetch_lines));
        const i32 eEip = kEntry + static_cast<i32>(offsetof(Block::Entry, jit_eip));
        const BranchKind kind = b_.branch;
        // R11 = the link, RCX = the next block, RDX = its entry.
        a.mov64(r11, reinterpret_cast<u64>(&b_.links[slot]));
        a.mov(8, rcx, mem(r11, static_cast<i32>(offsetof(Block::Link, to))));
        a.test(8, rcx, rcx);
        a.jcc(E, slow);
        a.mov64(rax, reinterpret_cast<u64>(ev_.link_epoch));
        a.mov(4, rax, mem(rax));
        a.alu(4, 7, rax, mem(r11, static_cast<i32>(offsetof(Block::Link, epoch))));
        a.jcc(NE, slow);
        a.mov(8, rdx, mem(rcx, eBody));
        a.test(8, rdx, rdx);
        a.jcc(E, slow);
        a.alu_imm(4, 7, mem(rcx, eEip), target);  // the next block is the target's (a link is only a hint)
        a.jcc(NE, slow);
        a.alu_imm(4, 7, ctx_at(kCtxBudget), 0);
        a.jcc(E, slow);
        // R8D = 1 on a mispredict; R9D = the conditional predictor's new state.
        if (kind == BranchKind::Conditional) {
            a.movzx(r9, 1, mem_abs(r10, &b_.exit_counter));
            a.mov32(r8, 0);
            a.alu_imm(4, 7, r9, 2);
            a.setcc(taken ? B : AE, r8);  // taken: predicted not taken below 2; not taken: predicted taken from 2
            if (taken) {
                a.lea(4, rax, mem(r9, 1));
                a.mov32(rsi, 3);
                a.alu(4, 7, rax, rsi);
                a.cmov(4, A, rax, rsi);
                a.mov(4, r9, rax);
            } else {
                a.lea(4, rax, mem(r9, -1));
                a.test(4, r9, r9);
                a.cmov(4, NE, r9, rax);
            }
        } else if (kind == BranchKind::Direct || kind == BranchKind::Call) {
            a.movzx(r8, 1, mem_abs(r10, &b_.exit_seen));
            a.alu_imm(4, 6, r8, 1);
        } else {
            a.mov32(r8, 0);
        }
        // R10 = the cycle the block ends at.
        a.lea(4, rax, mem(r8, r8, 1, 0));
        a.shift_imm(4, 4, rax, 2);  // x12
        static_assert(tp::kBlockMispredict == 12);
        a.mov(8, r10, ctx_at(kCtxBase));
        a.alu(8, 0, r10, rax);
        a.alu_imm(8, 0, r10, b_.cycles);
        // The limit with the slowest fetch.
        a.mov(4, rax, mem(rcx, eCycles));
        a.alu(8, 0, rax, r10);
        a.alu_imm(8, 0, rax, tp::kL2Latency + tp::kDramLatency);
        a.alu(8, 7, rax, ctx_at(kCtxLimit));
        a.jcc(AE, slow);
        // Nothing pending: a device\x27s write into code, a penalty not yet charged.
        a.mov64(rax, reinterpret_cast<u64>(cpu_.memory().bus().code_pending_byte()));
        a.alu_imm(1, 7, mem(rax), 0);
        a.jcc(NE, slow);
        a.mov64(rax, reinterpret_cast<u64>(cpu_.memory().penalty_word()));
        a.alu_imm(8, 7, mem(rax), 0);
        a.jcc(NE, slow);
        a.mov64(rax, reinterpret_cast<u64>(cpu_.memory().device_cycles_word()));
        a.alu_imm(8, 7, mem(rax), 0);
        a.jcc(NE, slow);
        // The fetch translation: a TLB hit on the next block\x27s page (the linear address is the target: flat CS).
        const u32 lin = st_.seg[kCs].base + target;
        const i32 tlb = static_cast<i32>(((lin >> 12) & (Memory::kTlbSize - 1)) * 32);
        a.alu_imm(4, 7, mem(kTlb, tlb), lin >> 12);
        a.jcc(NE, slow);
        a.mov(8, rax, mem(kTlb, tlb + 8));
        a.alu(8, 7, rax, mem(rcx, eHost));
        a.jcc(NE, slow);
        // The fetch: one or two lines, each its set\x27s most recent in L1I (else the model has work: the engine\x27s).
        const unsigned isets = cpu_.cache_timing().l1i_sets();
        a.mov64(rax, reinterpret_cast<u64>(cpu_.cache_timing().l1i_mru()));
        a.movzx(rdi, 1, mem(rcx, eLines));
        a.alu_imm(4, 7, rdi, 2);
        a.jcc(A, slow);
        a.mov(4, rsi, mem(rcx, eLine));
        a.lea(4, rdi, mem(rsi, -1));
        a.alu_imm(4, 4, rdi, isets - 1);
        a.alu(4, 7, rsi, mem(rax, rdi, 2, 0));
        a.jcc(NE, slow);
        Label one = a.label();
        a.alu_imm(1, 7, mem(rcx, eLines), 1);
        a.jcc(E, one);
        a.lea(4, rdi, mem(rsi, 0));
        a.alu_imm(4, 4, rdi, isets - 1);
        a.lea(4, rsi, mem(rsi, 1));
        a.alu(4, 7, rsi, mem(rax, rdi, 2, 0));
        a.jcc(NE, slow);
        a.bind(one);
        // ---- committed from here ----
        if (kind == BranchKind::Conditional) a.mov(1, mem_abs(rax, &b_.exit_counter), r9);
        if (kind != BranchKind::None) {
            a.mov_imm(1, mem_abs(rax, &b_.exit_seen), 1);
            a.mov_imm(4, mem_abs(rax, &b_.exit_target), target);
        }
        if (kind == BranchKind::Call) {
            a.mov64(rax, reinterpret_cast<u64>(ev_.rsb_top));
            a.mov(4, rsi, mem(rax));
            a.mov64(rdi, reinterpret_cast<u64>(ev_.rsb));
            a.mov_imm(4, mem(rdi, rsi, 2, 0), next_eip(b_.n - 1u));
            a.lea(4, rsi, mem(rsi, 1));
            a.alu_imm(4, 4, rsi, 15);
            a.mov(4, mem(rax), rsi);
        }
        // Statistics in the run's own context (the engine folds them in once the run is over).
        a.alu(4, 0, ctx_at(kCtxMispredicts), r8);
        a.alu(4, 0, ctx_at(kCtxInlineMispredicts), r8);
        a.alu_imm(4, 0, ctx_at(kCtxInlineChains), 1);
        a.alu_imm(8, 0, ctx_at(kCtxInlineSched), b_.cycles);
        a.alu_imm(8, 0, ctx_at(kCtxInlineInsns), b_.n);
        a.alu_imm(4, 5, ctx_at(kCtxBudget), 1);
        a.alu_imm(8, 0, ctx_at(kCtxInsnsDone), b_.n);
        a.mov(8, ctx_at(kCtxBase), r10);
        a.mov(8, ctx_at(kCtxBlock), rcx);
        a.mov_imm(4, ctx_at(kCtxEntryEip), target);
        a.mov_imm(1, ctx_at(kCtxEnded), 0);
        a.alu_imm(8, 0, mem(rcx, eRuns), 1);
        a.jmp(rdx);
    }
    // A memory operand at an absolute host address (through `scratch`, loaded first).
    Mem mem_abs(Reg scratch, const void* p) {
        a.mov64(scratch, reinterpret_cast<u64>(p));
        return mem(scratch);
    }
    void leave(unsigned i, u32 eip, u32 kind) {
        if (sse_active_) sse_flush();  // (code only: other paths go on with it active)
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
            a.alu_imm(4, 0, ctx_at(kCtxStores), 1);  // the spin skip's evidence
            // Write tracking: a watched page's first write is the interpreter's (it reports and unwatches the page);
            // with every page logged, the page is marked here (twice is the same if a later check fails).
            Memory& m = cpu_.memory();
            if (m.dirty_tracking() && m.selective()) {
                a.mov64(r11, reinterpret_cast<u64>(m.watched_map()));
                a.bt(8, mem(r11), rdx);
                a.jcc(B, fail);
            } else if (m.dirty_tracking()) {
                a.mov64(r11, reinterpret_cast<u64>(m.dirty_map()));
                a.op_rm(8, {0x0F, 0xAB}, rdx, mem(r11));  // bts
            }
        }
        a.mov(4, rdi, mem(kTlb, rcx, 0, 4));  // physical page
        a.mov(4, rdx, rax);
        a.alu_imm(4, 4, rdx, 0xFFF);
        a.alu(4, 1, rdi, rdx);  // or: the physical address
        a.alu(8, 0, rsi, rdx);  // the host byte
        if (write) log_store(size);
    }
    // The spin probe's store log (only when the context asks: two instructions otherwise). RSI = the host bytes.
    void log_store(unsigned size) {
        Label done = a.label();
        a.alu_imm(1, 7, ctx_at(kCtxLogging), 0);
        a.jcc(E, done);
        for (unsigned part = 0; part < (size == 16 ? 2u : 1u); ++part) {
            const unsigned n = size > 8 ? 8 : size;
            Label over = a.label();
            a.mov(4, rcx, ctx_at(kCtxLogCount));
            a.alu_imm(4, 7, rcx, 256);
            a.jcc(AE, over);
            a.shift_imm(4, 4, rcx, 4);
            a.lea(8, r10, mem(kCtx, rcx, 0, kCtxLog));
            a.lea(8, r11, mem(rsi, static_cast<i32>(8 * part)));
            a.mov64(rax, static_cast<u64>(n) << 56);
            a.alu(8, 1, r11, rax);
            a.mov(8, mem(r10, 0), r11);
            const Mem at = mem(rsi, static_cast<i32>(8 * part));
            if (n == 8) a.mov(8, rax, at);
            else if (n == 4) a.mov(4, rax, at);
            else a.movzx(rax, n, at);
            a.mov(8, mem(r10, 8), rax);
            a.bind(over);
            a.alu_imm(4, 0, ctx_at(kCtxLogCount), 1);
        }
        a.bind(done);
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
        a.mov(8, ctx_at(kSaveRsi), rsi);
        a.mov(8, ctx_at(kSaveRax), rax);
        a.mov(8, ctx_at(kSaveRdi), rdi);
        a.mov(4, kArg1, rdi);
        a.mov(8, kArg0, kCtx);
        a.mov32(kArg2, (i << 8) | (store ? 1u : 0u));
        a.call_abs(reinterpret_cast<const void*>(&h_cache));
        a.mov(8, rsi, ctx_at(kSaveRsi));
        a.mov(8, rax, ctx_at(kSaveRax));
        a.mov(8, rdi, ctx_at(kSaveRdi));
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

    // ---- x87 (the forms a profile of the boot found hot; the rest stay with the interpreter) ----
    // The host runs the guest's arithmetic under the control word the interpreter's HostFpu uses (every exception
    // masked, the guest's precision and rounding), set once and kept until C++ code runs; FNCLEX with it, so the
    // host's sticky flags after it are the native instructions' own (the guest's are sticky too: OR-ing is exact).
    void x87_enter() {
        if (cw_active_) return;
        guest_cw();
        cw_active_ = true;
    }
    void x87_leave() {
        if (!cw_active_) return;
        default_cw();
        cw_active_ = false;
    }
    void guest_cw() {
        a.movzx(rax, 2, st_at(kFcw));
        a.alu_imm(4, 4, rax, 0x0F00);
        a.alu_imm(4, 1, rax, 0x3F);
        a.mov(2, ctx_at(kCtxHostCw), rax);
        a.x87_mem(0xD9, 5, ctx_at(kCtxHostCw));  // fldcw
        a.fnclex();
    }
    void default_cw() { a.x87_mem(0xD9, 5, ctx_at(kCtxDefCw)); }
    // R8 = TOP.
    void x87_top() {
        a.movzx(r8, 2, st_at(kFsw));
        a.shift_imm(4, 5, r8, 11);
        a.alu_imm(4, 4, r8, 7);
    }
    // dst = the physical register of ST(i) (TOP in R8).
    void x87_phys(Reg dst, unsigned i) {
        a.lea(4, dst, mem(r8, static_cast<i32>(i)));
        a.alu_imm(4, 4, dst, 7);
    }
    // Jump to `fail` unless physical register `p` is empty (want_empty) or in use.
    void x87_tag_test(Reg p, bool want_empty, Label fail) {
        a.movzx(rdx, 2, st_at(kFtw));
        a.lea(4, rcx, mem(p, p, 0, 0));
        a.shift_cl(4, 5, rdx);
        a.alu_imm(4, 4, rdx, 3);
        a.alu_imm(4, 7, rdx, 3);
        a.jcc(want_empty ? NE : E, fail);
    }
    // dst = the address of physical register `p`'s 80-bit value.
    void x87_addr(Reg dst, Reg p) {
        a.mov(4, dst, p);
        a.shift_imm(8, 4, dst, 4);
        a.lea(8, dst, mem(kSt, dst, 0, kSt0));
    }
    // Jump to `fail` unless the value at [addr] is a normal number or a zero (no operand check can fire).
    void x87_operand_ok(Reg addr, Label fail) {
        Label zero = a.label(), ok = a.label();
        a.movzx(rdx, 2, mem(addr, 8));
        a.alu_imm(4, 4, rdx, 0x7FFF);
        a.jcc(E, zero);
        a.alu_imm(4, 7, rdx, 0x7FFF);
        a.jcc(E, fail);
        a.test_imm(1, mem(addr, 7), 0x80);  // the integer bit
        a.jcc(E, fail);
        a.jmp(ok);
        a.bind(zero);
        a.alu_imm(8, 7, mem(addr, 0), 0);
        a.jcc(NE, fail);
        a.bind(ok);
    }
    // Jump to `fail` unless the double at [RSI] is a normal number or a zero.
    void x87_m64_ok(Label fail) {
        Label zero = a.label(), ok = a.label();
        a.mov(8, rax, mem(rsi));
        a.mov(8, rdx, rax);
        a.shift_imm(8, 5, rdx, 52);
        a.alu_imm(4, 4, rdx, 0x7FF);
        a.jcc(E, zero);
        a.alu_imm(4, 7, rdx, 0x7FF);
        a.jcc(E, fail);
        a.jmp(ok);
        a.bind(zero);
        a.shift_imm(8, 4, rax, 1);  // the sign out: zero only if every other bit is
        a.test(8, rax, rax);
        a.jcc(NE, fail);
        a.bind(ok);
    }
    // Jump to `fail` unless the single at [RSI] is a normal number or a zero.
    void x87_m32_ok(Label fail) {
        Label zero = a.label(), ok = a.label();
        a.mov(4, rax, mem(rsi));
        a.mov(4, rdx, rax);
        a.alu_imm(4, 4, rdx, 0x7F800000);
        a.jcc(E, zero);
        a.alu_imm(4, 7, rdx, 0x7F800000);
        a.jcc(E, fail);
        a.jmp(ok);
        a.bind(zero);
        a.test_imm(4, rax, 0x7FFFFFFF);
        a.jcc(NE, fail);
        a.bind(ok);
    }
    // EDX = the tag of the value at [addr], as tag_of computes it: 0 valid, 1 zero, 2 special.
    void x87_tag_of(Reg addr) {
        Label zero_exp = a.label(), special = a.label(), done = a.label();
        a.movzx(rdx, 2, mem(addr, 8));
        a.alu_imm(4, 4, rdx, 0x7FFF);
        a.jcc(E, zero_exp);
        a.alu_imm(4, 7, rdx, 0x7FFF);
        a.jcc(E, special);
        a.test_imm(1, mem(addr, 7), 0x80);
        a.jcc(E, special);
        a.mov32(rdx, 0);
        a.jmp(done);
        a.bind(zero_exp);
        a.mov32(rdx, 1);
        a.alu_imm(8, 7, mem(addr, 0), 0);
        a.jcc(E, done);
        a.bind(special);
        a.mov32(rdx, 2);
        a.bind(done);
    }
    // Physical register `p`'s tag = EDX.
    void x87_set_tag(Reg p) {
        a.lea(4, rcx, mem(p, p, 0, 0));
        a.mov32(rax, 3);
        a.shift_cl(4, 4, rax);
        a.unary(4, 2, rax);
        a.alu(2, 4, st_at(kFtw), rax);
        a.shift_cl(4, 4, rdx);
        a.alu(2, 1, st_at(kFtw), rdx);
    }
    void x87_set_empty(Reg p) {
        a.mov32(rdx, 3);
        x87_set_tag(p);
    }
    void x87_set_top(Reg t) {
        a.mov(4, rax, t);
        a.shift_imm(4, 4, rax, 11);
        a.alu_imm(2, 4, st_at(kFsw), 0xC7FF);
        a.alu(2, 1, st_at(kFsw), rax);
    }
    void x87_clear_c1() { a.alu_imm(2, 4, st_at(kFsw), 0xFDFF); }
    // After a host operation: its exception flags join the status word (all masked: no ES); C1 from it if asked.
    void x87_status(bool c1) {
        a.fnstsw_ax();
        a.mov(4, rdx, rax);
        a.alu_imm(4, 4, rdx, 0x3F);
        a.alu(2, 1, st_at(kFsw), rdx);
        if (c1) {
            a.alu_imm(4, 4, rax, 0x200);
            x87_clear_c1();
            a.alu(2, 1, st_at(kFsw), rax);
        }
    }
    // pop(): ST(0)'s register empty, TOP + 1 (TOP in R8).
    void x87_pop() {
        x87_set_empty(r8);
        a.lea(4, r9, mem(r8, 1));
        a.alu_imm(4, 4, r9, 7);
        x87_set_top(r9);
    }
    // The last-instruction pointers run() keeps for a non-control instruction.
    void x87_pointers(unsigned i, const Insn& in) {
        a.mov_imm(4, st_at(kFip), eip_of(i));
        a.movzx(rax, 2, st_at(seg_selector(kCs)));
        a.mov(2, st_at(kFcs), rax);
        a.mov_imm(2, st_at(kFop), static_cast<u32>(((in.opcode & 7) << 8) | in.modrm));
        if (in.mem) {
            a.mov(4, rax, ctx_at(kCtxEa));
            a.mov(4, st_at(kFdp), rax);
            a.movzx(rax, 2, st_at(seg_selector(in.mem_seg)));
            a.mov(2, st_at(kFds), rax);
        }
    }
    // #NM (CR0.EM/TS) and a pending #MF (ES) are the interpreter's; so is any unmasked exception.
    void x87_preamble(Label f, bool control) {
        a.test_imm(4, st_at(kCr0Off), cr0::EM | cr0::TS);
        a.jcc(NE, f);
        if (control) return;
        a.test_imm(2, st_at(kFsw), 0x80);
        a.jcc(NE, f);
        a.movzx(rax, 2, st_at(kFcw));
        a.alu_imm(4, 4, rax, 0x3F);
        a.alu_imm(4, 7, rax, 0x3F);
        a.jcc(NE, f);
    }
    // Copy the 80-bit value at [from] to [to].
    void x87_copy(Reg to, Reg from) {
        a.mov(8, rax, mem(from, 0));
        a.mov(8, mem(to, 0), rax);
        a.movzx(rax, 2, mem(from, 8));
        a.mov(2, mem(to, 8), rax);
    }

    enum class X87Form { None, LoadM32, StoreM32, LoadM64, StoreM64, ArithM32, CompareM32, ArithD8, ArithDcDe, LoadReg, StoreReg, Exchange, Sign, LoadConst, StoreSw };

    bool x87(unsigned i, const Insn& in, Label after) {
        if (!kNativeX87) return false;
        const u8 esc = in.opcode;
        X87Form k = X87Form::None;
        if (in.mem) {
            if (esc == 0xD9 && in.reg == 0) k = X87Form::LoadM32;
            else if (esc == 0xDD && in.reg == 0) k = X87Form::LoadM64;
            else if (esc == 0xDD && (in.reg == 2 || in.reg == 3)) k = X87Form::StoreM64;
            else if (esc == 0xD9 && (in.reg == 2 || in.reg == 3)) k = X87Form::StoreM32;
            else if (esc == 0xD8 && (in.reg == 2 || in.reg == 3)) k = X87Form::CompareM32;
            else if (esc == 0xD8) k = X87Form::ArithM32;
        } else {
            if (esc == 0xD8 && in.reg != 2 && in.reg != 3) k = X87Form::ArithD8;
            else if ((esc == 0xDC || esc == 0xDE) && in.reg != 2 && in.reg != 3) k = X87Form::ArithDcDe;
            else if (esc == 0xD9 && in.modrm >= 0xC0 && in.modrm <= 0xC7) k = X87Form::LoadReg;
            else if (esc == 0xD9 && in.modrm >= 0xC8 && in.modrm <= 0xCF) k = X87Form::Exchange;
            else if (esc == 0xD9 && (in.modrm == 0xE0 || in.modrm == 0xE1)) k = X87Form::Sign;
            else if (esc == 0xD9 && (in.modrm == 0xE8 || in.modrm == 0xEE)) k = X87Form::LoadConst;
            else if (esc == 0xDD && (in.reg == 2 || in.reg == 3)) k = X87Form::StoreReg;
            else if (esc == 0xDF && in.modrm == 0xE0) k = X87Form::StoreSw;
        }
        if (k == X87Form::None) return false;
        const bool write = k == X87Form::StoreM32 || k == X87Form::StoreM64;
        const bool m64 = k == X87Form::LoadM64 || k == X87Form::StoreM64;
        const unsigned msize = m64 ? 8 : 4;
        if (in.mem && !segment_ok(in.mem_seg, msize, write)) return false;
        const bool host = k == X87Form::StoreM32 || k == X87Form::LoadM64 || k == X87Form::StoreM64 || k == X87Form::ArithM32 || k == X87Form::CompareM32 || k == X87Form::ArithD8 || k == X87Form::ArithDcDe ||
                          k == X87Form::LoadM32;
        const bool cw_before = cw_active_;
        const Label f = fallback(i, after, cw_before, host || cw_before);
        x87_preamble(f, k == X87Form::StoreSw);
        if (k == X87Form::StoreSw) {
            a.movzx(rax, 2, st_at(kFsw));
            a.mov(2, greg(kEax, 2), rax);
            host_flags_ = host_cf_ = false;  // the preamble's TEST took them
            finish(i, false);
            return true;
        }
        if (in.mem) {
            effective_address(in);
            a.mov(4, ctx_at(kCtxEa), rax);
            access_checks(in.mem_seg, msize, write, f);
            if (m64) {
                // The interpreter reads or writes the two halves: native when they share a cache line.
                a.mov(4, rax, rsi);
                a.alu_imm(4, 4, rax, 31);
                a.alu_imm(4, 7, rax, 24);
                a.jcc(A, f);
            }
            if (k == X87Form::LoadM64) x87_m64_ok(f);
            else if (!write) x87_m32_ok(f);
        }
        // Every check before any effect: the stack, the operands' classes.
        x87_top();
        const unsigned ri = in.rm;
        switch (k) {
            case X87Form::LoadM32: case X87Form::LoadM64: case X87Form::LoadConst:
                x87_phys(r9, 7);
                x87_tag_test(r9, true, f);
                break;
            case X87Form::StoreM32: case X87Form::StoreM64:
                x87_tag_test(r8, false, f);
                break;
            case X87Form::ArithM32: case X87Form::CompareM32: case X87Form::Sign: case X87Form::StoreReg:
                x87_tag_test(r8, false, f);
                x87_addr(r10, r8);
                x87_operand_ok(r10, f);
                break;
            case X87Form::ArithD8: case X87Form::ArithDcDe: case X87Form::Exchange:
                x87_phys(r9, ri);
                x87_tag_test(r8, false, f);
                x87_tag_test(r9, false, f);
                x87_addr(r10, r8);
                x87_addr(r11, r9);
                x87_operand_ok(r10, f);
                x87_operand_ok(r11, f);
                break;
            case X87Form::LoadReg:
                x87_phys(r9, ri);
                x87_tag_test(r9, false, f);
                x87_addr(r11, r9);
                x87_operand_ok(r11, f);
                x87_phys(r9, 7);
                x87_tag_test(r9, true, f);
                break;
            default: break;
        }
        if (in.mem) {
            cache(i, write);  // the interpreter's one access (clobbers R8-R11)
            x87_top();
        }
        x87_pointers(i, in);
        if (host) x87_enter();
        switch (k) {
            case X87Form::LoadM32: case X87Form::LoadM64: {
                x87_phys(r9, 7);
                x87_addr(r10, r9);
                a.x87_mem(m64 ? 0xDD : 0xD9, 0, mem(rsi));   // fld qword / dword
                a.x87_mem(0xDB, 7, mem(r10));   // fstp tbyte
                x87_clear_c1();
                x87_set_top(r9);
                x87_tag_of(r10);
                x87_set_tag(r9);
                break;
            }
            case X87Form::LoadConst: {
                x87_phys(r9, 7);
                x87_addr(r10, r9);
                const bool one = in.modrm == 0xE8;
                a.mov64(rax, one ? 0x8000000000000000ull : 0);
                a.mov(8, mem(r10, 0), rax);
                a.mov_imm(2, mem(r10, 8), one ? 0x3FFF : 0);
                x87_clear_c1();
                x87_set_top(r9);
                a.mov32(rdx, one ? 0 : 1);
                x87_set_tag(r9);
                break;
            }
            case X87Form::StoreM32: case X87Form::StoreM64: {
                x87_addr(r10, r8);
                a.x87_mem(0xDB, 5, mem(r10));   // fld tbyte
                a.x87_mem(m64 ? 0xDD : 0xD9, 3, mem(rsi));   // fstp qword / dword
                x87_status(true);
                if (in.reg == 3) x87_pop();
                break;
            }
            case X87Form::ArithM32: {
                x87_addr(r10, r8);
                a.x87_mem(0xDB, 5, mem(r10));
                a.x87_mem(0xD8, in.reg, mem(rsi));  // the operation with the single, as the interpreter's on its conversion
                x87_status(true);
                a.x87_mem(0xDB, 7, mem(r10));
                x87_tag_of(r10);
                x87_set_tag(r8);
                break;
            }
            case X87Form::CompareM32: {
                x87_addr(r10, r8);
                a.x87_mem(0xDB, 5, mem(r10));
                a.x87_mem(0xD8, 3, mem(rsi));  // fcomp dword: the host's C3 C2 C0 are the guest's for ordered operands
                a.fnstsw_ax();
                a.alu_imm(4, 4, rax, 0x4500);
                a.alu_imm(2, 4, st_at(kFsw), 0xB8FF);  // C0 C1 C2 C3 cleared
                a.alu(2, 1, st_at(kFsw), rax);
                if (in.reg == 3) x87_pop();
                break;
            }
            case X87Form::ArithD8: case X87Form::ArithDcDe: {
                // D8: ST(0) = ST(0) op ST(i); DC/DE: ST(i) = ST(i) op' ST(0) (the reversed SUB/DIV encodings).
                const bool d8 = k == X87Form::ArithD8;
                static constexpr u8 kD8[8] = {0xC1, 0xC9, 0, 0, 0xE1, 0xE9, 0xF1, 0xF9};  // fadd fmul - - fsub fsubr fdiv fdivr st, st(1)
                static constexpr u8 kDc[8] = {0xC1, 0xC9, 0, 0, 0xE9, 0xE1, 0xF9, 0xF1};
                const Reg dst = d8 ? r10 : r11, src = d8 ? r11 : r10;
                a.x87_mem(0xDB, 5, mem(src));  // the second operand (host ST(1))
                a.x87_mem(0xDB, 5, mem(dst));  // the first (host ST(0))
                a.x87_raw(0xD8, d8 ? kD8[in.reg] : kDc[in.reg]);
                x87_status(true);
                a.x87_mem(0xDB, 7, mem(dst));
                a.x87_raw(0xDD, 0xD8);  // fstp st(0): drop the second
                x87_tag_of(dst);
                x87_set_tag(d8 ? r8 : r9);
                if (esc == 0xDE) x87_pop();
                break;
            }
            case X87Form::LoadReg: {
                x87_phys(r9, 7);
                x87_addr(r10, r9);
                x87_copy(r10, r11);
                x87_clear_c1();
                x87_set_top(r9);
                x87_tag_of(r10);
                x87_set_tag(r9);
                break;
            }
            case X87Form::StoreReg: {
                x87_phys(r9, ri);
                x87_addr(r11, r9);
                x87_copy(r11, r10);
                x87_tag_of(r11);
                x87_set_tag(r9);
                x87_clear_c1();
                if (in.reg == 3) x87_pop();
                break;
            }
            case X87Form::Exchange: {
                // ST(0) and ST(i) swap; each register's tag from its new value.
                a.mov(8, rax, mem(r10, 0));
                a.mov(8, rcx, mem(r11, 0));
                a.mov(8, mem(r10, 0), rcx);
                a.mov(8, mem(r11, 0), rax);
                a.movzx(rax, 2, mem(r10, 8));
                a.movzx(rcx, 2, mem(r11, 8));
                a.mov(2, mem(r10, 8), rcx);
                a.mov(2, mem(r11, 8), rax);
                x87_tag_of(r10);
                x87_set_tag(r8);
                x87_tag_of(r11);
                x87_set_tag(r9);
                x87_clear_c1();
                break;
            }
            case X87Form::Sign: {
                if (in.modrm == 0xE0) a.alu_imm(1, 6, mem(r10, 9), 0x80);  // FCHS: xor the sign
                else a.alu_imm(1, 4, mem(r10, 9), 0x7F);                   // FABS: clear it
                x87_tag_of(r10);
                x87_set_tag(r8);
                x87_clear_c1();
                break;
            }
            default: break;
        }
        host_flags_ = host_cf_ = false;
        finish(i, in.mem);
        return true;
    }

    // SHL/SHR/SAR by an immediate or CL: the interpreter's formulas (its values for the undefined OF and AF too):
    // CF = SHL: bit `bits` of a << n; SHR: bit n-1 of a; SAR: bit n-1 of a sign-extended. OF = SHL: msb(r) ^ CF;
    // SHR: msb(a) ^ msb(r); SAR: 0. A count of zero changes nothing (the destination is read, not written).
    bool shift(unsigned i, const Insn& in, Label after) {
        const unsigned size = in.size, bits = size * 8;
        const bool m = in.dst.kind == OpndKind::Mem;
        if (!m && in.dst.kind != OpndKind::Reg) return false;
        const bool by_cl = in.src.kind == OpndKind::Reg;
        if (!by_cl && in.src.kind != OpndKind::Imm) return false;
        if (by_cl && in.src.reg != 1) return false;  // CL
        const unsigned imm = by_cl ? 0 : (in.imm & 0x1F);
        if (m && !segment_ok(in.mem_seg, size, true)) return false;
        if (m) {
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, size, true, f);
            cache(i, false);  // the interpreter reads the destination before it looks at the count
        }
        Label done = a.label();
        if (!by_cl && imm == 0) {
            if (m) host_flags_ = host_cf_ = false;
            finish(i, m);
            return true;
        }
        if (by_cl) {
            a.movzx(rcx, 1, greg(1, 1));
            a.alu_imm(4, 4, rcx, 0x1F);
            a.jcc(E, done);  // count 0: no write, no flags
        } else {
            a.mov32(rcx, imm);
        }
        if (m) {
            cache(i, true);  // keeps RSI, RAX, RDI; RCX is reloaded below
            if (by_cl) {
                a.movzx(rcx, 1, greg(1, 1));
                a.alu_imm(4, 4, rcx, 0x1F);
            } else {
                a.mov32(rcx, imm);
            }
            load_host(rax, size);
        } else {
            load_greg(rax, in.dst.reg, size);
        }
        // EAX = a (zero-extended), ECX = n (1..31). EDX = r, R9D = CF, R10D = OF.
        switch (in.op) {
            case Op::Shl:
                a.mov(8, rdx, rax);
                a.shift_cl(8, 4, rdx);
                a.mov(8, r9, rdx);
                a.shift_imm(8, 5, r9, static_cast<u8>(bits));
                a.alu_imm(4, 4, r9, 1);
                a.alu_imm(4, 4, rdx, mask_of(size));
                a.mov(4, r10, rdx);
                a.shift_imm(4, 5, r10, static_cast<u8>(bits - 1));
                a.alu(4, 6, r10, r9);
                break;
            case Op::Shr:
                a.mov(4, rdx, rax);
                a.shift_cl(4, 5, rdx);
                a.lea(4, rcx, mem(rcx, -1));
                a.mov(4, r9, rax);
                a.shift_cl(4, 5, r9);
                a.alu_imm(4, 4, r9, 1);
                a.mov(4, r10, rax);
                a.alu(4, 6, r10, rdx);
                a.shift_imm(4, 5, r10, static_cast<u8>(bits - 1));
                a.alu_imm(4, 4, r10, 1);
                break;
            case Op::Sar:
                if (size == 4) a.mov(4, rdx, rax);
                else a.movsx(rdx, size, rax);
                a.mov(4, r9, rdx);
                a.shift_cl(4, 7, rdx);
                a.alu_imm(4, 4, rdx, mask_of(size));
                a.lea(4, rcx, mem(rcx, -1));
                a.shift_cl(4, 7, r9);
                a.alu_imm(4, 4, r9, 1);
                a.mov32(r10, 0);
                break;
            default: return false;
        }
        if (m) a.mov(size, mem(rsi), rdx);
        else a.mov(size, greg(in.dst.reg, size), rdx);
        a.mov(4, st_at(kLazyR), rdx);
        a.mov(1, st_at(kLazyC), r9);
        a.mov(1, st_at(kLazyOf), r10);
        a.mov_imm(1, st_at(kLazyAf), 0);
        a.mov_imm(2, st_at(kLazyOp), static_cast<u32>(LazyOp::Result) | (size << 8));
        a.bind(done);
        // After a count of zero the flags are the previous instruction's: the record stays what it was, so what the
        // translator knew of it no longer holds on both paths.
        known_ = by_cl ? Known::Unknown : Known::Result;
        known_size_ = size;
        host_flags_ = host_cf_ = false;
        finish(i, m);
        return true;
    }

    // CDQ/CWD and CWDE/CBW: no flags.
    bool convert(unsigned i, const Insn& in) {
        if (in.op == Op::Cwd) {
            if (in.size == 4) {
                a.mov(4, rax, greg(kEax, 4));
                a.shift_imm(4, 7, rax, 31);
                a.mov(4, greg(kEdx, 4), rax);
            } else {
                a.movsx(rax, 2, greg(kEax, 2));
                a.shift_imm(4, 7, rax, 15);
                a.mov(2, greg(kEdx, 2), rax);
            }
        } else {
            if (in.size == 4) {
                a.movsx(rax, 2, greg(kEax, 2));
                a.mov(4, greg(kEax, 4), rax);
            } else {
                a.movsx(rax, 1, greg(kEax, 1));
                a.mov(2, greg(kEax, 2), rax);
            }
        }
        host_flags_ = host_cf_ = false;
        finish(i, false);
        return true;
    }

    // DIV/IDIV r/m32: a zero divisor or a quotient out of range is the interpreter's (#DE), found before any effect;
    // the flags are left as they were (the interpreter only marks them undefined).
    bool divide(unsigned i, const Insn& in, Label after) {
        if (in.size != 4) return false;
        const bool m = in.dst.kind == OpndKind::Mem;
        if (!m && in.dst.kind != OpndKind::Reg) return false;
        if (m && !segment_ok(in.mem_seg, 4, false)) return false;
        const Label f = fallback(i, after);
        if (m) {
            effective_address(in);
            access_checks(in.mem_seg, 4, false, f);
            a.mov(4, rcx, mem(rsi));
        } else {
            a.mov(4, rcx, greg(in.dst.reg, 4));
        }
        a.test(4, rcx, rcx);
        a.jcc(E, f);
        a.mov(4, rax, greg(kEax, 4));
        a.mov(4, rdx, greg(kEdx, 4));
        if (in.op == Op::Div) {
            a.alu(4, 7, rdx, rcx);  // EDX >= divisor: the quotient does not fit
            a.jcc(AE, f);
            if (m) {
                a.mov(8, ctx_at(kTmp0), rcx);
                cache(i, false);
                a.mov(8, rcx, ctx_at(kTmp0));
                a.mov(4, rax, greg(kEax, 4));
                a.mov(4, rdx, greg(kEdx, 4));
            }
            a.op_rr(4, {0xF7}, 6, rcx);  // div ecx
        } else {
            // In 64 bits: EDX:EAX / sign-extended divisor; the quotient must fit 32 bits signed.
            a.shift_imm(8, 4, rdx, 32);
            a.alu(8, 1, rax, rdx);       // or rax, rdx: the 64-bit dividend
            a.op_rr(8, {0x63}, rcx, rcx);  // movsxd rcx, ecx
            a.mov64(rdx, 0x8000000000000000ull);
            a.alu(8, 7, rax, rdx);
            Label fine = a.label();
            a.jcc(NE, fine);
            a.alu_imm(8, 7, rcx, 0xFFFFFFFFu);  // INT64_MIN / -1
            a.jcc(E, f);
            a.bind(fine);
            a.mov(8, ctx_at(kTmp0), rcx);
            a.mov(8, ctx_at(kTmp1), rax);
            a.byte(0x48);
            a.byte(0x99);                  // cqo
            a.op_rr(8, {0xF7}, 7, rcx);    // idiv rcx
            a.op_rr(8, {0x63}, r9, rax);   // movsxd r9, eax
            a.alu(8, 7, r9, rax);
            a.jcc(NE, f);
            if (m) {
                cache(i, false);
                a.mov(8, rcx, ctx_at(kTmp0));
                a.mov(8, rax, ctx_at(kTmp1));
                a.byte(0x48);
                a.byte(0x99);
                a.op_rr(8, {0xF7}, 7, rcx);
            }
        }
        a.mov(4, greg(kEax, 4), rax);
        a.mov(4, greg(kEdx, 4), rdx);
        host_flags_ = host_cf_ = false;
        finish(i, m);
        return true;
    }

    // An indirect CALL or JMP through a register or memory (32-bit); CALL pushes the next EIP. The accesses in the
    // interpreter's order: the target's read, then the push.
    bool indirect(unsigned i, const Insn& in, Label after) {
        if (!last(i) || in.size != 4) return false;
        const bool m = in.dst.kind == OpndKind::Mem;
        if (!m && in.dst.kind != OpndKind::Reg) return false;
        const bool call = in.op == Op::CallRm;
        if (m && !segment_ok(in.mem_seg, 4, false)) return false;
        if (call && !segment_ok(kSs, 4, true)) return false;
        const Label f = fallback(i, after);
        if (m) {
            effective_address(in);
            access_checks(in.mem_seg, 4, false, f);
            a.mov(8, ctx_at(kTmp0), rsi);
            a.mov(8, ctx_at(kTmp1), rdi);
        }
        if (call) {
            a.mov(4, rax, greg(kEsp, 4));
            a.lea(4, rax, mem(rax, -4));
            a.mov(4, ctx_at(kNewEsp), rax);
            access_checks(kSs, 4, true, f);
        }
        if (m) {
            a.mov(8, ctx_at(kTmp2), rsi);  // the stack slot (if any)
            a.mov(8, ctx_at(kTmp3), rdi);
            a.mov(8, rsi, ctx_at(kTmp0));
            a.mov(8, rdi, ctx_at(kTmp1));
            cache(i, false);
            a.mov(4, rax, mem(rsi));
            a.mov(4, ctx_at(kCtxEa), rax);  // the target
            a.mov(8, rsi, ctx_at(kTmp2));
            a.mov(8, rdi, ctx_at(kTmp3));
        } else {
            a.mov(4, rax, greg(in.dst.reg, 4));
            a.mov(4, ctx_at(kCtxEa), rax);
        }
        if (call) {
            cache(i, true);
            a.mov_imm(4, mem(rsi), next_eip(i));
            a.mov(4, rcx, ctx_at(kNewEsp));
            a.mov(4, greg(kEsp, 4), rcx);
        }
        a.mov(4, rax, ctx_at(kCtxEa));
        a.mov(4, st_at(kEipOff), rax);
        end(i, kNoEip, 0);
        return true;
    }

    // PUSH r/m32 from memory: the read, then the push.
    bool push_mem(unsigned i, const Insn& in, Label after) {
        const Opnd o = in.src.kind != OpndKind::None ? in.src : in.dst;
        if (in.size != 4 || o.kind != OpndKind::Mem) return false;
        if (!segment_ok(in.mem_seg, 4, false) || !segment_ok(kSs, 4, true)) return false;
        const Label f = fallback(i, after);
        effective_address(in);
        access_checks(in.mem_seg, 4, false, f);
        a.mov(8, ctx_at(kTmp0), rsi);
        a.mov(8, ctx_at(kTmp1), rdi);
        a.mov(4, rax, greg(kEsp, 4));
        a.lea(4, rax, mem(rax, -4));
        a.mov(4, ctx_at(kNewEsp), rax);
        access_checks(kSs, 4, true, f);
        a.mov(8, ctx_at(kTmp2), rsi);
        a.mov(8, ctx_at(kTmp3), rdi);
        a.mov(8, rsi, ctx_at(kTmp0));
        a.mov(8, rdi, ctx_at(kTmp1));
        cache(i, false);
        a.mov(4, rax, mem(rsi));
        a.mov(4, ctx_at(kCtxEa), rax);
        a.mov(8, rsi, ctx_at(kTmp2));
        a.mov(8, rdi, ctx_at(kTmp3));
        cache(i, true);
        a.mov(4, rax, ctx_at(kCtxEa));
        a.mov(4, mem(rsi), rax);
        a.mov(4, rcx, ctx_at(kNewEsp));
        a.mov(4, greg(kEsp, 4), rcx);
        host_flags_ = host_cf_ = false;
        finish(i, true);
        return true;
    }

    // LEAVE (32-bit): ESP = EBP + 4 after the read at SS:EBP; EBP = the value read.
    bool leave_frame(unsigned i, const Insn& in, Label after) {
        if (in.size != 4 || !segment_ok(kSs, 4, false)) return false;
        const Label f = fallback(i, after);
        a.mov(4, rax, greg(kEbp, 4));
        a.mov(4, ctx_at(kNewEsp), rax);
        access_checks(kSs, 4, false, f);
        cache(i, false);
        a.mov(4, rax, mem(rsi));
        a.mov(4, rcx, ctx_at(kNewEsp));
        a.lea(4, rcx, mem(rcx, 4));
        a.mov(4, greg(kEsp, 4), rcx);
        a.mov(4, greg(kEbp, 4), rax);
        host_flags_ = host_cf_ = false;
        finish(i, true);
        return true;
    }

    // ROL/ROR by an immediate (register destination): the interpreter materialises the flags and sets CF and OF in
    // them; a count of zero changes nothing.
    bool rotate(unsigned i, const Insn& in) {
        if (in.dst.kind != OpndKind::Reg || in.src.kind != OpndKind::Imm) return false;
        const unsigned size = in.size, bits = size * 8;
        const unsigned n = in.imm & 0x1F;
        if (n == 0) {
            finish(i, false);
            return true;
        }
        a.mov(8, kArg0, kSt);
        a.call_abs(reinterpret_cast<const void*>(&h_materialise));
        load_greg(rax, in.dst.reg, size);
        a.shift_imm(size, in.op == Op::Rol ? 0 : 1, rax, static_cast<u8>(n));  // the host rotates modulo the size
        a.mov(size, greg(in.dst.reg, size), rax);
        // CF and OF from the result: ROL: CF = bit 0, OF = msb ^ CF; ROR: CF = msb, OF = msb ^ the bit below it.
        a.movzx(rcx, size == 4 ? 2 : size, rax);  // (only the low bits matter below)
        if (size == 4) a.mov(4, rcx, rax);
        a.mov(4, rdx, rcx);
        if (in.op == Op::Rol) {
            a.alu_imm(4, 4, rdx, 1);                                   // CF
            a.shift_imm(4, 5, rcx, static_cast<u8>(bits - 1));
            a.alu_imm(4, 4, rcx, 1);
            a.alu(4, 6, rcx, rdx);                                     // OF = msb ^ CF
        } else {
            a.shift_imm(4, 5, rdx, static_cast<u8>(bits - 1));
            a.alu_imm(4, 4, rdx, 1);                                   // CF = msb
            a.shift_imm(4, 5, rcx, static_cast<u8>(bits - 2));
            a.alu_imm(4, 4, rcx, 1);
            a.alu(4, 6, rcx, rdx);                                     // OF = msb ^ next
        }
        a.shift_imm(4, 4, rcx, 11);
        a.alu(4, 1, rcx, rdx);                                         // CF | OF << 11
        a.mov(4, rax, st_at(kFlagsOff));
        a.alu_imm(4, 4, rax, ~(fl::CF | fl::OF));
        a.alu(4, 1, rax, rcx);
        a.mov(4, st_at(kFlagsOff), rax);
        known_ = Known::Unknown;
        host_flags_ = host_cf_ = false;
        finish(i, false);
        return true;
    }

    // ---- SSE (the Pentium III's: no 66/F2 forms) ----
    // The host runs the guest's arithmetic under the MXCSR the interpreter's HostSse uses (the guest's rounding and
    // flush-to-zero, every exception masked, no DAZ), set once per straight run. The host's sticky flags since then
    // join the guest's MXCSR before any C++ code runs and at every exit (the guest's are sticky too; they are seen
    // only through STMXCSR/FXSAVE, which are the interpreter's). Lanes holding a NaN or a denormal, unmasked
    // exceptions, a misaligned or page-crossing operand, CR4.OSFXSR clear or CR0.EM/TS set: the interpreter.
    void sse_enter() {
        if (sse_active_) return;
        sse_load();
        sse_active_ = true;
    }
    void sse_load() {
        a.mov(4, rax, st_at(kMxcsr));
        a.alu_imm(4, 4, rax, 0xE000);
        a.alu_imm(4, 1, rax, 0x1F80);
        a.mov(4, ctx_at(kCtxSseHost), rax);
        a.ldmxcsr(ctx_at(kCtxSseHost));
    }
    // The flags since sse_enter join the guest's; C++'s MXCSR comes back. Emits code only (`sse_leave` also forgets).
    void sse_flush() {
        a.stmxcsr(ctx_at(kCtxSseHost));
        a.mov(4, rax, ctx_at(kCtxSseHost));
        a.alu_imm(4, 4, rax, 0x3F);
        a.alu(4, 1, st_at(kMxcsr), rax);
        a.ldmxcsr(ctx_at(kCtxSseDef));
    }
    void sse_leave() {
        if (!sse_active_) return;
        sse_flush();
        sse_active_ = false;
    }
    Mem xmm_at(unsigned r) { return st_at(kXmm + static_cast<i32>(16 * (r & 7))); }
    // A constant vector of four copies of `v`.
    Mem vconst(u32 v) {
        static constexpr std::array<u32, 4 * 6> kConsts = {
            0x7FFFFFFFu, 0x7FFFFFFFu, 0x7FFFFFFFu, 0x7FFFFFFFu,  // 0: abs
            0x7F800000u, 0x7F800000u, 0x7F800000u, 0x7F800000u,  // 1: infinity
            0x00800000u, 0x00800000u, 0x00800000u, 0x00800000u,  // 2: the smallest normal
            0, 0, 0, 0,                                          // 3: zero
        };
        unsigned k = v == 0x7FFFFFFFu ? 0 : v == 0x7F800000u ? 1 : v == 0x00800000u ? 2 : 3;
        a.mov64(r11, reinterpret_cast<u64>(kConsts.data() + 4 * k));
        return mem(r11);
    }
    // Jump to `fail` when a lane of xmm `x` selected by `lanes` (bit per lane) is a NaN or a denormal. Uses xmm4/5.
    void sse_lanes_ok(unsigned x, unsigned lanes, Label fail) {
        a.sse_rr(0, 0x28, 4, x);                       // movaps xmm4, x
        a.sse(0x66, 0xDB, 4, vconst(0x7FFFFFFFu));     // pand: |lane|
        a.sse(0, 0x28, 5, vconst(0x00800000u));        // xmm5 = smallest normal
        a.sse_rr(0x66, 0x66, 5, 4);                    // pcmpgtd xmm5, xmm4: |lane| < smallest normal
        a.sse(0x66, 0x76, 4, vconst(0u));              // pcmpeqd xmm4, 0... (xmm4 is clobbered: recompute)
        // xmm5 = denormal or zero; xmm4 = zero. Denormal = xmm5 & ~xmm4 (pandn: xmm4 = ~xmm4 & xmm5).
        a.sse_rr(0x66, 0xDF, 4, 5);                    // pandn xmm4, xmm5
        a.movmskps(rax, 4);
        a.alu_imm(4, 4, rax, lanes);
        a.jcc(NE, fail);
        a.sse_rr(0, 0x28, 4, x);
        a.sse(0x66, 0xDB, 4, vconst(0x7FFFFFFFu));
        a.sse(0x66, 0x66, 4, vconst(0x7F800000u));     // pcmpgtd xmm4, inf: a NaN
        a.movmskps(rax, 4);
        a.alu_imm(4, 4, rax, lanes);
        a.jcc(NE, fail);
    }
    // The guards every SSE form shares: CR4.OSFXSR, CR0.EM/TS, every MXCSR exception masked.
    void sse_preamble(Label f) {
        a.test_imm(4, st_at(kCr4Off), cr4::OSFXSR);
        a.jcc(E, f);
        a.test_imm(4, st_at(kCr0Off), cr0::EM | cr0::TS);
        a.jcc(NE, f);
        a.mov(4, rax, st_at(kMxcsr));
        a.alu_imm(4, 4, rax, 0x1F80);
        a.alu_imm(4, 7, rax, 0x1F80);
        a.jcc(NE, f);
    }
    // The memory operand for an SSE form: RSI = the host bytes, EDI = the physical address; `align` 16 checks the
    // offset as the interpreter does (#GP: its own), else the linear address must be aligned to `align` (the
    // interpreter's element accesses then stay in one cache line).
    void sse_memory(const Insn& in, unsigned size, bool write, unsigned align, bool offset_align, Label f) {
        effective_address(in);
        if (offset_align) {
            a.test_imm(4, rax, align - 1);
            a.jcc(NE, f);
        }
        access_checks(in.mem_seg, size, write, f);
        if (!offset_align && align > 1) {
            a.test_imm(4, rsi, align - 1);  // the host page is aligned: the host address carries the linear offset
            a.jcc(NE, f);
        }
    }

    bool sse(unsigned i, const Insn& in, Label after) {
        if (in.prefix_66 || in.rep == 2 || in.lock) return false;
        const bool scalar = in.rep == 1;
        const u8 op = in.opcode;
        const unsigned d = in.reg;
        enum class K { None, Load, Store, Arith, Logic, Shuf, Unpack, Cmp, MovMsk, LoadLow64, StoreLow64, LoadHigh64, StoreHigh64 } k = K::None;
        switch (op) {
            case 0x10: case 0x28: k = K::Load; break;
            case 0x11: case 0x29: k = K::Store; break;
            case 0x2B: if (!scalar && in.mem) k = K::Store; break;
            case 0x51: case 0x58: case 0x59: case 0x5C: case 0x5D: case 0x5E: case 0x5F: k = K::Arith; break;
            case 0x54: case 0x55: case 0x56: case 0x57: if (!scalar) k = K::Logic; break;
            case 0xC6: if (!scalar) k = K::Shuf; break;
            case 0x14: case 0x15: if (!scalar) k = K::Unpack; break;
            case 0xC2: if ((in.imm & 0xFF) <= 7) k = K::Cmp; break;
            case 0x50: if (!scalar && !in.mem) k = K::MovMsk; break;
            case 0x12: if (!scalar) k = K::LoadLow64; break;   // MOVLPS xmm, m64 | MOVHLPS
            case 0x13: if (!scalar && in.mem) k = K::StoreLow64; break;
            case 0x16: if (!scalar) k = K::LoadHigh64; break;  // MOVHPS xmm, m64 | MOVLHPS
            case 0x17: if (!scalar && in.mem) k = K::StoreHigh64; break;
            default: break;
        }
        if (k == K::None) return false;
        const bool store = k == K::Store || k == K::StoreLow64 || k == K::StoreHigh64;
        const unsigned size = (k == K::LoadLow64 || k == K::StoreLow64 || k == K::LoadHigh64 || k == K::StoreHigh64) ? 8 : scalar ? 4 : 16;
        if (in.mem && !segment_ok(in.mem_seg, size, store)) return false;
        const bool host_math = k == K::Arith || k == K::Cmp;
        const bool sse_before = sse_active_;
        const Label f = fallback(i, after, cw_active_, cw_active_, sse_before, host_math || sse_before);
        sse_preamble(f);
        // The second operand into xmm1 (memory: checked here, read after the cache lookup below).
        if (in.mem) {
            // Packed forms the interpreter checks for alignment (MOVAPS, arithmetic, logic, ...) fault there; MOVUPS
            // and the 64-bit moves are only native when their elements stay in one line.
            const bool checked = !scalar && size == 16 && op != 0x10 && op != 0x11;
            sse_memory(in, size, store, size == 16 ? 16 : 8, checked, f);
        }
        a.sse(0, 0x10, 0, xmm_at(d));  // xmm0 = the destination register (movups)
        if (k == K::Arith || k == K::Cmp) {
            const unsigned lanes = scalar ? 1 : 15;
            if (in.mem) a.sse(scalar ? 0xF3 : 0, 0x10, 1, mem(rsi));  // movss/movups xmm1, [rsi]
            else a.sse(0, 0x10, 1, xmm_at(in.rm));
            if (op != 0x51) sse_lanes_ok(0, lanes, f);  // SQRT reads only its source
            sse_lanes_ok(1, lanes, f);
        }
        if (in.mem) {
            cache(i, store);  // keeps RSI and RDI
        }
        if (host_math) sse_enter();
        switch (k) {
            case K::Load:
                if (scalar) {
                    if (in.mem) a.sse(0xF3, 0x10, 0, mem(rsi));            // movss: the upper lanes cleared
                    else a.sse(0xF3, 0x10, 0, xmm_at(in.rm));            // movss xmm0, m32 would clear: use the register form
                    if (!in.mem) {
                        a.sse(0, 0x10, 0, xmm_at(d));
                        a.sse(0, 0x10, 1, xmm_at(in.rm));
                        a.sse_rr(0xF3, 0x10, 0, 1);                       // movss xmm0, xmm1: the low lane only
                    }
                } else {
                    a.sse(0, 0x10, 0, in.mem ? mem(rsi) : xmm_at(in.rm));
                }
                a.sse(0, 0x11, 0, xmm_at(d));
                break;
            case K::Store:
                if (in.mem) a.sse(scalar ? 0xF3 : 0, 0x11, 0, mem(rsi));
                else if (scalar) {
                    a.sse(0, 0x10, 1, xmm_at(in.rm));
                    a.sse_rr(0xF3, 0x10, 1, 0);
                    a.sse(0, 0x11, 1, xmm_at(in.rm));
                } else {
                    a.sse(0, 0x11, 0, xmm_at(in.rm));
                }
                break;
            case K::Arith:
                a.sse_rr(scalar ? 0xF3 : 0, op, 0, 1);
                a.sse(0, 0x11, 0, xmm_at(d));
                break;
            case K::Cmp:
                a.sse_rr(scalar ? 0xF3 : 0, 0xC2, 0, 1);
                a.byte(static_cast<u8>(in.imm & 7));
                a.sse(0, 0x11, 0, xmm_at(d));
                break;
            case K::Logic: case K::Shuf: case K::Unpack:
                // The source into a register: a memory operand of these must be 16-aligned, the state's are not.
                a.sse(0, 0x10, 1, in.mem ? mem(rsi) : xmm_at(in.rm));
                a.sse_rr(0, op, 0, 1);
                if (k == K::Shuf) a.byte(static_cast<u8>(in.imm));
                a.sse(0, 0x11, 0, xmm_at(d));
                break;
            case K::MovMsk:
                a.sse(0, 0x10, 1, xmm_at(in.rm));
                a.movmskps(rax, 1);
                a.mov(4, greg(static_cast<u8>(d), 4), rax);
                break;
            case K::LoadLow64: case K::LoadHigh64:
                // Memory: the 64 bits into the low (high) half. Register: MOVHLPS (MOVLHPS) from the source.
                if (in.mem) a.sse(0, op, 0, mem(rsi));
                else {
                    a.sse(0, 0x10, 1, xmm_at(in.rm));
                    a.sse_rr(0, op, 0, 1);
                }
                a.sse(0, 0x11, 0, xmm_at(d));
                break;
            case K::StoreLow64: case K::StoreHigh64:
                a.sse(0, op, 0, mem(rsi));
                break;
            default: break;
        }
        host_flags_ = host_cf_ = false;
        finish(i, in.mem);
        return true;
    }

    // ---- MMX, as SSE2 on the low 64 bits of an xmm register (the integer lanes are the same operations; the high
    // unpacks first move the high halves down, the packs first join both operands). The interpreter's effects on
    // the x87 state: TOP = 0 and every tag valid (mmx_touch), a written register's exponent bits all ones. EMMS
    // empties the tags. CR0.EM/TS and a pending x87 exception are the interpreter's.
    Mem mm_at(unsigned r) { return st_at(kSt0 + static_cast<i32>(16 * (r & 7))); }
    void mm_touch() {
        a.alu_imm(2, 4, st_at(kFsw), 0xC7FF);
        a.mov_imm(2, st_at(kFtw), 0);
    }
    void mm_store_xmm0(unsigned r) {
        a.sse(0x66, 0xD6, 0, mm_at(r));  // movq m64, xmm0
        a.mov_imm(2, st_at(kSt0 + static_cast<i32>(16 * (r & 7)) + 8), 0xFFFF);
    }
    void mm_load(unsigned x, const Mem& m) { a.sse(0xF3, 0x7E, x, m); }  // movq xmm, m64

    bool mmx(unsigned i, const Insn& in, Label after) {
        if (!kNativeX87) return false;  // the MMX registers are the x87 slots: the same layout requirement
        const u8 op = in.opcode;
        const bool mmx_map = (op >= 0x60 && op <= 0x7F && op != 0x7A && op != 0x7B && op != 0x7C && op != 0x7D) || op == 0xC4 || op == 0xC5 ||
                             (op >= 0xD1 && op <= 0xFE && op != 0xD4 && op != 0xD6 && op != 0xE6 && op != 0xF4 && op != 0xFB);
        if (!mmx_map || in.prefix_66 || in.rep == 2 || in.lock) return false;
        const unsigned d = in.reg;
        enum class K { None, Emms, MovdIn, MovdOut, MovqIn, MovqOut, Shift, Shuffle, Binary } k = K::None;
        bool high_unpack = false, pack = false;
        switch (op) {
            case 0x77: k = K::Emms; break;
            case 0x6E: k = K::MovdIn; break;
            case 0x7E: k = K::MovdOut; break;
            case 0x6F: k = K::MovqIn; break;
            case 0x7F: k = K::MovqOut; break;
            case 0xE7: if (in.mem) k = K::MovqOut; break;
            case 0x71: case 0x72: case 0x73:
                if (!in.mem && (in.reg == 2 || in.reg == 6 || (in.reg == 4 && op != 0x73))) k = K::Shift;
                break;
            case 0x70: k = K::Shuffle; break;
            case 0x68: case 0x69: case 0x6A: high_unpack = true; k = K::Binary; break;
            case 0x63: case 0x67: case 0x6B: pack = true; k = K::Binary; break;
            case 0x60: case 0x61: case 0x62: case 0x64: case 0x65: case 0x66: case 0x74: case 0x75: case 0x76:
            case 0xD1: case 0xD2: case 0xD3: case 0xD5: case 0xD8: case 0xD9: case 0xDA: case 0xDB: case 0xDC: case 0xDD: case 0xDE: case 0xDF:
            case 0xE0: case 0xE1: case 0xE2: case 0xE3: case 0xE4: case 0xE5: case 0xE8: case 0xE9: case 0xEA: case 0xEB: case 0xEC: case 0xED:
            case 0xEE: case 0xEF: case 0xF1: case 0xF2: case 0xF3: case 0xF5: case 0xF6: case 0xF8: case 0xF9: case 0xFA: case 0xFC: case 0xFD:
            case 0xFE:
                k = K::Binary;
                break;
            default: break;
        }
        if (k == K::None) return false;
        const bool store = k == K::MovqOut || (k == K::MovdOut && in.mem);
        const unsigned size = (k == K::MovdIn || k == K::MovdOut) ? 4 : 8;
        if (in.mem && !segment_ok(in.mem_seg, size, store)) return false;
        const Label f = fallback(i, after);
        a.test_imm(4, st_at(kCr0Off), cr0::EM | cr0::TS);
        a.jcc(NE, f);
        a.test_imm(2, st_at(kFsw), 0x80);  // a pending x87 exception: #MF first
        a.jcc(NE, f);
        if (in.mem) {
            effective_address(in);
            access_checks(in.mem_seg, size, store, f);
            if (size == 8) {
                // The interpreter reads or writes the two halves: native when they share a cache line.
                a.mov(4, rax, rsi);
                a.alu_imm(4, 4, rax, 31);
                a.alu_imm(4, 7, rax, 24);
                a.jcc(A, f);
            }
            cache(i, store);
        }
        switch (k) {
            case K::Emms:
                a.mov_imm(2, st_at(kFtw), 0xFFFF);
                break;
            case K::MovdIn:
                mm_touch();
                if (in.mem) a.mov(4, rax, mem(rsi));
                else a.mov(4, rax, greg(in.rm, 4));
                a.mov(4, mm_at(d), rax);
                a.mov_imm(4, st_at(kSt0 + static_cast<i32>(16 * (d & 7)) + 4), 0);
                a.mov_imm(2, st_at(kSt0 + static_cast<i32>(16 * (d & 7)) + 8), 0xFFFF);
                break;
            case K::MovdOut:
                mm_touch();
                a.mov(4, rax, mm_at(d));
                if (in.mem) a.mov(4, mem(rsi), rax);
                else a.mov(4, greg(in.rm, 4), rax);
                break;
            case K::MovqIn:
                mm_touch();
                mm_load(0, in.mem ? mem(rsi) : mm_at(in.rm));
                mm_store_xmm0(d);
                break;
            case K::MovqOut:
                mm_touch();
                mm_load(0, mm_at(d));
                if (in.mem) a.sse(0x66, 0xD6, 0, mem(rsi));
                else mm_store_xmm0(in.rm);
                break;
            case K::Shift:
                mm_touch();
                mm_load(0, mm_at(in.rm));
                a.sse_rr(0x66, op, in.reg, 0);  // 66 0F 71/72/73 /reg ib on xmm0
                a.byte(static_cast<u8>(in.imm));
                mm_store_xmm0(in.rm);
                break;
            case K::Shuffle:
                mm_touch();
                mm_load(1, in.mem ? mem(rsi) : mm_at(in.rm));
                a.sse_rr(0xF2, 0x70, 0, 1);  // pshuflw xmm0, xmm1: the low four words as PSHUFW
                a.byte(static_cast<u8>(in.imm));
                mm_store_xmm0(d);
                break;
            case K::Binary:
                mm_touch();
                mm_load(0, mm_at(d));
                mm_load(1, in.mem ? mem(rsi) : mm_at(in.rm));
                if (high_unpack) {
                    a.sse_rr(0x66, 0x73, 2, 0);  // psrlq xmm0, 32
                    a.byte(32);
                    a.sse_rr(0x66, 0x73, 2, 1);
                    a.byte(32);
                    a.sse_rr(0x66, static_cast<u8>(op - 8), 0, 1);  // the low unpack of the high halves
                } else if (pack) {
                    a.sse_rr(0x66, 0x6C, 0, 1);  // punpcklqdq: a in the low half, b in the high
                    a.sse_rr(0x66, op, 0, 0);    // pack the eight elements: the low 64 bits are MMX's result
                } else {
                    a.sse_rr(0x66, op, 0, 1);
                }
                mm_store_xmm0(d);
                break;
            default: break;
        }
        host_flags_ = host_cf_ = false;
        finish(i, in.mem);
        return true;
    }

    // IMUL r, r/m and IMUL r, r/m, imm (16/32-bit): the product's low part, CF = OF = it overflowed the size; the
    // interpreter's record is a result kind (SF/ZF/PF from the low part, AF clear).
    bool imul(unsigned i, const Insn& in, Label after) {
        const unsigned size = in.size;
        if ((size != 2 && size != 4) || in.dst.kind != OpndKind::Reg) return false;
        const bool m = in.src.kind == OpndKind::Mem;
        if (!m && in.src.kind != OpndKind::Reg) return false;
        if (m && !segment_ok(in.mem_seg, size, false)) return false;
        if (m) {
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, size, false, f);
            cache(i, false);
            if (size == 4) a.mov(4, rcx, mem(rsi));
            else a.movsx(rcx, 2, mem(rsi));
        } else if (size == 4) {
            a.mov(4, rcx, greg(in.src.reg, 4));
        } else {
            a.movsx(rcx, 2, greg(in.src.reg, 2));
        }
        if (in.op == Op::Imul3) a.mov32(rax, size == 4 ? in.imm : static_cast<u32>(static_cast<i32>(static_cast<i16>(in.imm & 0xFFFF))));
        else if (size == 4) a.mov(4, rax, greg(in.dst.reg, 4));
        else a.movsx(rax, 2, greg(in.dst.reg, 2));
        a.imul(4, rax, rcx);
        if (size == 4) {
            a.setcc(O, rdx);
        } else {
            a.movsx(rcx, 2, rax);  // the low part sign-extended differs from the product: it overflowed
            a.alu(4, 7, rcx, rax);
            a.setcc(NE, rdx);
            a.alu_imm(4, 4, rax, 0xFFFF);
        }
        a.mov(size, greg(in.dst.reg, size), rax);
        a.mov(4, st_at(kLazyR), rax);
        a.mov(1, st_at(kLazyC), rdx);
        a.mov(1, st_at(kLazyOf), rdx);
        a.mov_imm(1, st_at(kLazyAf), 0);
        a.mov_imm(2, st_at(kLazyOp), static_cast<u32>(LazyOp::Result) | (size << 8));
        known_ = Known::Result;
        known_size_ = size;
        host_flags_ = host_cf_ = false;
        finish(i, m);
        return true;
    }

    // MUL r/m32: EDX:EAX = EAX * src; CF = OF = the high half is not zero.
    bool mul(unsigned i, const Insn& in, Label after) {
        if (in.size != 4) return false;
        const bool m = in.dst.kind == OpndKind::Mem;
        if (!m && in.dst.kind != OpndKind::Reg) return false;
        if (m && !segment_ok(in.mem_seg, 4, false)) return false;
        if (m) {
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, 4, false, f);
            cache(i, false);
            a.mov(4, rcx, mem(rsi));
        } else {
            a.mov(4, rcx, greg(in.dst.reg, 4));
        }
        a.mov(4, rax, greg(kEax, 4));
        a.unary(4, 4, rcx);  // mul ecx
        a.mov(4, greg(kEax, 4), rax);
        a.mov(4, greg(kEdx, 4), rdx);
        a.mov(4, st_at(kLazyR), rax);
        a.test(4, rdx, rdx);
        a.setcc(NE, rdx);
        a.mov(1, st_at(kLazyC), rdx);
        a.mov(1, st_at(kLazyOf), rdx);
        a.mov_imm(1, st_at(kLazyAf), 0);
        a.mov_imm(2, st_at(kLazyOp), static_cast<u32>(LazyOp::Result) | (4u << 8));
        known_ = Known::Result;
        known_size_ = 4;
        host_flags_ = host_cf_ = false;
        finish(i, m);
        return true;
    }

    // ADC/SBB: the carry in from the flags before; the record keeps it (a = destination, b = source).
    bool adc_sbb(unsigned i, const Insn& in, Label after) {
        const unsigned size = in.size;
        const bool dst_mem = in.dst.kind == OpndKind::Mem, src_mem = in.src.kind == OpndKind::Mem;
        if (!dst_mem && in.dst.kind != OpndKind::Reg) return false;
        if (!src_mem && in.src.kind != OpndKind::Reg && in.src.kind != OpndKind::Imm) return false;
        if (dst_mem && src_mem) return false;
        const bool touched = dst_mem || src_mem;
        if (touched && !segment_ok(in.mem_seg, size, dst_mem)) return false;
        cf_to_edx();
        a.mov(1, ctx_at(kIncCf), rdx);
        if (touched) {
            const Label f = fallback(i, after);
            effective_address(in);
            access_checks(in.mem_seg, size, dst_mem, f);
            cache(i, false);
            if (dst_mem) cache(i, true);
        }
        if (dst_mem) load_host(rax, size);
        else load_greg(rax, in.dst.reg, size);
        if (src_mem) load_host(rcx, size);
        else if (in.src.kind == OpndKind::Imm) a.mov32(rcx, in.imm & mask_of(size));
        else load_greg(rcx, in.src.reg, size);
        a.mov(4, st_at(kLazyA), rax);
        a.mov(4, st_at(kLazyB), rcx);
        a.movzx(rdx, 1, ctx_at(kIncCf));
        a.mov(1, st_at(kLazyC), rdx);
        a.op_rr(4, {0x0F, 0xBA}, 4, rdx);  // bt edx, 0: the carry into the host CF
        a.byte(0);
        a.alu(size, in.op == Op::Adc ? 2 : 3, rax, rcx);
        a.mov(4, st_at(kLazyR), rax);
        if (dst_mem) a.mov(size, mem(rsi), rax);
        else a.mov(size, greg(in.dst.reg, size), rax);
        a.mov_imm(2, st_at(kLazyOp), static_cast<u32>(in.op == Op::Adc ? LazyOp::Adc : LazyOp::Sbb) | (size << 8));
        known_ = Known::Unknown;
        host_flags_ = host_cf_ = !touched;
        finish(i, touched);
        return true;
    }

    // XADD r/m, r: the sum into the destination first, then the old destination into the source register.
    bool xadd(unsigned i, const Insn& in, Label after) {
        const unsigned size = in.size;
        const bool m = in.dst.kind == OpndKind::Mem;
        if ((!m && in.dst.kind != OpndKind::Reg) || in.src.kind != OpndKind::Reg || in.lock) return false;
        if (m && !segment_ok(in.mem_seg, size, true)) return false;
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
        load_greg(rcx, in.src.reg, size);
        a.mov(4, st_at(kLazyA), rax);
        a.mov(4, st_at(kLazyB), rcx);
        a.mov(4, rdx, rax);
        a.alu(size, 0, rdx, rcx);
        a.mov(4, st_at(kLazyR), rdx);
        if (m) a.mov(size, mem(rsi), rdx);
        else a.mov(size, greg(in.dst.reg, size), rdx);
        a.mov(size, greg(in.src.reg, size), rax);
        set_lazy(LazyOp::Add, size);
        known_ = Known::Add;
        known_size_ = size;
        host_flags_ = host_cf_ = false;
        finish(i, m);
        return true;
    }

    // ---- instructions ----
    bool native(unsigned i, Label after) {
        const Insn& in = b_.insns[i];
        if (in.lock && in.op != Op::Nop) return false;
        if (in.mem && in.adsize != 4) return false;
        switch (in.op) {
            case Op::Nop: case Op::NopEv: case Op::Prefetch:  // nothing at all, a memory operand included
                if (last(i)) end(i, next_eip(i), 0);
                return true;
            case Op::Mov: return mov(i, in, after);
            case Op::Movzx: case Op::Movsx: return extend(i, in, after);
            case Op::Lea: return lea(i, in);
            case Op::Add: case Op::Or: case Op::And: case Op::Sub: case Op::Xor: case Op::Cmp: case Op::Test: return alu(i, in, after);
            case Op::Inc: case Op::Dec: return incdec(i, in, after);
            case Op::Neg: case Op::Not: return negnot(i, in, after);
            case Op::Push: return push(i, in, after) || push_mem(i, in, after);
            case Op::Pop: return pop(i, in, after);
            case Op::Jcc: return jcc(i, in);
            case Op::Jmp: return jmp(i, in);
            case Op::Call: return call(i, in, after);
            case Op::Ret: return ret(i, in, after);
            case Op::Setcc: return setcc(i, in);
            case Op::Cmovcc: return cmov(i, in);
            case Op::Xchg: return xchg(i, in);
            case Op::X87: return x87(i, in, after);
            case Op::Simd: return mmx(i, in, after) || sse(i, in, after);
            case Op::Shl: case Op::Shr: case Op::Sar: return shift(i, in, after);
            case Op::Rol: case Op::Ror: return rotate(i, in);
            case Op::Imul2: case Op::Imul3: return imul(i, in, after);
            case Op::Mul: return mul(i, in, after);
            case Op::Adc: case Op::Sbb: return adc_sbb(i, in, after);
            case Op::Xadd: return xadd(i, in, after);
            case Op::Cwd: case Op::Cbw: return convert(i, in);
            case Op::Div: case Op::Idiv: return divide(i, in, after);
            case Op::CallRm: case Op::JmpRm: return indirect(i, in, after);
            case Op::Leave: return leave_frame(i, in, after);
            default: return false;
        }
    }
    // The end of a native non-branch instruction: the last one leaves the block; others check the exit flag.
    void finish(unsigned i, bool touched_memory) {
        if (last(i)) end(i, next_eip(i), 0);
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
        a.mov(1, ctx_at(kIncCf), rdx);
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
        a.movzx(rcx, 1, ctx_at(kIncCf));
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
        a.mov(4, ctx_at(kNewEsp), rax);  // the new ESP
        access_checks(kSs, 4, true, f);
        if (o.kind == OpndKind::Reg) load_greg(rax, o.reg, 4);
        else a.mov32(rax, in.imm);
        cache(i, true);
        a.mov(4, mem(rsi), rax);
        a.mov(4, rcx, ctx_at(kNewEsp));
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
        a.mov(4, ctx_at(kNewEsp), rax);
        access_checks(kSs, 4, false, f);
        cache(i, false);
        a.mov(4, rax, mem(rsi));
        a.mov(4, rcx, ctx_at(kNewEsp));
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
        const bool cw = cw_active_, sse_on = sse_active_;  // each path leaves the host x87 and SSE states itself
        end(i, next_eip(i), 0);
        a.bind(taken);
        cw_active_ = cw;
        sse_active_ = sse_on;
        end(i, static_cast<u32>(next_eip(i) + in.imm), 1);
        return true;
    }
    bool jmp(unsigned i, const Insn& in) {
        if (!last(i) || in.size != 4) return false;
        end(i, static_cast<u32>(next_eip(i) + in.imm), 0);
        return true;
    }
    bool call(unsigned i, const Insn& in, Label after) {
        if (!last(i) || in.size != 4 || !segment_ok(kSs, 4, true)) return false;
        const Label f = fallback(i, after);
        a.mov(4, rax, greg(kEsp, 4));
        a.lea(4, rax, mem(rax, -4));
        a.mov(4, ctx_at(kNewEsp), rax);
        access_checks(kSs, 4, true, f);
        cache(i, true);
        a.mov_imm(4, mem(rsi), next_eip(i));
        a.mov(4, rcx, ctx_at(kNewEsp));
        a.mov(4, greg(kEsp, 4), rcx);
        end(i, static_cast<u32>(next_eip(i) + in.imm), 0);
        return true;
    }
    bool ret(unsigned i, const Insn& in, Label after) {
        if (!last(i) || in.size != 4 || !segment_ok(kSs, 4, false)) return false;
        const Label f = fallback(i, after);
        a.mov(4, rax, greg(kEsp, 4));
        a.mov(4, ctx_at(kNewEsp), rax);
        access_checks(kSs, 4, false, f);
        cache(i, false);
        a.mov(4, rax, mem(rsi));
        a.mov(4, st_at(kEipOff), rax);
        a.mov(4, rcx, ctx_at(kNewEsp));
        a.lea(4, rcx, mem(rcx, static_cast<i32>(4 + in.imm)));
        a.mov(4, greg(kEsp, 4), rcx);
        end(i, kNoEip, 0);
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

bool Jit::compile(Block& b, Cpu& cpu, BlockEngine& engine) {
    if (!buffer_ || b.rep || b.mode != 3) return false;
    b.jit_eip = cpu.state().eip;
    for (unsigned g = 0; g < kSegCount; ++g) b.jit_seg[g] = cpu.state().seg[g];
    b.jit_dirty = cpu.memory().dirty_tracking();
    Translator t(b, cpu, engine.native_view());
    t.run();
    if (!t.a.finish()) return false;
    const std::size_t n = t.a.code.size();
    if (used_ + n > capacity_) return false;
    std::memcpy(buffer_ + used_, t.a.code.data(), n);
    b.native = buffer_ + used_;
    b.entry.body = buffer_ + used_ + t.body;
    b.entry.jit_eip = b.jit_eip;
    used_ += (n + 15) & ~std::size_t{15};
    ++compiled_;
    return true;
}

#else  // no x86-64 host: blocks stay interpreted

bool Jit::compile(Block&, Cpu&, BlockEngine&) { return false; }

#endif

}  // namespace mv::p6
