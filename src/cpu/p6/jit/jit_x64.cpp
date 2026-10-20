#include "cpu/p6/jit/jit.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
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
// The working set of a title runs to hundreds of MiB of generated code (Forza: ~140 k live blocks, ~2.5 KiB each):
// at 64 MiB the cache flushed every few seconds and every block was translated about five times. Address space,
// not memory: only the pages written are backed.
constexpr std::size_t kBufferBytes = std::size_t{512} << 20;
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
constexpr i32 kCtxBlock = static_cast<i32>(offsetof(JitCtx, block)), kCtxChainLimit = static_cast<i32>(offsetof(JitCtx, chain_limit));
constexpr i32 kCtxInsnsDone = static_cast<i32>(offsetof(JitCtx, insns_done)), kCtxProbeEip = static_cast<i32>(offsetof(JitCtx, probe_eip));
constexpr i32 kCtxStores = static_cast<i32>(offsetof(JitCtx, stores)), kCtxMispredicts = static_cast<i32>(offsetof(JitCtx, mispredicts));
constexpr i32 kCtxInlineChains = static_cast<i32>(offsetof(JitCtx, inline_chains)), kCtxInlineMispredicts = static_cast<i32>(offsetof(JitCtx, inline_mispredicts));
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
constexpr Reg kArg0 = rcx, kArg1 = rdx, kArg2 = r8, kArg3 = r9;
#else
constexpr Reg kArg0 = rdi, kArg1 = rsi, kArg2 = rdx, kArg3 = rcx;
#endif

// A chain cell's copy (see Translator::join): fixed bytes, so a link is patched at fixed offsets from its start.
//   0 lea rax, [r10 + cycles]      (imm 3: the next block's cycles; 0x7FFFFFFF unlinked: the limit check always fails)
//   7 cmp rax, [rbx + chain_limit]; 14 jae slow
//  20 cmp dword [rbx + probe_eip], target; 30 je slow
//  36 skip (2 bytes: nop, or a short jump over the translation check, or over it and every line check)
//  38 cmp dword [r13 + tlb], tag (disp 41, imm 45); 49 jne slow; 55 cmp dword [r13 + tlb + 4], page (disp 58, imm 62); 66 jne slow
//  72 skip (2 bytes: over the line checks not used)
//  74 four line checks of 17 bytes: cmp dword [r15 + l1i - l1d + set * 4], line + 1 (disp +3, imm +7); jne slow
// 142 the commit
constexpr unsigned kCellLeaImm = 3, kCellSkipT = 36, kCellTagDisp = 41, kCellTagImm = 45, kCellPageDisp = 58, kCellPageImm = 62, kCellSkipL = 72,
                   kCellLines = 74, kCellLine = 17, kCellEnd = 142;
constexpr u32 kCellUnlinked = 0x7FFFFFFFu;
constexpr i32 kCtxTargets = static_cast<i32>(offsetof(JitCtx, targets));

Mem greg(u8 r, unsigned size) {
    if (size == 1) return r < 4 ? mem(kSt, kGpr + 4 * r) : mem(kSt, kGpr + 4 * (r - 4) + 1);
    return mem(kSt, kGpr + 4 * (r & 7));
}
Mem st_at(i32 off) { return mem(kSt, off); }
Mem ctx_at(i32 off) { return mem(kCtx, off); }

// ---- helpers the generated code calls ----

// Execute instruction `i` with the interpreter, as the block interpreter does: the clock at its start, its
// penalties added, the run left when it faulted, reached a device, rewrote its own page or passed the limit.
u32 h_exec(JitCtx* c, u32 i, Block* blk) {
    Block& b = *blk;
    c->block = blk;  // the block running (a kill of it is seen: BlockEngine::kill_block)
    Cpu& cpu = *c->cpu;
    Memory& mem = cpu.memory();
    const u64 walks = mem.page_walks();
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
    // A walk may have replaced a translation a chain cell relies on without checking (its own page's): the engine's
    // chain checks the next exit.
    if (mem.page_walks() != walks) std::atomic_ref<u64>(c->chain_limit).store(0, std::memory_order_relaxed);
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
void h_cache(JitCtx* c, u32 phys, u32 arg, const Block* b) {
    const unsigned i = arg >> 8;
    const u64 p = c->cpu->cache_timing().data_access(phys, (arg & 1) != 0);
    c->base += p;
    c->penalties += p;
    // The limit at the next boundary: leave there. Past the block's end only: a later boundary may reach it, which
    // the interpreter finds instruction by instruction — hand the rest of the block over to it.
    if (c->base + b->after[i] >= c->limit) c->exit = 1;
    else if (c->base + b->cycles >= c->limit) c->exit = 2;
}

// The end of a block whose last instruction left normally: the engine costs the exit and returns where the run goes
// on (the next block's native code, or null: back to the dispatcher).
void* h_chain(JitCtx* c, Block::Link* slot) { return c->engine->chain(*c, slot); }

void h_materialise(CpuState* s) { s->materialise(); }
u32 h_eflags(const CpuState* s) { return s->eflags(); }  // computed, the lazy record left as it is (LAHF)
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

    // MV_JIT_MAP's profile marks: where each part of the code starts (offset, kind), kept only with the map on.
    static bool marking() {
        static const bool on = [] {
            const char* p = std::getenv("MV_JIT_MAP");
            return p && *p;
        }();
        return on;
    }
    std::vector<std::pair<u32, std::string>> marks;
    // Each exit slot's chain cell copies (offsets of their starts and final jumps; 0: none).
    std::array<std::array<u32, 2>, 2> cell{}, cell_jump{};
    std::string op_;
    void mark(const char* kind, bool with_op = false) {
        if (marking()) marks.emplace_back(static_cast<u32>(a.size()), with_op ? std::string(kind) + op_ : std::string(kind));
    }

    void run() {
        epilogue_ = a.label();
        mark("prologue");
        prologue();
        // Inside a run the host's x87 control word and MXCSR are the guest's (loaded here once, kept across chains):
        // every block's body starts that way, and only the paths into C++ (helpers, the engine's chain, the epilogue)
        // put C++'s back, flushing the guest's sticky flags first.
        if (kNativeX87) guest_cw();
        sse_load();
        body = a.size();
        cw_active_ = kNativeX87;
        sse_active_ = true;
        for (unsigned i = 0; i < b_.n; ++i) {
            Label after = a.label();
            if (marking()) op_ = op_name(b_.insns[i].op);
            mark("insn:", true);
            if (!native(i, after)) inline_helper(i);
            a.bind(after);
        }
        // The last instruction always leaves (native or helper); nothing falls through to here.
        a.bind(epilogue_);
        mark("epilogue");
        epilogue();
        mark("stubs");
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
    Label ways_;               // the current exit's way search (see end)
    bool ways_used_{false};
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
        a.mov64(kCodeMap, reinterpret_cast<u64>(ev_.code_granules));
    }
    void epilogue() {
        // The block the run ended in (chain cells do not keep the context's block current).
        a.mov64(rcx, reinterpret_cast<u64>(&b_));
        a.mov(8, ctx_at(kCtxBlock), rcx);
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
        a.mov64(kArg2, reinterpret_cast<u64>(&b_));
        a.call_abs(reinterpret_cast<const void*>(&h_exec));
    }
    void inline_helper(unsigned i) {
        mark("helper:", true);
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
    // The block's normal end at EIP `eip` (kNoEip: already stored, a return): a chain cell for the exit (see join),
    // and where it does not take the exit, the engine, which costs it through link slot `slot` and may chain into the
    // next block's native code.
    static constexpr u64 kNoEip = ~u64{0};
    void end(unsigned i, u64 eip, unsigned slot) {
        mark("chain");
        if (kNativeX87) x87_enter();  // the next block's body starts with the guest's words (after a helper: again)
        sse_enter();
        Label slow = a.label();
        ways_ = a.label();
        ways_used_ = false;
        const BranchKind kind = b_.branch;
        const bool has_cell = eip != kNoEip && (kind == BranchKind::None || kind == BranchKind::Conditional || kind == BranchKind::Direct ||
                                                kind == BranchKind::Call);
        // Taken as the engine counts it: a jump to the next instruction is not taken.
        if (has_cell) chain_cell(static_cast<u32>(eip), slot, static_cast<u32>(eip) != next_eip(b_.n - 1u), slow);
        else if (eip == kNoEip && b_.jit_key && (kind == BranchKind::Return || kind == BranchKind::Indirect || kind == BranchKind::CallIndirect))
            chain_target(slow);
        if (ways_used_) {
            // A code line that is in its set but not the set's most recent (ESI = the line plus one, RDX = where to go
            // on): the model's own update, then back; a miss is the engine's.
            a.bind(ways_);
            mark("chain-ways");
            fetch_line(slow);
            a.jmp(rdx);
        }
        a.bind(slow);
        mark("chain-slow");
        if (eip != kNoEip) a.mov_imm(4, st_at(kEipOff), static_cast<u32>(eip));
        // The engine reads the whole state (the spin probe compares it): the guest's flags in, C++'s words back.
        sse_flush();
        if (kNativeX87) default_cw();
        a.mov(8, kArg0, kCtx);
        a.mov64(kArg1, reinterpret_cast<u64>(&b_.links[slot]));
        a.call_abs(reinterpret_cast<const void*>(&h_chain));
        Label back = a.label();
        a.test(8, rax, rax);
        a.jcc(E, back);
        a.mov(8, r11, rax);
        if (kNativeX87) guest_cw();
        sse_load();
        a.jmp(r11);
        a.bind(back);
        a.mov32(rax, (i << 2) | kJitEnd);
        a.jmp(epilogue_);
        cw_active_ = false;  // (the epilogue's state; the caller restores what a second exit starts from)
        sse_active_ = false;
    }
    // A return's or an indirect jump's chain (the target in the state's EIP), when the exit was predicted (a return:
    // the return stack's top is the target; an indirect jump: its last target) so its cost is the schedule alone: the
    // target cache's entry for the EIP, made for this block's assumptions, with the checks of a chain cell (the limit,
    // the spin probe, the page's translation, the code lines), then the commit (the return stack popped, or pushed for
    // an indirect call). Anything else is the engine's.
    void chain_target(Label slow) {
        mark("chain-target");
        const BranchKind kind = b_.branch;
        a.mov(4, rax, st_at(kEipOff));
        if (kind == BranchKind::Return) {
            a.mov64(rdx, reinterpret_cast<u64>(ev_.rsb_top));
            a.mov(4, rcx, mem(rdx));
            a.lea(4, rcx, mem(rcx, 15));
            a.alu_imm(4, 4, rcx, 15);
            a.mov64(rsi, reinterpret_cast<u64>(ev_.rsb));
            a.alu(4, 7, rax, mem(rsi, rcx, 2, 0));
            a.jcc(NE, slow);
        } else {
            a.mov64(rdx, reinterpret_cast<u64>(&b_.exit_target));
            a.alu(4, 7, rax, mem(rdx));
            a.jcc(NE, slow);
            a.mov64(rdx, reinterpret_cast<u64>(&b_.exit_seen));
            a.alu_imm(1, 7, mem(rdx), 0);
            a.jcc(E, slow);
        }
        a.mov(4, r8, rax);
        a.shift_imm(4, 5, r8, 12);
        a.alu(4, 6, r8, rax);
        a.alu_imm(4, 4, r8, JitCtx::kTargets - 1);
        a.shift_imm(4, 4, r8, 6);
        a.lea(8, r9, mem(kCtx, r8, 0, kCtxTargets));
        a.alu(4, 7, rax, mem(r9, 0));
        a.jcc(NE, slow);
        a.alu_imm(4, 7, mem(r9, 4), b_.jit_key);
        a.jcc(NE, slow);
        a.mov(8, r10, ctx_at(kCtxBase));
        a.alu_imm(8, 0, r10, b_.cycles);
        a.mov(4, r11, mem(r9, 8));
        a.alu(8, 0, r11, r10);
        a.alu(8, 7, r11, ctx_at(kCtxChainLimit));
        a.jcc(AE, slow);
        a.alu(4, 7, rax, ctx_at(kCtxProbeEip));
        a.jcc(E, slow);
        a.mov(4, r11, mem(r9, 12));
        a.mov(4, rdi, mem(r9, 16));
        a.alu(4, 7, rdi, mem(kTlb, r11, 0, 0));
        a.jcc(NE, slow);
        a.mov(4, rdi, mem(r9, 20));
        a.alu(4, 7, rdi, mem(kTlb, r11, 0, 4));
        a.jcc(NE, slow);
        Label one = a.label();
        a.mov(8, r11, mem(r9, 32));
        a.mov(4, rdi, mem(r9, 24));
        a.alu(4, 7, rdi, mem(kMru, r11, 0, 0));
        a.jcc(NE, slow);
        a.mov(4, rdi, mem(r9, 28));
        a.test(4, rdi, rdi);
        a.jcc(E, one);
        a.mov(8, r11, mem(r9, 40));
        a.alu(4, 7, rdi, mem(kMru, r11, 0, 0));
        a.jcc(NE, slow);
        a.bind(one);
        // ---- committed from here ----
        if (kind == BranchKind::Return) {
            a.mov(4, mem(rdx), rcx);
        } else if (kind == BranchKind::CallIndirect) {
            a.mov64(rdx, reinterpret_cast<u64>(ev_.rsb_top));
            a.mov(4, rcx, mem(rdx));
            a.mov64(rsi, reinterpret_cast<u64>(ev_.rsb));
            a.mov_imm(4, mem(rsi, rcx, 2, 0), next_eip(b_.n - 1u));
            a.lea(4, rcx, mem(rcx, 1));
            a.alu_imm(4, 4, rcx, 15);
            a.mov(4, mem(rdx), rcx);
        }
        a.mov(8, ctx_at(kCtxBase), r10);
        a.alu_imm(8, 0, ctx_at(kCtxInsnsDone), b_.n);
        a.alu_imm(4, 0, ctx_at(kCtxInlineChains), 1);
        a.mov(8, rax, mem(r9, 48));
        a.jmp(rax);
    }
    // One code line (ESI = the line plus one) in L1I: the most recent of its set, or found in a way of it and made so
    // (the age from the clock, as Timing::code_fetch does); else `slow`. Clobbers RAX RDI R11; keeps RCX RDX R8-R10.
    void fetch_line(Label slow) {
        Timing& t = cpu_.cache_timing();
        const unsigned isets = t.l1i_sets();
        Label ok = a.label();
        a.lea(4, rdi, mem(rsi, -1));
        a.alu_imm(4, 4, rdi, isets - 1);
        a.mov64(rax, reinterpret_cast<u64>(t.l1i_mru()));
        a.alu(4, 7, rsi, mem(rax, rdi, 2, 0));
        a.jcc(E, ok);
        if (t.l1i_ways() != 4 || t.ideal()) {
            a.jmp(slow);
            a.bind(ok);
            return;
        }
        Label found = a.label();
        a.mov64(r11, reinterpret_cast<u64>(t.l1i_tags()));
        a.shift_imm(4, 4, rdi, 2);  // the set's first way
        Label w[4] = {a.label(), a.label(), a.label(), a.label()};
        for (unsigned k = 0; k < 4; ++k) {
            a.alu(4, 7, rsi, mem(r11, rdi, 2, static_cast<i32>(4 * k)));
            a.jcc(E, w[k]);
        }
        a.jmp(slow);
        for (unsigned k = 3; k >= 1; --k) {
            a.bind(w[k]);
            a.lea(4, rdi, mem(rdi, static_cast<i32>(k)));
            a.jmp(found);
        }
        a.bind(w[0]);
        a.bind(found);
        a.mov64(r11, reinterpret_cast<u64>(t.lru_clock_word()));
        a.mov(8, rax, mem(r11));
        a.lea(8, rax, mem(rax, 1));
        a.mov(8, mem(r11), rax);
        a.mov64(r11, reinterpret_cast<u64>(t.l1i_last_use()));
        a.mov(8, mem(r11, rdi, 3, 0), rax);
        a.shift_imm(4, 5, rdi, 2);  // the set again
        a.mov64(rax, reinterpret_cast<u64>(t.l1i_mru_mutable()));
        a.mov(4, mem(rax, rdi, 2, 0), rsi);
        a.bind(ok);
    }
    // An exit's chain cell: the exit's prediction worked out (a conditional exit: the fast copy when its two-bit
    // predictor is saturated the way it went, which changes nothing; else the copy that steps it), R10 = the cycle the
    // block ends at, then a copy of the checks and the commit (join), patched for the block the exit leads to.
    void chain_cell(u32 target, unsigned slot, bool taken, Label slow) {
        const BranchKind kind = b_.branch;
        if (kind == BranchKind::Conditional) {
            Label step = a.label();
            a.mov64(rax, reinterpret_cast<u64>(&b_.exit_counter));
            a.alu_imm(1, 7, mem(rax), taken ? 3 : 0);
            a.jcc(NE, step);
            a.mov(8, r10, ctx_at(kCtxBase));
            a.alu_imm(8, 0, r10, b_.cycles);
            join(target, slot, 0, slow);
            a.bind(step);
            mark("chain-step");
            // R9 = the predictor's new state, R8 = 1 on a mispredict (RAX = its address).
            a.movzx(r9, 1, mem(rax));
            a.mov32(r8, 0);
            a.alu_imm(4, 7, r9, 2);
            a.setcc(taken ? B : AE, r8);  // taken: predicted not taken below 2; not taken: predicted taken from 2
            if (taken) {
                a.lea(4, rcx, mem(r9, 1));
                a.mov32(rsi, 3);
                a.alu(4, 7, rcx, rsi);
                a.cmov(4, A, rcx, rsi);
                a.mov(4, r9, rcx);
            } else {
                a.lea(4, rcx, mem(r9, -1));
                a.test(4, r9, r9);
                a.cmov(4, NE, r9, rcx);
            }
            static_assert(tp::kBlockMispredict == 12);
            a.mov(8, r10, ctx_at(kCtxBase));
            a.lea(8, rcx, mem(r8, r8, 1, 0));
            a.lea(8, r10, mem(r10, rcx, 2, static_cast<i32>(b_.cycles)));  // + 12 on a mispredict
            join(target, slot, 1, slow);
        } else {
            a.mov(8, r10, ctx_at(kCtxBase));
            a.alu_imm(8, 0, r10, b_.cycles);
            join(target, slot, 0, slow);
        }
    }
    // One copy of a chain cell's checks and commit (R10 = the cycle this block ends at), in the fixed layout patch_link
    // fills (kCell*): the next block must end before the chain limit (which also holds the budget, a device's write
    // into code, a helper's page walk), not be the spin probe's EIP, have its page's translation in the TLB (unless it
    // is this block's page: nothing replaced that translation since this block's own was checked — native code never
    // walks) and its code lines the most recent of their sets in L1I (unless they are this block's: this block's fetch
    // made them so and only fetches touch L1I), so its fetch changes no cache state. Then the run moves on and the next
    // block's code is entered. Unlinked, the first check always fails: `slow`, where the engine does it all.
    void join(u32 target, unsigned slot, unsigned copy, Label slow) {
        mark(copy ? "chain-cell-step" : "chain-cell");
        const std::size_t at = a.size();
        cell[slot][copy] = static_cast<u32>(at);
        const u8 lea[] = {0x49, 0x8D, 0x82};  // lea rax, [r10 + imm32]
        for (u8 v : lea) a.byte(v);
        a.dword(kCellUnlinked);
        const u8 cmp_limit[] = {0x48, 0x3B, 0x83};  // cmp rax, [rbx + disp32]
        for (u8 v : cmp_limit) a.byte(v);
        a.dword(static_cast<u32>(kCtxChainLimit));
        a.jcc(AE, slow);
        a.byte(0x81);  // cmp dword [rbx + disp32], imm32
        a.byte(0xBB);
        a.dword(static_cast<u32>(kCtxProbeEip));
        a.dword(target);
        a.jcc(E, slow);
        a.byte(0x66);  // skip (patched)
        a.byte(0x90);
        for (int k = 0; k < 2; ++k) {
            a.byte(0x41);  // cmp dword [r13 + disp32], imm32
            a.byte(0x81);
            a.byte(0xBD);
            a.dword(0);
            a.dword(0);
            a.jcc(NE, slow);
        }
        a.byte(0x66);  // skip (patched)
        a.byte(0x90);
        std::array<Label, 4> line_miss{}, line_back{};
        for (unsigned k = 0; k < 4; ++k) {
            line_miss[k] = a.label();
            line_back[k] = a.label();
            a.byte(0x41);  // cmp dword [r15 + disp32], imm32
            a.byte(0x81);
            a.byte(0xBF);
            a.dword(0);
            a.dword(0);
            a.jcc(NE, line_miss[k]);
            a.bind(line_back[k]);
        }
        if (a.size() - at != kCellEnd) std::abort();  // the layout patch_link relies on
        // ---- committed from here ----
        a.mov(8, ctx_at(kCtxBase), r10);
        a.alu_imm(8, 0, ctx_at(kCtxInsnsDone), b_.n);
        a.alu_imm(4, 0, ctx_at(kCtxInlineChains), 1);
        if (copy) {
            a.mov64(rax, reinterpret_cast<u64>(&b_.exit_counter));
            a.mov(1, mem(rax), r9);
            a.alu(4, 0, ctx_at(kCtxMispredicts), r8);
            a.alu(4, 0, ctx_at(kCtxInlineMispredicts), r8);
        }
        if (b_.branch == BranchKind::Call) {
            a.mov64(rax, reinterpret_cast<u64>(ev_.rsb_top));
            a.mov(4, rsi, mem(rax));
            a.mov64(rdi, reinterpret_cast<u64>(ev_.rsb));
            a.mov_imm(4, mem(rdi, rsi, 2, 0), next_eip(b_.n - 1u));
            a.lea(4, rsi, mem(rsi, 1));
            a.alu_imm(4, 4, rsi, 15);
            a.mov(4, mem(rax), rsi);
        }
        cell_jump[slot][copy] = static_cast<u32>(a.size());
        a.jmp(slow);  // patched: the next block's body
        // Each line check's way search (out of the way): ESI = the check's own immediate (10 bytes before where it goes
        // on), RDX = where it goes on.
        for (unsigned k = 0; k < 4; ++k) {
            a.bind(line_miss[k]);
            a.lea_rip(rdx, line_back[k]);
            a.mov(4, rsi, mem(rdx, -10));
            a.jmp(ways_);
        }
        ways_used_ = true;
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
        mark("access");
        const Segment& s = st_.seg[seg];
        if (s.limit != 0xFFFFFFFFu) {
            a.alu_imm(4, 7, rax, s.limit - (size - 1));
            a.jcc(A, fail);
        }
        if (s.base) a.lea(4, rax, mem(rax, static_cast<i32>(s.base)));
        // The page of the access's last byte against the tag of the entry its first byte indexes: a page crossing
        // (or a wrap past 4 GiB) cannot match.
        a.lea(4, rdx, mem(rax, static_cast<i32>(size - 1)));
        a.shift_imm(4, 5, rdx, 12);
        a.mov(4, rcx, rax);
        a.shift_imm(4, 5, rcx, 12);
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
        a.mov(4, rdi, mem(kTlb, rcx, 0, 4));  // physical page
        a.mov(4, rdx, rax);
        a.alu_imm(4, 4, rdx, 0xFFF);
        a.alu(4, 1, rdi, rdx);  // or: the physical address
        a.alu(8, 0, rsi, rdx);  // the host byte
        if (write) {
            // Never near translated code's bytes (the interpreter drops the blocks the write overlaps).
            a.mov(8, rdx, rsi);
            a.alu(8, 5, rdx, kRam);
            a.shift_imm(8, 5, rdx, kCodeGranuleBits);
            a.alu_imm(1, 7, mem(kCodeMap, rdx, 0, 0), 0);
            a.jcc(NE, fail);
            a.mov_imm(4, ctx_at(kCtxStores), 1);  // the spin skip's evidence (whether any store ran: a plain store, no read)
            // Write tracking: a watched page's first write is the interpreter's (it reports and unwatches the page);
            // with every page logged, the page is marked here (twice is the same if a later check fails).
            Memory& m = cpu_.memory();
            if (m.dirty_tracking()) a.shift_imm(8, 5, rdx, 12 - kCodeGranuleBits);  // the RAM page
            if (m.dirty_tracking() && m.selective()) {
                a.mov64(r11, reinterpret_cast<u64>(m.watched_map()));
                a.bt(8, mem(r11), rdx);
                a.jcc(B, fail);
            } else if (m.dirty_tracking()) {
                a.mov64(r11, reinterpret_cast<u64>(m.dirty_map()));
                a.op_rm(8, {0x0F, 0xAB}, rdx, mem(r11));  // bts
            }
            log_store(size);
        }
        mark("body:", true);
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
    // An SSE form holds its operands in xmm0/xmm1 across its lookup (its checks come before any effect): the call to
    // the model keeps them (C++ may use any vector register).
    bool keep_xmm_{false};
    void cache(unsigned i, bool store) {
        mark("cache");
        Timing& t = cpu_.cache_timing();
        const unsigned sets = t.l1d_sets();
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
        if (!every_access && t.l1d_ways() == 4 && !t.ideal()) {
            // A hit on another way of the set: the model's own update (Cache::access_ways) — the way's age from the
            // clock (which data_access advances first), the set's most recent line. No penalty, so nothing about the
            // limit changes either. ECX = the line plus one, EDX = the set.
            Label miss = a.label(), found = a.label();
            a.mov64(r8, reinterpret_cast<u64>(t.l1d_tags()));
            a.mov(4, r9, rdx);
            a.shift_imm(4, 4, r9, 2);  // the set's first way
            for (unsigned w = 0; w < 4; ++w) {
                Label next = a.label();
                a.alu(4, 7, rcx, mem(r8, r9, 2, static_cast<i32>(4 * w)));
                a.jcc(NE, next);
                if (w) a.lea(4, r9, mem(r9, static_cast<i32>(w)));
                a.jmp(found);
                a.bind(next);
            }
            a.jmp(miss);
            a.bind(found);
            a.mov64(r8, reinterpret_cast<u64>(t.lru_clock_word()));
            a.mov(8, r10, mem(r8));
            a.lea(8, r10, mem(r10, 1));
            a.mov(8, mem(r8), r10);
            a.mov64(r8, reinterpret_cast<u64>(t.l1d_last_use()));
            a.mov(8, mem(r8, r9, 3, 0), r10);
            a.mov(4, mem(kMru, rdx, 2, 0), rcx);
            a.jmp(hit);
            a.bind(miss);
        }
        if (keep_xmm_) {
            a.sse(0, 0x11, 0, ctx_at(kTmp0));  // movups [ctx+tmp0], xmm0 (tmp0-tmp3: 32 bytes)
            a.sse(0, 0x11, 1, ctx_at(kTmp2));
        }
        a.mov(8, ctx_at(kSaveRsi), rsi);
        a.mov(8, ctx_at(kSaveRax), rax);
        a.mov(8, ctx_at(kSaveRdi), rdi);
        a.mov(4, kArg1, rdi);
        a.mov(8, kArg0, kCtx);
        a.mov32(kArg2, (i << 8) | (store ? 1u : 0u));
        a.mov64(kArg3, reinterpret_cast<u64>(&b_));
        a.call_abs(reinterpret_cast<const void*>(&h_cache));
        a.mov(8, rsi, ctx_at(kSaveRsi));
        a.mov(8, rax, ctx_at(kSaveRax));
        a.mov(8, rdi, ctx_at(kSaveRdi));
        if (keep_xmm_) {
            a.sse(0, 0x10, 0, ctx_at(kTmp0));
            a.sse(0, 0x10, 1, ctx_at(kTmp2));
        }
        a.bind(hit);
        mark("body:", true);
    }
    // The interpreter's element accesses (4 bytes each, in address order) each looked up at its first byte, as
    // Memory::note does: any alignment is exact (a second look at a set's most recent line changes nothing). EDI kept.
    void cache_elems(unsigned i, bool store, unsigned elems) {
        cache(i, store);
        for (unsigned e = 1; e < elems; ++e) {
            a.lea(4, rdi, mem(rdi, 4));
            cache(i, store);
        }
        if (elems > 1) a.lea(4, rdi, mem(rdi, -4 * static_cast<i32>(elems - 1)));
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

    enum class X87Form { None, LoadM32, StoreM32, LoadM64, StoreM64, ArithM32, CompareM32, ArithD8, ArithDcDe, LoadReg, StoreReg, Exchange, Sign, LoadConst, StoreSw,
                         CompareReg, CompareFlags, LoadInt, StoreInt, Sqrt };

    // FLDCW and FNSTCW m16. FLDCW waits (a pending exception is the interpreter's) and keeps its pointers; a word that
    // unmasks an exception is the interpreter's too (it may arm one). FNSTCW is a control instruction (no pointers).
    bool control_word(unsigned i, const Insn& in, Label after) {
        const bool load = in.reg == 5;
        if (!segment_ok(in.mem_seg, 2, !load)) return false;
        const bool cw = cw_active_;
        const Label f = fallback(i, after, cw, cw);
        x87_preamble(f, !load);
        effective_address(in);
        a.mov(4, ctx_at(kCtxEa), rax);
        access_checks(in.mem_seg, 2, !load, f);
        if (load) {
            a.movzx(rcx, 2, mem(rsi));
            a.alu_imm(4, 4, rcx, 0x3F);
            a.alu_imm(4, 7, rcx, 0x3F);
            a.jcc(NE, f);
            cache(i, false);
            a.movzx(rcx, 2, mem(rsi));
            a.alu_imm(4, 4, rcx, 0x1F7F);
            a.alu_imm(4, 1, rcx, 0x40);
            a.mov(2, st_at(kFcw), rcx);
            a.alu_imm(2, 4, st_at(kFsw), 0x7F7F);  // ES and B clear: no pending exception is unmasked
            x87_pointers(i, in);
            if (cw_active_) guest_cw();  // the host runs under the new word
        } else {
            cache(i, true);
            a.movzx(rax, 2, st_at(kFcw));
            a.mov(2, mem(rsi), rax);
        }
        host_flags_ = host_cf_ = false;
        finish(i, true);
        return true;
    }
    // FWAIT: #NM with CR0.MP and TS, #MF for a pending exception with CR0.NE: both the interpreter's (as any pending
    // exception); otherwise nothing.
    bool fwait(unsigned i, Label after) {
        const Label f = fallback(i, after);
        a.test_imm(2, st_at(kFsw), 0x80);
        a.jcc(NE, f);
        a.mov(4, rax, st_at(kCr0Off));
        a.alu_imm(4, 4, rax, cr0::MP | cr0::TS);
        a.alu_imm(4, 7, rax, cr0::MP | cr0::TS);
        a.jcc(E, f);
        host_flags_ = host_cf_ = false;
        finish(i, false);
        return true;
    }

    bool x87(unsigned i, const Insn& in, Label after) {
        if (!kNativeX87) return false;
        const u8 esc = in.opcode;
        if (in.mem && esc == 0xD9 && (in.reg == 5 || in.reg == 7)) return control_word(i, in, after);
        X87Form k = X87Form::None;
        if (in.mem) {
            if (esc == 0xD9 && in.reg == 0) k = X87Form::LoadM32;
            else if (esc == 0xDD && in.reg == 0) k = X87Form::LoadM64;
            else if (esc == 0xDD && (in.reg == 2 || in.reg == 3)) k = X87Form::StoreM64;
            else if (esc == 0xD9 && (in.reg == 2 || in.reg == 3)) k = X87Form::StoreM32;
            else if ((esc == 0xD8 || esc == 0xDC) && (in.reg == 2 || in.reg == 3)) k = X87Form::CompareM32;  // m32fp / m64fp
            else if (esc == 0xD8 || esc == 0xDC) k = X87Form::ArithM32;
            else if ((esc == 0xDB || esc == 0xDF) && in.reg == 0) k = X87Form::LoadInt;    // FILD m32 / m16
            else if (esc == 0xDF && in.reg == 5) k = X87Form::LoadInt;                     // FILD m64
            else if ((esc == 0xDB || esc == 0xDF) && (in.reg == 2 || in.reg == 3)) k = X87Form::StoreInt;  // FIST(P) m32 / m16
            else if (esc == 0xDF && in.reg == 7) k = X87Form::StoreInt;                    // FISTP m64
        } else {
            if (esc == 0xD8 && (in.reg == 2 || in.reg == 3)) k = X87Form::CompareReg;      // FCOM(P) ST(i)
            else if (esc == 0xDA && in.modrm == 0xE9) k = X87Form::CompareReg;            // FUCOMPP (ST(1), popped twice)
            else if ((esc == 0xDB || esc == 0xDF) && (in.reg == 5 || in.reg == 6)) k = X87Form::CompareFlags;  // F(U)COMI(P) ST(i)
            else if (esc == 0xD8) k = X87Form::ArithD8;
            else if ((esc == 0xDC || esc == 0xDE) && in.reg != 2 && in.reg != 3) k = X87Form::ArithDcDe;
            else if (esc == 0xD9 && in.modrm >= 0xC0 && in.modrm <= 0xC7) k = X87Form::LoadReg;
            else if (esc == 0xD9 && in.modrm >= 0xC8 && in.modrm <= 0xCF) k = X87Form::Exchange;
            else if (esc == 0xD9 && (in.modrm == 0xE0 || in.modrm == 0xE1)) k = X87Form::Sign;
            else if (esc == 0xD9 && (in.modrm == 0xFA || in.modrm == 0xFC)) k = X87Form::Sqrt;  // FSQRT, FRNDINT
            else if (esc == 0xD9 && (in.modrm == 0xE8 || in.modrm == 0xEE)) k = X87Form::LoadConst;
            else if (esc == 0xDD && (in.reg == 2 || in.reg == 3)) k = X87Form::StoreReg;
            else if (esc == 0xDF && in.modrm == 0xE0) k = X87Form::StoreSw;
        }
        if (k == X87Form::None) return false;
        const bool write = k == X87Form::StoreM32 || k == X87Form::StoreM64 || k == X87Form::StoreInt;
        const bool m64 = k == X87Form::LoadM64 || k == X87Form::StoreM64 || ((k == X87Form::ArithM32 || k == X87Form::CompareM32) && esc == 0xDC);
        const bool integer = k == X87Form::LoadInt || k == X87Form::StoreInt;
        // An integer's size: m16 under DF /0 /2 /3, m64 under DF /5 /7, m32 under DB.
        const unsigned msize = integer ? (esc == 0xDB ? 4u : (in.reg == 5 || in.reg == 7) ? 8u : 2u) : m64 ? 8 : 4;
        if (in.mem && !segment_ok(in.mem_seg, msize, write)) return false;
        const bool host = k == X87Form::StoreM32 || k == X87Form::LoadM64 || k == X87Form::StoreM64 || k == X87Form::ArithM32 || k == X87Form::CompareM32 || k == X87Form::ArithD8 || k == X87Form::ArithDcDe ||
                          k == X87Form::LoadM32 || k == X87Form::StoreInt || k == X87Form::Sqrt;
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
            if (m64 && !write) x87_m64_ok(f);
            else if (!write && !integer) x87_m32_ok(f);
        }
        // Every check before any effect: the stack, the operands' classes.
        x87_top();
        const unsigned ri = in.rm;
        switch (k) {
            case X87Form::LoadM32: case X87Form::LoadM64: case X87Form::LoadConst: case X87Form::LoadInt:
                x87_phys(r9, 7);
                x87_tag_test(r9, true, f);
                break;
            case X87Form::StoreM32: case X87Form::StoreM64:
                x87_tag_test(r8, false, f);
                break;
            case X87Form::ArithM32: case X87Form::CompareM32: case X87Form::Sign: case X87Form::StoreReg: case X87Form::StoreInt: case X87Form::Sqrt:
                x87_tag_test(r8, false, f);
                x87_addr(r10, r8);
                x87_operand_ok(r10, f);
                if (k == X87Form::Sqrt && in.modrm == 0xFA) {
                    a.test_imm(1, mem(r10, 9), 0x80);  // a negative operand (or -0) is the interpreter's
                    a.jcc(NE, f);
                }
                break;
            case X87Form::ArithD8: case X87Form::ArithDcDe: case X87Form::Exchange: case X87Form::CompareReg: case X87Form::CompareFlags:
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
            cache_elems(i, write, msize > 4 ? 2 : 1);  // the interpreter's accesses: one, or the two halves (clobbers R8-R11)
            x87_top();
        }
        x87_pointers(i, in);
        if (k == X87Form::CompareFlags) {
            // The interpreter materialises the lazy flags (OF SF AF stay theirs) before it sets ZF PF CF. A call: the
            // host x87 stack is empty here, as the ABI wants; the registers the checks left are computed again.
            a.mov(8, kArg0, kSt);
            a.call_abs(reinterpret_cast<const void*>(&h_materialise));
            x87_top();
            x87_phys(r9, ri);
            x87_addr(r10, r8);
            x87_addr(r11, r9);
        }
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
                a.x87_mem(esc, in.reg, mem(rsi));  // the operation with the single or double, as the interpreter's on its conversion
                x87_status(true);
                a.x87_mem(0xDB, 7, mem(r10));
                x87_tag_of(r10);
                x87_set_tag(r8);
                break;
            }
            case X87Form::CompareM32: {
                x87_addr(r10, r8);
                a.x87_mem(0xDB, 5, mem(r10));
                a.x87_mem(esc, 3, mem(rsi));  // fcomp dword / qword: the host's C3 C2 C0 are the guest's for ordered operands
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
            case X87Form::Sqrt: {
                // The host's square root (or rounding to an integer) under the guest's precision and rounding, as the
                // interpreter's (which leaves C1).
                a.x87_mem(0xDB, 5, mem(r10));
                a.x87_raw(0xD9, in.modrm);  // fsqrt / frndint
                x87_status(false);
                a.x87_mem(0xDB, 7, mem(r10));
                x87_tag_of(r10);
                x87_set_tag(r8);
                break;
            }
            case X87Form::LoadInt: {
                // FILD: every integer is exact in 80 bits (no control word matters).
                x87_phys(r9, 7);
                x87_addr(r10, r9);
                a.x87_mem(msize == 4 ? 0xDB : 0xDF, msize == 8 ? 5 : 0, mem(rsi));  // fild word / dword / qword
                a.x87_mem(0xDB, 7, mem(r10));  // fstp tbyte
                x87_clear_c1();
                x87_set_top(r9);
                x87_tag_of(r10);
                x87_set_tag(r9);
                break;
            }
            case X87Form::StoreInt: {
                // FIST(P) under the guest's rounding, as the interpreter's host conversion: out of range gives the
                // integer indefinite with IE (masked: the preamble), C1 from the rounding.
                x87_addr(r10, r8);
                a.x87_mem(0xDB, 5, mem(r10));  // fld tbyte
                a.x87_mem(msize == 4 ? 0xDB : 0xDF, msize == 8 ? 7 : 3, mem(rsi));  // fistp word / dword / qword
                x87_status(true);
                if (in.reg != 2) x87_pop();  // FIST /2 keeps it
                break;
            }
            case X87Form::CompareReg: {
                // FCOM(P) ST(i): C3 C2 C0 from the host's compare of the two (ordered: the checks), C1 cleared.
                a.x87_mem(0xDB, 5, mem(r11));  // ST(i), then ST(0) on top
                a.x87_mem(0xDB, 5, mem(r10));
                a.x87_raw(0xDE, 0xD9);  // fcompp
                a.fnstsw_ax();
                a.alu_imm(4, 4, rax, 0x4500);
                a.alu_imm(2, 4, st_at(kFsw), 0xB8FF);
                a.alu(2, 1, st_at(kFsw), rax);
                if (esc == 0xDA) {
                    // FUCOMPP: the same for ordered operands (the checks), then two pops.
                    x87_pop();
                    x87_top();
                    x87_pop();
                } else if (in.reg == 3) {
                    x87_pop();
                }
                break;
            }
            case X87Form::CompareFlags: {
                // F(U)COMI(P): ZF PF CF from the host's compare; the other flags as materialised; C1 cleared.
                a.x87_mem(0xDB, 5, mem(r11));
                a.x87_mem(0xDB, 5, mem(r10));
                a.x87_raw(0xDF, 0xF1);  // fcomip st, st(1)
                a.x87_raw(0xDD, 0xD8);  // fstp st(0)
                a.byte(0x9C);  // pushfq
                a.pop(rax);
                a.alu_imm(4, 4, rax, fl::ZF | fl::PF | fl::CF);
                a.mov(4, rcx, st_at(kFlagsOff));
                a.alu_imm(4, 4, rcx, ~(fl::ZF | fl::PF | fl::CF));
                a.alu(4, 1, rcx, rax);
                a.mov(4, st_at(kFlagsOff), rcx);
                x87_clear_c1();
                if (esc == 0xDF) x87_pop();
                known_ = Known::Unknown;
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
        alignas(16) static constexpr std::array<u32, 4 * 6> kConsts = {  // aligned: legacy SSE memory operands fault otherwise (MSVC aligns arrays to 4)
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
        enum class K { None, Load, Store, Arith, Logic, Shuf, Unpack, Cmp, MovMsk, LoadLow64, StoreLow64, LoadHigh64, StoreHigh64, Comi, CvtSi2Ss, CvtSs2Si } k = K::None;
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
            case 0x2E: case 0x2F: if (!scalar) k = K::Comi; break;  // UCOMISS / COMISS: a scalar compare into EFLAGS
            case 0x2A: if (scalar) k = K::CvtSi2Ss; break;            // CVTSI2SS xmm, r/m32
            case 0x2C: case 0x2D: if (scalar) k = K::CvtSs2Si; break; // CVT(T)SS2SI r32, xmm/m32
            default: break;
        }
        if (k == K::None) return false;
        const bool store = k == K::Store || k == K::StoreLow64 || k == K::StoreHigh64;
        const unsigned size = (k == K::LoadLow64 || k == K::StoreLow64 || k == K::LoadHigh64 || k == K::StoreHigh64) ? 8 : (scalar || k == K::Comi) ? 4 : 16;
        if (in.mem && !segment_ok(in.mem_seg, size, store)) return false;
        const bool host_math = k == K::Arith || k == K::Cmp || k == K::Comi || k == K::CvtSi2Ss || k == K::CvtSs2Si;
        const bool sse_before = sse_active_;
        const Label f = fallback(i, after, cw_active_, cw_active_, sse_before, host_math || sse_before);
        sse_preamble(f);
        // The second operand into xmm1 (memory: checked here, read after the cache lookup below).
        // The interpreter's 4-byte element accesses this form's one lookup stands for (see cache_elems).
        unsigned elems = size / 4;
        if (in.mem) {
            // Packed forms the interpreter checks for alignment (MOVAPS, arithmetic, logic, ...) fault there; an
            // aligned offset in a segment based on a multiple of 16 keeps the four elements in one line (one lookup).
            const bool checked = !scalar && size == 16 && op != 0x10 && op != 0x11;
            if (checked && (st_.seg[in.mem_seg].base & 15) == 0) elems = 1;
            sse_memory(in, size, store, checked ? 16 : 1, checked, f);
        }
        a.sse(0, 0x10, 0, xmm_at(d));  // xmm0 = the destination register (movups)
        if (k == K::Arith || k == K::Cmp || k == K::Comi) {
            const bool one = scalar || k == K::Comi;
            const unsigned lanes = one ? 1 : 15;
            if (in.mem) a.sse(one ? 0xF3 : 0, 0x10, 1, mem(rsi));  // movss/movups xmm1, [rsi]
            else a.sse(0, 0x10, 1, xmm_at(in.rm));
            if (op != 0x51) sse_lanes_ok(0, lanes, f);  // SQRT reads only its source
            sse_lanes_ok(1, lanes, f);
        }
        if (k == K::CvtSs2Si) {
            // The source into xmm1; a NaN, an infinity or anything of magnitude 2^31 or more is the interpreter's (it
            // gives the integer indefinite with IE itself, without the host).
            if (in.mem) a.sse(0xF3, 0x10, 1, mem(rsi));
            else a.sse(0, 0x10, 1, xmm_at(in.rm));
            a.sse_rr(0x66, 0x7E, 1, 0);  // movd eax, xmm1
            a.alu_imm(4, 4, rax, 0x7F800000);
            a.alu_imm(4, 7, rax, 158u << 23);
            a.jcc(AE, f);
        }
        if (in.mem) {
            keep_xmm_ = true;
            cache_elems(i, store, elems);  // keeps RSI and RDI (and xmm0/xmm1)
            keep_xmm_ = false;
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
            case K::Comi: {
                // No NaN or denormal (checked above): the compare raises no MXCSR exception and its ZF, PF and CF are
                // the architecture's; OF, SF and AF are cleared. It defines all six arithmetic flags, so the lazy record
                // is simply replaced (no materialising).
                a.sse_rr(0, op, 0, 1);
                a.byte(0x9C);  // pushfq
                a.pop(rax);
                a.alu_imm(4, 4, rax, fl::ZF | fl::PF | fl::CF);
                a.mov(4, rcx, st_at(kFlagsOff));
                a.alu_imm(4, 4, rcx, ~(fl::ZF | fl::PF | fl::CF | fl::OF | fl::SF | fl::AF));
                a.alu(4, 1, rcx, rax);
                a.mov(4, st_at(kFlagsOff), rcx);
                a.mov_imm(1, st_at(kLazyOp), static_cast<u32>(LazyOp::None));
                known_ = Known::Unknown;
                break;
            }
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
            case K::CvtSi2Ss:
                // The host's conversion under the guest's rounding, as the interpreter's (inexact: PE).
                if (in.mem) a.sse(0xF3, 0x2A, 0, mem(rsi));
                else {
                    a.mov(4, rax, greg(in.rm, 4));
                    a.sse_rr(0xF3, 0x2A, 0, 0);  // cvtsi2ss xmm0, eax
                }
                a.sse(0, 0x11, 0, xmm_at(d));
                break;
            case K::CvtSs2Si:
                a.sse_rr(0xF3, op, 0, 1);  // cvt(t)ss2si eax, xmm1: truncating, or under the guest's rounding
                a.mov(4, greg(static_cast<u8>(d), 4), rax);
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
        enum class K { None, Emms, MovdIn, MovdOut, MovqIn, MovqOut, Shift, Shuffle, Binary, Pextrw } k = K::None;
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
            case 0xC5: if (!in.mem) k = K::Pextrw; break;  // PEXTRW r32, mm, imm8
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
            cache_elems(i, store, size / 4);  // the interpreter reads or writes 64 bits as two halves
        }
        switch (k) {
            case K::Pextrw:
                mm_touch();
                a.movzx(rax, 2, mem(kSt, kSt0 + static_cast<i32>(16 * (in.rm & 7)) + static_cast<i32>(2 * (in.imm & 3))));
                a.mov(4, greg(static_cast<u8>(d), 4), rax);
                break;
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
            case Op::Lahf: return lahf(i);
            case Op::Wait: return fwait(i, after);
            case Op::Movs: case Op::Stos: return string_once(i, in, after);
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
    // MOVS and STOS without a REP prefix (32-bit addressing): the destination's translation first, then the source's
    // read and the destination's write in the interpreter's order; ESI/EDI step by the size, down with DF.
    bool string_once(unsigned i, const Insn& in, Label after) {
        if (in.rep || in.adsize != 4) return false;
        const unsigned size = in.size;
        const bool movs = in.op == Op::Movs;
        const u8 src_seg = in.seg != kSegNone ? in.seg : static_cast<u8>(kDs);
        if (!segment_ok(kEs, size, true) || (movs && !segment_ok(src_seg, size, false))) return false;
        const Label f = fallback(i, after);
        a.mov(4, rax, greg(kEdi, 4));
        access_checks(kEs, size, true, f);
        if (movs) {
            a.mov(8, ctx_at(kTmp2), rsi);
            a.mov(8, ctx_at(kTmp3), rdi);
            a.mov(4, rax, greg(kEsi, 4));
            access_checks(src_seg, size, false, f);
            cache(i, false);
            load_host(rax, size);
            a.mov(4, ctx_at(kCtxEa), rax);
            a.mov(8, rsi, ctx_at(kTmp2));
            a.mov(8, rdi, ctx_at(kTmp3));
            cache(i, true);
            a.mov(4, rax, ctx_at(kCtxEa));
        } else {
            cache(i, true);
            load_greg(rax, kEax, size);
        }
        a.mov(size, mem(rsi), rax);
        a.mov32(rcx, size);
        a.mov32(rdx, static_cast<u32>(-static_cast<i32>(size)));
        a.test_imm(4, st_at(kFlagsOff), fl::DF);
        a.cmov(4, NE, rcx, rdx);
        a.alu(4, 0, greg(kEdi, 4), rcx);
        if (movs) a.alu(4, 0, greg(kEsi, 4), rcx);
        host_flags_ = host_cf_ = false;
        finish(i, true);
        return true;
    }
    // LAHF: AH = the flags' low byte, computed as the interpreter computes it (no materialising).
    bool lahf(unsigned i) {
        a.mov(8, kArg0, kSt);
        a.call_abs(reinterpret_cast<const void*>(&h_eflags));
        a.mov(1, greg(4, 1), rax);  // AH
        host_flags_ = host_cf_ = false;
        finish(i, false);
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
    b.jit_key = engine.snapshot_key(b);
    Translator t(b, cpu, engine.native_view());
    t.run();
    if (!t.a.finish()) return false;
    const std::size_t n = t.a.code.size();
    if (used_ + n > capacity_) return false;
    std::memcpy(buffer_ + used_, t.a.code.data(), n);
    b.native = buffer_ + used_;
    b.entry.body = buffer_ + used_ + t.body;
    b.entry.jit_eip = b.jit_eip;
    for (unsigned k = 0; k < 2; ++k) {
        b.links[k].cell = t.cell[k];
        b.links[k].jump = t.cell_jump[k];
        b.links[k].patched = false;
    }
    // MV_JIT_MAP=FILE: the generated code's ranges and their guest blocks, for a sampling profiler.
    static std::FILE* map = [] {
        const char* p = std::getenv("MV_JIT_MAP");
        return p && *p ? std::fopen(p, "w") : nullptr;
    }();
    if (map) {
        std::fprintf(map, "%llx %llx %08x %u", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(b.native)),
                     static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(b.native) + n), b.jit_eip, b.n);
        for (const auto& [off, kind] : t.marks) std::fprintf(map, " %x:%s", off, kind.c_str());
        std::fputc('\n', map);
        std::fflush(map);
    }
    used_ += (n + 15) & ~std::size_t{15};
    ++compiled_;
    return true;
}

namespace {
void put32(u8* p, u32 v) { std::memcpy(p, &v, 4); }
}  // namespace

bool Jit::patch_link(Block& from, unsigned slot, const Block& to, Cpu& cpu) {
    Block::Link& l = from.links[slot];
    if (!from.native || !l.cell[0] || !to.entry.body) return false;
    const u32 cs = from.jit_seg[kCs].base;
    const u32 la = cs + from.jit_eip, lb = cs + to.jit_eip;
    const bool same_page = (la >> 12) == (lb >> 12) && to.ram_page == from.ram_page;
    const unsigned idx = (lb >> 12) & (Memory::kTlbSize - 1);
    const Memory::TlbEntry& e = cpu.memory().tlb()[idx];
    if (!same_page && (e.tag != (lb >> 12) || e.host != to.entry.host_page)) return false;
    const bool own = to.entry.fetch_line >= from.entry.fetch_line &&
                     to.entry.fetch_line + to.entry.fetch_lines <= from.entry.fetch_line + from.entry.fetch_lines;
    const unsigned lines = own ? 0u : to.entry.fetch_lines;
    if (lines > 4) return false;
    const Timing& t = cpu.cache_timing();
    const i64 l1i = reinterpret_cast<const u8*>(t.l1i_mru()) - reinterpret_cast<const u8*>(t.l1d_mru());
    u8* code = static_cast<u8*>(from.native);
    for (unsigned c = 0; c < 2; ++c) {
        if (!l.cell[c]) continue;
        u8* p = code + l.cell[c];
        put32(p + kCellLeaImm, to.cycles);
        if (same_page) {
            p[kCellSkipT] = 0xEB;
            p[kCellSkipT + 1] = static_cast<u8>(lines ? kCellSkipL - (kCellSkipT + 2) : kCellEnd - (kCellSkipT + 2));
        } else {
            p[kCellSkipT] = 0x66;
            p[kCellSkipT + 1] = 0x90;
            put32(p + kCellTagDisp, idx * 32);
            put32(p + kCellTagImm, lb >> 12);
            put32(p + kCellPageDisp, idx * 32 + 4);
            put32(p + kCellPageImm, e.phys_page);
        }
        if (lines == 4) {
            p[kCellSkipL] = 0x66;
            p[kCellSkipL + 1] = 0x90;
        } else {
            p[kCellSkipL] = 0xEB;
            p[kCellSkipL + 1] = static_cast<u8>((4 - lines) * kCellLine);
        }
        for (unsigned j = 0; j < lines; ++j) {
            u8* q = p + kCellLines + (4 - lines + j) * kCellLine;
            const u32 line = to.entry.fetch_line + j;  // plus one, as the MRU keeps it
            put32(q + 3, static_cast<u32>(static_cast<i32>(l1i + ((line - 1) & (t.l1i_sets() - 1)) * 4)));
            put32(q + 7, line);
        }
        u8* j = code + l.jump[c];
        put32(j + 1, static_cast<u32>(static_cast<u8*>(to.entry.body) - (j + 5)));
    }
    l.patched = true;
    return true;
}

void Jit::unpatch_link(Block& from, unsigned slot) {
    Block::Link& l = from.links[slot];
    if (from.native && l.patched)
        for (unsigned c = 0; c < 2; ++c)
            if (l.cell[c]) put32(static_cast<u8*>(from.native) + l.cell[c] + kCellLeaImm, kCellUnlinked);
    l.patched = false;
}

#else  // no x86-64 host: blocks stay interpreted

bool Jit::compile(Block&, Cpu&, BlockEngine&) { return false; }
bool Jit::patch_link(Block&, unsigned, const Block&, Cpu&) { return false; }
void Jit::unpatch_link(Block& from, unsigned slot) { from.links[slot].patched = false; }

#endif

}  // namespace mv::p6
