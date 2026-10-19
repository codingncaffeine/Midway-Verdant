#include "cpu/p6/block.h"

#include "cpu/p6/jit/jit.h"
#include "debug/trace.h"
#include "cpu/p6/timing.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string_view>

namespace mv::p6 {

namespace {

struct PageEnd {};

// Code bytes from the block's page; a byte past the page ends the block before the instruction that wants it.
class PageSource : public CodeSource {
public:
    PageSource(const u8* page, unsigned off) : page_(page), off_(off) {}
    void at(unsigned off) { off_ = off; }
    u8 byte(unsigned i) override {
        if (off_ + i >= 4096) throw PageEnd{};
        return page_[off_ + i];
    }

private:
    const u8* page_;
    unsigned off_;
};

bool is_string(Op op) { return op == Op::Movs || op == Op::Cmps || op == Op::Stos || op == Op::Lods || op == Op::Scas || op == Op::Ins || op == Op::Outs; }

// Instructions after which the next one may see a different machine: control transfers, interrupt-flag changes,
// segment and descriptor-table loads, control and debug registers, MSRs, port I/O, halts.
bool ends_block(const Insn& in) {
    switch (in.op) {
        case Op::Jmp: case Op::Jcc: case Op::JmpFar: case Op::JmpRm: case Op::JmpFarRm: case Op::Call: case Op::CallFar: case Op::CallRm:
        case Op::CallFarRm: case Op::Ret: case Op::Retf: case Op::Iret: case Op::Loop: case Op::Loope: case Op::Loopne: case Op::Jcxz:
        case Op::Int: case Op::Int3: case Op::Into: case Op::Int1: case Op::Sysenter: case Op::Sysexit:
        case Op::Cli: case Op::Sti: case Op::Popf: case Op::MovSeg: case Op::PopSeg: case Op::Lds: case Op::Les: case Op::Lfs: case Op::Lgs:
        case Op::Lss: case Op::MovToCr: case Op::MovToDr: case Op::Lgdt: case Op::Lidt: case Op::Lldt: case Op::Ltr: case Op::Lmsw: case Op::Clts:
        case Op::Invlpg: case Op::Wrmsr: case Op::Hlt: case Op::In: case Op::Out: case Op::Ins: case Op::Outs: case Op::Wbinvd: case Op::Invd:
        case Op::Rsm: case Op::Ud2: case Op::Invalid: case Op::Bound:
            return true;
        default: return false;
    }
}

bool same_segments(const Block& a, const Block& b) {
    for (unsigned g = 0; g < kSegCount; ++g)
        if (a.jit_seg[g].base != b.jit_seg[g].base || a.jit_seg[g].limit != b.jit_seg[g].limit || a.jit_seg[g].ar != b.jit_seg[g].ar) return false;
    return true;
}

BranchKind branch_kind(Op op) {
    switch (op) {
        case Op::Jcc: case Op::Loop: case Op::Loope: case Op::Loopne: case Op::Jcxz: return BranchKind::Conditional;
        case Op::Jmp: case Op::JmpFar: case Op::Int: case Op::Int3: case Op::Into: case Op::Int1: case Op::Sysenter: case Op::Sysexit: return BranchKind::Direct;
        case Op::JmpRm: case Op::JmpFarRm: return BranchKind::Indirect;
        case Op::Ret: case Op::Retf: case Op::Iret: return BranchKind::Return;
        case Op::Call: case Op::CallFar: return BranchKind::Call;
        case Op::CallRm: case Op::CallFarRm: return BranchKind::CallIndirect;
        default: return BranchKind::None;
    }
}

}  // namespace

BlockEngine::BlockEngine(Cpu& cpu, Bus& bus) : cpu_(cpu), bus_(bus), fast_(kFastSize), page_blocks_(bus.ram_size() / 4096) {
    bus_.set_code_tracking(true);
    cpu_.memory().set_code_watcher(this);
    if (const char* l = std::getenv("MV_P6_BLOCK_LOG"); l && *l) {
        std::string spec(l);
        const auto c2 = spec.rfind(':'), c1 = spec.rfind(':', c2 - 1);
        if (c1 != std::string::npos && c2 != std::string::npos && c1 < c2) {
            log_from_ = std::strtoull(spec.c_str() + c1 + 1, nullptr, 0);
            log_to_ = std::strtoull(spec.c_str() + c2 + 1, nullptr, 0);
            block_log_ = std::fopen(spec.substr(0, c1).c_str(), "w");
        }
    }
    const char* j = std::getenv("MV_P6_JIT");
    if (!(j && std::string_view(j) == "0")) {
        jit_ = std::make_unique<Jit>();
        if (!jit_->available()) jit_.reset();
    }
}

BlockEngine::~BlockEngine() {
    cpu_.memory().set_code_watcher(nullptr);
    bus_.set_code_tracking(false);
}

void BlockEngine::clear() {
    for (u32 p = 0; p < page_blocks_.size(); ++p)
        if (!page_blocks_[p].empty()) kill_page(p);
    graveyard_.clear();
}

Block* BlockEngine::find() {
    if (!running_ && !graveyard_.empty()) graveyard_.clear();
    const CpuState& s = cpu_.state();
    const u32 lin = s.seg[kCs].base + s.eip;
    const u8 mode = static_cast<u8>((s.code32() ? 1 : 0) | (s.stack32() ? 2 : 0));
    const u8* host;
    try {
        host = cpu_.memory().host_pointer(lin, Access::Fetch);
    } catch (const Fault&) {
        return nullptr;  // the step raises the fetch fault
    }
    if (!host) return nullptr;
    const u32 page = bus_.ram_page(host);
    const unsigned off = lin & 0xFFFu;
    FastEntry& f = fast_[fast_index(lin, mode)];
    Block* b = nullptr;
    if (f.epoch == epoch_ && f.lin == lin && f.mode == mode && f.block->ram_page == page && f.block->page_off == off) {
        b = f.block;
    } else {
        const auto it = blocks_.find(key(page, off, mode));
        b = it != blocks_.end() ? it->second.get() : build(host - off, off, page, mode);
        if (!b) return nullptr;
        if (!b->lin) b->lin = lin;
        f = FastEntry{lin, mode, epoch_, b};
    }
    // The whole block inside the code segment (the step faults at the instruction that leaves it).
    const u32 last = s.eip + b->bytes - 1u;
    if (last < s.eip || last > s.seg[kCs].limit) return nullptr;
    return b;
}

Block* BlockEngine::build(const u8* page, unsigned off, u32 ram_page, u8 mode) {
    auto b = std::make_unique<Block>();
    b->ram_page = ram_page;
    b->page_off = static_cast<u16>(off);
    b->mode = mode;
    PageSource src(page, off);
    unsigned at = off;
    Insn in;
    while (b->insns.size() < kMaxBlockInsns) {
        src.at(at);
        try {
            decode(src, (mode & 1) != 0, in);
        } catch (const PageEnd&) {
            break;
        } catch (const Fault&) {
            break;  // invalid here: the step raises it when execution gets there
        }
        if (in.rep && is_string(in.op)) {
            if (b->insns.empty()) {
                b->off[0] = static_cast<u16>(at - off);
                b->insns.push_back(in);
                b->rep = true;
                at += in.len;
            }
            break;
        }
        b->off[b->insns.size()] = static_cast<u16>(at - off);
        b->insns.push_back(in);
        at += in.len;
        if (ends_block(in)) break;
    }
    if (b->insns.empty()) return nullptr;
    b->n = static_cast<u8>(b->insns.size());
    b->bytes = static_cast<u16>(at - off);
    b->branch = branch_kind(b->insns.back().op);
    std::array<const Insn*, kMaxBlockInsns> list{};
    for (unsigned i = 0; i < b->n; ++i) list[i] = &b->insns[i];
    b->cycles = schedule_block(cpu_.scheduler(), list.data(), b->n, b->after.data());
    b->entry.cycles = b->cycles;
    b->entry.n = b->n;
    b->entry.host_page = bus_.ram() + static_cast<std::size_t>(ram_page) * 4096u;
    {
        const PhysAddr code = static_cast<PhysAddr>(ram_page) * 4096u + static_cast<PhysAddr>(off);
        b->entry.fetch_line = (code >> 5) + 1;
        b->entry.fetch_lines = static_cast<u8>(((code + b->bytes - 1) >> 5) - (code >> 5) + 1);
    }
    Block* raw = b.get();
    page_blocks_[ram_page].push_back(raw);
    bus_.set_code_page(ram_page, true);
    blocks_.emplace(key(ram_page, off, mode), std::move(b));
    ++stats_.built;
    return raw;
}

std::string BlockEngine::hottest(unsigned count) const {
    std::vector<const Block*> all;
    for (const auto& [k, b] : blocks_) all.push_back(b.get());
    std::sort(all.begin(), all.end(), [](const Block* a, const Block* b) { return a->entry.runs * a->cycles > b->entry.runs * b->cycles; });
    std::string s;
    for (unsigned i = 0; i < count && i < all.size(); ++i)
        s += std::format(" {:08x}x{}:{}c*{}", all[i]->lin, all[i]->n, all[i]->cycles, all[i]->entry.runs);
    return s;
}

void BlockEngine::kill_page(u32 ram_page) {
    if (ram_page >= page_blocks_.size()) return;
    auto& list = page_blocks_[ram_page];
    for (Block* b : list) {
        if (b == running_ || (current_ && current_->block == b)) running_dead_ = true;
        const auto it = blocks_.find(key(b->ram_page, b->page_off, b->mode));
        if (it != blocks_.end()) {
            graveyard_.push_back(std::move(it->second));
            blocks_.erase(it);
        }
        ++stats_.killed;
    }
    list.clear();
    bus_.set_code_page(ram_page, false);
    ++epoch_;
    ++link_epoch_;
    pending_slot_ = nullptr;
    ++stats_.pages_written;
}

void BlockEngine::code_written(u32 ram_page) { kill_page(ram_page); }

void BlockEngine::take_device_writes() {
    written_.clear();
    bus_.take_code_writes(written_);
    for (u32 p : written_) {
        kill_page(p);
        ++stats_.device_page_writes;
    }
}

u32 BlockEngine::exit_cost(Block& b, u32 entry_eip) {
    const CpuState& s = cpu_.state();
    if (probing_ && b.branch != BranchKind::None) {
        // A spin probe: each predictor's state before the pass first changes it.
        bool seen = false;
        for (const ProbedExit& p : probed_exits_) seen = seen || p.block == &b;
        if (!seen) probed_exits_.push_back({&b, b.exit_counter, b.exit_seen, b.exit_target});
    }
    const u32 fall = (entry_eip + b.bytes) & (s.code32() ? 0xFFFFFFFFu : 0xFFFFu);
    const u32 target = s.eip;
    bool mispredict = false;
    switch (b.branch) {
        case BranchKind::None: return 0;
        case BranchKind::Conditional: {
            const bool taken = target != fall;
            mispredict = (b.exit_counter >= 2) != taken;
            if (taken && b.exit_counter < 3) ++b.exit_counter;
            if (!taken && b.exit_counter > 0) --b.exit_counter;
            break;
        }
        case BranchKind::Direct: mispredict = !b.exit_seen; break;
        case BranchKind::Call:
            mispredict = !b.exit_seen;
            rsb_[rsb_top_] = fall;
            rsb_top_ = (rsb_top_ + 1) & 15;
            break;
        case BranchKind::Indirect: mispredict = !b.exit_seen || b.exit_target != target; break;
        case BranchKind::CallIndirect:
            mispredict = !b.exit_seen || b.exit_target != target;
            rsb_[rsb_top_] = fall;
            rsb_top_ = (rsb_top_ + 1) & 15;
            break;
        case BranchKind::Return:
            rsb_top_ = (rsb_top_ + 15) & 15;
            mispredict = rsb_[rsb_top_] != target;
            break;
    }
    b.exit_seen = true;
    b.exit_target = target;
    if (!mispredict) return 0;
    ++stats_.mispredicts;
    return tp::kBlockMispredict;
}

BlockEngine::Run BlockEngine::begin(Block& b, u64 limit) {
    ++b.entry.runs;
    const PhysAddr code = static_cast<PhysAddr>(b.ram_page) * 4096u + b.page_off;
    // Penalties still pending (a page walk for the fetch, a delivery's stack) belong to the run from its start.
    const u64 pending = cpu_.memory().take_penalty();
    const Run r{&b, cpu_.cycles(), cpu_.state().eip, pending + cpu_.cache_timing().code_fetch(code, b.bytes), limit};
    cpu_.set_cycles_only(r.start + r.extra);
    return r;
}

BlockEngine::Next BlockEngine::exec(Run& r, unsigned i) {
    Memory& mem = cpu_.memory();
    Block& b = *r.b;
    const u64 devices = mem.device_events();
    running_ = &b;
    running_dead_ = false;
    const bool ok = cpu_.exec_decoded(b.insns[i]);
    running_ = nullptr;
    ++stats_.block_insns;
    const u64 penalty = mem.take_penalty();
    r.extra += penalty;
    stats_.penalty_cycles += penalty;
    if (!ok) {
        // The faulting instruction retires (as the step counts it), then the delivery.
        stats_.schedule_cycles += b.after[i];
        cpu_.set_cycles_only(r.start + r.extra + b.after[i] + tp::kInterruptDelivery);
        return Next::Exit;
    }
    if (b.rep) {
        const unsigned n = cpu_.rep_iterations();
        stats_.rep_cycles += static_cast<u64>(b.cycles) * (n > 1 ? n : 1);
        cpu_.set_cycles_only(r.start + r.extra + static_cast<u64>(b.cycles) * (n > 1 ? n : 1));
        return Next::Exit;
    }
    if (i + 1 == b.n) {
        const u32 exit = exit_cost(b, r.entry_eip);
        stats_.schedule_cycles += b.cycles;
        stats_.exit_cycles += exit;
        cpu_.set_cycles_only(r.start + r.extra + b.cycles + exit);
        return Next::Exit;
    }
    const u64 next = r.start + r.extra + b.after[i];
    cpu_.set_cycles_only(next);
    if (mem.device_events() != devices || running_dead_ || next >= r.limit) {
        stats_.schedule_cycles += b.after[i];
        return Next::Exit;
    }
    return Next::Continue;
}

unsigned BlockEngine::run(Block& b, u64 limit) {
    Run r = begin(b, limit);
    cpu_.cache_timing().trace_ = nullptr;
    if (block_log_ && r.start >= log_from_ && r.start <= log_to_) {
        const CpuState& s = cpu_.state();
        u64 h = 0xcbf29ce484222325ull;
        for (u32 g : s.gpr) h = (h ^ g) * 0x100000001b3ull;
        h = (h ^ s.eflags()) * 0x100000001b3ull;
        const TimingStats& t = cpu_.cache_timing().stats();
        cpu_.cache_timing().trace_ = block_log_;
        std::fprintf(block_log_, "  schedule %u:", b.cycles);
        for (unsigned k = 0; k < b.n; ++k) std::fprintf(block_log_, " %u", b.after[k]);
        std::fprintf(block_log_, " extra %llu\n", static_cast<unsigned long long>(r.extra));
        std::fprintf(block_log_, "%llu %08x %u %016llx l1d %llu l2 %llu l1i %llu caches %016llx limit %llu inhibit %d\n", static_cast<unsigned long long>(r.start), s.eip, b.n, static_cast<unsigned long long>(h),
                     static_cast<unsigned long long>(t.l1d_misses), static_cast<unsigned long long>(t.l2_misses), static_cast<unsigned long long>(t.l1i_misses),
                     static_cast<unsigned long long>(cpu_.cache_timing().cache_hash()), static_cast<unsigned long long>(r.limit), s.inhibit_interrupts ? 1 : 0);
    }
    ++stats_.whole_runs;
    if (jit_ && !b.rep) {
        // Hot blocks are translated; a block whose assumptions no longer hold is translated again, a few times.
        if (!b.native && !b.jit_failed && b.entry.runs >= hot_) {
            if (!jit_->compile(b, cpu_, *this)) {
                jit_->flush(*this);
                if (!jit_->compile(b, cpu_, *this)) b.jit_failed = true;
            }
            ++b.jit_compiles;
            ++link_epoch_;
        }
        // The last native exit's slot learns where it led, when a chain there could be valid (same snapshot).
        if (pending_slot_) {
            // Only the block the exit itself led to: an interrupt, a fault or a fallback step in between runs another.
            if (b.native && b.jit_eip == r.entry_eip && r.entry_eip == pending_target_ && b.jit_dirty == pending_from_->jit_dirty &&
                same_segments(b, *pending_from_) && b.jit_eip != idle_[0] && b.jit_eip != idle_[1]) {
                pending_slot_->to = &b;
                pending_slot_->epoch = link_epoch_;
            }
            else if (b.native && r.entry_eip != pending_target_) {
                if (++stats_.links_refused <= 8)
                    trace::log(trace::Cat::Cpu, "link refused: an exit of the block at {:08x} went to {:08x}, the next block run starts at {:08x}", pending_from_->lin,
                               pending_target_, r.entry_eip);
            }
            pending_slot_ = nullptr;
        }
        if (b.native && r.start + r.extra + b.cycles < r.limit) {
            if (Jit::eligible(b, cpu_)) return run_native(r);
            ++stats_.native_ineligible;
            if (b.jit_compiles < 4) b.native = b.entry.body = nullptr;
            else b.jit_failed = true;
            ++link_epoch_;
        }
    }
    unsigned i = 0;
    while (exec(r, i) == Next::Continue) ++i;
    return i + 1;
}

}  // namespace mv::p6

namespace mv::p6 {

// A native run: the generated code executes the block (with the interpreter for what it does not cover), maybe
// chaining into further blocks, and returns the instruction the last block ended at; the clock is then set exactly
// as `exec` would have set it.
unsigned BlockEngine::run_native(Run& r) {
    Memory& mem = cpu_.memory();
    JitCtx c{};
    c.st = &cpu_.state();
    c.cpu = &cpu_;
    c.engine = this;
    c.block = r.b;
    c.base = r.start + r.extra;
    c.limit = r.limit;
    c.devices = mem.device_events();
    c.error = &jit_error_;
    c.entry_eip = r.entry_eip;
    c.helper_ops = stats_.helper_ops.data();
    c.chain_budget = chain_budget_;
    c.chain_extra = 4096;
    c.start_eip = r.entry_eip;
    // A spin probe: about one pass (a short budget: the run goes on to where it began) with the stores logged.
    if (probe_eip_ && probe_eip_ == r.entry_eip && chain_budget_ >= 64) {
        // Every transition through the engine (a budget of 0: the generated code never commits a chain), one pass:
        // the chain goes on until it is back where it began. The predictors and the return stack are recorded.
        c.logging = 1;
        c.chain_budget = 0;
        probing_ = true;
        probed_exits_.clear();
        rsb_at_probe_ = rsb_;
        rsb_top_at_probe_ = rsb_top_;
    }
    // The state the run began from, for the spin skip (a copy of the registers; the run is long when it matters).
    const bool spin_watch = chain_budget_ >= 64;
    if (spin_watch) std::memcpy(&spin_start_, &cpu_.state(), sizeof spin_start_);  // bytes, padding too: compared as bytes
    cpu_.clear_faulted();
    running_ = r.b;
    running_dead_ = false;
    current_ = &c;
    const u32 result = Jit::entry(*r.b)(&c);
    current_ = nullptr;
    probing_ = false;
    running_ = nullptr;
    if (jit_error_) {
        const std::exception_ptr e = jit_error_;
        jit_error_ = nullptr;
        std::rethrow_exception(e);
    }
    Block& b = *c.block;  // the block the run ended in
    const unsigned i = result >> 2;
    const unsigned kind = result & 3;
    const u64 total = c.insns_done + i + 1;
    cpu_.add_instructions(total - c.helpers);
    stats_.block_insns += i + 1;
    ++stats_.native_runs;
    if ((stats_.native_runs & 0xFFFF) == 0) not_fixed_.fill(0);  // a loop that warmed up gets another chance
    stats_.native_insns += total - c.helpers;
    stats_.native_helpers += c.helpers;
    stats_.chains += c.inline_chains;
    stats_.whole_runs += c.inline_chains;
    stats_.mispredicts += c.inline_mispredicts;
    stats_.exit_cycles += static_cast<u64>(c.inline_mispredicts) * tp::kBlockMispredict;
    stats_.schedule_cycles += c.inline_sched;
    stats_.block_insns += c.inline_insns;
    const u64 left = mem.take_penalty();
    c.base += left;
    stats_.penalty_cycles += c.penalties + left;
    if (kind == kJitEnd) {
        if (c.ended) {
            cpu_.set_cycles_only(c.end_cycles);  // the exit was costed where the chain stopped
            pending_slot_ = c.last_slot;
            pending_target_ = c.last_target;
            pending_from_ = &b;
            if ((spin_watch || c.logging) && r.extra == 0) spin_skip(c, r.start, total - c.helpers);
        } else {
            const u32 exit = exit_cost(b, c.entry_eip);
            stats_.schedule_cycles += b.cycles;
            stats_.exit_cycles += exit;
            cpu_.set_cycles_only(c.base + b.cycles + exit);
        }
    } else if (kind == kJitExit && c.exit == 2) {
        // The interpreter finishes the block from the next instruction, with the penalties so far.
        cpu_.set_cycles_only(c.base + b.after[i]);
        Run rest{&b, c.base, c.entry_eip, 0, c.limit};
        ++stats_.native_handoffs;
        unsigned j = i + 1;
        while (exec(rest, j) == Next::Continue) ++j;
        return static_cast<unsigned>(c.insns_done) + j + 1;
    } else {
        stats_.schedule_cycles += b.after[i];
        cpu_.set_cycles_only(c.base + b.after[i] + (kind == kJitFault ? tp::kInterruptDelivery : 0));
    }
    return static_cast<unsigned>(total);
}

// The end of a native block whose last instruction left normally. Its exit is costed as `exec` costs it. Then the run
// goes on in the next block's native code only where the dispatcher would have done nothing in between: the next
// deadline and the stop cycle lie past the next block even with the slowest code fetch (so no event fires and no
// interrupt becomes deliverable — native code changes neither IF nor the PIC), no device wrote into code, the fetch
// translation hits the TLB on the page the block came from (find() would walk nothing), it is not the idle loop, and
// the link names that block's current translation with the same segments. Then its run begins as `begin` would.
void* BlockEngine::chain(JitCtx& c, Block::Link* slot) {
    Block& a = *c.block;
    const u32 exit = exit_cost(a, c.entry_eip);
    stats_.schedule_cycles += a.cycles;
    stats_.exit_cycles += exit;
    const u64 end = c.base + a.cycles + exit;
    c.end_cycles = end;
    c.ended = 1;
    c.last_slot = slot;
    c.last_target = cpu_.state().eip;
    Block* b = slot->to;
    if (exit) ++c.mispredicts;
    if (!b || slot->epoch != link_epoch_ || !b->native) return nullptr;
    // Past the budget a run goes on (a little) until it is back where it began: the spin skip compares the two.
    // (only while the run could still be a skippable pass, and not from where that already failed)
    if (c.chain_budget == 0 && (back_at_start(c) || c.chain_extra == 0 || c.helpers || c.penalties || not_a_fixed_point(c.start_eip))) return nullptr;
    const CpuState& s = cpu_.state();
    if (b->jit_eip != s.eip || s.eip == idle_[0] || s.eip == idle_[1]) return nullptr;
    constexpr u64 kSlowestFetch = tp::kL2Latency + tp::kDramLatency;
    if (end + b->cycles + kSlowestFetch >= c.limit) return nullptr;
    if (bus_.code_writes_pending() || cpu_.memory().penalty_pending()) return nullptr;
    const u32 lin = s.seg[kCs].base + s.eip;
    const Memory::TlbEntry& e = cpu_.memory().tlb()[(lin >> 12) & (Memory::kTlbSize - 1)];
    if (e.tag != (lin >> 12) || !e.host || bus_.ram_page(e.host) != b->ram_page || (lin & 0xFFFu) != b->page_off) return nullptr;
    if (c.chain_budget) --c.chain_budget;
    else --c.chain_extra;
    ++b->entry.runs;
    ++stats_.whole_runs;
    ++stats_.chains;
    stats_.block_insns += a.n;
    c.insns_done += a.n;
    const PhysAddr code = static_cast<PhysAddr>(b->ram_page) * 4096u + b->page_off;
    const u64 fetch = cpu_.cache_timing().code_fetch(code, b->bytes);
    c.penalties += fetch;
    c.base = end + fetch;
    c.block = b;
    c.entry_eip = s.eip;
    c.ended = 0;
    running_ = b;
    cpu_.set_cycles_only(c.base);
    return b->entry.body;
}

void BlockEngine::disable_jit() { jit_.reset(); }

void BlockEngine::drop_native() {
    ++link_epoch_;
    pending_slot_ = nullptr;
    for (auto& [k, b] : blocks_) b->native = b->entry.body = nullptr;
    for (auto& b : graveyard_) b->native = b->entry.body = nullptr;
}

}  // namespace mv::p6

namespace mv::p6 {

// The spin skip (Dolphin's idle skipping, made exact). A native run that wrote no memory, ran no instruction through
// the interpreter (so reached no device and walked no page), took no cache penalty (every access hit: the caches'
// contents and LRU order end each pass as they began) and no mispredict (the predictors are saturated), and ended
// at the EIP it began at with the whole CPU state as it began, is a pass of a loop that repeats exactly until
// something outside the CPU changes memory or raises an interrupt — which only events do (and the devices' own
// threads, whose timing is the host's either way). So the clock and the instruction count move on by whole passes,
// stopping a pass short of the limit; the run after that reaches the limit's instruction boundary as it would have.
void BlockEngine::spin_skip(const JitCtx& c, u64 start, u64 instructions) {
    const bool probe = c.logging != 0;
    if (probe) probe_eip_ = 0;  // one probe a candidate; a later run that qualifies arms another
    const u32 chained = probe ? 1u : chain_budget_ - c.chain_budget;
    if (!probe && chained < 64) return;
    if (c.helpers) { ++stats_.spin_refused[0]; return; }
    if (c.penalties) { ++stats_.spin_refused[1]; return; }
    if (c.mispredicts && !probe) {
        // Mispredicts: a periodic pattern leaves each predictor as it found it after a pass; a probe checks that.
        ++stats_.spin_refused[2];
        if (std::memcmp(&cpu_.state(), &spin_start_, sizeof(CpuState)) == 0 && cpu_.state().eip == c.start_eip) probe_eip_ = c.start_eip;
        return;
    }
    if (probe && !predictors_undone()) { ++stats_.spin_refused[2]; return; }
    if (instructions == 0) return;
    const CpuState& s = cpu_.state();
    if (s.eip != c.start_eip) {
        ++stats_.spin_refused[3];
        return;
    }
    if (std::memcmp(&s, &spin_start_, sizeof s) != 0) {
        ++stats_.spin_refused[4];
        remember_not_fixed(c.start_eip);
        return;
    }
    if (c.stores) {
        // Stores: a pass that leaves memory as it found it is still a fixed point. A long run arms a probe; the
        // probe checks every location its stores touched holds its first replaced bytes again.
        if (!probe) {
            probe_eip_ = c.start_eip;
            return;
        }
        if (c.log_count > 256) { ++stats_.spin_refused[5]; return; }
        if (!stores_undone(c)) { ++stats_.spin_refused[6]; return; }
    }
    const u64 now = cpu_.cycles();
    const u64 period = now - start;
    if (period == 0 || now >= c.limit) return;
    const u64 passes = (c.limit - now) / period;
    if (passes < 2) return;
    const u64 k = passes - 1;
    cpu_.set_cycles_only(now + k * period);
    cpu_.add_instructions(k * instructions);
    ++stats_.spin_skips;
    stats_.spin_cycles += k * period;
}

}  // namespace mv::p6

namespace mv::p6 {

// Whether every byte the probe's stores touched holds the value it had before the first store to it.
bool BlockEngine::stores_undone(const JitCtx& c) const {
    // Each byte's value before the first store to it, from the log in order; then every such byte compared.
    using Byte = ProbeByte;
    std::vector<Byte>& bytes = probe_bytes_;
    bytes.clear();
    for (u32 i = 0; i < c.log_count; ++i) {
        const u8* p = reinterpret_cast<const u8*>(c.log[i].where & 0x00FFFFFFFFFFFFFFull);
        const unsigned n = static_cast<unsigned>(c.log[i].where >> 56);
        for (unsigned k = 0; k < n; ++k) bytes.push_back({p + k, i, static_cast<u8>(c.log[i].old >> (8 * k))});
    }
    std::sort(bytes.begin(), bytes.end(), [](const Byte& a, const Byte& b) { return a.at != b.at ? a.at < b.at : a.order < b.order; });
    for (std::size_t k = 0; k < bytes.size(); ++k)
        if ((k == 0 || bytes[k].at != bytes[k - 1].at) && *bytes[k].at != bytes[k].original) return false;
    return true;
}

}  // namespace mv::p6

namespace mv::p6 {

// Whether the probe's pass left every predictor it touched, and the return stack, as it found them.
bool BlockEngine::predictors_undone() const {
    for (const ProbedExit& p : probed_exits_)
        if (p.block->exit_counter != p.counter || p.block->exit_seen != p.seen || p.block->exit_target != p.target) return false;
    return rsb_ == rsb_at_probe_ && rsb_top_ == rsb_top_at_probe_;
}

}  // namespace mv::p6

namespace mv::p6 {

// Back where the run began: the same EIP and registers (a nested loop's inner block recurs before its period ends).
bool BlockEngine::back_at_start(const JitCtx& c) const {
    const CpuState& s = cpu_.state();
    return s.eip == c.start_eip && s.gpr == spin_start_.gpr && s.eflags() == spin_start_.eflags();
}

}  // namespace mv::p6
