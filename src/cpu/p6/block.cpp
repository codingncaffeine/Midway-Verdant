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

BlockEngine::BlockEngine(Cpu& cpu, Bus& bus)
    : cpu_(cpu), bus_(bus), fast_(kFastSize), page_blocks_(bus.ram_size() / 4096), code_granules_(bus.ram_size() >> kCodeGranuleBits),
      ctx_(std::make_unique<JitCtx>()) {
    bus_.set_code_tracking(true);
    cpu_.memory().set_code_watcher(this, code_granules_.data());
    ctx_->st = &cpu_.state();
    ctx_->cpu = &cpu_;
    ctx_->engine = this;
    ctx_->error = &jit_error_;
    ctx_->helper_ops = stats_.helper_ops.data();
    bus_.set_code_write_kill(&ctx_->chain_limit);
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
    bus_.set_code_write_kill(nullptr);
    cpu_.memory().set_code_watcher(nullptr, nullptr);
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
    ++stats_.lookups;
    if (f.epoch == epoch_ && f.lin == lin && f.mode == mode && f.block->ram_page == page && f.block->page_off == off) {
        b = f.block;
        ++stats_.fast_hits;
    } else {
        const auto it = blocks_.find(key(page, off, mode));
        b = it != blocks_.end() ? it->second.get() : build(host - off, off, page, mode);
        if (!b->lin) b->lin = lin;
        f = FastEntry{lin, mode, epoch_, b};
        const u32 idx = fast_index(lin, mode);
        if (std::find(b->fast_slots.begin(), b->fast_slots.end(), idx) == b->fast_slots.end()) b->fast_slots.push_back(idx);
    }
    if (b->n == 0) return nullptr;  // no block can start here: the step runs it
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
    if (b->insns.empty()) {
        // The first instruction straddles the page's end, or does not decode: an empty block remembers it, so the
        // next visit steps at once (no build, no exception); it holds the page's bytes from here to the end (all the
        // decode read), and a write over them drops it as it would a block.
        b->bytes = static_cast<u16>(4096 - off);
        ++stats_.empty_blocks;
    }
    b->n = static_cast<u8>(b->insns.size());
    if (b->n) b->bytes = static_cast<u16>(at - off);
    b->branch = b->n ? branch_kind(b->insns.back().op) : BranchKind::None;
    if (b->n == 2 && b->insns[1].op == Op::Jcc && b->insns[1].cc == 5 && b->insns[1].imm == static_cast<u32>(-static_cast<i32>(b->bytes))) {
        const Insn& d = b->insns[0];
        const bool by_one = d.dst.kind == OpndKind::Reg && d.size == 4 && !d.lock &&
                            (d.op == Op::Dec || (d.src.kind == OpndKind::Imm && ((d.op == Op::Sub && d.imm == 1) || (d.op == Op::Add && d.imm == 0xFFFFFFFFu))));
        if (by_one) b->countdown = d.dst.reg;
    }
    std::array<const Insn*, kMaxBlockInsns> list{};
    for (unsigned i = 0; i < b->n; ++i) list[i] = &b->insns[i];
    b->cycles = b->n ? schedule_block(cpu_.scheduler(), list.data(), b->n, b->after.data()) : 0;
    b->entry.cycles = b->cycles;
    b->entry.n = b->n;
    b->entry.host_page = bus_.ram() + static_cast<std::size_t>(ram_page) * 4096u;
    {
        const PhysAddr code = static_cast<PhysAddr>(ram_page) * 4096u + static_cast<PhysAddr>(off);
        b->entry.fetch_line = (code >> 5) + 1;
        b->entry.fetch_lines = static_cast<u8>(((code + b->bytes - 1) >> 5) - (code >> 5) + 1);
    }
    Block* raw = b.get();
    for (Block::Link& l : raw->links) l.from = raw;
    page_blocks_[ram_page].push_back(raw);
    bus_.set_code_page(ram_page, true);
    {
        // A write of up to 16 bytes that overlaps the code starts at most 15 bytes before it (writes never cross a
        // page here: the interpreter splits them, the generated code refuses them).
        const unsigned from = off >= 15 ? off - 15 : 0, to = off + b->bytes;
        u8* g = code_granules_.data() + (static_cast<std::size_t>(ram_page) << (12 - kCodeGranuleBits));
        for (unsigned k = from >> kCodeGranuleBits; k <= (to - 1) >> kCodeGranuleBits; ++k) g[k] = 1;
    }
    blocks_.emplace(key(ram_page, off, mode), std::move(b));
    if (raw->n) ++stats_.built;
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

void BlockEngine::kill_block(Block* b) {
    if (b == running_ || (current_ && current_->block == b)) running_dead_ = true;
    forget(*b);
    const auto it = blocks_.find(key(b->ram_page, b->page_off, b->mode));
    if (it != blocks_.end()) {
        graveyard_.push_back(std::move(it->second));
        blocks_.erase(it);
    }
    ++stats_.killed;
}

void BlockEngine::kill_page(u32 ram_page) {
    if (ram_page >= page_blocks_.size()) return;
    auto& list = page_blocks_[ram_page];
    for (Block* b : list) kill_block(b);
    list.clear();
    mark_granules(ram_page);
    bus_.set_code_page(ram_page, false);
    pending_slot_ = nullptr;
    ++stats_.pages_written;
}

void BlockEngine::kill_range(u32 ram_page, unsigned off, unsigned len) {
    if (ram_page >= page_blocks_.size()) return;
    if (off + len > 4096) kill_range(ram_page + 1, 0, off + len - 4096);
    auto& list = page_blocks_[ram_page];
    std::size_t kept = 0;
    bool any = false;
    for (Block* b : list) {
        if (off < b->page_off + b->bytes && off + len > b->page_off) {
            kill_block(b);
            any = true;
        } else {
            list[kept++] = b;  // its code and its predictor stay (a store elsewhere in the page changes neither: D30)
        }
    }
    list.resize(kept);
    mark_granules(ram_page);
    if (list.empty()) bus_.set_code_page(ram_page, false);
    if (any) {
        pending_slot_ = nullptr;
        ++stats_.pages_written;
    } else {
        ++stats_.code_page_writes_kept;
    }
}

void BlockEngine::mark_granules(u32 ram_page) {
    u8* g = code_granules_.data() + (static_cast<std::size_t>(ram_page) << (12 - kCodeGranuleBits));
    std::memset(g, 0, std::size_t{1} << (12 - kCodeGranuleBits));
    for (const Block* b : page_blocks_[ram_page]) {
        const unsigned off = b->page_off, from = off >= 15 ? off - 15 : 0, to = off + b->bytes;
        for (unsigned k = from >> kCodeGranuleBits; k <= (to - 1) >> kCodeGranuleBits; ++k) g[k] = 1;
    }
}

void BlockEngine::set_link(Block::Link& slot, Block* to) {
    if (slot.to == to) {
        if (to && !slot.patched) Jit::patch_link(*slot.from, static_cast<unsigned>(&slot - slot.from->links.data()), *to, cpu_);
        return;
    }
    if (slot.patched) Jit::unpatch_link(*slot.from, static_cast<unsigned>(&slot - slot.from->links.data()));
    if (Block* old = slot.to) {
        auto& in = old->incoming;
        if (const auto it = std::find(in.begin(), in.end(), &slot); it != in.end()) {
            *it = in.back();
            in.pop_back();
        }
    }
    slot.to = to;
    if (to) {
        to->incoming.push_back(&slot);
        Jit::patch_link(*slot.from, static_cast<unsigned>(&slot - slot.from->links.data()), *to, cpu_);
    }
}

void BlockEngine::drop_links(Block& b) {
    for (u32 i : b.target_slots) {
        JitCtx::Target& t = ctx_->targets[i];
        if (t.block == &b) t = JitCtx::Target{};
    }
    b.target_slots.clear();
    for (Block::Link* s : b.incoming) {
        if (s->patched) Jit::unpatch_link(*s->from, static_cast<unsigned>(s - s->from->links.data()));
        s->to = nullptr;
    }
    b.incoming.clear();
    for (Block::Link& s : b.links) set_link(s, nullptr);
}

u32 BlockEngine::snapshot_key(const Block& b) {
    for (std::size_t k = 0; k < snapshots_.size(); ++k) {
        const auto& [seg, dirty] = snapshots_[k];
        bool same = dirty == b.jit_dirty;
        for (unsigned g = 0; same && g < kSegCount; ++g) same = seg[g].base == b.jit_seg[g].base && seg[g].limit == b.jit_seg[g].limit && seg[g].ar == b.jit_seg[g].ar;
        if (same) return static_cast<u32>(k + 1);
    }
    snapshots_.emplace_back(b.jit_seg, b.jit_dirty);
    return static_cast<u32>(snapshots_.size());
}

void BlockEngine::set_idle_eips(u32 a, u32 b) {
    idle_ = {a, b};
    ++link_epoch_;  // links made before are not taken by the engine's chain
    for (auto& [k, blk] : blocks_)
        if (blk->jit_eip == a || blk->jit_eip == b) drop_links(*blk);  // nor by a chain cell
}

void BlockEngine::forget(Block& b) {
    drop_links(b);
    for (u32 i : b.fast_slots)
        if (fast_[i].block == &b) fast_[i] = FastEntry{};
    b.fast_slots.clear();
}

void BlockEngine::code_written(u32 ram_page, unsigned off, unsigned len) { kill_range(ram_page, off, len); }

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
    if (jit_ && b.countdown != 0xFF) countdown_skip(b, limit);
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
            // Links are only ever learned to blocks with native code, so a first translation leaves every existing link
            // as valid as it was; a translation again (the block's assumptions changed) may not (a flush bumped it).
            if (++b.jit_compiles > 1) drop_links(b);
        }
        // The last native exit's slot learns where it led, when a chain there could be valid (same snapshot).
        if (pending_slot_) {
            // Only the block the exit itself led to: an interrupt, a fault or a fallback step in between runs another.
            if (b.native && b.jit_eip == r.entry_eip && r.entry_eip == pending_target_ && b.jit_dirty == pending_from_->jit_dirty &&
                same_segments(b, *pending_from_) && b.jit_eip != idle_[0] && b.jit_eip != idle_[1]) {
                set_link(*pending_slot_, &b);
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
            drop_links(b);
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
    JitCtx& c = *ctx_;
    c.block = r.b;
    c.base = r.start + r.extra;
    c.limit = r.limit;
    c.devices = mem.device_events();
    c.helpers = 0;
    c.exit = 0;
    c.ended = 0;
    c.end_cycles = 0;
    c.insns_done = 0;
    c.last_slot = nullptr;
    c.last_target = 0;
    c.penalties = 0;
    c.stores = 0;
    c.mispredicts = 0;
    c.start_eip = r.entry_eip;
    c.chain_extra = 4096;
    c.inline_chains = 0;
    c.inline_mispredicts = 0;
    c.logging = 0;
    c.log_count = 0;
    c.stop_why = 0;
    c.probe = 0;
    c.probe_eip = 0xFFFFFFFFu;
    c.probe_misses = 0;
    c.budget_end = c.base + static_cast<u64>(chain_budget_) * kBudgetCycles;
    // A spin probe with stores or mispredicts, armed by an earlier pass at this EIP: one pass with the stores logged
    // and every transition through the engine (a chain limit of 0: the generated code never commits a chain), so the
    // predictors and the return stack are recorded as exit_cost changes them. The run starts at the probe's EIP.
    if (probe_eip_ && probe_eip_ == r.entry_eip && chain_budget_ >= 64 && r.extra == 0) {
        c.logging = 1;
        probing_ = true;
        probed_exits_.clear();
        rsb_at_probe_ = rsb_;
        rsb_top_at_probe_ = rsb_top_;
        arm_probe(c, r.entry_eip, r.start, 0);
    } else if (chain_budget_ >= 64 && r.extra == 0 && known_spin(r.entry_eip)) {
        // A known spin loop: probed from its start, so the run ends when it is back where it began.
        c.budget_end = c.base + 256 * kBudgetCycles;
        c.chain_extra = 0;
        arm_probe(c, r.entry_eip, r.start, 0);
        ++stats_.spin_short_runs;
    }
    open_chain_limit(c);
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
    stats_.block_insns += total;
    ++stats_.native_runs;
    if ((stats_.native_runs & 0xFFFF) == 0) not_fixed_.fill(0);  // a loop that warmed up gets another chance
    stats_.native_insns += total - c.helpers;
    stats_.native_helpers += c.helpers;
    stats_.chains += c.inline_chains;
    stats_.whole_runs += c.inline_chains;
    stats_.mispredicts += c.inline_mispredicts;
    stats_.exit_cycles += static_cast<u64>(c.inline_mispredicts) * tp::kBlockMispredict;
    const u64 left = mem.take_penalty();
    c.base += left;
    stats_.penalty_cycles += c.penalties + left;
    ++stats_.run_end[kind == kJitEnd ? (c.stop_why & 15u) : kind == kJitFault ? 17u : 16u];
    if (c.logging) probe_eip_ = 0;  // one logging probe a candidate; a later pass that qualifies arms another
    if (kind == kJitEnd) {
        if (c.ended) {
            cpu_.set_cycles_only(c.end_cycles);  // the exit was costed where the chain stopped
            pending_slot_ = c.last_slot;
            pending_target_ = c.last_target;
            pending_from_ = &b;
            if (c.probe == 2) spin_skip(c, total - c.probe_insns);
        } else {
            const u32 exit = exit_cost(b, b.jit_eip);
            stats_.exit_cycles += exit;
            cpu_.set_cycles_only(c.base + b.cycles + exit);
        }
    } else if (kind == kJitExit && c.exit == 2) {
        // The interpreter finishes the block from the next instruction, with the penalties so far.
        cpu_.set_cycles_only(c.base + b.after[i]);
        Run rest{&b, c.base, b.jit_eip, 0, c.limit};
        ++stats_.native_handoffs;
        unsigned j = i + 1;
        while (exec(rest, j) == Next::Continue) ++j;
        return static_cast<unsigned>(c.insns_done) + j + 1;
    } else {
        cpu_.set_cycles_only(c.base + b.after[i] + (kind == kJitFault ? tp::kInterruptDelivery : 0));
    }
    // The schedules' share of the run (statistics): what the clock moved less the penalties and mispredicts.
    stats_.schedule_cycles += cpu_.cycles() - r.start - (c.penalties + r.extra + left) - c.mispredicts * tp::kBlockMispredict -
                              (kind == kJitFault ? tp::kInterruptDelivery : 0);
    return static_cast<unsigned>(total);
}

// The generated code's chain limit for the run from here: the limit or the budget's end, whichever comes first (0 for
// a logging probe, whose every transition is the engine's); then 0 if a device wrote into code meanwhile.
void BlockEngine::open_chain_limit(JitCtx& c) {
    const u64 v = c.logging ? 0 : std::min(c.limit, c.budget_end);
    std::atomic_ref<u64>(c.chain_limit).store(v, std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (bus_.code_writes_pending()) std::atomic_ref<u64>(c.chain_limit).store(0, std::memory_order_seq_cst);
}

// The end of a native block whose last instruction left normally, when its chain cell did not take the exit. Its exit is
// costed as `exec` costs it. Then the run goes on in the next block's native code only where the dispatcher would have
// done nothing in between: the next deadline and the stop cycle lie past the next block even with the slowest code
// fetch (so no event fires and no interrupt becomes deliverable — native code changes neither IF nor the PIC), no
// device wrote into code, the fetch translation hits the TLB on the page the block came from (find() would walk
// nothing), it is not the idle loop, and the link names that block's current translation with the same segments. Then
// its run begins as `begin` would. Past the chain budget the run goes on (a little) only to see whether it comes back
// to the state it was in there (the spin probe).
void* BlockEngine::chain(JitCtx& c, Block::Link* slot) {
    Block& a = *slot->from;
    c.block = &a;
    const CpuState& s = cpu_.state();
    {
        // Statistics: why the generated code's own chain did not take this exit.
        const Block* t = slot->to;
        const u32 lin = s.seg[kCs].base + s.eip;
        const Memory::TlbEntry& e = cpu_.memory().tlb()[(lin >> 12) & (Memory::kTlbSize - 1)];
        const u64 end_max = c.base + a.cycles + tp::kBlockMispredict;
        unsigned why = 7;
        if (a.branch != BranchKind::None && a.branch != BranchKind::Conditional && a.branch != BranchKind::Direct && a.branch != BranchKind::Call) why = 6;
        else if (c.probe == 1 && s.eip == c.probe_eip) why = 8;
        else if (!t || !slot->patched || slot->epoch != link_epoch_ || !t->entry.body || t->entry.jit_eip != s.eip) why = 0;
        else if (end_max + t->entry.cycles >= c.limit) why = 2;
        else if (end_max + t->entry.cycles >= c.budget_end) why = 1;
        else if (c.chain_limit == 0) why = 3;
        else if (e.tag != (lin >> 12) || e.host != t->entry.host_page) why = 4;
        else {
            const u32* mru = cpu_.cache_timing().l1i_mru();
            const unsigned sets = cpu_.cache_timing().l1i_sets();
            const u32 l = t->entry.fetch_line;
            for (unsigned k = 0; k < t->entry.fetch_lines; ++k)
                if (mru[(l - 1 + k) & (sets - 1)] != l + k) why = 5;
        }
        ++stats_.slow_chain[why];
    }
    const u32 exit = exit_cost(a, a.jit_eip);
    stats_.schedule_cycles += a.cycles;
    stats_.exit_cycles += exit;
    const u64 end = c.base + a.cycles + exit;
    c.end_cycles = end;
    c.ended = 1;
    c.last_slot = slot;
    c.last_target = s.eip;
    if (exit) ++c.mispredicts;
    // Back at the probe's EIP: a pass of a loop when the whole state is as it was there (the spin skip then checks the
    // rest); a nested loop's inner block recurs before the outer period ends, so a few other states are let by.
    if (c.probe == 1 && s.eip == c.probe_eip) {
        if (std::memcmp(&s, &spin_start_, sizeof s) == 0) {
            c.probe = 2;
            return c.stop_why = 7, nullptr;
        }
        if (++c.probe_misses >= 16) {
            ++stats_.spin_refused[4];
            remember_not_fixed(c.probe_eip);
            return c.stop_why = 8, nullptr;
        }
    }
    Block* b = slot->to;
    if (!b || slot->epoch != link_epoch_ || !b->native || b->jit_eip != s.eip) {
        // No link, a stale one, or one to where the exit went last time (a return, a jump table): the block the
        // dispatcher would run next, from the fast cache, under the conditions a link to it is learned (run()), and
        // the slot learns it now instead of after a trip through the dispatcher.
        b = nullptr;
        const u32 at = s.seg[kCs].base + s.eip;
        const FastEntry& f = fast_[fast_index(at, a.mode)];
        if (f.epoch == epoch_ && f.lin == at && f.mode == a.mode) {
            Block* n = f.block;
            if (n->native && n->jit_eip == s.eip && n->jit_dirty == a.jit_dirty && same_segments(*n, a) && s.eip != idle_[0] && s.eip != idle_[1] &&
                s.eip + n->bytes - 1u >= s.eip && s.eip + n->bytes - 1u <= s.seg[kCs].limit) {
                b = n;
                set_link(*slot, n);
                slot->epoch = link_epoch_;
                ++stats_.relinks;
            }
        }
        if (!b) return c.stop_why = 1, nullptr;
    }
    // The budget: a logging probe counts its transitions; otherwise the first block past the budget's end arms the
    // probe here (not where one already failed) and the run goes on a little, else it stops.
    if (c.logging) {
        if (c.chain_extra == 0) return c.stop_why = 2, nullptr;
        --c.chain_extra;
    } else if (end + b->cycles >= c.budget_end) {
        if (c.probe || chain_budget_ == 0 || not_a_fixed_point(s.eip)) return c.stop_why = 2, nullptr;
        arm_probe(c, s.eip, end, c.insns_done + a.n);
        c.budget_end = end + c.chain_extra * kBudgetCycles;
        c.chain_extra = 0;
    }
    if (b->jit_eip != s.eip || s.eip == idle_[0] || s.eip == idle_[1]) return c.stop_why = 3, nullptr;
    constexpr u64 kSlowestFetch = tp::kL2Latency + tp::kDramLatency;
    if (end + b->cycles + kSlowestFetch >= c.limit) return c.stop_why = 4, nullptr;
    if (bus_.code_writes_pending() || cpu_.memory().penalty_pending()) return c.stop_why = 5, nullptr;
    const u32 lin = s.seg[kCs].base + s.eip;
    const Memory::TlbEntry& e = cpu_.memory().tlb()[(lin >> 12) & (Memory::kTlbSize - 1)];
    if (e.tag != (lin >> 12) || !e.host || bus_.ram_page(e.host) != b->ram_page || (lin & 0xFFFu) != b->page_off) return c.stop_why = 6, nullptr;
    // A link the cell could not hold when it was learned (its page not in the TLB then) is patched now it is.
    if (!slot->patched && slot->to == b) Jit::patch_link(a, static_cast<unsigned>(slot - a.links.data()), *b, cpu_);
    // A return or an indirect jump: the generated code chains there itself next time (the target cache).
    if ((a.branch == BranchKind::Return || a.branch == BranchKind::Indirect || a.branch == BranchKind::CallIndirect) && b->entry.fetch_lines <= 2 && b->jit_key) {
        const unsigned i = JitCtx::target_index(s.eip);
        JitCtx::Target& t = c.targets[i];
        if (t.block && t.block != b) {
            auto& v = t.block->target_slots;
            v.erase(std::remove(v.begin(), v.end(), i), v.end());
        }
        const Timing& tm = cpu_.cache_timing();
        const i64 l1i = reinterpret_cast<const u8*>(tm.l1i_mru()) - reinterpret_cast<const u8*>(tm.l1d_mru());
        const u32 l = b->entry.fetch_line;
        t = JitCtx::Target{s.eip, b->jit_key, b->cycles, ((lin >> 12) & (Memory::kTlbSize - 1)) * 32, lin >> 12, e.phys_page,
                           {l, b->entry.fetch_lines == 2 ? l + 1 : 0u},
                           {l1i + static_cast<i64>(((l - 1) & (tm.l1i_sets() - 1)) * 4), l1i + static_cast<i64>((l & (tm.l1i_sets() - 1)) * 4)},
                           b->entry.body, b};
        if (std::find(b->target_slots.begin(), b->target_slots.end(), i) == b->target_slots.end()) b->target_slots.push_back(i);
    }
    ++b->entry.runs;
    ++stats_.whole_runs;
    ++stats_.chains;
    c.insns_done += a.n;
    const PhysAddr code = static_cast<PhysAddr>(b->ram_page) * 4096u + b->page_off;
    const u64 fetch = cpu_.cache_timing().code_fetch(code, b->bytes);
    c.penalties += fetch;
    c.base = end + fetch;
    c.block = b;
    c.ended = 0;
    running_ = b;
    open_chain_limit(c);
    cpu_.set_cycles_only(c.base);
    return b->entry.body;
}

void BlockEngine::disable_jit() { jit_.reset(); }

void BlockEngine::drop_native() {
    ++link_epoch_;
    pending_slot_ = nullptr;
    // The code (and its chain cells) is gone with the buffer: every link with it, no unpatching.
    auto reset = [](Block& b) {
        b.native = b.entry.body = nullptr;
        for (Block::Link& l : b.links) l = Block::Link{nullptr, 0, &b, {}, {}, false};
        b.incoming.clear();
        b.target_slots.clear();
    };
    for (JitCtx::Target& t : ctx_->targets) t = JitCtx::Target{};
    for (auto& [k, b] : blocks_) reset(*b);
    for (auto& b : graveyard_) reset(*b);
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
void BlockEngine::spin_skip(const JitCtx& c, u64 instructions) {
    const bool probe = c.logging != 0;
    if (c.helpers != c.probe_helpers) { ++stats_.spin_refused[0]; return; }
    if (c.penalties != c.probe_penalties) { ++stats_.spin_refused[1]; return; }
    if (c.mispredicts != c.probe_mispredicts && !probe) {
        // Mispredicts: a periodic pattern leaves each predictor as it found it after a pass; a probe checks that.
        ++stats_.spin_refused[2];
        probe_eip_ = c.probe_eip;
        return;
    }
    if (probe && !predictors_undone()) { ++stats_.spin_refused[2]; return; }
    if (instructions == 0) return;
    if (c.stores) {
        // Stores: a pass that leaves memory as it found it is still a fixed point. The next run (it starts here) is a
        // logging probe, which checks every location its stores touched holds its first replaced bytes again.
        if (!probe) {
            probe_eip_ = c.probe_eip;
            return;
        }
        if (c.log_count > 256) { ++stats_.spin_refused[5]; return; }
        if (!stores_undone(c)) { ++stats_.spin_refused[6]; return; }
    }
    const u64 now = cpu_.cycles();
    const u64 period = now - c.probe_cycle;
    if (period == 0 || now >= c.limit) return;
    const u64 passes = (c.limit - now) / period;
    if (passes < 2) return;
    const u64 k = passes - 1;
    cpu_.set_cycles_only(now + k * period);
    cpu_.add_instructions(k * instructions);
    remember_spin(c.probe_eip);
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
    // (The last target matters only to an indirect exit's prediction; a direct one's chain cells leave it.)
    for (const ProbedExit& p : probed_exits_) {
        const bool indirect = p.block->branch == BranchKind::Indirect || p.block->branch == BranchKind::CallIndirect;
        if (p.block->exit_counter != p.counter || p.block->exit_seen != p.seen || (indirect && p.block->exit_target != p.target)) return false;
    }
    return rsb_ == rsb_at_probe_ && rsb_top_ == rsb_top_at_probe_;
}

}  // namespace mv::p6

namespace mv::p6 {

// A countdown loop (`dec ecx; jnz self`: the kernel's stall loop) skipped arithmetically. While its exit stays taken
// and predicted (the two-bit counter saturated), its code line the most recent of its set and nothing is pending, each
// pass costs the schedule and nothing else and changes only the register and the lazy flags; the next pass the run
// executes rewrites the flags as they would be (a DEC's CF is carried unchanged through every pass). So whole passes
// move the clock, the instruction count and the register on, leaving the register at least one and a pass before
// the limit, so the run then executes as it would have to the limit's boundary. (Called before the run begins.)
void BlockEngine::countdown_skip(Block& b, u64 limit) {
    CpuState& s = cpu_.state();
    const u32 r = s.gpr[b.countdown];
    if (r < 3 || b.exit_counter != 3 || !b.exit_seen || b.cycles == 0 || !(b.mode & 1) || (s.flags & fl::TF)) return;
    if (cpu_.memory().penalty_pending() || bus_.code_writes_pending() || s.inhibit_interrupts) return;
    const Timing& t = cpu_.cache_timing();
    const u32* mru = t.l1i_mru();
    const unsigned sets = t.l1i_sets();
    const u32 l = b.entry.fetch_line;
    for (unsigned k = 0; k < b.entry.fetch_lines; ++k)
        if (mru[(l - 1 + k) & (sets - 1)] != l + k) return;
    const u64 now = cpu_.cycles();
    if (now >= limit) return;
    const u64 passes = (limit - now) / b.cycles;
    if (passes < 3) return;
    const u64 k = std::min<u64>(r - 2, passes - 2);
    s.gpr[b.countdown] = r - static_cast<u32>(k);
    cpu_.set_cycles_only(now + k * b.cycles);
    cpu_.add_instructions(2 * k);
    b.entry.runs += k;
    stats_.schedule_cycles += k * b.cycles;
    stats_.block_insns += 2 * k;
    ++stats_.countdown_skips;
    stats_.countdown_passes += k;
}

void BlockEngine::arm_probe(JitCtx& c, u32 eip, u64 at, u64 insns) {
    std::memcpy(&spin_start_, &cpu_.state(), sizeof spin_start_);  // bytes, padding too: compared as bytes
    c.probe = 1;
    c.probe_misses = 0;
    c.probe_eip = eip;
    c.probe_cycle = at;
    c.probe_insns = insns;
    c.probe_helpers = c.helpers;
    c.probe_penalties = c.penalties;
    c.probe_mispredicts = c.mispredicts;
    c.stores = 0;
}

}  // namespace mv::p6
