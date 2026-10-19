#include "cpu/p6/block.h"

#include "cpu/p6/timing.h"

#include <algorithm>
#include <format>

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
                b->insns.push_back(in);
                b->rep = true;
                at += in.len;
            }
            break;
        }
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
    std::sort(all.begin(), all.end(), [](const Block* a, const Block* b) { return a->runs * a->cycles > b->runs * b->cycles; });
    std::string s;
    for (unsigned i = 0; i < count && i < all.size(); ++i)
        s += std::format(" {:08x}x{}:{}c*{}", all[i]->lin, all[i]->n, all[i]->cycles, all[i]->runs);
    return s;
}

void BlockEngine::kill_page(u32 ram_page) {
    if (ram_page >= page_blocks_.size()) return;
    auto& list = page_blocks_[ram_page];
    for (Block* b : list) {
        if (b == running_) running_dead_ = true;
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
    ++b.runs;
    const PhysAddr code = static_cast<PhysAddr>(b.ram_page) * 4096u + b.page_off;
    const Run r{&b, cpu_.cycles(), cpu_.state().eip, cpu_.cache_timing().code_fetch(code, b.bytes), limit};
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
    ++stats_.whole_runs;
    unsigned i = 0;
    while (exec(r, i) == Next::Continue) ++i;
    return i + 1;
}

}  // namespace mv::p6
