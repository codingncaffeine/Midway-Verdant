#include "cpu/p6/p6_backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>

#include "core/machine.h"
#include "cpu/p6/block.h"
#include "cpu/p6/jit/jit.h"
#include "cpu/p6/oracle.h"
#include "debug/trace.h"

namespace mv {

namespace {
// Instructions per turn of the machine lock: enough to amortise the lock, short enough that the io thread
// (milestones, resets, the window) is never kept waiting for long.
constexpr unsigned kBatchInstructions = 20000;
}  // namespace

P6Backend::P6Backend(bool with_oracle) : with_oracle_(with_oracle) {}

P6Backend::~P6Backend() {
    stop();
    join();
    trace::set_clock_source(nullptr);
}

void P6Backend::fail(Stop why, std::string detail) {
    stop_reason_ = why;
    stop_detail_ = std::move(detail);
    stop_requested_ = true;
}

bool P6Backend::init(Machine& machine, std::string& error) {
    machine_ = &machine;
    cpu_ = std::make_unique<p6::Cpu>(machine.bus());
    cpu_->memory().set_rom_range(Flash::kBase, Flash::kWindow);
    cpu_->memory().set_dirty_tracking(dirty_tracking_, selective_requested_);
    cpu_->memory().set_device_hook([this] { sync_time(); });
    cpu_->set_interrupt_source([this] { return machine_->pic().pending_vector(); }, [this] { return machine_->pic().acknowledge(); });
    // Trace lines carry the interpreter's own clock, exact to the instruction.
    trace::set_clock_source(cpu_->cycles_pointer());
    if (const char* s = std::getenv("MV_P6_STOP_CYCLES"); s && *s) stop_at_cycles_ = std::strtoull(s, nullptr, 0);
    if (const char* s = std::getenv("MV_P6_DELIVERY_LOG"); s && *s) delivery_log_ = std::fopen(s, "w");
    if (const char* s = std::getenv("MV_IDLE_SKIP"); s && *s == '0') idle_skip_ = false;
    if (with_oracle_) {
        oracle_ = std::make_unique<p6::Oracle>();
        if (!oracle_->init(machine, *cpu_, error)) return false;
        cpu_->set_single_iteration(true);
    }
    // Block timing and the block engine (D26) unless the oracle (it compares per instruction) or
    // MV_P6_TIMING=pipeline (the per-instruction model, the block model's calibration reference) asks otherwise.
    const char* timing = std::getenv("MV_P6_TIMING");
    if (!oracle_ && !(timing && std::string_view(timing) == "pipeline")) {
        cpu_->set_block_timing(true);
        blocks_ = std::make_unique<p6::BlockEngine>(*cpu_, machine.bus());
    }
    reset();
    return true;
}

void P6Backend::reset() {
    cpu_->reset(cpuid_signature_);
    cpu_->set_cycles(cycles_.load());
    if (oracle_) oracle_->reset();
    if (blocks_) blocks_->clear();
    trace::log(trace::Cat::Cpu, "reset: p6 interpreter, real mode, CS f000:fff0, signature {:08x}", cpuid_signature_);
}

void P6Backend::start() {
    stop_requested_ = false;
    thread_ = std::thread([this] { thread_main(); });
}

void P6Backend::stop() {
    stop_requested_ = true;
    if (machine_) machine_->wake();
}

void P6Backend::join() {
    if (thread_.joinable()) thread_.join();
}

void P6Backend::kick() {
    ++stats_.kicks;
    if (machine_) machine_->wake_cpu();
}

// Bring the timeline up to the core's clock (called before every device access and when a deadline passes).
void P6Backend::sync_time() {
    const Cycles now = cpu_->cycles();
    cycles_.store(now, std::memory_order_relaxed);
    machine_->scheduler().advance_to(now);
}

void P6Backend::run_batch() {
    if (blocks_) {
        run_batch_blocks();
        return;
    }
    p6::Cpu& cpu = *cpu_;
    Scheduler& sched = machine_->scheduler();
    for (unsigned i = 0; i < kBatchInstructions && !stop_requested_; ++i) {
        // Events due at or before now fire first, so an interrupt they raise is seen at this boundary.
        if (cpu.cycles() >= sched.next_deadline()) sync_time();
        if (cpu.take_interrupt()) {
            ++stats_.irq_injections;
            ++stats_.by_vector[cpu.last_vector()];
            fold_delivery();
        }
        if (idle_eips_[0]) idle_turn();
        if (cpu.state().halted) {
            ++stats_.hlt_exits;
            // Sleep the clock to the next event; with no event and no interrupt the guest is dead.
            if (!(cpu.state().flags & p6::fl::IF)) {
                fail(Stop::Requested, "HLT with interrupts disabled: the guest halted for good");
                return;
            }
            if (!sched.has_pending()) {
                // Nothing on the timeline: wait for the io thread (a reset request, a stop).
                machine_->wait_for_wake([this] { return stop_requested_.load(); });
                return;
            }
            cpu.set_cycles(sched.next_deadline());
            sync_time();
            continue;
        }
        if (oracle_) {
            if (!oracle_->step(cpu)) {
                fail(Stop::InternalError, oracle_->last_error());
                return;
            }
        } else {
            cpu.step();
        }
        if (cpu.triple_fault()) {
            fail(Stop::TripleFault, std::format("triple fault at {}", cpu.describe()));
            return;
        }
        if (cpu.last_faulted()) fold_delivery();
        if (cpu.cycles() >= stop_at_cycles_) {
            fail(Stop::Requested, std::format("the stop cycle {} was reached", stop_at_cycles_));
            return;
        }
    }
    cycles_.store(cpu.cycles(), std::memory_order_relaxed);
}

// The block engine's loop (D26). Between blocks: device writes into code, due events, an interrupt, a halt. A block
// returns at the first instruction boundary at or past the next deadline (or the stop cycle), so events and
// interrupts land on the boundaries the per-instruction loop would give them.
void P6Backend::run_batch_blocks() {
    p6::Cpu& cpu = *cpu_;
    p6::BlockEngine& engine = *blocks_;
    Scheduler& sched = machine_->scheduler();
    const u64 budget_end = cpu.instructions() + kBatchInstructions;
    while (cpu.instructions() < budget_end && !stop_requested_) {
        engine.apply_device_writes();
        if (cpu.cycles() >= stop_at_cycles_) {
            fail(Stop::Requested, std::format("the stop cycle {} was reached", stop_at_cycles_));
            return;
        }
        between_instructions();
        if (idle_eips_[0]) idle_turn();
        if (cpu.state().halted) {
            ++stats_.hlt_exits;
            if (!(cpu.state().flags & p6::fl::IF)) {
                fail(Stop::Requested, "HLT with interrupts disabled: the guest halted for good");
                return;
            }
            if (!sched.has_pending()) {
                machine_->wait_for_wake([this] { return stop_requested_.load(); });
                return;
            }
            cpu.set_cycles(sched.next_deadline());
            sync_time();
            continue;
        }
        p6::Block* b = engine.find();
        if (!b) {
            engine.note_fallback();
            cpu.step();
            if (cpu.last_faulted()) fold_delivery();
        } else {
            // After STI or a load of SS one instruction runs before interrupts are looked at again.
            const u64 limit = cpu.state().inhibit_interrupts ? cpu.cycles() : std::min<u64>(sched.next_deadline(), stop_at_cycles_);
                        engine.run(*b, limit);
            if (cpu.last_faulted()) fold_delivery();
        }
        if (cpu.triple_fault()) {
            fail(Stop::TripleFault, std::format("triple fault at {}", cpu.describe()));
            return;
        }
    }
    cycles_.store(cpu.cycles(), std::memory_order_relaxed);
}

bool P6Backend::between_instructions() {
    p6::Cpu& cpu = *cpu_;
    if (cpu.cycles() >= machine_->scheduler().next_deadline()) sync_time();
    // The PIC's INTR line follows its every change (pending_vector() >= 0): without it there is nothing to take.
    if (!machine_->pic().intr_asserted() || !cpu.take_interrupt()) return false;
    ++stats_.irq_injections;
    ++stats_.by_vector[cpu.last_vector()];
    fold_delivery();
    return true;
}

// The loops the kernel spins in when it has nothing to run (the KVM backend's D22 finder, same shape): STI, two NOPs,
// CLI, then a compare of the DPC list head with itself, in the kernel image identity-mapped at 0x80000000. The check
// sits on the second NOP, where interrupts are enabled and the loop has not yet looked for work.
void P6Backend::find_idle_loop() {
    ++idle_scans_;
    static constexpr u8 kShape[] = {0xFB, 0x90, 0x90, 0xFA, 0x3B, 0x6D, 0x00, 0x74};
    constexpr u32 kFrom = 0x10000, kTo = 0x80000;
    const Bus& bus = machine_->bus();
    if (bus.ram_size() < kTo) return;
    const u8* ram = bus.ram();
    std::array<u32, 2> found{};
    u32 matches = 0;
    for (u32 a = kFrom; a + sizeof kShape <= kTo; ++a) {
        if (ram[a] != 0xFB || std::memcmp(ram + a, kShape, sizeof kShape) != 0) continue;
        if (matches < found.size()) found[matches] = 0x80000000u + a + 2;
        ++matches;
    }
    if (matches == 0 || matches > found.size()) return;
    idle_eips_ = found;
    if (blocks_) blocks_->set_idle_eips(found[0], found[1]);
    trace::log(trace::Cat::Cpu, "the kernel idle loops are at {:08x} {:08x}: their idle turns halt", found[0], found[1]);
}

// A pass through the idle loop with no interrupt delivered since the previous pass looked for work and found none;
// on one CPU only an interrupt can bring work, so the core halts as at HLT (the guest cannot tell).
bool P6Backend::idle_turn() {
    p6::CpuState& s = cpu_->state();
    if ((s.eip != idle_eips_[0] && s.eip != idle_eips_[1]) || s.seg[p6::kCs].base != 0) return false;
    ++idle_hits_;
    const bool quiet = stats_.irq_injections == idle_deliveries_at_hit_;
    idle_deliveries_at_hit_ = stats_.irq_injections;
    if (!quiet || !(s.flags & p6::fl::IF)) return false;
    ++idle_halts_;
    s.halted = true;
    return true;
}

void P6Backend::thread_main() {
    while (!stop_requested_) {
        try {
            {
                std::unique_lock lock(machine_->lock());
                // Frame stepping (libretro) and real-time pacing: the CPU holds between batches (batches end on
                // instruction counts, so a run stays deterministic; held, its clock stands still).
                if (!machine_->pace_cpu(lock, cpu_->cycles(), stop_requested_)) continue;
                if (idle_skip_ && !idle_eips_[0] && idle_scans_ < 4000 && (stats_.exits % 64) == 0) find_idle_loop();
                run_batch();
                ++stats_.exits;
            }
            // A mutex hands over to nobody in particular: give the io thread (milestones, resets, the window) a
            // real chance at the lock between batches.
            std::this_thread::yield();
        } catch (const p6::Unimplemented& u) {
            fail(Stop::InternalError, std::format("unimplemented: {} at {:04x}:{:08x} [{}] {}", u.what, cpu_->last_cs(), cpu_->last_eip(),
                                                  cpu_->last_bytes(), cpu_->describe()));
        }
    }
    machine_->wake();
}

void P6Backend::fold_delivery() {
    const p6::CpuState& s = cpu_->state();
    auto mix = [this](u64 v) {
        delivery_hash_ ^= v;
        delivery_hash_ *= 0x100000001b3ull;
    };
    mix(cpu_->cycles());
    mix(s.eip);
    mix(s.eflags());
    for (u32 r : s.gpr) mix(r);
    ++deliveries_;
    if (delivery_log_)
        std::fprintf(delivery_log_, "%llu %llu %08x %016llx\n", static_cast<unsigned long long>(cpu_->cycles()), static_cast<unsigned long long>(cpu_->instructions()), s.eip,
                     static_cast<unsigned long long>(delivery_hash_));
}

CpuBackend::Stats P6Backend::stats() const {
    Stats s = stats_;
    s.instructions = cpu_ ? cpu_->instructions() : 0;
    s.cycles = cycles_.load();
    return s;
}

bool P6Backend::read_regs(Regs& out) {
    if (!cpu_) return false;
    const p6::CpuState& s = cpu_->state();
    out.eax = s.gpr[p6::kEax];
    out.ebx = s.gpr[p6::kEbx];
    out.ecx = s.gpr[p6::kEcx];
    out.edx = s.gpr[p6::kEdx];
    out.esi = s.gpr[p6::kEsi];
    out.edi = s.gpr[p6::kEdi];
    out.ebp = s.gpr[p6::kEbp];
    out.esp = s.gpr[p6::kEsp];
    out.eip = s.eip;
    out.eflags = s.eflags();
    out.cr0 = s.cr0;
    out.cr3 = s.cr3;
    out.cr4 = s.cr4;
    out.cs = s.seg[p6::kCs].selector;
    out.ds = s.seg[p6::kDs].selector;
    out.es = s.seg[p6::kEs].selector;
    out.fs = s.seg[p6::kFs].selector;
    out.gs = s.seg[p6::kGs].selector;
    out.ss = s.seg[p6::kSs].selector;
    out.tsc = cpu_->cycles() + s.tsc;
    return true;
}

void P6Backend::take_dirty_pages(std::vector<u32>& pages) {
    if (cpu_) cpu_->memory().take_dirty_pages(pages);
}

std::string P6Backend::report() const {
    std::string r = std::format("cpu p6 memory: {} TLB misses, {} page walks", cpu_ ? cpu_->memory().tlb_misses() : 0, cpu_ ? cpu_->memory().page_walks() : 0);
    if (cpu_) {
        const p6::TimingStats& t = cpu_->timing().stats();
        const double ipc = t.cycles ? static_cast<double>(t.instructions) / static_cast<double>(t.cycles) : 0.0;
        r += std::format("\ncpu p6 pipeline: {:.3f} IPC, {} uops; L1I misses {}, L1D misses {}, L2 misses {}; branches {} mispredicted {}; device loads {} stores {}, port I/O {}; stalls: dependency {}, rob {}, fetch {}",
                         ipc, t.uops, t.l1i_misses, t.l1d_misses, t.l2_misses, t.branches, t.mispredicts, t.device_loads, t.device_stores, t.port_ios,
                         t.stall_dependency, t.stall_rob, t.stall_fetch);
    }
    if (blocks_) {
        const p6::BlockStats& b = blocks_->stats();
        r += std::format("\ncpu p6 blocks: {} built, {} live, {} killed ({} page writes, {} by devices); runs {} whole; {} fallback steps; {} instructions in blocks; {} exit mispredicts; cycles: schedules {}, penalties {}, exits {}, REP {}",
                         b.built, blocks_->live_blocks(), b.killed, b.pages_written, b.device_page_writes, b.whole_runs, b.fallback_steps,
                         b.block_insns, b.mispredicts, b.schedule_cycles, b.penalty_cycles, b.exit_cycles, b.rep_cycles);
        if (p6::Jit* j = blocks_->jit())
            r += std::format("\ncpu p6 recompiler: {} blocks translated ({} KiB, {} flushes); {} native runs, {} native instructions, {} by the interpreter inside them; {} hand-offs, {} runs refused (assumptions changed); {} blocks entered by a chain; {} spin skips covering {} cycles",
                             j->compiled(), j->code_bytes() / 1024, j->flushes(), b.native_runs, b.native_insns, b.native_helpers, b.native_handoffs, b.native_ineligible, b.chains,
                             b.spin_skips, b.spin_cycles);
        {
            std::vector<std::pair<u64, unsigned>> ops;
            for (unsigned k = 0; k < 256; ++k)
                if (k < 256 && b.helper_ops[k]) ops.emplace_back(b.helper_ops[k], k);
            std::sort(ops.rbegin(), ops.rend());
            r += "\ncpu p6 interpreted inside native code:";
            for (std::size_t k = 0; k < ops.size() && k < 16; ++k) r += std::format(" {} {}", p6::op_name(static_cast<p6::Op>(ops[k].second)), ops[k].first);
            std::vector<std::pair<u64, unsigned>> x87;
            for (unsigned k = 0; k < 256; ++k)
                if (b.helper_ops[256 + k]) x87.emplace_back(b.helper_ops[256 + k], k);
            std::sort(x87.rbegin(), x87.rend());
            r += "\ncpu p6 x87 forms inside native code (escape/mem/reg):";
            for (std::size_t k = 0; k < x87.size() && k < 24; ++k)
                r += std::format(" {:02x}{}{:x} {}", 0xD8 + x87[k].second / 32, (x87[k].second & 16) ? "m" : "r", x87[k].second & 15, x87[k].first);
            std::vector<std::pair<u64, unsigned>> sse;
            for (unsigned k = 0; k < 2048; ++k)
                if (b.helper_ops[512 + k]) sse.emplace_back(b.helper_ops[512 + k], k);
            std::sort(sse.rbegin(), sse.rend());
            static constexpr const char* kPfx[4] = {"", "66.", "F3.", "F2."};
            r += "\ncpu p6 SSE/MMX forms inside native code (prefix.0F opcode, m/r):";
            for (std::size_t k = 0; k < sse.size() && k < 24; ++k)
                r += std::format(" {}{:02x}{} {}", kPfx[(sse[k].second / 2) % 4], (sse[k].second / 8) % 256, (sse[k].second & 1) ? "m" : "r", sse[k].first);
        }
        r += "\ncpu p6 hottest blocks (address x instructions: cycles * runs):" + blocks_->hottest(12);
    }
    if (idle_eips_[0]) r += std::format("\ncpu p6: the kernel idle loops at {:08x} {:08x}: {} passes, {} turned into halts", idle_eips_[0], idle_eips_[1], idle_hits_, idle_halts_);
    else if (idle_skip_) r += std::format("\ncpu p6: the kernel idle loop was not found ({} scans): it spins", idle_scans_);
    r += std::format("\ncpu p6 delivery hash {:016x} over {} deliveries; stopped at cycle {}, {} instructions", delivery_hash_, deliveries_, cpu_ ? cpu_->cycles() : 0,
                     cpu_ ? cpu_->instructions() : 0);
    if (oracle_) r += "\n" + oracle_->summary();
    return r;
}

std::unique_ptr<CpuBackend> make_p6_backend(bool with_oracle) { return std::make_unique<P6Backend>(with_oracle); }

}  // namespace mv
