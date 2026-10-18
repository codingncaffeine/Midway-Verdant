#include "cpu/p6/p6_backend.h"

#include <format>

#include "core/machine.h"
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
    cpu_->memory().set_dirty_tracking(dirty_tracking_);
    cpu_->memory().set_device_hook([this] { sync_time(); });
    cpu_->set_interrupt_source([this] { return machine_->pic().pending_vector(); }, [this] { return machine_->pic().acknowledge(); });
    // Trace lines carry the interpreter's own clock, exact to the instruction.
    trace::set_clock_source(cpu_->cycles_pointer());
    if (with_oracle_) {
        oracle_ = std::make_unique<p6::Oracle>();
        if (!oracle_->init(machine, *cpu_, error)) return false;
        cpu_->set_single_iteration(true);
    }
    reset();
    return true;
}

void P6Backend::reset() {
    cpu_->reset(cpuid_signature_);
    cpu_->set_cycles(cycles_.load());
    if (oracle_) oracle_->reset();
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
    if (machine_) machine_->wake();
}

// Bring the timeline up to the core's clock (called before every device access and when a deadline passes).
void P6Backend::sync_time() {
    const Cycles now = cpu_->cycles();
    cycles_.store(now, std::memory_order_relaxed);
    machine_->scheduler().advance_to(now);
}

void P6Backend::run_batch() {
    p6::Cpu& cpu = *cpu_;
    Scheduler& sched = machine_->scheduler();
    for (unsigned i = 0; i < kBatchInstructions && !stop_requested_; ++i) {
        // Events due at or before now fire first, so an interrupt they raise is seen at this boundary.
        if (cpu.cycles() >= sched.next_deadline()) sync_time();
        if (cpu.take_interrupt()) {
            ++stats_.irq_injections;
            ++stats_.by_vector[cpu.last_vector()];
        }
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
    }
    cycles_.store(cpu.cycles(), std::memory_order_relaxed);
}

void P6Backend::thread_main() {
    while (!stop_requested_) {
        try {
            {
                std::lock_guard lock(machine_->lock());
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
    if (oracle_) r += "\n" + oracle_->summary();
    return r;
}

std::unique_ptr<CpuBackend> make_p6_backend(bool with_oracle) { return std::make_unique<P6Backend>(with_oracle); }

}  // namespace mv
