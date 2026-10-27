#pragma once
#if defined(__linux__)
#include <time.h>

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <map>
#include <unordered_map>
#include <vector>

#include "core/guest_clock.h"
#include "cpu/backend.h"

struct kvm_run;
struct kvm_coalesced_mmio_ring;

namespace mv {

// The fast mode on Linux: a KVM virtual machine whose memory slots are the console's RAM (twice: at zero and at
// the GPU-side alias), the aliased flash as a read-only slot, and the last page with the MCPX ROM laid over it
// until the firmware hides it. Everything not backed by a slot exits to the machine's bus. Interrupts come from
// the machine's 8259 pair through KVM_INTERRUPT with the interrupt-window dance; no in-kernel irqchip.
// Guest time is the guest clock (D16, core/guest_clock.h): the machine's io thread reads it live (the vCPU thread's
// CPU time so far) to fire device events while the guest runs, as it read the host clock before, and paces the vCPU.
class KvmBackend : public CpuBackend {
public:
    KvmBackend();
    ~KvmBackend() override;
    std::string_view name() const override { return "kvm"; }

    bool init(Machine& machine, std::string& error) override;
    void reset() override;
    void start() override;
    void stop() override;
    void join() override;
    void kick() override;
    Stop stop_reason() const override { return stop_reason_; }
    std::string stop_detail() const override { return stop_detail_; }
    Stats stats() const override { return stats_; }
    bool read_regs(Regs& out) override;
    void set_dirty_tracking(bool on) override { dirty_tracking_ = on; }
    void take_dirty_pages(std::vector<u32>& pages) override;
    void request_selective_tracking() override { selective_requested_ = true; }
    bool selective_tracking() const override { return selective_; }
    void watch_pages(u32 first, u32 count) override;
    void set_exact_reports(bool on) override { exact_reports_.store(on, std::memory_order_relaxed); }
    void set_break_eips(const std::vector<u32>& eips) override { break_eips_ = eips; }
    bool owns_timeline() const override { return guest_clock_on_; }
    Cycles timeline_cycles() const override;
    double timeline_rate() const override;
    void timeline_advanced() override;
    bool hardware_assisted() const override { return true; }
    std::string clock_name() const override;
    std::string report() const override;
    HostAccounts host_accounts() const override {
        const auto& c = clock_.stats();
        return {c.in_guest_ns, handler_ns_, event_ns_, pause_ns_, busy_guest_ns_, c.out_counted_ns, idle_spin_ns_, c.warped_cycles, idle_halts_, stats_.exits, stats_.mmio_exits};
    }

    // Replace the top page's backing when the MCPX ROM is hidden, and again when a reset shows it.
    void on_rom_hidden();
    void on_rom_shown();

private:
    void thread_main();
    void set_top_page(const void* backing);
    bool set_slot(u32 slot, u64 guest_phys, u64 size, void* host, bool readonly, std::string& error, bool log_dirty = false);
    bool setup_cpuid(std::string& error);
    void handle_io();
    void handle_mmio();
    // The writes KVM's coalesced MMIO ring holds, to the bus in order (with the machine lock held).
    void drain_coalesced_writes();
    void inject_interrupt_if_possible();
    void wait_for_interrupt();
    // The guest clock's helpers: move the guest's TSC by `delta` cycles through its offset (false when the host has
    // no KVM_VCPU_TSC_OFFSET); read and set the TSC; the TSC against the clock at a frame's end (an instrument).
    bool tsc_offset_add(i64 delta);
    // The backstop: a one-shot timer that kicks this vCPU a little after the next event is due, in case the io
    // thread (starved by host load) has not fired it; with the machine lock held. `always` re-arms even later.
    void arm_backstop(bool always);
    u64 read_tsc();
    void set_tsc(u64 value);
    void note_frame_tsc();
    void fail(Stop why, std::string detail);
    void arm_breakpoints();

    Machine* machine_{nullptr};
    int kvm_fd_{-1}, vm_fd_{-1}, vcpu_fd_{-1};
    kvm_run* run_{nullptr};
    std::size_t run_size_{0};
    kvm_coalesced_mmio_ring* coalesced_{nullptr};  // in the run area (null: an exit per device write)
    u32 coalesced_slots_{0};
    u64 coalesced_writes_{0}, coalesce_failures_{0};
    int coalesced_zones_{0};
    std::thread thread_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> running_{false};
    Stop stop_reason_{Stop::None};
    std::string stop_detail_;
    Stats stats_;
    // MV_PROFILE_EIP=1: where the guest was each time the vCPU was kicked (a sampling profiler for spin loops
    // that make no exits), printed when the CPU thread ends.
    // MV_PROFILE_EIP=FROM-TO samples only from vblank FROM to vblank TO (a stretch of gameplay, not the whole run).
    bool profile_eip_{false};
    u64 profile_from_{0}, profile_to_{~0ull};
    unsigned profile_us_{0};  // MV_PROFILE_EIP_US: sample on the vCPU thread's CPU clock every N us (0 = at kicks)
    timer_t profile_timer_{};
    bool profile_timer_ok_{false};
    std::unordered_map<u32, u64> eip_samples_;
    std::unordered_map<u32, u64> eip_busy_samples_;  // MV_PROFILE_EIP: the samples taken while the GPU thread had work
    u64 next_write_back_poll_ns_{0};  // the next exit that may land background write-backs (Nv2aStub::poll_write_backs)
    u64 busy_guest_ns_{0};                          // and the vCPU time in the guest while it had
    bool tsc_scaled_{false};
    bool dirty_tracking_{false};
    std::vector<u64> dirty_bitmap_;
    bool selective_requested_{false}, selective_{false};
    std::vector<u64> watched_;  // selective tracking: the pages whose next write is reported
    std::atomic<bool> exact_reports_{false};
    // Watches asked for on another thread (the FIFO thread's renderer) wait for the vCPU thread: a clear while the guest
    // runs flushes its TLB and costs it guest time. Applied at the next exit, conservatively: a page already marked
    // written then (perhaps after the texture read it) is reported as written.
    std::mutex watch_mutex_;
    std::vector<std::pair<u32, u32>> watch_queue_;
    std::atomic<bool> watch_pending_{false};
    std::vector<u32> watch_reported_;
    void apply_watch(u32 first, u32 count, bool conservative);
    void apply_pending_watches();
    u32 cpuid_signature_{0x0000068A};
    // --break-eip: hardware execution breakpoints, armed at init and dropped after the first debug exit.
    std::vector<u32> break_eips_;
    bool breakpoints_armed_{false};
    // MV_WATCH_AT_BREAK=REG+HEX[*N]: when a --break-eip address is reached, a hardware write watch (DR3, 4 bytes) on
    // that register plus the offset, and every write to it logged with the writer's registers and stack, N times.
    std::string watch_reg_;
    u32 watch_offset_{0}, watch_va_{0}, watch_hits_{0}, watch_max_{20};
    u32 mmio_breaks_{0};

    // The guest clock (MV_GUEST_CLOCK=wall turns it off: the host clock drives the timeline, as before D16).
    bool guest_clock_on_{true};
    GuestClock clock_;
    bool tsc_offset_ctl_{false};
    u64 tsc_offset_{0};  // the vCPU's TSC offset as last set (read once)
    bool tsc_offset_known_{false};
    // Exact exits (D33): the pending exit's device cost, the last run's CPU time, the fastest device exit seen and the
    // overhead taken off each run (four fifths of it).
    Cycles device_pending_{0};
    bool read_exit_{false};  // the last run ended in a device read or a polling write: it keeps the host's price
    bool counted_device_exit();
    bool poll_step(u64 key);
    std::array<u64, 16> poll_ring_{};  // the last device writes (address, value, size)
    unsigned poll_pos_{0}, poll_period_{0}, poll_streak_{0};
    u64 poll_exits_{0};
    u64 run_entered_cpu_{0}, last_run_cpu_{0}, exit_overhead_ns_{0}, device_exits_{0}, calibration_runs_{0};
    std::array<u64, 200> run_hist_{};  // MMIO runs by CPU time, 100 ns buckets
    void device_exit(Cycles cost, bool calibrates);
    // MV_DEBUG_TSC_LEAD: the TSC lead's change inside runs and across exits, by exit reason (two MSR reads an exit).
    bool debug_lead_{false}, lead_left_set_{false};
    i64 lead_entered_{0}, lead_left_{0}, lead_in_{0};
    unsigned lead_reason_{0};
    std::map<unsigned, std::pair<u64, i64>> lead_out_;
    std::atomic<clockid_t> vcpu_cpu_clock_{CLOCK_THREAD_CPUTIME_ID};  // the vCPU thread's CPU clock, for other threads
    std::atomic<bool> vcpu_clock_known_{false};
    timer_t backstop_{};
    bool backstop_ok_{false};
    u64 backstop_at_ns_{0};  // the host time it is armed for (machine lock)
    u64 backstop_shots_{0};
    bool halted_{false};    // HLT: the guest waits for an interrupt; guest time skips to each next event
    std::string stopped_at_;  // the first HLT with interrupts off: where, and the stack (for the unload report)
    // The kernel's idle loop (fast mode with the guest clock): it spins with interrupts enabled instead of halting, so
    // an idle console burns host time. Found in RAM once the kernel is there, a hardware execution breakpoint on it
    // turns a full turn of the loop with no interrupt since (it has looked for work and found none) into a halt, which
    // skips guest time to the next event as HLT does. MV_IDLE_SKIP=0 leaves the loop to spin.
    std::array<u32, 2> idle_eips_{};  // the loops found (0 = none), one hardware breakpoint each
    u32 idle_eip_{0};                 // the first, 0 until found
    bool idle_skip_{true};
    u64 idle_injections_at_hit_{~0ull}, idle_hits_{0}, idle_halts_{0}, idle_scans_{0};
    bool at_idle_{false};   // the guest is in the idle loop: a hit on it since the last injected interrupt
    u64 idle_spin_ns_{0};
    void find_idle_loop();
    Cycles tsc_epoch_{0};   // guest time at which the TSC was zero
    u64 vblank_seen_{0};
    u64 break_from_{0};  // MV_BREAK_FROM=VBLANK: the --break-eip breakpoints are armed from that vblank on
    u32 break_hits_{0};  // MV_BREAK_KEEP: hits logged so far
    bool frame_tsc_due_{false};
    // Instruments for the report: interrupt latency from INTR to injection in guest time (buckets up to 10, 50,
    // 100, 250, 500, 1000 us and beyond); the busiest port and MMIO page exits.
    u64 irq_latency_buckets_[7]{};
    Cycles irq_latency_max_{0};
    std::unordered_map<u32, u64> exit_sources_;
    // Host time on this thread firing device events (advance_time) and in the exit handlers, and the frame steps.
    u64 event_ns_{0}, handler_ns_{0}, pause_ns_{0};
    u64 tsc_rewinds_{0}, tsc_adjust_failures_{0}, guest_tsc_writes_{0}, tsc_rewound_{0};
    bool tsc_writes_trapped_{false};
    i64 tsc_lead_last_{0}, tsc_lead_max_{0}, tsc_lead_min_{0};
    double tsc_lead_sum_{0};
    u64 tsc_lead_samples_{0};
};

}  // namespace mv
#endif
