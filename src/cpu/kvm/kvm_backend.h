#pragma once
#if defined(__linux__)
#include <time.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/guest_clock.h"
#include "cpu/backend.h"

struct kvm_run;

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
    void set_break_eips(const std::vector<u32>& eips) override { break_eips_ = eips; }
    bool owns_timeline() const override { return guest_clock_on_; }
    Cycles timeline_cycles() const override;
    double timeline_rate() const override;
    void timeline_advanced() override;
    bool hardware_assisted() const override { return true; }
    std::string clock_name() const override;
    std::string report() const override;

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
    std::thread thread_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> running_{false};
    Stop stop_reason_{Stop::None};
    std::string stop_detail_;
    Stats stats_;
    // MV_PROFILE_EIP=1: where the guest was each time the vCPU was kicked (a sampling profiler for spin loops
    // that make no exits), printed when the CPU thread ends.
    bool profile_eip_{false};
    std::unordered_map<u32, u64> eip_samples_;
    bool tsc_scaled_{false};
    bool dirty_tracking_{false};
    std::vector<u64> dirty_bitmap_;
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
    std::atomic<clockid_t> vcpu_cpu_clock_{CLOCK_THREAD_CPUTIME_ID};  // the vCPU thread's CPU clock, for other threads
    std::atomic<bool> vcpu_clock_known_{false};
    timer_t backstop_{};
    bool backstop_ok_{false};
    u64 backstop_at_ns_{0};  // the host time it is armed for (machine lock)
    u64 backstop_shots_{0};
    bool halted_{false};    // HLT: the guest waits for an interrupt; guest time skips to each next event
    Cycles tsc_epoch_{0};   // guest time at which the TSC was zero
    u64 vblank_seen_{0};
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
