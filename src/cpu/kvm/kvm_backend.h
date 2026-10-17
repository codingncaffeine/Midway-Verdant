#pragma once
#if defined(__linux__)
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cpu/backend.h"

struct kvm_run;

namespace mv {

// The fast mode on Linux: a KVM virtual machine whose memory slots are the console's RAM (twice: at zero and at
// the GPU-side alias), the aliased flash as a read-only slot, and the last page with the MCPX ROM laid over it
// until the firmware hides it. Everything not backed by a slot exits to the machine's bus. Interrupts come from
// the machine's 8259 pair through KVM_INTERRUPT with the interrupt-window dance; no in-kernel irqchip.
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
    void fail(Stop why, std::string detail);

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
    bool tsc_scaled_{false};
    bool dirty_tracking_{false};
    std::vector<u64> dirty_bitmap_;
    u32 cpuid_signature_{0x0000068A};
};

}  // namespace mv
#endif
