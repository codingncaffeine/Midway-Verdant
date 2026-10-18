#pragma once
// The accurate mode: the interpreted Pentium III as a CpuBackend. Its instruction stream is the master clock.
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cpu/backend.h"
#include "cpu/p6/cpu.h"

namespace mv {

namespace p6 {
class Oracle;
}

class P6Backend : public CpuBackend {
public:
    explicit P6Backend(bool with_oracle);
    ~P6Backend() override;
    std::string_view name() const override { return "p6"; }

    bool init(Machine& machine, std::string& error) override;
    void reset() override;
    void start() override;
    void stop() override;
    void join() override;
    void kick() override;
    Stop stop_reason() const override { return stop_reason_; }
    std::string stop_detail() const override { return stop_detail_; }
    Stats stats() const override;
    bool read_regs(Regs& out) override;
    void set_dirty_tracking(bool on) override { dirty_tracking_ = on; }
    void take_dirty_pages(std::vector<u32>& pages) override;
    bool owns_timeline() const override { return true; }
    Cycles timeline_cycles() const override { return cycles_.load(std::memory_order_relaxed); }
    std::string report() const override;

    p6::Cpu& cpu() { return *cpu_; }

private:
    void thread_main();
    void run_batch();
    void sync_time();
    void fail(Stop why, std::string detail);

    Machine* machine_{nullptr};
    std::unique_ptr<p6::Cpu> cpu_;
    std::unique_ptr<p6::Oracle> oracle_;
    bool with_oracle_{false};
    std::thread thread_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<u64> cycles_{0};
    Stop stop_reason_{Stop::None};
    std::string stop_detail_;
    Stats stats_;
    bool dirty_tracking_{false};
    u32 cpuid_signature_{0x0000068Au};
};

}  // namespace mv
