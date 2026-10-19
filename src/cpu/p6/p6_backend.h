#pragma once
// The accurate mode: the interpreted Pentium III as a CpuBackend. Its instruction stream is the master clock.
#include <array>
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cpu/backend.h"
#include "cpu/p6/cpu.h"

namespace mv {

namespace p6 {
class Oracle;
class BlockEngine;
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
    void request_selective_tracking() override { selective_requested_ = true; }
    bool selective_tracking() const override { return cpu_ && cpu_->memory().selective(); }
    void watch_pages(u32 first, u32 count) override {
        if (cpu_) cpu_->memory().watch(first, count);
    }
    bool owns_timeline() const override { return true; }
    Cycles timeline_cycles() const override { return cycles_.load(std::memory_order_relaxed); }
    std::string report() const override;

    p6::Cpu& cpu() { return *cpu_; }

private:
    void thread_main();
    void run_batch();
    void run_batch_blocks();
    // The between-instruction duties: due events, an interrupt. Returns true when one was delivered.
    bool between_instructions();
    void sync_time();
    void fail(Stop why, std::string detail);

    Machine* machine_{nullptr};
    std::unique_ptr<p6::Cpu> cpu_;
    std::unique_ptr<p6::Oracle> oracle_;
    std::unique_ptr<p6::BlockEngine> blocks_;
    bool with_oracle_{false};
    std::thread thread_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<u64> cycles_{0};
    Stop stop_reason_{Stop::None};
    std::string stop_detail_;
    Stats stats_;
    bool dirty_tracking_{false};
    bool selective_requested_{false};
    u32 cpuid_signature_{0x0000068Au};
    // Instruments: MV_P6_STOP_CYCLES=N stops the core at the first instruction boundary at or past cycle N (a run's end
    // the io thread does not pick); the delivery hash folds the state at every interrupt and fault delivery, so two
    // runs (two executors) compare by one number.
    u64 stop_at_cycles_{~u64{0}};
    u64 delivery_hash_{0xcbf29ce484222325ull};
    u64 deliveries_{0};
    // The rest of the architectural state at every delivery (x87, SSE, segments, control registers), and RAM at the end.
    u64 state_hash_{0xcbf29ce484222325ull};
    std::FILE* delivery_log_{nullptr};  // MV_P6_DELIVERY_LOG=FILE: a line per delivery (cycle, instructions, EIP, hash, state hash)
    void fold_delivery();
    void note_hold();
    // The kernel idle loop halts (D22, here by address): found in RAM by its shape, a pass through it with no
    // interrupt delivered since the last pass halts the core to the next event. MV_IDLE_SKIP=0: it spins.
    void find_idle_loop();
    bool idle_turn();
    bool idle_skip_{true};
    std::array<u32, 2> idle_eips_{};
    u64 idle_scans_{0}, idle_hits_{0}, idle_halts_{0}, idle_deliveries_at_hit_{~u64{0}};
};

}  // namespace mv
