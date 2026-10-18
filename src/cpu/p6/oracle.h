#pragma once
// The KVM lock-step oracle: a hypervisor vCPU on the same RAM and flash that replays every instruction the
// interpreter executes, from the interpreter's own pre-state, and the comparison names the first divergence.
#include <string>
#include <vector>

#include "core/types.h"
#include "cpu/p6/cpu.h"

namespace mv {
class Machine;
}

namespace mv::p6 {

class Oracle {
public:
    Oracle();
    ~Oracle();
    Oracle(const Oracle&) = delete;
    Oracle& operator=(const Oracle&) = delete;

    bool init(Machine& machine, Cpu& cpu, std::string& error);
    void reset();
    // Run one instruction on the interpreter (with its writes buffered), replay it on the hypervisor, compare,
    // apply the memory writes and resynchronise on a divergence. False when the oracle itself failed.
    bool step(Cpu& cpu);
    std::string last_error() const { return error_; }
    std::string summary() const;

    struct Counts {
        u64 steps{0}, compared{0}, divergences{0}, resyncs{0}, skipped{0}, lockstep_breaks{0}, io_mismatches{0}, memory_mismatches{0};
        u64 undefined_flag_differences{0};
        u64 fpu_pointer_differences{0};
    };
    const Counts& counts() const { return counts_; }

private:
    struct Impl;
    Impl* impl_{nullptr};
    Counts counts_;
    std::string error_;
};

}  // namespace mv::p6
