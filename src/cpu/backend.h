#pragma once
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "core/types.h"

namespace mv {

class Machine;

// A CPU backend runs the guest's instruction stream against the machine: the hardware-virtualized fast mode
// (KVM, WHPX) or the cycle-counted interpreter. The machine owns the bus, the devices, the scheduler and the
// lock that serialises device access; a backend calls back into it on every exit and takes interrupts from it.
class CpuBackend {
public:
    enum class Stop { None, Requested, TripleFault, InternalError };

    virtual ~CpuBackend() = default;
    virtual std::string_view name() const = 0;

    // Prepare the virtual CPU against the machine's memory. Fails with a message when the host cannot provide it.
    virtual bool init(Machine& machine, std::string& error) = 0;
    // Put the CPU in its power-on state: real mode at the reset vector, TSC at zero.
    virtual void reset() = 0;
    // Start executing on the backend's own thread; `stop` asks it to leave at the next exit and `join` waits.
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void join() = 0;
    // Wake the CPU: an interrupt line changed, or a stop was requested. Safe from any thread.
    virtual void kick() = 0;

    virtual Stop stop_reason() const = 0;
    virtual std::string stop_detail() const = 0;

    // Counters for the end-of-run report.
    struct Stats {
        u64 exits{0}, mmio_exits{0}, pio_exits{0}, hlt_exits{0}, irq_injections{0}, irq_window_exits{0}, kicks{0}, intr_exits{0};
        u64 by_reason[64]{};
        u64 by_vector[256]{};
    };
    virtual Stats stats() const = 0;

    // Registers for the report and the debugger (fast mode reads them from the hypervisor).
    struct Regs {
        u32 eax, ebx, ecx, edx, esi, edi, ebp, esp, eip, eflags, cr0, cr3, cr4;
        u16 cs, ds, es, fs, gs, ss;
        u64 tsc;
    };
    virtual bool read_regs(Regs& out) = 0;
};

// The fast mode on this host, or nullptr with a reason when there is none.
std::unique_ptr<CpuBackend> make_fast_backend(std::string& why_not);

}  // namespace mv
