#pragma once
// The interpreted Pentium III: one instruction at a time against the machine's bus. The backend owns the
// thread, the lock and the clock; this class owns the architectural state and executes.
#include <functional>
#include <string>

#include "core/bus.h"
#include "core/types.h"
#include "cpu/p6/decode.h"
#include "cpu/p6/memory.h"
#include "cpu/p6/state.h"
#include "cpu/p6/timing.h"

namespace mv::p6 {

// Whether the x87 unit runs on the host's own 80-bit x87 (GCC/Clang on x86). Elsewhere (MSVC, other hosts)
// long double is a double and extended-precision results, their rounding and C1 are approximations.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
inline constexpr bool kHostX87 = true;
#else
inline constexpr bool kHostX87 = false;
#endif

class Cpu {
public:
    explicit Cpu(Bus& bus);

    CpuState& state() { return st_; }
    const CpuState& state() const { return st_; }
    Memory& memory() { return mem_; }

    // Power-on: real mode at the reset vector, EDX = the CPUID signature, everything else as the SDM lists.
    void reset(u32 cpuid_signature);

    // The interrupt source: `pending` answers the vector that would be delivered (or -1), `acknowledge`
    // takes it (the INTA cycle).
    void set_interrupt_source(std::function<int()> pending, std::function<int()> acknowledge) {
        irq_pending_ = std::move(pending);
        irq_ack_ = std::move(acknowledge);
    }

    // Execute one instruction. Faults it raises are delivered through the IDT inside the call (a double
    // fault too); a triple fault sets `triple_fault()`. Throws `Unimplemented` for what the core lacks.
    void step();
    // Take a pending external interrupt if the core accepts one now (IF set, no inhibit): delivers it and
    // returns true. Wakes a halted core.
    bool take_interrupt();
    // Deliver an external interrupt vector now (the oracle harness drives delivery itself).
    void deliver_external(u8 vector);

    bool triple_fault() const { return triple_fault_; }
    u64 cycles() const { return cycles_; }
    const u64* cycles_pointer() const { return &cycles_; }
    u64 instructions() const { return instructions_; }
    // The clock jumps forward (a HLT sleeping to the next event).
    void set_cycles(u64 c) {
        cycles_ = c;
        timing_.skip_to(c);
    }
    const Timing& timing() const { return timing_; }
    // In single-iteration mode a REP string instruction executes one element per step (what a single-stepping
    // hardware oracle does).
    void set_single_iteration(bool on) { single_iteration_ = on; }

    // The last instruction decoded (for reports), and where it started.
    const Insn& last_insn() const { return insn_; }
    u32 last_eip() const { return insn_eip_; }
    u16 last_cs() const { return insn_cs_; }
    // The bytes of the last instruction, as hex, for reports.
    std::string last_bytes();
    // A one-line description of the state for reports.
    std::string describe() const;
    // Flags the last instruction left undefined and registers the oracle must not compare (RDTSC's result).
    u32 undefined_flags() const { return st_.undefined_flags; }
    bool last_was_rdtsc() const { return last_rdtsc_; }
    bool last_was_cpuid() const { return last_cpuid_; }
    bool last_faulted() const { return last_faulted_; }
    bool last_was_rep_step() const { return last_rep_step_; }
    bool last_delivered_interrupt() const { return last_delivered_; }
    u8 last_vector() const { return last_vector_; }

    // ---- used by the execution units (public for the unit files) ----
    u32 effective_address(const Insn& in) const;
    u32 linear(u8 seg, u32 offset, unsigned size, Access access);
    u32 read_op(const Insn& in, const Opnd& o, unsigned size);
    void write_op(const Insn& in, const Opnd& o, unsigned size, u32 value);
    void push(unsigned size, u32 value);
    u32 pop(unsigned size);
    u32 stack_pointer() const { return st_.stack32() ? st_.gpr[kEsp] : st_.reg16(kEsp); }
    void set_stack_pointer(u32 v) {
        if (st_.stack32()) st_.gpr[kEsp] = v;
        else st_.set_reg16(kEsp, static_cast<u16>(v));
    }
    bool condition(u8 cc) const;

    // Segmentation and descriptors (system.cpp).
    struct Descriptor {
        u32 lo{0}, hi{0};
        u32 base() const { return (hi & 0xFF000000u) | ((hi & 0xFFu) << 16) | (lo >> 16); }
        u32 limit() const {
            u32 l = (lo & 0xFFFFu) | (hi & 0x000F0000u);
            if (hi & (1u << 23)) l = (l << 12) | 0xFFFu;
            return l;
        }
        u32 ar() const { return ((hi >> 8) & 0xFFu) | ((hi >> 8) & 0xF000u); }
        u8 type() const { return static_cast<u8>((hi >> 8) & 0xF); }
        bool system() const { return (hi & (1u << 12)) == 0; }
        unsigned dpl() const { return (hi >> 13) & 3; }
        bool present() const { return (hi & (1u << 15)) != 0; }
        bool code() const { return !system() && (type() & 8) != 0; }
        bool conforming() const { return code() && (type() & 4) != 0; }
        bool data() const { return !system() && (type() & 8) == 0; }
        bool writable() const { return data() && (type() & 2) != 0; }
        bool readable() const { return data() || (type() & 2) != 0; }
    };
    // Read the descriptor a selector names; #GP(selector) past the table's limit.
    Descriptor read_descriptor(u16 selector, u8 fault_vector = kGP);
    void set_accessed(u16 selector);
    void load_segment(u8 seg, u16 selector);  // MOV/POP sreg and LxS
    void load_cs(u16 selector, const Descriptor& d);
    void load_cached(u8 seg, u16 selector, const Descriptor& d);
    void set_real_mode_segment(u8 seg, u16 selector);

    void jump_far(u16 selector, u32 offset, const Insn& in);
    void call_far(u16 selector, u32 offset, const Insn& in);
    void ret_far(unsigned size, u32 pop_bytes);
    void iret(unsigned size);
    void deliver(u8 vector, bool external, bool software, bool has_error, u32 error_code);

    void write_cr(unsigned n, u32 v);
    u32 read_cr(unsigned n);
    u64 read_msr(u32 index);
    void write_msr(u32 index, u64 v);
    void cpuid();

    // x87 (fpu.cpp) and SIMD (simd.cpp) units.
    void exec_x87(const Insn& in);
    void exec_simd(const Insn& in);
    void fxsave(const Insn& in);
    void fxrstor(const Insn& in);
    void ldmxcsr(const Insn& in);
    void stmxcsr(const Insn& in);

private:
    class Fetcher;
    void execute(const Insn& in);
    void exec_alu(const Insn& in);
    void exec_shift(const Insn& in);
    void exec_string(const Insn& in);
    void exec_system(const Insn& in);
    void fault_entry(const Fault& f);
    void account(bool interrupt_delivery);

    Bus& bus_;
    CpuState st_;
    Memory mem_;
    Insn insn_{};
    u32 insn_eip_{0};
    u16 insn_cs_{0};
    u64 cycles_{0};
    u64 instructions_{0};
    bool triple_fault_{false};
    bool single_iteration_{false};
    bool last_rdtsc_{false}, last_cpuid_{false}, last_faulted_{false}, last_rep_step_{false}, last_delivered_{false};
    u8 last_vector_{0};
    Timing timing_;
    PhysAddr fetch_phys_{0};
    unsigned rep_iterations_{0};
    unsigned top_before_{0};
    std::function<int()> irq_pending_, irq_ack_;
    u32 cpuid_signature_{0x0000068Au};
};

}  // namespace mv::p6
