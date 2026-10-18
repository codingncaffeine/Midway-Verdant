#pragma once
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "apu/dsp/decode.h"
#include "core/types.h"

// A DSP56300-family core as the APU's global and encode processors use it: 24-bit words, two 56-bit accumulators,
// four 24-bit input registers, eight address/offset/modifier register triples, a 16-deep system stack, hardware
// DO loops and REP, fractional arithmetic with convergent rounding and transfer limiting. Written from the public
// DSP56300 Family Manual (chapters 3, 4, 5, 12 and 13); the peripheral space X:$FFFF80-$FFFFFF belongs to the
// owner (the APU adds its own registers there). Cycle counts follow the manual's basic rule (one cycle per word)
// until the Appendix A timing table is applied.
namespace mv::dsp {

class Peripherals {
public:
    virtual ~Peripherals() = default;
    virtual u32 peripheral_read(u32 address) = 0;              // full 24-bit address, $FFFF80..$FFFFFF
    virtual void peripheral_write(u32 address, u32 value) = 0;
    // True when `bit` of the peripheral register at `address` cannot change before the owner's next frame: a program
    // polling it in a tight loop then only spins out its cycle budget, which Core::run skips (same registers, same
    // cycle count).
    virtual bool static_until_next_frame(u32 address, u32 bit) const {
        (void)address;
        (void)bit;
        return false;
    }
};

class Core {
public:
    struct Config {
        u32 x_words{4096}, y_words{2048}, p_words{4096};
        // A second view of X memory: X:mix_alias .. +0x400 reaches X:mix_target .. +0x400 (0 = none). See apu.cpp.
        u32 mix_alias{0}, mix_target{0};
    };
    Core(const Config& config, Peripherals* peripherals);

    void reset();
    // Executes one instruction; returns the cycles it took. Stops (returns 0) once an error is recorded.
    u32 step();
    // Runs until `cycles` have elapsed, an error happens, or `halted()` becomes true (the owner sets halt from a
    // peripheral write). Returns the cycles executed.
    u64 run(u64 cycles);

    bool error() const { return !error_.empty(); }
    const std::string& error_text() const { return error_; }
    bool halted() const { return halted_; }
    void set_halted(bool h) { halted_ = h; }
    u64 cycles() const { return cycles_; }
    u64 instructions() const { return instructions_; }

    // Memories (24-bit words in the low bits).
    u32 read_x(u32 address);
    u32 read_y(u32 address);
    u32 read_p(u32 address) const;
    void write_x(u32 address, u32 value);
    void write_y(u32 address, u32 value);
    void write_p(u32 address, u32 value);
    // The X-memory index an address reaches (the mix-buffer alias applied); x_memory().size() when out of range.
    std::size_t x_index(u32 address) const {
        if (cfg_.mix_alias && address >= cfg_.mix_alias && address < cfg_.mix_alias + 0x400u) address = address - cfg_.mix_alias + cfg_.mix_target;
        return address < x_.size() ? address : x_.size();
    }
    std::vector<u32>& x_memory() { return x_; }
    std::vector<u32>& y_memory() { return y_; }
    std::vector<u32>& p_memory() { return p_; }

    // Registers.
    struct Regs {
        i64 a{0}, b{0};  // 56-bit, sign-extended in 64
        u32 x0{0}, x1{0}, y0{0}, y1{0};
        std::array<u32, 8> r{}, n{}, m{};
        u32 pc{0}, sr{0}, omr{0}, la{0}, lc{0}, sp{0}, sz{0}, vba{0}, sc{0}, ep{0};
        std::array<u32, 16> ssh{}, ssl{};
    };
    Regs& regs() { return r_; }
    // Instrument: while set, every instruction is written there before it runs (address, text, the data ALU
    // registers and R0-R7).
    void set_trace(std::FILE* f) { trace_ = f; }
    const Regs& regs() const { return r_; }

    // The Status Register bits (chapter 5).
    enum Sr : u32 {
        C = 1u << 0, V = 1u << 1, Z = 1u << 2, N = 1u << 3, U = 1u << 4, E = 1u << 5, L = 1u << 6, S = 1u << 7,
        I0 = 1u << 8, I1 = 1u << 9, S0 = 1u << 10, S1 = 1u << 11, SC_ = 1u << 13, DM = 1u << 14, LF = 1u << 15, FV = 1u << 16,
        SA = 1u << 17, CE = 1u << 19, SM = 1u << 20, RM = 1u << 21
    };

    // A listing line for the instruction at `address` (debugging).
    std::string disassemble_at(u32 address) const;

private:
    struct Cached {
        Decoded d;
        bool valid{false};
    };
    const Decoded& decoded(u32 address);
    void execute(const Decoded& d, u32 ext, u32 at);
    void fail(const std::string& why);

    // Data ALU.
    i64& acc(u32 which) { return which ? r_.b : r_.a; }
    static i64 sext56(i64 v) { return (v << 8) >> 8; }
    i64 source56(u32 jjj, u32 dest) const;  // Table 12-20 source operands as 56-bit values
    u32 read_reg5(u32 code, bool limit);     // 5-bit register codes (Table 12-16), with the limiter on A/B
    void write_reg5(u32 code, u32 value);
    u32 read_reg6(u32 code);                 // 6-bit codes (Table 12-13): all on-chip registers
    void write_reg6(u32 code, u32 value);
    u32 limit24(i64 acc56);
    void set_ccr_result(i64 result, bool carry_in_known, bool carry, bool overflow);
    void shift_flags(i64 before, i64 after, u32 count, bool left);
    void set_ccr_logical(i64 acc56);
    void alu(const Decoded& d, u32 ext, i64 pre_a, i64 pre_b);
    i64 multiply(u32 qqq, bool negate) const;
    i64 round56(i64 v) const;
    bool condition(u32 cccc) const;

    // AGU.
    u32 ea_address(u32 mmmrrr, u32 ext, bool& used_ext, bool update);  // resolves and (optionally) updates Rn
    u32 agu_update(u32 r, u32 m, i32 delta) const;
    void parallel_move(const Decoded& d, u32 ext, i64 pre_a, i64 pre_b);

    // Program control.
    void push(u32 pc_value, u32 sr_value);
    void pop(u32& pc_value, u32& sr_value);
    void do_loop(u32 count, u32 last_address, u32 first_address);
    void end_of_loop_check(u32 executed_at);
    u32 read_space(bool y, u32 address);
    void write_space(bool y, u32 address, u32 value);

    Config cfg_;
    Peripherals* periph_;
    std::vector<u32> x_, y_, p_;
    std::FILE* trace_{nullptr};
    std::vector<Cached> cache_;
    Regs r_;
    u64 cycles_{0}, instructions_{0};
    std::string error_;
    bool halted_{false};
    // REP state: the next instruction repeats `rep_` times.
    u32 rep_remaining_{0};
    u32 rep_saved_lc_{0};
    bool rep_active_{false};
    // A tight wait on a frame-static peripheral bit, noted by the bit-jump that keeps looping: the loop is the jump
    // alone, or the jump and one ADD #xx / INC on an accumulator before it (an idle counter).
    struct IdleWait {
        bool pending{false};
        u32 dest{0};
        i64 delta{0};
        u32 per_iteration{0};
    } idle_;
    void note_idle_wait(u32 target, u32 at, u32 jump_words);
};

}  // namespace mv::dsp
