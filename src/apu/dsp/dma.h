#pragma once
#include <string>
#include <vector>

#include "apu/dsp/core.h"
#include "core/types.h"

// The DSP's DMA engine as the APU's processors see it: a list of command blocks in DSP memory, each moving
// samples between the DSP's X/Y/P memories and a buffer in the APU's scratch space (plain or circular) or one of
// the processor's FIFOs, converting between 24-bit words and the buffer's sample format. Documented by the
// public xboxdevwiki DSP page (command block words, control word fields, format conversions); register
// behaviour per the public reference emulator.
namespace mv::dsp {

class DmaMemory {
public:
    virtual ~DmaMemory() = default;
    // Scratch space of the processor (an address space the owner maps through its page table).
    virtual bool scratch_read(u32 offset, u8* out, u32 len) = 0;
    virtual bool scratch_write(u32 offset, const u8* in, u32 len) = 0;
    // FIFOs: 0-1 input (read), 0-3 output (write); the owner keeps the ring pointers.
    virtual bool fifo_read(u32 fifo, u8* out, u32 len) = 0;
    virtual bool fifo_write(u32 fifo, const u8* in, u32 len) = 0;
};

class Dma {
public:
    enum Reg : u32 { NextBlock = 0, StartBlock = 1, Control = 2, Configuration = 3 };
    static constexpr u32 kEol = 1u << 14;
    static constexpr u32 kControlRunning = 1u << 4, kControlStopped = 1u << 5, kControlFrozen = 1u << 3;

    Dma(Core& core, DmaMemory& memory) : core_(core), memory_(memory) {}

    u32 read(Reg r);
    void write(Reg r, u32 value);
    bool eol();
    void clear_eol() { eol_ = false; }
    const std::string& last_error() const { return error_; }
    u64 transfers() const { return transfers_; }
    u64 bytes_moved() const { return bytes_; }

private:
    void run();
    void settle();
    bool run_block(u32 block_address);
    u32 read_dsp(u32 address) const;
    void write_dsp(u32 address, u32 value);

    Core& core_;
    DmaMemory& memory_;
    u32 next_block_{0}, start_block_{0}, control_{0}, configuration_{0};
    bool eol_{false};
    // The list's data moves at START; the engine reports itself running (and holds the end-of-list condition) until
    // the transfer time has passed on the core's clock: one cycle per byte, the first timing model.
    u64 busy_until_{0};
    bool pending_eol_{false};
    std::string error_;
    u64 transfers_{0}, bytes_{0};
    std::vector<u8> buffer_;
};

}  // namespace mv::dsp
