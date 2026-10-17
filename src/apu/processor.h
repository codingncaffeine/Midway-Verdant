#pragma once
#include <functional>
#include <map>
#include <string>
#include <utility>

#include "apu/dsp/core.h"
#include "apu/dsp/dma.h"
#include "core/types.h"

// One of the APU's programmable processors (the global processor or the encode processor): a DSP56300 core, its
// DMA engine, and the few peripheral registers the Xbox adds in the DSP's I/O space: $FFFFC4 halt (bit 0 stops
// the core until the next frame), $FFFFC5 interrupt status (bit 1 start of frame, bit 7 DMA end of list; writes
// clear), $FFFFD4-$FFFFD7 the DMA registers. The boot ROM's work on reset: program memory loaded from the first
// 0x800 words of scratch. Behaviour per the public reference emulator and the xboxdevwiki DSP page.
namespace mv::apu {

class Processor : public dsp::Peripherals {
public:
    static constexpr u32 kInterruptStartFrame = 1u << 1, kInterruptDmaEol = 1u << 7;

    Processor(const dsp::Core::Config& config, dsp::DmaMemory& memory);

    dsp::Core& core() { return core_; }
    dsp::Dma& dma() { return dma_; }

    // Reset (held) and release: the release bootstraps program memory from scratch and starts at P:0.
    void hold_reset();
    void release_reset();
    bool running() const { return running_; }

    // A frame: raise the start-of-frame condition and run the core until it halts itself or the cycle budget
    // is spent. Returns the cycles used.
    u64 run_frame(u64 cycle_budget);

    u32 peripheral_read(u32 address) override;
    void peripheral_write(u32 address, u32 value) override;
    const std::string& last_problem() const { return problem_; }
    u32 interrupts() const { return interrupts_; }
    // Instruments: how often the program touched each peripheral register (reads, writes), by address.
    const std::map<u32, std::pair<u64, u64>>& peripheral_traffic() const { return traffic_; }

private:
    dsp::Core core_;
    dsp::Dma dma_;
    dsp::DmaMemory& memory_;
    bool running_{false};
    u32 interrupts_{0};
    std::string problem_;
    std::map<u32, std::pair<u64, u64>> traffic_;
};

}  // namespace mv::apu
