#include "apu/processor.h"

#include <cstring>
#include <format>
#include <vector>

namespace mv::apu {

Processor::Processor(const dsp::Core::Config& config, dsp::DmaMemory& memory) : core_(config, this), dma_(core_, memory), memory_(memory) {}

void Processor::hold_reset() {
    running_ = false;
    core_.reset();
    interrupts_ = 0;
}

void Processor::release_reset() {
    core_.reset();
    interrupts_ = 0;
    // The boot ROM: the first 0x800 words of scratch (32-bit each, the word in the low 24 bits) into P memory.
    std::vector<u8> image(0x800 * 4, 0);
    if (!memory_.scratch_read(0, image.data(), static_cast<u32>(image.size()))) {
        problem_ = "bootstrap: the scratch space is not readable";
        return;
    }
    for (u32 i = 0; i < 0x800; ++i) {
        u32 w;
        std::memcpy(&w, &image[i * 4], 4);
        core_.write_p(i, w & 0xFFFFFFu);
    }
    running_ = true;
}

u64 Processor::run_frame(u64 cycle_budget) {
    if (!running_) return 0;
    interrupts_ |= kInterruptStartFrame;
    core_.set_halted(false);
    const u64 used = core_.run(cycle_budget);
    if (core_.error() && problem_.empty()) problem_ = core_.error_text();
    return used;
}

u32 Processor::peripheral_read(u32 address) {
    switch (address) {
        case 0xFFFFC5: return interrupts_ | (dma_.eol() ? kInterruptDmaEol : 0u);
        case 0xFFFFD4: return dma_.read(dsp::Dma::NextBlock);
        case 0xFFFFD5: return dma_.read(dsp::Dma::StartBlock);
        case 0xFFFFD6: return dma_.read(dsp::Dma::Control);
        case 0xFFFFD7: return dma_.read(dsp::Dma::Configuration);
        default: return 0;
    }
}

void Processor::peripheral_write(u32 address, u32 value) {
    switch (address) {
        case 0xFFFFC4:
            if (value & 1u) core_.set_halted(true);
            return;
        case 0xFFFFC5:
            interrupts_ &= ~value;
            if (value & kInterruptDmaEol) dma_.clear_eol();
            return;
        case 0xFFFFD4: dma_.write(dsp::Dma::NextBlock, value); return;
        case 0xFFFFD5: dma_.write(dsp::Dma::StartBlock, value); return;
        case 0xFFFFD6: dma_.write(dsp::Dma::Control, value); return;
        case 0xFFFFD7: dma_.write(dsp::Dma::Configuration, value); return;
        default: return;
    }
}

}  // namespace mv::apu
