#include "apu/dsp/dma.h"

#include <algorithm>
#include <cstring>
#include <format>

namespace mv::dsp {

namespace {
constexpr u32 kWord = 0xFFFFFFu;

// Sample formats (control word bits 10-12).
enum Format : u32 { F8 = 0, F16 = 1, F24Msb = 2, F32 = 3, F24Lsb = 6 };

u32 bytes_per_sample(u32 format) {
    switch (format) {
        case F8: return 1;
        case F16: return 2;
        case F24Msb: case F32: case F24Lsb: return 4;
        default: return 0;
    }
}
}  // namespace

u32 Dma::read_dsp(u32 address) const {
    // DSP addresses: 0x0000-0x17FF X, 0x1800-0x27FF Y, 0x2800-0x37FF P.
    if (address < 0x1800) {
        const std::size_t i = core_.x_index(address);
        return i < core_.x_memory().size() ? core_.x_memory()[i] : 0;
    }
    if (address < 0x2800) return (address - 0x1800) < core_.y_memory().size() ? core_.y_memory()[address - 0x1800] : 0;
    if (address < 0x3800) return core_.read_p(address - 0x2800);
    return 0;
}

void Dma::write_dsp(u32 address, u32 value) {
    value &= kWord;
    if (address < 0x1800) {
        if (const std::size_t i = core_.x_index(address); i < core_.x_memory().size()) core_.x_memory()[i] = value;
    } else if (address < 0x2800) {
        if (address - 0x1800 < core_.y_memory().size()) core_.y_memory()[address - 0x1800] = value;
    } else if (address < 0x3800) {
        core_.write_p(address - 0x2800, value);
    }
}

void Dma::settle() {
    if (pending_eol_ && core_.cycles() >= busy_until_) {
        pending_eol_ = false;
        if (walk_pending_) {
            walk_pending_ = false;
            run();  // the list goes on: the next stretch of it
            return;
        }
        eol_ = true;
        control_ = (control_ | kControlStopped) & ~kControlRunning;
    }
}

void Dma::reset() {
    next_block_ = start_block_ = control_ = configuration_ = 0;
    eol_ = pending_eol_ = walk_pending_ = false;
    busy_until_ = 0;
}

bool Dma::eol() {
    settle();
    return eol_;
}

u32 Dma::read(Reg r) {
    settle();
    switch (r) {
        case NextBlock: return next_block_;
        case StartBlock: return start_block_;
        case Control: return control_;
        case Configuration: return configuration_;
    }
    return 0;
}

void Dma::write(Reg r, u32 value) {
    value &= kWord;
    switch (r) {
        case NextBlock: next_block_ = value; return;
        case StartBlock: start_block_ = value; return;
        case Configuration: configuration_ = value; return;
        case Control:
            switch (value & 7u) {
                case 1:  // start
                    control_ = (control_ | kControlRunning) & ~kControlStopped;
                    run();
                    break;
                case 2:  // stop
                    control_ = (control_ | kControlStopped) & ~kControlRunning;
                    break;
                case 3: control_ |= kControlFrozen; break;
                case 4:
                    control_ &= ~kControlFrozen;
                    run();
                    break;
                case 5:  // abort
                    control_ = (control_ | kControlStopped) & ~kControlRunning;
                    break;
                default: break;
            }
            return;
    }
}

void Dma::run() {
    if (!(control_ & kControlRunning) || (control_ & kControlFrozen)) return;
    const u64 bytes_before = bytes_;
    // The engine walks the list over time, a byte a cycle: a stretch of it at once, then the rest once the processor
    // has run that long. A list need not end (one title's points at a block whose next word is sample data, itself),
    // and then the engine keeps moving data while the processor runs on, until it is stopped, aborted or reset.
    u32 blocks = 0;
    while (!(next_block_ & kEol) && bytes_ - bytes_before < kStretchBytes && blocks < kStretchBlocks) {
        const u32 block = next_block_ & 0x3FFFu;
        ++blocks;
        if (!run_block(block)) break;
    }
    const u64 moved = bytes_ - bytes_before;
    walk_pending_ = !(next_block_ & kEol);
    if (walk_pending_ && !unended_logged_) {
        unended_logged_ = true;
        if (error_.empty()) error_ = std::format("DMA list not ended after {} blocks, next {:06x}: it runs on over time", blocks, next_block_);
    }
    // A finished list: the engine stays busy for the transfer time, then stops and raises end-of-list.
    // Even an empty list is observed running for a few cycles: the kernel's program waits to see the bit.
    // A stretch that did not end costs at least a cycle a block, so a list of empty blocks still takes time.
    busy_until_ = core_.cycles() + std::max<u64>(moved, walk_pending_ ? std::max<u64>(blocks, 16) : 16);
    pending_eol_ = true;
}

bool Dma::run_block(u32 block) {
    const u32 next = read_dsp(block), control = read_dsp(block + 1), count = read_dsp(block + 2), dsp_offset = read_dsp(block + 3);
    u32 scratch_offset = read_dsp(block + 4);
    const u32 scratch_base = read_dsp(block + 5), scratch_size = read_dsp(block + 6) + 1;
    next_block_ = next;
    const bool interleave = control & 1u, to_buffer = (control >> 1) & 1u, writeback = (control >> 4) & 1u;
    if (trace_)
        std::fprintf(trace_, "DMA block %04x: control %06x (%s buffer %u) count %06x dsp %06x scratch off %06x base %06x size %06x\n", block, control,
                     to_buffer ? "dsp->" : "dsp<-", (control >> 5) & 0xFu, count, dsp_offset, scratch_offset, scratch_base, scratch_size);
    const u32 buffer_id = (control >> 5) & 0xFu, format = (control >> 10) & 7u, step = (control >> 14) & 0x3FFu;
    const u32 sample_bytes = bytes_per_sample(format);
    if (sample_bytes == 0) {
        error_ = std::format("DMA block {:04x}: sample format {} is skipped", block, format);
        return true;  // the hardware skips the transfer
    }
    u32 samples;
    u32 channels = 1, blocks = 0;
    if (interleave) {
        channels = (count & 0xFu) + 1;
        blocks = count >> 4;
        samples = channels * blocks;
    } else {
        samples = count & kWord;
    }
    const u32 words = format == F32 ? samples * 2 : samples;
    const u32 bytes = samples * sample_bytes;
    buffer_.assign(bytes, 0);
    const u32 stride = step ? step : blocks;

    auto dsp_index = [&](u32 i) {
        if (!interleave) return dsp_offset + i;
        const u32 block_index = i / channels, channel = i % channels;
        return dsp_offset + block_index + channel * stride;
    };

    if (to_buffer) {
        for (u32 i = 0; i < samples; ++i) {
            u8* out = &buffer_[i * sample_bytes];
            switch (format) {
                case F8: {
                    // Rounded to 8 bits with the sign bit flipped (unsigned bytes), saturated.
                    const u32 w = read_dsp(dsp_index(i));
                    const i32 s = static_cast<i32>(w << 8) >> 8;
                    i32 r = (s + 0x8000) >> 16;
                    r = std::clamp(r, -128, 127);
                    out[0] = static_cast<u8>(static_cast<u32>(r) ^ 0x80u);
                    break;
                }
                case F16: {
                    const u32 w = read_dsp(dsp_index(i)) >> 8;
                    out[0] = static_cast<u8>(w);
                    out[1] = static_cast<u8>(w >> 8);
                    break;
                }
                case F24Msb: {
                    const u32 w = read_dsp(dsp_index(i)) << 8;
                    std::memcpy(out, &w, 4);
                    break;
                }
                case F24Lsb: {
                    const u32 w = read_dsp(dsp_index(i));
                    std::memcpy(out, &w, 4);
                    break;
                }
                case F32: {
                    const u32 lo = read_dsp(dsp_offset + 2 * i), hi = read_dsp(dsp_offset + 2 * i + 1);
                    const u32 w = (hi << 8) | (lo >> 16);
                    std::memcpy(out, &w, 4);
                    break;
                }
                default: break;
            }
        }
        bool ok = false;
        if (buffer_id <= 3) {
            ok = memory_.fifo_write(buffer_id, buffer_.data(), bytes);
        } else if (buffer_id == 0xE) {
            // Circular scratch buffer: wrap at base + size.
            u32 done = 0;
            ok = true;
            if (scratch_offset >= scratch_size) scratch_offset = 0;
            while (done < bytes && ok) {
                const u32 chunk = std::min(bytes - done, scratch_size - scratch_offset);
                ok = memory_.scratch_write(scratch_base + scratch_offset, &buffer_[done], chunk);
                scratch_offset += chunk;
                if (scratch_offset >= scratch_size) scratch_offset = 0;
                done += chunk;
            }
        } else if (buffer_id == 0xF) {
            ok = memory_.scratch_write(scratch_base + scratch_offset, buffer_.data(), bytes);
            scratch_offset += bytes;
        } else {
            error_ = std::format("DMA block {:04x}: buffer {:x} is not known", block, buffer_id);
        }
        if (!ok && error_.empty()) error_ = std::format("DMA block {:04x}: the write to buffer {:x} failed", block, buffer_id);
    } else {
        bool ok = false;
        if (buffer_id <= 1) {
            ok = memory_.fifo_read(buffer_id, buffer_.data(), bytes);
        } else if (buffer_id == 0xE) {
            u32 done = 0;
            ok = true;
            if (scratch_offset >= scratch_size) scratch_offset = 0;
            while (done < bytes && ok) {
                const u32 chunk = std::min(bytes - done, scratch_size - scratch_offset);
                ok = memory_.scratch_read(scratch_base + scratch_offset, &buffer_[done], chunk);
                scratch_offset += chunk;
                if (scratch_offset >= scratch_size) scratch_offset = 0;
                done += chunk;
            }
        } else if (buffer_id == 0xF) {
            ok = memory_.scratch_read(scratch_base + scratch_offset, buffer_.data(), bytes);
            scratch_offset += bytes;
        } else {
            error_ = std::format("DMA block {:04x}: buffer {:x} cannot be read", block, buffer_id);
        }
        if (ok) {
            for (u32 i = 0; i < samples; ++i) {
                const u8* in = &buffer_[i * sample_bytes];
                switch (format) {
                    case F8: write_dsp(dsp_index(i), static_cast<u32>(in[0] ^ 0x80u) << 16); break;
                    case F16: write_dsp(dsp_index(i), (static_cast<u32>(in[0]) | (static_cast<u32>(in[1]) << 8)) << 8); break;
                    case F24Msb: {
                        u32 w;
                        std::memcpy(&w, in, 4);
                        write_dsp(dsp_index(i), w >> 8);
                        break;
                    }
                    case F24Lsb: {
                        u32 w;
                        std::memcpy(&w, in, 4);
                        write_dsp(dsp_index(i), w & kWord);
                        break;
                    }
                    case F32: {
                        u32 w;
                        std::memcpy(&w, in, 4);
                        write_dsp(dsp_offset + 2 * i, (w & 0xFFu) << 16);
                        write_dsp(dsp_offset + 2 * i + 1, w >> 8);
                        break;
                    }
                    default: break;
                }
            }
        } else if (error_.empty()) {
            error_ = std::format("DMA block {:04x}: the read from buffer {:x} failed", block, buffer_id);
        }
    }
    (void)words;
    if (writeback) write_dsp(block + 4, scratch_offset);
    ++transfers_;
    bytes_ += bytes;
    return true;
}

}  // namespace mv::dsp
