#include "nv2a/pgraph/twod.h"

#include <cstring>
#include <format>

#include "debug/trace.h"

namespace mv::pgraph {

u32 TwoD::bytes_per_pixel(u32 format) {
    switch (format) {
        case 0x01: return 1;                          // Y8
        case 0x02: case 0x03: case 0x04: case 0x05: return 2;  // X1R5G5B5 (two variants), R5G6B5, Y16
        case 0x06: case 0x07: case 0x08: case 0x09: case 0x0A: case 0x0B: return 4;  // X8R8G8B8, X1A7R8G8B8, A8R8G8B8, Y32
        default: return 0;
    }
}

bool TwoD::method(u32 cls, u32 method, u32 value) {
    if (cls == kClassSurfaces) {
        switch (method) {
            case kSurfDmaSource: dma_source_ = value; return true;
            case kSurfDmaDestination: dma_destination_ = value; return true;
            case kSurfFormat: format_ = value & 0xFFu; return true;
            case kSurfPitch: pitch_ = value; return true;
            case kSurfOffsetSource: offset_source_ = value; return true;
            case kSurfOffsetDestination: offset_destination_ = value; return true;
            default: return false;
        }
    }
    if (cls == kClassBlit) {
        switch (method) {
            case kBlitSurfaces: return true;  // the one surfaces object above
            case kBlitOperation: operation_ = value; return true;
            case kBlitPointIn: point_in_ = value; return true;
            case kBlitPointOut: point_out_ = value; return true;
            case kBlitSize:
                size_ = value;
                if (!blit()) trace::log(trace::Cat::Gpu, "2d: blit refused: {}", error_);
                return true;
            default: return false;
        }
    }
    return false;
}

bool TwoD::blit() {
    // Points and the size: x (width) in 15:0, y (height) in 31:16.
    const u32 w = size_ & 0xFFFFu, h = size_ >> 16;
    const u32 xin = point_in_ & 0xFFFFu, yin = point_in_ >> 16, xout = point_out_ & 0xFFFFu, yout = point_out_ >> 16;
    if (w == 0 || h == 0) return true;
    if (operation_ != kOperationSrcCopy) {
        error_ = std::format("operation {} (only SRCCOPY is modelled)", operation_);
        return false;
    }
    const u32 bpp = bytes_per_pixel(format_);
    if (bpp == 0) {
        error_ = std::format("colour format {:02x}", format_);
        return false;
    }
    const DmaObject src = resolver_.resolve(dma_source_), dst = resolver_.resolve(dma_destination_);
    if (!src.valid || !dst.valid) {
        error_ = std::format("DMA objects {:08x}/{:08x} do not resolve", dma_source_, dma_destination_);
        return false;
    }
    const u64 src_pitch = pitch_ & 0xFFFFu, dst_pitch = pitch_ >> 16, row = static_cast<u64>(w) * bpp;
    // Physical first and last bytes of each side, checked against the DMA windows and RAM.
    const u64 s0 = static_cast<u64>(offset_source_) + yin * src_pitch + static_cast<u64>(xin) * bpp, s1 = s0 + (h - 1) * src_pitch + row;
    const u64 d0 = static_cast<u64>(offset_destination_) + yout * dst_pitch + static_cast<u64>(xout) * bpp, d1 = d0 + (h - 1) * dst_pitch + row;
    if (s1 > static_cast<u64>(src.limit) + 1 || d1 > static_cast<u64>(dst.limit) + 1) {
        error_ = "rectangle outside its DMA object";
        return false;
    }
    auto phys = [](u64 a) { return a >= 0xF0000000ull ? a - 0xF0000000ull : a; };  // the GPU's RAM alias
    const u64 ps = phys(src.base + s0), pd = phys(dst.base + d0);
    if (ps + (s1 - s0) > ram_size_ || pd + (d1 - d0) > ram_size_) {
        error_ = "rectangle outside RAM";
        return false;
    }
    if (gpu_copy_ && gpu_copy_(static_cast<u32>(ps), static_cast<u32>(src_pitch), static_cast<u32>(pd), static_cast<u32>(dst_pitch), w, h, bpp)) {
        ++blits_;
        ++gpu_blits_;
        trace::log(trace::Cat::Gpu, "2d: blit {}x{} ({} B/px) {:08x} -> {:08x} on the GPU", w, h, bpp, ps, pd);
        return true;
    }
    // The source must be in RAM, and so must whatever was rendered into the destination: the blit may cover only
    // part of it, and the rest has to survive into the reloaded picture.
    if (before_read_) {
        before_read_(static_cast<u32>(ps), static_cast<u32>(s1 - s0));
        before_read_(static_cast<u32>(pd), static_cast<u32>(d1 - d0));
    }
    if (ram_copy_) ram_copy_(static_cast<u32>(pd), static_cast<u32>(dst_pitch), static_cast<u32>(ps), static_cast<u32>(src_pitch), static_cast<u32>(row), h);
    else
        for (u32 y = 0; y < h; ++y) std::memmove(ram_ + pd + y * dst_pitch, ram_ + ps + y * src_pitch, row);
    if (after_write_) after_write_(static_cast<u32>(pd), static_cast<u32>(d1 - d0));
    ++blits_;
    trace::log(trace::Cat::Gpu, "2d: blit {}x{} ({} B/px) {:08x} -> {:08x}", w, h, bpp, ps, pd);
    return true;
}

}  // namespace mv::pgraph
