#pragma once
#include <functional>
#include <string>

#include "nv2a/pgraph/kelvin.h"

// The 2D engine as the console's software drives it: the 2D surfaces object (class 0x62: a source and a destination
// surface, each a DMA object, an offset and a pitch, one colour format for both) and the image blit (class 0x9F:
// copy a rectangle from a point in the source to a point in the destination). The kernel's boot animation copies its
// finished logo render into a texture this way and shows the copy for the closing logo.
// Method numbers per nxdk's pbkit nv_objects.h (MIT); the colour formats per nouveau's list of the NV04 2D surface
// formats [verify]. Only SRCCOPY is modelled; the other operations, the clip rectangle and the colour key are not.
namespace mv::pgraph {

class TwoD {
public:
    static constexpr u32 kClassSurfaces = 0x62, kClassBlit = 0x9F;
    // Class 0x62.
    static constexpr u32 kSurfDmaSource = 0x184, kSurfDmaDestination = 0x188, kSurfFormat = 0x300, kSurfPitch = 0x304, kSurfOffsetSource = 0x308,
                         kSurfOffsetDestination = 0x30C;
    // Class 0x9F.
    static constexpr u32 kBlitSurfaces = 0x19C, kBlitOperation = 0x2FC, kBlitPointIn = 0x300, kBlitPointOut = 0x304, kBlitSize = 0x308;
    static constexpr u32 kOperationSrcCopy = 3;

    // `ram` is the guest RAM the engine reads and writes; `resolver` turns a DMA object handle into its window.
    // Around a blit the host hears which RAM it reads (a render target there must be in RAM first) and writes
    // (textures decoded from there are stale).
    TwoD(u8* ram, u64 ram_size, const HandleResolver& resolver) : ram_(ram), ram_size_(ram_size), resolver_(resolver) {}
    void set_hooks(std::function<void(u32 phys, u32 len)> before_read, std::function<void(u32 phys, u32 len)> after_write) {
        before_read_ = std::move(before_read);
        after_write_ = std::move(after_write);
    }

    // A method of either class; true when the engine took it. The blit runs on SIZE.
    bool method(u32 cls, u32 method, u32 value);
    // Bytes per pixel of a 2D surface colour format (0 = unknown).
    static u32 bytes_per_pixel(u32 format);

    u64 blits() const { return blits_; }
    const std::string& last_error() const { return error_; }

private:
    bool blit();

    u8* ram_;
    u64 ram_size_;
    const HandleResolver& resolver_;
    std::function<void(u32, u32)> before_read_, after_write_;
    u32 dma_source_{0}, dma_destination_{0}, format_{0}, pitch_{0}, offset_source_{0}, offset_destination_{0};
    u32 operation_{kOperationSrcCopy}, point_in_{0}, point_out_{0}, size_{0};
    u64 blits_{0};
    std::string error_;
};

}  // namespace mv::pgraph
