#include "nv2a/scanout.h"

#include <cstring>

namespace mv {

void scanout_to_xrgb(const u8* fb, u32 pitch, u32 bytes_per_pixel, bool rgb565, u32 w, u32 h, u8* out) {
    for (u32 y = 0; y < h; ++y) {
        const u8* src = fb + static_cast<std::size_t>(y) * pitch;
        u8* dst = out + static_cast<std::size_t>(y) * w * 4;
        if (bytes_per_pixel == 4) {
            std::memcpy(dst, src, static_cast<std::size_t>(w) * 4);
            continue;
        }
        for (u32 x = 0; x < w; ++x, dst += 4) {
            const u32 p = static_cast<u32>(src[2 * x]) | (static_cast<u32>(src[2 * x + 1]) << 8);
            const u32 r5 = (p >> (rgb565 ? 11 : 10)) & 0x1Fu;
            const u32 g = rgb565 ? (p >> 5) & 0x3Fu : (p >> 5) & 0x1Fu;
            const u32 b5 = p & 0x1Fu;
            dst[0] = static_cast<u8>((b5 << 3) | (b5 >> 2));
            dst[1] = static_cast<u8>(rgb565 ? ((g << 2) | (g >> 4)) : ((g << 3) | (g >> 2)));
            dst[2] = static_cast<u8>((r5 << 3) | (r5 >> 2));
            dst[3] = 0;
        }
    }
}

}  // namespace mv
