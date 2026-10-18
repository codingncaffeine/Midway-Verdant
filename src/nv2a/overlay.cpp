#include "nv2a/overlay.h"

#include <algorithm>

#include "nv2a/yuv.h"

namespace mv {

void draw_overlay(const Overlay& ov, const u8* ram, u64 ram_size, u8* xrgb, u32 pitch, u32 w, u32 h) {
    if (ov.in_w == 0 || ov.in_h == 0 || ov.out_w == 0 || ov.out_h == 0 || ov.pitch == 0) return;
    const u64 last = static_cast<u64>(ov.address) + static_cast<u64>(ov.pitch) * ov.in_h;
    if (last > ram_size) return;
    const u32 x_end = std::min(w, ov.out_x + ov.out_w), y_end = std::min(h, ov.out_y + ov.out_h);
    for (u32 y = ov.out_y; y < y_end; ++y) {
        // Source row and column in 12.20 fixed point from the 12.4 start.
        const u64 t = (static_cast<u64>(ov.t0) << 16) + static_cast<u64>(y - ov.out_y) * ov.dtdy;
        const u32 sy = std::min(static_cast<u32>(t >> 20), ov.in_h - 1);
        const u8* row = ram + ov.address + static_cast<u64>(sy) * ov.pitch;
        u8* out = xrgb + static_cast<u64>(y) * pitch;
        for (u32 x = ov.out_x; x < x_end; ++x) {
            u32* px = reinterpret_cast<u32*>(out + 4u * x);
            if (ov.color_key && (*px & 0xFFFFFFu) != (ov.key & 0xFFFFFFu)) continue;
            const u64 s = (static_cast<u64>(ov.s0) << 16) + static_cast<u64>(x - ov.out_x) * ov.dsdx;
            const u32 sx = std::min(static_cast<u32>(s >> 20), ov.in_w - 1);
            const u8* pair = row + (sx & ~1u) * 2u;  // two pixels share U and V
            int y0, u, y1, v;
            if (ov.yuy2) {
                y0 = pair[0]; u = pair[1]; y1 = pair[2]; v = pair[3];
            } else {
                u = pair[0]; y0 = pair[1]; v = pair[2]; y1 = pair[3];
            }
            *px = yuv601_to_xrgb((sx & 1u) ? y1 : y0, u, v);
        }
    }
}

}  // namespace mv

namespace mv {
void apply_gamma(const std::array<u8, 768>& lut, u8* xrgb, u32 pitch, u32 w, u32 h) {
    for (u32 y = 0; y < h; ++y) {
        u8* p = xrgb + static_cast<std::size_t>(y) * pitch;
        for (u32 x = 0; x < w; ++x, p += 4) {
            p[0] = lut[p[0] * 3u + 2];  // blue
            p[1] = lut[p[1] * 3u + 1];  // green
            p[2] = lut[p[2] * 3u + 0];  // red
        }
    }
}
}  // namespace mv
