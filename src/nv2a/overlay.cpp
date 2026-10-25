#include "nv2a/overlay.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "nv2a/yuv.h"

namespace mv {

void draw_overlay(const Overlay& ov, const u8* ram, u64 ram_size, u8* xrgb, u32 pitch, u32 w, u32 h, u32 y_from, u32 y_to) {
    if (ov.in_w == 0 || ov.in_h == 0 || ov.out_w == 0 || ov.out_h == 0 || ov.pitch == 0) return;
    const u64 last = static_cast<u64>(ov.address) + static_cast<u64>(ov.pitch) * ov.in_h;
    if (last > ram_size) return;
    const u32 x_end = std::min(w, ov.out_x + ov.out_w), y_end = std::min({h, ov.out_y + ov.out_h, y_to});
    if (ov.out_x >= x_end) return;
    // Each source row converted once, and each screen column's source column worked out once: at a high internal
    // resolution many screen pixels sample one texel (a 6x picture is 11 million of them, converted one by one before).
    std::vector<u32> line(ov.in_w), cols(x_end - ov.out_x);
    for (u32 x = ov.out_x; x < x_end; ++x) {
        // Source column in 12.20 fixed point from the 12.4 start.
        const u64 s = (static_cast<u64>(ov.s0) << 16) + static_cast<u64>(x - ov.out_x) * ov.dsdx;
        cols[x - ov.out_x] = std::min(static_cast<u32>(s >> 20), ov.in_w - 1);
    }
    u32 line_sy = ~0u, prev_sy = ~0u;
    const u8* prev_out = nullptr;
    for (u32 y = std::max(ov.out_y, y_from); y < y_end; ++y) {
        const u64 t = (static_cast<u64>(ov.t0) << 16) + static_cast<u64>(y - ov.out_y) * ov.dtdy;
        const u32 sy = std::min(static_cast<u32>(t >> 20), ov.in_h - 1);
        u8* out = xrgb + static_cast<u64>(y) * pitch;
        if (!ov.color_key && sy == prev_sy) {
            // The same source row over the same columns as the row above: the same pixels.
            std::memcpy(out + 4u * ov.out_x, prev_out + 4u * ov.out_x, 4u * static_cast<std::size_t>(x_end - ov.out_x));
            continue;
        }
        if (sy != line_sy) {
            const u8* row = ram + ov.address + static_cast<u64>(sy) * ov.pitch;
            for (u32 sx = 0; sx < ov.in_w; ++sx) {
                const u8* pair = row + (sx & ~1u) * 2u;  // two pixels share U and V
                int y0, u, y1, v;
                if (ov.yuy2) {
                    y0 = pair[0]; u = pair[1]; y1 = pair[2]; v = pair[3];
                } else {
                    u = pair[0]; y0 = pair[1]; v = pair[2]; y1 = pair[3];
                }
                line[sx] = yuv601_to_xrgb((sx & 1u) ? y1 : y0, u, v);
            }
            line_sy = sy;
        }
        u32* px = reinterpret_cast<u32*>(out) + ov.out_x;
        for (u32 i = 0; i < x_end - ov.out_x; ++i) {
            if (ov.color_key && (px[i] & 0xFFFFFFu) != (ov.key & 0xFFFFFFu)) continue;
            px[i] = line[cols[i]];
        }
        prev_sy = sy;
        prev_out = out;
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
