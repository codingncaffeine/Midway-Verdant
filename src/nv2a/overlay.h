#pragma once
#include "core/types.h"

// The NV2A video overlay (PVIDEO): a YUV 4:2:2 buffer in RAM scaled onto a rectangle of the screen, over the
// frame buffer or only where the frame buffer shows the colour key. Games play their full-motion video through
// it. Registers per the nouveau / envytools PVIDEO block (two buffers: BASE, LIMIT, OFFSET, SIZE_IN, POINT_IN,
// DS_DX, DT_DY, POINT_OUT, SIZE_OUT, FORMAT), behaviour cross-checked against xemu (read, never copied).
namespace mv {

struct Overlay {
    u32 address{0}, limit{0};  // the source's first byte in RAM and the last byte the buffer may use
    u32 pitch{0}, in_w{0}, in_h{0};
    u32 s0{0}, t0{0};          // source start, 12.4 fixed point
    u32 dsdx{1u << 20}, dtdy{1u << 20};  // source step per screen pixel, 12.20 fixed point
    u32 out_x{0}, out_y{0}, out_w{0}, out_h{0};
    bool yuy2{false};          // byte order Y0 U Y1 V (format colour 1); else U Y0 V Y1
    bool color_key{false};
    u32 key{0};                // the frame-buffer colour the overlay replaces, 0x00RRGGBB
};

// Draws the overlay into an XRGB8888 picture of the frame buffer (w x h, pitch bytes per line).
void draw_overlay(const Overlay& ov, const u8* ram, u64 ram_size, u8* xrgb, u32 pitch, u32 w, u32 h);

}  // namespace mv
