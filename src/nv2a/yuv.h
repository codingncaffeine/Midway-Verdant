#pragma once
#include <algorithm>

#include "core/types.h"

// YUV 4:2:2 as the NV2A's video paths read it (the PVIDEO overlay and the YUV texture formats): BT.601 in video range,
// Y 16-235 and Cb/Cr 16-240, converted with 8.8 fixed-point coefficients and rounded. Two pixels share one U and V.
namespace mv {

// One pixel to 0x00RRGGBB.
inline u32 yuv601_to_xrgb(int y, int u, int v) {
    const int c = (y - 16) * 298, d = u - 128, e = v - 128;
    const int r = std::clamp((c + 409 * e + 128) >> 8, 0, 255);
    const int g = std::clamp((c - 100 * d - 208 * e + 128) >> 8, 0, 255);
    const int b = std::clamp((c + 516 * d + 128) >> 8, 0, 255);
    return (static_cast<u32>(r) << 16) | (static_cast<u32>(g) << 8) | static_cast<u32>(b);
}

}  // namespace mv
