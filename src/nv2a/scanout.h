#pragma once
#include "core/types.h"

// The frame buffer's rows as the display reads them, converted to XRGB8888 (w x h, w * 4 bytes a row): 32-bit pixels
// are copied; 16-bit ones are widened from R5G6B5 or X1R5G5B5, each field's top bits repeated into the bits below it,
// so full intensity stays full.
namespace mv {

void scanout_to_xrgb(const u8* fb, u32 pitch, u32 bytes_per_pixel, bool rgb565, u32 w, u32 h, u8* out);

}  // namespace mv
