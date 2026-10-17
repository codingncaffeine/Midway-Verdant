#pragma once
#include <string>
#include <vector>

#include "core/types.h"
#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pgraph/kelvin_state.h"

// Texture images as the Kelvin texture units read them: swizzled (Morton order) power-of-two images with mip
// chains and cube faces, linear "rect" images with a pitch, DXT-compressed blocks, palettised indices. Every
// format decodes to one of three host pixel layouts. Format numbers and layouts per local-notes/hw/nv2a-pgraph.md.
namespace mv::pgraph {

enum class HostPixel : u8 { Rgba8, R16, R32f };

struct FormatInfo {
    u32 bytes_per_pixel{0};  // 0 = unknown format
    bool linear{false};
    bool compressed{false};
    bool depth{false};
    u32 block_bytes{0};  // DXT: bytes per 4x4 block
};
FormatInfo format_info(u32 color_format);

struct DecodedTexture {
    HostPixel pixel{HostPixel::Rgba8};
    u32 width{0}, height{0}, depth{1}, levels{1}, faces{1};
    // One image per (face, level), face-major; each holds width*height*depth texels of the host pixel.
    std::vector<std::vector<u8>> images;
    std::string error;
    u32 bytes_per_texel() const { return pixel == HostPixel::Rgba8 ? 4 : (pixel == HostPixel::R16 ? 2 : 4); }
    const std::vector<u8>& image(u32 face, u32 level) const { return images[face * levels + level]; }
};

// Byte offset of texel (x, y, z) in a swizzled image of the given log2 sizes (Morton interleave, x in the low bit).
u32 swizzled_offset(u32 x, u32 y, u32 z, u32 log_w, u32 log_h, u32 log_d);

// The bytes the texture occupies in memory from its offset (all levels and faces), 0 for an unknown format.
u32 texture_size_in_memory(const TextureStage& stage);

// Decodes a texture stage's image from guest memory. `base` is the physical address the stage's offset resolves to.
bool decode_texture(const TextureStage& stage, u32 base, const MemoryView& memory, const u8* palette, DecodedTexture& out);

// Decoders of single images (exposed for tests): each writes RGBA8 into `out` (w*h*4 bytes).
void decode_dxt1(const u8* blocks, u32 w, u32 h, u8* out);
void decode_dxt3(const u8* blocks, u32 w, u32 h, u8* out);
void decode_dxt5(const u8* blocks, u32 w, u32 h, u8* out);

}  // namespace mv::pgraph
