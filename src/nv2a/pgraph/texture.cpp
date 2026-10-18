#include "nv2a/pgraph/texture.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "nv2a/yuv.h"

namespace mv::pgraph {

namespace {

constexpr u32 kCubeFaceAlignment = 128;

// Writes one RGBA8 texel.
inline void put(u8* dst, u32 r, u32 g, u32 b, u32 a) {
    dst[0] = static_cast<u8>(r);
    dst[1] = static_cast<u8>(g);
    dst[2] = static_cast<u8>(b);
    dst[3] = static_cast<u8>(a);
}

inline u32 expand5(u32 v) { return (v << 3) | (v >> 2); }
inline u32 expand6(u32 v) { return (v << 2) | (v >> 4); }
inline u32 expand4(u32 v) { return v * 17; }
inline u32 expand1(u32 v) { return v ? 255 : 0; }

// Decodes one source texel of `format` at `p` into RGBA8 (or R16/R32f into `dst` for depth and 16-bit formats).
void decode_texel(u32 format, const u8* p, const u8* palette, HostPixel pixel, u8* dst) {
    u16 h = 0;
    u32 w = 0;
    switch (format) {
        case 0x00: case 0x13:  // Y8
            put(dst, p[0], p[0], p[0], 255);
            return;
        case 0x01: case 0x1B:  // AY8
            put(dst, p[0], p[0], p[0], p[0]);
            return;
        case 0x02: case 0x10:  // A1R5G5B5
            std::memcpy(&h, p, 2);
            put(dst, expand5((h >> 10) & 31), expand5((h >> 5) & 31), expand5(h & 31), expand1(h >> 15));
            return;
        case 0x03: case 0x1C:  // X1R5G5B5
            std::memcpy(&h, p, 2);
            put(dst, expand5((h >> 10) & 31), expand5((h >> 5) & 31), expand5(h & 31), 255);
            return;
        case 0x04: case 0x1D:  // A4R4G4B4
            std::memcpy(&h, p, 2);
            put(dst, expand4((h >> 8) & 15), expand4((h >> 4) & 15), expand4(h & 15), expand4(h >> 12));
            return;
        case 0x05: case 0x11:  // R5G6B5
            std::memcpy(&h, p, 2);
            put(dst, expand5(h >> 11), expand6((h >> 5) & 63), expand5(h & 31), 255);
            return;
        case 0x06: case 0x12:  // A8R8G8B8: bytes B G R A
            put(dst, p[2], p[1], p[0], p[3]);
            return;
        case 0x07: case 0x1E:  // X8R8G8B8
            put(dst, p[2], p[1], p[0], 255);
            return;
        case 0x0B: {  // I8 with an A8R8G8B8 palette
            const u8* e = palette ? palette + static_cast<std::size_t>(p[0]) * 4 : nullptr;
            if (e) put(dst, e[2], e[1], e[0], e[3]);
            else put(dst, p[0], p[0], p[0], 255);
            return;
        }
        case 0x16: case 0x29:  // R8B8: bytes B R
            put(dst, p[1], 0, p[0], 255);
            return;
        case 0x17: case 0x28:  // G8B8: bytes B G
            put(dst, 0, p[1], p[0], 255);
            return;
        case 0x19: case 0x1F:  // A8: white with the byte as alpha, so a texture modulated by a colour keeps the colour
            put(dst, 255, 255, 255, p[0]);
            return;
        case 0x1A: case 0x20:  // A8Y8: bytes Y A
            put(dst, p[0], p[0], p[0], p[1]);
            return;
        case 0x3A: case 0x3F:  // A8B8G8R8: bytes R G B A
            put(dst, p[0], p[1], p[2], p[3]);
            return;
        case 0x3B: case 0x40:  // B8G8R8A8: bytes A R G B
            put(dst, p[1], p[2], p[3], p[0]);
            return;
        case 0x3C: case 0x41:  // R8G8B8A8: bytes A B G R
            put(dst, p[3], p[2], p[1], p[0]);
            return;
        case 0x2C: case 0x2D: case 0x30: case 0x31: case 0x32: case 0x35:  // 16-bit depth and Y16
            std::memcpy(dst, p, 2);
            return;
        case 0x2A: case 0x2B: case 0x2E: case 0x2F: {
            // X8 Y24 depth: the 24-bit depth in bits 31-8 above the stencil byte (the Z24S8 surface's own word),
            // scaled to 0..1. The float forms are read as fixed point.
            std::memcpy(&w, p, 4);
            const float f = static_cast<float>(w >> 8) / 16777215.0f;
            std::memcpy(dst, &f, 4);
            return;
        }
        default:
            (void)pixel;
            put(dst, 255, 0, 255, 255);  // an unsupported format shows as magenta
            return;
    }
}

// The YUV 4:2:2 formats (linear only): every two pixels are four bytes holding both lumas and one U and V. 0x24
// (CR8YB8CB8YA8) is Y0 U Y1 V in memory, YUY2; 0x25 (YB8CR8YA8CB8) is U Y0 V Y1, UYVY. `pair` is the group's first
// byte, `odd` picks its second pixel.
bool yuv_format(u32 format) { return format == 0x24 || format == 0x25; }

void decode_yuv_texel(u32 format, const u8* pair, bool odd, u8* dst) {
    int y0, u, y1, v;
    if (format == 0x24) {
        y0 = pair[0]; u = pair[1]; y1 = pair[2]; v = pair[3];
    } else {
        u = pair[0]; y0 = pair[1]; v = pair[2]; y1 = pair[3];
    }
    const u32 rgb = yuv601_to_xrgb(odd ? y1 : y0, u, v);
    put(dst, (rgb >> 16) & 0xFFu, (rgb >> 8) & 0xFFu, rgb & 0xFFu, 255);
}

u32 dxt_color565(const u8* p) {
    u16 h;
    std::memcpy(&h, p, 2);
    return h;
}

void dxt_palette(u16 c0, u16 c1, bool dxt1, u8 colors[4][4]) {
    const u32 r0 = expand5(c0 >> 11), g0 = expand6((c0 >> 5) & 63), b0 = expand5(c0 & 31);
    const u32 r1 = expand5(c1 >> 11), g1 = expand6((c1 >> 5) & 63), b1 = expand5(c1 & 31);
    put(colors[0], r0, g0, b0, 255);
    put(colors[1], r1, g1, b1, 255);
    if (!dxt1 || c0 > c1) {
        put(colors[2], (2 * r0 + r1) / 3, (2 * g0 + g1) / 3, (2 * b0 + b1) / 3, 255);
        put(colors[3], (r0 + 2 * r1) / 3, (g0 + 2 * g1) / 3, (b0 + 2 * b1) / 3, 255);
    } else {
        put(colors[2], (r0 + r1) / 2, (g0 + g1) / 2, (b0 + b1) / 2, 255);
        put(colors[3], 0, 0, 0, 0);
    }
}

void decode_dxt_colors(const u8* block, bool dxt1, u32 bx, u32 by, u32 w, u32 h, u8* out, const u8* alphas) {
    u8 colors[4][4];
    const u16 c0 = static_cast<u16>(dxt_color565(block)), c1 = static_cast<u16>(dxt_color565(block + 2));
    dxt_palette(c0, c1, dxt1, colors);
    u32 bits;
    std::memcpy(&bits, block + 4, 4);
    for (u32 y = 0; y < 4; ++y) {
        for (u32 x = 0; x < 4; ++x) {
            const u32 px = bx * 4 + x, py = by * 4 + y;
            if (px >= w || py >= h) continue;
            const u32 idx = (bits >> (2 * (y * 4 + x))) & 3;
            u8* d = out + (static_cast<std::size_t>(py) * w + px) * 4;
            std::memcpy(d, colors[idx], 4);
            if (alphas) d[3] = alphas[y * 4 + x];
        }
    }
}

}  // namespace

FormatInfo format_info(u32 f) {
    switch (f) {
        case 0x00: case 0x01: case 0x0B: case 0x19: return {1, false};
        case 0x13: case 0x1B: case 0x1F: return {1, true};
        case 0x02: case 0x03: case 0x04: case 0x05: case 0x1A: case 0x27: case 0x28: case 0x29: case 0x32: return {2, false};
        case 0x10: case 0x11: case 0x16: case 0x17: case 0x1C: case 0x1D: case 0x20: case 0x35: return {2, true};
        case 0x24: case 0x25: return {2, true};  // YUV 4:2:2: 4 bytes per 2 pixels
        case 0x06: case 0x07: case 0x3A: case 0x3B: case 0x3C: return {4, false};
        case 0x12: case 0x1E: case 0x3F: case 0x40: case 0x41: return {4, true};
        case 0x0C: return {0, false, true, false, 8};
        case 0x0E: case 0x0F: return {0, false, true, false, 16};
        case 0x2C: case 0x2D: return {2, false, false, true};
        case 0x30: case 0x31: return {2, true, false, true};
        case 0x2A: case 0x2B: return {4, false, false, true};
        case 0x2E: case 0x2F: return {4, true, false, true};
        default: return {};
    }
}

u32 swizzled_offset(u32 x, u32 y, u32 z, u32 log_w, u32 log_h, u32 log_d) {
    u32 offset = 0, bit = 0;
    const u32 n = std::max({log_w, log_h, log_d});
    for (u32 i = 0; i < n; ++i) {
        if (i < log_w) offset |= ((x >> i) & 1u) << bit++;
        if (i < log_h) offset |= ((y >> i) & 1u) << bit++;
        if (i < log_d) offset |= ((z >> i) & 1u) << bit++;
    }
    return offset;
}

void decode_dxt1(const u8* blocks, u32 w, u32 h, u8* out) {
    const u32 bw = (w + 3) / 4, bh = (h + 3) / 4;
    for (u32 by = 0; by < bh; ++by)
        for (u32 bx = 0; bx < bw; ++bx) decode_dxt_colors(blocks + (by * bw + bx) * 8, true, bx, by, w, h, out, nullptr);
}

void decode_dxt3(const u8* blocks, u32 w, u32 h, u8* out) {
    const u32 bw = (w + 3) / 4, bh = (h + 3) / 4;
    for (u32 by = 0; by < bh; ++by) {
        for (u32 bx = 0; bx < bw; ++bx) {
            const u8* b = blocks + (by * bw + bx) * 16;
            u8 alphas[16];
            for (u32 i = 0; i < 16; ++i) alphas[i] = static_cast<u8>(expand4((b[i / 2] >> (4 * (i & 1))) & 15));
            decode_dxt_colors(b + 8, false, bx, by, w, h, out, alphas);
        }
    }
}

void decode_dxt5(const u8* blocks, u32 w, u32 h, u8* out) {
    const u32 bw = (w + 3) / 4, bh = (h + 3) / 4;
    for (u32 by = 0; by < bh; ++by) {
        for (u32 bx = 0; bx < bw; ++bx) {
            const u8* b = blocks + (by * bw + bx) * 16;
            const u32 a0 = b[0], a1 = b[1];
            u32 table[8] = {a0, a1, 0, 0, 0, 0, 0, 0};
            if (a0 > a1) {
                for (u32 i = 1; i < 7; ++i) table[i + 1] = ((7 - i) * a0 + i * a1) / 7;
            } else {
                for (u32 i = 1; i < 5; ++i) table[i + 1] = ((5 - i) * a0 + i * a1) / 5;
                table[6] = 0;
                table[7] = 255;
            }
            u64 bits = 0;
            for (u32 i = 0; i < 6; ++i) bits |= static_cast<u64>(b[2 + i]) << (8 * i);
            u8 alphas[16];
            for (u32 i = 0; i < 16; ++i) alphas[i] = static_cast<u8>(table[(bits >> (3 * i)) & 7]);
            decode_dxt_colors(b + 8, false, bx, by, w, h, out, alphas);
        }
    }
}

namespace {

u32 level_bytes(const FormatInfo& f, u32 w, u32 h, u32 d) {
    if (f.compressed) return ((w + 3) / 4) * ((h + 3) / 4) * f.block_bytes * d;
    return w * h * d * f.bytes_per_pixel;
}

u32 mip_count(const TextureStage& stage) {
    u32 levels = std::max<u32>(stage.mip_levels(), 1);
    const u32 largest = std::max(stage.log_width(), stage.log_height());
    return std::min(levels, largest + 1);
}

}  // namespace

u32 texture_size_in_memory(const TextureStage& stage) {
    const FormatInfo f = format_info(stage.color_format());
    if (f.bytes_per_pixel == 0 && !f.compressed) return 0;
    if (f.linear) return stage.rect_height() * stage.pitch();
    u32 w = 1u << stage.log_width(), h = 1u << stage.log_height(), d = stage.dimensionality() == 3 ? (1u << stage.log_depth()) : 1;
    u32 total = 0;
    for (u32 l = 0; l < mip_count(stage); ++l) {
        total += level_bytes(f, std::max(w, 1u), std::max(h, 1u), std::max(d, 1u));
        w >>= 1;
        h >>= 1;
        if (stage.dimensionality() == 3) d >>= 1;
    }
    if (stage.cubemap()) total = ((total + kCubeFaceAlignment - 1) & ~(kCubeFaceAlignment - 1)) * 6;
    return total;
}

bool decode_texture(const TextureStage& stage, u32 base, const MemoryView& memory, const u8* palette, DecodedTexture& out) {
    out = DecodedTexture{};
    const u32 format = stage.color_format();
    const FormatInfo f = format_info(format);
    if (f.bytes_per_pixel == 0 && !f.compressed) {
        out.error = std::format("texture format {:02x} is not supported yet", format);
        return false;
    }
    out.pixel = (format == 0x2A || format == 0x2B || format == 0x2E || format == 0x2F) ? HostPixel::R32f
              : (f.depth || format == 0x32 || format == 0x35) ? HostPixel::R16 : HostPixel::Rgba8;
    const u32 texel = out.bytes_per_texel();

    if (f.linear) {
        out.width = stage.rect_width();
        out.height = stage.rect_height();
        if (out.width == 0 || out.height == 0) {
            out.error = "linear texture with an empty image rect";
            return false;
        }
        const u32 pitch = stage.pitch();
        const bool yuv = yuv_format(format);
        // A YUV line of odd width still ends on a whole two-pixel group.
        const u32 row_bytes = (yuv ? (out.width + 1) & ~1u : out.width) * f.bytes_per_pixel;
        const u8* src = memory.at(base, pitch * (out.height - 1) + row_bytes);
        if (!src) {
            out.error = std::format("linear texture at {:08x} ({}x{} pitch {}) is outside RAM", base, out.width, out.height, pitch);
            return false;
        }
        std::vector<u8> img(static_cast<std::size_t>(out.width) * out.height * texel);
        for (u32 y = 0; y < out.height; ++y) {
            for (u32 x = 0; x < out.width; ++x) {
                u8* dst = &img[(static_cast<std::size_t>(y) * out.width + x) * texel];
                if (yuv) decode_yuv_texel(format, src + y * pitch + (x & ~1u) * 2, (x & 1u) != 0, dst);
                else decode_texel(format, src + y * pitch + x * f.bytes_per_pixel, palette, out.pixel, dst);
            }
        }
        out.images.push_back(std::move(img));
        return true;
    }

    const u32 log_w = stage.log_width(), log_h = stage.log_height();
    const bool volume = stage.dimensionality() == 3;
    const u32 log_d = volume ? stage.log_depth() : 0;
    out.width = 1u << log_w;
    out.height = 1u << log_h;
    out.depth = 1u << log_d;
    out.levels = mip_count(stage);
    out.faces = stage.cubemap() ? 6 : 1;
    const u32 face_bytes = texture_size_in_memory(stage) / out.faces;
    u32 face_base = base;
    for (u32 face = 0; face < out.faces; ++face) {
        u32 offset = face_base;
        u32 w = out.width, h = out.height, d = out.depth;
        u32 lw = log_w, lh = log_h, ld = log_d;
        for (u32 level = 0; level < out.levels; ++level) {
            const u32 bytes = level_bytes(f, w, h, d);
            const u8* src = memory.at(offset, bytes);
            if (!src) {
                out.error = std::format("texture level {} face {} at {:08x} ({} bytes) is outside RAM", level, face, offset, bytes);
                return false;
            }
            std::vector<u8> img(static_cast<std::size_t>(w) * h * d * texel);
            if (f.compressed) {
                if (format == 0x0C) decode_dxt1(src, w, h, img.data());
                else if (format == 0x0E) decode_dxt3(src, w, h, img.data());
                else decode_dxt5(src, w, h, img.data());
            } else {
                for (u32 z = 0; z < d; ++z)
                    for (u32 y = 0; y < h; ++y)
                        for (u32 x = 0; x < w; ++x)
                            decode_texel(format, src + swizzled_offset(x, y, z, lw, lh, ld) * f.bytes_per_pixel, palette, out.pixel,
                                         &img[((static_cast<std::size_t>(z) * h + y) * w + x) * texel]);
            }
            out.images.push_back(std::move(img));
            offset += bytes;
            w = std::max(w >> 1, 1u);
            h = std::max(h >> 1, 1u);
            lw = lw ? lw - 1 : 0;
            lh = lh ? lh - 1 : 0;
            if (volume) {
                d = std::max(d >> 1, 1u);
                ld = ld ? ld - 1 : 0;
            }
        }
        face_base += face_bytes;
    }
    return true;
}

}  // namespace mv::pgraph
