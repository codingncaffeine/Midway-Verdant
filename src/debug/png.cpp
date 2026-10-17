#include "debug/png.h"

#include <cstdio>
#include <vector>

namespace mv::png {

namespace {

u32 crc32(const u8* p, std::size_t n, u32 crc = 0xFFFFFFFFu) {
    static u32 table[256];
    static bool ready = false;
    if (!ready) {
        for (u32 i = 0; i < 256; ++i) {
            u32 c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        ready = true;
    }
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

void put32(std::vector<u8>& v, u32 x) {
    v.push_back(static_cast<u8>(x >> 24));
    v.push_back(static_cast<u8>(x >> 16));
    v.push_back(static_cast<u8>(x >> 8));
    v.push_back(static_cast<u8>(x));
}

void chunk(std::vector<u8>& out, const char* type, const std::vector<u8>& data) {
    put32(out, static_cast<u32>(data.size()));
    std::vector<u8> body(type, type + 4);
    body.insert(body.end(), data.begin(), data.end());
    out.insert(out.end(), body.begin(), body.end());
    put32(out, crc32(body.data(), body.size()) ^ 0xFFFFFFFFu);
}

}  // namespace

bool write(const std::filesystem::path& path, u32 width, u32 height, const u8* rgba, u32 stride, std::string& error) {
    // Filter byte 0 before every row, then zlib "stored" blocks (no compression) and the adler-32 of the raw data.
    std::vector<u8> raw;
    raw.reserve((static_cast<std::size_t>(width) * 4 + 1) * height);
    for (u32 y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba + static_cast<std::size_t>(y) * stride, rgba + static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(width) * 4);
    }
    u32 a = 1, b = 0;
    for (u8 c : raw) {
        a = (a + c) % 65521u;
        b = (b + a) % 65521u;
    }
    std::vector<u8> z = {0x78, 0x01};
    std::size_t pos = 0;
    do {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - pos);
        const bool last = pos + n >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<u8>(n));
        z.push_back(static_cast<u8>(n >> 8));
        z.push_back(static_cast<u8>(~n));
        z.push_back(static_cast<u8>((~n) >> 8));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos), raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
        pos += n;
    } while (pos < raw.size());
    put32(z, (b << 16) | a);

    std::vector<u8> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<u8> ihdr;
    put32(ihdr, width);
    put32(ihdr, height);
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(6);  // RGBA
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    chunk(out, "IHDR", ihdr);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});
    std::FILE* f = std::fopen(path.string().c_str(), "wb");
    if (!f) {
        error = "cannot create " + path.string();
        return false;
    }
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return true;
}

}  // namespace mv::png
