#include "crypto/sha1.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace mv::crypto {

Sha1::Sha1() : h_{0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u} {}

void Sha1::process_block(const u8* block) {
    u32 w[80];
    for (int t = 0; t < 16; ++t) {
        w[t] = (static_cast<u32>(block[4 * t]) << 24) | (static_cast<u32>(block[4 * t + 1]) << 16) |
               (static_cast<u32>(block[4 * t + 2]) << 8) | static_cast<u32>(block[4 * t + 3]);
    }
    for (int t = 16; t < 80; ++t) w[t] = std::rotl(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1);
    u32 a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
    for (int t = 0; t < 80; ++t) {
        u32 f, k;
        if (t < 20) {
            f = (b & c) | (~b & d);
            k = 0x5A827999u;
        } else if (t < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1u;
        } else if (t < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDCu;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6u;
        }
        const u32 temp = std::rotl(a, 5) + f + e + k + w[t];
        e = d;
        d = c;
        c = std::rotl(b, 30);
        b = a;
        a = temp;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
}

void Sha1::update(std::span<const u8> data) {
    if (data.empty()) return;  // an empty span may hold a null pointer, which memcpy must never see
    total_bytes_ += data.size();
    std::size_t off = 0;
    if (buffered_) {
        const std::size_t take = std::min(data.size(), buffer_.size() - buffered_);
        std::memcpy(buffer_.data() + buffered_, data.data(), take);
        buffered_ += take;
        off = take;
        if (buffered_ == buffer_.size()) {
            process_block(buffer_.data());
            buffered_ = 0;
        }
    }
    while (data.size() - off >= 64) {
        process_block(data.data() + off);
        off += 64;
    }
    if (off < data.size()) {
        std::memcpy(buffer_.data(), data.data() + off, data.size() - off);
        buffered_ = data.size() - off;
    }
}

Sha1::Digest Sha1::finish() {
    const u64 bits = total_bytes_ * 8;
    const u8 one = 0x80;
    update(std::span<const u8>(&one, 1));
    const u8 zero = 0;
    while (buffered_ != 56) update(std::span<const u8>(&zero, 1));
    u8 len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<u8>(bits >> (56 - 8 * i));
    update(std::span<const u8>(len, 8));
    Digest d;
    for (int i = 0; i < 5; ++i) {
        d[static_cast<std::size_t>(4 * i)] = static_cast<u8>(h_[static_cast<std::size_t>(i)] >> 24);
        d[static_cast<std::size_t>(4 * i + 1)] = static_cast<u8>(h_[static_cast<std::size_t>(i)] >> 16);
        d[static_cast<std::size_t>(4 * i + 2)] = static_cast<u8>(h_[static_cast<std::size_t>(i)] >> 8);
        d[static_cast<std::size_t>(4 * i + 3)] = static_cast<u8>(h_[static_cast<std::size_t>(i)]);
    }
    return d;
}

Sha1::Digest hmac_sha1(std::span<const u8> key, std::span<const u8> part1, std::span<const u8> part2) {
    std::array<u8, 64> k{};
    if (key.size() > 64) {
        const auto hashed = Sha1::hash(key);
        std::memcpy(k.data(), hashed.data(), hashed.size());
    } else {
        std::memcpy(k.data(), key.data(), key.size());
    }
    std::array<u8, 64> ipad, opad;
    for (std::size_t i = 0; i < 64; ++i) {
        ipad[i] = static_cast<u8>(k[i] ^ 0x36);
        opad[i] = static_cast<u8>(k[i] ^ 0x5C);
    }
    Sha1 inner;
    inner.update(ipad);
    inner.update(part1);
    inner.update(part2);
    const auto inner_digest = inner.finish();
    Sha1 outer;
    outer.update(opad);
    outer.update(inner_digest);
    return outer.finish();
}

Sha1::Digest hmac_sha1(std::span<const u8> key, std::span<const u8> message) {
    return hmac_sha1(key, message, std::span<const u8>{});
}

Rc4::Rc4(std::span<const u8> key) {
    for (std::size_t i = 0; i < 256; ++i) s_[i] = static_cast<u8>(i);
    u8 j = 0;
    for (std::size_t i = 0; i < 256; ++i) {
        j = static_cast<u8>(j + s_[i] + key[i % key.size()]);
        std::swap(s_[i], s_[j]);
    }
}

void Rc4::process(std::span<u8> data) {
    for (u8& byte : data) {
        i_ = static_cast<u8>(i_ + 1);
        j_ = static_cast<u8>(j_ + s_[i_]);
        std::swap(s_[i_], s_[j_]);
        byte ^= s_[static_cast<u8>(s_[i_] + s_[j_])];
    }
}

}  // namespace mv::crypto
