#pragma once
#include <array>
#include <cstddef>
#include <span>

#include "core/types.h"

namespace mv::crypto {

// SHA-1 as specified in FIPS 180-1 (the console uses it for the EEPROM, HDD password and disc security).
class Sha1 {
public:
    using Digest = std::array<u8, 20>;
    Sha1();
    void update(std::span<const u8> data);
    void update(const void* data, std::size_t len) { update(std::span<const u8>(static_cast<const u8*>(data), len)); }
    Digest finish();
    static Digest hash(std::span<const u8> data) {
        Sha1 s;
        s.update(data);
        return s.finish();
    }

private:
    void process_block(const u8* block);
    std::array<u32, 5> h_;
    std::array<u8, 64> buffer_{};
    std::size_t buffered_{0};
    u64 total_bytes_{0};
};

// HMAC-SHA1 (RFC 2104).
Sha1::Digest hmac_sha1(std::span<const u8> key, std::span<const u8> message);
Sha1::Digest hmac_sha1(std::span<const u8> key, std::span<const u8> part1, std::span<const u8> part2);

// RC4 stream cipher (the 1987 cipher as publicly described; encryption and decryption are the same operation).
class Rc4 {
public:
    explicit Rc4(std::span<const u8> key);
    void process(std::span<u8> data);

private:
    std::array<u8, 256> s_{};
    u8 i_{0}, j_{0};
};

}  // namespace mv::crypto
