#include "smbus/eeprom_image.h"

#include <cstring>
#include <format>

#include "crypto/sha1.h"

namespace mv::eeprom {

namespace {
void put32(u8* p, u32 v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<u8>(v >> (8 * i));
}
u32 get32(const u8* p) {
    return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) | (static_cast<u32>(p[2]) << 16) | (static_cast<u32>(p[3]) << 24);
}
}  // namespace

u32 section_checksum_read(std::span<const u8> section) {
    u64 sum = 0;
    u32 carries = 0;
    for (std::size_t i = 0; i + 4 <= section.size(); i += 4) {
        sum += get32(section.data() + i);
        if (sum > 0xFFFFFFFFu) {
            ++carries;
            sum &= 0xFFFFFFFFu;
        }
    }
    sum += carries;
    if (sum > 0xFFFFFFFFu) sum += 1;
    return static_cast<u32>(sum & 0xFFFFFFFFu);
}

u32 section_checksum_make(std::span<const u8> data) {
    // One's-complement sum of the data; the complement makes the whole section sum to all ones.
    u64 sum = 0;
    for (std::size_t i = 0; i + 4 <= data.size(); i += 4) {
        sum += get32(data.data() + i);
        sum = (sum & 0xFFFFFFFFu) + (sum >> 32);
    }
    sum = (sum & 0xFFFFFFFFu) + (sum >> 32);
    return ~static_cast<u32>(sum);
}

std::array<u8, 256> generate(std::span<const u8, 16> key, const Settings& s) {
    std::array<u8, 256> img{};
    // Encrypted block: confounder (8) || hdd key (16) || region (4), hashed then enciphered.
    std::array<u8, 28> plain{};
    std::memcpy(plain.data(), s.confounder.data(), 8);
    std::memcpy(plain.data() + 8, s.hdd_key.data(), 16);
    put32(plain.data() + 24, s.region);
    const auto data_hash = crypto::hmac_sha1(key, plain);
    const auto rc4_key = crypto::hmac_sha1(key, data_hash);
    std::array<u8, 28> enc = plain;
    crypto::Rc4(rc4_key).process(enc);
    std::memcpy(img.data() + 0x00, data_hash.data(), 20);
    std::memcpy(img.data() + 0x14, enc.data(), 28);
    // Factory section 0x34-0x5F, checksum at 0x30.
    std::memcpy(img.data() + 0x34, s.serial.data(), 12);
    std::memcpy(img.data() + 0x40, s.mac.data(), 6);
    std::memcpy(img.data() + 0x48, s.online_key.data(), 16);
    put32(img.data() + 0x58, s.video_standard);
    put32(img.data() + 0x30, section_checksum_make(std::span<const u8>(img.data() + 0x34, 0x2C)));
    // User section 0x64-0xBF, checksum at 0x60.
    put32(img.data() + 0x64, static_cast<u32>(s.timezone_bias_minutes));
    std::memcpy(img.data() + 0x68, "GMT\0", 4);
    std::memcpy(img.data() + 0x6C, "GDT\0", 4);
    put32(img.data() + 0x78, 0x00020005u);  // standard time starts: month 10? kept zero-ish: unused when bias is 0
    put32(img.data() + 0x78, 0);
    put32(img.data() + 0x7C, 0);
    put32(img.data() + 0x88, 0);
    put32(img.data() + 0x8C, 0);
    put32(img.data() + 0x90, s.language);
    put32(img.data() + 0x94, s.video_settings);
    put32(img.data() + 0x98, s.audio_settings);
    put32(img.data() + 0x9C, 0);  // games parental control: max rating
    put32(img.data() + 0xA0, 0);  // passcode
    put32(img.data() + 0xA4, 0);  // movies parental control
    put32(img.data() + 0xBC, 1);  // DVD playback kit zone 1
    put32(img.data() + 0x60, section_checksum_make(std::span<const u8>(img.data() + 0x64, 0x5C)));
    return img;
}

std::optional<std::array<u8, 16>> verify(std::span<const u8, 16> key, std::span<const u8, 256> image, std::string* why) {
    std::array<u8, 20> stored_hash{};
    std::memcpy(stored_hash.data(), image.data(), 20);
    const auto rc4_key = crypto::hmac_sha1(key, stored_hash);
    std::array<u8, 28> dec{};
    std::memcpy(dec.data(), image.data() + 0x14, 28);
    crypto::Rc4(rc4_key).process(dec);
    const auto computed = crypto::hmac_sha1(key, dec);
    if (computed != stored_hash) {
        if (why) *why = "the encrypted block's hash does not match this key";
        return std::nullopt;
    }
    if (section_checksum_read(std::span<const u8>(image.data() + 0x30, 0x30)) != 0xFFFFFFFFu) {
        if (why) *why = "factory section checksum";
        return std::nullopt;
    }
    if (section_checksum_read(std::span<const u8>(image.data() + 0x60, 0x60)) != 0xFFFFFFFFu) {
        if (why) *why = "user section checksum";
        return std::nullopt;
    }
    std::array<u8, 16> hdd_key{};
    std::memcpy(hdd_key.data(), dec.data() + 8, 16);
    return hdd_key;
}

}  // namespace mv::eeprom
