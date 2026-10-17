#pragma once
#include <array>
#include <optional>
#include <span>
#include <string>

#include "core/types.h"

namespace mv {

// The 256-byte EEPROM as the kernel reads it: an encrypted block (HDD key and region) authenticated with the
// kernel's own EEPROM key, then the factory and user sections, each guarded by a one's-complement checksum.
// Layout and algorithms per xboxdevwiki EEPROM and Hard_Drive pages (see local-notes/hw/smbus-smc-eeprom.md).
namespace eeprom {

constexpr u32 kRegionNorthAmerica = 0x00000001, kRegionJapan = 0x00000002, kRegionEurope = 0x00000004;
constexpr u32 kVideoNtscM = 0x00400100, kVideoNtscJ = 0x00400200, kVideoPalI = 0x00800300, kVideoPalM = 0x00400400;
constexpr u32 kLanguageEnglish = 1;

struct Settings {
    std::array<u8, 16> hdd_key{};
    u32 region{kRegionNorthAmerica};
    std::array<u8, 12> serial{{'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '1', '2'}};
    std::array<u8, 6> mac{{0x00, 0x50, 0xF2, 0x12, 0x34, 0x56}};
    std::array<u8, 16> online_key{};
    u32 video_standard{kVideoNtscM};
    u32 language{kLanguageEnglish};
    u32 video_settings{0};
    u32 audio_settings{0};
    i32 timezone_bias_minutes{0};
    std::array<u8, 8> confounder{{0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}};
};

// The one's-complement sum the kernel checks: over the whole section (checksum dword first) it must be all ones.
u32 section_checksum_read(std::span<const u8> section);
// The checksum dword that makes `data` (the section without its checksum) verify.
u32 section_checksum_make(std::span<const u8> data);

// Build an image the kernel with `eeprom_key` accepts.
std::array<u8, 256> generate(std::span<const u8, 16> eeprom_key, const Settings& s);

// Check an image against a key: returns the decrypted HDD key when the hash verifies.
std::optional<std::array<u8, 16>> verify(std::span<const u8, 16> eeprom_key, std::span<const u8, 256> image, std::string* why = nullptr);

}  // namespace eeprom
}  // namespace mv
