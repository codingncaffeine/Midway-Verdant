#pragma once
#include <array>
#include <optional>
#include <string>

#include "core/bus.h"

namespace mv {

// Reads the unpacked kernel (a PE image at 0x80010000, identity-mapped at physical 0x10000) out of guest RAM:
// its export table has no names, every export is by ordinal, and a few exports are data the emulator needs to
// know about its own guest — the EEPROM key the kernel validates the EEPROM with, and the version.
class KernelImage {
public:
    static constexpr PhysAddr kBase = 0x10000;
    static constexpr u32 kVirtualBase = 0x80010000;
    static constexpr u32 kOrdinalEepromKey = 321;
    static constexpr u32 kOrdinalHardwareInfo = 322;
    static constexpr u32 kOrdinalKrnlVersion = 324;

    struct Version {
        u16 major, minor, build, qfe;
    };

    explicit KernelImage(const Bus& bus) : bus_(bus) {}

    // True when a PE image with an export table is present at the base.
    bool present(std::string* why = nullptr) const;
    // The physical address an export's data lives at, or nullopt.
    std::optional<PhysAddr> export_address(u32 ordinal) const;
    std::optional<Version> version() const;
    std::optional<std::array<u8, 16>> eeprom_key() const;

private:
    u32 read32(PhysAddr a) const;
    u16 read16(PhysAddr a) const;
    const Bus& bus_;
};

}  // namespace mv
