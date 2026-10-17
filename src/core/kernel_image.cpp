#include "core/kernel_image.h"

#include <cstring>

namespace mv {

u32 KernelImage::read32(PhysAddr a) const {
    u32 v = 0;
    bus_.read_ram(a, &v, 4);
    return v;
}

u16 KernelImage::read16(PhysAddr a) const {
    u16 v = 0;
    bus_.read_ram(a, &v, 2);
    return v;
}

bool KernelImage::present(std::string* why) const {
    u8 mz[2]{};
    if (!bus_.read_ram(kBase, mz, 2) || mz[0] != 'M' || mz[1] != 'Z') {
        if (why) *why = "no MZ header at 0x80010000";
        return false;
    }
    const u32 pe = read32(kBase + 0x3C);
    if (pe == 0 || pe > 0x1000 || read32(kBase + pe) != 0x00004550u) {  // "PE\0\0"
        if (why) *why = "no PE signature";
        return false;
    }
    return export_address(kOrdinalKrnlVersion).has_value();
}

std::optional<PhysAddr> KernelImage::export_address(u32 ordinal) const {
    const u32 pe = read32(kBase + 0x3C);
    if (pe == 0 || pe > 0x1000) return std::nullopt;
    const PhysAddr opt = kBase + pe + 24;  // optional header
    const u16 magic = read16(opt);
    if (magic != 0x10B) return std::nullopt;  // PE32
    const u32 image_base = read32(opt + 28);
    const u32 dirs = read32(opt + 92);
    if (dirs < 1) return std::nullopt;
    const u32 export_rva = read32(opt + 96);
    const u32 export_size = read32(opt + 100);
    if (export_rva == 0 || export_size == 0) return std::nullopt;
    (void)image_base;
    const PhysAddr exp = kBase + export_rva;  // sections are identity-mapped: RVA == file offset == RAM offset
    const u32 base_ordinal = read32(exp + 16);
    const u32 count = read32(exp + 20);
    const u32 functions_rva = read32(exp + 28);
    if (ordinal < base_ordinal || ordinal - base_ordinal >= count || count > 1024) return std::nullopt;
    const u32 rva = read32(kBase + functions_rva + 4 * (ordinal - base_ordinal));
    if (rva == 0) return std::nullopt;
    return kBase + rva;
}

std::optional<KernelImage::Version> KernelImage::version() const {
    const auto at = export_address(kOrdinalKrnlVersion);
    if (!at) return std::nullopt;
    Version v{read16(*at), read16(*at + 2), read16(*at + 4), read16(*at + 6)};
    return v;
}

std::optional<std::array<u8, 16>> KernelImage::eeprom_key() const {
    const auto at = export_address(kOrdinalEepromKey);
    if (!at) return std::nullopt;
    std::array<u8, 16> key{};
    if (!bus_.read_ram(*at, key.data(), 16)) return std::nullopt;
    return key;
}

}  // namespace mv
