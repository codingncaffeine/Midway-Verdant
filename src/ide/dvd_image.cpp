#include "ide/dvd_image.h"

#include <cstring>
#include <format>

namespace mv {

namespace {
constexpr char kVolumeMagic[] = "MICROSOFT*XBOX*MEDIA";
constexpr u64 kVolumeDescriptorOffset = 32 * DvdImage::kSectorSize;
}  // namespace

DvdImage::~DvdImage() {
    if (file_) std::fclose(file_);
}

bool DvdImage::read_at(u64 offset, u32 bytes, u8* out) const {
#if defined(_WIN32)
    if (_fseeki64(file_, static_cast<long long>(offset), SEEK_SET) != 0) return false;
#else
    if (fseeko(file_, static_cast<off_t>(offset), SEEK_SET) != 0) return false;
#endif
    return std::fread(out, 1, bytes, file_) == bytes;
}

bool DvdImage::has_volume_descriptor(u64 partition) const {
    u8 magic[sizeof kVolumeMagic - 1];
    return read_at(partition + kVolumeDescriptorOffset, sizeof magic, magic) && std::memcmp(magic, kVolumeMagic, sizeof magic) == 0;
}

bool DvdImage::open(const std::filesystem::path& path, std::string& error) {
    std::error_code ec;
    const u64 size = std::filesystem::file_size(path, ec);
    if (ec) {
        error = std::format("cannot stat {}: {}", path.string(), ec.message());
        return false;
    }
    file_ = std::fopen(path.string().c_str(), "rb");
    if (!file_) {
        error = std::format("cannot open {}", path.string());
        return false;
    }
    if (has_volume_descriptor(0)) {
        kind_ = Kind::Xiso;
        base_ = 0;
    } else if (size > kRedumpGamePartition + kVolumeDescriptorOffset && has_volume_descriptor(kRedumpGamePartition)) {
        kind_ = Kind::Redump;
        base_ = kRedumpGamePartition;
    } else {
        std::fclose(file_);
        file_ = nullptr;
        error = std::format("{} is not an Xbox game disc image (no XDVDFS volume at sector 32, nor at the game partition of a full dump)",
                            path.string());
        return false;
    }
    sectors_ = (size - base_) / kSectorSize;
    return true;
}

bool DvdImage::read(u64 lba, u32 count, u8* out) const {
    if (!file_ || lba + count > sectors_) return false;
    return read_at(base_ + lba * kSectorSize, count * kSectorSize, out);
}

}  // namespace mv
