#include "ide/disk_image.h"

#include <format>

namespace mv {

DiskImage::~DiskImage() {
    if (file_) std::fclose(file_);
}

bool DiskImage::open(const std::filesystem::path& path, bool read_only, std::string& error) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        error = std::format("cannot stat {}: {}", path.string(), ec.message());
        return false;
    }
    if (size % 512 != 0) {
        error = std::format("{} is not a whole number of 512-byte sectors", path.string());
        return false;
    }
    file_ = std::fopen(path.string().c_str(), read_only ? "rb" : "r+b");
    if (!file_) {
        error = std::format("cannot open {}", path.string());
        return false;
    }
    read_only_ = read_only;
    sectors_ = size / 512;
    return true;
}

bool DiskImage::read(u64 lba, u32 count, u8* out) const {
    if (!file_ || lba + count > sectors_) return false;
#if defined(_WIN32)
    if (_fseeki64(file_, static_cast<long long>(lba * 512), SEEK_SET) != 0) return false;
#else
    if (fseeko(file_, static_cast<off_t>(lba * 512), SEEK_SET) != 0) return false;
#endif
    return std::fread(out, 512, count, file_) == count;
}

bool DiskImage::write(u64 lba, u32 count, const u8* in) {
    if (!file_ || read_only_ || lba + count > sectors_) return false;
#if defined(_WIN32)
    if (_fseeki64(file_, static_cast<long long>(lba * 512), SEEK_SET) != 0) return false;
#else
    if (fseeko(file_, static_cast<off_t>(lba * 512), SEEK_SET) != 0) return false;
#endif
    return std::fwrite(in, 512, count, file_) == count;
}

void DiskImage::flush() {
    if (file_) std::fflush(file_);
}

}  // namespace mv
