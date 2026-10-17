#pragma once
#include <cstdio>
#include <filesystem>
#include <string>

#include "core/types.h"

namespace mv {

// A raw disk image on the host: 512-byte sectors, read and written in place. Sparse files stay sparse.
class DiskImage {
public:
    DiskImage() = default;
    ~DiskImage();
    DiskImage(const DiskImage&) = delete;
    DiskImage& operator=(const DiskImage&) = delete;

    bool open(const std::filesystem::path& path, bool read_only, std::string& error);
    bool is_open() const { return file_ != nullptr; }
    u64 sectors() const { return sectors_; }
    bool read_only() const { return read_only_; }
    bool read(u64 lba, u32 count, u8* out) const;
    bool write(u64 lba, u32 count, const u8* in);
    void flush();

private:
    std::FILE* file_{nullptr};
    u64 sectors_{0};
    bool read_only_{false};
};

}  // namespace mv
