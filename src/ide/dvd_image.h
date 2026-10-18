#pragma once
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

#include "core/types.h"

namespace mv {

// A game disc image on the host, read in 2048-byte sectors of the disc's Xbox (game) partition: an XISO holds that
// partition alone; a full dump (the Redump layout) holds the video partition first and the game partition at byte
// 0x18300000. Either is recognised by the XDVDFS volume descriptor ("MICROSOFT*XBOX*MEDIA") in sector 32 of the
// partition. Read-only.
class DvdImage {
public:
    enum class Kind { Xiso, Redump };
    static constexpr u32 kSectorSize = 2048;
    static constexpr u64 kRedumpGamePartition = 0x18300000;

    DvdImage() = default;
    ~DvdImage();
    DvdImage(const DvdImage&) = delete;
    DvdImage& operator=(const DvdImage&) = delete;

    bool open(const std::filesystem::path& path, std::string& error);
    bool is_open() const { return file_ != nullptr; }
    Kind kind() const { return kind_; }
    std::string_view kind_name() const { return kind_ == Kind::Xiso ? "XISO" : "full dump"; }
    // Sectors of the game partition.
    u64 sectors() const { return sectors_; }
    bool read(u64 lba, u32 count, u8* out) const;

private:
    bool read_at(u64 offset, u32 bytes, u8* out) const;
    bool has_volume_descriptor(u64 partition) const;

    std::FILE* file_{nullptr};
    Kind kind_{Kind::Xiso};
    u64 base_{0};
    u64 sectors_{0};
};

}  // namespace mv
