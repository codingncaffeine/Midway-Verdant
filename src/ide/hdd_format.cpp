#include "ide/hdd_format.h"

#include <cstdio>
#include <cstring>
#include <format>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#endif

namespace mv {

namespace {

constexpr u64 kClusterLimit16 = 65525;  // fewer clusters than this: 16-bit allocation entries

u64 round_up(u64 v, u64 to) { return (v + to - 1) / to * to; }

bool write_at(std::FILE* f, u64 offset, const void* data, std::size_t len) {
#if defined(_WIN32)
    if (_fseeki64(f, static_cast<long long>(offset), SEEK_SET) != 0) return false;
#else
    if (fseeko(f, static_cast<off_t>(offset), SEEK_SET) != 0) return false;
#endif
    return std::fwrite(data, 1, len, f) == len;
}

// A 64-byte directory entry for an empty folder.
void folder_entry(u8* e, const char* name, u32 cluster) {
    std::memset(e, 0xFF, 64);
    const std::size_t n = std::strlen(name);
    e[0] = static_cast<u8>(n);
    e[1] = 0x10;  // directory
    std::memcpy(e + 2, name, n);
    std::memcpy(e + 0x2C, &cluster, 4);
    const u32 size = 0;
    std::memcpy(e + 0x30, &size, 4);
    const u32 stamp = 0x2D010000u;  // 2002-08-01 00:00, FAT date in the high half
    for (int i = 0; i < 3; ++i) std::memcpy(e + 0x34 + 4 * i, &stamp, 4);
}

// Marks a freshly created file sparse where the filesystem needs to be told: NTFS allocates the whole length of
// an extended file unless the sparse attribute is set first; ext4, btrfs, xfs and APFS leave unwritten ranges
// unallocated by themselves.
bool make_sparse(const std::filesystem::path& path) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD returned = 0;
    const BOOL ok = DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);
    CloseHandle(h);
    return ok != FALSE;
#else
    (void)path;
    return true;
#endif
}

bool format_fatx(std::FILE* f, const HddPartition& p, u32 serial, bool with_save_folders) {
    const u64 cluster_bytes = u64{kFatxSectorsPerCluster} * 512;
    const u64 clusters = p.size / cluster_bytes;
    const bool wide = clusters >= kClusterLimit16;
    const u64 data = fatx_data_offset(p.size, kFatxSectorsPerCluster);

    std::vector<u8> super(4096, 0xFF);
    std::memcpy(super.data(), "FATX", 4);
    std::memcpy(super.data() + 4, &serial, 4);
    const u32 spc = kFatxSectorsPerCluster, root = 1;
    std::memcpy(super.data() + 8, &spc, 4);
    std::memcpy(super.data() + 12, &root, 4);
    super[16] = super[17] = 0;
    if (!write_at(f, p.offset, super.data(), super.size())) return false;

    // The allocation table's head: media, the root's end of chain, and E:'s two folders (one cluster each).
    std::vector<u8> fat(4096, 0);
    const unsigned used = with_save_folders ? 4 : 2;
    for (unsigned i = 0; i < used; ++i) {
        if (wide) {
            const u32 v = i == 1 ? 0xFFFFFFFFu : 0xFFFFFFF8u;
            std::memcpy(fat.data() + 4 * i, &v, 4);
        } else {
            const u16 v = i == 1 ? 0xFFFF : 0xFFF8;
            std::memcpy(fat.data() + 2 * i, &v, 2);
        }
    }
    if (!write_at(f, p.offset + 0x1000, fat.data(), fat.size())) return false;
    // The rest of the table is zero: the sparse file already reads as zero.

    std::vector<u8> cluster(cluster_bytes, 0xFF);
    if (with_save_folders) {
        folder_entry(cluster.data(), "TDATA", 2);
        folder_entry(cluster.data() + 64, "UDATA", 3);
    }
    if (!write_at(f, p.offset + data, cluster.data(), cluster.size())) return false;
    if (with_save_folders) {
        std::vector<u8> empty(cluster_bytes, 0xFF);
        if (!write_at(f, p.offset + data + cluster_bytes, empty.data(), empty.size())) return false;
        if (!write_at(f, p.offset + data + 2 * cluster_bytes, empty.data(), empty.size())) return false;
    }
    return true;
}

}  // namespace

u64 fatx_data_offset(u64 partition_bytes, u32 sectors_per_cluster) {
    const u64 clusters = partition_bytes / (u64{sectors_per_cluster} * 512);
    const u64 entry = clusters >= kClusterLimit16 ? 4 : 2;
    return 0x1000 + round_up((clusters + 1) * entry, 4096);
}

bool create_blank_hdd(const std::filesystem::path& path, u32 serial_seed, std::string& error) {
    std::FILE* f = std::fopen(path.string().c_str(), "wb");
    if (!f) {
        error = std::format("cannot create {}", path.string());
        return false;
    }
    // Sparse before anything lands past the first bytes: on NTFS an un-flagged file would allocate all 8 GB.
    std::fclose(f);
    if (!make_sparse(path)) {
        error = std::format("cannot make {} sparse", path.string());
        return false;
    }
    f = std::fopen(path.string().c_str(), "r+b");
    if (!f) {
        error = std::format("cannot reopen {}", path.string());
        return false;
    }
    bool ok = true;
    u32 serial = serial_seed ? serial_seed : 0x4D560001u;
    for (const HddPartition& p : kRetailPartitions) {
        ok = ok && format_fatx(f, p, serial, std::strcmp(p.name, "E") == 0);
        serial += 0x01010101u;
    }
    std::fclose(f);
    if (!ok) {
        error = std::format("writing {} failed", path.string());
        return false;
    }
    // The full length, unwritten space left sparse (it reads as zero).
    std::error_code ec;
    std::filesystem::resize_file(path, kRetailDiskBytes, ec);
    if (ec) {
        error = std::format("cannot size {}: {}", path.string(), ec.message());
        return false;
    }
    return true;
}

}  // namespace mv
