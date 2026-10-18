#pragma once
#include <filesystem>
#include <string>

#include "core/types.h"

// A blank console hard disk: a sparse 8 GiB raw image in the retail layout (no partition table; the kernel's
// fixed offsets), the five partitions formatted FATX and E: holding the empty TDATA and UDATA folders games save
// under. C: is empty, so without a game disc the kernel finds no dashboard; games boot. Layout from xboxdevwiki
// Hard_Drive / FATX and our own measurements of a working image (local-notes/hw/ide.md):
//   superblock (4 KiB): "FATX", serial, sectors per cluster, root cluster 1, then 0xFF;
//   allocation table at +0x1000: 16-bit entries under 65525 clusters, 32-bit above, entry 0 = media
//   (0xFFF8 / 0xFFFFFFF8), entry 1 = the root's end of chain, size rounded up to 4 KiB;
//   data area after it, cluster 1 first; free directory entries are 0xFF.
namespace mv {

struct HddPartition {
    const char* name;
    u64 offset, size;
};
inline constexpr HddPartition kRetailPartitions[] = {
    {"X", 0x00080000ull, 0x2EE00000ull}, {"Y", 0x2EE80000ull, 0x2EE00000ull}, {"Z", 0x5DC80000ull, 0x2EE00000ull},
    {"C", 0x8CA80000ull, 0x1F400000ull}, {"E", 0xABE80000ull, 0x131F00000ull},
};
inline constexpr u64 kRetailDiskBytes = 8ull * 1024 * 1024 * 1024;
inline constexpr u32 kFatxSectorsPerCluster = 32;  // 16 KiB clusters

// Where a FATX volume's data area (cluster 1) starts, from its size: the allocation table's length.
u64 fatx_data_offset(u64 partition_bytes, u32 sectors_per_cluster);

// Creates (or overwrites) `path` as a blank retail drive. `serial_seed` seeds the volume serials.
bool create_blank_hdd(const std::filesystem::path& path, u32 serial_seed, std::string& error);

}  // namespace mv
