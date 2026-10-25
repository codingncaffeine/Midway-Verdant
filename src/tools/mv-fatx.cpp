// mv-fatx: reads files from a console hard-disk image's FATX partitions (the retail layout: no partition table, the
// kernel's fixed offsets). Raw and qcow2 images, through the emulator's own disk image reader.
//   mv-fatx ls  IMAGE PARTITION [PATH]          list a folder (partition letter X Y Z C E, path with / or \)
//   mv-fatx get IMAGE PARTITION PATH OUT        copy a file, or a folder recursively, to OUT
// FATX per xboxdevwiki: a 4 KiB superblock ("FATX", serial, sectors per cluster, root cluster), the allocation table
// at +0x1000 (16-bit entries below 65525 clusters, 32-bit above), then the data area, cluster 1 first. Directory
// entries are 64 bytes: name length (0xE5 deleted, 0x00 or 0xFF the end), attributes (0x10 folder), a 42-byte name,
// the first cluster at 0x2C and the size at 0x30.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ide/disk_image.h"
#include "ide/hdd_format.h"

namespace {

using mv::u32;
using mv::u64;
using mv::u8;

struct Entry {
    std::string name;
    bool folder{false};
    u32 cluster{0}, size{0};
};

class Volume {
public:
    bool open(const mv::DiskImage& disk, const mv::HddPartition& p, std::string& error) {
        disk_ = &disk;
        part_ = p;
        u8 super[512];
        if (!read_bytes(p.offset, super, sizeof super) || std::memcmp(super, "FATX", 4) != 0) {
            error = std::string("partition ") + p.name + " has no FATX superblock";
            return false;
        }
        std::memcpy(&sectors_per_cluster_, super + 8, 4);
        std::memcpy(&root_, super + 12, 4);
        cluster_bytes_ = sectors_per_cluster_ * 512;
        clusters_ = p.size / cluster_bytes_;
        wide_ = clusters_ >= 65525;
        data_ = mv::fatx_data_offset(p.size, sectors_per_cluster_);
        return true;
    }
    u32 next(u32 cluster) const {
        u8 b[4]{};
        read_bytes(part_.offset + 0x1000 + u64{cluster} * (wide_ ? 4 : 2), b, wide_ ? 4 : 2);
        u32 v = 0;
        std::memcpy(&v, b, wide_ ? 4 : 2);
        if (!wide_ && v >= 0xFFF0) v |= 0xFFFF0000u;
        return v;
    }
    bool end_of_chain(u32 c) const { return c == 0 || c >= 0xFFFFFFF0u || c > clusters_; }
    u64 cluster_count() const { return clusters_; }
    u32 cluster_size() const { return cluster_bytes_; }
    std::vector<u8> read_chain(u32 cluster, u64 limit) const {
        std::vector<u8> out, buf(cluster_bytes_);
        for (u32 guard = 0; !end_of_chain(cluster) && out.size() < limit && guard < clusters_; ++guard) {
            read_bytes(part_.offset + data_ + u64{cluster - 1} * cluster_bytes_, buf.data(), buf.size());
            out.insert(out.end(), buf.begin(), buf.end());
            cluster = next(cluster);
        }
        if (out.size() > limit) out.resize(limit);
        return out;
    }
    std::vector<Entry> list(u32 cluster) const {
        std::vector<Entry> entries;
        const std::vector<u8> d = read_chain(cluster, ~u64{0});
        for (std::size_t at = 0; at + 64 <= d.size(); at += 64) {
            const u8* e = d.data() + at;
            if (e[0] == 0x00 || e[0] == 0xFF) break;
            if (e[0] == 0xE5 || e[0] > 42) continue;
            Entry x;
            x.name.assign(reinterpret_cast<const char*>(e + 2), e[0]);
            x.folder = (e[1] & 0x10) != 0;
            std::memcpy(&x.cluster, e + 0x2C, 4);
            std::memcpy(&x.size, e + 0x30, 4);
            entries.push_back(x);
        }
        return entries;
    }
    u32 root() const { return root_; }

private:
    bool read_bytes(u64 offset, u8* out, std::size_t len) const {
        // The image reads whole sectors; gather the bytes asked for.
        const u64 first = offset / 512, last = (offset + len + 511) / 512;
        std::vector<u8> sectors((last - first) * 512);
        if (!disk_->read(first, static_cast<u32>(last - first), sectors.data())) return false;
        std::memcpy(out, sectors.data() + (offset - first * 512), len);
        return true;
    }
    const mv::DiskImage* disk_{nullptr};
    mv::HddPartition part_{};
    u32 sectors_per_cluster_{0}, root_{1}, cluster_bytes_{0};
    u64 clusters_{0}, data_{0};
    bool wide_{false};
};

bool equal_nocase(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
    return true;
}

// Walks PATH from the root; the entry it names (a synthetic folder entry for the root).
bool resolve(const Volume& v, const std::string& path, Entry& out) {
    out = {"", true, v.root(), 0};
    std::size_t at = 0;
    while (at < path.size()) {
        std::size_t end = path.find_first_of("/\\", at);
        if (end == std::string::npos) end = path.size();
        const std::string part = path.substr(at, end - at);
        at = end + 1;
        if (part.empty()) continue;
        if (!out.folder) return false;
        bool found = false;
        for (const Entry& e : v.list(out.cluster))
            if (equal_nocase(e.name, part)) {
                out = e;
                found = true;
                break;
            }
        if (!found) return false;
    }
    return true;
}

bool copy_out(const Volume& v, const Entry& e, const std::filesystem::path& to) {
    if (e.folder) {
        std::filesystem::create_directories(to);
        bool ok = true;
        for (const Entry& c : v.list(e.cluster)) ok = copy_out(v, c, to / c.name) && ok;
        return ok;
    }
    const std::vector<u8> bytes = v.read_chain(e.cluster, e.size);
    std::ofstream f(to, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: mv-fatx ls IMAGE PARTITION [PATH]\n       mv-fatx get IMAGE PARTITION PATH OUT\n       mv-fatx chain IMAGE PARTITION PATH\n");
        return 2;
    }
    const std::string cmd = argv[1];
    mv::DiskImage disk;
    std::string error;
    if (!disk.open(argv[2], true, error)) {
        std::fprintf(stderr, "mv-fatx: %s\n", error.c_str());
        return 1;
    }
    const mv::HddPartition* part = nullptr;
    for (const auto& p : mv::kRetailPartitions)
        if (std::toupper(static_cast<unsigned char>(argv[3][0])) == p.name[0]) part = &p;
    if (!part) {
        std::fprintf(stderr, "mv-fatx: no partition %s (X Y Z C E)\n", argv[3]);
        return 2;
    }
    Volume v;
    if (!v.open(disk, *part, error)) {
        std::fprintf(stderr, "mv-fatx: %s\n", error.c_str());
        return 1;
    }
    Entry e;
    const std::string path = argc > 4 ? argv[4] : "";
    if (!resolve(v, path, e)) {
        std::fprintf(stderr, "mv-fatx: %s not found on %s:\n", path.c_str(), part->name);
        return 1;
    }
    if (cmd == "ls") {
        if (!e.folder) {
            std::printf("%10u  %s\n", e.size, e.name.c_str());
            return 0;
        }
        for (const Entry& c : v.list(e.cluster)) std::printf("%10u  %s%s\n", c.size, c.name.c_str(), c.folder ? "/" : "");
        return 0;
    }
    if (cmd == "get" && argc > 5) return copy_out(v, e, argv[5]) ? 0 : 1;
    if (cmd == "chain") {
        // The cluster chain of PATH as runs of consecutive clusters (first-last), to tell a broken chain from bad data.
        u32 run_start = e.cluster, prev = e.cluster, n = 0;
        for (u32 c = e.cluster; !v.end_of_chain(c) && n <= v.cluster_count(); c = v.next(c), ++n) {
            if (c != e.cluster && c != prev + 1) {
                std::printf("%u-%u\n", run_start, prev);
                run_start = c;
            }
            prev = c;
        }
        std::printf("%u-%u (%u clusters of %u bytes)\n", run_start, prev, n, v.cluster_size());
        return 0;
    }
    std::fprintf(stderr, "mv-fatx: unknown command %s\n", cmd.c_str());
    return 2;
}
