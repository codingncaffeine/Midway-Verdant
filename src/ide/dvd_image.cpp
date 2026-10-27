#include "ide/dvd_image.h"

#include "ide/disk_image.h"  // note_slow_io

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <format>
#if defined(__linux__)
#include <fcntl.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace mv {

namespace {
constexpr char kVolumeMagic[] = "MICROSOFT*XBOX*MEDIA";
constexpr u64 kVolumeDescriptorOffset = 32 * DvdImage::kSectorSize;
}  // namespace

DvdImage::~DvdImage() {
    warm_stop_ = true;
    if (warmer_.joinable()) warmer_.join();
    {
        std::lock_guard g(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (reader_.joinable()) reader_.join();
    for (Part& p : parts_)
        if (p.file) std::fclose(p.file);
}

// Bytes of the whole image, across the parts they fall in.
bool DvdImage::read_at(u64 offset, u32 bytes, u8* out) const {
    std::lock_guard g(file_mutex_);
    for (const Part& p : parts_) {
        if (bytes == 0) break;
        if (offset >= p.start + p.size) continue;
        if (offset < p.start) return false;
        const u32 n = static_cast<u32>(std::min<u64>(bytes, p.start + p.size - offset));
#if defined(_WIN32)
        if (_fseeki64(p.file, static_cast<long long>(offset - p.start), SEEK_SET) != 0) return false;
#else
        if (fseeko(p.file, static_cast<off_t>(offset - p.start), SEEK_SET) != 0) return false;
#endif
        if (std::fread(out, 1, n, p.file) != n) return false;
        offset += n;
        out += n;
        bytes -= n;
    }
    return bytes == 0;
}

void DvdImage::note_past_end(u64 lba, u32 count) const {
    if (past_end_reads_.fetch_add(1, std::memory_order_relaxed) == 0) first_past_end_lba_.store(lba, std::memory_order_relaxed);
    (void)count;
}

bool DvdImage::has_volume_descriptor(u64 partition) const {
    u8 magic[sizeof kVolumeMagic - 1];
    return read_at(partition + kVolumeDescriptorOffset, sizeof magic, magic) && std::memcmp(magic, kVolumeMagic, sizeof magic) == 0;
}

bool DvdImage::open(const std::filesystem::path& path, std::string& error) {
    // The parts: the file itself, and for NAME.1.EXT the NAME.2.EXT, NAME.3.EXT ... that follow it.
    std::vector<std::filesystem::path> paths{path};
    const std::string stem = path.stem().string();
    const bool first_part = stem.size() > 2 && stem.ends_with(".1");
    if (first_part) {
        const std::string name = stem.substr(0, stem.size() - 2);
        for (int n = 2; n < 100; ++n) {
            const auto next = path.parent_path() / std::format("{}.{}{}", name, n, path.extension().string());
            std::error_code exists_ec;
            if (!std::filesystem::is_regular_file(next, exists_ec)) break;
            paths.push_back(next);
        }
        if (paths.size() == 1)
            note_ = std::format("{} is the first part of a split image and {}.2{} is not beside it: the rest of the disc is missing", path.filename().string(),
                                name, path.extension().string());
    }
    for (const auto& p : paths) {
        std::error_code ec;
        const u64 part_size = std::filesystem::file_size(p, ec);
        if (ec) {
            error = std::format("cannot stat {}: {}", p.string(), ec.message());
            return false;
        }
        std::FILE* f = std::fopen(p.string().c_str(), "rb");
        if (!f) {
            error = std::format("cannot open {}", p.string());
            return false;
        }
        parts_.push_back({p, f, size_, part_size});
        size_ += part_size;
    }
    const u64 size = size_;
    if (has_volume_descriptor(0)) {
        kind_ = Kind::Xiso;
        base_ = 0;
    } else if (size > kRedumpGamePartition + kVolumeDescriptorOffset && has_volume_descriptor(kRedumpGamePartition)) {
        kind_ = Kind::Redump;
        base_ = kRedumpGamePartition;
    } else {
        for (Part& p : parts_) std::fclose(p.file);
        parts_.clear();
        error = std::format("{} is not an Xbox game disc image (no XDVDFS volume at sector 32, nor at the game partition of a full dump)",
                            path.string());
        return false;
    }
    sectors_ = (size - base_) / kSectorSize;
#if defined(__linux__)
    warmer_ = std::thread([this] { warm_main(); });
#endif
    return true;
}

bool DvdImage::read(u64 lba, u32 count, u8* out) const {
    io_log("dvd-read", lba, count);
    if (parts_.empty()) return false;
    if (lba + count > sectors_) {
        note_past_end(lba, count);
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = read_at(base_ + lba * kSectorSize, count * kSectorSize, out);
    const auto ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    host_read_ns_.fetch_add(ns, std::memory_order_relaxed);
    note_slow_io("dvd read", lba, count, ns);
    return ok;
}

void DvdImage::begin_read(u64 lba, u32 count) {
    io_log("dvd-read", lba, count);
    std::unique_lock lock(mutex_);
    if (!reader_.joinable()) reader_ = std::thread([this] { reader_main(); });
    done_.wait(lock, [this] { return finished_; });
    req_lba_ = lba;
    req_count_ = count;
    requested_ = true;
    finished_ = false;
    wake_.notify_one();
}

bool DvdImage::ready() const {
    std::lock_guard lock(mutex_);
    return finished_;
}

bool DvdImage::end_read(std::vector<u8>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return finished_; });
    out = std::move(result_);
    result_.clear();
    const auto ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    host_read_ns_.fetch_add(ns, std::memory_order_relaxed);
    note_slow_io("dvd wait", req_lba_, req_count_, ns);
    return ok_;
}

void DvdImage::reader_main() {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "mv-dvd-reader");
#endif
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_.wait(lock, [this] { return stop_ || requested_; });
        if (stop_) return;
        requested_ = false;
        const u64 lba = req_lba_;
        const u32 count = req_count_;
        std::vector<u8> out(static_cast<std::size_t>(count) * kSectorSize);
        bool ok = true;
        if (ahead_count_ && lba >= ahead_lba_ && lba + count <= ahead_lba_ + ahead_count_) {
            std::memcpy(out.data(), ahead_.data() + static_cast<std::size_t>(lba - ahead_lba_) * kSectorSize, out.size());
            read_ahead_hits_.fetch_add(1, std::memory_order_relaxed);
        } else if (lba + count > sectors_) {
            note_past_end(lba, count);
            ok = false;
        } else {
            lock.unlock();
            ok = read_at(base_ + lba * kSectorSize, count * kSectorSize, out.data());
            // Instrument: MV_DEBUG_DVD_DELAY_MS=N makes every read the read-ahead missed take N ms more (a cold spinning disc).
            static const long delay_ms = [] { const char* d = std::getenv("MV_DEBUG_DVD_DELAY_MS"); return d && *d ? std::atol(d) : 0L; }();
            if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            lock.lock();
        }
        result_ = std::move(out);
        ok_ = ok;
        finished_ = true;
        done_.notify_all();
        // On ahead from the read's end, unless the next request is already waiting or the block still holds half a
        // read-ahead's worth past it.
        const u64 next = lba + count;
        if (requested_ || stop_ || next >= sectors_) continue;
        if (ahead_count_ && next >= ahead_lba_ && next + kReadAheadSectors / 2 <= ahead_lba_ + ahead_count_) continue;
        const u32 n = static_cast<u32>(std::min<u64>(kReadAheadSectors, sectors_ - next));
        std::vector<u8> block(static_cast<std::size_t>(n) * kSectorSize);
        lock.unlock();
        const bool block_ok = read_at(base_ + next * kSectorSize, n * kSectorSize, block.data());
        lock.lock();
        if (block_ok) {
            ahead_ = std::move(block);
            ahead_lba_ = next;
            ahead_count_ = n;
        }
    }
}

void DvdImage::warm_main() {
#if defined(__linux__)
    const u64 size = size_;
    pthread_setname_np(pthread_self(), "mv-dvd-warm");
    // Only when the image fits in half the memory the host has available.
    u64 available_kb = 0;
    if (std::FILE* m = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, m))
            if (std::sscanf(line, "MemAvailable: %llu kB", reinterpret_cast<unsigned long long*>(&available_kb)) == 1) break;
        std::fclose(m);
    }
    if (available_kb == 0 || size / 1024 > available_kb / 2) return;
    // The idle I/O class: the drive's own reads go first.
    constexpr int kIoprioWhoProcess = 1, kIoprioClassIdle = 3, kIoprioClassShift = 13;
    syscall(SYS_ioprio_set, kIoprioWhoProcess, 0, kIoprioClassIdle << kIoprioClassShift);
    std::vector<u8> chunk(4u << 20);
    for (const Part& p : parts_) {
        if (p.start + p.size <= base_) continue;
        const int fd = ::open(p.path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return;
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        for (u64 off = base_ > p.start ? base_ - p.start : 0; off < p.size && !warm_stop_.load(std::memory_order_relaxed);) {
            const ssize_t n = ::pread(fd, chunk.data(), chunk.size(), static_cast<off_t>(off));
            if (n <= 0) break;
            off += static_cast<u64>(n);
            warmed_bytes_.fetch_add(static_cast<u64>(n), std::memory_order_relaxed);
        }
        ::close(fd);
    }
#endif
}

}  // namespace mv
