#include "ide/dvd_image.h"

#include <algorithm>
#include <chrono>
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
    if (file_) std::fclose(file_);
}

bool DvdImage::read_at(u64 offset, u32 bytes, u8* out) const {
    std::lock_guard g(file_mutex_);
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
#if defined(__linux__)
    warmer_ = std::thread([this, p = path, size] { warm_main(p, size); });
#endif
    return true;
}

bool DvdImage::read(u64 lba, u32 count, u8* out) const {
    if (!file_ || lba + count > sectors_) return false;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = read_at(base_ + lba * kSectorSize, count * kSectorSize, out);
    host_read_ns_.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
    return ok;
}

void DvdImage::begin_read(u64 lba, u32 count) {
    std::unique_lock lock(mutex_);
    if (!reader_.joinable()) reader_ = std::thread([this] { reader_main(); });
    done_.wait(lock, [this] { return finished_; });
    req_lba_ = lba;
    req_count_ = count;
    requested_ = true;
    finished_ = false;
    wake_.notify_one();
}

bool DvdImage::end_read(std::vector<u8>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return finished_; });
    out = std::move(result_);
    result_.clear();
    host_read_ns_.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
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
        } else {
            lock.unlock();
            ok = read_at(base_ + lba * kSectorSize, count * kSectorSize, out.data());
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

void DvdImage::warm_main([[maybe_unused]] std::filesystem::path path, [[maybe_unused]] u64 size) {
#if defined(__linux__)
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
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    std::vector<u8> chunk(4u << 20);
    for (u64 off = base_; off < size && !warm_stop_.load(std::memory_order_relaxed);) {
        const ssize_t n = ::pread(fd, chunk.data(), chunk.size(), static_cast<off_t>(off));
        if (n <= 0) break;
        off += static_cast<u64>(n);
        warmed_bytes_.fetch_add(static_cast<u64>(n), std::memory_order_relaxed);
    }
    ::close(fd);
#endif
}

}  // namespace mv
