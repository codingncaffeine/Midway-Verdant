#include "ide/disk_image.h"

#include <mutex>

#include "debug/trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace mv {

namespace {
// Adds the host time from its construction to its destruction to an image's I/O instrument.
struct Timed {
    std::atomic<u64>& total;
    const char* what;
    u64 lba{0};
    u32 count{0};
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~Timed() {
        const auto ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        total.fetch_add(ns, std::memory_order_relaxed);
        note_slow_io(what, lba, count, ns);
    }
};
}  // namespace

void io_log(const char* what, u64 lba, u32 count) {
    static std::FILE* log = [] { const char* f = std::getenv("MV_DEBUG_IO_LOG"); return f && *f ? std::fopen(f, "w") : nullptr; }();
    if (!log) return;
    static std::mutex m;
    std::lock_guard g(m);
    std::fprintf(log, "%llu %s %llu %u\n", static_cast<unsigned long long>(trace::now()), what, static_cast<unsigned long long>(lba), count);
    std::fflush(log);
}

void note_slow_io(const char* what, u64 lba, u32 count, u64 ns) {
    static const u64 limit = [] { const char* v = std::getenv("MV_DEBUG_IO_SLOW_MS"); return v && *v ? static_cast<u64>(std::atoll(v)) * 1000000u : 0u; }();
    if (limit && ns >= limit) std::fprintf(stderr, "io: %s lba %llu x %u took %.1f ms\n", what, static_cast<unsigned long long>(lba), count, static_cast<double>(ns) / 1e6);
}

DiskImage::~DiskImage() {
    {
        std::lock_guard g(pending_mutex_);
        write_stop_ = true;  // the writer empties the cache, then ends
    }
    write_wake_.notify_all();
    if (writer_.joinable()) writer_.join();
    {
        std::lock_guard g(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (reader_.joinable()) reader_.join();
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

bool DiskImage::read_at(u64 lba, u32 count, u8* out) const {
    if (!file_ || lba + count > sectors_) return false;
    std::lock_guard g(file_mutex_);
#if defined(_WIN32)
    if (_fseeki64(file_, static_cast<long long>(lba * 512), SEEK_SET) != 0) return false;
#else
    if (fseeko(file_, static_cast<off_t>(lba * 512), SEEK_SET) != 0) return false;
#endif
    if (std::fread(out, 512, count, file_) != count) return false;
    overlay_pending(lba, count, out);
    return true;
}

void DiskImage::overlay_pending(u64 lba, u32 count, u8* out) const {
    std::lock_guard g(pending_mutex_);
    for (auto it = pending_.lower_bound(lba); it != pending_.end() && it->first < lba + count; ++it)
        std::memcpy(out + (it->first - lba) * 512, it->second.data.data(), 512);
}

bool DiskImage::read(u64 lba, u32 count, u8* out) const {
    io_log("hdd-read", lba, count);
    Timed timed{host_read_ns_, "hdd read", lba, count};
    return read_at(lba, count, out);
}

void DiskImage::begin_read(u64 lba, u32 count) {
    io_log("hdd-read", lba, count);
    std::unique_lock lock(mutex_);
    if (!reader_.joinable()) reader_ = std::thread([this] { reader_main(); });
    done_.wait(lock, [this] { return finished_; });
    req_lba_ = lba;
    req_count_ = count;
    requested_ = true;
    finished_ = false;
    wake_.notify_one();
}

bool DiskImage::ready() const {
    std::lock_guard lock(mutex_);
    return finished_;
}

bool DiskImage::end_read(std::vector<u8>& out) {
    Timed timed{host_read_ns_, "hdd wait", req_lba_, req_count_};
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return finished_; });
    out = std::move(result_);
    result_.clear();
    return ok_;
}

void DiskImage::reader_main() {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "mv-hdd-reader");
#endif
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_.wait(lock, [this] { return stop_ || requested_; });
        if (stop_) return;
        requested_ = false;
        const u64 lba = req_lba_;
        const u32 count = req_count_;
        lock.unlock();
        std::vector<u8> out(static_cast<std::size_t>(count) * 512);
        const bool ok = read_at(lba, count, out.data());
        // Instrument: MV_DEBUG_HDD_DELAY_MS=N makes every read on this thread take N ms more (a cold, busy host disk).
        static const long delay_ms = [] { const char* d = std::getenv("MV_DEBUG_HDD_DELAY_MS"); return d && *d ? std::atol(d) : 0L; }();
        if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        lock.lock();
        result_ = std::move(out);
        ok_ = ok;
        finished_ = true;
        done_.notify_all();
    }
}

bool DiskImage::write(u64 lba, u32 count, const u8* in) {
    io_log("hdd-write", lba, count);
    Timed timed{host_read_ns_, "hdd write", lba, count};
    if (!file_ || read_only_ || lba + count > sectors_) return false;
    std::unique_lock lock(pending_mutex_);
    if (!writer_.joinable()) writer_ = std::thread([this] { writer_main(); });
    write_room_.wait(lock, [this] { return pending_.size() < kMaxPendingSectors; });
    const u64 gen = ++write_generation_;
    for (u32 i = 0; i < count; ++i) {
        Pending& p = pending_[lba + i];
        std::memcpy(p.data.data(), in + static_cast<std::size_t>(i) * 512, 512);
        p.generation = gen;
    }
    write_wake_.notify_one();
    return true;
}

// The guest's FLUSH CACHE: the writer is already at it; the host's own cache makes it durable in its time.
void DiskImage::flush() {
    write_wake_.notify_one();
}

// Takes runs of consecutive cached sectors (up to 1 MiB each), writes each to the file, and drops the sectors no newer
// write replaced meanwhile. Holding file_mutex_ across the write and the drop keeps every read seeing one or the other.
void DiskImage::writer_main() {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "mv-hdd-writer");
#endif
    constexpr u32 kRun = 2048;
    std::vector<u8> run;
    std::vector<u64> gens;
    std::unique_lock lock(pending_mutex_);
    for (;;) {
        write_wake_.wait(lock, [this] { return write_stop_ || !pending_.empty(); });
        if (pending_.empty()) return;  // stopping, all written
        const u64 first = pending_.begin()->first;
        run.clear();
        gens.clear();
        for (auto it = pending_.begin(); it != pending_.end() && it->first == first + gens.size() && gens.size() < kRun; ++it) {
            run.insert(run.end(), it->second.data.begin(), it->second.data.end());
            gens.push_back(it->second.generation);
        }
        lock.unlock();
        bool ok = false;
        {
            std::lock_guard f(file_mutex_);
#if defined(_WIN32)
            ok = _fseeki64(file_, static_cast<long long>(first * 512), SEEK_SET) == 0;
#else
            ok = fseeko(file_, static_cast<off_t>(first * 512), SEEK_SET) == 0;
#endif
            ok = ok && std::fwrite(run.data(), 512, gens.size(), file_) == gens.size() && std::fflush(file_) == 0;
            lock.lock();
            for (std::size_t i = 0; i < gens.size(); ++i) {
                const auto it = pending_.find(first + i);
                if (it != pending_.end() && it->second.generation == gens[i]) pending_.erase(it);
            }
        }
        if (!ok) {
            static bool told = false;
            if (!told) std::fprintf(stderr, "[midway-verdant] hard disk image: a write at sector %llu failed; the image may be incomplete\n", static_cast<unsigned long long>(first));
            told = true;
        }
        write_room_.notify_all();
    }
}

}  // namespace mv
