#pragma once
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/types.h"

namespace mv {

// Instrument: MV_DEBUG_IO_SLOW_MS=N prints each disc or disk image call on the caller's thread that took N ms or more.
void note_slow_io(const char* what, u64 lba, u32 count, u64 ns);
// Instrument: MV_DEBUG_IO_LOG=FILE gets a line per disc and hard-disk request (what, first sector, sectors, guest cycle).
void io_log(const char* what, u64 lba, u32 count);

// A raw disk image on the host: 512-byte sectors, read and written in place. Sparse files stay sparse.
// DMA reads run on a thread of the image's own, as the disc's do: a cold read from a spinning host disk (the image's
// blocks after the host booted, or a fresh copy of it) held whole frames for 60 ms on the CPU's thread.
// Writes land in a write cache of the image's own (as the drive's) and a writer thread puts them in the file: a title
// rewriting one sector waited 40-70 ms a write for the host to finish writing the page out before changing it again
// (stable pages; PGR2 lost 13 s of frames to it). Reads see the cache first; the destructor writes it all out.
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
    // A read at once, on the caller's thread.
    bool read(u64 lba, u32 count, u8* out) const;
    // A read on the image's thread: begun now, its sectors collected by end_read (which waits for them if the host
    // has not finished; ready() says whether it has). One at a time: a second begin waits for the first to finish.
    void begin_read(u64 lba, u32 count);
    bool ready() const;
    bool end_read(std::vector<u8>& out);
    // Instrument: host time the caller's thread spent inside the reads, writes and flushes of every image of this kind,
    // or waiting for a read of the image's thread (the frame log's I/O column).
    static u64 host_read_ns() { return host_read_ns_.load(std::memory_order_relaxed); }
    bool write(u64 lba, u32 count, const u8* in);
    void flush();

private:
    bool read_at(u64 lba, u32 count, u8* out) const;  // the stream under file_mutex_, untimed
    void reader_main();

    inline static std::atomic<u64> host_read_ns_{0};
    std::FILE* file_{nullptr};
    u64 sectors_{0};
    bool read_only_{false};
    mutable std::mutex file_mutex_;  // one seek-and-transfer at a time on the stream

    void writer_main();
    void overlay_pending(u64 lba, u32 count, u8* out) const;  // cached writes over a read (file_mutex_ held)

    // The write cache: sectors written and not yet in the file, each with the generation of its last write (the writer
    // drops an entry only when no newer write came while it wrote it). Bounded: a write waits while it is full.
    struct Pending {
        std::array<u8, 512> data;
        u64 generation{0};
    };
    std::map<u64, Pending> pending_;
    mutable std::mutex pending_mutex_;
    std::condition_variable write_wake_, write_room_;
    u64 write_generation_{0};
    bool write_stop_{false};
    std::thread writer_;
    static constexpr std::size_t kMaxPendingSectors = 128 * 1024;  // 64 MiB

    // The reader thread: the request in hand and its result.
    std::thread reader_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, done_;
    bool stop_{false}, requested_{false}, finished_{true}, ok_{false};
    u64 req_lba_{0};
    u32 req_count_{0};
    std::vector<u8> result_;
};

}  // namespace mv
