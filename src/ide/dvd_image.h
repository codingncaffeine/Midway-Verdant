#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/types.h"

namespace mv {

// A game disc image on the host, read in 2048-byte sectors of the disc's Xbox (game) partition: an XISO holds that
// partition alone; a full dump (the Redump layout) holds the video partition first and the game partition at byte
// 0x18300000. Either is recognised by the XDVDFS volume descriptor ("MICROSOFT*XBOX*MEDIA") in sector 32 of the
// partition. Read-only.
//
// A split image (cut under 4 GiB for FAT32 drives: NAME.1.iso, NAME.2.iso, ...) is read as one: opening NAME.1.iso takes
// the parts that follow it in order. A first part with no second is still opened, with a note() saying the rest is
// missing (one title's quick race then waited for ever on cars whose data lay past the end).
//
// The drive's reads run on a thread of the image's own (as the reference emulators of other consoles read their
// discs): a read starts when the drive takes the command and is collected when its data phase begins, so the host's
// seek overlaps the drive's modeled time instead of stalling the CPU's thread (a cold read from a spinning disk held
// single frames for 70-270 ms). After each read the thread reads on ahead, since titles stream their files in order.
class DvdImage {
public:
    enum class Kind { Xiso, Redump };
    static constexpr u32 kSectorSize = 2048;
    static constexpr u64 kRedumpGamePartition = 0x18300000;
    static constexpr u32 kReadAheadSectors = 256;  // 512 KiB past the end of each read

    DvdImage() = default;
    ~DvdImage();
    DvdImage(const DvdImage&) = delete;
    DvdImage& operator=(const DvdImage&) = delete;

    bool open(const std::filesystem::path& path, std::string& error);
    bool is_open() const { return !parts_.empty(); }
    // The files the image was read from, and a warning about it (empty when there is none).
    std::size_t parts() const { return parts_.size(); }
    const std::string& note() const { return note_; }
    // Instrument: reads that asked for sectors past the image's end (an incomplete image), and the first such sector.
    u64 past_end_reads() const { return past_end_reads_.load(std::memory_order_relaxed); }
    u64 first_past_end_lba() const { return first_past_end_lba_.load(std::memory_order_relaxed); }
    void note_past_end(u64 lba, u32 count) const;
    Kind kind() const { return kind_; }
    std::string_view kind_name() const { return kind_ == Kind::Xiso ? "XISO" : "full dump"; }
    // Sectors of the game partition.
    u64 sectors() const { return sectors_; }
    // A read at once, on the caller's thread.
    bool read(u64 lba, u32 count, u8* out) const;
    // A read on the image's thread: begun now, its sectors collected by end_read (which waits for them if the host
    // has not delivered them yet). One read is in flight at a time; a new begin_read waits out the one before.
    void begin_read(u64 lba, u32 count);
    bool end_read(std::vector<u8>& out);
    // Whether the read in flight has its sectors (end_read would not wait).
    bool ready() const;
    // Instruments: host time the caller's thread spent reading or waiting for a read; reads the read-ahead answered.
    static u64 host_read_ns() { return host_read_ns_.load(std::memory_order_relaxed); }
    u64 read_ahead_hits() const { return read_ahead_hits_.load(std::memory_order_relaxed); }
    u64 warmed_bytes() const { return warmed_bytes_.load(std::memory_order_relaxed); }

private:
    bool read_at(u64 offset, u32 bytes, u8* out) const;
    bool has_volume_descriptor(u64 partition) const;
    void reader_main();
    void warm_main();

    inline static std::atomic<u64> host_read_ns_{0};
    // The image's files in order, each with its byte offset in the whole image.
    struct Part {
        std::filesystem::path path;
        std::FILE* file{nullptr};
        u64 start{0}, size{0};
    };
    std::vector<Part> parts_;
    u64 size_{0};  // all parts together
    std::string note_;
    mutable std::atomic<u64> past_end_reads_{0}, first_past_end_lba_{0};
    mutable std::mutex file_mutex_;  // one seek-and-read at a time on the streams
    Kind kind_{Kind::Xiso};
    u64 base_{0};
    u64 sectors_{0};

    // The reader thread: the request in hand, its result, and the read-ahead block.
    std::thread reader_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, done_;
    bool stop_{false}, requested_{false}, finished_{true}, ok_{false};
    u64 req_lba_{0};
    u32 req_count_{0};
    std::vector<u8> result_;
    std::vector<u8> ahead_;
    u64 ahead_lba_{0};
    u32 ahead_count_{0};
    std::atomic<u64> read_ahead_hits_{0};
    // The page cache warmed with the whole image at idle I/O priority after it opens (Linux), so a title's reads find
    // it in memory instead of seeking a spinning disk; stopped when the image closes or memory is short.
    std::thread warmer_;
    std::atomic<bool> warm_stop_{false};
    std::atomic<u64> warmed_bytes_{0};
};

}  // namespace mv
