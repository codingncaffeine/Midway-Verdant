#pragma once
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/types.h"

// The pushbuffer capture: the GPU's command stream as the puller saw it, with the RAM pages the CPU (or a
// device) wrote before each pusher run, the RAMIN objects behind every handle the stream names, and the
// scanout register writes. A capture replays through the renderer with no console attached, so the renderer
// is developed and regression-tested offline against streams from real boots and games.
//
// File layout (little-endian): the 8-byte magic, u32 version, u32 RAM size, then records, each a one-byte
// type followed by its fields:
//   Pages    u32 first page, u32 count, count * 4096 bytes       RAM contents as of the next run
//   Method   u32 packed (subchannel:3 | class:16 | method:13), u32 value
//   Object   u32 handle, u32 dw[4]                               the RAMIN object the handle resolved to
//   Run      u64 cycles, u32 get, u32 put                        a pusher run begins
//   Register u32 offset, u32 value                               a scanout-related GPU register write
namespace mv::capture {

constexpr u32 kVersion = 1;
constexpr std::size_t kPageSize = 4096;

enum class Rec : u8 { Pages = 1, Method = 2, Object = 3, Run = 4, Register = 5 };

struct Record {
    Rec type{Rec::Run};
    // Pages
    u32 first_page{0};
    u32 page_count{0};
    const u8* page_data{nullptr};  // valid until the next read
    // Method
    u32 subchannel{0};
    u32 cls{0};
    u32 method{0};
    u32 value{0};
    // Object
    u32 handle{0};
    u32 dw[4]{};
    // Run
    u64 cycles{0};
    u32 get{0}, put{0};
    // Register
    u32 offset{0};
};

inline u32 pack_method(u32 subchannel, u32 cls, u32 method) {
    return (subchannel << 29) | ((cls & 0xFFFFu) << 13) | (method & 0x1FFCu);
}

class Writer {
public:
    Writer() = default;
    ~Writer();
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    bool open(const std::filesystem::path& path, u32 ram_size, std::string& error);
    void close();
    bool is_open() const { return file_ != nullptr; }

    void pages(u32 first_page, u32 count, const u8* data);
    void method(u32 subchannel, u32 cls, u32 method, u32 value);
    void object(u32 handle, const u32 dw[4]);
    void run(u64 cycles, u32 get, u32 put);
    void reg(u32 offset, u32 value);

    u64 bytes_written() const { return bytes_; }
    u64 pages_written() const { return pages_; }
    u64 methods_written() const { return methods_; }

private:
    void put(const void* p, std::size_t n);
    void put8(u8 v) { put(&v, 1); }
    void put32(u32 v) { put(&v, 4); }
    void put64(u64 v) { put(&v, 8); }

    std::FILE* file_{nullptr};
    u64 bytes_{0}, pages_{0}, methods_{0};
};

class Reader {
public:
    Reader() = default;
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    bool open(const std::filesystem::path& path, std::string& error);
    void close();
    u32 ram_size() const { return ram_size_; }
    // The next record, or false at the end (or on a truncated file: `error()` then says so).
    bool next(Record& out);
    std::string error() const { return error_; }

private:
    bool get(void* p, std::size_t n);
    std::FILE* file_{nullptr};
    u32 ram_size_{0};
    std::vector<u8> page_buffer_;
    std::string error_;
};

}  // namespace mv::capture
