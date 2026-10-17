#include "nv2a/capture.h"

#include <cstring>
#include <format>

namespace mv::capture {

namespace {
constexpr char kMagic[8] = {'M', 'V', 'P', 'B', 'C', 'A', 'P', '1'};
}

Writer::~Writer() { close(); }

bool Writer::open(const std::filesystem::path& path, u32 ram_size, std::string& error) {
    close();
    file_ = std::fopen(path.string().c_str(), "wb");
    if (!file_) {
        error = std::format("cannot create the capture file {}", path.string());
        return false;
    }
    put(kMagic, sizeof kMagic);
    put32(kVersion);
    put32(ram_size);
    return true;
}

void Writer::close() {
    if (file_) std::fclose(file_);
    file_ = nullptr;
}

void Writer::put(const void* p, std::size_t n) {
    if (!file_) return;
    std::fwrite(p, 1, n, file_);
    bytes_ += n;
}

void Writer::pages(u32 first_page, u32 count, const u8* data) {
    if (count == 0) return;
    put8(static_cast<u8>(Rec::Pages));
    put32(first_page);
    put32(count);
    put(data, static_cast<std::size_t>(count) * kPageSize);
    pages_ += count;
}

void Writer::method(u32 subchannel, u32 cls, u32 method, u32 value) {
    put8(static_cast<u8>(Rec::Method));
    put32(pack_method(subchannel, cls, method));
    put32(value);
    ++methods_;
}

void Writer::object(u32 handle, const u32 dw[4]) {
    put8(static_cast<u8>(Rec::Object));
    put32(handle);
    for (int i = 0; i < 4; ++i) put32(dw[i]);
}

void Writer::run(u64 cycles, u32 get, u32 put_ptr) {
    put8(static_cast<u8>(Rec::Run));
    put64(cycles);
    put32(get);
    put32(put_ptr);
}

void Writer::reg(u32 offset, u32 value) {
    put8(static_cast<u8>(Rec::Register));
    put32(offset);
    put32(value);
}

Reader::~Reader() {
    if (file_) std::fclose(file_);
}

bool Reader::get(void* p, std::size_t n) {
    if (!file_) return false;
    if (std::fread(p, 1, n, file_) != n) {
        if (!std::feof(file_)) error_ = "read error";
        return false;
    }
    return true;
}

bool Reader::open(const std::filesystem::path& path, std::string& error) {
    file_ = std::fopen(path.string().c_str(), "rb");
    if (!file_) {
        error = std::format("cannot open the capture file {}", path.string());
        return false;
    }
    char magic[8];
    u32 version = 0;
    if (!get(magic, 8) || std::memcmp(magic, kMagic, 8) != 0) {
        error = "not a pushbuffer capture (bad magic)";
        return false;
    }
    if (!get(&version, 4) || version != kVersion) {
        error = std::format("capture version {} (this build reads {})", version, kVersion);
        return false;
    }
    if (!get(&ram_size_, 4)) {
        error = "truncated header";
        return false;
    }
    return true;
}

bool Reader::next(Record& out) {
    u8 type = 0;
    if (!get(&type, 1)) return false;
    out = Record{};
    switch (static_cast<Rec>(type)) {
        case Rec::Pages: {
            out.type = Rec::Pages;
            if (!get(&out.first_page, 4) || !get(&out.page_count, 4)) {
                error_ = "truncated pages record";
                return false;
            }
            const std::size_t n = static_cast<std::size_t>(out.page_count) * kPageSize;
            if (n > static_cast<std::size_t>(ram_size_)) {
                error_ = "pages record larger than RAM";
                return false;
            }
            page_buffer_.resize(n);
            if (!get(page_buffer_.data(), n)) {
                error_ = "truncated page data";
                return false;
            }
            out.page_data = page_buffer_.data();
            return true;
        }
        case Rec::Method: {
            out.type = Rec::Method;
            u32 packed = 0;
            if (!get(&packed, 4) || !get(&out.value, 4)) {
                error_ = "truncated method record";
                return false;
            }
            out.subchannel = packed >> 29;
            out.cls = (packed >> 13) & 0xFFFFu;
            out.method = packed & 0x1FFCu;
            return true;
        }
        case Rec::Object:
            out.type = Rec::Object;
            if (!get(&out.handle, 4) || !get(out.dw, 16)) {
                error_ = "truncated object record";
                return false;
            }
            return true;
        case Rec::Run:
            out.type = Rec::Run;
            if (!get(&out.cycles, 8) || !get(&out.get, 4) || !get(&out.put, 4)) {
                error_ = "truncated run record";
                return false;
            }
            return true;
        case Rec::Register:
            out.type = Rec::Register;
            if (!get(&out.offset, 4) || !get(&out.value, 4)) {
                error_ = "truncated register record";
                return false;
            }
            return true;
    }
    error_ = std::format("unknown record type {}", type);
    return false;
}

}  // namespace mv::capture
