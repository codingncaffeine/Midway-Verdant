#include "nv2a/pgraph/census.h"

#ifdef MV_STATE_CENSUS
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace mv::pgraph::census {
namespace {

constexpr u32 kValues = 24;  // distinct values kept per method; more than that and the method counts as free-form

struct Entry {
    u64 writes{0};
    u32 count{0};
    bool overflow{false};
    bool consumed{false};
    std::array<u32, kValues> value{};
    void note(u32 v) {
        ++writes;
        for (u32 i = 0; i < count; ++i)
            if (value[i] == v) return;
        if (count < kValues) value[count++] = v;
        else overflow = true;
    }
};

struct Register {
    u64 reads{0}, writes{0};
    Entry values;
};

struct Tables {
    std::mutex lock;  // the classes and registers are touched from the CPU's thread and the FIFO thread
    std::array<Entry, 0x800> kelvin{};
    std::array<u8, 0x800> kelvin_read{};
    std::map<u32, Entry> classes;     // class << 16 | method
    std::map<u32, Register> registers;
    std::map<std::pair<std::string, u32>, u64> facts;
    ~Tables() { dump(); }
    void dump();
};

Tables& tables() {
    static Tables t;
    return t;
}

// The packed Kelvin methods whose free-form fields (sizes, offsets, levels of detail) would hide the modes beside
// them: the census keeps only the mode bits of these.
u32 mode_bits(u32 method, u32 value) {
    if (method >= 0x1B00 && method < 0x1C00) {
        switch (method & 0x3Fu) {
            case 0x00: return 0;                    // offset
            case 0x04: return value & 0x0000FFFFu;  // format: DMA, cube, border source, dimensions, color format
            case 0x0C: return value & 0x4000003Fu;  // control 0: enable, anisotropy, image field, alpha kill, color key
            case 0x10: return 0;                    // control 1: pitch
            case 0x14: return value & 0xFFFFE000u;  // filter: kernel, min, mag, signed channels
            case 0x1C: return 0;                    // image rectangle
            case 0x20: return value & 0x0000003Fu;  // palette: DMA and length
            default: return value;
        }
    }
    if (method == 0x0208) return value & 0x0000FFFFu;                  // surface format without the sizes
    if (method >= 0x1760 && method < 0x17A0) return value & 0xFFu;      // vertex array format: type and size
    if (method >= 0x1720 && method < 0x1760) return value & 0x80000000u;  // vertex array offset: which DMA object
    return value;
}

void print_entry(std::FILE* f, const Entry& e) {
    std::fprintf(f, " w=%llu v=", static_cast<unsigned long long>(e.writes));
    for (u32 i = 0; i < e.count; ++i) std::fprintf(f, "%s%x", i ? "," : "", e.value[i]);
    if (e.overflow) std::fprintf(f, ",*");
}

void Tables::dump() {
    const char* path = std::getenv("MV_CENSUS_FILE");
    std::FILE* f = path && *path ? std::fopen(path, "w") : stderr;
    if (!f) return;
    for (u32 i = 0; i < kelvin.size(); ++i) {
        const Entry& e = kelvin[i];
        if (!e.writes) continue;
        std::fprintf(f, "kelvin %04x %s%s", i << 2, e.consumed ? "s" : "-", kelvin_read[i] ? "r" : "-");
        print_entry(f, e);
        std::fputc('\n', f);
    }
    for (const auto& [key, e] : classes) {
        std::fprintf(f, "class %02x %04x --", key >> 16, key & 0xFFFFu);
        print_entry(f, e);
        std::fputc('\n', f);
    }
    for (const auto& [off, r] : registers) {
        std::fprintf(f, "mmio %06x r=%llu", off, static_cast<unsigned long long>(r.reads));
        print_entry(f, r.values);
        std::fputc('\n', f);
    }
    for (const auto& [key, count] : facts) std::fprintf(f, "fact %llu x [%x] %s\n", static_cast<unsigned long long>(count), key.second, key.first.c_str());
    if (f != stderr) std::fclose(f);
}

}  // namespace

void kelvin_write(u32 method, u32 value, bool consumed) {
    Entry& e = tables().kelvin[(method >> 2) & 0x7FFu];
    e.consumed = consumed;
    e.note(mode_bits(method, value));
}

void kelvin_read(u32 index) { tables().kelvin_read[index & 0x7FFu] = 1; }

void class_write(u32 cls, u32 method, u32 value) {
    Tables& t = tables();
    const std::lock_guard<std::mutex> hold(t.lock);
    t.classes[(cls << 16) | (method & 0xFFFFu)].note(value);
}

void mmio(u32 offset, bool write, u32 value) {
    // The instance memory (objects, the hash table) and the vertex-program image are data, not registers.
    if (offset >= 0x700000 && offset < 0x800000) offset = 0x700000;
    Tables& t = tables();
    const std::lock_guard<std::mutex> hold(t.lock);
    Register& r = t.registers[offset & ~3u];
    if (write) r.values.note(value);
    else ++r.reads;
}

void draw_fact(const char* what, u32 detail) {
    Tables& t = tables();
    const std::lock_guard<std::mutex> hold(t.lock);
    ++t.facts[{what, detail}];
}

}  // namespace mv::pgraph::census
#endif
