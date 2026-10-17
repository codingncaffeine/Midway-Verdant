// mv-pbcap: reads a pushbuffer capture written by the emulator (--capture) and reports what is in it: the
// pusher runs, the methods by class, the frames (one per semaphore release of the 3D class), the draws and
// clears per frame, the RAM pages carried. The renderer's offline instrument.
//   mv-pbcap summary FILE        the report
//   mv-pbcap methods FILE        every method as "class method value", one per line (diffs against traces)
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "nv2a/capture.h"

namespace {

struct ClassStats {
    std::map<mv::u32, mv::u64> by_method;
    mv::u64 total{0};
};

struct Frame {
    mv::u64 methods{0}, draws{0}, clears{0}, pages{0};
    mv::u64 first_cycles{0};
    mv::u32 semaphore_value{0};
};

int summary(const char* path) {
    mv::capture::Reader r;
    std::string error;
    if (!r.open(path, error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    std::map<mv::u32, ClassStats> classes;
    std::vector<Frame> frames(1);
    mv::u64 runs = 0, pages = 0, page_bytes = 0, objects = 0, registers = 0, methods = 0;
    mv::u64 cycles = 0;
    mv::u32 last_scanout = 0;
    mv::capture::Record rec;
    while (r.next(rec)) {
        switch (rec.type) {
            case mv::capture::Rec::Pages:
                ++pages;
                page_bytes += static_cast<mv::u64>(rec.page_count) * mv::capture::kPageSize;
                frames.back().pages += rec.page_count;
                break;
            case mv::capture::Rec::Run:
                ++runs;
                cycles = rec.cycles;
                break;
            case mv::capture::Rec::Object:
                ++objects;
                break;
            case mv::capture::Rec::Register:
                ++registers;
                if (rec.offset == 0x600800) last_scanout = rec.value;
                break;
            case mv::capture::Rec::Method: {
                ++methods;
                auto& c = classes[rec.cls];
                ++c.total;
                ++c.by_method[rec.method];
                Frame& f = frames.back();
                if (f.methods == 0) f.first_cycles = cycles;
                ++f.methods;
                if (rec.cls == 0x97) {
                    if (rec.method == 0x17FC && rec.value != 0) ++f.draws;
                    if (rec.method == 0x1D94) ++f.clears;
                    if (rec.method == 0x1D70) {
                        f.semaphore_value = rec.value;
                        frames.emplace_back();
                    }
                }
                break;
            }
        }
    }
    if (!r.error().empty()) std::fprintf(stderr, "mv-pbcap: %s (reporting what was read)\n", r.error().c_str());
    std::printf("ram %u MiB; %llu pusher runs, %llu methods, %llu objects, %llu register writes, %llu page records (%llu KiB)\n",
                r.ram_size() >> 20, static_cast<unsigned long long>(runs), static_cast<unsigned long long>(methods),
                static_cast<unsigned long long>(objects), static_cast<unsigned long long>(registers), static_cast<unsigned long long>(pages),
                static_cast<unsigned long long>(page_bytes >> 10));
    std::printf("scanout start (last PCRTC START write): %08x\n", last_scanout);
    for (const auto& [cls, stats] : classes) {
        std::printf("class %04x: %llu methods, %zu distinct\n", cls, static_cast<unsigned long long>(stats.total), stats.by_method.size());
        std::vector<std::pair<mv::u64, mv::u32>> top;
        for (const auto& [m, n] : stats.by_method) top.emplace_back(n, m);
        std::sort(top.rbegin(), top.rend());
        std::size_t shown = 0;
        for (const auto& [n, m] : top) {
            if (shown++ >= 24) break;
            std::printf("  %04x x%llu", m, static_cast<unsigned long long>(n));
        }
        std::printf("\n");
    }
    std::printf("frames (ended by a 3D semaphore release): %zu\n", frames.size() - (frames.back().methods == 0 ? 1 : 0));
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const Frame& f = frames[i];
        if (f.methods == 0) continue;
        std::printf("  frame %zu: %llu methods, %llu draws, %llu clears, %llu pages, at %.3f ms, semaphore %08x\n", i,
                    static_cast<unsigned long long>(f.methods), static_cast<unsigned long long>(f.draws),
                    static_cast<unsigned long long>(f.clears), static_cast<unsigned long long>(f.pages),
                    static_cast<double>(f.first_cycles) / 733333.333, f.semaphore_value);
    }
    return 0;
}

int methods(const char* path) {
    mv::capture::Reader r;
    std::string error;
    if (!r.open(path, error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    mv::capture::Record rec;
    while (r.next(rec)) {
        switch (rec.type) {
            case mv::capture::Rec::Method: std::printf("[%04x] %04x <- %08x\n", rec.cls, rec.method, rec.value); break;
            case mv::capture::Rec::Run: std::printf("run at %llu cycles get %08x put %08x\n", static_cast<unsigned long long>(rec.cycles), rec.get, rec.put); break;
            case mv::capture::Rec::Pages: std::printf("pages %u + %u\n", rec.first_page, rec.page_count); break;
            case mv::capture::Rec::Object:
                std::printf("object %08x: %08x %08x %08x %08x\n", rec.handle, rec.dw[0], rec.dw[1], rec.dw[2], rec.dw[3]);
                break;
            case mv::capture::Rec::Register: std::printf("register %06x <- %08x\n", rec.offset, rec.value); break;
        }
    }
    if (!r.error().empty()) {
        std::fprintf(stderr, "mv-pbcap: %s\n", r.error().c_str());
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: mv-pbcap summary|methods FILE\n");
        return 2;
    }
    if (std::strcmp(argv[1], "summary") == 0) return summary(argv[2]);
    if (std::strcmp(argv[1], "methods") == 0) return methods(argv[2]);
    std::fprintf(stderr, "mv-pbcap: unknown command %s\n", argv[1]);
    return 2;
}
