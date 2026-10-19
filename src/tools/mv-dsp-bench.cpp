// mv-dsp-bench: runs a snapshot of the APU's global processor (MV_DEBUG_GP_SNAPSHOT=N in a game) for a number of
// frames, on the compiled path and on the reference path, and reports the host time each took; --check runs the two
// side by side and compares their whole state after every frame. The peripherals are a stand-in: the start-of-frame
// processor is the real one (its DMA engine, flags and halt register); the memory behind its DMA reads as zeros.
//   mv-dsp-bench SNAPSHOT FRAMES [--check]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "apu/dsp/core.h"
#include "apu/processor.h"

#if defined(__linux__)
#include <sys/prctl.h>
#endif

using namespace mv;

namespace {

constexpr dsp::Core::Config kGpConfig{0x1800, 0x800, 0x1000, 0xC00, 0x1400};  // as the APU builds its GP
constexpr u64 kCyclesPerFrame = 160'000'000 / 1500;

// The DMA's far side: scratch and FIFO reads give zeros, writes are dropped (the program's own work is what is timed).
struct NullMemory : dsp::DmaMemory {
    bool scratch_read(u32, u8* out, u32 len) override { std::memset(out, 0, len); return true; }
    bool scratch_write(u32, const u8*, u32) override { return true; }
    bool fifo_read(u32, u8* out, u32 len) override { std::memset(out, 0, len); return true; }
    bool fifo_write(u32, const u8*, u32) override { return true; }
};

// The real processor (its DMA engine, interrupt flags and halt register) around a core restored from the snapshot.
struct Rig {
    NullMemory memory;
    apu::Processor p{kGpConfig, memory};
    dsp::Core& core{p.core()};
    bool start(const std::vector<u32>& snap) {
        p.release_reset();  // running, with a program of zeros the snapshot replaces
        return core.restore(snap);
    }
    void frame() { p.run_frame(kCyclesPerFrame, true); }
};

}  // namespace

int main(int argc, char** argv) {
#if defined(__linux__)
    // MV_ALLOW_PROFILER=1: a sampling profiler that is not our parent (eu-stack) may attach.
    if (const char* p = std::getenv("MV_ALLOW_PROFILER"); p && *p == '1') prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
#endif
    if (argc < 3) {
        std::fprintf(stderr, "usage: mv-dsp-bench SNAPSHOT FRAMES [--check]\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) {
        std::fprintf(stderr, "mv-dsp-bench: cannot open %s\n", argv[1]);
        return 1;
    }
    std::vector<u32> snap;
    u32 w;
    while (std::fread(&w, 4, 1, f) == 1) snap.push_back(w);
    std::fclose(f);
    const long frames = std::atol(argv[2]);
    const bool check = argc > 3 && std::string(argv[3]) == "--check";

    if (check) {
        Rig fast, ref;
        ref.core.set_reference(true);
        if (!fast.start(snap) || !ref.start(snap)) {
            std::fprintf(stderr, "mv-dsp-bench: the snapshot does not fit the GP's configuration\n");
            return 1;
        }
        for (long i = 0; i < frames; ++i) {
            fast.frame();
            ref.frame();
            const auto a = fast.core.snapshot(), b = ref.core.snapshot();
            if (a != b || fast.core.error() != ref.core.error()) {
                std::size_t first = 0;
                while (first < a.size() && first < b.size() && a[first] == b[first]) ++first;
                std::printf("DIFFER after frame %ld: first differing word %zu (compiled %08x, reference %08x); errors '%s' / '%s'\n", i, first,
                            first < a.size() ? a[first] : 0u, first < b.size() ? b[first] : 0u, fast.core.error_text().c_str(), ref.core.error_text().c_str());
                return 1;
            }
        }
        std::printf("identical: %ld frames, %llu instructions, the whole state compared after each frame\n", frames,
                    static_cast<unsigned long long>(fast.core.instructions()));
        return 0;
    }
    if (argc > 3 && std::string(argv[3]) == "--hist") {
        Rig r;
        std::vector<u64> hist(0x1000, 0);
        r.core.set_pc_histogram(&hist);
        if (!r.start(snap)) return 1;
        for (long i = 0; i < frames; ++i) r.frame();
        u64 total = 0;
        for (u64 v : hist) total += v;
        for (int k = 0; k < 16; ++k) {
            std::size_t best = 0;
            for (std::size_t a = 0; a < hist.size(); ++a)
                if (hist[a] > hist[best]) best = a;
            if (!hist[best]) break;
            std::printf("P:%04zx %5.2f%%  %s\n", best, 100.0 * static_cast<double>(hist[best]) / static_cast<double>(total), r.core.disassemble_at(static_cast<u32>(best)).c_str());
            hist[best] = 0;
        }
        return 0;
    }
    for (const bool reference : {false, true}) {
        Rig r;
        r.core.set_reference(reference);
        if (!r.start(snap)) {
            std::fprintf(stderr, "mv-dsp-bench: the snapshot does not fit the GP's configuration\n");
            return 1;
        }
        const u64 i0 = r.core.instructions();
        const auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < frames; ++i) r.frame();
        const double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        const u64 n = r.core.instructions() - i0;
        std::printf("%-9s %ld frames: %.1f us a frame, %llu instructions (%.2f ns each)%s%s\n", reference ? "reference" : "compiled", frames, ns / 1e3 / static_cast<double>(frames),
                    static_cast<unsigned long long>(n), n ? ns / static_cast<double>(n) : 0.0, r.core.error() ? "; error: " : "", r.core.error_text().c_str());
    }
    return 0;
}
