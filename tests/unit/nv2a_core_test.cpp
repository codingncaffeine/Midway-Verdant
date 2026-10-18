#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"
#include "nv2a/nv2a_stub.h"
#include "nv2a/scanout.h"

using namespace mv;

namespace {

// Channel 0 with a Kelvin object and one DMA object over the first 64 KiB of RAM: the pushbuffer at 0x1000, the
// semaphore at 0x8000, reports from 0x9000. The register block is unmapped, so offsets are its own.
struct Channel {
    static constexpr u32 kKelvin = 0x11, kDma = 0x22;           // handles
    static constexpr u32 kKelvinInstance = 0x100, kDmaInstance = 0x101;
    static constexpr u32 kPushbuffer = 0x1000, kSemaphore = 0x8000;
    Bus bus{1u << 20};
    Scheduler sched;
    Nv2aStub gpu{bus, sched, nullptr};
    std::vector<u32> words;

    static u32 hash(u32 handle) {  // a 9-bit table index (RAMHT size field 0), channel 0
        u32 h = 0;
        for (; handle; handle >>= 9) h ^= handle & 0x1FFu;
        return h;
    }
    void ramin(u32 offset, u32 v) { gpu.mmio_write(Nv2aStub::kPramin + offset, 4, v); }
    Channel() {
        ramin(hash(kKelvin) * 8, kKelvin);
        ramin(hash(kKelvin) * 8 + 4, kKelvinInstance);
        ramin(hash(kDma) * 8, kDma);
        ramin(hash(kDma) * 8 + 4, kDmaInstance);
        ramin(kKelvinInstance * 16, 0x97);
        ramin(kDmaInstance * 16, 0x3D);         // a DMA object at base 0
        ramin(kDmaInstance * 16 + 4, 0xFFFF);   // limit
        ramin(kDmaInstance * 16 + 8, 0x0);
        gpu.mmio_write(Nv2aStub::kCache1DmaInstance, 4, kDmaInstance);
        gpu.mmio_write(Nv2aStub::kCache1DmaGet, 4, kPushbuffer);
        gpu.mmio_write(Nv2aStub::kCache1DmaPut, 4, kPushbuffer);
        gpu.mmio_write(Nv2aStub::kCache1DmaPush, 4, 1);
        method(0, kKelvin);
        method(0x01A4, kDma);   // SET_CONTEXT_DMA_SEMAPHORE
        method(0x1D6C, kSemaphore);
        method(0x01A8, kDma);   // SET_CONTEXT_DMA_REPORT
        kick();
    }
    void method(u32 m, u32 v) {
        words.push_back((1u << 18) | m);
        words.push_back(v);
    }
    // Copies the queued words behind the previous ones and moves PUT past them.
    void kick() {
        const u32 put = gpu.reg(Nv2aStub::kCache1DmaPut);
        bus.write_ram(put, words.data(), words.size() * 4);
        gpu.mmio_write(Nv2aStub::kCache1DmaPut, 4, put + static_cast<u32>(words.size() * 4));
        words.clear();
    }
    u32 ram32(u32 at) {
        u32 v;
        std::memcpy(&v, bus.ram() + at, 4);
        return v;
    }
};

}  // namespace

// Double buffering as titles program it: the write index moves when a frame is finished, FLIP_STALL holds the
// puller while it would draw into the buffer still on the screen (read == write), and the vblank handler's write
// of INCREMENT bit 1 moves the read index and lets the puller go on.
TEST(Nv2aFlip, FlipStallHoldsThePullerUntilTheReadIndexMoves) {
    Channel c;
    c.method(0x0128, 2);  // SET_FLIP_MODULO
    c.method(0x0120, 0);  // SET_FLIP_READ
    c.method(0x0124, 0);  // SET_FLIP_WRITE
    c.method(0x012C, 0);  // FLIP_INCREMENT_WRITE: write 1, read 0 — the other buffer is free
    c.method(0x0130, 0);  // FLIP_STALL: passes
    c.method(0x1D70, 1);  // semaphore release
    c.method(0x012C, 0);  // write 0 again: that buffer is on the screen
    c.method(0x0130, 0);  // FLIP_STALL: holds
    c.method(0x1D70, 2);
    c.kick();
    EXPECT_EQ(c.ram32(Channel::kSemaphore), 1u) << "the second frame waits for the display";
    EXPECT_TRUE(c.gpu.flip_stalled());
    const u32 surface = c.gpu.mmio_read(Nv2aStub::kPgraphSurface, 4);
    EXPECT_EQ((surface >> 20) & 7u, 0u);  // write
    EXPECT_EQ((surface >> 24) & 7u, 0u);  // read
    EXPECT_EQ((surface >> 28) & 7u, 2u);  // modulo
    c.gpu.mmio_write(Nv2aStub::kPgraphIncrement, 4, 1);  // the 2D read index only: still held
    EXPECT_EQ(c.ram32(Channel::kSemaphore), 1u);
    c.gpu.mmio_write(Nv2aStub::kPgraphIncrement, 4, 2);
    EXPECT_FALSE(c.gpu.flip_stalled());
    EXPECT_EQ(c.ram32(Channel::kSemaphore), 2u);
    EXPECT_EQ((c.gpu.mmio_read(Nv2aStub::kPgraphSurface, 4) >> 24) & 7u, 1u);
}

// GET_REPORT writes 16 bytes at the report object + offset: the timer, a count, and a status word of zero (done),
// which is what a visibility test's poll waits for.
TEST(Nv2aReport, GetReportWritesTheReportWithItsStatusDone) {
    Channel c;
    const u32 at = 0x9000;
    const u32 sentinel[4] = {0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA};
    c.bus.write_ram(at, sentinel, 16);
    c.sched.advance_to(cycles_from_ns(1'000'000));
    c.method(0x17C8, 1);              // CLEAR_REPORT_VALUE
    c.method(0x17D0, (1u << 24) | at);  // GET_REPORT, z-pass pixel count
    c.kick();
    EXPECT_EQ(c.gpu.reports(), 1u);
    EXPECT_NE(c.ram32(at), 0xAAAAAAAAu) << "timestamp";
    EXPECT_GT(c.ram32(at + 8), 0u) << "the count";
    EXPECT_EQ(c.ram32(at + 12), 0u) << "status: done";
}

// PCRTC_RASTER counts the scanline: at the start of our frame (the vertical blank) it is the first line below the
// picture, it rises through the frame and stays below the frame's total.
TEST(Nv2aRaster, TheRasterRegisterCountsLinesThroughTheFrame) {
    Bus bus(4096);
    Scheduler sched;
    Nv2aStub gpu(bus, sched, nullptr);
    const Cycles frame = muldiv64(clk::kCpuHzNum * 1001, 1, clk::kCpuHzDen * 60000);
    sched.advance_to(frame * 10);
    const u32 at_blank = gpu.mmio_read(Nv2aStub::kPcrtcRaster, 4);
    EXPECT_EQ(at_blank, 480u);
    u32 previous = at_blank, wraps = 0;
    for (int i = 1; i < 100; ++i) {
        sched.advance_to(frame * 10 + frame * static_cast<u64>(i) / 100);
        const u32 line = gpu.mmio_read(Nv2aStub::kPcrtcRaster, 4);
        EXPECT_LT(line, 525u);
        if (line < previous) ++wraps;
        previous = line;
    }
    EXPECT_EQ(wraps, 1u) << "one wrap from the bottom of the blank to line 0";
}

TEST(Scanout, SixteenBitPixelsAreWidenedPerLayout) {
    const u8 fb[8] = {0x00, 0xF8, 0xE0, 0x07, 0x1F, 0x00, 0x00, 0x7C};  // R5G6B5 red, green, blue; X1R5G5B5 red
    u8 out[16]{};
    scanout_to_xrgb(fb, 8, 2, true, 3, 1, out);
    EXPECT_EQ(out[2], 255);  // red
    EXPECT_EQ(out[1], 0);
    EXPECT_EQ(out[5], 255);  // green
    EXPECT_EQ(out[8], 255);  // blue
    EXPECT_EQ(out[10], 0);
    scanout_to_xrgb(fb + 6, 2, 2, false, 1, 1, out);
    EXPECT_EQ(out[2], 255);
    EXPECT_EQ(out[1], 0);
}
