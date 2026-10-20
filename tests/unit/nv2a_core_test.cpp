#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"
#include "nv2a/gpu_timing.h"
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

// SET_REFERENCE writes the puller's reference counter, which software reads back through the USER window; the
// window's DMA_GET writes reach the pusher's GET.
TEST(Nv2aFifo, TheUserWindowReadsTheReferenceAndWritesGet) {
    Channel c;
    c.method(0x0050, 0x1234);  // SET_REFERENCE
    c.kick();
    EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kUser + 0x48, 4), 0x1234u);
    c.gpu.mmio_write(Nv2aStub::kUser + 0x44, 4, Channel::kPushbuffer);
    EXPECT_EQ(c.gpu.reg(Nv2aStub::kCache1DmaGet), Channel::kPushbuffer);
    EXPECT_NE(c.gpu.mmio_read(Nv2aStub::kCache1DmaCtl, 4) & (1u << 31), 0u);
    EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kPramdacPllTestCounter, 4) & Nv2aStub::kPllLocked, Nv2aStub::kPllLocked);
    // The bound object's class is mirrored where software-method handlers read it.
    EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kPgraphCtxSwitch1, 4) & 0xFFu, 0x97u);
}

// PTIMER ALARM_0 raises PTIMER INTR bit 0 when TIME_0 reaches it, with the interrupt enabled.
TEST(Nv2aTimer, TheAlarmRaisesItsInterruptWhenTheTimeReachesIt) {
    Bus bus(4096);
    Scheduler sched;
    bool irq = false;
    Nv2aStub gpu(bus, sched, [&](bool level) { irq = level; });
    gpu.mmio_write(Nv2aStub::kPmcIntrEn0, 4, 1);
    sched.advance_to(cycles_from_ns(1'000'000));
    const u32 now = gpu.mmio_read(Nv2aStub::kPtimerTime0, 4);
    const u32 ticks_per_ms = now;  // the counter has run for 1 ms
    gpu.mmio_write(Nv2aStub::kPtimerAlarm0, 4, now + ticks_per_ms);  // one more millisecond
    gpu.mmio_write(Nv2aStub::kPtimerIntrEn0, 4, 1);
    while (sched.has_pending() && sched.next_deadline() < cycles_from_ns(1'900'000)) sched.advance_to(sched.next_deadline());
    sched.advance_to(cycles_from_ns(1'900'000));
    EXPECT_EQ(gpu.mmio_read(Nv2aStub::kPtimerIntr0, 4) & 1u, 0u) << "not yet";
    while (sched.has_pending() && sched.next_deadline() <= cycles_from_ns(2'100'000)) sched.advance_to(sched.next_deadline());
    sched.advance_to(cycles_from_ns(2'100'000));
    EXPECT_EQ(gpu.mmio_read(Nv2aStub::kPtimerIntr0, 4) & 1u, 1u) << "reached";
    EXPECT_TRUE(irq);
    gpu.mmio_write(Nv2aStub::kPtimerIntr0, 4, 1);
    EXPECT_FALSE(irq);
}

// A write to RAM between two pusher runs makes the pages it touched newer than any cache built before it; pages it
// did not touch keep their generation, and reading changes nothing.
TEST(Nv2aWrites, APusherRunStampsThePagesWrittenSinceTheLastOne) {
    Channel c;
    c.bus.set_dirty_tracking(true);
    c.gpu.set_write_tracking(nullptr);
    c.method(0x0050, 1);
    c.kick();
    const u32 built = c.gpu.write_generation();
    EXPECT_LE(c.gpu.newest_write(0x20000, 0x2000), built);
    const u32 texel = 0xFFFFFFFF;
    c.bus.write_ram(0x21000, &texel, 4);
    c.method(0x0050, 2);
    c.kick();
    EXPECT_GT(c.gpu.newest_write(0x20000, 0x2000), built) << "the texture's second page was written";
    EXPECT_LE(c.gpu.newest_write(0x20000, 0x1000), built) << "its first page was not";
    EXPECT_GT(c.gpu.newest_write(0x21000, 4), built) << "a second reader still sees it";
}

// The pattern class's SET_MONOCHROME_COLOR0 lands in PGRAPH PATT_COLOR0, where software reads it back: Direct3D pushes
// a value through it and polls the register to know the engine has got that far (a title waited on it for ever).
TEST(Nv2aFence, ThePatternColourMethodIsReadBackFromPgraph) {
    Channel c;
    constexpr u32 kPattern = 0x33, kPatternInstance = 0x102;
    c.ramin(Channel::hash(kPattern) * 8, kPattern);
    c.ramin(Channel::hash(kPattern) * 8 + 4, kPatternInstance);
    c.ramin(kPatternInstance * 16, Nv2aStub::kClassPattern);
    auto on_subchannel_1 = [&](u32 m, u32 v) {
        c.words.push_back((1u << 18) | (1u << 13) | m);
        c.words.push_back(v);
    };
    on_subchannel_1(0, kPattern);
    on_subchannel_1(Nv2aStub::kMethodPatternColor0, 0x5EED0001);
    c.kick();
    EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kPgraphPattColor0, 4), 0x5EED0001u);
    // The same method on the 3D class is not the pattern colour.
    c.method(Nv2aStub::kMethodPatternColor0, 0x12345678);
    c.kick();
    EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kPgraphPattColor0, 4), 0x5EED0001u);
}

// The timed pusher (PLAN D28): 1000 methods nobody can observe, then SET_REFERENCE and a semaphore release. Timed, the
// kick changes nothing the CPU can see; the engine reads busy and GET trails; the reference lands exactly when the GPU
// clock reaches it (not a cycle before), the semaphore one method later, then the engine is idle with GET at PUT.
// Untimed (the fast mode), both land at the kick.
TEST(Nv2aTiming, ObservableMethodsLandWhenTheGpuClockReachesThem) {
    for (const bool timed : {false, true}) {
        SCOPED_TRACE(timed ? "timed" : "untimed");
        Channel c;
        c.gpu.set_timed(timed);
        const u64 clocks_before = c.gpu.gpu_clocks();
        for (int i = 0; i < 1000; ++i) c.method(Nv2aStub::kMethodNoOperation, 0);
        c.method(0x0050, 0x77);              // SET_REFERENCE
        c.method(0x1D70, 0xCAFE0001);        // BACK_END_WRITE_SEMAPHORE_RELEASE
        c.kick();
        const u32 put = c.gpu.reg(Nv2aStub::kCache1DmaPut);
        auto semaphore = [&] {
            u32 v = 0;
            c.bus.read_ram(Channel::kSemaphore, &v, 4);
            return v;
        };
        if (!timed) {
            EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kUser + 0x48, 4), 0x77u);
            EXPECT_EQ(semaphore(), 0xCAFE0001u);
            EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kPgraphStatus, 4) & 1u, 0u);
            continue;
        }
        EXPECT_NE(c.gpu.mmio_read(Nv2aStub::kUser + 0x48, 4), 0x77u) << "the reference waits for the GPU";
        EXPECT_NE(semaphore(), 0xCAFE0001u);
        EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kPgraphStatus, 4) & 1u, 1u) << "busy";
        EXPECT_LT(c.gpu.mmio_read(Nv2aStub::kUser + 0x44, 4), put) << "GET trails";
        // The reference's time: the GPU clock after the 1000 methods (each one clock), from the clock at the kick.
        const Cycles at = gpu_timing::cycles_from_gpu(clocks_before + 1000);
        c.sched.advance_to(at - 1);
        EXPECT_NE(c.gpu.mmio_read(Nv2aStub::kUser + 0x48, 4), 0x77u) << "not a cycle early";
        const u32 get_mid = c.gpu.mmio_read(Nv2aStub::kUser + 0x44, 4);
        EXPECT_GT(get_mid, Channel::kPushbuffer) << "GET has moved with the GPU clock";
        EXPECT_LT(get_mid, put);
        c.sched.advance_to(at);
        EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kUser + 0x48, 4), 0x77u) << "on time";
        EXPECT_NE(semaphore(), 0xCAFE0001u) << "the semaphore is one method later";
        const Cycles sem_at = gpu_timing::cycles_from_gpu(clocks_before + 1001);
        c.sched.advance_to(sem_at);
        EXPECT_EQ(semaphore(), 0xCAFE0001u);
        c.sched.advance_to(gpu_timing::cycles_from_gpu(clocks_before + 1002));
        EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kPgraphStatus, 4) & 1u, 0u) << "idle";
        EXPECT_EQ(c.gpu.mmio_read(Nv2aStub::kUser + 0x44, 4), put);
    }
}
