#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "apu/dsp/dma.h"
#include "apu/processor.h"
#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pgraph/replay.h"

using namespace mv;
using namespace mv::dsp;

namespace {

// A flat scratch space and four FIFO logs.
struct FlatMemory : DmaMemory {
    std::vector<u8> scratch;
    std::vector<std::vector<u8>> fifo{4};
    std::vector<u8> fifo_in;
    bool scratch_read(u32 offset, u8* out, u32 len) override {
        if (static_cast<std::size_t>(offset) + len > scratch.size()) return false;
        std::memcpy(out, &scratch[offset], len);
        return true;
    }
    bool scratch_write(u32 offset, const u8* in, u32 len) override {
        if (static_cast<std::size_t>(offset) + len > scratch.size()) return false;
        std::memcpy(&scratch[offset], in, len);
        return true;
    }
    bool fifo_read(u32, u8* out, u32 len) override {
        if (fifo_in.size() < len) return false;
        std::memcpy(out, fifo_in.data(), len);
        return true;
    }
    bool fifo_write(u32 f, const u8* in, u32 len) override {
        fifo[f].insert(fifo[f].end(), in, in + len);
        return true;
    }
};

struct NullPeripherals : Peripherals {
    u32 peripheral_read(u32) override { return 0; }
    void peripheral_write(u32, u32) override {}
};

void put_block(Core& core, u32 at, u32 next, u32 control, u32 count, u32 dsp, u32 offset, u32 base, u32 size_minus_1) {
    const u32 words[7] = {next, control, count, dsp, offset, base, size_minus_1};
    for (u32 i = 0; i < 7; ++i) core.write_x(at + i, words[i]);
}

}  // namespace

TEST(DspDma, ScratchToXMemoryIn24BitLsbFormatAndBack) {
    NullPeripherals np;
    Core core(Core::Config{}, &np);
    FlatMemory mem;
    mem.scratch.assign(4096, 0);
    const u32 samples[3] = {0x123456, 0x800000, 0x7FFFFF};
    std::memcpy(&mem.scratch[0x100], samples, 12);
    Dma dma(core, mem);
    // Block at X:$40: scratch (0xF) -> DSP X:$200, 3 samples, format 24-bit lsb (6), end of list.
    put_block(core, 0x40, Dma::kEol, (0xFu << 5) | (6u << 10), 3, 0x200, 0x100, 0, 0);
    dma.write(Dma::StartBlock, 0x40);
    dma.write(Dma::NextBlock, 0x40);
    dma.write(Dma::Control, 1);  // start
    EXPECT_TRUE(dma.read(Dma::Control) & Dma::kControlRunning) << "running for the transfer time";
    EXPECT_FALSE(dma.eol());
    for (int i = 0; i < 64; ++i) core.step();  // NOPs: 64 cycles pass, more than the 12 bytes take
    EXPECT_TRUE(dma.eol());
    EXPECT_FALSE(dma.read(Dma::Control) & Dma::kControlRunning);
    EXPECT_EQ(core.read_x(0x200), 0x123456u);
    EXPECT_EQ(core.read_x(0x201), 0x800000u);
    EXPECT_EQ(core.read_x(0x202), 0x7FFFFFu);
    EXPECT_TRUE(dma.last_error().empty()) << dma.last_error();
    // And back out, 16-bit, to FIFO 0: the upper 16 bits of each word.
    put_block(core, 0x48, Dma::kEol, 2u | (0u << 5) | (1u << 10), 3, 0x200, 0, 0, 0);
    dma.clear_eol();
    dma.write(Dma::NextBlock, 0x48);
    dma.write(Dma::Control, 1);
    for (int i = 0; i < 64; ++i) core.step();
    ASSERT_EQ(mem.fifo[0].size(), 6u);
    EXPECT_EQ(mem.fifo[0][0], 0x34);
    EXPECT_EQ(mem.fifo[0][1], 0x12);
    EXPECT_EQ(mem.fifo[0][2], 0x00);
    EXPECT_EQ(mem.fifo[0][3], 0x80);
}

TEST(DspDma, CircularScratchWrapsAndWritesTheOffsetBack) {
    NullPeripherals np;
    Core core(Core::Config{}, &np);
    FlatMemory mem;
    mem.scratch.assign(4096, 0);
    for (u32 i = 0; i < 4; ++i) core.write_x(0x300 + i, 0x111111u * (i + 1));
    // Circular buffer of 8 bytes at scratch 0x200, offset 4, 4 samples of 24-bit msb (16 bytes): wraps twice.
    put_block(core, 0x50, Dma::kEol, 2u | (1u << 4) | (0xEu << 5) | (2u << 10), 4, 0x300, 4, 0x200, 7);
    Dma dma(core, mem);
    dma.write(Dma::NextBlock, 0x50);
    dma.write(Dma::Control, 1);
    EXPECT_TRUE(dma.last_error().empty()) << dma.last_error();
    u32 w;
    std::memcpy(&w, &mem.scratch[0x204], 4);
    EXPECT_EQ(w, 0x33333300u) << "sample 3 lands at offset 4 after two wraps";
    std::memcpy(&w, &mem.scratch[0x200], 4);
    EXPECT_EQ(w, 0x44444400u) << "sample 4 at the base";
    EXPECT_EQ(core.read_x(0x54), 4u) << "offset written back after the wrap";
}

namespace {
// Scratch and FIFO spaces backed by the capture's RAM image through the kernel's own page tables.
struct CaptureMemory : DmaMemory {
    u8* ram{nullptr};
    u32 size{0};
    u32 scratch_table{0}, scratch_entries{0}, fifo_table{0}, fifo_entries{0};
    std::map<u32, u64> fifo_bytes;
    bool translate(u32 table, u32 entries, u32 offset, u32& phys) const {
        const u32 page = offset / 4096;
        if (page >= entries) return false;
        u32 entry;
        if (static_cast<u64>(table) + page * 8 + 4 > size) return false;
        std::memcpy(&entry, ram + table + page * 8, 4);
        phys = entry + (offset % 4096);
        return phys + 1 <= size;
    }
    bool copy(u32 table, u32 entries, u32 offset, u8* out, const u8* in, u32 len) {
        while (len) {
            u32 phys;
            if (!translate(table, entries, offset, phys)) return false;
            const u32 chunk = std::min(len, 4096 - (offset % 4096));
            if (phys + chunk > size) return false;
            if (out) std::memcpy(out, ram + phys, chunk);
            else if (in) std::memcpy(ram + phys, in, chunk);
            if (out) out += chunk;
            if (in) in += chunk;
            offset += chunk;
            len -= chunk;
        }
        return true;
    }
    bool scratch_read(u32 offset, u8* out, u32 len) override { return copy(scratch_table, scratch_entries, offset, out, nullptr, len); }
    bool scratch_write(u32 offset, const u8* in, u32 len) override { return copy(scratch_table, scratch_entries, offset, nullptr, in, len); }
    bool fifo_read(u32, u8*, u32) override { return false; }
    bool fifo_write(u32 f, const u8*, u32 len) override {
        fifo_bytes[f] += len;
        return true;
    }
};
}  // namespace

TEST(DspDma, TheKernelsGpProgramRunsFramesAgainstTheCapturedMemory) {
    std::filesystem::path capture = std::filesystem::path(MV_SOURCE_DIR) / "scratch" / "boot-anim.mvcap";
    if (const char* env = std::getenv("MV_BOOT_CAPTURE")) capture = env;
    if (!std::filesystem::exists(capture)) GTEST_SKIP() << "no boot-animation capture at " << capture;
    pgraph::Replay r;
    std::string error;
    ASSERT_TRUE(r.open(capture, error)) << error;
    pgraph::Kelvin k;
    ASSERT_TRUE(r.run(k, nullptr, error, 0)) << error;
    CaptureMemory mem;
    mem.ram = r.mutable_ram();
    mem.size = r.memory().size;
    // The kernel's page tables (session 1 trace): scratch SGEs at 0x0398c000 (0x608 entries), FIFO SGEs at 0x03fa8000.
    mem.scratch_table = 0x0398c000;
    mem.scratch_entries = 0x608;
    mem.fifo_table = 0x03fa8000;
    mem.fifo_entries = 2;
    apu::Processor gp(Core::Config{0x1800, 0x800, 0x1000}, mem);
    gp.release_reset();
    ASSERT_TRUE(gp.running()) << gp.last_problem();
    EXPECT_EQ(gp.core().read_p(0), 0x0BF080u) << "the boot ROM loaded the program from scratch";
    u32 frames_halted = 0;
    for (u32 frame = 0; frame < 4; ++frame) {
        const u64 used = gp.run_frame(2'000'000);
        std::printf("frame %u: %llu cycles, %llu instructions so far, halted %d, PC %04x, DMA transfers %llu (%llu bytes)%s%s\n", frame,
                    static_cast<unsigned long long>(used), static_cast<unsigned long long>(gp.core().instructions()), gp.core().halted(),
                    gp.core().regs().pc, static_cast<unsigned long long>(gp.dma().transfers()), static_cast<unsigned long long>(gp.dma().bytes_moved()),
                    gp.core().error() ? " ERROR: " : "", gp.core().error() ? gp.core().error_text().c_str() : "");
        if (gp.core().error()) {
            std::printf("  at %s\n", gp.core().disassemble_at(gp.core().regs().pc).c_str());
            break;
        }
        if (gp.core().halted()) ++frames_halted;
    }
    EXPECT_EQ(frames_halted, 4u) << "the program halts itself at the end of every frame";
    EXPECT_FALSE(gp.core().error()) << gp.core().error_text();
    EXPECT_EQ(mem.fifo_bytes[0], 128u) << "one frame of 32 samples, four bytes each, to output FIFO 0";
    EXPECT_TRUE(gp.dma().last_error().empty()) << gp.dma().last_error();
    for (const auto& [f, n] : mem.fifo_bytes) std::printf("  FIFO %u: %llu bytes\n", f, static_cast<unsigned long long>(n));
}
