#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

#include "core/bus.h"
#include "core/scheduler.h"
#include "ide/ata_disk.h"
#include "ide/dvd_image.h"
#include "ide/hdd_format.h"
#include "ide/disk_image.h"
#include "ide/ide.h"

using namespace mv;

namespace {

// The console's cable: a locked hard disk as master, the DVD drive as slave, driven through the task-file
// ports the way the kernel drives them (select, command, poll the alternate status, move data).
struct Rig {
    Bus bus{4096};
    Scheduler sched;
    int irq_rises{0};
    bool irq_level{false};
    IdeController ide{bus, sched, [this](bool level) {
                          if (level && !irq_level) ++irq_rises;
                          irq_level = level;
                      }};
    DiskImage image;  // no file behind it: identify and the security commands never touch the platters
    AtaDisk* disk{nullptr};

    Rig() {
        auto d = std::make_unique<AtaDisk>(image);
        disk = d.get();
        d->set_locked(true);
        ide.attach(0, std::move(d));
        ide.attach(1, std::make_unique<AtapiCdrom>());
    }
    u8 status() { return static_cast<u8>(bus.pio_read(0x1F7, 1)); }      // acknowledges the interrupt
    u8 alt_status() { return static_cast<u8>(bus.pio_read(0x3F6, 1)); }  // does not
    void select_master() { bus.pio_write(0x1F6, 1, 0xA0); }
    void command(u8 c) {
        select_master();
        bus.pio_write(0x1F7, 1, c);
    }
    // Poll like software does, advancing the timeline a microsecond at a time; true when BSY cleared in time.
    bool wait_not_busy(Cycles budget = 10'000'000) {
        const Cycles end = sched.now() + budget;
        while (alt_status() & ata::kStatusBsy) {
            if (sched.now() >= end) return false;
            sched.advance_to(sched.now() + 733);
        }
        return true;
    }
    void write_block(const std::array<u8, 512>& block) {
        for (std::size_t i = 0; i < block.size(); i += 2) bus.pio_write(0x1F0, 2, static_cast<u32>(block[i] | (block[i + 1] << 8)));
    }
    std::array<u8, 512> read_block() {
        std::array<u8, 512> out{};
        for (std::size_t i = 0; i < out.size(); i += 2) {
            const u16 v = static_cast<u16>(bus.pio_read(0x1F0, 2));
            out[i] = static_cast<u8>(v);
            out[i + 1] = static_cast<u8>(v >> 8);
        }
        return out;
    }
    // Word 128 of IDENTIFY DEVICE: bit 1 security enabled, bit 2 locked.
    u16 identify_security_word() {
        command(0xEC);
        if (!wait_not_busy()) return 0xFFFF;
        if (!(status() & ata::kStatusDrq)) return 0xFFFF;
        const auto id = read_block();
        return static_cast<u16>(id[256] | (id[257] << 8));
    }
    static std::array<u8, 512> password_block(u8 fill) {
        std::array<u8, 512> b{};
        for (std::size_t i = 2; i < 34; ++i) b[i] = fill;  // word 0 = user password; the password follows
        return b;
    }
};

}  // namespace

TEST(Ide, SecurityUnlockCompletesAndInterrupts) {
    Rig r;
    EXPECT_TRUE(r.disk->locked());
    r.command(0xF2);
    EXPECT_TRUE(r.wait_not_busy());
    ASSERT_TRUE(r.alt_status() & ata::kStatusDrq) << "the drive must ask for the 512-byte password block";
    r.write_block(Rig::password_block(0x5A));
    // After the block the drive works, then reports ready with an interrupt: the kernel waits on exactly that.
    ASSERT_TRUE(r.wait_not_busy()) << "the drive stayed BUSY after the password block";
    const u8 st = r.status();
    EXPECT_TRUE(st & ata::kStatusDrdy);
    EXPECT_FALSE(st & ata::kStatusErr);
    EXPECT_FALSE(st & ata::kStatusDrq);
    EXPECT_EQ(r.irq_rises, 1);
    EXPECT_FALSE(r.disk->locked());
    // The drive takes the next command, and identifies as enabled but unlocked.
    const u16 security = r.identify_security_word();
    EXPECT_EQ(security & 0x0002, 0x0002);
    EXPECT_EQ(security & 0x0004, 0);
}

TEST(Ide, SecurityUnlockWithTheWrongPasswordAbortsAndStaysLocked) {
    Rig r;
    r.disk->set_password_oracle([] {
        std::array<u8, 32> pw{};
        pw.fill(0x11);
        return pw;
    });
    r.command(0xF2);
    ASSERT_TRUE(r.wait_not_busy());
    r.write_block(Rig::password_block(0x22));
    ASSERT_TRUE(r.wait_not_busy());
    const u8 st = r.status();
    EXPECT_TRUE(st & ata::kStatusErr);
    EXPECT_EQ(r.bus.pio_read(0x1F1, 1) & ata::kErrorAbrt, ata::kErrorAbrt);
    EXPECT_EQ(r.irq_rises, 1);
    EXPECT_TRUE(r.disk->locked());
    EXPECT_EQ(r.identify_security_word() & 0x0006, 0x0006);
}

TEST(Ide, SoftwareResetKeepsTheDriveUnlocked) {
    Rig r;
    r.command(0xF2);
    ASSERT_TRUE(r.wait_not_busy());
    r.write_block(Rig::password_block(0x5A));
    ASSERT_TRUE(r.wait_not_busy());
    ASSERT_FALSE(r.disk->locked());
    // SRST through the device control register: the security state survives it (only power cycles lock the drive).
    r.bus.pio_write(0x3F6, 1, 0x04);
    r.bus.pio_write(0x3F6, 1, 0x00);
    EXPECT_FALSE(r.disk->locked());
    EXPECT_EQ(r.identify_security_word() & 0x0006, 0x0002);
}

namespace {
// The kernel's media check on the DVD drive: TEST UNIT READY fails with CHECK CONDITION, then REQUEST SENSE
// fetches the 18 sense bytes — over PIO, or over the bus master when the PACKET command's features bit 0 asks.
struct DvdRig : Rig {
    static constexpr u16 kBm = 0xFF60;
    DvdRig() {
        ide.config_write(0x20, 4, kBm | 1);  // BAR4: the bus-master block
        const u32 prd[2] = {0x200, 0x80000040u};  // one entry: 64 bytes at 0x200, end of table
        bus.write_ram(0x100, prd, sizeof prd);
        bus.pio_write(kBm + 4, 4, 0x100);
        bus.pio_write(kBm + 2, 1, 0x06);  // clear the error and interrupt bits
    }
    void packet(const std::array<u8, 12>& cdb, bool dma, u16 byte_count) {
        bus.pio_write(0x1F6, 1, 0xB0);  // the slave
        bus.pio_write(0x1F1, 1, dma ? 0x01 : 0x00);
        bus.pio_write(0x1F4, 1, byte_count & 0xFF);
        bus.pio_write(0x1F5, 1, byte_count >> 8);
        bus.pio_write(0x1F7, 1, 0xA0);
        ASSERT_TRUE(wait_not_busy());
        ASSERT_TRUE(alt_status() & ata::kStatusDrq) << "the drive must ask for the command block";
        for (std::size_t i = 0; i < cdb.size(); i += 2) bus.pio_write(0x1F0, 2, static_cast<u32>(cdb[i] | (cdb[i + 1] << 8)));
    }
    void test_unit_ready_fails() {
        packet({0x00}, false, 0);
        ASSERT_TRUE(wait_not_busy());
        EXPECT_TRUE(status() & ata::kStatusErr);
        EXPECT_EQ(bus.pio_read(0x1F1, 1) >> 4, 0x02u) << "sense key NOT READY in the error register";
    }
};
}  // namespace

TEST(Ide, AtapiRequestSenseOverPio) {
    DvdRig r;
    r.test_unit_ready_fails();
    r.packet({0x03, 0, 0, 0, 18}, false, 18);
    ASSERT_TRUE(r.wait_not_busy());
    ASSERT_TRUE(r.status() & ata::kStatusDrq);
    EXPECT_EQ(r.bus.pio_read(0x1F2, 1), 0x02u) << "interrupt reason: data to the host";
    EXPECT_EQ(r.bus.pio_read(0x1F4, 1) | (r.bus.pio_read(0x1F5, 1) << 8), 18u);
    std::array<u8, 18> sense{};
    for (std::size_t i = 0; i < sense.size(); i += 2) {
        const u16 v = static_cast<u16>(r.bus.pio_read(0x1F0, 2));
        sense[i] = static_cast<u8>(v);
        sense[i + 1] = static_cast<u8>(v >> 8);
    }
    EXPECT_EQ(sense[0], 0x70);
    EXPECT_EQ(sense[2], 0x02);
    EXPECT_EQ(sense[12], 0x3A);
    ASSERT_TRUE(r.wait_not_busy());
    const u8 st = r.status();
    EXPECT_TRUE(st & ata::kStatusDrdy);
    EXPECT_FALSE(st & ata::kStatusDrq);
    EXPECT_EQ(r.bus.pio_read(0x1F2, 1), 0x03u) << "interrupt reason: command complete";
}

TEST(Ide, AtapiRequestSenseOverTheBusMaster) {
    DvdRig r;
    r.test_unit_ready_fails();
    const int rises_before = r.irq_rises;
    r.packet({0x03, 0, 0, 0, 18}, true, 0);
    // The driver starts the engine (read to memory) once the command block is written.
    r.bus.pio_write(DvdRig::kBm + 0, 1, 0x09);
    ASSERT_TRUE(r.wait_not_busy());
    const u8 bm = static_cast<u8>(r.bus.pio_read(DvdRig::kBm + 2, 1));
    EXPECT_TRUE(bm & 0x04) << "bus-master interrupt bit";
    EXPECT_FALSE(bm & 0x01) << "engine stopped";
    EXPECT_FALSE(bm & 0x02) << "no DMA error";
    const u8 st = r.status();
    EXPECT_TRUE(st & ata::kStatusDrdy);
    EXPECT_FALSE(st & ata::kStatusDrq) << "a DMA data phase never asks the host to read the data port";
    EXPECT_FALSE(st & ata::kStatusErr);
    EXPECT_EQ(r.bus.pio_read(0x1F2, 1), 0x03u) << "interrupt reason: command complete";
    EXPECT_EQ(r.irq_rises, rises_before + 1) << "one completion interrupt";
    std::array<u8, 18> sense{};
    ASSERT_TRUE(r.bus.read_ram(0x200, sense.data(), sense.size()));
    EXPECT_EQ(sense[0], 0x70);
    EXPECT_EQ(sense[2], 0x02);
    EXPECT_EQ(sense[12], 0x3A);
}

TEST(Ide, AtapiErrorWhileDmaIsArmedSetsTheBusMasterInterruptBit) {
    // The kernel arms the engine for a packet's data phase and its handler tests the bus-master interrupt bit
    // first; a controller sets that bit on every device interrupt edge (SFF-8038i), including a plain error.
    DvdRig r;
    r.packet({0x28, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, true, 0);  // READ(10) with no disc in the drive
    r.bus.pio_write(DvdRig::kBm + 0, 1, 0x09);
    ASSERT_TRUE(r.wait_not_busy());
    const u8 bm = static_cast<u8>(r.bus.pio_read(DvdRig::kBm + 2, 1));
    EXPECT_TRUE(bm & 0x04) << "interrupt bit set by the device's INTRQ edge";
    EXPECT_FALSE(bm & 0x02) << "no DMA error: the engine never ran";
    EXPECT_TRUE(r.status() & ata::kStatusErr);
    EXPECT_EQ(r.bus.pio_read(0x1F1, 1) >> 4, 0x02u) << "NOT READY";
}

TEST(Ide, AtapiModeSense10ReturnsTheSecurityPageWithoutMedia) {
    // The kernel's DVD initialisation reads mode page 0x3E (28 bytes: an 8-byte header and the 20-byte page) over
    // DMA before any disc is involved, and rejects the drive unless the page's byte 3 reads 1.
    DvdRig r;
    r.packet({0x5A, 0x00, 0x3E, 0, 0, 0, 0, 0x00, 0x1C, 0, 0, 0}, true, 0);
    r.bus.pio_write(DvdRig::kBm + 0, 1, 0x09);
    ASSERT_TRUE(r.wait_not_busy());
    const u8 st = r.status();
    EXPECT_FALSE(st & ata::kStatusErr) << "a mode page describes the drive, not the medium";
    EXPECT_TRUE(static_cast<u8>(r.bus.pio_read(DvdRig::kBm + 2, 1)) & 0x04);
    std::array<u8, 28> reply{};
    ASSERT_TRUE(r.bus.read_ram(0x200, reply.data(), reply.size()));
    EXPECT_EQ((reply[0] << 8) | reply[1], 26) << "mode data length excludes its own field";
    EXPECT_EQ(reply[6], 0);
    EXPECT_EQ(reply[7], 0) << "no block descriptors";
    EXPECT_EQ(reply[8], 0x3E) << "page code";
    EXPECT_EQ(reply[9], 18) << "page length";
    EXPECT_EQ(reply[10], 0) << "video partition selected";
    EXPECT_EQ(reply[11], 1) << "the byte the kernel insists on";
    EXPECT_EQ(reply[12], 0) << "not authenticated";
}

TEST(Ide, TheResetLineLocksTheDriveAgain) {
    Rig r;
    r.command(0xF2);
    ASSERT_TRUE(r.wait_not_busy());
    r.write_block(Rig::password_block(0x5A));
    ASSERT_TRUE(r.wait_not_busy());
    ASSERT_FALSE(r.disk->locked());
    r.ide.reset();
    EXPECT_TRUE(r.disk->locked()) << "a hardware reset puts a security-enabled drive back in the locked state";
    EXPECT_FALSE(r.irq_level);
    EXPECT_EQ(r.identify_security_word() & 0x0006, 0x0006);
}

namespace {
// A file with the XDVDFS volume magic at sector 32 of a partition starting at `partition` (sparse elsewhere).
std::filesystem::path make_disc(const char* name, u64 partition, u64 total, bool with_magic) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::FILE* f = std::fopen(path.string().c_str(), "wb");
    EXPECT_NE(f, nullptr);
    if (with_magic) {
        std::fseek(f, static_cast<long>(partition + 32 * 2048), SEEK_SET);
        std::fwrite("MICROSOFT*XBOX*MEDIA", 1, 20, f);
    }
    std::fclose(f);
    std::filesystem::resize_file(path, total);
    return path;
}
}  // namespace

TEST(DvdImage, RecognisesAnXisoAFullDumpAndRefusesOtherFiles) {
    std::filesystem::path xiso, full, plain;
    {  // the images close at the end of this block: Windows refuses to delete an open file
    std::string error;
    xiso = make_disc("mv-test-disc.xiso", 0, 64 * 2048, true);
    DvdImage a;
    ASSERT_TRUE(a.open(xiso, error)) << error;
    EXPECT_EQ(a.kind(), DvdImage::Kind::Xiso);
    EXPECT_EQ(a.sectors(), 64u);
    u8 sector[2048];
    ASSERT_TRUE(a.read(32, 1, sector));
    EXPECT_EQ(std::memcmp(sector, "MICROSOFT*XBOX*MEDIA", 20), 0);
    EXPECT_FALSE(a.read(64, 1, sector));  // past the end

    // A full dump: the game partition at byte 0x18300000, sector 0 of the disc is the video partition.
    full = make_disc("mv-test-disc.iso", DvdImage::kRedumpGamePartition, DvdImage::kRedumpGamePartition + 100 * 2048, true);
    DvdImage b;
    ASSERT_TRUE(b.open(full, error)) << error;
    EXPECT_EQ(b.kind(), DvdImage::Kind::Redump);
    EXPECT_EQ(b.sectors(), 100u);
    ASSERT_TRUE(b.read(32, 1, sector));
    EXPECT_EQ(std::memcmp(sector, "MICROSOFT*XBOX*MEDIA", 20), 0);

    // Control: the same sizes without the magic are not game discs.
    plain = make_disc("mv-test-disc-plain.iso", 0, 64 * 2048, false);
    DvdImage c;
    EXPECT_FALSE(c.open(plain, error));
    EXPECT_NE(error.find("not an Xbox game disc"), std::string::npos) << error;

    }
    std::filesystem::remove(xiso);
    std::filesystem::remove(full);
    std::filesystem::remove(plain);
}

TEST(DvdImage, ASplitImageIsReadAsOneDiscAndAFirstPartAloneSaysTheRestIsMissing) {
    // NAME.1.iso + NAME.2.iso (cut for FAT32): the second part continues the first. Every sector carries its own
    // number so a read across the cut can be checked byte for byte.
    const auto dir = std::filesystem::temp_directory_path();
    const auto one = dir / "mv-test-split.1.iso", two = dir / "mv-test-split.2.iso", lone = dir / "mv-test-lone.1.iso";
    auto write_sectors = [](const std::filesystem::path& p, u32 first, u32 count, bool magic) {
        std::FILE* f = std::fopen(p.string().c_str(), "wb");
        ASSERT_NE(f, nullptr);
        std::vector<u8> s(2048);
        for (u32 i = 0; i < count; ++i) {
            std::fill(s.begin(), s.end(), 0);
            const u32 n = first + i;
            std::memcpy(s.data() + 100, &n, 4);
            if (magic && n == 32) std::memcpy(s.data(), "MICROSOFT*XBOX*MEDIA", 20);
            std::fwrite(s.data(), 1, s.size(), f);
        }
        std::fclose(f);
    };
    write_sectors(one, 0, 40, true);
    write_sectors(two, 40, 24, false);
    write_sectors(lone, 0, 40, true);
    {
        std::string error;
        DvdImage d;
        ASSERT_TRUE(d.open(one, error)) << error;
        EXPECT_EQ(d.parts(), 2u);
        EXPECT_EQ(d.sectors(), 64u) << "both parts";
        EXPECT_TRUE(d.note().empty());
        std::vector<u8> buf(4 * 2048);
        ASSERT_TRUE(d.read(38, 4, buf.data())) << "a read across the cut";
        for (u32 i = 0; i < 4; ++i) {
            u32 n = 0;
            std::memcpy(&n, buf.data() + i * 2048 + 100, 4);
            EXPECT_EQ(n, 38 + i) << "sector " << 38 + i;
        }
        d.begin_read(60, 4);  // the drive's threaded path, in the second part
        std::vector<u8> out;
        ASSERT_TRUE(d.end_read(out));
        u32 last = 0;
        std::memcpy(&last, out.data() + 3 * 2048 + 100, 4);
        EXPECT_EQ(last, 63u);

        DvdImage a;
        ASSERT_TRUE(a.open(lone, error)) << error;
        EXPECT_EQ(a.parts(), 1u);
        EXPECT_EQ(a.sectors(), 40u);
        EXPECT_NE(a.note().find("mv-test-lone.2.iso is not beside it"), std::string::npos) << a.note();
        u8 sector[2048];
        EXPECT_FALSE(a.read(40, 1, sector));
        EXPECT_EQ(a.past_end_reads(), 1u);
        EXPECT_EQ(a.first_past_end_lba(), 40u);
    }
    std::filesystem::remove(one);
    std::filesystem::remove(two);
    std::filesystem::remove(lone);
}

TEST(HddFormat, BlankDriveMatchesTheMeasuredRetailLayout) {
    // Where the data area starts, as measured on a working drive image (local-notes/hw/ide.md): 16-bit tables on
    // the cache partitions and C:, a 32-bit table on E:.
    EXPECT_EQ(fatx_data_offset(0x2EE00000ull, 32), 0x19000u);   // X, Y, Z
    EXPECT_EQ(fatx_data_offset(0x1F400000ull, 32), 0x11000u);   // C
    EXPECT_EQ(fatx_data_offset(0x131F00000ull, 32), 0x133000u); // E

    const auto path = std::filesystem::temp_directory_path() / "mv-test-blank-hdd.raw";
    std::string error;
    ASSERT_TRUE(create_blank_hdd(path, 0x12345678u, error)) << error;
    EXPECT_EQ(std::filesystem::file_size(path), kRetailDiskBytes);
    {
        DiskImage disk;
        ASSERT_TRUE(disk.open(path, true, error)) << error;
        u8 s[512];
        for (const HddPartition& p : kRetailPartitions) {
            ASSERT_TRUE(disk.read(p.offset / 512, 1, s));
            EXPECT_EQ(std::memcmp(s, "FATX", 4), 0) << p.name;
            EXPECT_EQ(s[8], 32u) << p.name;   // sectors per cluster
            EXPECT_EQ(s[12], 1u) << p.name;   // root cluster
            EXPECT_EQ(s[18], 0xFFu) << p.name;
            ASSERT_TRUE(disk.read((p.offset + 0x1000) / 512, 1, s));
            EXPECT_EQ(s[0], 0xF8u) << p.name;  // media entry
            const u64 root = p.offset + fatx_data_offset(p.size, 32);
            ASSERT_TRUE(disk.read(root / 512, 1, s));
            if (std::strcmp(p.name, "E") == 0) {
                EXPECT_EQ(s[0], 5u);
                EXPECT_EQ(s[1], 0x10u);
                EXPECT_EQ(std::memcmp(s + 2, "TDATA", 5), 0);
                EXPECT_EQ(std::memcmp(s + 64 + 2, "UDATA", 5), 0);
                EXPECT_EQ(s[0x2C], 2u);
                EXPECT_EQ(s[64 + 0x2C], 3u);
            } else {
                EXPECT_EQ(s[0], 0xFFu) << p.name << ": an empty root directory";
            }
        }
    }
    std::filesystem::remove(path);
}

TEST(Ide, TheDvdDriveIsBecomingReadyWhileTheDiscSpinsUp) {
    std::filesystem::path path;
    {
    DvdRig r;
    path = make_disc("mv-test-spinup.xiso", 0, 64 * 2048, true);
    auto disc = std::make_unique<DvdImage>();
    std::string error;
    ASSERT_TRUE(disc->open(path, error)) << error;
    static_cast<AtapiCdrom*>(r.ide.primary().device(1))->insert(std::move(disc));
    // Just after power-on: NOT READY, LOGICAL UNIT IS IN PROCESS OF BECOMING READY (02/04/01), not "no medium".
    r.test_unit_ready_fails();
    r.packet({0x03, 0, 0, 0, 18}, false, 18);
    ASSERT_TRUE(r.wait_not_busy());
    std::array<u8, 18> sense{};
    for (std::size_t i = 0; i < sense.size(); i += 2) {
        const u16 v = static_cast<u16>(r.bus.pio_read(0x1F0, 2));
        sense[i] = static_cast<u8>(v);
        sense[i + 1] = static_cast<u8>(v >> 8);
    }
    EXPECT_EQ(sense[2], 0x02);
    EXPECT_EQ(sense[12], 0x04);
    EXPECT_EQ(sense[13], 0x01);
    // Still spinning just before the spin-up time; ready just after it.
    r.sched.advance_to(cycles_from_ns(AtapiCdrom::kSpinUpNs - 10'000'000));
    r.test_unit_ready_fails();
    r.sched.advance_to(cycles_from_ns(AtapiCdrom::kSpinUpNs + 10'000'000));
    r.packet({0x00}, false, 0);
    ASSERT_TRUE(r.wait_not_busy());
    EXPECT_FALSE(r.status() & ata::kStatusErr) << "the disc is up to speed: TEST UNIT READY passes";
    }
    std::filesystem::remove(path);
}

// Both devices on the cable latch the command-block registers: a driver may load the LBA, then select the drive.
TEST(Ide, TaskFileWritesReachBothDevices) {
    Rig r;
    r.bus.pio_write(0x1F6, 1, 0xB0);  // the slave
    r.bus.pio_write(0x1F3, 1, 0x5A);  // LBA low
    r.bus.pio_write(0x1F2, 1, 0x07);  // sector count
    r.bus.pio_write(0x1F6, 1, 0xA0);  // the master
    EXPECT_EQ(r.bus.pio_read(0x1F3, 1), 0x5Au);
    EXPECT_EQ(r.bus.pio_read(0x1F2, 1), 0x07u);
}

// Writes land in the image's write cache and a writer thread puts them in the file: every read sees the latest write
// at once (from the cache or the file, never an older sector), and everything is in the file once the image is gone.
TEST(DiskImage, WritesAreReadBackAtOnceAndReachTheFile) {
    const auto path = std::filesystem::temp_directory_path() / "mv-test-write-cache.raw";
    constexpr u64 kSectors = 80 * 2048;  // 80 MiB: more than the cache holds
    {
        std::FILE* f = std::fopen(path.string().c_str(), "wb");
        ASSERT_NE(f, nullptr);
        std::fseek(f, static_cast<long>(kSectors * 512 - 1), SEEK_SET);
        std::fputc(0, f);
        std::fclose(f);
    }
    std::string error;
    std::vector<u8> big(static_cast<std::size_t>(kSectors) * 512);
    for (std::size_t i = 0; i < big.size(); i += 512) std::memset(&big[i], static_cast<int>(1 + i / 512 % 250), 512);
    {
        DiskImage disk;
        ASSERT_TRUE(disk.open(path, false, error)) << error;
        // One sector rewritten many times, each read straight after.
        u8 s[512];
        for (u32 v = 0; v < 2000; ++v) {
            std::memset(s, static_cast<int>(v & 0xFF), sizeof s);
            ASSERT_TRUE(disk.write(5, 1, s));
            u8 r[512];
            ASSERT_TRUE(disk.read(5, 1, r));
            ASSERT_EQ(r[0], static_cast<u8>(v)) << "write " << v;
        }
        // A large copy (the cache fills and drains), its far end read at once, in one read and on the reader thread.
        for (u64 at = 0; at < kSectors - 64; at += 2048) ASSERT_TRUE(disk.write(64 + at, static_cast<u32>(std::min<u64>(2048, kSectors - 64 - at)), &big[(64 + at) * 512]));
        std::vector<u8> tail(16 * 512);
        ASSERT_TRUE(disk.read(kSectors - 16, 16, tail.data()));
        EXPECT_EQ(std::memcmp(tail.data(), &big[(kSectors - 16) * 512], tail.size()), 0) << "the last sectors written";
        disk.begin_read(kSectors - 8, 8);
        std::vector<u8> got;
        ASSERT_TRUE(disk.end_read(got));
        EXPECT_EQ(std::memcmp(got.data(), &big[(kSectors - 8) * 512], got.size()), 0) << "the reader thread sees the cache too";
    }
    {
        DiskImage disk;
        ASSERT_TRUE(disk.open(path, true, error)) << error;
        u8 r[512];
        ASSERT_TRUE(disk.read(5, 1, r));
        EXPECT_EQ(r[0], static_cast<u8>(1999 & 0xFF)) << "the last write is in the file";
        std::vector<u8> all(static_cast<std::size_t>(kSectors - 64) * 512);
        ASSERT_TRUE(disk.read(64, static_cast<u32>(kSectors - 64), all.data()));
        EXPECT_EQ(std::memcmp(all.data(), &big[64 * 512], all.size()), 0) << "the whole copy is in the file";
    }
    std::filesystem::remove(path);
}
