#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/bus.h"
#include "nv2a/capture.h"

using namespace mv;

TEST(Capture, RoundTripsEveryRecordType) {
    const auto path = std::filesystem::temp_directory_path() / "mv-capture-test.mvcap";
    std::vector<u8> page(capture::kPageSize * 2);
    for (std::size_t i = 0; i < page.size(); ++i) page[i] = static_cast<u8>(i * 7);
    {
        capture::Writer w;
        std::string error;
        ASSERT_TRUE(w.open(path, 64u << 20, error)) << error;
        w.run(123456789ull, 0x1000, 0x2000);
        w.pages(5, 2, page.data());
        w.object(0xD, std::array<u32, 4>{0x00000097u, 0x11223344u, 0, 0}.data());
        w.method(0, 0x97, 0x17FC, 6);
        w.method(3, 0x9F, 0x02FC, 3);
        w.method(7, 0, 0, 0xD);
        w.reg(0x600800, 0x03C00000);
        EXPECT_EQ(w.methods_written(), 3u);
        EXPECT_EQ(w.pages_written(), 2u);
    }
    capture::Reader r;
    std::string error;
    ASSERT_TRUE(r.open(path, error)) << error;
    EXPECT_EQ(r.ram_size(), 64u << 20);
    capture::Record rec;
    ASSERT_TRUE(r.next(rec));
    EXPECT_EQ(rec.type, capture::Rec::Run);
    EXPECT_EQ(rec.cycles, 123456789ull);
    EXPECT_EQ(rec.get, 0x1000u);
    EXPECT_EQ(rec.put, 0x2000u);
    ASSERT_TRUE(r.next(rec));
    EXPECT_EQ(rec.type, capture::Rec::Pages);
    EXPECT_EQ(rec.first_page, 5u);
    EXPECT_EQ(rec.page_count, 2u);
    EXPECT_EQ(rec.page_data[4096 + 1], static_cast<u8>(4097 * 7));
    ASSERT_TRUE(r.next(rec));
    EXPECT_EQ(rec.type, capture::Rec::Object);
    EXPECT_EQ(rec.handle, 0xDu);
    EXPECT_EQ(rec.dw[1], 0x11223344u);
    ASSERT_TRUE(r.next(rec));
    EXPECT_EQ(rec.type, capture::Rec::Method);
    EXPECT_EQ(rec.subchannel, 0u);
    EXPECT_EQ(rec.cls, 0x97u);
    EXPECT_EQ(rec.method, 0x17FCu);
    EXPECT_EQ(rec.value, 6u);
    ASSERT_TRUE(r.next(rec));
    EXPECT_EQ(rec.subchannel, 3u);
    EXPECT_EQ(rec.cls, 0x9Fu);
    EXPECT_EQ(rec.method, 0x2FCu);
    ASSERT_TRUE(r.next(rec));
    EXPECT_EQ(rec.subchannel, 7u);
    EXPECT_EQ(rec.cls, 0u);
    EXPECT_EQ(rec.method, 0u);
    EXPECT_EQ(rec.value, 0xDu);
    ASSERT_TRUE(r.next(rec));
    EXPECT_EQ(rec.type, capture::Rec::Register);
    EXPECT_EQ(rec.offset, 0x600800u);
    EXPECT_EQ(rec.value, 0x03C00000u);
    EXPECT_FALSE(r.next(rec));
    EXPECT_TRUE(r.error().empty()) << r.error();
    r.close();  // Windows refuses to delete an open file
    std::filesystem::remove(path);
}

TEST(Capture, RefusesAForeignFile) {
    const auto path = std::filesystem::temp_directory_path() / "mv-capture-foreign.bin";
    {
        std::FILE* f = std::fopen(path.string().c_str(), "wb");
        ASSERT_NE(f, nullptr);
        std::fputs("not a capture at all, just text", f);
        std::fclose(f);
    }
    capture::Reader r;
    std::string error;
    EXPECT_FALSE(r.open(path, error));
    EXPECT_NE(error.find("magic"), std::string::npos) << error;
    std::filesystem::remove(path);
}

TEST(BusDirty, DeviceWritesMarkPagesOnlyWhenTracking) {
    Bus bus(1u << 20);
    std::vector<u32> pages;
    const u32 word = 0x12345678;
    bus.write_ram(0x3FFE, &word, 4);  // untracked: nothing recorded
    bus.take_dirty_pages(pages);
    EXPECT_TRUE(pages.empty());

    bus.set_dirty_tracking(true);
    bus.write_ram(0x3FFE, &word, 4);  // straddles pages 3 and 4
    bus.write(0xF0000000u + 0x9000, 2, 0xABCD);  // through the GPU-side alias: page 9
    bus.write64(0x20000, 1);  // page 32
    bus.take_dirty_pages(pages);
    EXPECT_EQ(pages, (std::vector<u32>{3, 4, 9, 32}));
    pages.clear();
    bus.take_dirty_pages(pages);
    EXPECT_TRUE(pages.empty()) << "taking the pages clears the record";
}
