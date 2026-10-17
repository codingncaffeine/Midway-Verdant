#include <gtest/gtest.h>

#include "smbus/eeprom_image.h"

using namespace mv;
using namespace mv::eeprom;

TEST(Eeprom, SectionChecksumMakesTheReadAlgorithmAllOnes) {
    std::array<u8, 0x2C> data{};
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<u8>(i * 37 + 11);
    const u32 ck = section_checksum_make(data);
    std::array<u8, 0x30> section{};
    for (int i = 0; i < 4; ++i) section[static_cast<std::size_t>(i)] = static_cast<u8>(ck >> (8 * i));
    std::copy(data.begin(), data.end(), section.begin() + 4);
    EXPECT_EQ(section_checksum_read(section), 0xFFFFFFFFu);
    // A corrupted byte breaks it (the control).
    section[10] ^= 0x01;
    EXPECT_NE(section_checksum_read(section), 0xFFFFFFFFu);
}

TEST(Eeprom, GeneratedImageVerifiesWithItsKeyOnly) {
    std::array<u8, 16> key{};
    for (std::size_t i = 0; i < 16; ++i) key[i] = static_cast<u8>(0xA0 + i);
    Settings s;
    for (std::size_t i = 0; i < 16; ++i) s.hdd_key[i] = static_cast<u8>(0x10 + i);
    s.region = kRegionEurope;
    const auto img = generate(key, s);
    std::string why;
    const auto hdd = verify(key, img, &why);
    ASSERT_TRUE(hdd.has_value()) << why;
    EXPECT_EQ(*hdd, s.hdd_key);
    // Plain fields landed where the kernel reads them.
    EXPECT_EQ(img[0x34], '1');
    EXPECT_EQ(img[0x40], 0x00);
    EXPECT_EQ(img[0x41], 0x50);
    EXPECT_EQ(img[0x42], 0xF2);
    EXPECT_EQ(img[0x58], 0x00);
    EXPECT_EQ(img[0x59], 0x01);
    EXPECT_EQ(img[0x5A], 0x40);
    EXPECT_EQ(img[0x90], 1);
    // A different key rejects it; a flipped encrypted byte rejects it.
    std::array<u8, 16> other = key;
    other[3] ^= 0x80;
    EXPECT_FALSE(verify(other, img, &why).has_value());
    auto bad = img;
    bad[0x20] ^= 0x01;
    EXPECT_FALSE(verify(key, bad, &why).has_value());
    auto bad2 = img;
    bad2[0x70] ^= 0x01;
    EXPECT_FALSE(verify(key, bad2, &why).has_value());
    EXPECT_EQ(why, "user section checksum");
}
