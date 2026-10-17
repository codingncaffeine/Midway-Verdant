#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "crypto/sha1.h"

using namespace mv;
using namespace mv::crypto;

namespace {
std::string hex(std::span<const u8> d) {
    std::string s;
    for (u8 b : d) {
        char buf[3];
        std::snprintf(buf, sizeof buf, "%02x", b);
        s += buf;
    }
    return s;
}
std::span<const u8> bytes(const char* s) { return {reinterpret_cast<const u8*>(s), std::strlen(s)}; }
}  // namespace

TEST(Crypto, Sha1KnownAnswers) {
    // FIPS 180-1 examples.
    EXPECT_EQ(hex(Sha1::hash(bytes("abc"))), "a9993e364706816aba3e25717850c26c9cd0d89d");
    EXPECT_EQ(hex(Sha1::hash(bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))),
              "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    EXPECT_EQ(hex(Sha1::hash(bytes(""))), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    // Streaming in odd pieces gives the same digest.
    Sha1 s;
    const char* msg = "The quick brown fox jumps over the lazy dog";
    s.update(bytes(msg).subspan(0, 7));
    s.update(bytes(msg).subspan(7));
    EXPECT_EQ(hex(s.finish()), "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12");
}

TEST(Crypto, HmacSha1Rfc2202) {
    std::array<u8, 20> key1{};
    key1.fill(0x0b);
    EXPECT_EQ(hex(hmac_sha1(key1, bytes("Hi There"))), "b617318655057264e28bc0b6fb378c8ef146be00");
    EXPECT_EQ(hex(hmac_sha1(bytes("Jefe"), bytes("what do ya want for nothing?"))), "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79");
    std::array<u8, 80> key_long{};
    key_long.fill(0xaa);
    EXPECT_EQ(hex(hmac_sha1(key_long, bytes("Test Using Larger Than Block-Size Key - Hash Key First"))),
              "aa4ae5e15272d00e95705637ce8a3b55ed402112");
    // Two-part messages hash as their concatenation.
    EXPECT_EQ(hex(hmac_sha1(bytes("Jefe"), bytes("what do ya want "), bytes("for nothing?"))),
              "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79");
}

TEST(Crypto, Rc4KnownAnswersAndRoundTrip) {
    // RFC 6229 test vector: key 0102030405, first 16 bytes of the keystream.
    const std::array<u8, 5> key{1, 2, 3, 4, 5};
    std::array<u8, 16> zeros{};
    Rc4(key).process(zeros);
    EXPECT_EQ(hex(zeros), "b2396305f03dc027ccc3524a0a1118a8");
    // Wikipedia's classic: Key "Key", plaintext "Plaintext".
    std::array<u8, 9> pt{'P', 'l', 'a', 'i', 'n', 't', 'e', 'x', 't'};
    Rc4(bytes("Key")).process(pt);
    EXPECT_EQ(hex(pt), "bbf316e8d940af0ad3");
    Rc4(bytes("Key")).process(pt);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(pt.data()), 9), "Plaintext");
}
