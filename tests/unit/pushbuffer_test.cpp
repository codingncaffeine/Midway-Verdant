#include <gtest/gtest.h>

#include <vector>

#include "core/bus.h"
#include "nv2a/pushbuffer.h"

using namespace mv;

namespace {
void put(Bus& bus, u32 addr, u32 v) { bus.write(addr, 4, v); }
u32 hdr(u32 count, u32 subch, u32 method, bool increasing = true) {
    return (increasing ? 0u : 0x40000000u) | (count << 18) | (subch << 13) | method;
}
}  // namespace

TEST(Pushbuffer, DecodesMethodsJumpsCallsAndReturns) {
    Bus bus(1u << 20);
    PushbufferWalker w(bus);
    std::vector<PushbufferMethod> seen;
    w.set_sink([&](const PushbufferMethod& m) { seen.push_back(m); });
    PushbufferWalker::DmaObject dma{0, (1u << 20) - 1, true};

    // At 0: a jump to 0x1000, as the console's kernel plants it.
    put(bus, 0x0, 0x1000 | 1u);
    // 0x1000: SET_OBJECT on subchannel 0, then three increasing methods from 0x200, then a call to a subroutine,
    // then a non-increasing burst of two values to method 0x1800 on subchannel 2.
    u32 p = 0x1000;
    put(bus, p, hdr(1, 0, 0x0000)); put(bus, p + 4, 0xD); p += 8;
    put(bus, p, hdr(3, 0, 0x0200)); put(bus, p + 4, 11); put(bus, p + 8, 22); put(bus, p + 12, 33); p += 16;
    put(bus, p, 0x2000 | 2u); p += 4;  // call
    put(bus, p, hdr(2, 2, 0x1800, false)); put(bus, p + 4, 0xAAAA); put(bus, p + 8, 0xBBBB); p += 12;
    const u32 put_at = p;
    // Subroutine at 0x2000: one method, then return.
    put(bus, 0x2000, hdr(1, 1, 0x0100)); put(bus, 0x2004, 0x77); put(bus, 0x2008, 0x00020000);

    const u32 get = w.run(dma, 0, put_at);
    EXPECT_EQ(get, put_at);
    EXPECT_FALSE(w.errored()) << w.last_error();
    ASSERT_EQ(seen.size(), 7u);
    EXPECT_EQ(seen[0].method, 0u);
    EXPECT_EQ(seen[0].value, 0xDu);
    EXPECT_EQ(seen[1].method, 0x200u);
    EXPECT_EQ(seen[2].method, 0x204u);
    EXPECT_EQ(seen[3].method, 0x208u);
    EXPECT_EQ(seen[3].value, 33u);
    EXPECT_EQ(seen[4].subchannel, 1u);
    EXPECT_EQ(seen[4].method, 0x100u);  // from the subroutine
    EXPECT_EQ(seen[5].subchannel, 2u);
    EXPECT_EQ(seen[5].method, 0x1800u);
    EXPECT_EQ(seen[6].method, 0x1800u);  // non-increasing
    EXPECT_EQ(seen[6].value, 0xBBBBu);
    EXPECT_EQ(w.methods_seen(), 7u);
}

TEST(Pushbuffer, StopsAtAnUnknownHeaderAndOutsideTheDmaObject) {
    Bus bus(1u << 20);
    PushbufferWalker w(bus);
    PushbufferWalker::DmaObject dma{0, 0x3FFF, true};
    put(bus, 0x100, 0xE0000000u);  // not a header we know
    const u32 get = w.run(dma, 0x100, 0x200);
    EXPECT_TRUE(w.errored());
    EXPECT_EQ(get, 0x104u);
    // A GET beyond the object's limit fails the fetch rather than reading stray memory.
    PushbufferWalker w2(bus);
    EXPECT_EQ(w2.run(dma, 0x4000, 0x4100), 0x4000u);
    EXPECT_TRUE(w2.errored());
    // An invalid object runs nothing.
    PushbufferWalker w3(bus);
    EXPECT_EQ(w3.run(PushbufferWalker::DmaObject{}, 0, 0x100), 0u);
}
