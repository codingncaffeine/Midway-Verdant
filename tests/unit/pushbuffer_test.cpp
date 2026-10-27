#include <gtest/gtest.h>

#include <algorithm>
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

TEST(Pushbuffer, ASinkCanStopTheWalkMidHeaderAndTheRestIsDeliveredOnResume) {
    Bus bus(1u << 20);
    PushbufferWalker w(bus);
    std::vector<PushbufferMethod> seen;
    w.set_sink([&](const PushbufferMethod& m) {
        seen.push_back(m);
        if (m.method == 0x0100 && m.value != 0) w.request_stop();  // the engine traps on this one
    });
    PushbufferWalker::DmaObject dma{0, (1u << 20) - 1, true};
    // One header of four increasing methods from 0x0100: the first carries data, so the walk must stop after it;
    // then a second header the resumed walk reaches.
    u32 p = 0x2000;
    put(bus, p, hdr(4, 1, 0x0100)); put(bus, p + 4, 0x304); put(bus, p + 8, 5); put(bus, p + 12, 6); put(bus, p + 16, 7); p += 20;
    put(bus, p, hdr(1, 1, 0x0300)); put(bus, p + 4, 9); p += 8;
    const u32 put_ptr = p;

    const u32 get1 = w.run(dma, 0x2000, put_ptr);
    EXPECT_EQ(get1, 0x2008u) << "stopped right after the trapping data word";
    EXPECT_TRUE(w.mid_header());
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_EQ(seen[0].method, 0x0100u);
    EXPECT_EQ(seen[0].value, 0x304u);
    EXPECT_FALSE(w.errored());

    const u32 get2 = w.run(dma, get1, put_ptr);
    EXPECT_EQ(get2, put_ptr);
    EXPECT_FALSE(w.mid_header());
    ASSERT_EQ(seen.size(), 5u);
    EXPECT_EQ(seen[1].method, 0x0104u) << "the header's remaining methods continue increasing";
    EXPECT_EQ(seen[1].value, 5u);
    EXPECT_EQ(seen[3].method, 0x010Cu);
    EXPECT_EQ(seen[4].method, 0x0300u);
    EXPECT_EQ(seen[4].value, 9u);
    EXPECT_EQ(seen[4].subchannel, 1u);
    EXPECT_EQ(w.methods_seen(), 5u);
}

// A bulk sink takes runs of a non-increasing header's words as the methods they are: the same methods, values, counts
// and GET as one by one, through a header it takes in parts, one it turns down, increasing headers it is never
// offered, a stop it asks for mid-run and a run cut short by the end of the DMA object.
TEST(Pushbuffer, ABulkSinkSeesTheSameMethodsAsOneByOne) {
    Bus bus(1u << 20);
    u32 p = 0x1000;
    put(bus, p, hdr(3, 0, 0x1800)); p += 4;  // increasing: never offered
    for (u32 i = 0; i < 3; ++i, p += 4) put(bus, p, 0x100 + i);
    put(bus, p, hdr(9, 0, 0x1800, false)); p += 4;  // taken in parts of at most four
    for (u32 i = 0; i < 9; ++i, p += 4) put(bus, p, 0x200 + i);
    put(bus, p, hdr(4, 0, 0x1818, false)); p += 4;  // turned down
    for (u32 i = 0; i < 4; ++i, p += 4) put(bus, p, 0x300 + i);
    put(bus, p, hdr(6, 1, 0x1800, false)); p += 4;  // the sink asks for a stop after its first part
    for (u32 i = 0; i < 6; ++i, p += 4) put(bus, p, 0x400 + i);
    const u32 put_at = p;

    struct Seen {
        u32 sub, method, value;
        bool operator==(const Seen&) const = default;
    };
    auto walk = [&](bool bulk, u32 limit, std::vector<Seen>& seen, PushbufferWalker& w) {
        w.set_sink([&](const PushbufferMethod& m) { seen.push_back({m.subchannel, m.method, m.value}); });
        bool stopped_once = false;
        if (bulk)
            w.set_bulk_sink([&](u32 sub, u32 method, const u32* words, u32 count) -> u32 {
                if (method != 0x1800) return 0;
                const u32 take = std::min(count, 4u);
                for (u32 i = 0; i < take; ++i) seen.push_back({sub, method, words[i]});
                if (sub == 1 && !stopped_once) {
                    stopped_once = true;
                    w.request_stop();
                }
                return take;
            });
        PushbufferWalker::DmaObject dma{0, limit, true};
        u32 get = 0x1000;
        for (int i = 0; i < 4 && get != put_at && !w.errored(); ++i) get = w.run(dma, get, put_at);
        return get;
    };
    for (const u32 limit : {(1u << 20) - 1, put_at - 9}) {  // the second ends the object two words into the last header
        SCOPED_TRACE(limit);
        PushbufferWalker a(bus), b(bus);
        std::vector<Seen> one, many;
        const u32 get_a = walk(false, limit, one, a);
        const u32 get_b = walk(true, limit, many, b);
        EXPECT_EQ(get_a, get_b);
        EXPECT_EQ(a.errored(), b.errored());
        EXPECT_EQ(a.methods_seen(), b.methods_seen());
        EXPECT_EQ(a.words_fetched(), b.words_fetched());
        EXPECT_TRUE(one == many) << one.size() << " methods one by one, " << many.size() << " with the bulk sink";
    }
}
