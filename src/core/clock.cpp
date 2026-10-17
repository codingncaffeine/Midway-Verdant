#include "core/clock.h"

namespace mv {

namespace {
// Quotient and remainder of a * b / c, exact for every u64 a and b, c < 2^64 with the product under 2^128.
struct QuotRem {
    u64 quotient;
    u64 remainder;
};

QuotRem muldiv64_qr(u64 a, u64 b, u64 c) {
#if defined(__SIZEOF_INT128__)
    __extension__ using u128 = unsigned __int128;
    const u128 product = static_cast<u128>(a) * b;
    return {static_cast<u64>(product / c), static_cast<u64>(product % c)};
#else
    // Portable 128-by-64 division: a*b as (hi, lo), then long division one bit at a time.
    const u64 a_lo = a & 0xFFFFFFFFu, a_hi = a >> 32;
    const u64 b_lo = b & 0xFFFFFFFFu, b_hi = b >> 32;
    const u64 p0 = a_lo * b_lo;
    const u64 p1 = a_lo * b_hi;
    const u64 p2 = a_hi * b_lo;
    const u64 p3 = a_hi * b_hi;
    const u64 mid = (p0 >> 32) + (p1 & 0xFFFFFFFFu) + (p2 & 0xFFFFFFFFu);
    const u64 lo = (p0 & 0xFFFFFFFFu) | (mid << 32);
    const u64 hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    u64 quotient = 0, remainder = 0;
    for (int i = 127; i >= 0; --i) {
        const u64 bit = (i >= 64) ? ((hi >> (i - 64)) & 1u) : ((lo >> i) & 1u);
        const u64 carry = remainder >> 63;
        remainder = (remainder << 1) | bit;
        if (carry || remainder >= c) {
            remainder -= c;
            if (i < 64) quotient |= (1ull << i);
        }
    }
    return {quotient, remainder};
#endif
}
}  // namespace

u64 muldiv64(u64 a, u64 b, u64 c) { return muldiv64_qr(a, b, c).quotient; }

u64 muldiv64_ceil(u64 a, u64 b, u64 c) {
    const QuotRem qr = muldiv64_qr(a, b, c);
    return qr.remainder ? qr.quotient + 1 : qr.quotient;
}

}  // namespace mv
