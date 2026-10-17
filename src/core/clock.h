#pragma once
#include "core/types.h"

namespace mv {

// The console's clocks, as exact ratios of the master clock. The CPU core clock is the master timeline:
// 133.333 MHz front-side bus times 5.5 = 733.333 MHz = 2200/3 MHz. The GPU core runs at 233.333 MHz = 700/3 MHz,
// so one GPU cycle is exactly 22/7 CPU cycles. The other clocks are not integer ratios; conversions go through
// muldiv64, which is exact in 128-bit arithmetic.
namespace clk {

inline constexpr u64 kCpuHzNum = 2'200'000'000;  // Hz = kCpuHzNum / kCpuHzDen
inline constexpr u64 kCpuHzDen = 3;
inline constexpr double kCpuHz = 733'333'333.333;

inline constexpr u64 kGpuHzNum = 700'000'000;
inline constexpr u64 kGpuHzDen = 3;

// 8254 timer input. xemu's model and the kernel's 1 ms divisor agree on 1.125 MHz; the wiki says "6% off a PC",
// which is 1.193182 / 1.125. To be confirmed from the divisor the kernel writes (Phase 1 trace).
inline constexpr u64 kPitHz = 1'125'000;
// ACPI power-management timer (standard rate; to be confirmed).
inline constexpr u64 kPmTimerHz = 3'579'545;
// Audio: 48 kHz, 32-sample frames.
inline constexpr u64 kAudioHz = 48'000;
inline constexpr u64 kApuFramesPerSecond = 1'500;

}  // namespace clk

// floor(a * b / c) without overflow for the ranges the timeline uses (a < 2^64, b, c < 2^40). Exact.
u64 muldiv64(u64 a, u64 b, u64 c);
// ceil(a * b / c): the first cycle at or after a device tick, so a deadline converted back never lands early.
u64 muldiv64_ceil(u64 a, u64 b, u64 c);

// Convert between the master timeline and a device clock given in Hz.
inline Cycles cycles_from_ticks(u64 ticks, u64 tick_hz) {
    return muldiv64(ticks, clk::kCpuHzNum, tick_hz * clk::kCpuHzDen);
}
inline u64 ticks_from_cycles(Cycles cycles, u64 tick_hz) {
    return muldiv64(cycles, tick_hz * clk::kCpuHzDen, clk::kCpuHzNum);
}
inline Cycles cycles_from_ticks_ceil(u64 ticks, u64 tick_hz) {
    return muldiv64_ceil(ticks, clk::kCpuHzNum, tick_hz * clk::kCpuHzDen);
}
inline Cycles cycles_from_ns(u64 ns) { return muldiv64(ns, clk::kCpuHzNum, 1'000'000'000ull * clk::kCpuHzDen); }
inline Cycles cycles_from_ns_ceil(u64 ns) { return muldiv64_ceil(ns, clk::kCpuHzNum, 1'000'000'000ull * clk::kCpuHzDen); }
inline u64 ns_from_cycles(Cycles cycles) { return muldiv64(cycles, 1'000'000'000ull * clk::kCpuHzDen, clk::kCpuHzNum); }

}  // namespace mv
