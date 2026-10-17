#pragma once
#include <cstddef>
#include <cstdint>

namespace mv {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// A point on the master timeline, in CPU cycles (733.333 MHz). u64 lasts ~800 years.
using Cycles = u64;

// Physical address of the guest machine (32-bit bus).
using PhysAddr = u32;

}  // namespace mv
