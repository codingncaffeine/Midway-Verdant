#pragma once
// What the work a pushbuffer asks for costs the NV2A, in its own clocks (NVPLL 233.33 MHz), for the accurate core's
// timed pusher (PLAN D28): the GPU takes console time, so a heavy frame misses its vblank as on hardware. Stage A
// counts the front end, vertices, primitive setup and clears; the pixels of draws come with stage B (occlusion
// queries). Published rates (Xbox technical specifications; GeForce 3 series): 4 pixel pipelines x 2 texture units
// at 233 MHz, 115 M vertices/s, two vertex shader units. Constants marked [estimate] await a console measurement.
#include "core/clock.h"
#include "core/types.h"
#include "nv2a/pgraph/kelvin_state.h"

namespace mv::gpu_timing {

inline constexpr u64 kMethodClocks = 1;           // the front end takes a method a clock [estimate]
inline constexpr u64 kVertexMinClocks = 2;        // 115 M vertices/s at 233 MHz
inline constexpr u64 kVertexUnits = 2;            // two vertex shaders: a program instruction per unit per clock
inline constexpr u64 kFixedFunctionInsns = 8;     // the fixed-function transform as a program length [estimate]
inline constexpr u64 kTriangleSetupClocks = 4;    // [estimate]
inline constexpr u64 kPixelsPerClock = 4;         // four pixel pipelines: clears
inline constexpr u32 kMaxProgramInsns = 136;      // the transform program memory

// Vertices the batch transforms (array elements, array ranges, inline and immediate vertices).
u64 vertex_count(const pgraph::KelvinState& state, const pgraph::DrawBatch& batch);
// Primitives a primitive type makes of `vertices` vertices (points and lines count as primitives too).
u64 primitive_count(pgraph::Primitive primitive, u64 vertices);
// Instructions of the transform program from its start slot to the one marked last (fixed function: the estimate).
u64 program_length(const pgraph::KelvinState& state);
u64 draw_clocks(const pgraph::KelvinState& state, const pgraph::DrawBatch& batch);
// CLEAR_SURFACE: the clip rectangle's pixels at four a clock.
u64 clear_clocks(const pgraph::KelvinState& state);

// GPU clocks and CPU cycles: exactly 7 GPU clocks per 22 CPU cycles (233.33 and 733.33 MHz).
inline Cycles cycles_from_gpu(u64 gpu_clocks) { return muldiv64_ceil(gpu_clocks, clk::kCpuHzNum, clk::kGpuHzNum); }
inline u64 gpu_from_cycles(Cycles cycles) { return muldiv64(cycles, clk::kGpuHzNum, clk::kCpuHzNum); }

}  // namespace mv::gpu_timing
