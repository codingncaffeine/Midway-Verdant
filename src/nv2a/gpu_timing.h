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
// Stage B, the pixels of a draw: the passes a pixel takes through its pipeline (two texture units a pass; two
// combiner stages a pass [estimate]), and the clocks for `pixels` of them at four pipelines.
inline constexpr u64 kTexturesPerPass = 2, kCombinerStagesPerPass = 2;
u64 pixel_passes(const pgraph::KelvinState& state);
// Stage C, memory: 6.4 GB/s (128-bit DDR at 200 MHz) at 233.33 MHz is 192/7 bytes a GPU clock, shared by everything.
// A pixel's traffic: its colour written (and read to blend), its depth read to test and written, and a texel per
// texture after the texture cache (one byte per texel [estimate]; Z compression not modelled yet). A pixel costs the
// longer of its pipeline time and its memory time.
inline constexpr u64 kBusBytesNum = 192, kBusBytesDen = 7;  // bytes per GPU clock
inline constexpr u64 kTexelBytes = 1;                        // [estimate]
u64 pixel_bytes(const pgraph::KelvinState& state);
// Calibration (measured, not the NV2A itself but its sibling): the GeForce 3 family (same 4x2 pipelines, 128-bit DDR;
// the Ti 200 has the Xbox's 6.4 GB/s) reached 0.83-0.89 of min(pipeline, memory) limits in 3DMark 2001's fill tests
// (AnandTech's GeForce3 Titanium review, Ti 200 1165.6 Mtexels/s multitextured; users' GeForce 3 and overclocked
// Ti 200 runs 715/1430 and 755/1556 Mtexels/s): a pixel takes 20/17 of its limit (0.85 efficiency).
inline constexpr u64 kFillEfficiencyNum = 20, kFillEfficiencyDen = 17;
inline u64 pixel_clocks(u64 pixels, u64 passes, u64 bytes = 0) {
    const u64 pipeline = (pixels * passes + kPixelsPerClock - 1) / kPixelsPerClock;
    const u64 memory = (pixels * bytes * kBusBytesDen + kBusBytesNum - 1) / kBusBytesNum;
    const u64 limit = pipeline > memory ? pipeline : memory;
    return (limit * kFillEfficiencyNum + kFillEfficiencyDen - 1) / kFillEfficiencyDen;
}

// GPU clocks and CPU cycles: exactly 7 GPU clocks per 22 CPU cycles (233.33 and 733.33 MHz). In 64 bits (a 128-bit
// division per method was 1.5 % of a title's CPU thread): exact while a count stays below 2^59 (25 years of either clock).
inline constexpr u64 kGpuToCpuNum = 22, kGpuToCpuDen = 7;
static_assert(clk::kCpuHzNum * kGpuToCpuDen == clk::kGpuHzNum * kGpuToCpuNum, "22 CPU cycles per 7 GPU clocks");
inline Cycles cycles_from_gpu(u64 gpu_clocks) { return (gpu_clocks * kGpuToCpuNum + kGpuToCpuDen - 1) / kGpuToCpuDen; }
inline u64 gpu_from_cycles(Cycles cycles) { return cycles / kGpuToCpuNum * kGpuToCpuDen + cycles % kGpuToCpuNum * kGpuToCpuDen / kGpuToCpuNum; }

}  // namespace mv::gpu_timing
