#include "nv2a/gpu_timing.h"

#include <algorithm>

namespace mv::gpu_timing {

static_assert(clk::kCpuHzDen == clk::kGpuHzDen, "the clock ratio is kCpuHzNum / kGpuHzNum");

u64 vertex_count(const pgraph::KelvinState& state, const pgraph::DrawBatch& batch) {
    u64 n = batch.indices.size() + batch.immediate_count;
    for (const auto& r : batch.ranges) n += r.second;
    if (!batch.inline_words.empty()) {
        // An inline vertex is every enabled attribute, each padded to whole words.
        u64 words = 0;
        for (const auto& a : state.attributes)
            if (a.size) words += (a.bytes() + 3) / 4;
        if (words) n += batch.inline_words.size() / words;
    }
    return n;
}

u64 primitive_count(pgraph::Primitive primitive, u64 v) {
    using pgraph::Primitive;
    switch (primitive) {
        case Primitive::Points: return v;
        case Primitive::Lines: return v / 2;
        case Primitive::LineLoop: return v >= 2 ? v : 0;
        case Primitive::LineStrip: return v >= 2 ? v - 1 : 0;
        case Primitive::Triangles: return v / 3;
        case Primitive::TriangleStrip:
        case Primitive::TriangleFan:
        case Primitive::Polygon: return v >= 3 ? v - 2 : 0;
        case Primitive::Quads: return v / 4 * 2;  // two triangles a quad
        case Primitive::QuadStrip: return v >= 4 ? (v - 2) / 2 * 2 : 0;
        default: return 0;
    }
}

u64 program_length(const pgraph::KelvinState& state) {
    if (!state.vertex_program_mode()) return kFixedFunctionInsns;
    u64 n = 0;
    for (u32 slot = state.program_start(); slot < pgraph::kProgramSlots && n < kMaxProgramInsns; ++slot) {
        ++n;
        if (state.program[slot][3] & 1u) break;  // the final instruction
    }
    return std::max<u64>(n, 1);
}

u64 draw_clocks(const pgraph::KelvinState& state, const pgraph::DrawBatch& batch) {
    const u64 vertices = vertex_count(state, batch);
    const u64 per_vertex = std::max(kVertexMinClocks, (program_length(state) + kVertexUnits - 1) / kVertexUnits);
    return vertices * per_vertex + primitive_count(batch.primitive, vertices) * kTriangleSetupClocks;
}

u64 pixel_passes(const pgraph::KelvinState& state) {
    u64 textures = 0;
    for (u32 stage = 0; stage < 4; ++stage)
        if (state.shader_stage_program(stage) != 0) ++textures;
    const u64 stages = state.combiner_stages();
    return std::max<u64>({1, (textures + kTexturesPerPass - 1) / kTexturesPerPass, (stages + kCombinerStagesPerPass - 1) / kCombinerStagesPerPass});
}

u64 clear_clocks(const pgraph::KelvinState& state) {
    const auto s = state.surface();
    const u64 pixels = static_cast<u64>(s.clip_width()) * s.clip_height();
    return (pixels + kPixelsPerClock - 1) / kPixelsPerClock;
}

}  // namespace mv::gpu_timing
