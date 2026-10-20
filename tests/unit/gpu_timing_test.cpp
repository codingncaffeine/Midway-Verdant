// The NV2A cost model of the timed pusher (gpu_timing.h, PLAN D28): what a draw transforms and sets up, the program
// length the vertex units run, and the exact clock ratio between the GPU and the CPU.
#include <gtest/gtest.h>

#include "nv2a/gpu_timing.h"

using namespace mv;
using namespace mv::pgraph;

TEST(GpuTiming, PrimitivesFromVertices) {
    EXPECT_EQ(gpu_timing::primitive_count(Primitive::Triangles, 300), 100u);
    EXPECT_EQ(gpu_timing::primitive_count(Primitive::Triangles, 301), 100u);
    EXPECT_EQ(gpu_timing::primitive_count(Primitive::TriangleStrip, 10), 8u);
    EXPECT_EQ(gpu_timing::primitive_count(Primitive::TriangleFan, 2), 0u);
    EXPECT_EQ(gpu_timing::primitive_count(Primitive::Quads, 8), 4u);
    EXPECT_EQ(gpu_timing::primitive_count(Primitive::QuadStrip, 6), 4u);
    EXPECT_EQ(gpu_timing::primitive_count(Primitive::LineStrip, 5), 4u);
    EXPECT_EQ(gpu_timing::primitive_count(Primitive::Points, 7), 7u);
}

TEST(GpuTiming, VerticesFromEveryBatchForm) {
    KelvinState s;
    DrawBatch b;
    b.primitive = Primitive::Triangles;
    b.indices = {0, 1, 2, 3, 4, 5};
    b.ranges = {{10, 9}, {40, 3}};
    b.immediate_count = 4;
    // Inline vertices: a float3 position and a 4-byte colour = 4 words each; 8 words = 2 vertices.
    s.attributes[0].type = AttribType::Float;
    s.attributes[0].size = 3;
    s.attributes[3].type = AttribType::UbD3d;
    s.attributes[3].size = 4;
    b.inline_words.assign(8, 0);
    EXPECT_EQ(gpu_timing::vertex_count(s, b), 6u + 12u + 4u + 2u);
}

TEST(GpuTiming, ProgramLengthEndsAtTheFinalInstruction) {
    KelvinState s;
    EXPECT_EQ(gpu_timing::program_length(s), gpu_timing::kFixedFunctionInsns) << "fixed function";
    s.r[(mthd::kTransformExecutionMode >> 2) & 0x7FF] = 2;  // program mode
    s.r[(mthd::kTransformProgramStart >> 2) & 0x7FF] = 10;
    s.program[10 + 5][3] = 1;  // the sixth instruction from the start is the last
    EXPECT_EQ(gpu_timing::program_length(s), 6u);
    // 300 triangle vertices: max(2, ceil(6 / 2)) = 3 clocks each, 100 triangles at the setup cost.
    DrawBatch b;
    b.primitive = Primitive::Triangles;
    b.ranges = {{0, 300}};
    EXPECT_EQ(gpu_timing::draw_clocks(s, b), 300u * 3 + 100u * gpu_timing::kTriangleSetupClocks);
}

TEST(GpuTiming, SevenGpuClocksAreTwentyTwoCpuCycles) {
    EXPECT_EQ(gpu_timing::cycles_from_gpu(7), 22u);
    EXPECT_EQ(gpu_timing::cycles_from_gpu(7'000'000), 22'000'000u);
    EXPECT_EQ(gpu_timing::cycles_from_gpu(1), 4u) << "rounded up: a method is not seen before it runs";
    EXPECT_EQ(gpu_timing::gpu_from_cycles(22), 7u);
    EXPECT_EQ(gpu_timing::gpu_from_cycles(21), 6u) << "rounded down: an idle GPU never starts in the future";
}
