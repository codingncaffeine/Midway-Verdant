#pragma once
#include <array>

#include "nv2a/shaders/vsh_translate.h"

// A CPU interpreter for the Kelvin vertex program: the same semantics the GLSL translator emits, run on the host
// for one vertex. An instrument (what does the program do with this vertex?) and a test oracle for the translator.
namespace mv::pgraph::vsh {

struct Vec4 {
    float v[4]{0.0f, 0.0f, 0.0f, 1.0f};
    float& operator[](std::size_t i) { return v[i]; }
    float operator[](std::size_t i) const { return v[i]; }
};

struct Outputs {
    Vec4 pos, d0, d1, b0, b1, fog, pts, t0, t1, t2, t3;
    // oPos as the program left it (screen space), before any host-side conversion.
};

// Runs `program` with the 16 input attributes and the 192 constants (as raw u32 words, like the state keeps them).
Outputs run(const Program& program, const std::array<Vec4, kAttributes>& inputs, const std::array<std::array<u32, 4>, kConstants>& constants);

}  // namespace mv::pgraph::vsh
