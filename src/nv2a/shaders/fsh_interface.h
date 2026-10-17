#pragma once
// The varyings between the vertex side (vsh_translate) and the fragment side (psh_translate), by location.
namespace mv::pgraph {
constexpr unsigned kVaryingD0 = 0, kVaryingD1 = 1, kVaryingB0 = 2, kVaryingB1 = 3, kVaryingT0 = 4, kVaryingFog = 8, kVaryingPos = 9;
}  // namespace mv::pgraph
