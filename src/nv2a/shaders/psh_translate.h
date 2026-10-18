#pragma once
#include <array>
#include <string>

#include "core/types.h"
#include "nv2a/pgraph/kelvin_state.h"

// The Kelvin pixel pipeline as GLSL: the four texture stages (shader stage programs), the register combiners
// (up to eight general stages and the final combiner, per the public NV_register_combiners specification), the
// alpha test. Field layouts per local-notes/hw/nv2a-pgraph.md.
namespace mv::pgraph::psh {

// What a fragment shader depends on, extracted from the state so equal keys share a compiled shader.
struct Key {
    u32 combiner_control{0};
    std::array<u32, 8> color_icw{}, color_ocw{}, alpha_icw{}, alpha_ocw{};
    u32 final_cw0{0}, final_cw1{0};
    u32 shader_stage_program{0};
    u32 other_stage_input{0};
    u32 clip_plane_mode{0};  // SET_SHADER_CLIP_PLANE_MODE: four bits per stage, one per component
    u32 dot_mapping{0};      // SET_DOT_RGBMAPPING: how stages 1-3 read their source's colour for a dot product
    u32 shadow_compare_func{0};
    std::array<u32, 4> texture_format{};  // only the fields that shape the shader (dimensionality, cube, colour)
    std::array<bool, 4> texture_enabled{};
    bool alpha_test{false};
    u32 alpha_func{0x207};
    bool smooth_shading{true};
    bool fog_enable{false};
    bool operator==(const Key&) const = default;
};
Key key_from_state(const KelvinState& state);

// Uniform block (std140, set 0, binding 1), see the layout in the generated shader.
struct Uniforms {
    float c0[9][4];
    float c1[9][4];
    float fog_color[4];
    float alpha_ref[4];   // x = reference 0..1
    float tex_scale[4][4];  // xy = 1 / texture size for linear textures (1 otherwise), z = depth compare scale, w = unused
    float eye[4];           // xyz = SET_EYE_VECTOR, the constant eye of the dot-reflect-specular-constant mode
};
static_assert(sizeof(Uniforms) == 25 * 16);
void uniforms_from_state(const KelvinState& state, const std::array<std::array<float, 2>, 4>& texture_scale, Uniforms& out);

enum class SamplerKind : u8 { None, Sampler2D, SamplerCube, Sampler3D, Shadow2D };

struct Translation {
    std::string glsl;
    std::string error;
    std::array<SamplerKind, 4> samplers{};  // what each texture binding (2 + stage) must receive
};
Translation translate(const Key& key);

constexpr u32 kSamplerBindingBase = 2;

}  // namespace mv::pgraph::psh
