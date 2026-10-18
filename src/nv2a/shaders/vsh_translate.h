#pragma once
#include <array>
#include <string>
#include <vector>

#include "core/types.h"
#include "nv2a/pgraph/kelvin_state.h"

// The Kelvin vertex program: a 128-bit instruction word per slot (four little-endian u32), a MAC operation and
// an ILU operation that may be paired, three operands with swizzle and negate, writes to a temporary, an output
// register or (vertex state shaders) a constant. Field layout per local-notes/hw/nv2a-pgraph.md (two public
// decoders agree on it); semantics per the public NV_vertex_program1_1 specification with the Kelvin additions.
namespace mv::pgraph::vsh {

enum class Mac : u8 { Nop, Mov, Mul, Add, Mad, Dp3, Dph, Dp4, Dst, Min, Max, Slt, Sge, Arl };
enum class Ilu : u8 { Nop, Mov, Rcp, Rcc, Rsq, Exp, Log, Lit };
enum class Source : u8 { None = 0, Temp = 1, Input = 2, Constant = 3 };

struct Operand {
    Source source{Source::None};
    u32 index{0};  // temp register, input attribute or constant slot
    std::array<u8, 4> swizzle{0, 1, 2, 3};
    bool negate{false};
    bool relative{false};  // constant indexed by A0
};

struct Instruction {
    Mac mac{Mac::Nop};
    Ilu ilu{Ilu::Nop};
    Operand a, b, c;
    u32 mac_mask{0};         // xyzw bits (x = 8)
    u32 ilu_mask{0};
    u32 temp{0};             // destination temporary for the MAC (and the ILU when not paired)
    u32 output_mask{0};
    u32 output_index{0};     // output register (or constant when output_is_constant)
    bool output_is_constant{false};
    bool output_from_ilu{false};
    bool final{false};
};

Instruction decode(const std::array<u32, 4>& token);
std::string disassemble(const Instruction& i);
std::string mask_string(u32 mask);  // "xyz" style; "" when empty

// The program as the hardware would run it: slots from the start address up to and including the final one.
struct Program {
    std::vector<Instruction> instructions;
    u32 start{0};
    bool truncated{false};  // no final instruction before the end of program memory
};
Program extract_program(const KelvinState& state);


struct Options {
    bool smooth_shading{true};
    u32 texture_count{4};
    // SET_FOG_ENABLE and SET_FOG_MODE: with fog on, the shader ends by turning the fog distance in oFog.x into
    // the factor the pixel side blends with.
    bool fog{false};
    u32 fog_mode{0x2601};
};

// GLSL (Vulkan 1.0, version 450) for the program: inputs v0..v15 at locations 0..15, one std140 uniform block
// (set 0, binding 0: vec4 c[192]; vec4 surface_size; vec4 clip_range), outputs at the locations the fragment side
// expects (see fsh_interface.h).
struct Translation {
    std::string glsl;
    std::string error;
    u16 inputs_read{0};  // attribute bits the program reads
    std::vector<std::string> disassembly;
};
Translation translate(const Program& program, const Options& options);

// The uniform block layout shared with the backend: the constant file, then surface_size, clip_range and the
// fixed function's block (fixed_function_uniforms).
constexpr u32 kConstantSlots = 192, kLights = 8, kLightVectors = 9;
constexpr u32 kFixedFunctionVectors = 4 + kLights * kLightVectors;
constexpr u32 kUniformBytes = (kConstantSlots + 2 + kFixedFunctionVectors) * 16;

// The fixed-function vertex pipeline (execution mode 0) as GLSL with the same interface as a translated program:
// skinning through the model-view matrices (XF slots 8+8i, inverses 12+8i), the normal into eye space, the lights
// of SET_LIGHT_ENABLE_MASK (infinite, local, spot) with the colour sources of SET_COLOR_MATERIAL, specular per
// SET_LIGHT_CONTROL and SET_SPECULAR_ENABLE, texgen per coordinate and the texture matrices (XF 0x44+8i), the fog
// coordinate per SET_FOG_GEN_MODE, and the position through the composite matrix (XF 0-3, viewport scale
// included) then the viewport offset (XF 0x3B), as the XDK's own program epilogue leaves it. The light data are
// uniforms (fixed_function_uniforms). Behaviour per the method definitions and the reference emulator's
// generator (read, not copied); [verify] marks what neither settles.
Translation fixed_function_translate(const KelvinState& state, const Options& options);
// The state words the generated shader depends on (for caching): two states with equal keys get the same GLSL.
std::vector<u32> fixed_function_key(const KelvinState& state);
// A state the generator cannot express (an invalid enumerant), with the reason.
bool fixed_function_supported(const KelvinState& state, std::string& why);
// The fixed function's uniforms after the constant file: material emission + alpha, scene ambient, specular
// curve, fog params (SET_FOG_PARAMS, for programs too), then nine per light: ambient, diffuse, specular, local position (w = range), attenuation,
// infinite direction, infinite half vector, spot falloff, spot direction. Writes kUniformBytes - 16 * kConstantSlots
// - 32 bytes (everything after surface_size and clip_range).
void fixed_function_uniforms(const KelvinState& state, float* out);

}  // namespace mv::pgraph::vsh
