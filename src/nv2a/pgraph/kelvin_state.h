#pragma once
#include <array>
#include <cstring>
#include <functional>
#include <vector>

#include "core/types.h"

// The state of the Kelvin 3D class (0x97) as its methods leave it, and the decoders of its packed fields.
// Method numbers and fields are documented in local-notes/hw/nv2a-pgraph.md (public nxdk header, envytools,
// behaviour read from xemu); nothing here is copied code.
namespace mv::pgraph {

namespace mthd {
// Kelvin (0x97) methods the state machine acts on; every other method is stored by number.
constexpr u32 kNoOperation = 0x0100, kWaitForIdle = 0x0110, kNotify = 0x0104;
constexpr u32 kSetFlipRead = 0x0120, kSetFlipWrite = 0x0124, kSetFlipModulo = 0x0128, kFlipIncrementWrite = 0x012C, kFlipStall = 0x0130;
constexpr u32 kDmaNotifies = 0x0180, kDmaA = 0x0184, kDmaB = 0x0188, kDmaState = 0x0190, kDmaColor = 0x0194, kDmaZeta = 0x0198,
              kDmaVertexA = 0x019C, kDmaVertexB = 0x01A0, kDmaSemaphore = 0x01A4, kDmaReport = 0x01A8;
constexpr u32 kSurfaceClipH = 0x0200, kSurfaceClipV = 0x0204, kSurfaceFormat = 0x0208, kSurfacePitch = 0x020C, kSurfaceColorOffset = 0x0210,
              kSurfaceZetaOffset = 0x0214;
constexpr u32 kCombinerAlphaIcw = 0x0260, kCombinerSpecFogCw0 = 0x0288, kCombinerSpecFogCw1 = 0x028C, kControl0 = 0x0290;
constexpr u32 kAlphaTestEnable = 0x0300, kBlendEnable = 0x0304, kCullFaceEnable = 0x0308, kDepthTestEnable = 0x030C, kDitherEnable = 0x0310,
              kLightingEnable = 0x0314, kColorMaterial = 0x0298, kMaterialEmission = 0x03A8, kMaterialAlpha = 0x03B4,
              kLightEnableMask = 0x03BC, kSceneAmbientColor = 0x0A10, kStencilTestEnable = 0x032C, kAlphaFunc = 0x033C, kAlphaRef = 0x0340, kBlendSfactor = 0x0344,
              kBlendDfactor = 0x0348, kBlendColor = 0x034C, kBlendEquation = 0x0350, kDepthFunc = 0x0354, kColorMask = 0x0358,
              kDepthMask = 0x035C, kStencilMask = 0x0360, kStencilFunc = 0x0364, kStencilRef = 0x0368, kStencilFuncMask = 0x036C,
              kStencilOpFail = 0x0370, kStencilOpZfail = 0x0374, kStencilOpZpass = 0x0378, kShadeModel = 0x037C, kClipMin = 0x0394,
              kClipMax = 0x0398, kCullFace = 0x039C, kFrontFace = 0x03A0;
constexpr u32 kViewportOffset = 0x0A20, kCombinerFactor0 = 0x0A60, kCombinerFactor1 = 0x0A80, kCombinerAlphaOcw = 0x0AA0,
              kCombinerColorIcw = 0x0AC0, kViewportScale = 0x0AF0;
constexpr u32 kTransformProgram = 0x0B00, kTransformConstant = 0x0B80;  // 32 words each
constexpr u32 kVertexDataArrayOffset = 0x1720, kVertexDataArrayFormat = 0x1760;  // 16 each
constexpr u32 kBeginEnd = 0x17FC, kArrayElement16 = 0x1800, kArrayElement32 = 0x1808, kDrawArrays = 0x1810, kInlineArray = 0x1818;
constexpr u32 kVertexData2fM = 0x1880, kVertexData2s = 0x1900, kVertexData4ub = 0x1940, kVertexData4sM = 0x1980, kVertexData4fM = 0x1A00;
constexpr u32 kVertex3f = 0x1500, kVertex4f = 0x1518, kNormal3f = 0x1530, kNormal3s = 0x1540, kDiffuse4f = 0x1550, kDiffuse3f = 0x1560,
              kDiffuse4ub = 0x156C, kSpecular4f = 0x1570, kSpecular3f = 0x1580, kSpecular4ub = 0x158C, kTexcoord0_2f = 0x1590,
              kTexcoord0_2s = 0x1598, kTexcoord0_4f = 0x15A0, kTexcoord0_4s = 0x15B0, kTexcoord1_2f = 0x15B8, kTexcoord1_2s = 0x15C0,
              kTexcoord1_4f = 0x15C8, kTexcoord1_4s = 0x15D8, kTexcoord2_2f = 0x15E0, kTexcoord2_2s = 0x15E8, kTexcoord2_4f = 0x15F0,
              kTexcoord2_4s = 0x1600, kTexcoord3_2f = 0x1608, kTexcoord3_2s = 0x1610, kTexcoord3_4f = 0x1620, kTexcoord3_4s = 0x1630;
constexpr u32 kFogCoord = 0x1698, kWeight1f = 0x169C, kWeight2f = 0x16A0, kWeight3f = 0x16B0, kWeight4f = 0x16C0;
constexpr u32 kTextureBase = 0x1B00, kTextureStride = 0x40;  // offset, format, address, control0, control1, filter, rect, palette, border
constexpr u32 kSemaphoreOffset = 0x1D6C, kSemaphoreRelease = 0x1D70, kZstencilClear = 0x1D8C, kColorClear = 0x1D90, kClearSurface = 0x1D94,
              kClearRectH = 0x1D98, kClearRectV = 0x1D9C;
constexpr u32 kCombinerColorOcw = 0x1E40, kCombinerControl = 0x1E60, kShaderStageProgram = 0x1E70, kDotRgbMapping = 0x1E74,
              kShaderOtherStageInput = 0x1E78, kTransformExecutionMode = 0x1E94, kTransformProgramCxtWriteEn = 0x1E98,
              kTransformProgramLoad = 0x1E9C, kTransformProgramStart = 0x1EA0, kTransformConstantLoad = 0x1EA4;
}  // namespace mthd

enum class Primitive : u8 {
    End = 0, Points = 1, Lines = 2, LineLoop = 3, LineStrip = 4, Triangles = 5, TriangleStrip = 6, TriangleFan = 7, Quads = 8, QuadStrip = 9,
    Polygon = 10
};

// Vertex attribute array types (SET_VERTEX_DATA_ARRAY_FORMAT bits 0-3).
enum class AttribType : u8 { UbD3d = 0, S1 = 1, Float = 2, UbOgl = 4, S32k = 5, Cmp = 6 };

constexpr u32 kAttributes = 16;
constexpr u32 kProgramSlots = 136;
constexpr u32 kConstants = 192;

struct VertexAttribute {
    AttribType type{AttribType::Float};
    u32 size{0};  // components; 0 = disabled
    u32 stride{0};
    u32 offset{0};  // within the vertex DMA object
    bool dma_b{false};
    // Bytes one element occupies in memory (size * component size).
    u32 bytes() const {
        switch (type) {
            case AttribType::UbD3d:
            case AttribType::UbOgl: return size;
            case AttribType::S1:
            case AttribType::S32k: return size * 2;
            case AttribType::Cmp: return 4;
            case AttribType::Float: return size * 4;
        }
        return size * 4;
    }
};

struct TextureStage {
    u32 offset{0}, format{0}, address{0}, control0{0}, control1{0}, filter{0}, image_rect{0}, palette{0}, border_color{0};
    bool enabled() const { return (control0 >> 30) & 1u; }
    u32 dma_select() const { return format & 3u; }
    bool cubemap() const { return (format >> 2) & 1u; }
    bool border_color_source() const { return (format >> 3) & 1u; }
    u32 dimensionality() const { return (format >> 4) & 0xFu; }
    u32 color_format() const { return (format >> 8) & 0xFFu; }
    u32 mip_levels() const { return (format >> 16) & 0xFu; }
    u32 log_width() const { return (format >> 20) & 0xFu; }
    u32 log_height() const { return (format >> 24) & 0xFu; }
    u32 log_depth() const { return (format >> 28) & 0xFu; }
    u32 pitch() const { return control1 >> 16; }
    u32 rect_width() const { return image_rect >> 16; }
    u32 rect_height() const { return image_rect & 0xFFFFu; }
    u32 wrap_u() const { return address & 0xFu; }
    u32 wrap_v() const { return (address >> 8) & 0xFu; }
    u32 wrap_p() const { return (address >> 16) & 0xFu; }
    u32 min_filter() const { return (filter >> 16) & 0x3Fu; }
    u32 mag_filter() const { return (filter >> 24) & 0xFu; }
    u32 min_lod() const { return (control0 >> 18) & 0xFFFu; }  // 4.8 fixed point
    u32 max_lod() const { return (control0 >> 6) & 0xFFFu; }
};

struct SurfaceState {
    u32 clip_h{0}, clip_v{0}, format{0}, pitch{0}, color_offset{0}, zeta_offset{0};
    u32 clip_x() const { return clip_h & 0xFFFFu; }
    u32 clip_width() const { return clip_h >> 16; }
    u32 clip_y() const { return clip_v & 0xFFFFu; }
    u32 clip_height() const { return clip_v >> 16; }
    u32 color_format() const { return format & 0xFu; }
    u32 zeta_format() const { return (format >> 4) & 0xFu; }
    bool swizzled() const { return ((format >> 8) & 0xFu) == 2; }
    u32 anti_aliasing() const { return (format >> 12) & 0xFu; }
    u32 log_width() const { return (format >> 16) & 0xFFu; }
    u32 log_height() const { return (format >> 24) & 0xFFu; }
    u32 color_pitch() const { return pitch & 0xFFFFu; }
    u32 zeta_pitch() const { return pitch >> 16; }
    // Bytes per pixel of the colour surface (0 = unknown format).
    u32 color_bytes() const {
        switch (color_format()) {
            case 1: case 2: case 3: return 2;
            case 4: case 5: case 6: case 7: case 8: return 4;
            case 9: return 1;
            case 10: return 2;
            default: return 0;
        }
    }
    u32 zeta_bytes() const { return zeta_format() == 1 ? 2 : (zeta_format() == 2 ? 4 : 0); }
};

// The whole method-addressable state: the last value written to every method (indexed by method / 4), plus the
// memories that streaming methods fill (the vertex program and its constants), decoded views on top.
struct KelvinState {
    std::array<u32, 0x800> r{};
    std::array<std::array<u32, 4>, kProgramSlots> program{};
    std::array<std::array<u32, 4>, kConstants> constants{};
    u32 program_load{0};   // instruction slot the next SET_TRANSFORM_PROGRAM word goes to (word index within the slot in program_word)
    u32 program_word{0};
    u32 constant_load{0};  // constant slot the next SET_TRANSFORM_CONSTANT word goes to
    u32 constant_word{0};
    std::array<VertexAttribute, kAttributes> attributes{};
    // The current value of each attribute for immediate-mode vertices (SET_VERTEX_DATA*), as floats.
    std::array<std::array<float, 4>, kAttributes> immediate{};

    u32 reg(u32 method) const { return r[(method >> 2) & 0x7FFu]; }
    float regf(u32 method) const {
        float f;
        const u32 v = reg(method);
        std::memcpy(&f, &v, 4);
        return f;
    }
    u32 dma_handle(u32 which) const { return reg(which); }
    SurfaceState surface() const {
        return {reg(mthd::kSurfaceClipH), reg(mthd::kSurfaceClipV), reg(mthd::kSurfaceFormat), reg(mthd::kSurfacePitch),
                reg(mthd::kSurfaceColorOffset), reg(mthd::kSurfaceZetaOffset)};
    }
    TextureStage texture(u32 stage) const {
        const u32 b = mthd::kTextureBase + stage * mthd::kTextureStride;
        return {reg(b), reg(b + 4), reg(b + 8), reg(b + 0xC), reg(b + 0x10), reg(b + 0x14), reg(b + 0x1C), reg(b + 0x20), reg(b + 0x24)};
    }
    u32 combiner_stages() const { return reg(mthd::kCombinerControl) & 0xFFu; }
    bool combiner_mux_msb() const { return (reg(mthd::kCombinerControl) >> 8) & 1u; }
    bool combiner_factor0_per_stage() const { return (reg(mthd::kCombinerControl) >> 12) & 1u; }
    bool combiner_factor1_per_stage() const { return (reg(mthd::kCombinerControl) >> 16) & 1u; }
    u32 shader_stage_program(u32 stage) const { return (reg(mthd::kShaderStageProgram) >> (5 * stage)) & 0x1Fu; }
    u32 transform_mode() const { return reg(mthd::kTransformExecutionMode) & 3u; }  // 0 fixed function, 2 program
    bool vertex_program_mode() const { return transform_mode() == 2; }
    u32 program_start() const { return reg(mthd::kTransformProgramStart) & 0xFFu; }
    bool alpha_test() const { return reg(mthd::kAlphaTestEnable) & 1u; }
    bool blend() const { return reg(mthd::kBlendEnable) & 1u; }
    bool cull_face() const { return reg(mthd::kCullFaceEnable) & 1u; }
    bool depth_test() const { return reg(mthd::kDepthTestEnable) & 1u; }
    bool stencil_test() const { return reg(mthd::kStencilTestEnable) & 1u; }
    bool depth_write() const { return reg(mthd::kDepthMask) & 1u; }
    u32 color_mask() const { return reg(mthd::kColorMask); }
    u32 clear_color() const { return reg(mthd::kColorClear); }
    u32 clear_zstencil() const { return reg(mthd::kZstencilClear); }
};

// A batch of geometry between SET_BEGIN_END(primitive) and SET_BEGIN_END(END), in whichever form the stream used.
struct DrawBatch {
    Primitive primitive{Primitive::End};
    std::vector<u32> indices;                  // ARRAY_ELEMENT16/32: indices into the vertex arrays
    std::vector<std::pair<u32, u32>> ranges;   // DRAW_ARRAYS: (first index, count)
    std::vector<u32> inline_words;             // INLINE_ARRAY: interleaved attribute data
    std::vector<std::array<float, 4>> immediate_vertices;  // SET_VERTEX_DATA*: 16 attribute values per vertex, vertex-major
    u32 immediate_count{0};
    u16 immediate_mask{0};  // which attributes the immediate vertices carry
    bool empty() const { return indices.empty() && ranges.empty() && inline_words.empty() && immediate_count == 0; }
};

}  // namespace mv::pgraph
