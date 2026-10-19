#include "nv2a/pgraph/kelvin.h"

#include <algorithm>
#include <cstring>
#include <format>

namespace mv::pgraph {

namespace {

float as_float(u32 v) {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

float s16_norm(u32 half) {
    const i16 s = static_cast<i16>(half & 0xFFFFu);
    return std::max(-1.0f, static_cast<float>(s) / 32767.0f);
}

float s16_plain(u32 half) { return static_cast<float>(static_cast<i16>(half & 0xFFFFu)); }

// The fixed-function registers that alias the vertex constant file: method -> (constant slot * 4 + component).
// Slots per the public XF context table: composite matrix 0, projection 4, model-view i at 8 + 8i with its inverse
// at 12 + 8i, lights 0x28.., eye position 0x38, fog 0x39, viewport scale 0x3A, offset 0x3B, texgen planes of
// texture i at 0x40 + 8i, texture matrix i at 0x44 + 8i.
bool xf_context_slot(u32 method, u32& slot_word) {
    auto range = [&](u32 base, u32 words, u32 slot) {
        if (method >= base && method < base + words * 4) {
            slot_word = slot * 4 + ((method - base) >> 2);
            return true;
        }
        return false;
    };
    if (range(0x0680, 16, 0x00)) return true;   // composite matrix
    if (range(0x0440, 16, 0x04)) return true;   // projection matrix
    for (u32 i = 0; i < 4; ++i) {
        if (range(0x0480 + 0x40 * i, 16, 0x08 + 8 * i)) return true;  // model-view i
        if (range(0x0580 + 0x40 * i, 16, 0x0C + 8 * i)) return true;  // inverse model-view i
        if (range(0x0840 + 0x40 * i, 16, 0x40 + 8 * i)) return true;  // texgen planes S T R Q of texture i
        if (range(0x06C0 + 0x40 * i, 16, 0x44 + 8 * i)) return true;  // texture matrix i
    }
    if (range(0x0A50, 4, 0x38)) return true;    // eye position
    if (range(0x09D0, 4, 0x39)) return true;    // fog plane (SET_FOG_PARAMS are registers of their own)
    if (range(0x0AF0, 4, 0x3A)) return true;    // viewport scale
    if (range(0x0A20, 4, 0x3B)) return true;    // viewport offset
    return false;
}

// Immediate-mode attribute numbers (the XDK's vertex declaration order).
constexpr u32 kAttrPosition = 0, kAttrWeight = 1, kAttrNormal = 2, kAttrDiffuse = 3, kAttrSpecular = 4, kAttrFog = 5, kAttrTexcoord0 = 9;

}  // namespace

const char* primitive_name(Primitive p) {
    switch (p) {
        case Primitive::End: return "end";
        case Primitive::Points: return "points";
        case Primitive::Lines: return "lines";
        case Primitive::LineLoop: return "line loop";
        case Primitive::LineStrip: return "line strip";
        case Primitive::Triangles: return "triangles";
        case Primitive::TriangleStrip: return "triangle strip";
        case Primitive::TriangleFan: return "triangle fan";
        case Primitive::Quads: return "quads";
        case Primitive::QuadStrip: return "quad strip";
        case Primitive::Polygon: return "polygon";
    }
    return "?";
}

Kelvin::Kelvin() {
    for (auto& v : state_.immediate) v = {0.0f, 0.0f, 0.0f, 1.0f};
}

void Kelvin::immediate_component(u32 attribute, u32 component, float v, bool completes_vertex) {
    if (attribute >= kAttributes || component >= 4) return;
    state_.immediate[attribute][component] = v;
    batch_.immediate_mask = static_cast<u16>(batch_.immediate_mask | (1u << attribute));
    if (completes_vertex && attribute == kAttrPosition) immediate_vertex_done();
}

void Kelvin::immediate_vertex_done() {
    if (!in_begin_) return;
    for (u32 a = 0; a < kAttributes; ++a) batch_.immediate_vertices.push_back(state_.immediate[a]);
    ++batch_.immediate_count;
}

void Kelvin::begin_end(u32 value) {
    if (value != 0) {
        in_begin_ = true;
        batch_ = DrawBatch{};
        batch_.primitive = static_cast<Primitive>(value & 0xFu);
        return;
    }
    in_begin_ = false;
    if (sink_ && !batch_.empty()) sink_->draw(state_, batch_);
    if (!batch_.empty()) latch_current_values(state_, vertex_handles_, vertex_memory_, batch_);
    batch_ = DrawBatch{};
}

namespace {

// What a method does, decided once per method slot: the checks below run in the order the methods overlap (the
// streaming ranges first, then the data forms, the named immediates, the switch), so the table gives the first match
// for every one of the 2048 slots and the per-word work is one lookup.
enum class Op : u8 {
    Plain, Program, Constant, XfContext, ArrayFormat, ArrayOffset, Data4f, Data2f, Data2s, Data4ub, Data4s, NamedFloat,
    NamedS16, NamedUb4, FogCoord, ProgramLoad, ConstantLoad, BeginEnd, Element16, Element32, DrawArrays, InlineArray,
    ClearSurface, SemaphoreRelease
};
struct Decoded {
    Op op{Op::Plain};
    u8 attr{0};
    u8 a{0};       // the component, word or half the method writes
    u8 b{0};       // the form's component count or word count
    bool norm{false};
    u16 slot{0};   // XF context: constant slot * 4 + component
};

Decoded decode(u32 method) {
    using namespace mthd;
    Decoded d;
    auto in = [&](u32 base, u32 bytes) { return method >= base && method < base + bytes; };
    if (in(kTransformProgram, 0x80)) return {Op::Program};
    if (in(kTransformConstant, 0x80)) return {Op::Constant};
    if (u32 slot_word; xf_context_slot(method, slot_word)) {
        d.op = Op::XfContext;
        d.slot = static_cast<u16>(slot_word);
        return d;
    }
    if (in(kVertexDataArrayFormat, 0x40)) return {Op::ArrayFormat, static_cast<u8>((method - kVertexDataArrayFormat) >> 2)};
    if (in(kVertexDataArrayOffset, 0x40)) return {Op::ArrayOffset, static_cast<u8>((method - kVertexDataArrayOffset) >> 2)};
    if (in(kVertexData4fM, 0x100)) {
        const u32 i = (method - kVertexData4fM) >> 2;
        return {Op::Data4f, static_cast<u8>(i >> 2), static_cast<u8>(i & 3u)};
    }
    if (in(kVertexData2fM, 0x80)) {
        const u32 i = (method - kVertexData2fM) >> 2;
        return {Op::Data2f, static_cast<u8>(i >> 1), static_cast<u8>(i & 1u)};
    }
    if (in(kVertexData2s, 0x40)) return {Op::Data2s, static_cast<u8>((method - kVertexData2s) >> 2)};
    if (in(kVertexData4ub, 0x40)) return {Op::Data4ub, static_cast<u8>((method - kVertexData4ub) >> 2)};
    if (in(kVertexData4sM, 0x80)) {
        const u32 i = (method - kVertexData4sM) >> 2;
        return {Op::Data4s, static_cast<u8>(i >> 1), static_cast<u8>(i & 1u)};
    }
    // The named immediate methods, in the order they were matched.
    auto named_float = [&](u32 base, u32 attr, u32 count) {
        if (!in(base, count * 4)) return false;
        d = {Op::NamedFloat, static_cast<u8>(attr), static_cast<u8>((method - base) >> 2), static_cast<u8>(count)};
        return true;
    };
    auto named_s16x2 = [&](u32 base, u32 attr, u32 words, bool normalised) {
        if (!in(base, words * 4)) return false;
        d = {Op::NamedS16, static_cast<u8>(attr), static_cast<u8>((method - base) >> 2), static_cast<u8>(words), normalised};
        return true;
    };
    auto named_ub4 = [&](u32 base, u32 attr) {
        if (method != base) return false;
        d = {Op::NamedUb4, static_cast<u8>(attr)};
        return true;
    };
    if (named_float(kVertex3f, kAttrPosition, 3) || named_float(kVertex4f, kAttrPosition, 4) || named_float(kNormal3f, kAttrNormal, 3) ||
        named_s16x2(kNormal3s, kAttrNormal, 2, true) || named_float(kDiffuse4f, kAttrDiffuse, 4) || named_float(kDiffuse3f, kAttrDiffuse, 3) ||
        named_ub4(kDiffuse4ub, kAttrDiffuse) || named_float(kSpecular4f, kAttrSpecular, 4) || named_float(kSpecular3f, kAttrSpecular, 3) ||
        named_ub4(kSpecular4ub, kAttrSpecular) || named_float(kWeight1f, kAttrWeight, 1) || named_float(kWeight2f, kAttrWeight, 2) ||
        named_float(kWeight3f, kAttrWeight, 3) || named_float(kWeight4f, kAttrWeight, 4))
        return d;
    if (method == kFogCoord) return {Op::FogCoord};
    static constexpr u32 kTex2f[4] = {kTexcoord0_2f, kTexcoord1_2f, kTexcoord2_2f, kTexcoord3_2f};
    static constexpr u32 kTex2s[4] = {kTexcoord0_2s, kTexcoord1_2s, kTexcoord2_2s, kTexcoord3_2s};
    static constexpr u32 kTex4f[4] = {kTexcoord0_4f, kTexcoord1_4f, kTexcoord2_4f, kTexcoord3_4f};
    static constexpr u32 kTex4s[4] = {kTexcoord0_4s, kTexcoord1_4s, kTexcoord2_4s, kTexcoord3_4s};
    for (u32 t = 0; t < 4; ++t) {
        if (named_float(kTex2f[t], kAttrTexcoord0 + t, 2) || named_s16x2(kTex2s[t], kAttrTexcoord0 + t, 1, false) ||
            named_float(kTex4f[t], kAttrTexcoord0 + t, 4) || named_s16x2(kTex4s[t], kAttrTexcoord0 + t, 2, false))
            return d;
    }
    switch (method) {
        case kTransformProgramLoad: return {Op::ProgramLoad};
        case kTransformConstantLoad: return {Op::ConstantLoad};
        case kBeginEnd: return {Op::BeginEnd};
        case kArrayElement16: return {Op::Element16};
        case kArrayElement32: return {Op::Element32};
        case kDrawArrays: return {Op::DrawArrays};
        case kInlineArray: return {Op::InlineArray};
        case kClearSurface: return {Op::ClearSurface};
        case kSemaphoreRelease: return {Op::SemaphoreRelease};
        default: return {Op::Plain};
    }
}

const std::array<Decoded, 0x800>& decoded_methods() {
    static const std::array<Decoded, 0x800> table = [] {
        std::array<Decoded, 0x800> t{};
        for (u32 i = 0; i < t.size(); ++i) t[i] = decode(i << 2);
        return t;
    }();
    return table;
}

}  // namespace

void Kelvin::method(u32 method, u32 value) {
    method &= 0x1FFCu;
    ++methods_;
    ++histogram_[method >> 2];
    state_.r[method >> 2] = value;
    const Decoded& d = decoded_methods()[method >> 2];
    switch (d.op) {
        case Op::Plain: return;
        case Op::Program:
            if (state_.program_load < kProgramSlots) state_.program[state_.program_load][state_.program_word] = value;
            if (++state_.program_word == 4) {
                state_.program_word = 0;
                ++state_.program_load;
            }
            return;
        case Op::Constant:
            if (state_.constant_load < kConstants) state_.constants[state_.constant_load][state_.constant_word] = value;
            if (++state_.constant_word == 4) {
                state_.constant_word = 0;
                ++state_.constant_load;
            }
            return;
        // The fixed-function transform registers live in the constant file too (the "XF context"): a program can
        // read the viewport scale and offset as c[58] and c[59], the matrices from slot 0 on, and the XDK's programs do.
        case Op::XfContext: state_.constants[d.slot >> 2][d.slot & 3u] = value; return;
        case Op::ArrayFormat: {
            VertexAttribute& a = state_.attributes[d.attr];
            a.type = static_cast<AttribType>(value & 0xFu);
            a.size = (value >> 4) & 0xFu;
            a.stride = value >> 8;
            return;
        }
        case Op::ArrayOffset: {
            VertexAttribute& a = state_.attributes[d.attr];
            a.offset = value & 0x7FFFFFFFu;
            a.dma_b = (value >> 31) & 1u;
            return;
        }
        case Op::Data4f: immediate_component(d.attr, d.a, as_float(value), d.a == 3); return;
        case Op::Data2f:
            if (d.a == 0) {
                state_.immediate[d.attr][2] = 0.0f;
                state_.immediate[d.attr][3] = 1.0f;
            }
            immediate_component(d.attr, d.a, as_float(value), d.a == 1);
            return;
        case Op::Data2s:
            state_.immediate[d.attr][2] = 0.0f;
            state_.immediate[d.attr][3] = 1.0f;
            immediate_component(d.attr, 0, s16_plain(value), false);
            immediate_component(d.attr, 1, s16_plain(value >> 16), true);
            return;
        case Op::Data4ub:
        case Op::NamedUb4:
            for (u32 c = 0; c < 4; ++c) immediate_component(d.attr, c, static_cast<float>((value >> (8 * c)) & 0xFFu) / 255.0f, c == 3);
            return;
        case Op::Data4s:
            immediate_component(d.attr, d.a * 2u, s16_plain(value), false);
            immediate_component(d.attr, d.a * 2u + 1, s16_plain(value >> 16), d.a == 1);
            return;
        // A short float form sets the components it leaves out as well, to those of (x, 0, 0, 1): a 2F texture
        // coordinate after a 4F one has z = 0 and w = 1 again (a projective lookup divides by that w), a 3F colour
        // has alpha 1, a 1F weight is (w, 0, 0, 1).
        case Op::NamedFloat:
            for (u32 c = d.b; c < 4; ++c) state_.immediate[d.attr][c] = c == 3 ? 1.0f : 0.0f;
            immediate_component(d.attr, d.a, as_float(value), d.a == d.b - 1u);
            return;
        case Op::NamedS16: {
            if (d.a == 0 && d.b == 1) {
                state_.immediate[d.attr][2] = 0.0f;
                state_.immediate[d.attr][3] = 1.0f;
            }
            const float lo = d.norm ? s16_norm(value) : s16_plain(value);
            const float hi = d.norm ? s16_norm(value >> 16) : s16_plain(value >> 16);
            immediate_component(d.attr, d.a * 2u, lo, false);
            immediate_component(d.attr, d.a * 2u + 1, hi, d.a == d.b - 1u);
            return;
        }
        case Op::FogCoord:
            // The fog coordinate fills all four components of its attribute.
            for (u32 c = 0; c < 4; ++c) immediate_component(kAttrFog, c, as_float(value), false);
            return;
        case Op::ProgramLoad:
            state_.program_load = value & 0xFFu;
            state_.program_word = 0;
            return;
        case Op::ConstantLoad:
            state_.constant_load = value & 0x1FFu;
            state_.constant_word = 0;
            return;
        case Op::BeginEnd: begin_end(value); return;
        case Op::Element16:
            batch_.indices.push_back(value & 0xFFFFu);
            batch_.indices.push_back(value >> 16);
            return;
        case Op::Element32: batch_.indices.push_back(value); return;
        case Op::DrawArrays: batch_.ranges.emplace_back(value & 0xFFFFFFu, (value >> 24) + 1); return;
        case Op::InlineArray: batch_.inline_words.push_back(value); return;
        case Op::ClearSurface:
            if (sink_) sink_->clear(state_, value);
            return;
        case Op::SemaphoreRelease:
            if (sink_) sink_->semaphore_release(state_, value);
            return;
    }
}

void decode_attribute(const VertexAttribute& a, const u8* p, float out[4]) {
    out[0] = 0.0f;
    out[1] = 0.0f;
    out[2] = 0.0f;
    out[3] = 1.0f;
    const u32 n = std::min<u32>(a.size, 4);
    switch (a.type) {
        case AttribType::Float:
            std::memcpy(out, p, n * 4);
            break;
        case AttribType::UbOgl:
            for (u32 i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]) / 255.0f;
            break;
        case AttribType::UbD3d: {
            // D3D colour bytes in memory are B G R A; the attribute reads as R G B A.
            float tmp[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            for (u32 i = 0; i < n; ++i) tmp[i] = static_cast<float>(p[i]) / 255.0f;
            out[0] = tmp[2];
            out[1] = tmp[1];
            out[2] = tmp[0];
            out[3] = n == 4 ? tmp[3] : 1.0f;
            break;
        }
        case AttribType::S1:
            for (u32 i = 0; i < n; ++i) {
                u16 h;
                std::memcpy(&h, p + 2 * i, 2);
                out[i] = s16_norm(h);
            }
            break;
        case AttribType::S32k:
            for (u32 i = 0; i < n; ++i) {
                u16 h;
                std::memcpy(&h, p + 2 * i, 2);
                out[i] = s16_plain(h);
            }
            break;
        case AttribType::Cmp: {
            u32 v;
            std::memcpy(&v, p, 4);
            i32 x = static_cast<i32>(v & 0x7FFu), y = static_cast<i32>((v >> 11) & 0x7FFu), z = static_cast<i32>((v >> 22) & 0x3FFu);
            if (x & 0x400) x -= 0x800;
            if (y & 0x400) y -= 0x800;
            if (z & 0x200) z -= 0x400;
            out[0] = std::max(-1.0f, static_cast<float>(x) / 1023.0f);
            out[1] = std::max(-1.0f, static_cast<float>(y) / 1023.0f);
            out[2] = std::max(-1.0f, static_cast<float>(z) / 511.0f);
            break;
        }
    }
}

bool fetch_vertices(const KelvinState& state, const HandleResolver& handles, const MemoryView& memory, const DrawBatch& batch,
                    FetchedVertices& out) {
    out = FetchedVertices{};
    out.primitive = batch.primitive;

    if (batch.immediate_count) {
        // An attribute no vertex of the batch set keeps the value it had before BEGIN for all of them.
        out.count = batch.immediate_count;
        out.enabled = static_cast<u16>(batch.immediate_mask | 1u);
        for (u32 a = 0; a < kAttributes; ++a) {
            if (!(out.enabled & (1u << a))) {
                out.constant[a] = state.immediate[a];
                continue;
            }
            auto& dst = out.attribute[a];
            dst.resize(static_cast<std::size_t>(out.count) * 4);
            for (u32 v = 0; v < out.count; ++v)
                for (u32 c = 0; c < 4; ++c) dst[static_cast<std::size_t>(v) * 4 + c] = batch.immediate_vertices[static_cast<std::size_t>(v) * kAttributes + a][c];
        }
        return true;
    }

    if (!batch.inline_words.empty()) {
        // Interleaved by attribute order, each enabled attribute's element rounded up to whole words.
        u32 stride = 0;
        for (const auto& a : state.attributes)
            if (a.size) stride += (a.bytes() + 3) & ~3u;
        if (stride == 0) {
            out.error = "inline array with no enabled attribute";
            return false;
        }
        const u32 bytes = static_cast<u32>(batch.inline_words.size() * 4);
        out.count = bytes / stride;
        const u8* data = reinterpret_cast<const u8*>(batch.inline_words.data());
        u32 offset = 0;
        for (u32 a = 0; a < kAttributes; ++a) {
            const VertexAttribute& fmt = state.attributes[a];
            if (!fmt.size) {
                out.constant[a] = state.immediate[a];
                continue;
            }
            out.enabled = static_cast<u16>(out.enabled | (1u << a));
            auto& dst = out.attribute[a];
            dst.resize(static_cast<std::size_t>(out.count) * 4);
            for (u32 v = 0; v < out.count; ++v) decode_attribute(fmt, data + static_cast<std::size_t>(v) * stride + offset, &dst[static_cast<std::size_t>(v) * 4]);
            offset += (fmt.bytes() + 3) & ~3u;
        }
        return true;
    }

    std::vector<u32> indices = batch.indices;
    for (const auto& [first, count] : batch.ranges)
        for (u32 i = 0; i < count; ++i) indices.push_back(first + i);
    if (indices.empty()) {
        out.error = "batch without geometry";
        return false;
    }
    const auto [min_it, max_it] = std::minmax_element(indices.begin(), indices.end());
    const u32 lo = *min_it, hi = *max_it;
    out.count = hi - lo + 1;
    out.indices.resize(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) out.indices[i] = indices[i] - lo;

    const DmaObject dma_a = handles.resolve(state.dma_handle(mthd::kDmaVertexA));
    const DmaObject dma_b = handles.resolve(state.dma_handle(mthd::kDmaVertexB));
    for (u32 a = 0; a < kAttributes; ++a) {
        const VertexAttribute& fmt = state.attributes[a];
        // A disabled attribute reads its current value for every vertex.
        if (!fmt.size) {
            out.constant[a] = state.immediate[a];
            continue;
        }
        const DmaObject& dma = fmt.dma_b ? dma_b : dma_a;
        if (!dma.valid) {
            out.error = std::format("attribute {} names an unresolved vertex DMA object", a);
            return false;
        }
        // A stride of 0 makes the attribute a constant: every vertex reads the array's first element.
        if (fmt.stride == 0) {
            if (fmt.offset + static_cast<u64>(fmt.bytes()) > static_cast<u64>(dma.limit) + 1) {
                out.error = std::format("attribute {} runs past the vertex DMA object's limit", a);
                return false;
            }
            const u64 at = static_cast<u64>(dma.base) + fmt.offset;
            const u8* p = at + fmt.bytes() <= 0xFFFFFFFFull ? memory.at(static_cast<u32>(at), fmt.bytes()) : nullptr;
            if (!p) {
                out.error = std::format("attribute {} at {:08x} is outside RAM", a, at);
                return false;
            }
            decode_attribute(fmt, p, out.constant[a].data());
            continue;
        }
        const u32 stride = fmt.stride;
        const u64 first = static_cast<u64>(dma.base) + fmt.offset + static_cast<u64>(lo) * stride;
        const u64 span = static_cast<u64>(out.count - 1) * stride + fmt.bytes();
        if (fmt.offset + static_cast<u64>(lo) * stride + span > static_cast<u64>(dma.limit) + 1) {
            out.error = std::format("attribute {} runs past the vertex DMA object's limit ({:x} > {:x})", a,
                                    fmt.offset + static_cast<u64>(lo) * stride + span, static_cast<u64>(dma.limit) + 1);
            return false;
        }
        if (first + span > 0xFFFFFFFFull) {
            out.error = std::format("attribute {} runs past the address space", a);
            return false;
        }
        const u8* p = memory.at(static_cast<u32>(first), static_cast<u32>(span));
        if (!p) {
            out.error = std::format("attribute {} at {:08x} (+{:x}) is outside RAM", a, first, span);
            return false;
        }
        out.enabled = static_cast<u16>(out.enabled | (1u << a));
        auto& dst = out.attribute[a];
        dst.resize(static_cast<std::size_t>(out.count) * 4);
        for (u32 v = 0; v < out.count; ++v) decode_attribute(fmt, p + static_cast<std::size_t>(v) * stride, &dst[static_cast<std::size_t>(v) * 4]);
    }
    return true;
}

void latch_current_values(KelvinState& state, const HandleResolver* handles, const MemoryView& memory, const DrawBatch& batch) {
    if (batch.immediate_count) return;  // the immediate methods wrote the current values as they went
    if (!batch.inline_words.empty()) {
        // The last whole vertex of the interleaved data, laid out as fetch_vertices reads it.
        u32 stride = 0;
        for (const auto& a : state.attributes)
            if (a.size) stride += (a.bytes() + 3) & ~3u;
        const std::size_t bytes = batch.inline_words.size() * 4;
        if (stride == 0 || bytes < stride) return;
        const u8* last = reinterpret_cast<const u8*>(batch.inline_words.data()) + (bytes / stride - 1) * stride;
        u32 offset = 0;
        for (u32 a = 0; a < kAttributes; ++a) {
            const VertexAttribute& fmt = state.attributes[a];
            if (!fmt.size) continue;
            float v[4];
            decode_attribute(fmt, last + offset, v);
            std::memcpy(state.immediate[a].data(), v, sizeof v);
            offset += (fmt.bytes() + 3) & ~3u;
        }
        return;
    }
    if (!handles) return;
    // The batch's last vertex as fetch_vertices orders them: the indices, then the DRAW_ARRAYS ranges.
    u32 index;
    if (!batch.ranges.empty()) index = batch.ranges.back().first + batch.ranges.back().second - 1;
    else if (!batch.indices.empty()) index = batch.indices.back();
    else return;
    const DmaObject dma_a = handles->resolve(state.dma_handle(mthd::kDmaVertexA));
    const DmaObject dma_b = handles->resolve(state.dma_handle(mthd::kDmaVertexB));
    for (u32 a = 0; a < kAttributes; ++a) {
        const VertexAttribute& fmt = state.attributes[a];
        if (!fmt.size) continue;
        const DmaObject& dma = fmt.dma_b ? dma_b : dma_a;
        if (!dma.valid) continue;
        const u64 rel = fmt.offset + static_cast<u64>(index) * fmt.stride;  // a stride of 0 reads the first element
        if (rel + fmt.bytes() > static_cast<u64>(dma.limit) + 1 || dma.base + rel + fmt.bytes() > 0xFFFFFFFFull) continue;
        const u8* p = memory.at(static_cast<u32>(dma.base + rel), fmt.bytes());
        if (!p) continue;
        float v[4];
        decode_attribute(fmt, p, v);
        std::memcpy(state.immediate[a].data(), v, sizeof v);
    }
}

}  // namespace mv::pgraph
