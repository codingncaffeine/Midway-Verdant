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
constexpr u32 kAttrPosition = 0, kAttrNormal = 2, kAttrDiffuse = 3, kAttrSpecular = 4, kAttrTexcoord0 = 9;

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
    batch_ = DrawBatch{};
}

void Kelvin::method(u32 method, u32 value) {
    method &= 0x1FFCu;
    ++methods_;
    ++histogram_[method >> 2];
    state_.r[method >> 2] = value;

    // Streaming ranges first.
    if (method >= mthd::kTransformProgram && method < mthd::kTransformProgram + 0x80) {
        if (state_.program_load < kProgramSlots) state_.program[state_.program_load][state_.program_word] = value;
        if (++state_.program_word == 4) {
            state_.program_word = 0;
            ++state_.program_load;
        }
        return;
    }
    if (method >= mthd::kTransformConstant && method < mthd::kTransformConstant + 0x80) {
        if (state_.constant_load < kConstants) state_.constants[state_.constant_load][state_.constant_word] = value;
        if (++state_.constant_word == 4) {
            state_.constant_word = 0;
            ++state_.constant_load;
        }
        return;
    }
    // The fixed-function transform registers live in the constant file too (the "XF context"): a program can read
    // the viewport scale and offset as c[58] and c[59], the matrices from slot 0 on, and the XDK's programs do.
    if (u32 slot_word; xf_context_slot(method, slot_word)) {
        state_.constants[slot_word >> 2][slot_word & 3u] = value;
        return;
    }
    if (method >= mthd::kVertexDataArrayFormat && method < mthd::kVertexDataArrayFormat + 0x40) {
        VertexAttribute& a = state_.attributes[(method - mthd::kVertexDataArrayFormat) >> 2];
        a.type = static_cast<AttribType>(value & 0xFu);
        a.size = (value >> 4) & 0xFu;
        a.stride = value >> 8;
        return;
    }
    if (method >= mthd::kVertexDataArrayOffset && method < mthd::kVertexDataArrayOffset + 0x40) {
        VertexAttribute& a = state_.attributes[(method - mthd::kVertexDataArrayOffset) >> 2];
        a.offset = value & 0x7FFFFFFFu;
        a.dma_b = (value >> 31) & 1u;
        return;
    }
    if (method >= mthd::kVertexData4fM && method < mthd::kVertexData4fM + 0x100) {
        const u32 i = (method - mthd::kVertexData4fM) >> 2;
        immediate_component(i >> 2, i & 3u, as_float(value), (i & 3u) == 3);
        return;
    }
    if (method >= mthd::kVertexData2fM && method < mthd::kVertexData2fM + 0x80) {
        const u32 i = (method - mthd::kVertexData2fM) >> 2;
        const u32 attr = i >> 1, comp = i & 1u;
        if (comp == 0) {
            state_.immediate[attr][2] = 0.0f;
            state_.immediate[attr][3] = 1.0f;
        }
        immediate_component(attr, comp, as_float(value), comp == 1);
        return;
    }
    if (method >= mthd::kVertexData2s && method < mthd::kVertexData2s + 0x40) {
        const u32 attr = (method - mthd::kVertexData2s) >> 2;
        state_.immediate[attr][2] = 0.0f;
        state_.immediate[attr][3] = 1.0f;
        immediate_component(attr, 0, s16_plain(value), false);
        immediate_component(attr, 1, s16_plain(value >> 16), true);
        return;
    }
    if (method >= mthd::kVertexData4ub && method < mthd::kVertexData4ub + 0x40) {
        const u32 attr = (method - mthd::kVertexData4ub) >> 2;
        for (u32 c = 0; c < 4; ++c) immediate_component(attr, c, static_cast<float>((value >> (8 * c)) & 0xFFu) / 255.0f, c == 3);
        return;
    }
    if (method >= mthd::kVertexData4sM && method < mthd::kVertexData4sM + 0x80) {
        const u32 i = (method - mthd::kVertexData4sM) >> 2;
        const u32 attr = i >> 1, half = i & 1u;
        immediate_component(attr, half * 2, s16_plain(value), false);
        immediate_component(attr, half * 2 + 1, s16_plain(value >> 16), half == 1);
        return;
    }
    // The named immediate methods.
    auto named_float = [&](u32 base, u32 attr, u32 count) -> bool {
        if (method < base || method >= base + count * 4) return false;
        const u32 comp = (method - base) >> 2;
        if (comp == 0 && count == 3) state_.immediate[attr][3] = 1.0f;
        immediate_component(attr, comp, as_float(value), comp == count - 1);
        return true;
    };
    auto named_s16x2 = [&](u32 base, u32 attr, u32 words, bool normalised) -> bool {
        if (method < base || method >= base + words * 4) return false;
        const u32 w = (method - base) >> 2;
        if (w == 0 && words == 1) {
            state_.immediate[attr][2] = 0.0f;
            state_.immediate[attr][3] = 1.0f;
        }
        const float lo = normalised ? s16_norm(value) : s16_plain(value);
        const float hi = normalised ? s16_norm(value >> 16) : s16_plain(value >> 16);
        immediate_component(attr, w * 2, lo, false);
        immediate_component(attr, w * 2 + 1, hi, w == words - 1);
        return true;
    };
    auto named_ub4 = [&](u32 base, u32 attr) -> bool {
        if (method != base) return false;
        for (u32 c = 0; c < 4; ++c) immediate_component(attr, c, static_cast<float>((value >> (8 * c)) & 0xFFu) / 255.0f, c == 3);
        return true;
    };
    if (named_float(mthd::kVertex3f, kAttrPosition, 3) || named_float(mthd::kVertex4f, kAttrPosition, 4) ||
        named_float(mthd::kNormal3f, kAttrNormal, 3) || named_s16x2(mthd::kNormal3s, kAttrNormal, 2, true) ||
        named_float(mthd::kDiffuse4f, kAttrDiffuse, 4) || named_float(mthd::kDiffuse3f, kAttrDiffuse, 3) ||
        named_ub4(mthd::kDiffuse4ub, kAttrDiffuse) || named_float(mthd::kSpecular4f, kAttrSpecular, 4) ||
        named_float(mthd::kSpecular3f, kAttrSpecular, 3) || named_ub4(mthd::kSpecular4ub, kAttrSpecular))
        return;
    static constexpr u32 kTex2f[4] = {mthd::kTexcoord0_2f, mthd::kTexcoord1_2f, mthd::kTexcoord2_2f, mthd::kTexcoord3_2f};
    static constexpr u32 kTex2s[4] = {mthd::kTexcoord0_2s, mthd::kTexcoord1_2s, mthd::kTexcoord2_2s, mthd::kTexcoord3_2s};
    static constexpr u32 kTex4f[4] = {mthd::kTexcoord0_4f, mthd::kTexcoord1_4f, mthd::kTexcoord2_4f, mthd::kTexcoord3_4f};
    static constexpr u32 kTex4s[4] = {mthd::kTexcoord0_4s, mthd::kTexcoord1_4s, mthd::kTexcoord2_4s, mthd::kTexcoord3_4s};
    for (u32 t = 0; t < 4; ++t) {
        if (named_float(kTex2f[t], kAttrTexcoord0 + t, 2) || named_s16x2(kTex2s[t], kAttrTexcoord0 + t, 1, false) ||
            named_float(kTex4f[t], kAttrTexcoord0 + t, 4) || named_s16x2(kTex4s[t], kAttrTexcoord0 + t, 2, false))
            return;
    }

    switch (method) {
        case mthd::kTransformProgramLoad:
            state_.program_load = value & 0xFFu;
            state_.program_word = 0;
            return;
        case mthd::kTransformConstantLoad:
            state_.constant_load = value & 0x1FFu;
            state_.constant_word = 0;
            return;
        case mthd::kBeginEnd:
            begin_end(value);
            return;
        case mthd::kArrayElement16:
            batch_.indices.push_back(value & 0xFFFFu);
            batch_.indices.push_back(value >> 16);
            return;
        case mthd::kArrayElement32:
            batch_.indices.push_back(value);
            return;
        case mthd::kDrawArrays:
            batch_.ranges.emplace_back(value & 0xFFFFFFu, (value >> 24) + 1);
            return;
        case mthd::kInlineArray:
            batch_.inline_words.push_back(value);
            return;
        case mthd::kClearSurface:
            if (sink_) sink_->clear(state_, value);
            return;
        case mthd::kSemaphoreRelease:
            if (sink_) sink_->semaphore_release(state_, value);
            return;
        default:
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
        out.count = batch.immediate_count;
        out.enabled = static_cast<u16>(batch.immediate_mask | 1u);
        for (u32 a = 0; a < kAttributes; ++a) {
            if (!(out.enabled & (1u << a))) continue;
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
            if (!fmt.size) continue;
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
        if (!fmt.size) continue;
        const DmaObject& dma = fmt.dma_b ? dma_b : dma_a;
        if (!dma.valid) {
            out.error = std::format("attribute {} names an unresolved vertex DMA object", a);
            return false;
        }
        const u32 stride = fmt.stride ? fmt.stride : fmt.bytes();
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

}  // namespace mv::pgraph
