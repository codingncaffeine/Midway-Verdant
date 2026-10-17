#pragma once
#include <array>
#include <string>
#include <vector>

#include "nv2a/pgraph/kelvin_state.h"

namespace mv::pgraph {

// A DMA object as a handle resolves to it: a window onto RAM.
struct DmaObject {
    u32 base{0};
    u32 limit{0};  // last valid offset
    u32 cls{0};
    bool valid{false};
};

class HandleResolver {
public:
    virtual ~HandleResolver() = default;
    virtual DmaObject resolve(u32 handle) const = 0;
};

// Guest RAM as the engine reads it (through the direct window or the 0xF0000000 alias).
struct MemoryView {
    const u8* ram{nullptr};
    u32 size{0};
    const u8* at(u32 phys, u32 len) const {
        u32 off = phys;
        if (off >= 0xF0000000u) off -= 0xF0000000u;
        if (!ram || off >= size || len > size - off) return nullptr;
        return ram + off;
    }
};

// What the state machine hands to a renderer.
class RenderSink {
public:
    virtual ~RenderSink() = default;
    virtual void clear(const KelvinState& state, u32 clear_bits) = 0;
    virtual void draw(const KelvinState& state, const DrawBatch& batch) = 0;
    virtual void semaphore_release(const KelvinState& state, u32 value) {
        (void)state;
        (void)value;
    }
};

// The Kelvin (class 0x97) method state machine: every method lands in the state; the streaming methods fill the
// program and constant memories or accumulate geometry; BEGIN/END and CLEAR_SURFACE reach the sink.
class Kelvin {
public:
    Kelvin();
    void set_sink(RenderSink* sink) { sink_ = sink; }
    void method(u32 method, u32 value);

    const KelvinState& state() const { return state_; }
    KelvinState& state() { return state_; }
    bool in_begin() const { return in_begin_; }
    const DrawBatch& batch() const { return batch_; }
    u64 methods_seen() const { return methods_; }
    const std::array<u64, 0x800>& histogram() const { return histogram_; }

private:
    void begin_end(u32 value);
    void immediate_component(u32 attribute, u32 component, float v, bool completes_vertex);
    void immediate_vertex_done();

    KelvinState state_;
    DrawBatch batch_;
    bool in_begin_{false};
    RenderSink* sink_{nullptr};
    u64 methods_{0};
    std::array<u64, 0x800> histogram_{};
};

// Geometry of one batch resolved to flat float4 attribute arrays and a rebased index list, ready for any backend.
struct FetchedVertices {
    Primitive primitive{Primitive::End};
    u32 count{0};
    u16 enabled{0};  // attributes with data
    std::array<std::vector<float>, kAttributes> attribute;  // count * 4 floats each, for enabled attributes
    std::vector<u32> indices;  // into 0..count-1; empty = sequential
    std::string error;  // non-empty when memory could not be read; the batch is then unusable
};

bool fetch_vertices(const KelvinState& state, const HandleResolver& handles, const MemoryView& memory, const DrawBatch& batch,
                    FetchedVertices& out);

// Decodes one attribute element at `p` by its array format into x y z w (defaults 0 0 0 1).
void decode_attribute(const VertexAttribute& a, const u8* p, float out[4]);

const char* primitive_name(Primitive p);

}  // namespace mv::pgraph
