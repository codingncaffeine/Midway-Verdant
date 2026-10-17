#pragma once
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "core/bus.h"
#include "core/types.h"

namespace mv {

// The command stream the CPU feeds the GPU: a ring of 32-bit words in RAM, reached through the DMA object
// PFIFO's CACHE1 is bound to. This walker is the pusher/puller's front half: it follows jumps and calls,
// decodes method headers (increasing and non-increasing), and hands each method to a sink. Layout per the
// public NV FIFO documentation (envytools "fifo/dma-pusher"): header bits 31:29 type, 28:18 count,
// 15:13 subchannel, 12:2 method; a word with its low bits 01 is a jump, 10 a call, 0x00020000 a return.
struct PushbufferMethod {
    u32 subchannel;
    u32 method;
    u32 value;
    u32 header_address;  // where in the ring the header sat (for tracing)
};

class PushbufferWalker {
public:
    using Sink = std::function<void(const PushbufferMethod&)>;

    struct DmaObject {
        u32 base{0};
        u32 limit{0};  // last valid offset
        bool valid{false};
    };

    explicit PushbufferWalker(Bus& bus) : bus_(bus) {}
    void set_sink(Sink s) { sink_ = std::move(s); }

    // Run from `get` until it reaches `put` or an error; returns the new GET. Jump targets, like GET and PUT,
    // are offsets within the DMA object (physical addresses when the object spans RAM from zero).
    u32 run(const DmaObject& dma, u32 get, u32 put);

    u64 methods_seen() const { return methods_; }
    u64 words_fetched() const { return words_; }
    bool errored() const { return error_; }
    std::string last_error() const { return last_error_; }

private:
    bool fetch(const DmaObject& dma, u32 offset, u32& out);

    Bus& bus_;
    Sink sink_;
    std::vector<u32> call_stack_;
    u64 methods_{0}, words_{0};
    bool error_{false};
    std::string last_error_;
};

}  // namespace mv
