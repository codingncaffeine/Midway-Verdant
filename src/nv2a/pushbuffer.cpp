#include "nv2a/pushbuffer.h"

#include <format>

#include "debug/trace.h"

namespace mv {

bool PushbufferWalker::fetch(const DmaObject& dma, u32 offset, u32& out) {
    if (!dma.valid || offset > dma.limit || offset + 4 > dma.limit + 1) return false;
    if (!bus_.read_ram(dma.base + offset, &out, 4)) return false;
    ++words_;
    return true;
}

u32 PushbufferWalker::run(const DmaObject& dma, u32 get, u32 put) {
    error_ = false;
    for (int guard = 0; guard < 4'000'000 && get != put; ++guard) {
        u32 w;
        if (!fetch(dma, get, w)) {
            error_ = true;
            last_error_ = std::format("fetch failed at offset {:08x} (dma base {:08x} limit {:08x})", get, dma.base, dma.limit);
            trace::log(trace::Cat::Gpu, "pushbuffer: {}", last_error_);
            return get;
        }
        if ((w & 0xE0000003u) == 0x20000000u) {  // old-style jump
            get = w & 0x1FFFFFFCu;
            continue;
        }
        if ((w & 3u) == 1u) {  // jump
            get = w & 0xFFFFFFFCu;
            continue;
        }
        if ((w & 3u) == 2u) {  // call
            if (call_stack_.size() > 64) {
                error_ = true;
                last_error_ = "call stack overflow";
                return get;
            }
            call_stack_.push_back(get + 4);
            get = w & 0xFFFFFFFCu;
            continue;
        }
        if (w == 0x00020000u) {  // return
            if (call_stack_.empty()) {
                error_ = true;
                last_error_ = std::format("return with an empty call stack at {:08x}", get);
                return get + 4;
            }
            get = call_stack_.back();
            call_stack_.pop_back();
            continue;
        }
        const u32 type = w & 0xE0030003u;
        if (type != 0 && type != 0x40000000u) {
            error_ = true;
            last_error_ = std::format("unknown header {:08x} at {:08x}", w, get);
            trace::log(trace::Cat::Gpu, "pushbuffer: {}", last_error_);
            return get + 4;
        }
        const bool increasing = type == 0;
        const u32 count = (w >> 18) & 0x7FFu;
        const u32 subchannel = (w >> 13) & 7u;
        u32 method = w & 0x1FFCu;
        const u32 header_at = get;
        get += 4;
        for (u32 i = 0; i < count; ++i) {
            u32 v;
            if (!fetch(dma, get, v)) {
                error_ = true;
                last_error_ = std::format("data fetch failed at {:08x}", get);
                return get;
            }
            get += 4;
            ++methods_;
            if (sink_) sink_({subchannel, method, v, header_at});
            if (increasing) method += 4;
        }
    }
    return get;
}

}  // namespace mv
