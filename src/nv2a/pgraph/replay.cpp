#include "nv2a/pgraph/replay.h"

#include "nv2a/pgraph/twod.h"

#include <cstring>

namespace mv::pgraph {

bool Replay::open(const std::filesystem::path& path, std::string& error) {
    if (!reader_.open(path, error)) return false;
    if (reader_.ram_size() == 0 || reader_.ram_size() > (256u << 20)) {
        error = "capture RAM size out of range";
        return false;
    }
    ram_.assign(reader_.ram_size(), 0);
    return true;
}

DmaObject Replay::resolve(u32 handle) const {
    DmaObject d;
    const auto it = objects_.find(handle);
    if (it == objects_.end()) return d;
    const auto& dw = it->second;
    d.cls = dw[0] & 0xFFu;
    if (d.cls != 0x02 && d.cls != 0x03 && d.cls != 0x3D) return d;
    d.base = (dw[2] & 0xFFFFF000u) | ((dw[0] >> 20) & 0xFFFu);
    d.limit = dw[1];
    d.valid = true;
    return d;
}

bool Replay::run(Kelvin& kelvin, Observer* observer, std::string& error, u32 until_frame) {
    capture::Record rec;
    TwoD twod(ram_.data(), ram_.size(), *this);
    twod.set_hooks(blit_before_read_, blit_after_write_);
    if (blit_ram_copy_) twod.set_ram_copy(blit_ram_copy_);
    while (reader_.next(rec)) {
        switch (rec.type) {
            case capture::Rec::Pages: {
                const std::size_t off = static_cast<std::size_t>(rec.first_page) * capture::kPageSize;
                const std::size_t len = static_cast<std::size_t>(rec.page_count) * capture::kPageSize;
                if (off + len <= ram_.size()) std::memcpy(ram_.data() + off, rec.page_data, len);
                if (page_observer_) page_observer_(rec.first_page, rec.page_count);
                break;
            }
            case capture::Rec::Object:
                objects_[rec.handle] = {rec.dw[0], rec.dw[1], rec.dw[2], rec.dw[3]};
                break;
            case capture::Rec::Register:
                if (rec.offset == 0x600800) scanout_start_ = rec.value;
                if (observer) observer->register_write(rec.offset, rec.value);
                break;
            case capture::Rec::Run:
                break;
            case capture::Rec::Method:
                ++methods_;
                if (rec.cls == 0x97) {
                    kelvin.method(rec.method, rec.value);
                    if (method_hook_) method_hook_(rec.method, rec.value);
                    if (rec.method == mthd::kSemaphoreRelease) {
                        if (observer) observer->frame_end(frames_, rec.value);
                        ++frames_;
                        if (until_frame && frames_ >= until_frame) return true;
                    }
                } else if (rec.cls != 0) {
                    if (rec.cls == TwoD::kClassSurfaces || rec.cls == TwoD::kClassBlit) twod.method(rec.cls, rec.method, rec.value);
                    if (observer) observer->other_method(rec.cls, rec.subchannel, rec.method, rec.value);
                }
                break;
        }
    }
    if (!reader_.error().empty()) {
        error = reader_.error();
        return false;
    }
    return true;
}

}  // namespace mv::pgraph
