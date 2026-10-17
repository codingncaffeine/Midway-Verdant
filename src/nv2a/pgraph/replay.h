#pragma once
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "nv2a/capture.h"
#include "nv2a/pgraph/kelvin.h"

namespace mv::pgraph {

// Replays a pushbuffer capture: keeps the RAM image the capture carries, the handle table, and feeds the Kelvin
// state machine every 3D method in order. Frames end at the 3D class's semaphore releases.
class Replay : public HandleResolver {
public:
    class Observer {
    public:
        virtual ~Observer() = default;
        virtual void frame_end(u32 frame_index, u32 semaphore_value) {
            (void)frame_index;
            (void)semaphore_value;
        }
        virtual void register_write(u32 offset, u32 value) {
            (void)offset;
            (void)value;
        }
        // A method of a class other than Kelvin (the 2D classes); not modelled yet.
        virtual void other_method(u32 cls, u32 subchannel, u32 method, u32 value) {
            (void)cls;
            (void)subchannel;
            (void)method;
            (void)value;
        }
    };

    bool open(const std::filesystem::path& path, std::string& error);
    // Runs the whole capture. `until_frame` stops after that many frames have ended (0 = all).
    bool run(Kelvin& kelvin, Observer* observer, std::string& error, u32 until_frame = 0);

    MemoryView memory() const { return {ram_.data(), static_cast<u32>(ram_.size())}; }
    DmaObject resolve(u32 handle) const override;
    u32 frames() const { return frames_; }
    u32 scanout_start() const { return scanout_start_; }
    u64 methods() const { return methods_; }

private:
    capture::Reader reader_;
    std::vector<u8> ram_;
    std::unordered_map<u32, std::array<u32, 4>> objects_;
    u32 frames_{0};
    u32 scanout_start_{0};
    u64 methods_{0};
};

}  // namespace mv::pgraph
