#pragma once
#include <array>
#include <functional>
#include <memory>
#include <mutex>

#include "usb/usb_device.h"

// The Xbox pad as the console sees it: a compound device, an internal three-port hub with the XID gamepad
// function on its port 1 (ports 2 and 3 are the memory-unit slots, empty here). Descriptors and reports from the
// Controller S (23-0923H) dump in xboxdevwiki's Xbox_Input_Devices; the hub class from USB 1.1 chapter 11. See
// local-notes/hw/usb.md.
namespace mv::usb {

// The gamepad's state as a host maps it: digital buttons, eight analog buttons (0-255) and two sticks.
struct PadState {
    enum Button : u16 { Up = 0x01, Down = 0x02, Left = 0x04, Right = 0x08, Start = 0x10, Back = 0x20, LeftThumb = 0x40, RightThumb = 0x80 };
    enum Analog : u8 { A, B, X, Y, Black, White, LeftTrigger, RightTrigger };
    u16 buttons{0};
    std::array<u8, 8> analog{};
    i16 left_x{0}, left_y{0}, right_x{0}, right_y{0};
    bool operator==(const PadState&) const = default;
};

class XidGamepad : public Device {
public:
    using Rumble = std::function<void(u16 left, u16 right)>;
    // From any thread: the host's mapping of its own controller.
    void set_state(const PadState& s);
    void set_rumble(Rumble r) { rumble_ = std::move(r); }
    // The 20-byte input report for a state.
    static std::array<u8, 20> report(const PadState& s);

protected:
    std::span<const u8> device_descriptor() const override;
    std::span<const u8> configuration_descriptor() const override;
    bool class_request(const Setup& s, std::span<const u8> data, std::vector<u8>& reply) override;
    Status endpoint_in(u8 endpoint, u8* data, u32 len, u32& actual) override;
    Status endpoint_out(u8 endpoint, const u8* data, u32 len) override;

private:
    PadState current() const;

    mutable std::mutex lock_;
    PadState state_{};
    PadState last_sent_{};
    bool sent_once_{false};
    Rumble rumble_;
};

class Hub : public Device {
public:
    static constexpr unsigned kPorts = 3;
    // Plugs a device into a downstream port (1-based); nullptr empties it.
    void attach(unsigned port, Device* device);
    void bus_reset() override;
    Device* find(u8 address) override;

protected:
    std::span<const u8> device_descriptor() const override;
    std::span<const u8> configuration_descriptor() const override;
    bool class_request(const Setup& s, std::span<const u8> data, std::vector<u8>& reply) override;
    Status endpoint_in(u8 endpoint, u8* data, u32 len, u32& actual) override;

private:
    struct Port {
        Device* device{nullptr};
        u16 status{0}, change{0};
    };
    std::array<Port, kPorts> ports_{};
};

// One controller port's worth: the hub with the gamepad on its port 1.
class XboxPad {
public:
    XboxPad() { hub.attach(1, &gamepad); }
    Hub hub;
    XidGamepad gamepad;
};

}  // namespace mv::usb
