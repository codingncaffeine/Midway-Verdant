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
    // Instruments: interrupt-IN polls of the report endpoint, and the reports actually delivered (a NAK otherwise).
    u64 polls() const { return polls_; }
    u64 reports() const { return reports_; }
    u64 control_requests() const { return control_requests_; }

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
    u64 polls_{0}, reports_{0}, control_requests_{0};
    Rumble rumble_;
};

class Hub : public Device {
public:
    static constexpr unsigned kMaxPorts = 4;
    // A pad's hub has three downstream ports, the gamepad's fixed on port 1; the 1.0-1.5 motherboard's hub
    // (the front-panel daughterboard on root port 1) has four, all removable.
    explicit Hub(unsigned ports = 3, u8 fixed_ports = 0x02, bool compound = true);
    unsigned ports() const { return nports_; }
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
    unsigned nports_;
    std::array<u8, 9> hub_descriptor_;
    std::array<Port, kMaxPorts> ports_{};
};

// The front panel of a 1.0-1.5 board: a four-port hub on root port 1 of the first OHCI. The kernel sets
// XboxHardwareInfo.Flags bit 0 for these boards (LPC bridge revision <= 0xB3), and XAPI then looks for the
// controllers BEHIND this hub, front ports 1-4 on its ports 3, 4, 1, 2; a pad on a root port is disabled.
struct FrontPanel {
    static constexpr unsigned kRootPort = 1;
    static constexpr unsigned kHubPortOfFrontPort[4] = {3, 4, 1, 2};
    Hub hub{4, 0x00, false};
};

// One controller port's worth: the hub with the gamepad on its port 1.
class XboxPad {
public:
    XboxPad() { hub.attach(1, &gamepad); }
    Hub hub;
    XidGamepad gamepad;
};

}  // namespace mv::usb
