#include "usb/xbox_pad.h"

#include <algorithm>
#include <cstring>

namespace mv::usb {

namespace {
// Controller S (23-0923H): USB 1.10, class 0, maxpacket0 8, VID 045E PID 0289, bcdDevice 1.20.
constexpr std::array<u8, 18> kPadDevice{18, 1, 0x10, 0x01, 0, 0, 0, 8, 0x5E, 0x04, 0x89, 0x02, 0x20, 0x01, 0, 0, 0, 1};
// One configuration: interface class 0x58 subclass 0x42, interrupt IN 0x81 and OUT 0x02, 32 bytes, 4 ms.
constexpr std::array<u8, 32> kPadConfiguration{9, 2, 32, 0, 1, 1, 0, 0x80, 50,
                                               9, 4, 0, 0, 2, 0x58, 0x42, 0, 0,
                                               7, 5, 0x81, 3, 32, 0, 4,
                                               7, 5, 0x02, 3, 32, 0, 4};
constexpr std::array<u8, 16> kXidDescriptor{16, 0x42, 0x00, 0x01, 0x01, 0x02, 20, 6, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// The pad's internal hub: class 9, three ports, one status-change endpoint.
constexpr std::array<u8, 18> kHubDevice{18, 1, 0x10, 0x01, 9, 0, 0, 8, 0x5E, 0x04, 0x88, 0x02, 0x01, 0x00, 0, 0, 0, 1};
constexpr std::array<u8, 25> kHubConfiguration{9, 2, 25, 0, 1, 1, 0, 0xA0, 50,
                                               9, 4, 0, 0, 1, 9, 0, 0, 0,
                                               7, 5, 0x81, 3, 1, 0, 0xFF};
// The hub class descriptor is built per hub (see the constructor): per-port power + per-port over-current,
// compound for a pad's hub, 100 ms power-good, 64 mA. The motherboard hub answers with the same device
// descriptor; XAPI matches hubs by class (9), and the board's own IDs are not known here.

enum : u16 { kPortConnection = 0x0001, kPortEnable = 0x0002, kPortSuspend = 0x0004, kPortReset = 0x0010, kPortPower = 0x0100 };
enum : u16 { kFeatPortEnable = 1, kFeatPortSuspend = 2, kFeatPortReset = 4, kFeatPortPower = 8, kFeatCConnection = 16, kFeatCEnable = 17,
             kFeatCSuspend = 18, kFeatCOverCurrent = 19, kFeatCReset = 20 };
}  // namespace

// ---- the gamepad ----

void XidGamepad::set_state(const PadState& s) {
    std::lock_guard g(lock_);
    state_ = s;
}

PadState XidGamepad::current() const {
    std::lock_guard g(lock_);
    return state_;
}

std::array<u8, 20> XidGamepad::report(const PadState& s) {
    std::array<u8, 20> r{};
    r[0] = 0x00;
    r[1] = 20;
    r[2] = static_cast<u8>(s.buttons);
    std::copy(s.analog.begin(), s.analog.end(), r.begin() + 4);
    const i16 axes[4] = {s.left_x, s.left_y, s.right_x, s.right_y};
    for (std::size_t i = 0; i < 4; ++i) {
        r[12 + 2 * i] = static_cast<u8>(axes[i]);
        r[13 + 2 * i] = static_cast<u8>(static_cast<u16>(axes[i]) >> 8);
    }
    return r;
}

std::span<const u8> XidGamepad::device_descriptor() const { return kPadDevice; }
std::span<const u8> XidGamepad::configuration_descriptor() const { return kPadConfiguration; }

bool XidGamepad::class_request(const Setup& s, std::span<const u8> data, std::vector<u8>& reply) {
    ++control_requests_;
    if (s.request_type == 0xC1 && s.request == 6 && s.value == 0x4200) {  // GET_DESCRIPTOR (XID)
        reply.assign(kXidDescriptor.begin(), kXidDescriptor.end());
        return true;
    }
    if (s.request_type == 0xC1 && s.request == 1) {  // GET_CAPABILITIES: every bit of the report is real
        if (s.value == 0x0100) {
            reply.assign(20, 0xFF);
            reply[0] = 0x00;
            reply[1] = 20;
            reply[3] = 0x00;
            return true;
        }
        if (s.value == 0x0200) {
            reply = {0x00, 6, 0xFF, 0xFF, 0xFF, 0xFF};
            return true;
        }
        return false;
    }
    if (s.request_type == 0xA1 && s.request == 1 && s.value == 0x0100) {  // GET_REPORT
        const auto r = report(current());
        reply.assign(r.begin(), r.end());
        return true;
    }
    if (s.request_type == 0x21 && s.request == 9 && s.value == 0x0200) {  // SET_REPORT: rumble
        return endpoint_out(2, data.data(), static_cast<u32>(data.size())) == Status::Ack;
    }
    return false;
}

Status XidGamepad::endpoint_in(u8 endpoint, u8* data, u32 len, u32& actual) {
    if (endpoint != 1) return Status::Stall;
    const PadState s = current();
    ++polls_;
    if (sent_once_ && s == last_sent_) return Status::Nak;  // nothing new since the last report
    const auto r = report(s);
    actual = std::min<u32>(len, static_cast<u32>(r.size()));
    std::memcpy(data, r.data(), actual);
    last_sent_ = s;
    sent_once_ = true;
    ++reports_;
    return Status::Ack;
}

Status XidGamepad::endpoint_out(u8 endpoint, const u8* data, u32 len) {
    if (endpoint != 2) return Status::Stall;
    if (len >= 6 && rumble_) rumble_(static_cast<u16>(data[2] | (data[3] << 8)), static_cast<u16>(data[4] | (data[5] << 8)));
    return Status::Ack;
}

// ---- the hub ----

Hub::Hub(unsigned ports, u8 fixed_ports, bool compound)
    : nports_(std::min(ports, kMaxPorts)),
      hub_descriptor_{9, 0x29, static_cast<u8>(std::min(ports, kMaxPorts)), static_cast<u8>(compound ? 0x0D : 0x09), 0x00, 50, 64,
                      fixed_ports, 0xFF} {}

void Hub::attach(unsigned port, Device* device) {
    if (port < 1 || port > nports_) return;
    Port& p = ports_[port - 1];
    p.device = device;
    if (p.status & kPortPower) {
        p.status = static_cast<u16>(device ? (p.status | kPortConnection) : (p.status & ~(kPortConnection | kPortEnable)));
        p.change |= kPortConnection;
    }
}

void Hub::bus_reset() {
    Device::bus_reset();
    for (Port& p : ports_) {
        p.status = 0;
        p.change = 0;
    }
}

Device* Hub::find(u8 address) {
    if (address == this->address()) return this;
    for (Port& p : ports_)
        if (p.device && (p.status & kPortEnable))
            if (Device* d = p.device->find(address)) return d;
    return nullptr;
}

std::span<const u8> Hub::device_descriptor() const { return kHubDevice; }
std::span<const u8> Hub::configuration_descriptor() const { return kHubConfiguration; }

bool Hub::class_request(const Setup& s, std::span<const u8>, std::vector<u8>& reply) {
    const unsigned port = s.index;
    const bool to_port = s.recipient() == 3;
    if (to_port && (port < 1 || port > nports_)) return false;
    switch (s.request) {
        case 0:  // GET_STATUS: the hub's (no local power loss, no over-current) or a port's status and change bits
            if (!to_port) {
                reply = {0, 0, 0, 0};
            } else {
                const Port& p = ports_[port - 1];
                reply = {static_cast<u8>(p.status), static_cast<u8>(p.status >> 8), static_cast<u8>(p.change), static_cast<u8>(p.change >> 8)};
            }
            return true;
        case 6:  // GET_DESCRIPTOR (hub)
            if ((s.value >> 8) != 0x29) return false;
            reply.assign(hub_descriptor_.begin(), hub_descriptor_.end());
            return true;
        case 3:    // SET_FEATURE
        case 1: {  // CLEAR_FEATURE
            if (!to_port) return true;  // C_HUB_LOCAL_POWER / C_HUB_OVER_CURRENT: nothing to clear
            Port& p = ports_[port - 1];
            const bool set = s.request == 3;
            switch (s.value) {
                case kFeatPortPower:
                    if (set) {
                        p.status |= kPortPower;
                        if (p.device) {
                            p.status |= kPortConnection;
                            p.change |= kPortConnection;
                        }
                    } else {
                        p.status = 0;
                    }
                    return true;
                case kFeatPortReset:
                    if (!set) return false;
                    if (p.device && (p.status & kPortConnection)) {
                        p.device->bus_reset();
                        p.status = static_cast<u16>((p.status | kPortEnable) & ~kPortReset);
                        p.change |= kPortReset;
                    }
                    return true;
                case kFeatPortEnable:
                    if (!set) p.status &= static_cast<u16>(~kPortEnable);
                    return true;
                case kFeatPortSuspend:
                    p.status = static_cast<u16>(set ? (p.status | kPortSuspend) : (p.status & ~kPortSuspend));
                    return true;
                case kFeatCConnection: p.change &= static_cast<u16>(~0x0001u); return true;
                case kFeatCEnable: p.change &= static_cast<u16>(~0x0002u); return true;
                case kFeatCSuspend: p.change &= static_cast<u16>(~0x0004u); return true;
                case kFeatCOverCurrent: p.change &= static_cast<u16>(~0x0008u); return true;
                case kFeatCReset: p.change &= static_cast<u16>(~0x0010u); return true;
                default: return false;
            }
        }
        default:
            return false;
    }
}

// The status-change endpoint: bit n set for each port n with a change, NAK while nothing changed.
Status Hub::endpoint_in(u8 endpoint, u8* data, u32 len, u32& actual) {
    if (endpoint != 1) return Status::Stall;
    u8 bitmap = 0;
    for (unsigned i = 0; i < nports_; ++i)
        if (ports_[i].change) bitmap = static_cast<u8>(bitmap | (1u << (i + 1)));
    if (!bitmap || len == 0) return Status::Nak;
    data[0] = bitmap;
    actual = 1;
    return Status::Ack;
}

}  // namespace mv::usb
