#pragma once
#include <array>
#include <span>
#include <vector>

#include "core/types.h"

// A full-speed USB 1.1 function as the host controller sees it: tokens addressed to it, answered with ACK (and the
// bytes moved), NAK or STALL. The default control pipe (endpoint 0) runs the standard SETUP / data / status
// sequence here and answers the chapter 9 standard requests from the descriptors a device supplies; class and
// vendor requests and the other endpoints belong to the device. Written from the USB 1.1 specification
// (chapters 8, 9) — see local-notes/hw/usb.md.
namespace mv::usb {

enum class Pid : u8 { Setup, Out, In };
enum class Status : u8 { Ack, Nak, Stall };

struct Setup {
    u8 request_type{0}, request{0};
    u16 value{0}, index{0}, length{0};
    bool device_to_host() const { return (request_type & 0x80) != 0; }
    u8 type() const { return static_cast<u8>((request_type >> 5) & 3); }  // 0 standard, 1 class, 2 vendor
    u8 recipient() const { return static_cast<u8>(request_type & 0x1F); }  // 0 device, 1 interface, 2 endpoint, 3 other
};

class Device {
public:
    virtual ~Device() = default;

    // USB bus reset (a port reset upstream): address 0, unconfigured, the control pipe idle.
    virtual void bus_reset();
    u8 address() const { return address_; }
    u8 configuration() const { return configuration_; }

    // A transaction addressed to this device: SETUP and OUT carry len bytes in data; IN may fill up to len bytes.
    // `actual` receives the bytes moved on an ACK.
    Status transact(Pid pid, u8 endpoint, u8* data, u32 len, u32& actual);

    // A hub routes a token for another address downstream; a plain function has nothing below it.
    virtual Device* find(u8 address) { return address == address_ ? this : nullptr; }

protected:
    virtual std::span<const u8> device_descriptor() const = 0;
    virtual std::span<const u8> configuration_descriptor() const = 0;
    // Class and vendor control requests. Device-to-host: fill `reply` (truncated to wLength by the caller).
    // Host-to-device: `data` holds the data stage. Return false to STALL.
    virtual bool class_request(const Setup& s, std::span<const u8> data, std::vector<u8>& reply);
    // Endpoints other than 0.
    virtual Status endpoint_in(u8 endpoint, u8* data, u32 len, u32& actual);
    virtual Status endpoint_out(u8 endpoint, const u8* data, u32 len);
    virtual void configured(u8 /*configuration*/) {}

private:
    bool standard_request(const Setup& s, std::vector<u8>& reply);
    void start_setup(const u8* packet);
    void finish_status_stage();

    enum class Stage : u8 { Idle, DataIn, DataOut, StatusIn, StatusOut, Stalled };
    u8 address_{0}, pending_address_{0}, configuration_{0};
    bool address_pending_{false};
    Stage stage_{Stage::Idle};
    Setup setup_{};
    std::vector<u8> reply_;     // the IN data stage still to send
    std::size_t reply_pos_{0};
    std::vector<u8> out_data_;  // the OUT data stage received so far
};

}  // namespace mv::usb
