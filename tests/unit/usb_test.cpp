// The OHCI controller driven through its own data structures in RAM, the way a driver does: an endpoint
// descriptor with SETUP / data / status transfer descriptors, run by the 1 ms frame timer; the pad found through
// its internal hub and read through its interrupt endpoint.
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <vector>

#include "core/bus.h"
#include "core/clock.h"
#include "core/scheduler.h"
#include "usb/ohci.h"
#include "usb/xbox_pad.h"

using namespace mv;

namespace {

constexpr u32 kHcca = 0x10000, kEd = 0x11000, kTd = 0x11100, kBuf = 0x12000, kData = 0x13000;
constexpr u32 kControl = 0x04, kCommandStatus = 0x08, kHccaReg = 0x18, kControlHead = 0x20, kBulkHead = 0x28, kPort1 = 0x54;

struct Rig {
    Bus bus{1u << 20};
    Scheduler sched;
    bool irq{false};
    Ohci ohci{"ohci-test", bus, sched, 1, [this](bool level) { irq = level; }};
    usb::XboxPad pad;

    Rig() {
        ohci.attach(1, &pad.hub);
        ohci.mmio_write(kHccaReg, 4, kHcca);
        ohci.mmio_write(kControl, 4, (2u << 6) | (1u << 4));  // operational, control list enabled
    }
    void w32(u32 a, u32 v) { bus.write_ram(a, &v, 4); }
    u32 r32(u32 a) {
        u32 v = 0;
        bus.read_ram(a, &v, 4);
        return v;
    }
    void frames(unsigned n) { sched.advance_to(sched.now() + n * cycles_from_ns(1'000'000) + 1); }

    // One TD: direction (0 SETUP, 1 OUT, 2 IN), the buffer, the toggle from the TD.
    void td(u32 at, u32 dir, u32 toggle, u32 buf, u32 len, u32 next, bool rounding = true, u32 delay = 7) {
        w32(at + 0, (rounding ? 1u << 18 : 0u) | (dir << 19) | (delay << 21) | ((2u | toggle) << 24) | (0xFu << 28));
        w32(at + 4, len ? buf : 0);
        w32(at + 8, next);
        w32(at + 12, len ? buf + len - 1 : 0);
    }
    // A control transfer to function `fa`; returns the IN data stage, or the failing TD's condition code in `cc`.
    std::vector<u8> control(u8 fa, std::array<u8, 8> setup, u32 in_len, u32& cc) {
        bus.write_ram(kBuf, setup.data(), 8);
        std::vector<u8> zero(in_len + 1, 0xEE);
        bus.write_ram(kData, zero.data(), zero.size());
        const u32 td1 = kTd, td2 = kTd + 0x10, td3 = kTd + 0x20, tail = kTd + 0x30;
        td(td1, 0, 0, kBuf, 8, in_len ? td2 : td3);
        if (in_len) td(td2, 2, 1, kData, in_len, td3);
        td(td3, in_len ? 1 : 2, 1, 0, 0, tail);  // the status stage runs the other way
        w32(kEd + 0, fa | (8u << 16));
        w32(kEd + 4, tail);
        w32(kEd + 8, td1);
        w32(kEd + 12, 0);
        ohci.mmio_write(kControlHead, 4, kEd);
        ohci.mmio_write(kCommandStatus, 4, 2u);  // ControlListFilled
        frames(3);
        cc = 0;
        for (u32 t : {td1, td2, td3}) {
            if (t == td2 && !in_len) continue;
            const u32 c = r32(t) >> 28;
            if (c != 0) {
                cc = c;
                break;
            }
        }
        std::vector<u8> out(in_len);
        if (in_len) bus.read_ram(kData, out.data(), in_len);
        return out;
    }
};

}  // namespace

TEST(Ohci, EnumeratesThePadBehindItsHubAndReadsAReport) {
    Rig r;
    // The root port: connected, then reset (enables it and the hub answers at address 0).
    EXPECT_EQ(r.ohci.mmio_read(kPort1, 4) & 1u, 1u);
    r.ohci.mmio_write(kPort1, 4, 1u << 4);
    EXPECT_EQ(r.ohci.mmio_read(kPort1, 4) & 2u, 2u);

    u32 cc = 0;
    auto hub = r.control(0, {0x80, 6, 0, 1, 0, 0, 18, 0}, 18, cc);  // GET_DESCRIPTOR(device)
    ASSERT_EQ(cc, 0u);
    EXPECT_EQ(hub[4], 9u);  // hub class
    r.control(0, {0x00, 5, 1, 0, 0, 0, 0, 0}, 0, cc);  // SET_ADDRESS 1
    ASSERT_EQ(cc, 0u);
    // Control: nothing answers at address 0 now (the pad's hub port is still off).
    r.control(0, {0x80, 6, 0, 1, 0, 0, 18, 0}, 18, cc);
    EXPECT_EQ(cc, 5u) << "device not responding";

    r.control(1, {0x00, 9, 1, 0, 0, 0, 0, 0}, 0, cc);  // SET_CONFIGURATION 1
    ASSERT_EQ(cc, 0u);
    r.control(1, {0x23, 3, 8, 0, 1, 0, 0, 0}, 0, cc);  // SET_FEATURE(PORT_POWER) port 1
    auto st = r.control(1, {0xA3, 0, 0, 0, 1, 0, 4, 0}, 4, cc);  // GET_STATUS port 1
    ASSERT_EQ(cc, 0u);
    EXPECT_EQ(st[0] & 1u, 1u) << "the gamepad is connected on hub port 1";
    r.control(1, {0x23, 3, 4, 0, 1, 0, 0, 0}, 0, cc);  // SET_FEATURE(PORT_RESET) port 1

    auto pad = r.control(0, {0x80, 6, 0, 1, 0, 0, 18, 0}, 18, cc);
    ASSERT_EQ(cc, 0u);
    EXPECT_EQ(pad[8] | (pad[9] << 8), 0x045E);
    EXPECT_EQ(pad[10] | (pad[11] << 8), 0x0289);
    r.control(0, {0x00, 5, 2, 0, 0, 0, 0, 0}, 0, cc);  // SET_ADDRESS 2
    auto xid = r.control(2, {0xC1, 6, 0x00, 0x42, 0, 0, 16, 0}, 16, cc);  // the XID descriptor
    ASSERT_EQ(cc, 0u);
    EXPECT_EQ(xid[1], 0x42u);
    EXPECT_EQ(xid[4], 1u);   // a game controller
    EXPECT_EQ(xid[6], 20u);  // 20-byte input report

    // The input report through interrupt endpoint 1: A pressed fully, the left stick hard right.
    usb::PadState s;
    s.analog[usb::PadState::A] = 0xFF;
    s.left_x = 32767;
    r.pad.gamepad.set_state(s);
    r.w32(kEd + 0x40, 2u | (1u << 7) | (2u << 11) | (32u << 16));  // function 2, endpoint 1, IN
    r.td(kTd + 0x40, 2, 0, kData, 20, kTd + 0x50);
    r.w32(kEd + 0x44, kTd + 0x50);
    r.w32(kEd + 0x48, kTd + 0x40);
    r.w32(kEd + 0x4C, 0);
    r.ohci.mmio_write(kBulkHead, 4, kEd + 0x40);
    r.ohci.mmio_write(kControl, 4, (2u << 6) | (1u << 5));  // bulk list (the same machinery as interrupt)
    r.ohci.mmio_write(kCommandStatus, 4, 4u);
    r.frames(2);
    EXPECT_EQ(r.r32(kTd + 0x40) >> 28, 0u);
    std::array<u8, 20> report{};
    r.bus.read_ram(kData, report.data(), 20);
    EXPECT_EQ(report[1], 20u);
    EXPECT_EQ(report[4], 0xFFu);
    EXPECT_EQ(report[12] | (report[13] << 8), 0x7FFF);

    // Control: with nothing new the endpoint NAKs and the next TD stays on the ED.
    r.td(kTd + 0x50, 2, 1, kData, 20, kTd + 0x60);
    r.w32(kEd + 0x44, kTd + 0x60);
    r.ohci.mmio_write(kCommandStatus, 4, 4u);
    r.frames(2);
    EXPECT_EQ(r.r32(kTd + 0x50) >> 28, 0xFu) << "not accessed: NAKed and retried";
}

// XAPI reads the configuration descriptor with a long wLength, the data stage in 8-byte TDs without
// bufferRounding, and only the status TD asks for an interrupt. The device's short packet ends the data stage with
// DataUnderrun and halts the ED, so the status TD never runs: the errored TD must post the done queue itself.
TEST(Ohci, AnErroredTransferPostsTheDoneQueueAtOnce) {
    Rig r;
    r.ohci.mmio_write(kPort1, 4, 1u << 4);
    const std::array<u8, 8> setup{0x80, 6, 0, 2, 0, 0, 0x30, 0};
    r.bus.write_ram(kBuf, setup.data(), 8);
    const u32 tail = kTd + 0x100;
    r.td(kTd, 0, 0, kBuf, 8, kTd + 0x10, true);
    for (u32 i = 0; i < 6; ++i) r.td(kTd + 0x10 * (i + 1), 2, (i + 1) & 1u, kData + 8 * i, 8, kTd + 0x10 * (i + 2), false);
    r.td(kTd + 0x70, 1, 1, 0, 0, tail, true, 0);  // the status stage: the only TD that interrupts
    r.w32(kEd + 0, 0u | (8u << 16));
    r.w32(kEd + 4, tail);
    r.w32(kEd + 8, kTd);
    r.w32(kEd + 12, 0);
    r.ohci.mmio_write(kControlHead, 4, kEd);
    r.ohci.mmio_write(kCommandStatus, 4, 2u);
    r.frames(3);
    EXPECT_EQ(r.r32(kTd + 0x40) >> 28, 9u) << "the hub's 25-byte descriptor ends in the fourth data TD: DataUnderrun";
    EXPECT_EQ(r.r32(kEd + 8) & 1u, 1u) << "the ED halted";
    EXPECT_EQ(r.r32(kTd + 0x70) >> 28, 0xFu) << "the status TD did not run";
    EXPECT_EQ(r.r32(kHcca + 0x84) & ~0xFu, kTd + 0x40) << "the errored TD heads the done queue in the HCCA";
    EXPECT_TRUE(r.ohci.mmio_read(0x0C, 4) & 2u) << "WritebackDoneHead";
}

TEST(UsbDevice, AReplyThatFillsItsLastPacketEndsWithAZeroLengthPacket) {
    usb::XidGamepad pad;
    u8 setup[8] = {0x80, 6, 0, 2, 0, 0, 0x50, 0};  // GET_DESCRIPTOR(configuration), wLength 80; the gamepad's is 32
    u32 actual = 0;
    ASSERT_EQ(pad.transact(usb::Pid::Setup, 0, setup, 8, actual), usb::Status::Ack);
    u8 buf[8];
    u32 total = 0;
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(pad.transact(usb::Pid::In, 0, buf, 8, actual), usb::Status::Ack);
        total += actual;
    }
    EXPECT_EQ(total, 32u);
    ASSERT_EQ(pad.transact(usb::Pid::In, 0, buf, 8, actual), usb::Status::Ack) << "not a stall";
    EXPECT_EQ(actual, 0u) << "the zero-length packet ends the data stage";
    EXPECT_EQ(pad.transact(usb::Pid::Out, 0, buf, 0, actual), usb::Status::Ack) << "then the status stage";
    // A reply of exactly wLength ends without one: the next IN is out of turn.
    u8 setup8[8] = {0x80, 6, 0, 1, 0, 0, 8, 0};
    ASSERT_EQ(pad.transact(usb::Pid::Setup, 0, setup8, 8, actual), usb::Status::Ack);
    ASSERT_EQ(pad.transact(usb::Pid::In, 0, buf, 8, actual), usb::Status::Ack);
    EXPECT_EQ(actual, 8u);
    EXPECT_EQ(pad.transact(usb::Pid::Out, 0, buf, 0, actual), usb::Status::Ack);
}

// The 1.0-1.5 front panel: a four-port, non-compound hub with the controllers on its ports 3, 4, 1, 2.
TEST(UsbHub, TheFrontPanelHubHasFourPortsAndFrontPortOneIsItsPortThree) {
    usb::FrontPanel panel;
    usb::XboxPad pad;
    panel.hub.attach(usb::FrontPanel::kHubPortOfFrontPort[0], &pad.hub);
    u32 actual = 0;
    u8 get_hub[8] = {0xA0, 6, 0, 0x29, 0, 0, 9, 0};
    ASSERT_EQ(panel.hub.transact(usb::Pid::Setup, 0, get_hub, 8, actual), usb::Status::Ack);
    u8 d[9] = {};
    ASSERT_EQ(panel.hub.transact(usb::Pid::In, 0, d, 8, actual), usb::Status::Ack);
    ASSERT_EQ(panel.hub.transact(usb::Pid::In, 0, d + 8, 1, actual), usb::Status::Ack);
    EXPECT_EQ(d[2], 4u) << "four downstream ports";
    EXPECT_EQ(d[3] & 4u, 0u) << "not part of a compound device";
    EXPECT_EQ(d[7], 0u) << "every port removable";
    for (unsigned port = 1; port <= 4; ++port) {
        u8 power[8] = {0x23, 3, 8, 0, static_cast<u8>(port), 0, 0, 0};
        panel.hub.transact(usb::Pid::Setup, 0, power, 8, actual);
        panel.hub.transact(usb::Pid::In, 0, d, 0, actual);
        u8 status[8] = {0xA3, 0, 0, 0, static_cast<u8>(port), 0, 4, 0};
        panel.hub.transact(usb::Pid::Setup, 0, status, 8, actual);
        u8 st[4] = {};
        ASSERT_EQ(panel.hub.transact(usb::Pid::In, 0, st, 4, actual), usb::Status::Ack);
        EXPECT_EQ(st[0] & 1u, port == 3 ? 1u : 0u) << "port " << port;
        panel.hub.transact(usb::Pid::Out, 0, d, 0, actual);
    }
}
