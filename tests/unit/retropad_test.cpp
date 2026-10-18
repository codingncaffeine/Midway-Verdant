// Every control of the Xbox pad reached from the RetroPad, alone: each RetroPad button and stick direction moves
// exactly one control of the console's pad, and exactly the byte (or bit) of the 20-byte input report the game
// reads for that control. A button that reached nothing, or two controls, fails here.
#include <gtest/gtest.h>

#include <array>

#include "libretro/libretro.h"
#include "libretro/retropad_map.h"
#include "usb/xbox_pad.h"

using namespace mv;
using P = usb::PadState;

namespace {

struct Expect {
    unsigned retropad_id;
    const char* xbox;
    unsigned report_byte;  // where the control lives in the report
    u8 report_mask;        // the bit, or 0xFF for an analog byte
};

// Report layout: byte 2 the digital buttons, bytes 4-11 the analog buttons A B X Y Black White LT RT,
// bytes 12-19 the sticks.
constexpr std::array<Expect, 16> kButtons{{
    {RETRO_DEVICE_ID_JOYPAD_B, "A", 4, 0xFF},
    {RETRO_DEVICE_ID_JOYPAD_A, "B", 5, 0xFF},
    {RETRO_DEVICE_ID_JOYPAD_Y, "X", 6, 0xFF},
    {RETRO_DEVICE_ID_JOYPAD_X, "Y", 7, 0xFF},
    {RETRO_DEVICE_ID_JOYPAD_R, "Black", 8, 0xFF},
    {RETRO_DEVICE_ID_JOYPAD_L, "White", 9, 0xFF},
    {RETRO_DEVICE_ID_JOYPAD_L2, "Left trigger", 10, 0xFF},
    {RETRO_DEVICE_ID_JOYPAD_R2, "Right trigger", 11, 0xFF},
    {RETRO_DEVICE_ID_JOYPAD_UP, "D-pad up", 2, 0x01},
    {RETRO_DEVICE_ID_JOYPAD_DOWN, "D-pad down", 2, 0x02},
    {RETRO_DEVICE_ID_JOYPAD_LEFT, "D-pad left", 2, 0x04},
    {RETRO_DEVICE_ID_JOYPAD_RIGHT, "D-pad right", 2, 0x08},
    {RETRO_DEVICE_ID_JOYPAD_START, "Start", 2, 0x10},
    {RETRO_DEVICE_ID_JOYPAD_SELECT, "Back", 2, 0x20},
    {RETRO_DEVICE_ID_JOYPAD_L3, "Left stick click", 2, 0x40},
    {RETRO_DEVICE_ID_JOYPAD_R3, "Right stick click", 2, 0x80},
}};

std::array<u8, 20> report_with(const libretro::InputState& in) { return usb::XidGamepad::report(libretro::map_retropad(in)); }

}  // namespace

TEST(RetroPad, EveryButtonReachesExactlyItsXboxControl) {
    const auto neutral = report_with([](unsigned, unsigned, unsigned) { return 0; });
    for (const Expect& e : kButtons) {
        const auto r = report_with([&](unsigned device, unsigned index, unsigned id) {
            return device == RETRO_DEVICE_JOYPAD && index == 0 && id == e.retropad_id ? 1 : 0;
        });
        for (unsigned b = 0; b < 20; ++b) {
            const u8 diff = static_cast<u8>(r[b] ^ neutral[b]);
            if (b == e.report_byte)
                EXPECT_EQ(diff, e.report_mask) << e.xbox << ": report byte " << b;
            else
                EXPECT_EQ(diff, 0u) << e.xbox << " also moved report byte " << b;
        }
    }
}

TEST(RetroPad, EveryStickDirectionReachesItsAxis) {
    struct Axis {
        unsigned index, id;
        int value;
        unsigned report_byte;  // the axis' low byte
        i16 expected;          // the pad's value (Y up is positive on the pad)
    };
    const Axis axes[] = {
        {RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X, 32767, 12, 32767},
        {RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y, -32767, 14, 32767},   // RetroPad up
        {RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X, -32767, 16, -32767},
        {RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y, 32767, 18, -32767},  // RetroPad down
    };
    for (const Axis& a : axes) {
        const auto r = report_with([&](unsigned device, unsigned index, unsigned id) {
            return device == RETRO_DEVICE_ANALOG && index == a.index && id == a.id ? a.value : 0;
        });
        for (unsigned b = 12; b < 20; b += 2) {
            const i16 v = static_cast<i16>(r[b] | (r[b + 1] << 8));
            EXPECT_EQ(v, b == a.report_byte ? a.expected : 0) << "axis at report byte " << b;
        }
        for (unsigned b = 0; b < 12; ++b) EXPECT_EQ(r[b], b == 1 ? 20u : 0u) << "a stick moved report byte " << b;
    }
}

TEST(RetroPad, AnalogButtonValuesPassThroughWhenTheFrontendReportsThem) {
    // A half-pulled right trigger (the frontend's analog button value) is half on the pad, not a full press.
    const auto s = libretro::map_retropad([](unsigned device, unsigned index, unsigned id) {
        if (device == RETRO_DEVICE_ANALOG && index == RETRO_DEVICE_INDEX_ANALOG_BUTTON && id == RETRO_DEVICE_ID_JOYPAD_R2) return 16384;
        return 0;
    });
    EXPECT_EQ(s.analog[P::RightTrigger], 128u);
    EXPECT_EQ(s.analog[P::LeftTrigger], 0u);
}
