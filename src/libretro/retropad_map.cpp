#include "libretro/retropad_map.h"

#include <algorithm>

#include "libretro/libretro.h"

namespace mv::libretro {

usb::PadState map_retropad(const InputState& input) {
    using P = usb::PadState;
    P s;
    auto pressed = [&](unsigned id) { return input(RETRO_DEVICE_JOYPAD, 0, id) != 0; };
    auto bit = [&](unsigned id, u16 b) {
        if (pressed(id)) s.buttons = static_cast<u16>(s.buttons | b);
    };
    bit(RETRO_DEVICE_ID_JOYPAD_UP, P::Up);
    bit(RETRO_DEVICE_ID_JOYPAD_DOWN, P::Down);
    bit(RETRO_DEVICE_ID_JOYPAD_LEFT, P::Left);
    bit(RETRO_DEVICE_ID_JOYPAD_RIGHT, P::Right);
    bit(RETRO_DEVICE_ID_JOYPAD_START, P::Start);
    bit(RETRO_DEVICE_ID_JOYPAD_SELECT, P::Back);
    bit(RETRO_DEVICE_ID_JOYPAD_L3, P::LeftThumb);
    bit(RETRO_DEVICE_ID_JOYPAD_R3, P::RightThumb);
    // An analog button: the frontend's analog value when it reports one (0..32767), else the digital press.
    auto analog_button = [&](unsigned id) -> u8 {
        const int a = input(RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_BUTTON, id);
        if (a > 0) return static_cast<u8>(std::min(255, a / 128));
        return pressed(id) ? 0xFF : 0x00;
    };
    s.analog[P::A] = analog_button(RETRO_DEVICE_ID_JOYPAD_B);
    s.analog[P::B] = analog_button(RETRO_DEVICE_ID_JOYPAD_A);
    s.analog[P::X] = analog_button(RETRO_DEVICE_ID_JOYPAD_Y);
    s.analog[P::Y] = analog_button(RETRO_DEVICE_ID_JOYPAD_X);
    s.analog[P::White] = analog_button(RETRO_DEVICE_ID_JOYPAD_L);
    s.analog[P::Black] = analog_button(RETRO_DEVICE_ID_JOYPAD_R);
    s.analog[P::LeftTrigger] = analog_button(RETRO_DEVICE_ID_JOYPAD_L2);
    s.analog[P::RightTrigger] = analog_button(RETRO_DEVICE_ID_JOYPAD_R2);
    auto axis = [&](unsigned index, unsigned id, bool invert) {
        const int v = input(RETRO_DEVICE_ANALOG, index, id);
        return static_cast<i16>(std::clamp(invert ? -v : v, -32768, 32767));
    };
    s.left_x = axis(RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X, false);
    s.left_y = axis(RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y, true);
    s.right_x = axis(RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X, false);
    s.right_y = axis(RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y, true);
    return s;
}

}  // namespace mv::libretro
