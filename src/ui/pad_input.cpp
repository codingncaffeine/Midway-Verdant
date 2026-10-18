#include "ui/pad_input.h"

#include <algorithm>
#include <cstdio>

namespace mv::ui {

using usb::PadState;

PadInput::~PadInput() {
    if (gamepad_) SDL_CloseGamepad(gamepad_);
}

void PadInput::attach(usb::XidGamepad& pad) {
    pad.set_rumble([this](u16 left, u16 right) { rumble_.store((static_cast<u32>(left) << 16) | right); });
}

void PadInput::handle_event(const SDL_Event& e) {
    if (e.type == SDL_EVENT_GAMEPAD_ADDED && !gamepad_) {
        gamepad_ = SDL_OpenGamepad(e.gdevice.which);
        if (gamepad_) std::fprintf(stderr, "input: %s drives front port 1\n", SDL_GetGamepadName(gamepad_));
    } else if (e.type == SDL_EVENT_GAMEPAD_REMOVED && gamepad_ && SDL_GetGamepadID(gamepad_) == e.gdevice.which) {
        SDL_CloseGamepad(gamepad_);
        gamepad_ = nullptr;
        std::fprintf(stderr, "input: gamepad removed; the keyboard drives front port 1\n");
    }
}

PadState PadInput::keyboard_state() const {
    const bool* k = SDL_GetKeyboardState(nullptr);
    PadState s;
    auto button = [&](SDL_Scancode sc, u16 bit) {
        if (k[sc]) s.buttons = static_cast<u16>(s.buttons | bit);
    };
    button(SDL_SCANCODE_UP, PadState::Up);
    button(SDL_SCANCODE_DOWN, PadState::Down);
    button(SDL_SCANCODE_LEFT, PadState::Left);
    button(SDL_SCANCODE_RIGHT, PadState::Right);
    button(SDL_SCANCODE_RETURN, PadState::Start);
    button(SDL_SCANCODE_BACKSPACE, PadState::Back);
    auto analog = [&](SDL_Scancode sc, PadState::Analog a) {
        if (k[sc]) s.analog[a] = 0xFF;
    };
    analog(SDL_SCANCODE_Z, PadState::A);
    analog(SDL_SCANCODE_X, PadState::B);
    analog(SDL_SCANCODE_C, PadState::X);
    analog(SDL_SCANCODE_V, PadState::Y);
    analog(SDL_SCANCODE_1, PadState::White);
    analog(SDL_SCANCODE_2, PadState::Black);
    analog(SDL_SCANCODE_Q, PadState::LeftTrigger);
    analog(SDL_SCANCODE_E, PadState::RightTrigger);
    auto axis = [&](SDL_Scancode neg, SDL_Scancode pos) {
        return static_cast<i16>((k[pos] ? 32767 : 0) - (k[neg] ? 32767 : 0));
    };
    s.left_x = axis(SDL_SCANCODE_A, SDL_SCANCODE_D);
    s.left_y = axis(SDL_SCANCODE_S, SDL_SCANCODE_W);
    s.right_x = axis(SDL_SCANCODE_J, SDL_SCANCODE_L);
    s.right_y = axis(SDL_SCANCODE_K, SDL_SCANCODE_I);
    return s;
}

PadState PadInput::gamepad_state() const {
    PadState s;
    if (!gamepad_) return s;
    auto button = [&](SDL_GamepadButton b, u16 bit) {
        if (SDL_GetGamepadButton(gamepad_, b)) s.buttons = static_cast<u16>(s.buttons | bit);
    };
    button(SDL_GAMEPAD_BUTTON_DPAD_UP, PadState::Up);
    button(SDL_GAMEPAD_BUTTON_DPAD_DOWN, PadState::Down);
    button(SDL_GAMEPAD_BUTTON_DPAD_LEFT, PadState::Left);
    button(SDL_GAMEPAD_BUTTON_DPAD_RIGHT, PadState::Right);
    button(SDL_GAMEPAD_BUTTON_START, PadState::Start);
    button(SDL_GAMEPAD_BUTTON_BACK, PadState::Back);
    button(SDL_GAMEPAD_BUTTON_LEFT_STICK, PadState::LeftThumb);
    button(SDL_GAMEPAD_BUTTON_RIGHT_STICK, PadState::RightThumb);
    auto analog = [&](SDL_GamepadButton b, PadState::Analog a) {
        if (SDL_GetGamepadButton(gamepad_, b)) s.analog[a] = 0xFF;
    };
    analog(SDL_GAMEPAD_BUTTON_SOUTH, PadState::A);
    analog(SDL_GAMEPAD_BUTTON_EAST, PadState::B);
    analog(SDL_GAMEPAD_BUTTON_WEST, PadState::X);
    analog(SDL_GAMEPAD_BUTTON_NORTH, PadState::Y);
    analog(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, PadState::White);
    analog(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, PadState::Black);
    auto trigger = [&](SDL_GamepadAxis a) { return static_cast<u8>(std::clamp(SDL_GetGamepadAxis(gamepad_, a) / 128, 0, 255)); };
    s.analog[PadState::LeftTrigger] = trigger(SDL_GAMEPAD_AXIS_LEFT_TRIGGER);
    s.analog[PadState::RightTrigger] = trigger(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER);
    // SDL's Y axes point down; the pad's point up.
    auto invert = [](i16 v) { return static_cast<i16>(v == -32768 ? 32767 : -v); };
    s.left_x = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_LEFTX);
    s.left_y = invert(SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_LEFTY));
    s.right_x = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_RIGHTX);
    s.right_y = invert(SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_RIGHTY));
    return s;
}

void PadInput::update(usb::XidGamepad& pad) {
    // The gamepad and the keyboard together: buttons OR'd, the stronger of each analog value.
    const PadState g = gamepad_state(), k = keyboard_state();
    PadState s;
    s.buttons = static_cast<u16>(g.buttons | k.buttons);
    for (std::size_t i = 0; i < s.analog.size(); ++i) s.analog[i] = std::max(g.analog[i], k.analog[i]);
    auto stronger = [](i16 a, i16 b) { return std::abs(static_cast<int>(a)) >= std::abs(static_cast<int>(b)) ? a : b; };
    s.left_x = stronger(g.left_x, k.left_x);
    s.left_y = stronger(g.left_y, k.left_y);
    s.right_x = stronger(g.right_x, k.right_x);
    s.right_y = stronger(g.right_y, k.right_y);
    if (!sent_ || !(s == last_)) {
        pad.set_state(s);
        last_ = s;
        sent_ = true;
    }
    const u32 r = rumble_.load();
    if (gamepad_ && r != rumble_applied_) {
        SDL_RumbleGamepad(gamepad_, static_cast<u16>(r >> 16), static_cast<u16>(r), 0);
        rumble_applied_ = r;
    }
}

}  // namespace mv::ui
