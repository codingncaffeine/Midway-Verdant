#pragma once
#include <SDL3/SDL.h>

#include <atomic>

#include "usb/xbox_pad.h"

// The host's controls mapped onto the console's pad on front port 1: the first SDL gamepad connected (A/B/X/Y,
// LB = White, RB = Black, the triggers and sticks, D-pad, Start, Back, stick clicks), and the keyboard as a
// fallback (arrows = D-pad, WASD = left stick, IJKL = right stick, Z X C V = A B X Y, Q E = triggers,
// 1 2 = White Black, Enter = Start, Backspace = Back). The console's rumble goes back to the gamepad.
namespace mv::ui {

class PadInput {
public:
    ~PadInput();
    // Wires the rumble of the console's pad back to the host gamepad.
    void attach(usb::XidGamepad& pad);
    void handle_event(const SDL_Event& e);
    // The mapped state, sent to the console's pad when it changes; the rumble applied. Call once per loop.
    void update(usb::XidGamepad& pad);

private:
    usb::PadState keyboard_state() const;
    usb::PadState gamepad_state() const;

    SDL_Gamepad* gamepad_{nullptr};
    usb::PadState last_{};
    bool sent_{false};
    std::atomic<u32> rumble_{0};  // left << 16 | right, from the console
    u32 rumble_applied_{0};
};

}  // namespace mv::ui
