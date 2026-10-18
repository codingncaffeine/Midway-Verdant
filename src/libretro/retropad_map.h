#pragma once
#include <functional>

#include "usb/xbox_pad.h"

// The RetroPad as the core reads it, onto the console's pad. Every control the Xbox pad has:
//   B, A, Y, X (bottom, right, left, top)  ->  A, B, X, Y      (analog buttons: full press, or the frontend's
//   L, R                                   ->  White, Black     analog button value when it reports one)
//   L2, R2                                 ->  left and right triggers
//   L3, R3                                 ->  the stick clicks
//   Select, Start                          ->  Back, Start
//   D-pad                                  ->  D-pad
//   left and right analog sticks           ->  the sticks (Y up is positive on the pad, down on the RetroPad)
namespace mv::libretro {

// The frontend's input_state for one port: (device, index, id) -> value.
using InputState = std::function<int(unsigned device, unsigned index, unsigned id)>;

usb::PadState map_retropad(const InputState& input);

}  // namespace mv::libretro
