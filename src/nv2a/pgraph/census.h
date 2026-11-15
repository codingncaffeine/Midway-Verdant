#pragma once
#include "core/types.h"

// The feature census (builds with MV_STATE_CENSUS=ON only; nothing here exists in a normal build): what a title
// asks of the GPU against what this code acts on. It records every method each graphics class receives with the
// distinct values written, whether the Kelvin state machine consumes the method itself, whether anything ever reads
// the stored value back (KelvinState::reg), and every register the guest touches. A method written with a value that
// is neither consumed nor read is a setting nothing here honors. The table goes to MV_CENSUS_FILE (or stderr) when
// the process ends; tools/census-report.sh joins the tables of many titles.
#ifdef MV_STATE_CENSUS
namespace mv::pgraph::census {

// A Kelvin (class 0x97) method as the state machine took it; `consumed`: the machine acts on it itself.
void kelvin_write(u32 method, u32 value, bool consumed);
// The stored value of a Kelvin method was read back (index = method / 4).
void kelvin_read(u32 index);
// A method of any other class.
void class_write(u32 cls, u32 method, u32 value);
// A guest access to a register of the GPU's window (the offset inside it).
void mmio(u32 offset, bool write, u32 value);
// Something true of one draw that the method table cannot show (two settings met in the same draw): counted per
// `what` and `detail`.
void draw_fact(const char* what, u32 detail);

}  // namespace mv::pgraph::census
#define MV_CENSUS_KELVIN_READ(index) ::mv::pgraph::census::kelvin_read(index)
#else
#define MV_CENSUS_KELVIN_READ(index) ((void)0)
#endif
