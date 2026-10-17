#pragma once
#include <array>
#include <string>

#include "core/types.h"

// The DSP56300 instruction decoder: a word is either one of the manual's non-parallel encodings (matched against
// the generated opcode table, most fixed bits first) or a parallel-move instruction (bits 23-8 the data bus move,
// bits 7-0 the Data ALU operation). Fields come out by their letter in the manual's pattern. Sources: the public
// DSP56300 Family Manual, chapter 12 (partial encodings, tables 12-13..12-20) and chapter 13 (opcodes).
namespace mv::dsp {

enum class Move : u8 { NotParallel, None, Immediate, Register, Update, X, Y, XR1, XR2, RY1, RY2, L, XY };

// The Data ALU operation of a parallel-move instruction (Tables 12-19 and 12-20).
enum class Alu : u8 {
    None, Mpy, Mpyr, Mac, Macr, Tfr, Addr, Tst, Cmp, Subr, Cmpm, Add, Rnd, Addl, Clr, Sub, Subl, Not, Asr, Lsr, Abs, Ror, Asl, Lsl, Neg, Rol,
    Adc, Sbc, Or, Eor, And
};

struct Decoded {
    u32 word{0};
    int pattern{-1};        // index into kOpcodePatterns, -1 for the parallel family
    Move move{Move::NotParallel};
    bool extension{false};  // the instruction has a second word
    // Parallel family: the ALU part.
    Alu alu{Alu::None};
    u8 alu_dest{0};         // 0 = A, 1 = B
    u8 alu_src{0};          // JJJ (Table 12-20) or QQQ (multiply operands)
    bool alu_negate{false}; // multiply: k2
    // Fields by letter ('a'..'z' then 'A'..'Z'), each the value of that letter's bits, MSB first.
    std::array<u32, 52> field{};
    std::array<u8, 52> width{};
    u32 f(char c) const { return field[index(c)]; }
    u8 w(char c) const { return width[index(c)]; }
    static std::size_t index(char c) { return static_cast<std::size_t>((c >= 'a' && c <= 'z') ? c - 'a' : 26 + (c - 'A')); }
};

Decoded decode(u32 word);
// The mnemonic of a decoded word ("JSR", "MOVE", "MAC", ...).
std::string mnemonic(const Decoded& d);
// A listing line: mnemonic, operands and the move, with the extension word when the instruction has one.
std::string disassemble(const Decoded& d, u32 extension);
const char* register_name(u32 code6);  // the six-bit on-chip register encoding (Table 12-13)
const char* alu_name(Alu a);

}  // namespace mv::dsp
