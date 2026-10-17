#include "apu/dsp/decode.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <vector>

#include "apu/dsp/opcode_table.h"

namespace mv::dsp {

namespace {

struct Compiled {
    u32 fixed_mask{0}, fixed_value{0};
    int fixed_bits{0};
    int index{0};
};

const std::vector<Compiled>& compiled() {
    static const std::vector<Compiled> table = [] {
        std::vector<Compiled> v;
        const int n = static_cast<int>(sizeof(kOpcodePatterns) / sizeof(kOpcodePatterns[0]));
        for (int i = 0; i < n; ++i) {
            Compiled c;
            c.index = i;
            const char* p = kOpcodePatterns[i].pattern;
            for (int bit = 0; bit < 24; ++bit) {
                const char ch = p[bit];
                const u32 mask = 1u << (23 - bit);
                if (ch == '0' || ch == '1') {
                    c.fixed_mask |= mask;
                    if (ch == '1') c.fixed_value |= mask;
                    ++c.fixed_bits;
                }
            }
            v.push_back(c);
        }
        // The most constrained pattern wins when several match; the table order breaks ties.
        std::stable_sort(v.begin(), v.end(), [](const Compiled& a, const Compiled& b) { return a.fixed_bits > b.fixed_bits; });
        return v;
    }();
    return table;
}

void extract_fields(Decoded& d, const char* pattern, u32 word) {
    for (int bit = 0; bit < 24; ++bit) {
        const char ch = pattern[bit];
        if (ch == '0' || ch == '1') continue;
        const std::size_t i = Decoded::index(ch);
        d.field[i] = (d.field[i] << 1) | ((word >> (23 - bit)) & 1u);
        ++d.width[i];
    }
}

bool ea_needs_extension(u32 mmmrrr) { return mmmrrr == 0b110000 || mmmrrr == 0b110100; }

// The data bus move field of a parallel instruction, from the formats on the MOVE pages (13-111..13-129).
void decode_move(Decoded& d, u32 w) {
    const u32 top = w >> 20;  // bits 23-20
    if (w >> 23) {  // 1wmmeeff Wrr MMRRR: X:Y: double move
        d.move = Move::XY;
        extract_fields(d, "1wmmeeffWrrMMRRRoooooooo", w);
        return;
    }
    if (top >= 0b0100) {  // 01...: X:, Y: or L:
        const u32 ddddd = ((w >> 20) & 3u) << 3 | ((w >> 16) & 7u);
        if (ddddd < 4) {
            d.move = Move::L;
            extract_fields(d, (w >> 14) & 1u ? "0100L0LLW1MMMRRRoooooooo" : "0100L0LLW0aaaaaaoooooooo", w);
        } else {
            d.move = (w >> 19) & 1u ? Move::Y : Move::X;
            extract_fields(d, (w >> 14) & 1u ? "01ddsdddW1MMMRRRoooooooo" : "01ddsdddW0aaaaaaoooooooo", w);
        }
        if (((w >> 14) & 1u) && ea_needs_extension((w >> 8) & 0x3Fu)) d.extension = true;
        return;
    }
    if (top >= 0b0010) {  // 001.....: immediate short, register, update or no move
        const u32 ddddd = (w >> 16) & 0x1Fu;
        if (ddddd >= 4) {
            d.move = Move::Immediate;
            extract_fields(d, "001dddddiiiiiiiioooooooo", w);
        } else if (((w >> 8) & 0xFFFFu) == 0x2000u) {
            d.move = Move::None;
        } else if (((w >> 13) & 0x7FFu) == 0b00100000010u) {
            d.move = Move::Update;
            extract_fields(d, "00100000010MMRRRoooooooo", w);
        } else {
            d.move = Move::Register;
            extract_fields(d, "001000eeeeedddddoooooooo", w);
        }
        return;
    }
    if (top == 0b0001) {  // X:R or R:Y class I
        if ((w >> 14) & 1u) {
            d.move = Move::RY1;
            extract_fields(d, "0001deffW1MMMRRRoooooooo", w);
        } else {
            d.move = Move::XR1;
            extract_fields(d, "0001ffdFW0MMMRRRoooooooo", w);
        }
        if (ea_needs_extension((w >> 8) & 0x3Fu)) d.extension = true;
        return;
    }
    // 0000 100d: class II
    if ((w >> 14) & 1u) {
        d.move = Move::RY2;
        extract_fields(d, "0000100d10MMMRRRoooooooo", w);
    } else {
        d.move = Move::XR2;
        extract_fields(d, "0000100d00MMMRRRoooooooo", w);
    }
}

void decode_alu(Decoded& d, u32 op) {
    d.alu_dest = static_cast<u8>((op >> 3) & 1u);
    if (op & 0x80u) {
        d.alu_src = static_cast<u8>((op >> 4) & 7u);  // QQQ
        d.alu_negate = (op & 4u) != 0;
        const bool mac = (op & 2u) != 0, round = (op & 1u) != 0;
        d.alu = mac ? (round ? Alu::Macr : Alu::Mac) : (round ? Alu::Mpyr : Alu::Mpy);
        return;
    }
    const u32 jjj = (op >> 4) & 7u, kkk = op & 7u;
    d.alu_src = static_cast<u8>(jjj);
    static constexpr Alu kRow0[8] = {Alu::None, Alu::Tfr, Alu::Addr, Alu::Tst, Alu::None, Alu::Cmp, Alu::Subr, Alu::Cmpm};
    static constexpr Alu kRow1[8] = {Alu::Add, Alu::Rnd, Alu::Addl, Alu::Clr, Alu::Sub, Alu::None, Alu::Subl, Alu::Not};
    static constexpr Alu kRow2[8] = {Alu::Add, Alu::Adc, Alu::Asr, Alu::Lsr, Alu::Sub, Alu::Sbc, Alu::Abs, Alu::Ror};
    static constexpr Alu kRow3[8] = {Alu::Add, Alu::Adc, Alu::Asl, Alu::Lsl, Alu::Sub, Alu::Sbc, Alu::Neg, Alu::Rol};
    static constexpr Alu kRowS[8] = {Alu::Add, Alu::Tfr, Alu::Or, Alu::Eor, Alu::Sub, Alu::Cmp, Alu::And, Alu::Cmpm};
    switch (jjj) {
        case 0: d.alu = kRow0[kkk]; break;
        case 1: d.alu = kRow1[kkk]; break;
        case 2: d.alu = kRow2[kkk]; break;
        case 3: d.alu = kRow3[kkk]; break;
        default: d.alu = kRowS[kkk]; break;
    }
}

const char* ea_string(u32 mmmrrr, std::string& buf) {
    const u32 mode = mmmrrr >> 3, r = mmmrrr & 7u;
    switch (mode) {
        case 0: buf = std::format("(R{})-N{}", r, r); break;
        case 1: buf = std::format("(R{})+N{}", r, r); break;
        case 2: buf = std::format("(R{})-", r); break;
        case 3: buf = std::format("(R{})+", r); break;
        case 4: buf = std::format("(R{})", r); break;
        case 5: buf = std::format("(R{}+N{})", r, r); break;
        case 7: buf = std::format("-(R{})", r); break;
        default: buf = mmmrrr == 0b110000 ? "abs" : (mmmrrr == 0b110100 ? "#imm" : "?"); break;
    }
    return buf.c_str();
}

const char* reg5(u32 ddddd) {
    static const char* kNames[32] = {"?", "?", "?", "?", "X0", "X1", "Y0", "Y1", "A0", "B0", "A2", "B2", "A1", "B1", "A", "B",
                                     "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"};
    return kNames[ddddd & 31u];
}

const char* alu_operands(const Decoded& d) {
    // The source operand text for the parallel family.
    static const char* kMul[8] = {"X0,X0", "Y0,Y0", "X1,X0", "Y1,Y0", "X0,Y1", "Y0,X0", "X1,Y0", "Y1,X1"};
    static const char* kSrc[8] = {"", "", "X", "Y", "X0", "Y0", "X1", "Y1"};
    switch (d.alu) {
        case Alu::Mpy: case Alu::Mpyr: case Alu::Mac: case Alu::Macr: return kMul[d.alu_src & 7u];
        default: return kSrc[d.alu_src & 7u];
    }
}

}  // namespace

const char* alu_name(Alu a) {
    switch (a) {
        case Alu::None: return "MOVE";
        case Alu::Mpy: return "MPY";
        case Alu::Mpyr: return "MPYR";
        case Alu::Mac: return "MAC";
        case Alu::Macr: return "MACR";
        case Alu::Tfr: return "TFR";
        case Alu::Addr: return "ADDR";
        case Alu::Tst: return "TST";
        case Alu::Cmp: return "CMP";
        case Alu::Subr: return "SUBR";
        case Alu::Cmpm: return "CMPM";
        case Alu::Add: return "ADD";
        case Alu::Rnd: return "RND";
        case Alu::Addl: return "ADDL";
        case Alu::Clr: return "CLR";
        case Alu::Sub: return "SUB";
        case Alu::Subl: return "SUBL";
        case Alu::Not: return "NOT";
        case Alu::Asr: return "ASR";
        case Alu::Lsr: return "LSR";
        case Alu::Abs: return "ABS";
        case Alu::Ror: return "ROR";
        case Alu::Asl: return "ASL";
        case Alu::Lsl: return "LSL";
        case Alu::Neg: return "NEG";
        case Alu::Rol: return "ROL";
        case Alu::Adc: return "ADC";
        case Alu::Sbc: return "SBC";
        case Alu::Or: return "OR";
        case Alu::Eor: return "EOR";
        case Alu::And: return "AND";
    }
    return "?";
}

const char* register_name(u32 code6) {
    static const char* kNames[64] = {"?", "?", "?", "?", "X0", "X1", "Y0", "Y1", "A0", "B0", "A2", "B2", "A1", "B1", "A", "B",
                                     "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7",
                                     "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "?", "?", "EP", "?", "?", "?", "?", "?",
                                     "VBA", "SC", "?", "?", "?", "?", "?", "?", "SZ", "SR", "OMR", "SP", "SSH", "SSL", "LA", "LC"};
    return kNames[code6 & 63u];
}

Decoded decode(u32 word) {
    Decoded d;
    d.word = word & 0xFFFFFFu;
    for (const Compiled& c : compiled()) {
        if ((d.word & c.fixed_mask) != c.fixed_value) continue;
        d.pattern = c.index;
        const OpcodePattern& p = kOpcodePatterns[c.index];
        extract_fields(d, p.pattern, d.word);
        const std::string ops = p.operands;
        if (ops.find("xxxx") != std::string::npos || ops.find("expr") != std::string::npos || ops.find("label") != std::string::npos ||
            ops.find("FOREVER") != std::string::npos)
            d.extension = true;
        if (d.w('M') == 3 && d.w('R') == 3 && ea_needs_extension((d.f('M') << 3) | d.f('R'))) d.extension = true;
        return d;
    }
    // Not a non-parallel encoding: the parallel family (every word with bits 23-20 >= 0001, or class II 0000100d).
    if ((d.word >> 20) == 0 && ((d.word >> 17) & 0x7u) != 0b100) {
        d.pattern = -2;  // unknown
        return d;
    }
    d.move = Move::None;
    decode_move(d, d.word);
    decode_alu(d, d.word & 0xFFu);
    return d;
}

std::string mnemonic(const Decoded& d) {
    if (d.pattern >= 0) return kOpcodePatterns[d.pattern].mnemonic;
    if (d.pattern == -2) return "???";
    return alu_name(d.alu);
}

std::string disassemble(const Decoded& d, u32 extension) {
    std::string ea;
    if (d.pattern == -2) return std::format("dc ${:06x}", d.word);
    if (d.pattern >= 0) {
        const OpcodePattern& p = kOpcodePatterns[d.pattern];
        std::string s = std::format("{} {}", p.mnemonic, p.operands);
        // Append the field values the pattern carries, in letter order, so the listing is checkable against the manual.
        std::string fields;
        for (const char* q = p.pattern; *q; ++q) {
            const char ch = *q;
            if (ch == '0' || ch == '1') continue;
            if (fields.find(std::format(" {}=", ch)) != std::string::npos) continue;
            fields += std::format(" {}={:x}", ch, d.f(ch));
        }
        if (d.w('M') == 3 && d.w('R') == 3) fields += std::format(" ea={}", ea_string((d.f('M') << 3) | d.f('R'), ea));
        s += " |" + fields;
        if (d.extension) s += std::format(" ext=${:06x}", extension & 0xFFFFFFu);
        return s;
    }
    std::string s = alu_name(d.alu);
    const char* ops = alu_operands(d);
    if (d.alu != Alu::None) s += std::format(" {}{}{}", ops, *ops ? "," : "", d.alu_dest ? "B" : "A");
    if (d.alu == Alu::Mpy || d.alu == Alu::Mpyr || d.alu == Alu::Mac || d.alu == Alu::Macr) s = std::format("{} {}{},{}", alu_name(d.alu), d.alu_negate ? "-" : "", ops, d.alu_dest ? "B" : "A");
    const u32 w = d.word;
    switch (d.move) {
        case Move::None: break;
        case Move::Immediate: s += std::format("  #${:02x},{}", d.f('i'), reg5(d.f('d'))); break;
        case Move::Register: s += std::format("  {},{}", reg5(d.f('e')), reg5(d.f('d'))); break;
        case Move::Update: s += std::format("  {}", ea_string((d.f('M') << 3) | d.f('R') | 0, ea)); break;
        case Move::X:
        case Move::Y: {
            const u32 ddddd = (d.f('d') >> 3) << 3 | (d.f('d') & 7u);
            const char* space = d.move == Move::X ? "X" : "Y";
            const bool write = d.f('W');
            std::string target = ((w >> 14) & 1u) ? std::format("{}:{}", space, ea_string((d.f('M') << 3) | d.f('R'), ea)) : std::format("{}:${:02x}", space, d.f('a'));
            if (((w >> 14) & 1u) && ((d.f('M') << 3) | d.f('R')) == 0b110100) target = std::format("#${:06x}", extension & 0xFFFFFFu);
            else if (((w >> 14) & 1u) && ((d.f('M') << 3) | d.f('R')) == 0b110000) target = std::format("{}:${:06x}", space, extension & 0xFFFFFFu);
            s += write ? std::format("  {},{}", target, reg5(ddddd)) : std::format("  {},{}", reg5(ddddd), target);
            break;
        }
        case Move::XR1: {
            static const char* kFf[4] = {"X0", "X1", "A", "B"};
            const std::string target = ((d.f('M') << 3) | d.f('R')) == 0b110100 ? std::format("#${:06x}", extension & 0xFFFFFFu) : std::format("X:{}", ea_string((d.f('M') << 3) | d.f('R'), ea));
            s += d.f('W') ? std::format("  {},{} {},{}", target, kFf[d.f('f') & 3u], d.f('d') ? "B" : "A", d.f('F') ? "Y1" : "Y0")
                          : std::format("  {},{} {},{}", kFf[d.f('f') & 3u], target, d.f('d') ? "B" : "A", d.f('F') ? "Y1" : "Y0");
            break;
        }
        case Move::RY1: {
            static const char* kFf[4] = {"Y0", "Y1", "A", "B"};
            const std::string target = ((d.f('M') << 3) | d.f('R')) == 0b110100 ? std::format("#${:06x}", extension & 0xFFFFFFu) : std::format("Y:{}", ea_string((d.f('M') << 3) | d.f('R'), ea));
            s += d.f('W') ? std::format("  {},{} {},{}", d.f('d') ? "B" : "A", d.f('e') ? "X1" : "X0", target, kFf[d.f('f') & 3u])
                          : std::format("  {},{} {},{}", d.f('d') ? "B" : "A", d.f('e') ? "X1" : "X0", kFf[d.f('f') & 3u], target);
            break;
        }
        case Move::XR2: s += std::format("  {},X:{} X0,{}", d.f('d') ? "B" : "A", ea_string((d.f('M') << 3) | d.f('R'), ea), d.f('d') ? "B" : "A"); break;
        case Move::RY2: s += std::format("  Y0,{} {},Y:{}", d.f('d') ? "B" : "A", d.f('d') ? "B" : "A", ea_string((d.f('M') << 3) | d.f('R'), ea)); break;
        case Move::L: {
            static const char* kL[8] = {"A10", "B10", "X", "Y", "A", "B", "AB", "BA"};
            const std::string target = ((w >> 14) & 1u) ? std::format("L:{}", ea_string((d.f('M') << 3) | d.f('R'), ea)) : std::format("L:${:02x}", d.f('a'));
            s += d.f('W') ? std::format("  {},{}", target, kL[d.f('L') & 7u]) : std::format("  {},{}", kL[d.f('L') & 7u], target);
            break;
        }
        case Move::XY: {
            static const char* kE[4] = {"X0", "X1", "A", "B"};
            static const char* kF[4] = {"Y0", "Y1", "A", "B"};
            static const char* kMode[4] = {"", "+N", "-", "+"};
            const u32 rx = d.f('R'), mx = d.f('M'), ry = (rx < 4 ? 4 : 0) + d.f('r'), my = d.f('m');
            const std::string xa = std::format("X:(R{}){}", rx, kMode[mx & 3u]), ya = std::format("Y:(R{}){}", ry, kMode[my & 3u]);
            s += d.f('W') ? std::format("  {},{}", xa, kE[d.f('e') & 3u]) : std::format("  {},{}", kE[d.f('e') & 3u], xa);
            s += d.f('w') ? std::format(" {},{}", ya, kF[d.f('f') & 3u]) : std::format(" {},{}", kF[d.f('f') & 3u], ya);
            break;
        }
        case Move::NotParallel: break;
    }
    return s;
}

}  // namespace mv::dsp
