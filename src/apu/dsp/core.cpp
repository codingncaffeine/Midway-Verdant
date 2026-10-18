#include "apu/dsp/core.h"

#include <cstring>
#include <format>

#include "apu/dsp/opcode_table.h"

namespace mv::dsp {

namespace {

constexpr u32 kWord = 0xFFFFFFu;
constexpr i64 kAccMin = -(i64{1} << 55), kAccMax = (i64{1} << 55) - 1;

i64 sext24(u32 v) { return static_cast<i64>(static_cast<i32>(v << 8) >> 8); }
i32 sext_bits(u32 v, u32 bits) { return static_cast<i32>(v << (32 - bits)) >> (32 - bits); }

// The non-parallel operations, classified once per pattern.
enum class Op : u8 {
    Unknown, Nop, Rts, Rti, Enddo, Illegal, Debug, Stop, Wait, Reset, Trap,
    JmpEa, JmpShort, JsrEa, JsrShort, JccShort, JccEa, JsccShort, JsccEa,
    BraLong, BraShort, BraRn, BsrLong, BsrShort, BsrRn, BccLong, BccShort, BccRn, BsccLong, BsccShort, BsccRn,
    BitJumpEa, BitJumpAa, BitJumpPp, BitJumpQq, BitJumpReg,         // JCLR/JSET/JSCLR/JSSET
    BitBranchEa, BitBranchAa, BitBranchPp, BitBranchQq, BitBranchReg,  // BRCLR/BRSET/BSCLR/BSSET
    BitOpEa, BitOpAa, BitOpPp, BitOpQq, BitOpReg,                    // BCLR/BSET/BCHG/BTST
    DoEa, DoAa, DoImm, DoReg, DoForever, DorEa, DorAa, DorImm, DorReg, DorForever,
    RepEa, RepAa, RepImm, RepReg,
    MoveLongDispX, MoveLongDispY, MoveShortDispX, MoveShortDispY,
    MovecEa, MovecAa, MovecReg, MovecImm, MovemEa, MovemAa,
    MovepPpEa, MovepXqqEa, MovepYqqEa, MovepPpP, MovepQqP, MovepPpReg, MovepXqqReg, MovepYqqReg,
    LuaEa, LuaDisp,
    AluImmShort, AluImmLong, AndiOri,
    AslImm, AslReg, AsrImm, AsrReg, LslImm, LslReg, LsrImm, LsrReg,
    MulImm,  // MPYI MACI MPYRI MACRI
    Tcc1, Tcc2, Tcc3, Inc, Dec,
    Cmpu, Div, MulShift, MulMixed, Dmac, Norm, Normf, Clb, Extract, Insert, Merge, Lra, Brkcc, Trapcc, Nop2,
};

struct OpInfo {
    Op op;
    u8 variant;  // the sub-operation (bit op kind, ALU op, condition) where the mnemonic decides
};

OpInfo classify(const char* mnemonic, const char* operands) {
    const std::string m = mnemonic, o = operands;
    auto has = [&](const char* s) { return o.find(s) != std::string::npos; };
    if (m == "NOP") return {Op::Nop, 0};
    if (m == "RTS") return {Op::Rts, 0};
    if (m == "RTI") return {Op::Rti, 0};
    if (m == "ENDDO") return {Op::Enddo, 0};
    if (m == "ILLEGAL") return {Op::Illegal, 0};
    if (m == "DEBUG") return {Op::Debug, 0};
    if (m == "STOP") return {Op::Stop, 0};
    if (m == "WAIT") return {Op::Wait, 0};
    if (m == "RESET") return {Op::Reset, 0};
    if (m == "TRAP") return {Op::Trap, 0};
    if (m == "JMP") return {o == "ea" ? Op::JmpEa : Op::JmpShort, 0};
    if (m == "JSR") return {o == "ea" ? Op::JsrEa : Op::JsrShort, 0};
    if (m == "Jcc") return {o == "ea" ? Op::JccEa : Op::JccShort, 0};
    if (m == "JScc") return {o == "ea" ? Op::JsccEa : Op::JsccShort, 0};
    if (m == "BRA") return {o == "xxxx" ? Op::BraLong : (o == "Rn" ? Op::BraRn : Op::BraShort), 0};
    if (m == "BSR") return {o == "xxxx" ? Op::BsrLong : (o == "Rn" ? Op::BsrRn : Op::BsrShort), 0};
    if (m == "Bcc") return {o == "xxxx" ? Op::BccLong : (o == "Rn" ? Op::BccRn : Op::BccShort), 0};
    if (m == "BScc") return {o == "xxxx" ? Op::BsccLong : (o == "Rn" ? Op::BsccRn : Op::BsccShort), 0};
    auto form = [&](Op ea, Op aa, Op pp, Op qq, Op reg) {
        if (has(":ea")) return ea;
        if (has(":aa")) return aa;
        if (has(":pp")) return pp;
        if (has(":qq")) return qq;
        return reg;
    };
    // Bit instructions: variant = 0 clear, 1 set, 2 change, 3 test; jumps/branches: variant bit0 = set-condition, bit1 = subroutine.
    if (m == "BCLR") return {form(Op::BitOpEa, Op::BitOpAa, Op::BitOpPp, Op::BitOpQq, Op::BitOpReg), 0};
    if (m == "BSET") return {form(Op::BitOpEa, Op::BitOpAa, Op::BitOpPp, Op::BitOpQq, Op::BitOpReg), 1};
    if (m == "BCHG") return {form(Op::BitOpEa, Op::BitOpAa, Op::BitOpPp, Op::BitOpQq, Op::BitOpReg), 2};
    if (m == "BTST") return {form(Op::BitOpEa, Op::BitOpAa, Op::BitOpPp, Op::BitOpQq, Op::BitOpReg), 3};
    if (m == "JCLR") return {form(Op::BitJumpEa, Op::BitJumpAa, Op::BitJumpPp, Op::BitJumpQq, Op::BitJumpReg), 0};
    if (m == "JSET") return {form(Op::BitJumpEa, Op::BitJumpAa, Op::BitJumpPp, Op::BitJumpQq, Op::BitJumpReg), 1};
    if (m == "JSCLR") return {form(Op::BitJumpEa, Op::BitJumpAa, Op::BitJumpPp, Op::BitJumpQq, Op::BitJumpReg), 2};
    if (m == "JSSET") return {form(Op::BitJumpEa, Op::BitJumpAa, Op::BitJumpPp, Op::BitJumpQq, Op::BitJumpReg), 3};
    if (m == "BRCLR") return {form(Op::BitBranchEa, Op::BitBranchAa, Op::BitBranchPp, Op::BitBranchQq, Op::BitBranchReg), 0};
    if (m == "BRSET") return {form(Op::BitBranchEa, Op::BitBranchAa, Op::BitBranchPp, Op::BitBranchQq, Op::BitBranchReg), 1};
    if (m == "BSCLR") return {form(Op::BitBranchEa, Op::BitBranchAa, Op::BitBranchPp, Op::BitBranchQq, Op::BitBranchReg), 2};
    if (m == "BSSET") return {form(Op::BitBranchEa, Op::BitBranchAa, Op::BitBranchPp, Op::BitBranchQq, Op::BitBranchReg), 3};
    if (m == "DO") {
        if (has("FOREVER")) return {Op::DoForever, 0};
        if (has(":ea")) return {Op::DoEa, 0};
        if (has(":aa")) return {Op::DoAa, 0};
        if (has("#xxx")) return {Op::DoImm, 0};
        return {Op::DoReg, 0};
    }
    if (m == "DOR") {
        if (has("FOREVER")) return {Op::DorForever, 0};
        if (has(":ea")) return {Op::DorEa, 0};
        if (has(":aa")) return {Op::DorAa, 0};
        if (has("#xxx")) return {Op::DorImm, 0};
        return {Op::DorReg, 0};
    }
    if (m == "REP") {
        if (has(":ea")) return {Op::RepEa, 0};
        if (has(":aa")) return {Op::RepAa, 0};
        if (has("#xxx")) return {Op::RepImm, 0};
        return {Op::RepReg, 0};
    }
    if (m == "MOVE") {
        if (has("X:(Rn + xxxx)")) return {Op::MoveLongDispX, 0};
        if (has("Y:(Rn + xxxx)")) return {Op::MoveLongDispY, 0};
        if (has("X:(Rn + xxx)")) return {Op::MoveShortDispX, 0};
        if (has("Y:(Rn + xxx)")) return {Op::MoveShortDispY, 0};
    }
    if (m == "MOVE(C)") {
        if (has(":ea")) return {Op::MovecEa, 0};
        if (has(":aa")) return {Op::MovecAa, 0};
        if (has("#xx")) return {Op::MovecImm, 0};
        return {Op::MovecReg, 0};
    }
    if (m == "MOVE(M)") return {has(":ea") ? Op::MovemEa : Op::MovemAa, 0};
    if (m == "MOVEP") {
        if (o == "[X or Y]:pp,[X or Y]:ea") return {Op::MovepPpEa, 0};
        if (o == "X:qq,[X or Y]:ea") return {Op::MovepXqqEa, 0};
        if (o == "Y:qq,[X or Y]:ea") return {Op::MovepYqqEa, 0};
        if (o == "[X or Y]:pp,P:ea") return {Op::MovepPpP, 0};
        if (o == "[X or Y]:qq,P:ea") return {Op::MovepQqP, 0};
        if (o == "[X or Y]:pp,D") return {Op::MovepPpReg, 0};
        if (o == "X:qq,D") return {Op::MovepXqqReg, 0};
        if (o == "Y:qq,D") return {Op::MovepYqqReg, 0};
    }
    if (m == "LUA/LEA") return {has("(Rn") ? Op::LuaDisp : Op::LuaEa, 0};
    // ALU immediates: variant = the kkk-style operation code used by alu_immediate: 0 ADD 1 SUB 2 CMP 3 AND 4 OR 5 EOR
    static const char* kImm[6] = {"ADD", "SUB", "CMP", "AND", "OR", "EOR"};
    for (u8 i = 0; i < 6; ++i)
        if (m == kImm[i] && has("#x")) return {has("#xxxx") ? Op::AluImmLong : Op::AluImmShort, i};
    if (m == "AND(I)") return {Op::AndiOri, 0};
    if (m == "OR(I)") return {Op::AndiOri, 1};
    if (m == "ASL") return {has("#ii") ? Op::AslImm : Op::AslReg, 0};
    if (m == "ASR") return {has("#ii") ? Op::AsrImm : Op::AsrReg, 0};
    if (m == "LSL") return {has("#ii") ? Op::LslImm : Op::LslReg, 0};
    if (m == "LSR") return {has("#ii") ? Op::LsrImm : Op::LsrReg, 0};
    if (m == "MPYI") return {Op::MulImm, 0};
    if (m == "MACI") return {Op::MulImm, 1};
    if (m == "MPYRI") return {Op::MulImm, 2};
    if (m == "MACRI") return {Op::MulImm, 3};
    if (m == "Tcc") return {o == "S1,D1" ? Op::Tcc1 : (o == "S2,D2" ? Op::Tcc3 : Op::Tcc2), 0};
    if (m == "INC") return {Op::Inc, 0};
    if (m == "CMPU") return {Op::Cmpu, 0};
    if (m == "DIV") return {Op::Div, 0};
    if (m == "MPY" && has("#n")) return {Op::MulShift, 0};
    if (m == "MAC" && has("#n")) return {Op::MulShift, 1};
    if (m == "MPYR" && has("#n")) return {Op::MulShift, 2};
    if (m == "MACR" && has("#n")) return {Op::MulShift, 3};
    if (m == "MPY" && has("uu")) return {Op::MulMixed, 0};
    if (m == "MACuu") return {Op::MulMixed, 1};
    if (m == "DMAC") return {Op::Dmac, 0};
    if (m == "NORM") return {Op::Norm, 0};
    if (m == "NORMF") return {Op::Normf, 0};
    if (m == "CLB") return {Op::Clb, 0};
    if (m == "EXTRACT") return {Op::Extract, static_cast<u8>(has("#CO") ? 2 : 0)};
    if (m == "EXTRACTU") return {Op::Extract, static_cast<u8>(has("#CO") ? 3 : 1)};
    if (m == "INSERT") return {Op::Insert, static_cast<u8>(has("#CO") ? 2 : 0)};
    if (m == "MERGE") return {Op::Merge, 0};
    if (m == "LRA") return {Op::Lra, static_cast<u8>(has("xxxx") ? 1 : 0)};
    if (m == "BRKcc") return {Op::Brkcc, 0};
    if (m == "TRAPcc") return {Op::Trapcc, 0};
    if (m == "DEBUGcc" || m == "PFLUSH" || m == "PFLUSHUN" || m == "PFREE" || m == "PLOCKR" || m == "PUNLOCKR" || m == "PUNLOCK") return {Op::Nop2, 0};
    if (m == "DEC") return {Op::Dec, 0};
    return {Op::Unknown, 0};
}

const std::vector<OpInfo>& op_table() {
    static const std::vector<OpInfo> t = [] {
        std::vector<OpInfo> v;
        for (const OpcodePattern& p : kOpcodePatterns) v.push_back(classify(p.mnemonic, p.operands));
        return v;
    }();
    return t;
}

}  // namespace

Core::Core(const Config& config, Peripherals* peripherals) : cfg_(config), periph_(peripherals) {
    x_.assign(cfg_.x_words, 0);
    y_.assign(cfg_.y_words, 0);
    p_.assign(cfg_.p_words, 0);
    cache_.assign(cfg_.p_words, Cached{});
    reset();
}

void Core::reset() {
    r_ = Regs{};
    r_.sr = I0 | I1;  // interrupts masked, no scaling, convergent rounding
    for (auto& m : r_.m) m = kWord;
    error_.clear();
    halted_ = false;
    rep_active_ = false;
    rep_remaining_ = 0;
}

void Core::fail(const std::string& why) {
    if (error_.empty()) error_ = why;
}

// ---- memory ----

u32 Core::read_x(u32 address) {
    address &= kWord;
    if (address >= 0xFFFF80u) return periph_ ? periph_->peripheral_read(address) & kWord : 0;
    return address < x_.size() ? x_[address] : 0;
}

u32 Core::read_y(u32 address) {
    address &= kWord;
    if (address >= 0xFFFF80u) return periph_ ? periph_->peripheral_read(address) & kWord : 0;
    return address < y_.size() ? y_[address] : 0;
}

u32 Core::read_p(u32 address) const {
    address &= kWord;
    return address < p_.size() ? p_[address] : 0;
}

void Core::write_x(u32 address, u32 value) {
    address &= kWord;
    if (address >= 0xFFFF80u) {
        if (periph_) periph_->peripheral_write(address, value & kWord);
        return;
    }
    if (address < x_.size()) x_[address] = value & kWord;
}

void Core::write_y(u32 address, u32 value) {
    address &= kWord;
    if (address >= 0xFFFF80u) {
        if (periph_) periph_->peripheral_write(address, value & kWord);
        return;
    }
    if (address < y_.size()) y_[address] = value & kWord;
}

void Core::write_p(u32 address, u32 value) {
    address &= kWord;
    if (address < p_.size()) {
        p_[address] = value & kWord;
        cache_[address].valid = false;
        if (address > 0) cache_[address - 1].valid = false;  // an extension word changed
    }
}

u32 Core::read_space(bool y, u32 address) { return y ? read_y(address) : read_x(address); }
void Core::write_space(bool y, u32 address, u32 value) {
    if (y) write_y(address, value);
    else write_x(address, value);
}

const Decoded& Core::decoded(u32 address) {
    Cached& c = cache_[address % cfg_.p_words];
    if (!c.valid) {
        c.d = decode(p_[address % cfg_.p_words]);
        c.valid = true;
    }
    return c.d;
}

std::string Core::disassemble_at(u32 address) const {
    const Decoded d = decode(read_p(address));
    return disassemble(d, d.extension ? read_p(address + 1) : 0);
}

// ---- data ALU helpers ----

u32 Core::limit24(i64 acc56) {
    // Representable in 24 bits when the eight extension bits and bit 47 all equal the sign.
    const i64 upper = acc56 >> 47;
    if (upper == 0 || upper == -1) return static_cast<u32>(acc56 >> 24) & kWord;
    r_.sr |= L;
    return acc56 < 0 ? 0x800000u : 0x7FFFFFu;
}

void Core::set_ccr_result(i64 result, bool carry_in_known, bool carry, bool overflow) {
    const i64 v = sext56(result);
    u32 sr = r_.sr & ~(E | U | N | Z | V | C);
    if (v < 0) sr |= N;
    if (v == 0) sr |= Z;
    const i64 ext = v >> 47;  // bits 55..47
    if (ext != 0 && ext != -1) sr |= E;
    if (((v >> 47) & 1) == ((v >> 46) & 1)) sr |= U;
    if (overflow) sr |= V | L;
    if (carry_in_known && carry) sr |= C;
    r_.sr = sr;
}

void Core::set_ccr_logical(i64 acc56) {
    // AND/OR/EOR/NOT: N from bit 47, Z if the 24-bit result is zero, V cleared; others unchanged.
    u32 sr = r_.sr & ~(N | Z | V);
    if ((acc56 >> 47) & 1) sr |= N;
    if (((acc56 >> 24) & kWord) == 0) sr |= Z;
    r_.sr = sr;
}

i64 Core::source56(u32 jjj, u32 dest) const {
    switch (jjj & 7u) {
        case 0: case 1: return dest ? r_.a : r_.b;  // the other accumulator
        case 2: return sext56((static_cast<i64>(sext24(r_.x1)) << 24) | r_.x0);  // 48-bit X = X1:X0
        case 3: return sext56((static_cast<i64>(sext24(r_.y1)) << 24) | r_.y0);
        case 4: return sext24(r_.x0) << 24;
        case 5: return sext24(r_.y0) << 24;
        case 6: return sext24(r_.x1) << 24;
        default: return sext24(r_.y1) << 24;
    }
}

i64 Core::multiply(u32 qqq, bool negate) const {
    static constexpr u8 kPairs[8][2] = {{0, 0}, {2, 2}, {1, 0}, {3, 2}, {0, 3}, {2, 0}, {1, 2}, {3, 1}};  // X0 X1 Y0 Y1 as 0 1 2 3
    const u32 regs[4] = {r_.x0, r_.x1, r_.y0, r_.y1};
    const i64 a = sext24(regs[kPairs[qqq & 7][0]]), b = sext24(regs[kPairs[qqq & 7][1]]);
    i64 product = (a * b) << 1;  // fractional: 1.23 x 1.23 = 2.46, left-aligned to 1.47
    if (negate) product = -product;
    return sext56(product);
}

i64 Core::round56(i64 v) const {
    // Convergent rounding at the A1/A0 boundary (no scaling).
    const u32 low = static_cast<u32>(v) & kWord;
    const bool odd = (v >> 24) & 1;
    if (low > 0x800000u || (low == 0x800000u && odd)) v += i64{1} << 24;
    v &= ~i64{kWord};
    return sext56(v);
}

bool Core::condition(u32 cccc) const {
    const bool c = r_.sr & C, v = r_.sr & V, z = r_.sr & Z, n = r_.sr & N, u = r_.sr & U, e = r_.sr & E, l = r_.sr & L;
    switch (cccc & 15u) {
        case 0x0: return !c;                 // CC / HS
        case 0x1: return n == v;             // GE
        case 0x2: return !z;                 // NE
        case 0x3: return !n;                 // PL
        case 0x4: return !(z || (!u && !e)); // NN
        case 0x5: return !e;                 // EC
        case 0x6: return !l;                 // LC
        case 0x7: return !(z || (n != v));   // GT
        case 0x8: return c;                  // CS / LO
        case 0x9: return n != v;             // LT
        case 0xA: return z;                  // EQ
        case 0xB: return n;                  // MI
        case 0xC: return z || (!u && !e);    // NR
        case 0xD: return e;                  // ES
        case 0xE: return l;                  // LS
        default: return z || (n != v);       // LE
    }
}

// Five-bit register codes (Table 12-16): X0 X1 Y0 Y1 A0 B0 A2 B2 A1 B1 A B R0-7 N0-7.
u32 Core::read_reg5(u32 code, bool limit) {
    switch (code & 31u) {
        case 4: return r_.x0;
        case 5: return r_.x1;
        case 6: return r_.y0;
        case 7: return r_.y1;
        case 8: return static_cast<u32>(r_.a) & kWord;
        case 9: return static_cast<u32>(r_.b) & kWord;
        case 10: return static_cast<u32>(r_.a >> 48) & 0xFFu;
        case 11: return static_cast<u32>(r_.b >> 48) & 0xFFu;
        case 12: return static_cast<u32>(r_.a >> 24) & kWord;
        case 13: return static_cast<u32>(r_.b >> 24) & kWord;
        case 14: return limit ? limit24(r_.a) : (static_cast<u32>(r_.a >> 24) & kWord);
        case 15: return limit ? limit24(r_.b) : (static_cast<u32>(r_.b >> 24) & kWord);
        default:
            if (code >= 16 && code < 24) return r_.r[code - 16];
            if (code >= 24) return r_.n[code - 24];
            return 0;
    }
}

void Core::write_reg5(u32 code, u32 value) {
    value &= kWord;
    switch (code & 31u) {
        case 4: r_.x0 = value; return;
        case 5: r_.x1 = value; return;
        case 6: r_.y0 = value; return;
        case 7: r_.y1 = value; return;
        case 8: r_.a = sext56((r_.a & ~i64{kWord}) | value); return;
        case 9: r_.b = sext56((r_.b & ~i64{kWord}) | value); return;
        case 10: r_.a = sext56((r_.a & ((i64{1} << 48) - 1)) | (static_cast<i64>(value & 0xFFu) << 48)); return;
        case 11: r_.b = sext56((r_.b & ((i64{1} << 48) - 1)) | (static_cast<i64>(value & 0xFFu) << 48)); return;
        case 12: r_.a = sext56((r_.a & ~(i64{kWord} << 24)) | (static_cast<i64>(value) << 24)); return;
        case 13: r_.b = sext56((r_.b & ~(i64{kWord} << 24)) | (static_cast<i64>(value) << 24)); return;
        case 14: r_.a = sext24(value) << 24; return;  // sign-extended into A2, A0 cleared
        case 15: r_.b = sext24(value) << 24; return;
        default:
            if (code >= 16 && code < 24) r_.r[code - 16] = value;
            else if (code >= 24) r_.n[code - 24] = value;
            return;
    }
}

// Six-bit codes (Table 12-13): the five-bit set with a leading 0, then 100FFF M, 101EEE EP, 110VVV VBA/SC, 111GGG.
u32 Core::read_reg6(u32 code) {
    code &= 63u;
    if (code < 32) return read_reg5(code, true);
    const u32 sub = code & 7u;
    switch (code >> 3) {
        case 4: return r_.m[sub];
        case 5: return sub == 2 ? r_.ep : 0;
        case 6: return sub == 0 ? r_.vba : (sub == 1 ? r_.sc : 0);
        default:
            switch (sub) {
                case 0: return r_.sz;
                case 1: return r_.sr;
                case 2: return r_.omr;
                case 3: return r_.sp;
                case 4: return r_.ssh[r_.sp & 15u];
                case 5: return r_.ssl[r_.sp & 15u];
                case 6: return r_.la;
                default: return r_.lc;
            }
    }
}

void Core::write_reg6(u32 code, u32 value) {
    code &= 63u;
    value &= kWord;
    if (code < 32) {
        write_reg5(code, value);
        return;
    }
    const u32 sub = code & 7u;
    switch (code >> 3) {
        case 4: r_.m[sub] = value; return;
        case 5: if (sub == 2) r_.ep = value; return;
        case 6: if (sub == 0) r_.vba = value; else if (sub == 1) r_.sc = value; return;
        default:
            switch (sub) {
                case 0: r_.sz = value; return;
                case 1: r_.sr = value; return;
                case 2: r_.omr = value; return;
                case 3: r_.sp = value; return;
                case 4: r_.ssh[r_.sp & 15u] = value; return;
                case 5: r_.ssl[r_.sp & 15u] = value; return;
                case 6: r_.la = value; return;
                default: r_.lc = value; return;
            }
    }
}

// ---- AGU ----

u32 Core::agu_update(u32 r, u32 m, i32 delta) const {
    const u32 mod = m & 0xFFFFu;
    if (mod == 0xFFFFu) return static_cast<u32>(static_cast<i32>(r) + delta) & kWord;  // linear
    if (mod == 0) {
        // Reverse carry: propagate the carry from the MSB toward the LSB.
        u32 rev_r = 0, rev_d = 0;
        for (int i = 0; i < 24; ++i) {
            rev_r |= ((r >> i) & 1u) << (23 - i);
            rev_d |= ((static_cast<u32>(delta) >> i) & 1u) << (23 - i);
        }
        const u32 sum = (rev_r + rev_d) & kWord;
        u32 out = 0;
        for (int i = 0; i < 24; ++i) out |= ((sum >> i) & 1u) << (23 - i);
        return out;
    }
    if (mod < 0x8000u) {
        // Modulo (mod + 1): the buffer starts at the lowest multiple of 2^k not below the register.
        u32 k = 0;
        while ((1u << k) < mod + 1) ++k;
        const u32 base = r & ~((1u << k) - 1u);
        const i32 size = static_cast<i32>(mod) + 1;
        i32 offset = static_cast<i32>(r - base) + delta;
        if (offset >= size) offset -= size;
        else if (offset < 0) offset += size;
        return (base + static_cast<u32>(offset)) & kWord;
    }
    // Multiple wrap-around modulo 2^k.
    const u32 span = (mod - 0x8000u) + 1u;
    const u32 base = r & ~(span - 1u);
    return (base | (static_cast<u32>(static_cast<i32>(r) + delta) & (span - 1u))) & kWord;
}

u32 Core::ea_address(u32 mmmrrr, u32 ext, bool& used_ext, bool update) {
    const u32 mode = (mmmrrr >> 3) & 7u, n = mmmrrr & 7u;
    used_ext = false;
    u32& reg = r_.r[n];
    const u32 off = r_.n[n], mod = r_.m[n];
    switch (mode) {
        case 0: { const u32 a = reg; if (update) reg = agu_update(reg, mod, -static_cast<i32>(off)); return a; }
        case 1: { const u32 a = reg; if (update) reg = agu_update(reg, mod, static_cast<i32>(off)); return a; }
        case 2: { const u32 a = reg; if (update) reg = agu_update(reg, mod, -1); return a; }
        case 3: { const u32 a = reg; if (update) reg = agu_update(reg, mod, 1); return a; }
        case 4: return reg;
        case 5: return agu_update(reg, mod, static_cast<i32>(off));
        case 7: { if (update) reg = agu_update(reg, mod, -1); return reg; }
        default:
            used_ext = true;
            return ext & kWord;  // absolute address (110000) or immediate data (110100): the caller knows which
    }
}

// ---- program control ----

void Core::push(u32 pc_value, u32 sr_value) {
    r_.sp = (r_.sp + 1) & 0xFFu;
    r_.ssh[r_.sp & 15u] = pc_value & kWord;
    r_.ssl[r_.sp & 15u] = sr_value & kWord;
}

void Core::pop(u32& pc_value, u32& sr_value) {
    pc_value = r_.ssh[r_.sp & 15u];
    sr_value = r_.ssl[r_.sp & 15u];
    r_.sp = (r_.sp - 1) & 0xFFu;
}

void Core::do_loop(u32 count, u32 last_address, u32 first_address) {
    push(r_.la, r_.lc);        // LA and LC
    r_.lc = count & kWord;
    push(first_address, r_.sr);  // the loop's first instruction and SR (with the previous LF)
    r_.la = last_address & kWord;
    r_.sr |= LF;
}

void Core::end_of_loop_check(u32 executed_at) {
    if (!(r_.sr & LF) || executed_at != r_.la) return;
    if (r_.lc == 1 && !(r_.sr & FV)) {
        // End of loop: SSL(LF) -> SR, SP - 1 -> SP; SSH -> LA, SSL -> LC, SP - 1 -> SP.
        u32 pc_value, sr_value;
        pop(pc_value, sr_value);
        r_.sr = (r_.sr & ~(LF | FV)) | (sr_value & (LF | FV));
        u32 la, lc;
        pop(la, lc);
        r_.la = la;
        r_.lc = lc;
        return;
    }
    r_.lc = (r_.lc - 1) & kWord;
    r_.pc = r_.ssh[r_.sp & 15u];
}

// ---- the parallel family ----

void Core::alu(const Decoded& d, u32 /*ext*/, i64 pre_a, i64 pre_b) {
    (void)pre_a;
    (void)pre_b;
    const u32 dst = d.alu_dest;
    i64& D = acc(dst);
    const i64 src = source56(d.alu_src, dst);
    switch (d.alu) {
        case Alu::None: return;
        case Alu::Mpy: case Alu::Mpyr: case Alu::Mac: case Alu::Macr: {
            const i64 p = multiply(d.alu_src, d.alu_negate);
            i64 r = (d.alu == Alu::Mac || d.alu == Alu::Macr) ? sext56(D + p) : p;
            const bool overflow = (d.alu == Alu::Mac || d.alu == Alu::Macr) && ((D < 0) == (p < 0)) && ((r < 0) != (D < 0));
            if (d.alu == Alu::Mpyr || d.alu == Alu::Macr) r = round56(r);
            D = r;
            set_ccr_result(D, false, false, overflow);
            return;
        }
        case Alu::Add: case Alu::Sub: case Alu::Cmp: {
            const i64 b = (d.alu == Alu::Add) ? src : -src;
            const i64 r = D + b;
            const bool overflow = ((D < 0) == (b < 0)) && ((sext56(r) < 0) != (D < 0));
            const bool carry = d.alu == Alu::Add ? (static_cast<u64>(D & ((i64{1} << 56) - 1)) + static_cast<u64>(src & ((i64{1} << 56) - 1))) >> 56
                                                 : static_cast<u64>(D & ((i64{1} << 56) - 1)) < static_cast<u64>(src & ((i64{1} << 56) - 1));
            if (d.alu != Alu::Cmp) D = sext56(r);
            set_ccr_result(sext56(r), true, carry, overflow);
            return;
        }
        case Alu::Cmpm: {
            const i64 a = D < 0 ? -D : D, b = src < 0 ? -src : src;
            set_ccr_result(sext56(a - b), true, static_cast<u64>(a) < static_cast<u64>(b), false);
            return;
        }
        case Alu::Tfr: D = src; set_ccr_result(D, false, false, false); return;
        case Alu::Tst: set_ccr_result(src, true, false, false); return;
        case Alu::Rnd: D = round56(D); set_ccr_result(D, false, false, false); return;
        case Alu::Clr: D = 0; set_ccr_result(0, false, false, false); return;
        case Alu::Addr: D = sext56((D >> 1) + src); set_ccr_result(D, false, false, false); return;
        case Alu::Subr: D = sext56((D >> 1) - src); set_ccr_result(D, false, false, false); return;
        case Alu::Addl: D = sext56((D << 1) + src); set_ccr_result(D, false, false, false); return;
        case Alu::Subl: D = sext56((D << 1) - src); set_ccr_result(D, false, false, false); return;
        case Alu::Not: {
            const i64 a1 = (~D) & (i64{kWord} << 24);
            D = (D & ~(i64{kWord} << 24)) | a1;
            set_ccr_logical(D);
            return;
        }
        case Alu::And: case Alu::Or: case Alu::Eor: {
            const i64 s = src & (i64{kWord} << 24), a1 = D & (i64{kWord} << 24);
            const i64 r = d.alu == Alu::And ? (a1 & s) : (d.alu == Alu::Or ? (a1 | s) : (a1 ^ s));
            D = (D & ~(i64{kWord} << 24)) | r;
            set_ccr_logical(D);
            return;
        }
        case Alu::Asr: {
            const bool c = D & 1;
            D = sext56(D >> 1);
            set_ccr_result(D, true, c, false);
            return;
        }
        case Alu::Asl: {
            const bool c = (D >> 55) & 1;
            const i64 r = sext56(D << 1);
            const bool overflow = ((r >> 55) & 1) != ((D >> 55) & 1);
            D = r;
            set_ccr_result(D, true, c, overflow);
            return;
        }
        case Alu::Lsr: {
            const u32 a1 = static_cast<u32>(D >> 24) & kWord;
            D = (D & ~(i64{kWord} << 24)) | (static_cast<i64>(a1 >> 1) << 24);
            r_.sr = (r_.sr & ~C) | ((a1 & 1u) ? C : 0u);
            set_ccr_logical(D);
            return;
        }
        case Alu::Lsl: {
            const u32 a1 = static_cast<u32>(D >> 24) & kWord;
            D = (D & ~(i64{kWord} << 24)) | (static_cast<i64>((a1 << 1) & kWord) << 24);
            r_.sr = (r_.sr & ~C) | ((a1 >> 23) & 1u ? C : 0u);
            set_ccr_logical(D);
            return;
        }
        case Alu::Ror: case Alu::Rol: {
            const u32 a1 = static_cast<u32>(D >> 24) & kWord;
            const u32 cin = (r_.sr & C) ? 1u : 0u;
            u32 out, cout;
            if (d.alu == Alu::Ror) { cout = a1 & 1u; out = (a1 >> 1) | (cin << 23); }
            else { cout = (a1 >> 23) & 1u; out = ((a1 << 1) & kWord) | cin; }
            D = (D & ~(i64{kWord} << 24)) | (static_cast<i64>(out) << 24);
            r_.sr = (r_.sr & ~C) | (cout ? C : 0u);
            set_ccr_logical(D);
            return;
        }
        case Alu::Abs: D = sext56(D < 0 ? -D : D); set_ccr_result(D, false, false, false); return;
        case Alu::Neg: D = sext56(-D); set_ccr_result(D, false, false, false); return;
        case Alu::Adc: case Alu::Sbc: {
            const i64 cin = (r_.sr & C) ? 1 : 0;
            const i64 r = d.alu == Alu::Adc ? D + src + cin : D - src - cin;
            D = sext56(r);
            set_ccr_result(D, false, false, false);
            return;
        }
    }
}

void Core::parallel_move(const Decoded& d, u32 ext, i64 pre_a, i64 pre_b) {
    // Reads of the accumulators use their values before the ALU operation of the same instruction.
    const i64 cur_a = r_.a, cur_b = r_.b;
    auto read_pre5 = [&](u32 code) {
        r_.a = pre_a;
        r_.b = pre_b;
        const u32 v = read_reg5(code, true);
        r_.a = cur_a;
        r_.b = cur_b;
        return v;
    };
    bool used_ext = false;
    switch (d.move) {
        case Move::NotParallel: case Move::None: return;
        case Move::Immediate: {
            const u32 code = d.f('d'), imm = d.f('i');
            // Signed fraction into X0 X1 Y0 Y1 A B (the eight MSBs), unsigned integer into the others.
            if ((code >= 4 && code <= 7) || code == 14 || code == 15) write_reg5(code, imm << 16);
            else write_reg5(code, imm);
            return;
        }
        case Move::Register: write_reg5(d.f('d'), read_pre5(d.f('e'))); return;
        case Move::Update: {
            const u32 mmrrr = (d.f('M') << 3) | d.f('R');
            ea_address(mmrrr, 0, used_ext, true);
            return;
        }
        case Move::X: case Move::Y: {
            const bool y = d.move == Move::Y;
            const u32 code = d.f('d');
            const bool write = d.f('W');
            u32 address;
            bool immediate = false;
            if ((d.word >> 14) & 1u) {
                const u32 mmmrrr = (d.f('M') << 3) | d.f('R');
                address = ea_address(mmmrrr, ext, used_ext, true);
                immediate = mmmrrr == 0b110100;
            } else {
                address = d.f('a');
            }
            if (write) write_reg5(code, immediate ? ext : read_space(y, address));
            else write_space(y, address, read_pre5(code));
            return;
        }
        case Move::L: {
            static constexpr u8 kHi[8] = {12, 13, 5, 7, 12, 13, 14, 15}, kLo[8] = {8, 9, 4, 6, 8, 9, 15, 14};
            const u32 l = d.f('L') & 7u;
            const bool write = d.f('W');
            u32 address;
            if ((d.word >> 14) & 1u) address = ea_address((d.f('M') << 3) | d.f('R'), ext, used_ext, true);
            else address = d.f('a');
            if (write) {
                const u32 hi = read_x(address), lo = read_y(address);
                if (l == 4 || l == 5) {  // A or B as a 48-bit value, sign-extended
                    acc(l - 4) = sext56((static_cast<i64>(sext24(hi)) << 24) | lo);
                } else if (l == 6) { write_reg5(14, hi); write_reg5(15, lo); }
                else if (l == 7) { write_reg5(15, hi); write_reg5(14, lo); }
                else { write_reg5(kHi[l], hi); write_reg5(kLo[l], lo); }
            } else {
                u32 hi, lo;
                if (l == 4 || l == 5) {
                    const i64 v = l == 4 ? pre_a : pre_b;
                    const i64 upper = v >> 47;
                    if (upper == 0 || upper == -1) { hi = static_cast<u32>(v >> 24) & kWord; lo = static_cast<u32>(v) & kWord; }
                    else { r_.sr |= L; hi = v < 0 ? 0x800000u : 0x7FFFFFu; lo = v < 0 ? 0 : kWord; }
                } else {
                    hi = read_pre5(kHi[l]);
                    lo = read_pre5(kLo[l]);
                }
                write_x(address, hi);
                write_y(address, lo);
            }
            return;
        }
        case Move::XR1: {
            static constexpr u8 kFf[4] = {4, 5, 14, 15};
            const u32 mmmrrr = (d.f('M') << 3) | d.f('R');
            const u32 address = ea_address(mmmrrr, ext, used_ext, true);
            const u32 s2 = d.f('d') ? read_pre5(15) : read_pre5(14);
            if (d.f('W')) write_reg5(kFf[d.f('f') & 3u], mmmrrr == 0b110100 ? ext : read_x(address));
            else write_x(address, read_pre5(kFf[d.f('f') & 3u]));
            write_reg5(d.f('F') ? 7 : 6, s2);  // S2 -> Y0/Y1
            return;
        }
        case Move::RY1: {
            static constexpr u8 kFf[4] = {6, 7, 14, 15};
            const u32 mmmrrr = (d.f('M') << 3) | d.f('R');
            const u32 address = ea_address(mmmrrr, ext, used_ext, true);
            const u32 s1 = d.f('d') ? read_pre5(15) : read_pre5(14);
            write_reg5(d.f('e') ? 5 : 4, s1);  // S1 -> X0/X1
            if (d.f('W')) write_reg5(kFf[d.f('f') & 3u], mmmrrr == 0b110100 ? ext : read_y(address));
            else write_y(address, read_pre5(kFf[d.f('f') & 3u]));
            return;
        }
        case Move::XR2: {  // A/B -> X:ea, X0 -> A/B
            const u32 address = ea_address((d.f('M') << 3) | d.f('R'), ext, used_ext, true);
            write_x(address, read_pre5(d.f('d') ? 15 : 14));
            write_reg5(d.f('d') ? 15 : 14, r_.x0);
            return;
        }
        case Move::RY2: {  // Y0 -> A/B, A/B -> Y:ea
            const u32 address = ea_address((d.f('M') << 3) | d.f('R'), ext, used_ext, true);
            write_y(address, read_pre5(d.f('d') ? 15 : 14));
            write_reg5(d.f('d') ? 15 : 14, r_.y0);
            return;
        }
        case Move::XY: {
            static constexpr u8 kE[4] = {4, 5, 14, 15}, kF[4] = {6, 7, 14, 15};
            const u32 rx = d.f('R'), mx = d.f('M'), ry = (rx < 4 ? 4 : 0) + d.f('r'), my = d.f('m');
            // Modes (Table 12-16 X:Y: operands): 01 (Rn)+Nn, 10 (Rn)-, 11 (Rn)+, 00 (Rn).
            auto resolve = [&](u32 n, u32 mode) {
                const u32 a = r_.r[n];
                switch (mode & 3u) {
                    case 1: r_.r[n] = agu_update(a, r_.m[n], static_cast<i32>(r_.n[n])); break;
                    case 2: r_.r[n] = agu_update(a, r_.m[n], -1); break;
                    case 3: r_.r[n] = agu_update(a, r_.m[n], 1); break;
                    default: break;
                }
                return a;
            };
            const u32 xa = resolve(rx, mx), ya = resolve(ry, my);
            const u32 xs = read_pre5(kE[d.f('e') & 3u]), ys = read_pre5(kF[d.f('f') & 3u]);
            if (d.f('W')) write_reg5(kE[d.f('e') & 3u], read_x(xa));
            else write_x(xa, xs);
            if (d.f('w')) write_reg5(kF[d.f('f') & 3u], read_y(ya));
            else write_y(ya, ys);
            return;
        }
    }
}

// ---- the non-parallel instructions ----

void Core::execute(const Decoded& d, u32 ext, u32 at) {
    if (d.pattern == -2) {
        fail(std::format("unknown instruction word {:06x} at P:{:04x}", d.word, at));
        return;
    }
    if (d.pattern < 0) {
        const i64 pre_a = r_.a, pre_b = r_.b;
        alu(d, ext, pre_a, pre_b);
        parallel_move(d, ext, pre_a, pre_b);
        return;
    }
    const OpInfo info = op_table()[static_cast<std::size_t>(d.pattern)];
    const u32 next = r_.pc;  // already past this instruction
    bool used_ext = false;
    auto mmmrrr = [&]() { return (d.f('M') << 3) | d.f('R'); };
    auto bit_source_address = [&](Op base, u32& address, bool& y, bool& is_reg) {
        is_reg = false;
        y = d.f('S') & 1u;
        switch (base) {
            case Op::BitOpEa: case Op::BitJumpEa: case Op::BitBranchEa: address = ea_address(mmmrrr(), ext, used_ext, true); break;
            case Op::BitOpAa: case Op::BitJumpAa: case Op::BitBranchAa: address = d.f('a'); break;
            case Op::BitOpPp: case Op::BitJumpPp: case Op::BitBranchPp: address = 0xFFFFC0u | d.f('p'); break;
            case Op::BitOpQq: case Op::BitJumpQq: case Op::BitBranchQq: address = 0xFFFF80u | d.f('q'); break;
            default: is_reg = true; address = d.f('D'); break;
        }
    };
    auto relative = [&](u32 disp, u32 bits) { return (at + static_cast<u32>(sext_bits(disp, bits))) & kWord; };
    auto jump_sub = [&](u32 target, bool sub) {
        if (sub) push(next, r_.sr);
        r_.pc = target & kWord;
    };
    switch (info.op) {
        case Op::Nop: case Op::Debug: case Op::Reset: return;
        case Op::Rts: {
            u32 pc_value, sr_value;
            pop(pc_value, sr_value);
            r_.pc = pc_value;
            return;
        }
        case Op::Rti: {
            u32 pc_value, sr_value;
            pop(pc_value, sr_value);
            r_.pc = pc_value;
            r_.sr = sr_value;
            return;
        }
        case Op::Enddo: {
            u32 pc_value, sr_value;
            pop(pc_value, sr_value);
            r_.sr = (r_.sr & ~(LF | FV)) | (sr_value & (LF | FV));
            u32 la, lc;
            pop(la, lc);
            r_.la = la;
            r_.lc = lc;
            return;
        }
        case Op::Illegal: fail(std::format("ILLEGAL at P:{:04x}", at)); return;
        case Op::Stop: case Op::Wait: halted_ = true; return;
        case Op::Trap: fail(std::format("TRAP at P:{:04x} (software interrupts are not modelled)", at)); return;
        case Op::JmpEa: jump_sub(ea_address(mmmrrr(), ext, used_ext, true), false); return;
        case Op::JmpShort: jump_sub(d.f('a'), false); return;
        case Op::JsrEa: jump_sub(ea_address(mmmrrr(), ext, used_ext, true), true); return;
        case Op::JsrShort: jump_sub(d.f('a'), true); return;
        case Op::JccShort: if (condition(d.f('C'))) jump_sub(d.f('a'), false); return;
        case Op::JccEa: { const u32 t = ea_address(mmmrrr(), ext, used_ext, true); if (condition(d.f('C'))) jump_sub(t, false); return; }
        case Op::JsccShort: if (condition(d.f('C'))) jump_sub(d.f('a'), true); return;
        case Op::JsccEa: { const u32 t = ea_address(mmmrrr(), ext, used_ext, true); if (condition(d.f('C'))) jump_sub(t, true); return; }
        case Op::BraLong: jump_sub(relative(ext, 24), false); return;
        case Op::BraShort: jump_sub(relative(d.f('a'), 9), false); return;
        case Op::BraRn: jump_sub((at + r_.r[d.f('R')]) & kWord, false); return;
        case Op::BsrLong: jump_sub(relative(ext, 24), true); return;
        case Op::BsrShort: jump_sub(relative(d.f('a'), 9), true); return;
        case Op::BsrRn: jump_sub((at + r_.r[d.f('R')]) & kWord, true); return;
        case Op::BccLong: if (condition(d.f('C'))) jump_sub(relative(ext, 24), false); return;
        case Op::BccShort: if (condition(d.f('C'))) jump_sub(relative(d.f('a'), 9), false); return;
        case Op::BccRn: if (condition(d.f('C'))) jump_sub((at + r_.r[d.f('R')]) & kWord, false); return;
        case Op::BsccLong: if (condition(d.f('C'))) jump_sub(relative(ext, 24), true); return;
        case Op::BsccShort: if (condition(d.f('C'))) jump_sub(relative(d.f('a'), 9), true); return;
        case Op::BsccRn: if (condition(d.f('C'))) jump_sub((at + r_.r[d.f('R')]) & kWord, true); return;
        case Op::BitJumpEa: case Op::BitJumpAa: case Op::BitJumpPp: case Op::BitJumpQq: case Op::BitJumpReg:
        case Op::BitBranchEa: case Op::BitBranchAa: case Op::BitBranchPp: case Op::BitBranchQq: case Op::BitBranchReg: {
            u32 address;
            bool y, is_reg;
            bit_source_address(info.op, address, y, is_reg);
            const u32 value = is_reg ? read_reg6(address) : read_space(y, address);
            const bool set = (value >> (d.f('b') & 31u)) & 1u;
            r_.sr = (r_.sr & ~C) | (set ? C : 0u);
            const bool want_set = info.variant & 1u, sub = info.variant & 2u;
            if (set != want_set) return;
            const bool branch = info.op >= Op::BitBranchEa && info.op <= Op::BitBranchReg;
            jump_sub(branch ? relative(ext, 24) : (ext & kWord), sub);
            return;
        }
        case Op::BitOpEa: case Op::BitOpAa: case Op::BitOpPp: case Op::BitOpQq: case Op::BitOpReg: {
            u32 address;
            bool y, is_reg;
            bit_source_address(info.op, address, y, is_reg);
            const u32 bit = 1u << (d.f('b') & 31u);
            const u32 value = is_reg ? read_reg6(address) : read_space(y, address);
            r_.sr = (r_.sr & ~C) | ((value & bit) ? C : 0u);
            u32 out = value;
            switch (info.variant) {
                case 0: out = value & ~bit; break;
                case 1: out = value | bit; break;
                case 2: out = value ^ bit; break;
                default: return;  // BTST
            }
            if (is_reg) write_reg6(address, out);
            else write_space(y, address, out);
            return;
        }
        case Op::DoEa: case Op::DoAa: case Op::DoImm: case Op::DoReg: case Op::DoForever:
        case Op::DorEa: case Op::DorAa: case Op::DorImm: case Op::DorReg: case Op::DorForever: {
            u32 count = 0;
            const bool y = d.f('S') & 1u;
            switch (info.op) {
                case Op::DoEa: case Op::DorEa: count = read_space(y, ea_address(mmmrrr(), ext, used_ext, true)); break;
                case Op::DoAa: case Op::DorAa: count = read_space(y, d.f('a')); break;
                case Op::DoImm: case Op::DorImm: count = (d.f('h') << 8) | d.f('i'); break;
                case Op::DoReg: case Op::DorReg: count = read_reg6(d.f('D')); break;
                default: break;
            }
            const bool relative_form = info.op >= Op::DorEa;
            const u32 last = relative_form ? relative(ext, 24) : ((ext - 1) & kWord);
            if (info.op == Op::DoForever || info.op == Op::DorForever) {
                do_loop(r_.lc, last, next);
                r_.sr |= FV;
                return;
            }
            if (count == 0 && !(r_.sr & SC_)) {
                r_.pc = (last + 1) & kWord;  // a zero count skips the loop
                return;
            }
            do_loop(count, last, next);
            return;
        }
        case Op::RepEa: case Op::RepAa: case Op::RepImm: case Op::RepReg: {
            u32 count = 0;
            const bool y = d.f('S') & 1u;
            switch (info.op) {
                case Op::RepEa: count = read_space(y, ea_address(mmmrrr(), ext, used_ext, true)); break;
                case Op::RepAa: count = read_space(y, d.f('a')); break;
                case Op::RepImm: count = (d.f('h') << 8) | d.f('i'); break;
                default: count = read_reg6(d.f('d')); break;
            }
            rep_saved_lc_ = r_.lc;
            r_.lc = count & kWord;
            rep_active_ = count != 0;
            rep_remaining_ = count;
            return;
        }
        case Op::MoveLongDispX: case Op::MoveLongDispY: {
            const bool y = info.op == Op::MoveLongDispY;
            const u32 address = (r_.r[d.f('R')] + ext) & kWord;
            if (d.f('W')) write_reg6(d.f('D'), read_space(y, address));
            else write_space(y, address, read_reg6(d.f('D')));
            return;
        }
        case Op::MoveShortDispX: case Op::MoveShortDispY: {
            const bool y = info.op == Op::MoveShortDispY;
            const u32 address = (r_.r[d.f('R')] + static_cast<u32>(sext_bits(d.f('a'), 7))) & kWord;
            // DDDD: 01DD = X0 X1 Y0 Y1, 1DDD = A0 B0 A2 B2 A1 B1 A B (the five-bit codes 4..15).
            const u32 code = d.f('D') & 15u;
            const u32 code5 = (code & 8u) ? (8u + (code & 7u)) : (4u + (code & 3u));
            if (d.f('W')) write_reg5(code5, read_space(y, address));
            else write_space(y, address, read_reg5(code5, true));
            return;
        }
        case Op::MovecEa: case Op::MovecAa: {
            const bool y = d.f('S') & 1u;
            const u32 reg = 32u | (d.f('d') & 31u);
            u32 address = 0;
            bool immediate = false;
            if (info.op == Op::MovecEa) {
                const u32 m = mmmrrr();
                address = ea_address(m, ext, used_ext, true);
                immediate = m == 0b110100;
            } else {
                address = d.f('a');
            }
            if (d.f('W')) write_reg6(reg, immediate ? ext : read_space(y, address));
            else write_space(y, address, read_reg6(reg));
            return;
        }
        case Op::MovecReg: {
            const u32 ctrl = 32u | (d.f('d') & 31u), other = d.f('e') & 63u;
            if (d.f('W')) write_reg6(ctrl, read_reg6(other));
            else write_reg6(other, read_reg6(ctrl));
            return;
        }
        case Op::MovecImm: write_reg6(32u | (d.f('d') & 31u), d.f('i')); return;
        case Op::MovemEa: case Op::MovemAa: {
            const u32 address = info.op == Op::MovemEa ? ea_address(mmmrrr(), ext, used_ext, true) : d.f('a');
            if (d.f('W')) write_reg6(d.f('d'), read_p(address));
            else write_p(address, read_reg6(d.f('d')));
            return;
        }
        case Op::MovepPpEa: case Op::MovepXqqEa: case Op::MovepYqqEa: {
            const u32 periph_address = info.op == Op::MovepPpEa ? (0xFFFFC0u | d.f('p')) : (0xFFFF80u | d.f('q'));
            const bool periph_y = info.op == Op::MovepPpEa ? (d.f('s') & 1u) : (info.op == Op::MovepYqqEa);
            const bool mem_y = d.f('S') & 1u;
            const u32 m = mmmrrr();
            const u32 address = ea_address(m, ext, used_ext, true);
            if (d.f('W')) write_space(periph_y, periph_address, m == 0b110100 ? ext : read_space(mem_y, address));
            else write_space(mem_y, address, read_space(periph_y, periph_address));
            return;
        }
        case Op::MovepPpP: case Op::MovepQqP: {
            const u32 periph_address = info.op == Op::MovepPpP ? (0xFFFFC0u | d.f('p')) : (0xFFFF80u | d.f('q'));
            const bool periph_y = info.op == Op::MovepPpP ? (d.f('s') & 1u) : (d.f('S') & 1u);
            const u32 address = ea_address(mmmrrr(), ext, used_ext, true);
            if (d.f('W')) write_space(periph_y, periph_address, read_p(address));
            else write_p(address, read_space(periph_y, periph_address));
            return;
        }
        case Op::MovepPpReg: case Op::MovepXqqReg: case Op::MovepYqqReg: {
            const u32 periph_address = info.op == Op::MovepPpReg ? (0xFFFFC0u | d.f('p')) : (0xFFFF80u | d.f('q'));
            const bool periph_y = info.op == Op::MovepPpReg ? (d.f('s') & 1u) : (info.op == Op::MovepYqqReg);
            // W = 1 writes the peripheral from the register (the kernel's program loads its DMA pointers this way).
            if (d.f('W')) write_space(periph_y, periph_address, read_reg6(d.f('d')));
            else write_reg6(d.f('d'), read_space(periph_y, periph_address));
            return;
        }
        case Op::LuaEa: {
            // MMRRR: 00 (Rn)-Nn, 01 (Rn)+Nn, 10 (Rn)-, 11 (Rn)+ ; the result goes to R or N (dddd: 0nnn R, 1nnn N).
            const u32 mm = d.f('M'), n = d.f('R');
            i32 delta = 0;
            switch (mm & 3u) {
                case 0: delta = -static_cast<i32>(r_.n[n]); break;
                case 1: delta = static_cast<i32>(r_.n[n]); break;
                case 2: delta = -1; break;
                default: delta = 1; break;
            }
            const u32 value = agu_update(r_.r[n], r_.m[n], delta);
            const u32 dd = d.f('d') & 15u;
            if (dd & 8u) r_.n[dd & 7u] = value; else r_.r[dd & 7u] = value;
            return;
        }
        case Op::LuaDisp: {
            const u32 value = (r_.r[d.f('R')] + static_cast<u32>(sext_bits(d.f('a'), 7))) & kWord;
            const u32 dd = d.f('d') & 15u;
            if (dd & 8u) r_.n[dd & 7u] = value; else r_.r[dd & 7u] = value;
            return;
        }
        case Op::AluImmShort: case Op::AluImmLong: {
            // ADD/SUB/CMP: the immediate is a 24-bit word added as the A1 part; AND/OR/EOR act on A1.
            const u32 imm24 = info.op == Op::AluImmLong ? (ext & kWord) : d.f('i');
            i64& D = acc(d.f('d'));
            const i64 src = sext24(imm24) << 24;
            switch (info.variant) {
                case 0: case 1: case 2: {
                    const i64 b = info.variant == 0 ? src : -src;
                    const i64 r = D + b;
                    const bool overflow = ((D < 0) == (b < 0)) && ((sext56(r) < 0) != (D < 0));
                    if (info.variant != 2) D = sext56(r);
                    set_ccr_result(sext56(r), false, false, overflow);
                    return;
                }
                default: {
                    const i64 s = static_cast<i64>(imm24) << 24, a1 = D & (i64{kWord} << 24);
                    const i64 r = info.variant == 3 ? (a1 & s) : (info.variant == 4 ? (a1 | s) : (a1 ^ s));
                    D = (D & ~(i64{kWord} << 24)) | r;
                    set_ccr_logical(D);
                    return;
                }
            }
        }
        case Op::AndiOri: {
            // EE: 00 MR (SR high byte), 01 CCR, 10 COM (OMR low), 11 EOM.
            const u32 imm = d.f('i') & 0xFFu, ee = d.f('E') & 3u;
            auto apply = [&](u32 v) { return info.variant ? (v | imm) : (v & imm); };
            switch (ee) {
                case 0: r_.sr = (r_.sr & ~0xFF00u) | (apply((r_.sr >> 8) & 0xFFu) << 8); return;
                case 1: r_.sr = (r_.sr & ~0xFFu) | apply(r_.sr & 0xFFu); return;
                case 2: r_.omr = (r_.omr & ~0xFFu) | apply(r_.omr & 0xFFu); return;
                default: r_.omr = (r_.omr & ~0xFF00u) | (apply((r_.omr >> 8) & 0xFFu) << 8); return;
            }
        }
        case Op::AslImm: case Op::AsrImm: {
            const u32 count = d.f('i') & 63u;
            const i64 s = acc(d.f('S'));
            i64 r = info.op == Op::AslImm ? sext56(s << count) : sext56(s >> count);
            acc(d.f('D')) = r;
            set_ccr_result(r, false, false, false);
            return;
        }
        case Op::AslReg: case Op::AsrReg: {
            static constexpr u8 kSss[8] = {0, 0, 12, 13, 4, 6, 5, 7};
            const u32 count = read_reg5(kSss[d.f('s') & 7u], false) & 63u;
            const i64 s = acc(d.f('S'));
            const i64 r = info.op == Op::AslReg ? sext56(s << count) : sext56(s >> count);
            acc(d.f('D')) = r;
            set_ccr_result(r, false, false, false);
            return;
        }
        case Op::LslImm: case Op::LsrImm: case Op::LslReg: case Op::LsrReg: {
            static constexpr u8 kSss[8] = {0, 0, 12, 13, 4, 6, 5, 7};
            const u32 count = (info.op == Op::LslImm || info.op == Op::LsrImm) ? (d.f('i') & 31u) : (read_reg5(kSss[d.f('s') & 7u], false) & 31u);
            i64& D = acc(d.f('D'));
            u32 a1 = static_cast<u32>(D >> 24) & kWord;
            a1 = (info.op == Op::LslImm || info.op == Op::LslReg) ? ((a1 << count) & kWord) : (a1 >> count);
            D = (D & ~(i64{kWord} << 24)) | (static_cast<i64>(a1) << 24);
            set_ccr_logical(D);
            return;
        }
        case Op::MulImm: {
            // (±)#xxxx * S (qq: 00 Y1, 01 X0, 10 Y0, 11 X1) into D; variant: 0 MPYI 1 MACI 2 MPYRI 3 MACRI.
            static constexpr u8 kQq[4] = {7, 4, 6, 5};
            const i64 a = sext24(ext), b = sext24(read_reg5(kQq[d.f('q') & 3u], false));
            i64 product = sext56((a * b) << 1);
            if (d.f('k')) product = -product;
            i64& D = acc(d.f('d'));
            i64 r = (info.variant & 1u) ? sext56(D + product) : product;
            if (info.variant & 2u) r = round56(r);
            D = r;
            set_ccr_result(D, false, false, false);
            return;
        }
        case Op::Tcc1: case Op::Tcc2: case Op::Tcc3: {
            if (!condition(d.f('C'))) return;
            if (info.op != Op::Tcc3) acc(d.f('d')) = source56(d.f('J'), d.f('d'));
            if (info.op != Op::Tcc1) r_.r[d.f('T') & 7u] = r_.r[d.f('t') & 7u];
            return;
        }
        case Op::Inc: case Op::Dec: {
            i64& D = acc(d.f('d'));
            D = sext56(D + (info.op == Op::Inc ? 1 : -1));
            set_ccr_result(D, false, false, false);
            return;
        }
        // ---- the non-parallel arithmetic (chapter 13; register tables 12-13 and 12-16) ----
        case Op::Cmpu: {
            // S2 - S1 on 48 unsigned bits: S1 a 24-bit register left-aligned, or the other accumulator's bits 47-0.
            static constexpr u8 kGgg[8] = {0, 0, 0, 0, 4, 6, 5, 7};
            constexpr u64 kMask48 = (u64{1} << 48) - 1;
            const u32 g = d.f('g') & 7u;
            const u64 s2 = static_cast<u64>(acc(d.f('d'))) & kMask48;
            const u64 s1 = g == 0 ? static_cast<u64>(acc(d.f('d') ^ 1u)) & kMask48 : static_cast<u64>(read_reg5(kGgg[g], false)) << 24;
            const u64 r = (s2 - s1) & kMask48;
            u32 sr = r_.sr & ~(N | Z | V | C);
            if ((r >> 47) & 1u) sr |= N;
            if (r == 0) sr |= Z;
            if (s2 < s1) sr |= C;
            r_.sr = sr;
            return;
        }
        case Op::Div: {
            // One non-restoring step: shift D left with C in, then add or subtract the divisor in the MSP by sign.
            static constexpr u8 kJj[4] = {4, 6, 5, 7};
            i64& D = acc(d.f('d'));
            const u32 s = read_reg5(kJj[d.f('J') & 3u], false);
            const bool differ = ((D >> 55) & 1) != ((s >> 23) & 1u);
            const i64 shifted = sext56((D << 1) | ((r_.sr & C) ? 1 : 0));
            const bool v = ((D >> 55) & 1) != ((shifted >> 55) & 1);
            const i64 r = sext56(differ ? shifted + (sext24(s) << 24) : shifted - (sext24(s) << 24));
            D = r;
            u32 sr = r_.sr & ~(V | C);
            if (v) sr |= V | L;
            if (!((r >> 55) & 1)) sr |= C;
            r_.sr = sr;
            return;
        }
        case Op::MulShift: {
            // (±)(S * 2^-n): S [Y1,X0,Y0,X1] left-aligned and shifted right n; variant 0 MPY 1 MAC 2 MPYR 3 MACR.
            static constexpr u8 kQq[4] = {7, 4, 6, 5};
            const u32 n = d.f('s') & 31u;
            i64 p = (sext24(read_reg5(kQq[d.f('Q') & 3u], false)) << 24) >> n;
            if (d.f('k')) p = -p;
            i64& D = acc(d.f('d'));
            const bool accumulate = info.variant & 1u;
            i64 r = accumulate ? sext56(D + p) : sext56(p);
            const bool overflow = accumulate && ((D < 0) == (p < 0)) && ((r < 0) != (D < 0));
            if (info.variant & 2u) r = round56(r);
            D = r;
            set_ccr_result(D, false, false, overflow);
            return;
        }
        case Op::MulMixed: case Op::Dmac: {
            // S1 * S2 with either operand unsigned (QQQQ pairs). MPY/MAC su|uu: s = 0 su, 1 uu. DMAC: ss 00, su 10,
            // uu 11, the accumulator shifted right 24 before accumulating.
            static constexpr u8 kQqqq[16][2] = {{4, 4}, {6, 6}, {5, 4}, {7, 6}, {4, 7}, {6, 4}, {5, 6}, {7, 5},
                                                {5, 5}, {7, 7}, {4, 5}, {6, 7}, {7, 4}, {4, 6}, {6, 5}, {5, 7}};
            const u32 q = d.f('Q') & 15u;
            const u32 a = read_reg5(kQqqq[q][0], false), b = read_reg5(kQqqq[q][1], false);
            u32 mode;  // 0 ss, 1 su, 2 uu
            if (info.op == Op::Dmac) {
                const u32 ss = d.f('s') & 3u;
                mode = ss == 0 ? 0u : (ss == 3 ? 2u : 1u);
            } else {
                mode = (d.f('s') & 1u) ? 2u : 1u;
            }
            const i64 x = mode == 2 ? static_cast<i64>(a) : sext24(a);
            const i64 y = mode == 0 ? sext24(b) : static_cast<i64>(b);
            i64 p = (x * y) << 1;
            if (d.f('k')) p = -p;
            i64& D = acc(d.f('d'));
            const i64 base = info.op == Op::Dmac ? (D >> 24) : D;
            const bool accumulate = info.op == Op::Dmac || (info.variant & 1u);
            const i64 r = accumulate ? sext56(base + p) : sext56(p);
            const bool overflow = accumulate && ((base < 0) == (p < 0)) && ((r < 0) != (base < 0));
            D = r;
            set_ccr_result(D, false, false, overflow);
            return;
        }
        case Op::Norm: {
            // One normalisation step from the E, U and Z bits as they stand.
            i64& D = acc(d.f('d'));
            const u32 rn = d.f('R') & 7u;
            bool v = false;
            if (!(r_.sr & (E | U | Z))) {
                const i64 r = sext56(D << 1);
                v = ((D >> 55) & 1) != ((r >> 55) & 1);
                D = r;
                r_.r[rn] = agu_update(r_.r[rn], r_.m[rn], -1);
            } else if (r_.sr & E) {
                D >>= 1;
                r_.r[rn] = agu_update(r_.r[rn], r_.m[rn], 1);
            } else {
                return;
            }
            set_ccr_result(D, false, false, v);
            return;
        }
        case Op::Normf: {
            // Shift D by the signed amount in S: right when positive, left when negative (CLB's count).
            static constexpr u8 kSss[8] = {0, 0, 12, 13, 4, 6, 5, 7};
            const i32 amount = sext_bits(read_reg5(kSss[d.f('s') & 7u], false), 24);
            i64& D = acc(d.f('D'));
            bool v = false;
            if (amount >= 0) {
                D = amount >= 56 ? (D < 0 ? -1 : 0) : (D >> amount);
            } else {
                const u32 n = static_cast<u32>(std::min(-amount, 56));
                const i64 r = n >= 56 ? 0 : sext56(D << n);
                const i64 top = n >= 56 ? D : (D >> (55 - n));  // the bits that pass through bit 55, and the sign
                v = top != 0 && top != -1;
                D = r;
            }
            set_ccr_result(D, false, false, v);
            return;
        }
        case Op::Clb: {
            // 9 minus the run of leading bits equal to bit 55 into the MSP, LSP cleared, EXP sign-extended.
            const i64 s = acc(d.f('S'));
            i32 result = 0;
            if (s != 0) {
                const i64 lead = (s >> 55) & 1;
                i32 run = 0;
                for (int bit = 55; bit >= 0 && ((s >> bit) & 1) == lead; --bit) ++run;
                result = 9 - run;
            }
            i64& D = acc(d.f('D'));
            D = static_cast<i64>(result) << 24;
            u32 sr = r_.sr & ~(N | Z | V);
            if (result < 0) sr |= N;
            if (result == 0) sr |= Z;
            r_.sr = sr;
            return;
        }
        case Op::Extract: case Op::Insert: {
            // Control word: offset in bits 5-0, width in bits 17-12, from S1 [A1,B1,X0,Y0,X1,Y1] or the extension word.
            static constexpr u8 kSss[8] = {0, 0, 12, 13, 4, 6, 5, 7};
            static constexpr u8 kQqq[8] = {0, 0, 8, 9, 4, 6, 5, 7};
            const u32 control = (info.variant & 2u) ? (ext & kWord) : read_reg5(kSss[d.f('S') & 7u], false);
            const u32 offset = control & 63u, width = (control >> 12) & 63u;
            const u64 mask = width >= 64 ? ~u64{0} : ((u64{1} << width) - 1);
            i64& D = acc(d.f('D'));
            if (info.op == Op::Extract) {
                const u64 src = static_cast<u64>(acc(d.f('s'))) & ((u64{1} << 56) - 1);
                const u64 field = offset >= 56 ? 0 : (src >> offset) & mask;
                i64 r = static_cast<i64>(field);
                if (!(info.variant & 1u) && width > 0 && width < 56 && ((field >> (width - 1)) & 1u)) r = static_cast<i64>(field | ~mask);
                D = sext56(r);
            } else {
                const u64 src = static_cast<u64>(read_reg5(kQqq[d.f('q') & 7u], false)) & mask;
                const u64 hole = offset >= 56 ? 0 : (mask << offset);
                D = sext56(static_cast<i64>((static_cast<u64>(D) & ~hole) | ((src << offset) & hole)));
            }
            set_ccr_result(D, true, false, false);
            return;
        }
        case Op::Merge: {
            // S[11:0] above D[35:24] in the MSP; the rest of D unchanged.
            static constexpr u8 kSss[8] = {0, 0, 12, 13, 4, 6, 5, 7};
            const u32 s = read_reg5(kSss[d.f('S') & 7u], false);
            i64& D = acc(d.f('D'));
            const u32 d1 = static_cast<u32>(D >> 24) & kWord;
            const u32 m = ((s & 0xFFFu) << 12) | (d1 & 0xFFFu);
            D = sext56((D & ~(i64{kWord} << 24)) | (static_cast<i64>(m) << 24));
            set_ccr_logical(D);
            return;
        }
        case Op::Lra:
            write_reg5(d.f('d'), (at + (info.variant ? ext : r_.r[d.f('R') & 7u])) & kWord);
            return;
        case Op::Brkcc:
            // Leave the current DO loop: the loop's stack entries go, execution continues after its last instruction.
            if (condition(d.f('C'))) {
                const u32 after = (r_.la + 1) & kWord;
                u32 pc_value, sr_value;
                pop(pc_value, sr_value);
                r_.sr = (r_.sr & ~(LF | FV)) | (sr_value & (LF | FV));
                u32 la, lc;
                pop(la, lc);
                r_.la = la;
                r_.lc = lc;
                jump_sub(after, false);
            }
            return;
        case Op::Trapcc:
            if (condition(d.f('C'))) fail(std::format("TRAPcc at P:{:04x} (software interrupts are not modelled)", at));
            return;
        case Op::Nop2:  // DEBUGcc and the instruction-cache controls: no debugger, no cache
            return;
        case Op::Unknown:
            fail(std::format("{} {} at P:{:04x} is not implemented", kOpcodePatterns[d.pattern].mnemonic, kOpcodePatterns[d.pattern].operands, at));
            return;
    }
}

u32 Core::step() {
    if (!error_.empty() || halted_) return 0;
    const u32 at = r_.pc & kWord;
    if (at >= cfg_.p_words) {
        fail(std::format("PC {:06x} outside program memory", at));
        return 0;
    }
    const Decoded d = decoded(at);
    const u32 ext = d.extension ? read_p(at + 1) : 0;
    const u32 words = d.extension ? 2 : 1;
    if (rep_active_) {
        // The repeated instruction runs LC times from one fetch, then LC is restored.
        execute(d, ext, at);
        if (--rep_remaining_ == 0) {
            rep_active_ = false;
            r_.lc = rep_saved_lc_;
            r_.pc = (at + words) & kWord;
        }
        ++instructions_;
        cycles_ += 1;
        return 1;
    }
    r_.pc = (at + words) & kWord;
    execute(d, ext, at);
    ++instructions_;
    cycles_ += words;
    // The loop address names the LAST word of the loop's final instruction (the kernel's DOR at P:03CA ends on
    // the extension word of a two-word ADD).
    if (!rep_active_) end_of_loop_check(at + words - 1);
    return words;
}

u64 Core::run(u64 cycles) {
    u64 done = 0;
    while (done < cycles && error_.empty() && !halted_) {
        const u32 c = step();
        if (c == 0) break;
        done += c;
    }
    return done;
}

}  // namespace mv::dsp
