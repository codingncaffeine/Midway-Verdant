#include "smbus/smc.h"

#include "debug/trace.h"

namespace mv {

Smc::Smc(Revision rev) : revision_(rev) {
    regs_[kRegCpuTemp] = 45;
    regs_[kRegBoardTemp] = 40;
    regs_[kRegFanSpeed] = 20;
    // The bootloader shuts the console down if every challenge byte is zero; a real SMC never hands out zeros.
    regs_[kRegChallenge0 + 0] = 0x52;
    regs_[kRegChallenge0 + 1] = 0x72;
    regs_[kRegChallenge0 + 2] = 0xEA;
    regs_[kRegChallenge0 + 3] = 0x46;
}

std::string Smc::version_string() const {
    switch (revision_) {
        case Revision::P01: return "P01";
        case Revision::P05: return "P05";
        case Revision::P11: return "P11";
        case Revision::P2L: return "P2L";
        case Revision::D01: return "D01";
    }
    return "P01";
}

std::optional<u8> Smc::read_byte(u8 command) {
    switch (command) {
        case kRegVersion: {
            const std::string v = version_string();
            const u8 c = static_cast<u8>(v[version_cursor_ % 3]);
            version_cursor_ = static_cast<u8>((version_cursor_ + 1) % 3);
            return c;
        }
        case kRegTrayState: return static_cast<u8>(tray_);
        case kRegAvPack: return static_cast<u8>(av_pack_);
        case kRegInterruptReason: {
            const u8 v = regs_[kRegInterruptReason];
            regs_[kRegInterruptReason] = 0;  // reading acknowledges
            return v;
        }
        default:
            if (command < regs_.size()) return regs_[command];
            trace::log(trace::Cat::Smbus, "smc read of unknown register {:02x}", command);
            return std::nullopt;
    }
}

bool Smc::write_byte(u8 command, u8 value) {
    switch (command) {
        case kRegVersion:
            version_cursor_ = 0;
            return true;
        case kRegReset:
            trace::log(trace::Cat::Boot, "SMC reset/power request {:02x}", value);
            if (on_reset_) on_reset_(value);
            return true;
        case kRegTrayEject:
            trace::log(trace::Cat::Boot, "SMC tray {}", value == 0 ? "eject" : "load");
            if (on_tray_) on_tray_(value == 0);
            return true;
        case kRegLedMode:
        case kRegLedSequence:
            regs_[command] = value;
            trace::log(trace::Cat::Boot, "SMC LED {} <- {:02x}", command == kRegLedMode ? "mode" : "sequence", value);
            return true;
        default:
            if (command == kRegResponse0 || command == kRegResponse0 + 1) {
                regs_[command] = value;
                if (++response_written_ == 2)
                    trace::log(trace::Cat::Boot, "SMC watchdog challenge answered: response {:04x}", response());
                return true;
            }
            if (command < regs_.size()) {
                regs_[command] = value;
                return true;
            }
            trace::log(trace::Cat::Smbus, "smc write to unknown register {:02x} <- {:02x}", command, value);
            return false;
    }
}

std::optional<u8> Adm1032::read_byte(u8 command) {
    switch (command) {
        case 0x00: return u8{40};  // local temperature
        case 0x01: return u8{45};  // remote (CPU) temperature
        case 0x02: return u8{0};   // status
        case 0x03: return regs_[0x03];
        case 0xFE: return u8{0x41};  // manufacturer ID
        case 0xFF: return u8{0x42};  // die revision
        default: return regs_[command];
    }
}

bool Adm1032::write_byte(u8 command, u8 value) {
    regs_[command] = value;
    return true;
}

VideoEncoder::VideoEncoder(Kind kind) : kind_(kind) {}

std::string_view VideoEncoder::name() const {
    switch (kind_) {
        case Kind::Conexant: return "conexant";
        case Kind::Focus: return "focus";
        case Kind::Xcalibur: return "xcalibur";
    }
    return "encoder";
}

std::optional<u8> VideoEncoder::read_byte(u8 command) { return regs_[command]; }

bool VideoEncoder::write_byte(u8 command, u8 value) {
    regs_[command] = value;
    return true;
}

}  // namespace mv
