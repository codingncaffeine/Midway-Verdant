#pragma once
#include <array>
#include <functional>
#include <string>

#include "smbus/smbus.h"

namespace mv {

// The System Management Controller, a PIC16LC63A at SMBus address 0x10 (Xyclops on 1.6 boards). It owns power,
// reset, the front LED, the fan, the tray, the AV pack sense, the temperatures, the second bootloader's watchdog
// challenge, and interrupts the CPU through a GPIO line for buttons and tray events. Register numbers follow
// the community documentation; those marked [verify] in local-notes/hw/smc.md are confirmed by traces.
class Smc : public SmbusDevice {
public:
    enum class Revision { P01, P05, P11, P2L, D01 };
    enum class AvPack : u8 { Scart = 0x00, Hdtv = 0x01, Vga = 0x02, Rfu = 0x03, SVideo = 0x04, Undefined = 0x05, Composite = 0x06, None = 0x07 };
    enum class Tray : u8 { Busy = 0x00, Open = 0x10, ClosedNoMedia = 0x40, ClosedMedia = 0x60 };

    // Register numbers.
    static constexpr u8 kRegVersion = 0x01, kRegReset = 0x02, kRegTrayState = 0x03, kRegAvPack = 0x04, kRegFanMode = 0x05,
                        kRegFanSpeed = 0x06, kRegLedMode = 0x07, kRegLedSequence = 0x08, kRegCpuTemp = 0x09, kRegBoardTemp = 0x0A,
                        kRegTrayEject = 0x0C, kRegInterruptReason = 0x11, kRegResetOnEject = 0x19, kRegInterruptEnable = 0x1A,
                        kRegScratch = 0x1B, kRegChallenge0 = 0x1C, kRegResponse0 = 0x20;
    static constexpr u8 kResetBitReset = 0x01, kResetBitPowerCycle = 0x40, kResetBitPowerOff = 0x80;

    explicit Smc(Revision rev = Revision::P01);
    std::string_view name() const override { return "smc"; }

    std::optional<u8> read_byte(u8 command) override;
    bool write_byte(u8 command, u8 value) override;

    void set_av_pack(AvPack p) { av_pack_ = p; }
    void set_tray(Tray t) { tray_ = t; }
    Tray tray() const { return tray_; }
    u8 led_mode() const { return regs_[kRegLedMode]; }
    u8 led_sequence() const { return regs_[kRegLedSequence]; }
    u8 fan_speed() const { return regs_[kRegFanSpeed]; }
    u8 scratch() const { return regs_[kRegScratch]; }
    bool challenge_answered() const { return response_written_ == 2; }
    u16 response() const { return static_cast<u16>(regs_[kRegResponse0] | (regs_[kRegResponse0 + 1] << 8)); }

    // Power and reset requests from the guest.
    void set_on_reset_request(std::function<void(u8 bits)> cb) { on_reset_ = std::move(cb); }
    // Eject/load requests reach the drive through this.
    void set_on_tray_command(std::function<void(bool eject)> cb) { on_tray_ = std::move(cb); }

private:
    std::string version_string() const;

    Revision revision_;
    std::array<u8, 0x22> regs_{};
    AvPack av_pack_{AvPack::Hdtv};
    Tray tray_{Tray::ClosedNoMedia};
    u8 version_cursor_{0};
    int response_written_{0};
    std::function<void(u8)> on_reset_;
    std::function<void(bool)> on_tray_;
};

// The 256-byte serial EEPROM at SMBus address 0x54. Byte and word reads at the command offset; writes land in
// the image, which a client may persist.
class Eeprom : public SmbusDevice {
public:
    std::string_view name() const override { return "eeprom"; }
    std::optional<u8> read_byte(u8 command) override {
        provision_if_needed();
        return image_[command];
    }
    bool write_byte(u8 command, u8 value) override {
        provision_if_needed();
        image_[command] = value;
        dirty_ = true;
        return true;
    }
    // An image loaded from the user's file is final; an empty one is provisioned by the machine the first time
    // the kernel touches it (by then the kernel, and its EEPROM key, are in RAM).
    void set_provisioner(std::function<bool(std::array<u8, 256>&)> p) { provisioner_ = std::move(p); }
    void mark_provisioned() { provisioned_ = true; }
    bool provisioned() const { return provisioned_; }
    std::array<u8, 256>& image() { return image_; }
    const std::array<u8, 256>& image() const { return image_; }
    bool dirty() const { return dirty_; }
    void clear_dirty() { dirty_ = false; }

private:
    void provision_if_needed() {
        if (provisioned_) return;
        provisioned_ = true;
        if (provisioner_ && provisioner_(image_)) dirty_ = true;
    }
    std::array<u8, 256> image_{};
    bool dirty_{false};
    bool provisioned_{false};
    std::function<bool(std::array<u8, 256>&)> provisioner_;
};

// The ADM1032 temperature monitor at 0x4C (not on 1.6 boards): local and remote temperatures and the usual
// limit registers.
class Adm1032 : public SmbusDevice {
public:
    std::string_view name() const override { return "adm1032"; }
    std::optional<u8> read_byte(u8 command) override;
    bool write_byte(u8 command, u8 value) override;

private:
    std::array<u8, 256> regs_{};
};

// A video encoder that answers its probes and stores its registers; the kernel configures it over SMBus and
// nothing downstream reads it yet. One class, three identities (Conexant 0x45, Focus 0x6A, Xcalibur 0x70).
class VideoEncoder : public SmbusDevice {
public:
    enum class Kind { Conexant, Focus, Xcalibur };
    explicit VideoEncoder(Kind kind);
    std::string_view name() const override;
    std::optional<u8> read_byte(u8 command) override;
    bool write_byte(u8 command, u8 value) override;
    Kind kind() const { return kind_; }
    u8 reg(u8 r) const { return regs_[r]; }

private:
    Kind kind_;
    std::array<u8, 256> regs_{};
};

}  // namespace mv
