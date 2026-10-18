#pragma once
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "ide/dvd_image.h"
#include "core/clock.h"
#include "ide/ide.h"

namespace mv {

// The hard disk: an ATA-5 class drive over a raw image, with the security feature set the console relies on.
// It powers up locked; the kernel unlocks it with the password it derives from the EEPROM's HDD key and the
// drive's model and serial. The expected password comes from an oracle the machine installs (it knows the
// EEPROM and the kernel's key); without one, any password unlocks.
class AtaDisk : public IdeDevice {
public:
    static constexpr std::size_t kMaxMultiple = 16;
    using PasswordOracle = std::function<std::optional<std::array<u8, 32>>()>;

    explicit AtaDisk(DiskImage& image);
    std::string_view name() const override { return "hdd"; }
    bool is_atapi() const override { return false; }

    void set_identity(std::string model, std::string serial, std::string firmware);
    const std::string& model() const { return model_; }
    const std::string& serial() const { return serial_; }
    void set_password_oracle(PasswordOracle o) { oracle_ = std::move(o); }
    void set_locked(bool locked) {
        security_enabled_ = locked;
        locked_ = locked;
    }
    bool locked() const { return locked_; }

    void reset() override;
    void hardware_reset() override;
    Cycles start_command(u8 command) override;
    void continue_command() override;
    u16 read_data() override;
    void write_data(u16 value) override;

    // The 512-byte block an IDENTIFY DEVICE returns.
    void build_identify(u8* out) const;
    // The password the console's kernel computes for this drive and the given HDD key.
    static std::array<u8, 32> password_for(std::span<const u8, 16> hdd_key, const std::string& model, const std::string& serial);

    u64 unlock_attempts() const { return unlock_attempts_; }
    bool ever_unlocked() const { return ever_unlocked_; }

private:
    enum class Pending { None, IdentifyReady, ReadReady, WriteDone, Complete, DmaRead, DmaWrite, SecurityData };
    void finish_ok();
    void finish_error(u8 error);
    void present_sectors(u32 count);
    void begin_pio_in(u32 bytes);
    void begin_pio_out(u32 bytes);
    u64 lba() const;
    void set_lba(u64 v);
    u32 count() const { return tf_.sector_count == 0 ? 256u : tf_.sector_count; }
    Cycles transfer_time(u32 sectors) const;
    void handle_security_data();

    DiskImage& image_;
    std::string model_{"MIDWAY VERDANT HD 8GB"}, serial_{"MV0000000001"}, firmware_{"MV01"};
    PasswordOracle oracle_;
    bool security_enabled_{true}, locked_{true}, frozen_{false};
    std::optional<std::array<u8, 32>> user_password_;  // set by SECURITY SET PASSWORD in this session
    u8 multiple_{0};
    u8 command_{0};
    Pending pending_{Pending::None};
    std::vector<u8> buffer_;
    std::size_t buffer_pos_{0};
    u32 sectors_left_{0};
    u64 current_lba_{0};
    u32 chunk_sectors_{0};
    u64 unlock_attempts_{0};
    bool ever_unlocked_{false};
};

// The DVD drive: an ATAPI device, empty or holding a game disc image. With a disc it answers the reads, the
// structure reads and the security page the console's kernel uses; an image carries no security sectors, so the
// drive reports the disc as already authenticated with the game partition selected.
class AtapiCdrom : public IdeDevice {
public:
    AtapiCdrom();
    std::string_view name() const override { return "dvd"; }
    bool is_atapi() const override { return true; }
    void reset() override;
    // Power-on or the reset line: the disc spins up again.
    void hardware_reset() override;
    // After power-on, the reset line or a tray load the drive spins the disc up and identifies it before it can be
    // read: until then a media command ends with NOT READY, LOGICAL UNIT IS IN PROCESS OF BECOMING READY (02/04/01),
    // the standard MMC answer, and the host polls. The time is a typical drive's [verify on a console]; without it the
    // kernel reads the game's header before its boot animation can take the game's logo.
    static constexpr u64 kSpinUpNs = 2'000'000'000;
    bool spinning_up() const;
    Cycles start_command(u8 command) override;
    void continue_command() override;
    u16 read_data() override;
    void write_data(u16 value) override;
    u64 packets() const { return packets_; }
    // Puts a disc in the tray (before power-on: the kernel finds it at initialisation).
    void insert(std::unique_ptr<DvdImage> disc) { disc_ = std::move(disc); }
    bool has_disc() const { return disc_ != nullptr; }
    u64 sectors_read() const { return sectors_read_; }

private:
    enum class Pending { None, IdentifyReady, PacketReady, DataReady, DmaData, DataOut, DmaDataOut, Complete };
    void build_identify(u8* out) const;
    void run_packet();
    void mode_sense10();
    void read_sectors(u64 lba, u32 count);
    void read_dvd_structure();
    void data_out(u32 len);
    void data_out_done();
    void next_pio_chunk();
    void pio_chunk_done();
    void data_in(const u8* data, u32 len, Cycles latency = 0);  // latency 0: the command overhead alone
    void check_condition(u8 sense_key, u8 asc, u8 ascq);
    void finish_ok();

    Pending pending_{Pending::None};
    std::array<u8, 12> cdb_{};
    std::size_t cdb_pos_{0};
    std::vector<u8> buffer_;
    std::size_t buffer_pos_{0};
    std::vector<u8> payload_;      // a PIO data-in phase larger than the host's byte count goes out in chunks
    std::size_t payload_pos_{0};
    std::unique_ptr<DvdImage> disc_;
    std::array<u8, 20> security_page_{};  // the last security page the host wrote (MODE SELECT page 0x3E)
    u64 sectors_read_{0};
    u8 sense_key_{0}, asc_{0}, ascq_{0};
    bool dma_{false};  // the current PACKET command's data phase goes through the bus master (features bit 0)
    u64 packets_{0};
    Cycles ready_at_{cycles_from_ns(kSpinUpNs)};  // power-on is time 0
};

}  // namespace mv
