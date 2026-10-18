#include "ide/ata_disk.h"

#include <algorithm>
#include <cstring>

#include "core/clock.h"
#include "crypto/sha1.h"
#include "debug/trace.h"

namespace mv {

namespace {
// IDENTIFY strings: pairs of characters swapped within each 16-bit word, space-padded.
void put_string(u8* words, std::size_t nwords, const std::string& s) {
    for (std::size_t i = 0; i < nwords; ++i) {
        const char a = 2 * i < s.size() ? s[2 * i] : ' ';
        const char b = 2 * i + 1 < s.size() ? s[2 * i + 1] : ' ';
        words[2 * i] = static_cast<u8>(b);
        words[2 * i + 1] = static_cast<u8>(a);
    }
}
void put16(u8* p, u16 v) {
    p[0] = static_cast<u8>(v);
    p[1] = static_cast<u8>(v >> 8);
}
void put32(u8* p, u32 v) {
    put16(p, static_cast<u16>(v));
    put16(p + 2, static_cast<u16>(v >> 16));
}
std::string trimmed_trailing_spaces(std::string s) {
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}
constexpr Cycles kCommandLatency = 60'000;  // ~80 us: a drive's command overhead
constexpr u64 kDvdNsPerByte = 125;          // ~8 MB/s: the drive reading a spinning disc
u32 be32(const u8* p) { return (static_cast<u32>(p[0]) << 24) | (static_cast<u32>(p[1]) << 16) | (static_cast<u32>(p[2]) << 8) | p[3]; }
void put_be16(u8* p, u32 v) {
    p[0] = static_cast<u8>(v >> 8);
    p[1] = static_cast<u8>(v);
}
void put_be24(u8* p, u32 v) {
    p[0] = static_cast<u8>(v >> 16);
    put_be16(p + 1, v);
}
}  // namespace

AtaDisk::AtaDisk(DiskImage& image) : image_(image) {}

void AtaDisk::set_identity(std::string model, std::string serial, std::string firmware) {
    model_ = std::move(model);
    serial_ = std::move(serial);
    firmware_ = std::move(firmware);
}

std::array<u8, 32> AtaDisk::password_for(std::span<const u8, 16> hdd_key, const std::string& model, const std::string& serial) {
    // HMAC-SHA1 over the model then the serial as they come out of IDENTIFY (byte order restored, trailing
    // spaces dropped), padded with zeros to the 32-byte ATA password.
    const std::string m = trimmed_trailing_spaces(model);
    const std::string s = trimmed_trailing_spaces(serial);
    const auto h = crypto::hmac_sha1(hdd_key, std::span<const u8>(reinterpret_cast<const u8*>(m.data()), m.size()),
                                     std::span<const u8>(reinterpret_cast<const u8*>(s.data()), s.size()));
    std::array<u8, 32> pw{};
    std::memcpy(pw.data(), h.data(), 20);
    return pw;
}

void AtaDisk::reset() {
    tf_.error = 0x01;
    tf_.sector_count = 1;
    tf_.lba_low = 1;
    tf_.lba_mid = 0;
    tf_.lba_high = 0;
    tf_.status = ata::kStatusDrdy | ata::kStatusDsc;
    pending_ = Pending::None;
    buffer_.clear();
}

void AtaDisk::hardware_reset() {
    // Only a power cycle or a hardware reset locks the drive again and lifts a freeze; SRST does neither.
    frozen_ = false;
    locked_ = security_enabled_;
    reset();
}

u64 AtaDisk::lba() const {
    if (tf_.device & 0x40) {
        return static_cast<u64>(tf_.lba_low) | (static_cast<u64>(tf_.lba_mid) << 8) | (static_cast<u64>(tf_.lba_high) << 16) |
               (static_cast<u64>(tf_.device & 0x0F) << 24);
    }
    // CHS with the default 16 heads x 63 sectors geometry.
    const u64 cyl = static_cast<u64>(tf_.lba_mid) | (static_cast<u64>(tf_.lba_high) << 8);
    const u64 head = tf_.device & 0x0F;
    const u64 sector = tf_.lba_low ? tf_.lba_low : 1;
    return (cyl * 16 + head) * 63 + sector - 1;
}

void AtaDisk::set_lba(u64 v) {
    if (tf_.device & 0x40) {
        tf_.lba_low = static_cast<u8>(v);
        tf_.lba_mid = static_cast<u8>(v >> 8);
        tf_.lba_high = static_cast<u8>(v >> 16);
        tf_.device = static_cast<u8>((tf_.device & 0xF0) | ((v >> 24) & 0x0F));
    } else {
        const u64 sector = v % 63 + 1;
        const u64 head = (v / 63) % 16;
        const u64 cyl = v / (63 * 16);
        tf_.lba_low = static_cast<u8>(sector);
        tf_.lba_mid = static_cast<u8>(cyl);
        tf_.lba_high = static_cast<u8>(cyl >> 8);
        tf_.device = static_cast<u8>((tf_.device & 0xF0) | head);
    }
}

Cycles AtaDisk::transfer_time(u32 sectors) const {
    // Ultra DMA mode 2 (33 MB/s) plus command overhead; rotational and seek time are not modelled yet [plan: Phase 11].
    return kCommandLatency + cycles_from_ns(static_cast<u64>(sectors) * 512u * 1000u / 33u);
}

void AtaDisk::finish_ok() {
    tf_.status = ata::kStatusDrdy | ata::kStatusDsc;
    tf_.error = 0;
    pending_ = Pending::None;
    channel_->raise_interrupt();
}

void AtaDisk::finish_error(u8 error) {
    tf_.status = ata::kStatusDrdy | ata::kStatusDsc | ata::kStatusErr;
    tf_.error = error;
    pending_ = Pending::None;
    buffer_.clear();
    channel_->raise_interrupt();
}

void AtaDisk::begin_pio_in(u32 bytes) {
    buffer_.resize(bytes);
    buffer_pos_ = 0;
}

void AtaDisk::begin_pio_out(u32 bytes) {
    buffer_.assign(bytes, 0);
    buffer_pos_ = 0;
}

void AtaDisk::build_identify(u8* out) const {
    std::memset(out, 0, 512);
    const u64 total = std::min<u64>(image_.sectors(), 0x0FFFFFFFu);
    const u32 cyl = static_cast<u32>(std::min<u64>(total / (16 * 63), 16383));
    put16(out + 0 * 2, 0x0040);
    put16(out + 1 * 2, static_cast<u16>(cyl));
    put16(out + 3 * 2, 16);
    put16(out + 6 * 2, 63);
    put_string(out + 10 * 2, 10, serial_);
    put_string(out + 23 * 2, 4, firmware_);
    put_string(out + 27 * 2, 20, model_);
    put16(out + 47 * 2, 0x8000 | kMaxMultiple);
    put16(out + 49 * 2, 0x0F00);  // LBA, DMA, IORDY
    put16(out + 51 * 2, 0x0200);
    put16(out + 53 * 2, 0x0007);
    put16(out + 54 * 2, static_cast<u16>(cyl));
    put16(out + 55 * 2, 16);
    put16(out + 56 * 2, 63);
    put32(out + 57 * 2, static_cast<u32>(cyl) * 16 * 63);
    put16(out + 59 * 2, multiple_ ? static_cast<u16>(0x0100 | multiple_) : 0);
    put32(out + 60 * 2, static_cast<u32>(total));
    put16(out + 63 * 2, 0x0407);  // multiword DMA 0-2, mode 2 active
    put16(out + 64 * 2, 0x0003);  // PIO 3 and 4
    put16(out + 65 * 2, 120);
    put16(out + 66 * 2, 120);
    put16(out + 67 * 2, 120);
    put16(out + 68 * 2, 120);
    put16(out + 80 * 2, 0x007E);  // ATA-2 .. ATA-6
    put16(out + 81 * 2, 0x0000);
    put16(out + 82 * 2, 0x4062);  // NOP, look-ahead, write cache, security
    put16(out + 83 * 2, 0x5000);  // FLUSH CACHE
    put16(out + 84 * 2, 0x4000);
    put16(out + 85 * 2, static_cast<u16>(0x4060 | (security_enabled_ ? 0x0002 : 0)));
    put16(out + 86 * 2, 0x1000);
    put16(out + 87 * 2, 0x4000);
    put16(out + 88 * 2, 0x0407);  // UDMA 0-2, mode 2 active
    put16(out + 93 * 2, 0x6001);
    u16 security = 0x0001;
    if (security_enabled_) security |= 0x0002;
    if (locked_) security |= 0x0004;
    if (frozen_) security |= 0x0008;
    security |= 0x0100;  // maximum security level
    put16(out + 128 * 2, security);
    // Integrity word: 0xA5 and a checksum making the byte sum of the block zero.
    out[510] = 0xA5;
    u8 sum = 0;
    for (int i = 0; i < 511; ++i) sum = static_cast<u8>(sum + out[i]);
    out[511] = static_cast<u8>(0x100 - sum);
}

Cycles AtaDisk::start_command(u8 command) {
    command_ = command;
    tf_.status = ata::kStatusBsy | ata::kStatusDsc;
    tf_.error = 0;
    const bool media = (command >= 0x20 && command <= 0x3F) || command == 0x40 || command == 0x41 || (command >= 0xC4 && command <= 0xCB) ||
                       command == 0xE7 || command == 0xEA;
    if (locked_ && media) {
        trace::log(trace::Cat::Pio, "hdd command {:02x} refused: drive locked", command);
        pending_ = Pending::Complete;
        tf_.error = ata::kErrorAbrt;
        return kCommandLatency;
    }
    switch (command) {
        case 0xEC:  // IDENTIFY DEVICE
            begin_pio_in(512);
            build_identify(buffer_.data());
            pending_ = Pending::IdentifyReady;
            return kCommandLatency;
        case 0x20:
        case 0x21:  // READ SECTORS
        case 0xC4:  // READ MULTIPLE
            current_lba_ = lba();
            sectors_left_ = count();
            chunk_sectors_ = command == 0xC4 ? std::max<u32>(multiple_, 1) : 1;
            if (command == 0xC4 && multiple_ == 0) {
                pending_ = Pending::Complete;
                tf_.error = ata::kErrorAbrt;
                return kCommandLatency;
            }
            pending_ = Pending::ReadReady;
            trace::log(trace::Cat::Pio, "hdd read {} sectors at lba {}", sectors_left_, current_lba_);
            return transfer_time(std::min(sectors_left_, chunk_sectors_));
        case 0x30:
        case 0x31:  // WRITE SECTORS
        case 0xC5:  // WRITE MULTIPLE
            current_lba_ = lba();
            sectors_left_ = count();
            chunk_sectors_ = command == 0xC5 ? std::max<u32>(multiple_, 1) : 1;
            if (image_.read_only()) {
                pending_ = Pending::Complete;
                tf_.error = ata::kErrorAbrt;
                return kCommandLatency;
            }
            trace::log(trace::Cat::Pio, "hdd write {} sectors at lba {}", sectors_left_, current_lba_);
            begin_pio_out(std::min(sectors_left_, chunk_sectors_) * 512);
            tf_.status = ata::kStatusDrdy | ata::kStatusDsc | ata::kStatusDrq;  // ready for the first block at once
            pending_ = Pending::WriteDone;
            return ~Cycles{0};
        case 0xC8:
        case 0xC9:  // READ DMA
            current_lba_ = lba();
            sectors_left_ = count();
            trace::log(trace::Cat::Pio, "hdd read dma {} sectors at lba {}", sectors_left_, current_lba_);
            pending_ = Pending::DmaRead;
            return transfer_time(sectors_left_);
        case 0xCA:
        case 0xCB:  // WRITE DMA
            current_lba_ = lba();
            sectors_left_ = count();
            if (image_.read_only()) {
                pending_ = Pending::Complete;
                tf_.error = ata::kErrorAbrt;
                return kCommandLatency;
            }
            trace::log(trace::Cat::Pio, "hdd write dma {} sectors at lba {}", sectors_left_, current_lba_);
            pending_ = Pending::DmaWrite;
            return transfer_time(sectors_left_);
        case 0x40:
        case 0x41:  // READ VERIFY
            pending_ = Pending::Complete;
            return transfer_time(count());
        case 0xC6:  // SET MULTIPLE
            if (tf_.sector_count > kMaxMultiple || (tf_.sector_count & (tf_.sector_count - 1))) {
                tf_.error = ata::kErrorAbrt;
            } else {
                multiple_ = tf_.sector_count;
            }
            pending_ = Pending::Complete;
            return kCommandLatency;
        case 0xEF:  // SET FEATURES
            switch (tf_.features) {
                case 0x02: case 0x82: case 0x03: case 0x55: case 0xAA: case 0x05: case 0x85: case 0x66: case 0xCC: break;
                default: tf_.error = ata::kErrorAbrt; break;
            }
            pending_ = Pending::Complete;
            return kCommandLatency;
        case 0xE5:  // CHECK POWER MODE
            tf_.sector_count = 0xFF;
            pending_ = Pending::Complete;
            return kCommandLatency;
        case 0x10: case 0x70: case 0x91: case 0xE0: case 0xE1: case 0xE2: case 0xE3: case 0xE6: case 0xE7: case 0xEA: case 0x00:
            if (command == 0xE7 || command == 0xEA) image_.flush();
            pending_ = Pending::Complete;
            return kCommandLatency;
        case 0x90:  // EXECUTE DEVICE DIAGNOSTIC
            tf_.error = 0x01;
            pending_ = Pending::Complete;
            return kCommandLatency;
        case 0xF5:  // SECURITY FREEZE LOCK
            frozen_ = true;
            pending_ = Pending::Complete;
            return kCommandLatency;
        case 0xF3:  // SECURITY ERASE PREPARE
            pending_ = Pending::Complete;
            return kCommandLatency;
        case 0xF1:  // SECURITY SET PASSWORD
        case 0xF2:  // SECURITY UNLOCK
        case 0xF4:  // SECURITY ERASE UNIT
        case 0xF6:  // SECURITY DISABLE PASSWORD
            if (frozen_ && command != 0xF2) {
                tf_.error = ata::kErrorAbrt;
                pending_ = Pending::Complete;
                return kCommandLatency;
            }
            begin_pio_out(512);
            tf_.status = ata::kStatusDrdy | ata::kStatusDsc | ata::kStatusDrq;
            pending_ = Pending::SecurityData;
            return ~Cycles{0};
        default:
            trace::log(trace::Cat::Pio, "hdd unknown command {:02x}", command);
            tf_.error = ata::kErrorAbrt;
            pending_ = Pending::Complete;
            return kCommandLatency;
    }
}

void AtaDisk::present_sectors(u32 n) {
    begin_pio_in(n * 512);
    if (!image_.read(current_lba_, n, buffer_.data())) {
        finish_error(ata::kErrorIdnf);
        return;
    }
    channel_->count_sectors(false, n);
    current_lba_ += n;
    sectors_left_ -= n;
    tf_.status = ata::kStatusDrdy | ata::kStatusDsc | ata::kStatusDrq;
    channel_->raise_interrupt();
}

void AtaDisk::continue_command() {
    switch (pending_) {
        case Pending::IdentifyReady:
            tf_.status = ata::kStatusDrdy | ata::kStatusDsc | ata::kStatusDrq;
            channel_->raise_interrupt();
            break;
        case Pending::ReadReady:
            present_sectors(std::min(sectors_left_, chunk_sectors_));
            break;
        case Pending::Complete:
            if (tf_.error)
                finish_error(tf_.error);
            else
                finish_ok();
            break;
        case Pending::DmaRead: {
            if (!channel_->dma_active()) return;  // the engine starts us again
            std::vector<u8> data(static_cast<std::size_t>(sectors_left_) * 512);
            if (!image_.read(current_lba_, sectors_left_, data.data())) {
                channel_->dma_complete(false);
                finish_error(ata::kErrorIdnf);
                return;
            }
            const bool ok = channel_->dma_transfer(data.data(), nullptr, static_cast<u32>(data.size()));
            channel_->count_sectors(false, sectors_left_);
            set_lba(current_lba_ + sectors_left_);
            channel_->dma_complete(ok);
            if (ok) finish_ok(); else finish_error(ata::kErrorAbrt);
            break;
        }
        case Pending::DmaWrite: {
            if (!channel_->dma_active()) return;
            std::vector<u8> data(static_cast<std::size_t>(sectors_left_) * 512);
            const bool ok = channel_->dma_transfer(nullptr, data.data(), static_cast<u32>(data.size())) &&
                            image_.write(current_lba_, sectors_left_, data.data());
            channel_->count_sectors(true, sectors_left_);
            set_lba(current_lba_ + sectors_left_);
            channel_->dma_complete(ok);
            if (ok) finish_ok(); else finish_error(ata::kErrorAbrt);
            break;
        }
        case Pending::WriteDone:
        case Pending::SecurityData:
        case Pending::None:
            break;
    }
}

u16 AtaDisk::read_data() {
    if (buffer_pos_ + 1 >= buffer_.size()) {
        if (buffer_pos_ + 1 == buffer_.size()) ++buffer_pos_;
        // Block consumed.
        if (buffer_pos_ >= buffer_.size()) {
            const u16 v = buffer_.size() >= 2 ? static_cast<u16>(buffer_[buffer_.size() - 2] | (buffer_[buffer_.size() - 1] << 8)) : 0;
            tf_.status = ata::kStatusDrdy | ata::kStatusDsc;
            if (pending_ == Pending::ReadReady && sectors_left_ > 0) {
                tf_.status |= ata::kStatusBsy;
                channel_->schedule(*this, transfer_time(std::min(sectors_left_, chunk_sectors_)));
            } else {
                set_lba(current_lba_);
                pending_ = Pending::None;
            }
            buffer_.clear();
            return v;
        }
    }
    const u16 v = static_cast<u16>(buffer_[buffer_pos_] | (buffer_[buffer_pos_ + 1] << 8));
    buffer_pos_ += 2;
    if (buffer_pos_ >= buffer_.size()) {
        tf_.status = ata::kStatusDrdy | ata::kStatusDsc;
        if (pending_ == Pending::ReadReady && sectors_left_ > 0) {
            tf_.status |= ata::kStatusBsy;
            channel_->schedule(*this, transfer_time(std::min(sectors_left_, chunk_sectors_)));
        } else {
            set_lba(current_lba_);
            pending_ = Pending::None;
        }
        buffer_.clear();
    }
    return v;
}

void AtaDisk::write_data(u16 value) {
    if (buffer_pos_ + 1 >= buffer_.size()) return;
    buffer_[buffer_pos_] = static_cast<u8>(value);
    buffer_[buffer_pos_ + 1] = static_cast<u8>(value >> 8);
    buffer_pos_ += 2;
    if (buffer_pos_ < buffer_.size()) return;
    // A full block arrived.
    tf_.status = ata::kStatusBsy | ata::kStatusDsc;
    if (pending_ == Pending::SecurityData) {
        handle_security_data();
        return;
    }
    if (pending_ == Pending::WriteDone) {
        const u32 n = static_cast<u32>(buffer_.size() / 512);
        if (!image_.write(current_lba_, n, buffer_.data())) {
            finish_error(ata::kErrorAbrt);
            return;
        }
        channel_->count_sectors(true, n);
        current_lba_ += n;
        sectors_left_ -= n;
        if (sectors_left_ == 0) {
            set_lba(current_lba_);
            buffer_.clear();
            channel_->schedule(*this, transfer_time(n));
            pending_ = Pending::Complete;
        } else {
            begin_pio_out(std::min(sectors_left_, chunk_sectors_) * 512);
            tf_.status = ata::kStatusDrdy | ata::kStatusDsc | ata::kStatusDrq;
            channel_->raise_interrupt();
        }
    }
}

void AtaDisk::handle_security_data() {
    std::array<u8, 32> pw{};
    std::memcpy(pw.data(), buffer_.data() + 2, 32);
    const bool master = (buffer_[0] & 0x01) != 0;
    buffer_.clear();
    switch (command_) {
        case 0xF2: {  // UNLOCK
            ++unlock_attempts_;
            std::optional<std::array<u8, 32>> expected = user_password_;
            if (!expected && oracle_) expected = oracle_();
            if (!locked_) {
                pending_ = Pending::Complete;  // unlocking an unlocked drive succeeds
            } else if (master) {
                tf_.error = ata::kErrorAbrt;  // maximum security: the master password cannot unlock
                pending_ = Pending::Complete;
            } else if (!expected || *expected == pw) {
                locked_ = false;
                ever_unlocked_ = true;
                trace::log(trace::Cat::Boot, "HDD unlocked{}", expected ? "" : " (no password oracle: any password accepted)");
                pending_ = Pending::Complete;
            } else {
                trace::log(trace::Cat::Boot, "HDD unlock refused: password mismatch");
                tf_.error = ata::kErrorAbrt;
                pending_ = Pending::Complete;
            }
            break;
        }
        case 0xF1:  // SET PASSWORD
            if (!master) {
                user_password_ = pw;
                security_enabled_ = true;
            }
            pending_ = Pending::Complete;
            break;
        case 0xF6: {  // DISABLE PASSWORD
            std::optional<std::array<u8, 32>> expected = user_password_;
            if (!expected && oracle_) expected = oracle_();
            if (locked_ || (expected && *expected != pw)) {
                tf_.error = ata::kErrorAbrt;
            } else {
                security_enabled_ = false;
                user_password_.reset();
            }
            pending_ = Pending::Complete;
            break;
        }
        case 0xF4:  // ERASE UNIT: accepted, the image is left alone
            security_enabled_ = false;
            locked_ = false;
            pending_ = Pending::Complete;
            break;
        default:
            tf_.error = ata::kErrorAbrt;
            pending_ = Pending::Complete;
            break;
    }
    channel_->schedule(*this, kCommandLatency);
}

// ---------------------------------------------------------------------------------------------------------------

AtapiCdrom::AtapiCdrom() = default;

void AtapiCdrom::reset() {
    tf_.error = 0x01;
    tf_.sector_count = 1;
    tf_.lba_low = 1;
    tf_.lba_mid = 0x14;
    tf_.lba_high = 0xEB;
    tf_.status = 0;  // ATAPI devices show DRDY clear after reset until a command
    pending_ = Pending::None;
    sense_key_ = 0;
    asc_ = 0;
    ascq_ = 0;
}

void AtapiCdrom::build_identify(u8* out) const {
    std::memset(out, 0, 512);
    put16(out + 0 * 2, 0x85C0);  // ATAPI, removable, CD/DVD, 12-byte packets, 50 us DRQ
    put_string(out + 10 * 2, 10, "MVDVD0000001");
    put_string(out + 23 * 2, 4, "MV01");
    put_string(out + 27 * 2, 20, "MIDWAY VERDANT DVD-ROM");
    put16(out + 49 * 2, 0x0B00);  // DMA, LBA, IORDY
    put16(out + 53 * 2, 0x0006);
    put16(out + 63 * 2, 0x0407);
    put16(out + 64 * 2, 0x0003);
    put16(out + 65 * 2, 120);
    put16(out + 66 * 2, 120);
    put16(out + 67 * 2, 120);
    put16(out + 68 * 2, 120);
    put16(out + 71 * 2, 30);
    put16(out + 72 * 2, 30);
    put16(out + 80 * 2, 0x007E);
    put16(out + 82 * 2, 0x4010);  // PACKET command feature set
    put16(out + 83 * 2, 0x4000);
    put16(out + 84 * 2, 0x4000);
    put16(out + 85 * 2, 0x4010);
    put16(out + 87 * 2, 0x4000);
    put16(out + 88 * 2, 0x0407);
}

Cycles AtapiCdrom::start_command(u8 command) {
    tf_.status = ata::kStatusBsy;
    tf_.error = 0;
    switch (command) {
        case 0xA1:  // IDENTIFY PACKET DEVICE
            buffer_.assign(512, 0);
            build_identify(buffer_.data());
            buffer_pos_ = 0;
            pending_ = Pending::IdentifyReady;
            return kCommandLatency;
        case 0xA0:  // PACKET: the 12-byte command block follows on the data port
            cdb_pos_ = 0;
            dma_ = (tf_.features & 0x01) != 0;
            tf_.sector_count = 0x01;  // interrupt reason: command
            tf_.status = ata::kStatusDrdy | ata::kStatusDrq;
            pending_ = Pending::PacketReady;
            return ~Cycles{0};
        case 0x08:  // DEVICE RESET
            reset();
            tf_.status = ata::kStatusDrdy;
            pending_ = Pending::Complete;
            return kCommandLatency;
        case 0xEC:  // IDENTIFY DEVICE on an ATAPI device: abort with the signature
            reset();
            tf_.error = ata::kErrorAbrt;
            tf_.status = ata::kStatusDrdy | ata::kStatusErr;
            pending_ = Pending::None;
            channel_->raise_interrupt();
            return ~Cycles{0};
        case 0xEF:  // SET FEATURES
        case 0xE0: case 0xE1: case 0xE5:
            if (command == 0xE5) tf_.sector_count = 0xFF;
            pending_ = Pending::Complete;
            return kCommandLatency;
        default:
            trace::log(trace::Cat::Pio, "dvd unknown command {:02x}", command);
            tf_.error = ata::kErrorAbrt;
            tf_.status = ata::kStatusDrdy | ata::kStatusErr;
            pending_ = Pending::None;
            channel_->raise_interrupt();
            return ~Cycles{0};
    }
}

void AtapiCdrom::finish_ok() {
    tf_.status = ata::kStatusDrdy;
    tf_.error = 0;
    tf_.sector_count = 0x03;  // interrupt reason: I/O | CoD = command complete
    pending_ = Pending::None;
    channel_->raise_interrupt();
}

void AtapiCdrom::check_condition(u8 sense_key, u8 asc, u8 ascq) {
    sense_key_ = sense_key;
    asc_ = asc;
    ascq_ = ascq;
    tf_.error = static_cast<u8>((sense_key << 4) | ata::kErrorAbrt);
    tf_.status = ata::kStatusDrdy | ata::kStatusErr;
    tf_.sector_count = 0x03;
    pending_ = Pending::None;
    channel_->raise_interrupt();
}

void AtapiCdrom::data_in(const u8* data, u32 len, Cycles latency) {
    if (latency == 0) latency = kCommandLatency;
    if (dma_) {
        // A DMA data phase: the bytes move through the bus master once the host starts its engine. No byte count,
        // no DRQ; the completion interrupt is the only one the host sees. The console's kernel fetches its sense
        // data this way and waits on exactly that interrupt.
        buffer_.assign(data, data + len);
        if (len & 1) buffer_.push_back(0);
        buffer_pos_ = 0;
        tf_.status = ata::kStatusBsy;
        pending_ = Pending::DmaData;
        channel_->schedule(*this, latency);
        return;
    }
    // PIO: the host set the byte count limit in the LBA mid/high registers before the PACKET command; a longer
    // transfer goes out in pieces of at most that size, an interrupt each.
    payload_.assign(data, data + len);
    payload_pos_ = 0;
    next_pio_chunk();
}

void AtapiCdrom::run_packet() {
    ++packets_;
    const u8 op = cdb_[0];
    trace::log(trace::Cat::Pio, "dvd packet {:02x} cdb {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x}{}", op, cdb_[0],
               cdb_[1], cdb_[2], cdb_[3], cdb_[4], cdb_[5], cdb_[6], cdb_[7], cdb_[8], cdb_[9], cdb_[10], cdb_[11], dma_ ? " dma" : "");
    switch (op) {
        case 0x00:  // TEST UNIT READY
            if (disc_)
                finish_ok();
            else
                check_condition(0x02, 0x3A, 0x00);  // not ready: medium not present
            break;
        case 0x03: {  // REQUEST SENSE
            u8 sense[18]{};
            sense[0] = 0x70;
            sense[2] = sense_key_;
            sense[7] = 10;
            sense[12] = asc_;
            sense[13] = ascq_;
            data_in(sense, sizeof sense);
            break;
        }
        case 0x12: {  // INQUIRY
            u8 inq[36]{};
            inq[0] = 0x05;  // CD/DVD device
            inq[1] = 0x80;  // removable
            inq[2] = 0x00;
            inq[3] = 0x21;
            inq[4] = 31;
            std::memcpy(inq + 8, "MIDWAY  ", 8);
            std::memcpy(inq + 16, "VERDANT DVD-ROM ", 16);
            std::memcpy(inq + 32, "MV01", 4);
            data_in(inq, sizeof inq);
            break;
        }
        case 0x1B:  // START STOP UNIT (load/eject)
            trace::log(trace::Cat::Boot, "DVD tray {}", (cdb_[4] & 0x02) ? ((cdb_[4] & 0x01) ? "load" : "eject") : "spin command");
            finish_ok();
            break;
        case 0x1E:  // PREVENT ALLOW MEDIUM REMOVAL
            finish_ok();
            break;
        case 0x4A: {  // GET EVENT STATUS NOTIFICATION: media class, no change
            u8 ev[8]{};
            ev[1] = 6;
            ev[2] = 0x04;  // media event class
            ev[3] = 0x10;  // supported: media
            ev[4] = 0x00;  // no change
            ev[5] = disc_ ? 0x02 : 0x00;  // media status: present, or no media
            data_in(ev, sizeof ev);
            break;
        }
        case 0x25: {  // READ CAPACITY: the last LBA of the game partition and the 2048-byte block size
            if (!disc_) {
                check_condition(0x02, 0x3A, 0x00);
                break;
            }
            const u32 last = static_cast<u32>(disc_->sectors() - 1);
            const u8 cap[8] = {static_cast<u8>(last >> 24), static_cast<u8>(last >> 16), static_cast<u8>(last >> 8), static_cast<u8>(last), 0, 0, 0x08, 0x00};
            data_in(cap, sizeof cap);
            break;
        }
        case 0x28:  // READ(10)
            read_sectors(be32(&cdb_[2]), static_cast<u32>((cdb_[7] << 8) | cdb_[8]));
            break;
        case 0xA8:  // READ(12)
            read_sectors(be32(&cdb_[2]), be32(&cdb_[6]));
            break;
        case 0xAD:  // READ DVD STRUCTURE
            read_dvd_structure();
            break;
        case 0x55:  // MODE SELECT(10): the parameter list (the security page with a challenge) comes from the host
            data_out(static_cast<u32>((cdb_[7] << 8) | cdb_[8]));
            break;
        case 0x43:  // READ TOC
            check_condition(disc_ ? 0x05 : 0x02, disc_ ? 0x24 : 0x3A, 0x00);
            break;
        case 0x5A:  // MODE SENSE(10)
            mode_sense10();
            break;
        case 0x46: {  // GET CONFIGURATION: profile list with DVD-ROM, no current profile
            u8 cfg[20]{};
            cfg[3] = 16;
            cfg[6] = 0x00;
            cfg[7] = 0x00;  // current profile: none
            cfg[8] = 0x00;
            cfg[9] = 0x00;
            cfg[10] = 0x03;
            cfg[11] = 8;
            cfg[12] = 0x00;
            cfg[13] = 0x10;  // DVD-ROM profile
            cfg[16] = 0x00;
            cfg[17] = 0x08;  // CD-ROM profile
            data_in(cfg, sizeof cfg);
            break;
        }
        default:
            trace::log(trace::Cat::Pio, "dvd unsupported packet {:02x}", op);
            check_condition(0x05, 0x20, 0x00);  // illegal request: invalid command
            break;
    }
}

// Mode pages describe the drive, so they answer with or without a disc. The console's kernel reads page 0x3E, the
// drive's security page (layout per the community documentation in local-notes/hw/dvd.md), at initialisation and
// insists on its byte 3 reading 1; page 0x2A is the standard capabilities page.
void AtapiCdrom::mode_sense10() {
    const u8 page = static_cast<u8>(cdb_[2] & 0x3F);
    const u32 allocation = static_cast<u32>((cdb_[7] << 8) | cdb_[8]);
    u8 reply[8 + 32]{};
    u32 page_len = 0;
    u8* p = reply + 8;
    switch (page) {
        case 0x3E:
            page_len = 20;
            p[0] = 0x3E;
            p[1] = 18;
            // With a disc: the game partition selected, authenticated, book type D (an Xbox game disc) version 1, and
            // the last challenge the host wrote. An image has no security sectors to answer challenges from.
            if (disc_) std::memcpy(p + 2, security_page_.data() + 2, 18);
            p[2] = disc_ ? 0x01 : 0x00;  // partition: game, or video
            p[3] = 0x01;
            p[4] = disc_ ? 0x01 : 0x00;  // authenticated
            p[5] = disc_ ? 0xD1 : 0x00;  // book type / version
            break;
        case 0x2A:
            page_len = 26;
            p[0] = 0x2A;
            p[1] = 24;
            p[2] = 0x3F;  // reads CD-R, CD-RW, method 2, DVD-ROM, DVD-R, DVD-RAM
            p[3] = 0x00;  // writes nothing
            p[4] = 0x71;  // audio play, composite, digital ports, mode 2 form 2, multi-session
            p[5] = 0x03;  // CD-DA stream accurate
            p[6] = 0x29;  // tray loader, eject, lock
            p[8] = 0x1B;  // maximum read speed 7056 kB/s
            p[9] = 0x90;
            p[11] = 0x01;  // 256 volume levels
            p[12] = 0x02;  // 512 KiB buffer
            p[14] = 0x1B;  // current read speed
            p[15] = 0x90;
            break;
        default:
            check_condition(0x05, 0x24, 0x00);  // illegal request: invalid field in CDB
            return;
    }
    const u32 total = 8 + page_len;
    reply[0] = static_cast<u8>((total - 2) >> 8);
    reply[1] = static_cast<u8>(total - 2);
    data_in(reply, allocation ? std::min(total, allocation) : total);
}

// READ(10)/READ(12) from the game partition. The data rate is the drive's: about 8 MB/s off a spinning disc.
void AtapiCdrom::read_sectors(u64 lba, u32 count) {
    if (!disc_) {
        check_condition(0x02, 0x3A, 0x00);
        return;
    }
    if (count == 0) {
        finish_ok();
        return;
    }
    if (lba + count > disc_->sectors()) {
        check_condition(0x05, 0x21, 0x00);  // illegal request: logical block address out of range
        return;
    }
    std::vector<u8> data(static_cast<std::size_t>(count) * DvdImage::kSectorSize);
    if (!disc_->read(lba, count, data.data())) {
        check_condition(0x03, 0x11, 0x00);  // medium error: unrecovered read error
        return;
    }
    sectors_read_ += count;
    data_in(data.data(), static_cast<u32>(data.size()), kCommandLatency + cycles_from_ns(static_cast<u64>(data.size()) * kDvdNsPerByte));
}

// READ DVD STRUCTURE: a 4-byte header (the data length that follows, plus 2) and the structure. Format 0x00 is the
// physical format (one layer of an Xbox game disc, its last sector in layer 0 always 0x2033AF), 0x04 the
// manufacturing information, 0xFD the Xbox security sectors (synthesised: an image does not carry them; the
// authenticated security page tells the kernel not to run the challenges).
void AtapiCdrom::read_dvd_structure() {
    if (!disc_) {
        check_condition(0x02, 0x3A, 0x00);
        return;
    }
    const u8 format = cdb_[7];
    const u32 allocation = static_cast<u32>((cdb_[8] << 8) | cdb_[9]);
    u32 len = 0;
    switch (format) {
        case 0x00: case 0x04: len = 2048; break;
        case 0xFD: len = 0x664 - 4; break;
        default:
            trace::log(trace::Cat::Pio, "dvd structure format {:02x} not supported", format);
            check_condition(0x05, 0x24, 0x00);
            return;
    }
    std::vector<u8> reply(4 + len, 0);
    put_be16(reply.data(), len + 2);
    u8* d = reply.data() + 4;
    if (format == 0x00) {
        d[0] = 0x01;          // book type DVD-ROM, part version 1
        d[1] = 0x02;          // 120 mm, maximum rate 10.08 Mbit/s
        d[2] = 0x31;          // two layers, opposite track path, embossed
        d[3] = 0x10;
        put_be24(d + 5, 0x030000);   // first physical sector of the data area
        put_be24(d + 9, 0xFCFFFF);   // last physical sector of the data area
        put_be24(d + 13, 0x2033AF);  // last sector in layer 0
    } else if (format == 0x04) {
        d[0] = 0x01;  // u32 1, as the dumps show
        std::memcpy(d + 8, "MVIMAGE0", 8);
        d[24] = 0x02;
    } else {
        d[768] = 1;   // challenge table version
        d[769] = 23;  // challenge entries
    }
    data_in(reply.data(), allocation ? std::min(static_cast<u32>(reply.size()), allocation) : static_cast<u32>(reply.size()));
}

// A data-out phase of len bytes: through the bus master when the PACKET asked for DMA, else PIO words on the data
// port (interrupt reason 0: data to the device).
void AtapiCdrom::data_out(u32 len) {
    if (len == 0) {
        finish_ok();
        return;
    }
    buffer_.clear();
    buffer_.reserve(len);
    payload_.assign(len, 0);
    payload_pos_ = 0;
    if (dma_) {
        tf_.status = ata::kStatusBsy;
        pending_ = Pending::DmaDataOut;
        channel_->schedule(*this, kCommandLatency);
        return;
    }
    tf_.lba_mid = static_cast<u8>(len);
    tf_.lba_high = static_cast<u8>(len >> 8);
    tf_.sector_count = 0x00;
    tf_.status = ata::kStatusDrdy | ata::kStatusDrq;
    pending_ = Pending::DataOut;
    channel_->raise_interrupt();
}

// The host's parameter list is in payload_. MODE SELECT(10) of page 0x3E: the security page with a challenge.
void AtapiCdrom::data_out_done() {
    if (cdb_[0] == 0x55 && payload_.size() >= 8 + 20 && (payload_[8] & 0x3F) == 0x3E) {
        std::memcpy(security_page_.data(), payload_.data() + 8, security_page_.size());
        trace::log(trace::Cat::Pio, "dvd security page written: partition {} authenticated {} challenge id {:02x} value {:08x}", security_page_[2],
                   security_page_[4], security_page_[7], be32(&security_page_[8]));
    }
    finish_ok();
}

// The host has read one PIO piece: the next, or the completion.
void AtapiCdrom::pio_chunk_done() {
    if (payload_pos_ < payload_.size())
        next_pio_chunk();
    else
        finish_ok();
}

// The next piece of a PIO data-in phase: at most the byte count the host allowed, even.
void AtapiCdrom::next_pio_chunk() {
    const u32 limit = static_cast<u32>(tf_.lba_mid) | (static_cast<u32>(tf_.lba_high) << 8);
    const std::size_t remaining = payload_.size() - payload_pos_;
    std::size_t n = remaining;
    if (limit && limit < n) n = std::max<std::size_t>(2, limit & ~1u);
    buffer_.assign(payload_.begin() + static_cast<std::ptrdiff_t>(payload_pos_), payload_.begin() + static_cast<std::ptrdiff_t>(payload_pos_ + n));
    payload_pos_ += n;
    if (buffer_.size() & 1) buffer_.push_back(0);
    buffer_pos_ = 0;
    tf_.lba_mid = static_cast<u8>(n);
    tf_.lba_high = static_cast<u8>(n >> 8);
    tf_.sector_count = 0x02;  // interrupt reason: I/O (data to host)
    tf_.status = ata::kStatusDrdy | ata::kStatusDrq;
    pending_ = Pending::DataReady;
    channel_->raise_interrupt();
}

void AtapiCdrom::continue_command() {
    switch (pending_) {
        case Pending::IdentifyReady:
            tf_.status = ata::kStatusDrdy | ata::kStatusDrq;
            channel_->raise_interrupt();
            break;
        case Pending::DmaData: {
            if (!channel_->dma_active()) return;  // the engine starts us again
            const bool ok = channel_->dma_transfer(buffer_.data(), nullptr, static_cast<u32>(buffer_.size()));
            buffer_.clear();
            channel_->dma_complete(ok);
            if (ok)
                finish_ok();
            else
                check_condition(0x0B, 0x00, 0x00);  // aborted command
            break;
        }
        case Pending::DmaDataOut: {
            if (!channel_->dma_active()) return;
            const bool ok = channel_->dma_transfer(nullptr, payload_.data(), static_cast<u32>(payload_.size()));
            channel_->dma_complete(ok);
            if (ok)
                data_out_done();
            else
                check_condition(0x0B, 0x00, 0x00);
            break;
        }
        case Pending::Complete:
            finish_ok();
            break;
        default:
            break;
    }
}

u16 AtapiCdrom::read_data() {
    if (buffer_pos_ + 1 >= buffer_.size()) {
        buffer_.clear();
        if (pending_ == Pending::DataReady) pio_chunk_done();
        return 0;
    }
    const u16 v = static_cast<u16>(buffer_[buffer_pos_] | (buffer_[buffer_pos_ + 1] << 8));
    buffer_pos_ += 2;
    if (buffer_pos_ >= buffer_.size()) {
        buffer_.clear();
        if (pending_ == Pending::IdentifyReady) {
            tf_.status = ata::kStatusDrdy;
            pending_ = Pending::None;
        } else if (pending_ == Pending::DataReady) {
            pio_chunk_done();
        }
    }
    return v;
}

void AtapiCdrom::write_data(u16 value) {
    if (pending_ == Pending::DataOut) {
        if (payload_pos_ < payload_.size()) payload_[payload_pos_++] = static_cast<u8>(value);
        if (payload_pos_ < payload_.size()) payload_[payload_pos_++] = static_cast<u8>(value >> 8);
        if (payload_pos_ >= payload_.size()) {
            tf_.status = ata::kStatusBsy;
            pending_ = Pending::None;
            data_out_done();
        }
        return;
    }
    if (pending_ != Pending::PacketReady) return;
    if (cdb_pos_ + 1 < cdb_.size()) {
        cdb_[cdb_pos_] = static_cast<u8>(value);
        cdb_[cdb_pos_ + 1] = static_cast<u8>(value >> 8);
        cdb_pos_ += 2;
    }
    if (cdb_pos_ >= cdb_.size()) {
        tf_.status = ata::kStatusBsy;
        pending_ = Pending::None;
        run_packet();
    }
}

}  // namespace mv
