#include "usb/usb_device.h"

#include <algorithm>
#include <cstring>

namespace mv::usb {

namespace {
enum : u8 { kGetStatus = 0, kClearFeature = 1, kSetFeature = 3, kSetAddress = 5, kGetDescriptor = 6, kGetConfiguration = 8,
            kSetConfiguration = 9, kGetInterface = 10, kSetInterface = 11 };
enum : u8 { kDescDevice = 1, kDescConfiguration = 2 };
}  // namespace

void Device::bus_reset() {
    address_ = 0;
    pending_address_ = 0;
    address_pending_ = false;
    configuration_ = 0;
    stage_ = Stage::Idle;
    reply_.clear();
    reply_pos_ = 0;
    out_data_.clear();
}

bool Device::class_request(const Setup&, std::span<const u8>, std::vector<u8>&) { return false; }
Status Device::endpoint_in(u8, u8*, u32, u32&) { return Status::Stall; }
Status Device::endpoint_out(u8, const u8*, u32) { return Status::Stall; }

bool Device::standard_request(const Setup& s, std::vector<u8>& reply) {
    switch (s.request) {
        case kGetStatus:
            reply = {0, 0};
            return true;
        case kClearFeature:
        case kSetFeature:
            return true;
        case kSetAddress:
            pending_address_ = static_cast<u8>(s.value & 0x7F);
            address_pending_ = true;
            return true;
        case kGetDescriptor: {
            std::span<const u8> d;
            switch (s.value >> 8) {
                case kDescDevice: d = device_descriptor(); break;
                case kDescConfiguration: d = configuration_descriptor(); break;
                default: return false;  // no string descriptors: every index is 0
            }
            reply.assign(d.begin(), d.end());
            return true;
        }
        case kGetConfiguration:
            reply = {configuration_};
            return true;
        case kSetConfiguration:
            configuration_ = static_cast<u8>(s.value);
            configured(configuration_);
            return true;
        case kGetInterface:
            reply = {0};
            return true;
        case kSetInterface:
            return true;
        default:
            return false;
    }
}

void Device::start_setup(const u8* packet) {
    setup_.request_type = packet[0];
    setup_.request = packet[1];
    setup_.value = static_cast<u16>(packet[2] | (packet[3] << 8));
    setup_.index = static_cast<u16>(packet[4] | (packet[5] << 8));
    setup_.length = static_cast<u16>(packet[6] | (packet[7] << 8));
    reply_.clear();
    reply_pos_ = 0;
    out_data_.clear();
    if (setup_.device_to_host()) {
        const bool ok = setup_.type() == 0 ? standard_request(setup_, reply_) : class_request(setup_, {}, reply_);
        if (!ok) {
            stage_ = Stage::Stalled;
            return;
        }
        if (reply_.size() > setup_.length) reply_.resize(setup_.length);
        stage_ = Stage::DataIn;
    } else if (setup_.length > 0) {
        stage_ = Stage::DataOut;
    } else {
        std::vector<u8> unused;
        const bool ok = setup_.type() == 0 ? standard_request(setup_, unused) : class_request(setup_, {}, unused);
        stage_ = ok ? Stage::StatusIn : Stage::Stalled;
    }
}

void Device::finish_status_stage() {
    if (address_pending_) {
        address_ = pending_address_;
        address_pending_ = false;
    }
    stage_ = Stage::Idle;
}

Status Device::transact(Pid pid, u8 endpoint, u8* data, u32 len, u32& actual) {
    actual = 0;
    if (endpoint != 0) {
        if (pid == Pid::In) return endpoint_in(endpoint, data, len, actual);
        if (pid == Pid::Out) {
            const Status st = endpoint_out(endpoint, data, len);
            if (st == Status::Ack) actual = len;
            return st;
        }
        return Status::Stall;
    }
    switch (pid) {
        case Pid::Setup:
            if (len != 8) return Status::Stall;
            start_setup(data);
            actual = 8;
            return Status::Ack;  // a SETUP is always acknowledged; a refused request stalls the next stage
        case Pid::In:
            switch (stage_) {
                case Stage::DataIn: {
                    const std::size_t n = std::min<std::size_t>(len, reply_.size() - reply_pos_);
                    if (n) std::memcpy(data, reply_.data() + reply_pos_, n);
                    reply_pos_ += n;
                    actual = static_cast<u32>(n);
                    // The data stage ends on a short packet or at wLength. A reply shorter than wLength that fills
                    // its last packet exactly still owes the host a zero-length packet (USB 1.1 5.5.3).
                    if (reply_pos_ >= reply_.size() && (n < len || reply_.size() >= setup_.length)) stage_ = Stage::StatusOut;
                    return Status::Ack;
                }
                case Stage::StatusIn:
                    finish_status_stage();
                    return Status::Ack;
                default:
                    return Status::Stall;
            }
        case Pid::Out:
            switch (stage_) {
                case Stage::DataOut: {
                    out_data_.insert(out_data_.end(), data, data + len);
                    actual = len;
                    if (out_data_.size() >= setup_.length) {
                        out_data_.resize(setup_.length);
                        std::vector<u8> unused;
                        const bool ok = setup_.type() == 0 ? standard_request(setup_, unused) : class_request(setup_, out_data_, unused);
                        stage_ = ok ? Stage::StatusIn : Stage::Stalled;
                    }
                    return Status::Ack;
                }
                case Stage::StatusOut:
                case Stage::DataIn:  // the host ended the data stage early
                    actual = len;
                    finish_status_stage();
                    return Status::Ack;
                default:
                    return Status::Stall;
            }
    }
    return Status::Stall;
}

}  // namespace mv::usb
