#include "smbus/smbus.h"

#include "core/clock.h"
#include "debug/trace.h"

namespace mv {

SmbusController::SmbusController(Bus& bus, Scheduler& sched, std::function<void(bool)> irq)
    : PciDevice("smbus", 0x10DE, 0x01B4, 0x0C0500, 0xB1), bus_(bus), sched_(sched), irq_(std::move(irq)) {
    define_bar(0, BarKind::Io, 16);
    define_bar(1, BarKind::Io, 16);
    define_bar(2, BarKind::Io, 32);
    set_cfg8(0x3C, 11);  // interrupt line, as the console's firmware leaves it
    set_cfg8(0x3D, 1);
    completion_.name = "smbus.complete";
    completion_.fire = [this](Cycles) { complete_cycle(); };
}

void SmbusController::attach(u8 address7, SmbusDevice& dev) { devices_[address7] = &dev; }

SmbusDevice* SmbusController::device_at(u8 address7) const {
    auto it = devices_.find(address7);
    return it == devices_.end() ? nullptr : it->second;
}

void SmbusController::on_bar_changed(int index, u32 old_base, u32 new_base) {
    if (index != 1) return;
    (void)old_base;
    if (mapped_at_) bus_.unmap_pio(static_cast<u16>(mapped_at_), 16);
    mapped_at_ = new_base;
    if (new_base) bus_.map_pio(static_cast<u16>(new_base), 16, this);
    trace::log(trace::Cat::Smbus, "controller I/O at {:04x}", new_base);
}

u32 SmbusController::pio_read(u16 port, unsigned size) {
    const u16 off = static_cast<u16>(port - mapped_at_);
    switch (off) {
        case 0x0: return size == 1 ? status_ : static_cast<u32>(status_);
        case 0x1: return 0;
        case 0x2: return control_;
        case 0x4: return address_;
        case 0x6: return size == 1 ? (data_ & 0xFF) : data_;
        case 0x7: return data_ >> 8;
        case 0x8: return command_;
        default: return 0;
    }
}

void SmbusController::pio_write(u16 port, unsigned size, u32 value) {
    const u16 off = static_cast<u16>(port - mapped_at_);
    switch (off) {
        case 0x0:
            status_ = static_cast<u8>(status_ & ~static_cast<u8>(value));  // write one to clear
            update_irq();
            break;
        case 0x2:
            control_ = static_cast<u8>(value);
            if (control_ & kControlAbort) {
                sched_.cancel(completion_);
                status_ = static_cast<u8>((status_ & ~kStatusBusy) | kStatusAbort);
                control_ = static_cast<u8>(control_ & ~kControlAbort);
            }
            if (control_ & kControlStart) {
                control_ = static_cast<u8>(control_ & ~kControlStart);
                start_cycle();
            }
            update_irq();
            break;
        case 0x4: address_ = static_cast<u8>(value); break;
        case 0x6:
            if (size == 1)
                data_ = static_cast<u16>((data_ & 0xFF00) | (value & 0xFF));
            else
                data_ = static_cast<u16>(value);
            break;
        case 0x7: data_ = static_cast<u16>((data_ & 0x00FF) | ((value & 0xFF) << 8)); break;
        case 0x8: command_ = static_cast<u8>(value); break;
        default: break;
    }
}

void SmbusController::start_cycle() {
    pending_.address7 = static_cast<u8>(address_ >> 1);
    pending_.read = (address_ & 1) != 0;
    pending_.cycle_type = static_cast<u8>(control_ & 0x07);
    pending_.command = command_;
    pending_.data = data_;
    pending_.acked = false;
    if (cycle_hook_) cycle_hook_(pending_);
    status_ = static_cast<u8>(status_ | kStatusBusy);
    // Bits on the wire: address+ack, command+ack, each data byte+ack, at 100 kHz.
    unsigned bytes = 1;
    switch (pending_.cycle_type) {
        case 0: bytes = 1; break;   // quick
        case 1: bytes = 2; break;   // byte
        case 2: bytes = 3; break;   // byte data
        case 3: bytes = 4; break;   // word data
        default: bytes = 4; break;
    }
    if (pending_.read && pending_.cycle_type >= 2) bytes += 1;  // repeated start + address again
    const u64 ns = static_cast<u64>(bytes) * 9u * 10'000u;
    sched_.schedule(completion_, sched_.now() + cycles_from_ns(ns));
}

void SmbusController::complete_cycle() {
    Transaction& t = pending_;
    SmbusDevice* dev = device_at(t.address7);
    bool ack = false;
    u16 result = 0;
    if (dev) {
        switch (t.cycle_type) {
            case 0: ack = dev->quick(t.read); break;
            case 1:
                if (t.read) {
                    auto v = dev->receive_byte();
                    ack = v.has_value();
                    if (ack) result = *v;
                } else {
                    ack = dev->send_byte(t.command);
                }
                break;
            case 2:
                if (t.read) {
                    auto v = dev->read_byte(t.command);
                    ack = v.has_value();
                    if (ack) result = *v;
                } else {
                    ack = dev->write_byte(t.command, static_cast<u8>(t.data));
                }
                break;
            case 3:
                if (t.read) {
                    auto v = dev->read_word(t.command);
                    ack = v.has_value();
                    if (ack) result = *v;
                } else {
                    ack = dev->write_word(t.command, t.data);
                }
                break;
            default: ack = false; break;
        }
    }
    t.acked = ack;
    if (ack && t.read) data_ = result;
    status_ = static_cast<u8>((status_ & ~kStatusBusy) | (ack ? kStatusComplete : kStatusProtocolError));
    if (t.read)
        trace::log(trace::Cat::Smbus, "{} {:02x} cmd {:02x} read  -> {:04x}{}", dev ? dev->name() : "(absent)", t.address7,
                   t.command, result, ack ? "" : " NACK");
    else
        trace::log(trace::Cat::Smbus, "{} {:02x} cmd {:02x} write <- {:04x}{}", dev ? dev->name() : "(absent)", t.address7,
                   t.command, t.data, ack ? "" : " NACK");
    if (log_.size() < 100000) log_.push_back(t);
    ++count_;
    update_irq();
}

void SmbusController::update_irq() {
    const bool level = (control_ & kControlInterrupt) && (status_ & (kStatusComplete | kStatusProtocolError | kStatusAbort));
    if (level != irq_level_) {
        irq_level_ = level;
        if (irq_) irq_(level);
    }
}

}  // namespace mv
