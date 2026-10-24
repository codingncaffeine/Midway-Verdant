#include "ide/ide.h"

#include <cstring>

#include "core/clock.h"
#include "debug/trace.h"

namespace mv {

IdeController::IdeController(Bus& bus, Scheduler& sched, std::function<void(bool)> irq14)
    : PciDevice("ide", 0x10DE, 0x01BC, 0x01018A, 0xB1), bus_(bus), primary_(std::make_unique<IdeChannel>(bus, sched, std::move(irq14))) {
    define_bar(4, BarKind::Io, 16);
    set_cfg8(0x3C, 14);
    set_cfg8(0x3D, 1);
    bus_.map_pio(0x1F0, 8, this);
    bus_.map_pio(0x3F6, 1, this);
}

void IdeController::attach(int position, std::unique_ptr<IdeDevice> dev) { primary_->attach(position, std::move(dev)); }

void IdeController::on_bar_changed(int index, u32 old_base, u32 new_base) {
    if (index != 4) return;
    (void)old_base;
    if (busmaster_at_) bus_.unmap_pio(static_cast<u16>(busmaster_at_), 16);
    busmaster_at_ = new_base;
    if (new_base) bus_.map_pio(static_cast<u16>(new_base), 16, this);
    trace::log(trace::Cat::Pio, "ide bus master at {:04x}", new_base);
}

u32 IdeController::pio_read(u16 port, unsigned size) {
    if (port >= 0x1F0 && port <= 0x1F7) return primary_->read_task(static_cast<u16>(port - 0x1F0), size);
    if (port == 0x3F6) return primary_->read_control(0);
    if (busmaster_at_ && port >= busmaster_at_ && port < busmaster_at_ + 8) return primary_->read_busmaster(static_cast<u16>(port - busmaster_at_), size);
    return 0xFF;
}

void IdeController::pio_write(u16 port, unsigned size, u32 value) {
    if (port >= 0x1F0 && port <= 0x1F7) {
        primary_->write_task(static_cast<u16>(port - 0x1F0), size, value);
    } else if (port == 0x3F6) {
        primary_->write_control(0, value);
    } else if (busmaster_at_ && port >= busmaster_at_ && port < busmaster_at_ + 8) {
        primary_->write_busmaster(static_cast<u16>(port - busmaster_at_), size, value);
    }
}

IdeChannel::IdeChannel(Bus& bus, Scheduler& sched, std::function<void(bool)> irq) : bus_(bus), sched_(sched), irq_(std::move(irq)) {
    for (int i = 0; i < 2; ++i) {
        events_[i].name = i == 0 ? "ide.master" : "ide.slave";
        events_[i].fire = [this, i](Cycles) {
            if (devices_[static_cast<std::size_t>(i)]) devices_[static_cast<std::size_t>(i)]->continue_command();
        };
    }
}

void IdeChannel::attach(int position, std::unique_ptr<IdeDevice> dev) {
    dev->attach(this);
    dev->hardware_reset();
    devices_[static_cast<std::size_t>(position)] = std::move(dev);
}

IdeDevice* IdeChannel::selected() const { return devices_[static_cast<std::size_t>(selected_)].get(); }

void IdeChannel::schedule(IdeDevice& dev, Cycles delay) {
    const int pos = devices_[0].get() == &dev ? 0 : 1;
    sched_.schedule(events_[pos], sched_.now() + delay);
}

void IdeChannel::raise_interrupt() {
    if (nien_) return;
    if (!irq_level_) {
        irq_level_ = true;
        // The bus master's interrupt bit follows the device's INTRQ edge whether or not a transfer ran (SFF-8038i):
        // the console's kernel tests it before the device status whenever it has armed the engine.
        bm_status_ |= 0x04;
        if (irq_) irq_(true);
    }
}

void IdeController::reset() { primary_->hardware_reset(); }

void IdeChannel::hardware_reset() {
    for (auto& ev : events_) sched_.cancel(ev);
    for (auto& d : devices_) {
        if (d) d->hardware_reset();
    }
    selected_ = 0;
    nien_ = hob_ = false;
    bm_command_ = bm_status_ = 0;
    bm_prdt_ = 0;
    if (irq_level_) {
        irq_level_ = false;
        if (irq_) irq_(false);
    }
}

void IdeChannel::software_reset() {
    for (auto& d : devices_) {
        if (d) d->reset();
    }
    if (irq_level_) {
        irq_level_ = false;
        if (irq_) irq_(false);
    }
}

u32 IdeChannel::read_task(u16 reg, unsigned size) {
    IdeDevice* dev = selected();
    if (!dev) {
        // No device answers: an empty position reads as zero status, like an open cable.
        return 0;
    }
    IdeDevice::TaskFile& tf = dev->tf();
    switch (reg) {
        case 0: {
            if (!(tf.status & ata::kStatusDrq)) return 0;
            if (size == 4) return static_cast<u32>(dev->read_data()) | (static_cast<u32>(dev->read_data()) << 16);
            return dev->read_data();
        }
        case 1: return hob_ ? tf.hob_features : tf.error;
        case 2: return hob_ ? tf.hob_sector_count : tf.sector_count;
        case 3: return hob_ ? tf.hob_lba_low : tf.lba_low;
        case 4: return hob_ ? tf.hob_lba_mid : tf.lba_mid;
        case 5: return hob_ ? tf.hob_lba_high : tf.lba_high;
        case 6: return tf.device;
        case 7:
            // Reading the status register acknowledges the interrupt.
            if (irq_level_) {
                irq_level_ = false;
                if (irq_) irq_(false);
            }
            return tf.status;
        default: return 0;
    }
}

// A task-file register write with its previous value kept as the high-order byte (48-bit addressing).
void IdeChannel::latch_task(IdeDevice::TaskFile& tf, u16 reg, u8 v) {
    switch (reg) {
        case 1: tf.hob_features = tf.features; tf.features = v; break;
        case 2: tf.hob_sector_count = tf.sector_count; tf.sector_count = v; break;
        case 3: tf.hob_lba_low = tf.lba_low; tf.lba_low = v; break;
        case 4: tf.hob_lba_mid = tf.lba_mid; tf.lba_mid = v; break;
        case 5: tf.hob_lba_high = tf.lba_high; tf.lba_high = v; break;
        default: break;
    }
}

void IdeChannel::write_task(u16 reg, unsigned size, u32 value) {
    if (reg == 6) {
        selected_ = (value & 0x10) ? 1 : 0;
        for (auto& d : devices_) {
            if (d) d->tf().device = static_cast<u8>(value);
        }
        return;
    }
    // Both devices on the cable latch the features, count and LBA registers (ATA: every command-block write but the
    // command reaches both); only the selected one takes data and commands. A driver may load the task file before it
    // selects the device.
    if (reg >= 1 && reg <= 5) {
        for (auto& d : devices_)
            if (d) latch_task(d->tf(), reg, static_cast<u8>(value));
        return;
    }
    IdeDevice* dev = selected();
    if (!dev) return;
    IdeDevice::TaskFile& tf = dev->tf();
    switch (reg) {
        case 0:
            if (tf.status & ata::kStatusDrq) {
                dev->write_data(static_cast<u16>(value));
                if (size == 4) dev->write_data(static_cast<u16>(value >> 16));
            }
            break;
        case 7: {
            if (tf.status & ata::kStatusBsy) return;  // commands written while busy are lost
            ++commands_;
            last_command_[static_cast<std::size_t>(selected_)] = static_cast<u8>(value);
            trace::log(trace::Cat::Pio, "ide {} command {:02x} (count {} lba {:02x}{:02x}{:02x} dev {:02x} feat {:02x})", dev->name(),
                       static_cast<u8>(value), tf.sector_count, tf.lba_high, tf.lba_mid, tf.lba_low, tf.device, tf.features);
            if (irq_level_) {
                irq_level_ = false;
                if (irq_) irq_(false);
            }
            const Cycles delay = dev->start_command(static_cast<u8>(value));
            if (delay != ~Cycles{0}) schedule(*dev, delay);
            break;
        }
        default: break;
    }
}

u32 IdeChannel::read_control(u16 reg) {
    (void)reg;
    IdeDevice* dev = selected();
    return dev ? dev->tf().status : 0;  // alternate status: no interrupt acknowledge
}

void IdeChannel::write_control(u16 reg, u32 value) {
    (void)reg;
    const bool srst = (value & 0x04) != 0;
    nien_ = (value & 0x02) != 0;
    hob_ = (value & 0x80) != 0;
    static bool in_reset = false;
    if (srst && !in_reset) {
        in_reset = true;
        trace::log(trace::Cat::Pio, "ide software reset");
        software_reset();
    } else if (!srst) {
        in_reset = false;
    }
}

u32 IdeChannel::read_busmaster(u16 reg, unsigned size) {
    switch (reg) {
        case 0: return bm_command_;
        case 2: return bm_status_;
        case 4: return size == 4 ? bm_prdt_ : (bm_prdt_ & 0xFF);
        case 5: return (bm_prdt_ >> 8) & 0xFF;
        case 6: return (bm_prdt_ >> 16) & 0xFF;
        case 7: return (bm_prdt_ >> 24) & 0xFF;
        default: return 0;
    }
}

void IdeChannel::write_busmaster(u16 reg, unsigned size, u32 value) {
    switch (reg) {
        case 0: {
            const u8 was = bm_command_;
            bm_command_ = static_cast<u8>(value & 0x09);
            if ((bm_command_ & 1u) && !(was & 1u)) {
                bm_status_ |= 0x01;  // active
                // A device whose DMA command was already issued proceeds now that the engine runs, but not before its
                // data is there: a read still due later (the drive reading the disc) keeps its time. Pulling every
                // transfer in to a microsecond after the engine started made the DVD and the disk ~50 times faster
                // than their modelled rates, and a game copying from disc to disk issued commands as fast as it could.
                if (IdeDevice* dev = selected()) {
                    const TimerEvent& due = events_[devices_[0].get() == dev ? 0 : 1];
                    if (!due.scheduled() || due.when <= sched_.now() + cycles_from_ns(1000)) schedule(*dev, cycles_from_ns(1000));
                }
            } else if (!(bm_command_ & 1u)) {
                bm_status_ &= static_cast<u8>(~0x01u);
            }
            break;
        }
        case 2:
            // Bits 1 and 2 are write-one-to-clear; bits 5 and 6 are software-owned.
            bm_status_ = static_cast<u8>((bm_status_ & ~(value & 0x06)) & ~0x60u) | static_cast<u8>(value & 0x60);
            break;
        case 4:
            if (size == 4)
                bm_prdt_ = value & ~3u;
            else
                bm_prdt_ = (bm_prdt_ & ~0xFFu) | (value & 0xFC);
            break;
        case 5: bm_prdt_ = (bm_prdt_ & ~0xFF00u) | ((value & 0xFF) << 8); break;
        case 6: bm_prdt_ = (bm_prdt_ & ~0xFF0000u) | ((value & 0xFF) << 16); break;
        case 7: bm_prdt_ = (bm_prdt_ & ~0xFF000000u) | ((value & 0xFF) << 24); break;
        default: break;
    }
}

bool IdeChannel::dma_transfer(const u8* to_memory, u8* from_memory, u32 len) {
    // Walk the physical region descriptor table: {address, byte count (0 = 64 KiB) | EOT in bit 31}.
    u32 entry = bm_prdt_;
    u32 done = 0;
    for (int guard = 0; guard < 8192 && done < len; ++guard) {
        u32 addr = 0, count_eot = 0;
        if (!bus_.read_ram(entry, &addr, 4) || !bus_.read_ram(entry + 4, &count_eot, 4)) return false;
        u32 count = count_eot & 0xFFFE;
        if (count == 0) count = 0x10000;
        const u32 n = std::min(count, len - done);
        if (to_memory) {
            if (!bus_.write_ram(addr & ~1u, to_memory + done, n)) return false;
        } else if (from_memory) {
            if (!bus_.read_ram(addr & ~1u, from_memory + done, n)) return false;
        }
        done += n;
        if (count_eot & 0x80000000u) break;
        entry += 8;
    }
    return done == len;
}

void IdeChannel::dma_complete(bool ok) {
    bm_status_ = static_cast<u8>((bm_status_ & ~0x01u) | 0x04u | (ok ? 0 : 0x02u));
    bm_command_ &= static_cast<u8>(~0x01u);
}

}  // namespace mv
