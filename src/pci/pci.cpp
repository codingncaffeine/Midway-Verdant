#include "pci/pci.h"

#include <format>

#include "debug/trace.h"

namespace mv {

std::string PciAddress::str() const { return std::format("{:02x}:{:02x}.{}", bus, device, function); }

PciDevice::PciDevice(std::string_view name, u16 vendor, u16 device, u32 class_code, u8 revision, u8 header_type)
    : name_(name), header_type_(header_type) {
    set_cfg16(0x00, vendor);
    set_cfg16(0x02, device);
    set_cfg8(0x08, revision);
    cfg_[0x09] = static_cast<u8>(class_code & 0xFF);
    cfg_[0x0A] = static_cast<u8>((class_code >> 8) & 0xFF);
    cfg_[0x0B] = static_cast<u8>((class_code >> 16) & 0xFF);
    cfg_[0x0E] = header_type;
}

u32 PciDevice::cfg32(u8 reg) const {
    return static_cast<u32>(cfg_[reg]) | (static_cast<u32>(cfg_[static_cast<std::size_t>(reg) + 1]) << 8) | (static_cast<u32>(cfg_[static_cast<std::size_t>(reg) + 2]) << 16) |
           (static_cast<u32>(cfg_[static_cast<std::size_t>(reg) + 3]) << 24);
}

void PciDevice::set_cfg16(u8 reg, u16 v) {
    cfg_[reg] = static_cast<u8>(v);
    cfg_[static_cast<std::size_t>(reg) + 1] = static_cast<u8>(v >> 8);
}

void PciDevice::set_cfg32(u8 reg, u32 v) {
    for (unsigned i = 0; i < 4; ++i) cfg_[static_cast<std::size_t>(reg) + i] = static_cast<u8>(v >> (8 * i));
}

void PciDevice::define_bar(int index, BarKind kind, u32 size, bool prefetchable) {
    auto& b = bars_[static_cast<std::size_t>(index)];
    b.kind = kind;
    b.size = size;
    b.prefetchable = prefetchable;
    u32 flags = 0;
    if (kind == BarKind::Io) flags = 1;
    if (kind == BarKind::Memory && prefetchable) flags = 8;
    set_cfg32(static_cast<u8>(0x10 + 4 * index), flags);
}

u32 PciDevice::bar_base(int index) const {
    const auto& b = bars_[static_cast<std::size_t>(index)];
    const u32 raw = cfg32(static_cast<u8>(0x10 + 4 * index));
    if (b.kind == BarKind::Io) return raw & 0xFFFFu & ~(b.size - 1);
    if (b.kind == BarKind::Memory) return raw & ~(b.size - 1) & ~0xFu;
    return 0;
}

void PciDevice::write_bar_dword(int index, u32 value) {
    auto& b = bars_[static_cast<std::size_t>(index)];
    const u8 reg = static_cast<u8>(0x10 + 4 * index);
    if (b.kind == BarKind::None) return;  // unimplemented BARs read as zero and ignore writes
    const u32 old_base = bar_base(index);
    const u32 mask = ~(b.size - 1);
    u32 stored;
    if (b.kind == BarKind::Io) {
        stored = (value & mask & 0xFFFFu) | 1u;
    } else {
        stored = (value & mask & ~0xFu) | (b.prefetchable ? 8u : 0u);
    }
    set_cfg32(reg, stored);
    const u32 new_base = bar_base(index);
    if (new_base != old_base) on_bar_changed(index, old_base, new_base);
}

void PciDevice::reset_config() {
    for (int i = 0; i < 6; ++i) write_bar_dword(i, 0);
    set_cfg16(0x04, 0);
    set_cfg16(0x06, 0);
    on_reset();
}

u32 PciDevice::config_read(u8 reg, unsigned size) {
    u32 v = 0;
    for (unsigned i = 0; i < size; ++i) v |= static_cast<u32>(cfg_[static_cast<u8>(reg + i)]) << (8 * i);
    return v;
}

void PciDevice::config_write(u8 reg, unsigned size, u32 value) {
    // Standard header: identification is read-only; status bits are write-one-to-clear; BARs are masked.
    const u8 end = static_cast<u8>(reg + size - 1);
    if (header_type_ == 0 && reg >= 0x10 && end <= 0x27) {
        // Assemble the dword the write lands in (partial writes merge into the current value).
        const int index = (reg - 0x10) / 4;
        const u8 dreg = static_cast<u8>(0x10 + 4 * index);
        u32 current = cfg32(dreg);
        for (unsigned i = 0; i < size; ++i) {
            const unsigned shift = 8u * static_cast<unsigned>(reg + i - dreg);
            current = (current & ~(0xFFu << shift)) | (((value >> (8 * i)) & 0xFFu) << shift);
        }
        write_bar_dword(index, current);
        on_config_write(reg, size, value);
        return;
    }
    for (unsigned i = 0; i < size; ++i) {
        const u8 r = static_cast<u8>(reg + i);
        const u8 byte = static_cast<u8>(value >> (8 * i));
        if (r <= 0x03 || (r >= 0x08 && r <= 0x0B) || r == 0x0E || (r >= 0x2C && r <= 0x2F) || r == 0x34 || r == 0x3D) {
            continue;  // vendor/device, revision/class, header type, subsystem, capabilities pointer, interrupt pin
        }
        if (r == 0x06 || r == 0x07) {
            cfg_[r] = static_cast<u8>(cfg_[r] & ~byte);  // status: write one to clear
            continue;
        }
        cfg_[r] = byte;
    }
    on_config_write(reg, size, value);
}

PciBus::PciBus(Bus& bus) : bus_(bus) { bus_.map_pio(0xCF8, 8, this); }

void PciBus::add(PciAddress at, PciDevice& dev) {
    dev.address_ = at;
    devices_[at.key()] = &dev;
}

PciDevice* PciBus::find(PciAddress at) const {
    auto it = devices_.find(at.key());
    return it == devices_.end() ? nullptr : it->second;
}

void PciBus::reset_all() {
    for (auto& [key, dev] : devices_) dev->reset_config();
    config_address_ = 0;
}

u32 PciBus::config_read(PciAddress at, u8 reg, unsigned size) {
    if (PciDevice* d = find(at)) {
        const u32 v = d->config_read(reg, size);
        trace::log(trace::Cat::Pci, "{} {} read  reg {:02x} size {} -> {:0{}x}", at.str(), d->name(), reg, size, v, size * 2);
        return v;
    }
    if (at.bus == 0 && at.device == 0 && (at.function == 1 || at.function == 2)) {
        trace::log(trace::Cat::Pci, "{} read of a function that freezes real hardware", at.str());
    } else {
        trace::log(trace::Cat::Pci, "{} read  reg {:02x}: no device", at.str(), reg);
    }
    return size == 4 ? 0xFFFFFFFFu : (size == 2 ? 0xFFFFu : 0xFFu);
}

void PciBus::config_write(PciAddress at, u8 reg, unsigned size, u32 value) {
    if (PciDevice* d = find(at)) {
        trace::log(trace::Cat::Pci, "{} {} write reg {:02x} size {} <- {:0{}x}", at.str(), d->name(), reg, size, value, size * 2);
        d->config_write(reg, size, value);
        return;
    }
    trace::log(trace::Cat::Pci, "{} write reg {:02x} <- {:08x}: no device", at.str(), reg, value);
}

u32 PciBus::pio_read(u16 port, unsigned size) {
    if (port < 0xCFC) {
        // CONFIG_ADDRESS, any width (byte reads return the addressed byte of the latch).
        const unsigned shift = 8u * (port & 3u);
        const u32 v = config_address_ >> shift;
        return size == 4 ? v : (size == 2 ? (v & 0xFFFFu) : (v & 0xFFu));
    }
    if ((config_address_ & 0x80000000u) == 0) return size == 4 ? 0xFFFFFFFFu : (size == 2 ? 0xFFFFu : 0xFFu);
    const PciAddress at{static_cast<u8>((config_address_ >> 16) & 0xFF), static_cast<u8>((config_address_ >> 11) & 0x1F),
                        static_cast<u8>((config_address_ >> 8) & 0x7)};
    const u8 reg = static_cast<u8>((config_address_ & 0xFC) | (port & 3u));
    return config_read(at, reg, size);
}

void PciBus::pio_write(u16 port, unsigned size, u32 value) {
    if (port < 0xCFC) {
        const unsigned shift = 8u * (port & 3u);
        const u32 mask = size == 4 ? 0xFFFFFFFFu : (size == 2 ? 0xFFFFu : 0xFFu);
        config_address_ = (config_address_ & ~(mask << shift)) | ((value & mask) << shift);
        return;
    }
    if ((config_address_ & 0x80000000u) == 0) return;
    const PciAddress at{static_cast<u8>((config_address_ >> 16) & 0xFF), static_cast<u8>((config_address_ >> 11) & 0x1F),
                        static_cast<u8>((config_address_ >> 8) & 0x7)};
    const u8 reg = static_cast<u8>((config_address_ & 0xFC) | (port & 3u));
    config_write(at, reg, size, value);
}

}  // namespace mv
