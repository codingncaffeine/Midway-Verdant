#include "mcpx/lpc.h"

#include "debug/trace.h"

namespace mv {

Lpc::Lpc(Bus& bus, Flash& flash) : PciDevice("lpc", 0x10DE, 0x01B2, 0x060100, 0xB2), bus_(bus), flash_(flash) {
    define_bar(0, BarKind::Io, 256);
}

void Lpc::attach_pm_block(PioHandler* pm, u32 size) {
    pm_ = pm;
    pm_size_ = size;
    if (bar_base(0) != 0) on_bar_changed(0, 0, bar_base(0));
}

void Lpc::on_config_write(u8 reg, unsigned size, u32 value) {
    const u8 end = static_cast<u8>(reg + size - 1);
    if (reg <= 0x80 && end >= 0x80) {
        const u32 byte = (value >> (8u * (0x80 - reg))) & 0xFFu;
        if (byte & 0x02u) flash_.hide_rom();
    }
    if (reg <= 0x8C && end >= 0x8C) {
        trace::log(trace::Cat::Boot, "LPC 0x8C <- {:08x} (IDE/NIC enable: {})", cfg32(0x8C), ide_nic_enabled());
    }
}

void Lpc::on_reset() {
    // The bridge's own registers: the ROM-hide bit and the function enables, as the boot code finds them.
    set_cfg32(0x80, 0);
    set_cfg32(0x8C, 0);
}

void Lpc::on_bar_changed(int index, u32 old_base, u32 new_base) {
    if (index != 0 || !pm_) return;
    if (pm_mapped_at_) bus_.unmap_pio(static_cast<u16>(pm_mapped_at_), pm_size_);
    pm_mapped_at_ = new_base;
    if (new_base) bus_.map_pio(static_cast<u16>(new_base), pm_size_, pm_);
    trace::log(trace::Cat::Boot, "PM block moved {:04x} -> {:04x}", old_base, new_base);
}

}  // namespace mv
