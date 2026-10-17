#pragma once
#include <array>
#include <functional>
#include <map>
#include <string>
#include <string_view>

#include "core/bus.h"
#include "core/types.h"

namespace mv {

// Bus/device/function packed as the configuration mechanism sees it.
struct PciAddress {
    u8 bus{0}, device{0}, function{0};
    constexpr u16 key() const { return static_cast<u16>((bus << 8) | (device << 3) | function); }
    std::string str() const;
};

// One PCI function: 256 bytes of configuration space with the standard semantics for the identification
// registers (read-only) and the base address registers (size-masked, probe with all ones). Device-specific
// registers from 0x40 up are stored and read back unless a subclass intercepts them.
class PciDevice {
public:
    enum class BarKind { None, Memory, Io };
    struct Bar {
        BarKind kind{BarKind::None};
        u32 size{0};
        bool prefetchable{false};
    };

    PciDevice(std::string_view name, u16 vendor, u16 device, u32 class_code, u8 revision, u8 header_type = 0);
    virtual ~PciDevice() = default;

    std::string_view name() const { return name_; }
    const PciAddress& address() const { return address_; }

    u32 config_read(u8 reg, unsigned size);
    void config_write(u8 reg, unsigned size, u32 value);

    // Raw access for defaults and tests.
    u8 cfg8(u8 reg) const { return cfg_[reg]; }
    u16 cfg16(u8 reg) const { return static_cast<u16>(cfg_[reg] | (cfg_[static_cast<std::size_t>(reg) + 1] << 8)); }
    u32 cfg32(u8 reg) const;
    void set_cfg8(u8 reg, u8 v) { cfg_[reg] = v; }
    void set_cfg16(u8 reg, u16 v);
    void set_cfg32(u8 reg, u32 v);

    void define_bar(int index, BarKind kind, u32 size, bool prefetchable = false);
    // The programmed base of a BAR (flags stripped), or 0 when unset.
    u32 bar_base(int index) const;
    const Bar& bar(int index) const { return bars_[static_cast<std::size_t>(index)]; }

protected:
    // Subclasses see every configuration write after the standard registers were updated; the BAR hook fires
    // whenever a BAR's base changes so the device can move its regions on the bus.
    virtual void on_config_write(u8 reg, unsigned size, u32 value) { (void)reg; (void)size; (void)value; }
    virtual void on_bar_changed(int index, u32 old_base, u32 new_base) { (void)index; (void)old_base; (void)new_base; }

private:
    friend class PciBus;
    void write_bar_dword(int index, u32 value);

    std::string name_;
    PciAddress address_{};
    u8 header_type_;
    std::array<u8, 256> cfg_{};
    std::array<Bar, 6> bars_{};
};

// Configuration mechanism #1 at ports 0xCF8-0xCFF, and the registry of functions behind it.
// Functions 0:0.1 and 0:0.2 freeze a real console when read; the emulated bus answers all ones and logs.
class PciBus : public PioHandler {
public:
    explicit PciBus(Bus& bus);
    std::string_view name() const override { return "pci"; }

    void add(PciAddress at, PciDevice& dev);
    PciDevice* find(PciAddress at) const;

    u32 config_read(PciAddress at, u8 reg, unsigned size);
    void config_write(PciAddress at, u8 reg, unsigned size, u32 value);

    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;

    u32 config_address() const { return config_address_; }

private:
    Bus& bus_;
    u32 config_address_{0};
    std::map<u16, PciDevice*> devices_;
};

}  // namespace mv
