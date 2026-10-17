#pragma once
#include <cstdio>
#include <format>
#include <string>
#include <string_view>

#include "core/types.h"

namespace mv::trace {

// Trace categories; each can be switched on by name (--trace post,pci,...).
enum class Cat : u32 {
    Post = 1u << 0,    // writes to the POST code port 0x80
    Pci = 1u << 1,     // PCI configuration space accesses
    Smbus = 1u << 2,   // SMBus transactions
    Mmio = 1u << 3,    // MMIO to unknown or logged registers
    Pio = 1u << 4,     // port I/O to unknown or logged ports
    Irq = 1u << 5,     // interrupt lines and injections
    Flash = 1u << 6,   // flash and MCPX ROM events
    Cpu = 1u << 7,     // CPU backend events (exits, resets, faults)
    Boot = 1u << 8,    // boot-chain milestones recognised from behaviour
    Timer = 1u << 9,   // PIT/RTC/PM timer programming
    Gpu = 1u << 10,    // NV2A register traffic
    Apu = 1u << 11,    // APU register traffic and processor events
    Aci = 1u << 12,    // AC97 controller register traffic and buffer events
    All = 0xFFFFFFFFu,
};

// Enabled categories; set once at start, read everywhere.
void enable(u32 mask);
u32 enabled();
bool parse_list(std::string_view list, u32& mask_out);  // "post,pci" -> mask; false on an unknown name
inline bool on(Cat c) { return (enabled() & static_cast<u32>(c)) != 0; }

// Where lines go: stderr by default, or a file.
void set_sink(std::FILE* f);

// The current timeline position, supplied by the machine so every line carries a cycle stamp.
void set_clock_source(const Cycles* now);

void emit(Cat c, std::string_view text);

template <class... Args>
void log(Cat c, std::format_string<Args...> fmt, Args&&... args) {
    if (!on(c)) return;
    emit(c, std::format(fmt, std::forward<Args>(args)...));
}

std::string_view name_of(Cat c);

}  // namespace mv::trace
