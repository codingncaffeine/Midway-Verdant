#include "debug/trace.h"

#include <array>
#include <mutex>
#include <utility>

namespace mv::trace {

namespace {
u32 g_enabled = 0;
std::FILE* g_sink = stderr;
const Cycles* g_now = nullptr;
std::mutex g_mutex;

struct Name {
    Cat cat;
    std::string_view name;
};
constexpr std::array<Name, 13> kNames{{{Cat::Post, "post"},
                                       {Cat::Pci, "pci"},
                                       {Cat::Smbus, "smbus"},
                                       {Cat::Mmio, "mmio"},
                                       {Cat::Pio, "pio"},
                                       {Cat::Irq, "irq"},
                                       {Cat::Flash, "flash"},
                                       {Cat::Cpu, "cpu"},
                                       {Cat::Boot, "boot"},
                                       {Cat::Timer, "timer"},
                                       {Cat::Gpu, "gpu"},
                                       {Cat::Apu, "apu"},
                                       {Cat::Aci, "aci"}}};
}  // namespace

void enable(u32 mask) { g_enabled = mask; }
u32 enabled() { return g_enabled; }
void set_sink(std::FILE* f) { g_sink = f ? f : stderr; }
void set_clock_source(const Cycles* now) { g_now = now; }

std::string_view name_of(Cat c) {
    for (const auto& n : kNames) {
        if (n.cat == c) return n.name;
    }
    return "?";
}

bool parse_list(std::string_view list, u32& mask_out) {
    u32 mask = 0;
    while (!list.empty()) {
        const auto comma = list.find(',');
        const std::string_view item = list.substr(0, comma);
        if (item == "all") {
            mask = static_cast<u32>(Cat::All);
        } else if (!item.empty()) {
            bool found = false;
            for (const auto& n : kNames) {
                if (n.name == item) {
                    mask |= static_cast<u32>(n.cat);
                    found = true;
                }
            }
            if (!found) return false;
        }
        if (comma == std::string_view::npos) break;
        list.remove_prefix(comma + 1);
    }
    mask_out = mask;
    return true;
}

void emit(Cat c, std::string_view text) {
    std::lock_guard lock(g_mutex);
    const Cycles now = g_now ? *g_now : 0;
    std::fprintf(g_sink, "[%12llu] %-5.*s %.*s\n", static_cast<unsigned long long>(now),
                 static_cast<int>(name_of(c).size()), name_of(c).data(), static_cast<int>(text.size()), text.data());
    std::fflush(g_sink);
}

}  // namespace mv::trace
