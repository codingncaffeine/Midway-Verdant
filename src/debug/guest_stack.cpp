#include "debug/guest_stack.h"

#include <format>

#include "core/kernel_image.h"

namespace mv::debug {

std::optional<PhysAddr> virt_to_phys(const Bus& bus, const CpuBackend::Regs& regs, u32 va) {
    if (!(regs.cr0 & 0x80000000u)) return static_cast<PhysAddr>(va);
    u32 pde = 0;
    if (!bus.read_ram(static_cast<PhysAddr>((regs.cr3 & ~0xFFFu) + ((va >> 22) * 4)), &pde, 4)) return std::nullopt;
    if (!(pde & 1)) return std::nullopt;
    if ((pde & 0x80) && (regs.cr4 & 0x10)) return static_cast<PhysAddr>((pde & 0xFFC00000u) | (va & 0x3FFFFFu));
    u32 pte = 0;
    if (!bus.read_ram(static_cast<PhysAddr>((pde & ~0xFFFu) + (((va >> 12) & 0x3FF) * 4)), &pte, 4)) return std::nullopt;
    if (!(pte & 1)) return std::nullopt;
    return static_cast<PhysAddr>((pte & ~0xFFFu) | (va & 0xFFF));
}

std::string describe_stack(const Bus& bus, const CpuBackend::Regs& r, unsigned words) {
    std::string s = std::format("eip {:08x} esp {:08x} ebp {:08x} eflags {:08x} cr3 {:08x} | eax {:08x} ebx {:08x} ecx {:08x} edx {:08x} esi {:08x} edi {:08x}",
                                r.eip, r.esp, r.ebp, r.eflags, r.cr3, r.eax, r.ebx, r.ecx, r.edx, r.esi, r.edi);
    // The code around eip, 48 bytes either side (eip at the bar), for disassembling offline.
    s += std::format("\n  code from {:08x}:", r.eip - 48);
    for (u32 va = r.eip - 48; va != r.eip + 48; ++va) {
        const auto pa = virt_to_phys(bus, r, va);
        u8 b = 0;
        if (va == r.eip) s += " |";
        s += (pa && bus.read_ram(*pa, &b, 1)) ? std::format(" {:02x}", b) : std::string(" --");
    }
    constexpr u32 kCodeLo = KernelImage::kVirtualBase, kCodeHi = KernelImage::kVirtualBase + 0x70000;
    for (unsigned i = 0; i < words; ++i) {
        const u32 va = r.esp + i * 4;
        if (i % 8 == 0) s += std::format("\n  {:08x}:", va);
        const auto pa = virt_to_phys(bus, r, va);
        u32 v = 0;
        if (!pa || !bus.read_ram(*pa, &v, 4)) {
            s += " --------";
            break;
        }
        s += std::format(" {:08x}{}", v, (v >= kCodeLo && v < kCodeHi) ? '*' : ' ');
    }
    return s;
}

}  // namespace mv::debug
