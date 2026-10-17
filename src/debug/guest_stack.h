#pragma once
#include <optional>
#include <string>

#include "core/bus.h"
#include "cpu/backend.h"

namespace mv::debug {

// Translates a guest virtual address through the paging structures the registers point at (32-bit paging,
// 4 KiB and 4 MiB pages); identity when paging is off; nullopt for an unmapped address.
std::optional<PhysAddr> virt_to_phys(const Bus& bus, const CpuBackend::Regs& regs, u32 va);

// The general registers and the words on the stack from esp up, eight per row; a word that falls inside the
// kernel image's address range is marked with a star — the return addresses among them are the call chain.
std::string describe_stack(const Bus& bus, const CpuBackend::Regs& regs, unsigned words = 96);

}  // namespace mv::debug
