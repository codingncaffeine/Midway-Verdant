#pragma once
#include <linux/kvm.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "core/types.h"

namespace mv::kvm {

// Entry i of the ring (the header declares the entries as a zero-length array, which GCC bounds-checks).
inline kvm_coalesced_mmio* coalesced_entry(kvm_coalesced_mmio_ring* ring, u32 i) {
    return reinterpret_cast<kvm_coalesced_mmio*>(reinterpret_cast<std::uintptr_t>(ring) + offsetof(kvm_coalesced_mmio_ring, coalesced_mmio)) + i;
}

// KVM's coalesced MMIO ring (KVM_CAP_COALESCED_MMIO): a write to a registered zone does not leave the guest; the
// kernel appends it at `last` and the VMM takes the writes from `first`, in the order the guest made them, handing
// each slot back as it goes. A full ring makes the kernel exit with the write as usual. `slots` is the ring's length
// (the page after the header, in entries); `sink(phys, len, data)` receives each write. Returns the writes taken.
template <typename Sink>
u32 drain_coalesced(kvm_coalesced_mmio_ring* ring, u32 slots, Sink&& sink) {
    std::atomic_ref<__u32> first_ref(ring->first), last_ref(ring->last);
    u32 first = first_ref.load(std::memory_order_relaxed);
    const u32 last = last_ref.load(std::memory_order_acquire);  // the entries before it are written
    u32 taken = 0;
    while (first != last) {
        const kvm_coalesced_mmio& e = *coalesced_entry(ring, first);
        sink(static_cast<u64>(e.phys_addr), static_cast<u32>(e.len), e.data);
        first = (first + 1) % slots;
        first_ref.store(first, std::memory_order_release);  // the slot is the kernel's again
        ++taken;
    }
    return taken;
}

}  // namespace mv::kvm
