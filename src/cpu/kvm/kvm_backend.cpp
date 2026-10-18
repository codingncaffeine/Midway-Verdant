#if defined(__linux__)
#include "cpu/kvm/kvm_backend.h"

#include <vector>
#include <cstdlib>
#include <fcntl.h>
#include <linux/kvm.h>
#include <pthread.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <format>

#include "core/machine.h"
#include "debug/guest_stack.h"
#include "debug/trace.h"

namespace mv {

namespace {
constexpr int kKickSignal = SIGUSR1;
// The kick: the handler marks the run area so a KVM_RUN entered after the signal leaves at once (the race the
// KVM documentation describes for immediate_exit), and the signal itself breaks a KVM_RUN already in progress.
std::atomic<kvm_run*> g_kick_target{nullptr};
void kick_handler(int) {
    if (kvm_run* r = g_kick_target.load(std::memory_order_relaxed)) r->immediate_exit = 1;
}

constexpr u32 kSlotRam = 0, kSlotRamAlias = 1, kSlotFlash = 2, kSlotTopPage = 3;

// The KVM structures end in flexible arrays: they are built in byte buffers, the entries addressed past the header.
struct MsrsOne {
    alignas(8) unsigned char bytes[sizeof(kvm_msrs) + sizeof(kvm_msr_entry)]{};
    kvm_msrs* header() { return reinterpret_cast<kvm_msrs*>(bytes); }
    kvm_msr_entry* entry() { return reinterpret_cast<kvm_msr_entry*>(bytes + sizeof(kvm_msrs)); }
};
struct CpuidFour {
    alignas(8) unsigned char bytes[sizeof(kvm_cpuid2) + 4 * sizeof(kvm_cpuid_entry2)]{};
    kvm_cpuid2* header() { return reinterpret_cast<kvm_cpuid2*>(bytes); }
    kvm_cpuid_entry2* entries() { return reinterpret_cast<kvm_cpuid_entry2*>(bytes + sizeof(kvm_cpuid2)); }
};
}  // namespace

KvmBackend::KvmBackend() {
    const char* p = std::getenv("MV_PROFILE_EIP");
    profile_eip_ = p && *p && *p != '0';
}

KvmBackend::~KvmBackend() {
    stop();
    join();
    g_kick_target.store(nullptr, std::memory_order_relaxed);
    if (run_) munmap(run_, run_size_);
    if (vcpu_fd_ >= 0) close(vcpu_fd_);
    if (vm_fd_ >= 0) close(vm_fd_);
    if (kvm_fd_ >= 0) close(kvm_fd_);
}

void KvmBackend::fail(Stop why, std::string detail) {
    stop_reason_ = why;
    stop_detail_ = std::move(detail);
    stop_requested_ = true;
}

bool KvmBackend::set_slot(u32 slot, u64 guest_phys, u64 size, void* host, bool readonly, std::string& error, bool log_dirty) {
    kvm_userspace_memory_region region{};
    region.slot = slot;
    region.flags = (readonly ? KVM_MEM_READONLY : 0u) | (log_dirty ? KVM_MEM_LOG_DIRTY_PAGES : 0u);
    region.guest_phys_addr = guest_phys;
    region.memory_size = size;
    region.userspace_addr = reinterpret_cast<u64>(host);
    if (ioctl(vm_fd_, KVM_SET_USER_MEMORY_REGION, &region) < 0) {
        error = std::format("KVM_SET_USER_MEMORY_REGION slot {} at {:#x} failed: {}", slot, guest_phys, std::strerror(errno));
        return false;
    }
    return true;
}

bool KvmBackend::setup_cpuid(std::string& error) {
    // A Coppermine Pentium III as the console's firmware and games see it: family 6, model 8, stepping 10;
    // FPU VME DE PSE TSC MSR PAE MCE CX8 SEP MTRR PGE MCA CMOV PAT PSE36 MMX FXSR SSE; no SSE2, no APIC in use.
    CpuidFour cpuid{};
    cpuid.header()->nent = 4;
    kvm_cpuid_entry2* e = cpuid.entries();
    e[0].function = 0;
    e[0].eax = 2;
    e[0].ebx = 0x756E6547;  // "Genu"
    e[0].edx = 0x49656E69;  // "ineI"
    e[0].ecx = 0x6C65746E;  // "ntel"
    e[1].function = 1;
    e[1].eax = cpuid_signature_;
    e[1].ebx = 0;
    e[1].ecx = 0;
    e[1].edx = 0x0383F9FF;
    e[2].function = 2;
    e[2].eax = 0x03020101;
    e[2].ebx = 0;
    e[2].ecx = 0;
    e[2].edx = 0x0C040841;
    e[3].function = 0x80000000;
    e[3].eax = 0;
    if (ioctl(vcpu_fd_, KVM_SET_CPUID2, cpuid.header()) < 0) {
        error = std::format("KVM_SET_CPUID2 failed: {}", std::strerror(errno));
        return false;
    }
    return true;
}

bool KvmBackend::init(Machine& machine, std::string& error) {
    machine_ = &machine;
    kvm_fd_ = open("/dev/kvm", O_RDWR | O_CLOEXEC);
    if (kvm_fd_ < 0) {
        error = std::format("cannot open /dev/kvm: {}", std::strerror(errno));
        return false;
    }
    const int version = ioctl(kvm_fd_, KVM_GET_API_VERSION, 0);
    if (version != KVM_API_VERSION) {
        error = std::format("KVM API version {} (expected {})", version, KVM_API_VERSION);
        return false;
    }
    if (ioctl(kvm_fd_, KVM_CHECK_EXTENSION, KVM_CAP_READONLY_MEM) <= 0) {
        error = "KVM lacks read-only memory slots";
        return false;
    }
    vm_fd_ = ioctl(kvm_fd_, KVM_CREATE_VM, 0);
    if (vm_fd_ < 0) {
        error = std::format("KVM_CREATE_VM failed: {}", std::strerror(errno));
        return false;
    }
    // Intel hosts need these for real-mode guests; harmless elsewhere.
    ioctl(vm_fd_, KVM_SET_TSS_ADDR, 0xFFFBD000);
    u64 identity = 0xFFFBC000;
    ioctl(vm_fd_, KVM_SET_IDENTITY_MAP_ADDR, &identity);

    Bus& bus = machine.bus();
    Flash& flash = machine.flash();
    if (!set_slot(kSlotRam, 0, bus.ram_size(), bus.ram(), false, error, dirty_tracking_)) return false;
    if (!set_slot(kSlotRamAlias, Bus::kRamAliasBase, bus.ram_size(), bus.ram(), false, error, dirty_tracking_)) return false;
    if (dirty_tracking_) dirty_bitmap_.assign((bus.ram_size() / 4096 + 63) / 64, 0);
    if (!set_slot(kSlotFlash, Flash::kBase, Flash::kWindow - 4096, const_cast<u8*>(flash.window()), true, error)) return false;
    const void* top = flash.rom_visible() ? flash.overlay_page() : flash.window() + Flash::kWindow - 4096;
    if (!set_slot(kSlotTopPage, 0xFFFFF000u, 4096, const_cast<void*>(top), true, error)) return false;

    vcpu_fd_ = ioctl(vm_fd_, KVM_CREATE_VCPU, 0);
    if (vcpu_fd_ < 0) {
        error = std::format("KVM_CREATE_VCPU failed: {}", std::strerror(errno));
        return false;
    }
    const int mmap_size = ioctl(kvm_fd_, KVM_GET_VCPU_MMAP_SIZE, 0);
    if (mmap_size <= 0) {
        error = "KVM_GET_VCPU_MMAP_SIZE failed";
        return false;
    }
    run_size_ = static_cast<std::size_t>(mmap_size);
    void* p = mmap(nullptr, run_size_, PROT_READ | PROT_WRITE, MAP_SHARED, vcpu_fd_, 0);
    if (p == MAP_FAILED) {
        error = std::format("mmap of the vCPU run area failed: {}", std::strerror(errno));
        return false;
    }
    run_ = static_cast<kvm_run*>(p);
    if (!setup_cpuid(error)) return false;
    arm_breakpoints();

    if (ioctl(kvm_fd_, KVM_CHECK_EXTENSION, KVM_CAP_TSC_CONTROL) > 0) {
        if (ioctl(vcpu_fd_, KVM_SET_TSC_KHZ, 733333) == 0) tsc_scaled_ = true;
    }
    if (!tsc_scaled_) trace::log(trace::Cat::Cpu, "TSC scaling unavailable: the guest TSC runs at host rate");

    // The kick signal stays deliverable: its handler sets immediate_exit on the run area.
    g_kick_target.store(run_, std::memory_order_relaxed);
    struct sigaction sa{};
    sa.sa_handler = kick_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(kKickSignal, &sa, nullptr);

    machine.flash().add_on_rom_hidden([this] { on_rom_hidden(); });
    machine.flash().add_on_rom_shown([this] { on_rom_shown(); });
    reset();
    return true;
}

void KvmBackend::set_top_page(const void* backing) {
    std::string error;
    // Replace the slot: delete (size 0) then add with the new backing.
    kvm_userspace_memory_region region{};
    region.slot = kSlotTopPage;
    region.memory_size = 0;
    region.guest_phys_addr = 0xFFFFF000u;
    ioctl(vm_fd_, KVM_SET_USER_MEMORY_REGION, &region);
    if (!set_slot(kSlotTopPage, 0xFFFFF000u, 4096, const_cast<void*>(backing), true, error)) fail(Stop::InternalError, error);
}

void KvmBackend::on_rom_hidden() { set_top_page(machine_->flash().window() + Flash::kWindow - 4096); }

void KvmBackend::on_rom_shown() { set_top_page(machine_->flash().overlay_page()); }

void KvmBackend::reset() {
    kvm_sregs sregs{};
    ioctl(vcpu_fd_, KVM_GET_SREGS, &sregs);
    auto seg = [](u16 selector, u64 base, u32 limit, u8 type) {
        kvm_segment s{};
        s.selector = selector;
        s.base = base;
        s.limit = limit;
        s.type = type;
        s.present = 1;
        s.s = 1;
        s.dpl = 0;
        s.db = 0;
        s.g = 0;
        return s;
    };
    sregs.cs = seg(0xF000, 0xFFFF0000, 0xFFFF, 0xB);
    sregs.ds = seg(0, 0, 0xFFFF, 0x3);
    sregs.es = seg(0, 0, 0xFFFF, 0x3);
    sregs.fs = seg(0, 0, 0xFFFF, 0x3);
    sregs.gs = seg(0, 0, 0xFFFF, 0x3);
    sregs.ss = seg(0, 0, 0xFFFF, 0x3);
    sregs.tr = seg(0, 0, 0xFFFF, 0xB);
    sregs.tr.s = 0;
    sregs.ldt = seg(0, 0, 0xFFFF, 0x2);
    sregs.ldt.s = 0;
    sregs.gdt.base = 0;
    sregs.gdt.limit = 0xFFFF;
    sregs.idt.base = 0;
    sregs.idt.limit = 0xFFFF;
    sregs.cr0 = 0x60000010;
    sregs.cr2 = 0;
    sregs.cr3 = 0;
    sregs.cr4 = 0;
    sregs.efer = 0;
    sregs.apic_base = 0;
    ioctl(vcpu_fd_, KVM_SET_SREGS, &sregs);

    kvm_regs regs{};
    regs.rip = 0xFFF0;
    regs.rflags = 0x2;
    regs.rdx = cpuid_signature_;
    ioctl(vcpu_fd_, KVM_SET_REGS, &regs);

    // TSC starts at zero at power-on.
    MsrsOne msrs{};
    msrs.header()->nmsrs = 1;
    msrs.entry()->index = 0x10;  // IA32_TIME_STAMP_COUNTER
    msrs.entry()->data = 0;
    ioctl(vcpu_fd_, KVM_SET_MSRS, msrs.header());
    trace::log(trace::Cat::Cpu, "reset: real mode, CS f000:fff0, signature {:08x}", cpuid_signature_);
}

void KvmBackend::start() {
    stop_requested_ = false;
    running_ = true;
    thread_ = std::thread([this] { thread_main(); });
}

void KvmBackend::stop() {
    stop_requested_ = true;
    kick();
}

void KvmBackend::join() {
    if (thread_.joinable()) thread_.join();
    if (profile_eip_ && !eip_samples_.empty()) {
        std::vector<std::pair<u32, u64>> top(eip_samples_.begin(), eip_samples_.end());
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        u64 total = 0;
        for (const auto& [eip, n] : top) total += n;
        std::fprintf(stderr, "eip profile (%llu samples):\n", static_cast<unsigned long long>(total));
        for (std::size_t i = 0; i < top.size() && i < 20; ++i)
            std::fprintf(stderr, "  %08x  %6llu  %5.1f%%\n", top[i].first, static_cast<unsigned long long>(top[i].second), 100.0 * static_cast<double>(top[i].second) / static_cast<double>(total));
        eip_samples_.clear();
    }
    running_ = false;
}

void KvmBackend::kick() {
    ++stats_.kicks;
    if (run_) run_->immediate_exit = 1;
    if (thread_.joinable()) pthread_kill(thread_.native_handle(), kKickSignal);
    machine_->wake();
}

void KvmBackend::inject_interrupt_if_possible() {
    Pic& pic = machine_->pic();
    if (pic.pending_vector() < 0) {
        run_->request_interrupt_window = 0;
        return;
    }
    if (run_->ready_for_interrupt_injection && run_->if_flag) {
        kvm_interrupt irq{};
        irq.irq = static_cast<u32>(pic.acknowledge());
        if (ioctl(vcpu_fd_, KVM_INTERRUPT, &irq) == 0) {
            ++stats_.irq_injections;
            ++stats_.by_vector[irq.irq & 0xFF];
            trace::log(trace::Cat::Irq, "injected vector {:02x}", irq.irq);
        }
        run_->request_interrupt_window = pic.pending_vector() >= 0 ? 1 : 0;
    } else {
        run_->request_interrupt_window = 1;
    }
}

void KvmBackend::wait_for_interrupt() {
    // HLT with interrupts enabled: sleep until a device raises a line (the io thread kicks us) or a stop.
    machine_->wait_for_wake([this] { return stop_requested_.load() || machine_->pic().pending_vector() >= 0; });
}

void KvmBackend::handle_io() {
    Bus& bus = machine_->bus();
    u8* data = reinterpret_cast<u8*>(run_) + run_->io.data_offset;
    const unsigned size = run_->io.size;
    for (u32 i = 0; i < run_->io.count; ++i, data += size) {
        if (run_->io.direction == KVM_EXIT_IO_OUT) {
            u32 v = 0;
            std::memcpy(&v, data, size);
            bus.pio_write(run_->io.port, size, v);
        } else {
            const u32 v = bus.pio_read(run_->io.port, size);
            std::memcpy(data, &v, size);
        }
    }
    ++stats_.pio_exits;
}

void KvmBackend::handle_mmio() {
    Bus& bus = machine_->bus();
    const PhysAddr addr = static_cast<PhysAddr>(run_->mmio.phys_addr);
    const unsigned len = run_->mmio.len;
    if (run_->mmio.is_write) {
        if (len == 8) {
            u64 v;
            std::memcpy(&v, run_->mmio.data, 8);
            bus.write64(addr, v);
        } else {
            u32 v = 0;
            std::memcpy(&v, run_->mmio.data, len);
            bus.write(addr, len, v);
        }
    } else {
        if (len == 8) {
            const u64 v = bus.read64(addr);
            std::memcpy(run_->mmio.data, &v, 8);
        } else {
            const u32 v = bus.read(addr, len);
            std::memcpy(run_->mmio.data, &v, len);
        }
    }
    ++stats_.mmio_exits;
}

void KvmBackend::thread_main() {
    sigset_t unblock;
    sigemptyset(&unblock);
    sigaddset(&unblock, kKickSignal);
    pthread_sigmask(SIG_UNBLOCK, &unblock, nullptr);

    while (!stop_requested_) {
        {
            std::lock_guard lock(machine_->lock());
            machine_->advance_time();
            inject_interrupt_if_possible();
        }
        run_->immediate_exit = 0;
        const int rc = ioctl(vcpu_fd_, KVM_RUN, 0);
        ++stats_.exits;
        if (rc < 0 && errno == EINTR) {
            ++stats_.intr_exits;
            if (profile_eip_) {
                kvm_regs r{};
                if (ioctl(vcpu_fd_, KVM_GET_REGS, &r) == 0) ++eip_samples_[static_cast<u32>(r.rip)];
            }
            continue;  // kicked: loop to inject or stop
        }
        if (rc < 0) {
            fail(Stop::InternalError, std::format("KVM_RUN failed: {}", std::strerror(errno)));
            break;
        }
        std::unique_lock lock(machine_->lock());
        machine_->advance_time();
        if (run_->exit_reason < 64) ++stats_.by_reason[run_->exit_reason];
        switch (run_->exit_reason) {
            case KVM_EXIT_IO: handle_io(); break;
            case KVM_EXIT_MMIO: handle_mmio(); break;
            case KVM_EXIT_HLT:
                ++stats_.hlt_exits;
                wait_for_interrupt();
                break;
            case KVM_EXIT_IRQ_WINDOW_OPEN: ++stats_.irq_window_exits; break;
            case KVM_EXIT_INTR: break;
            case KVM_EXIT_DEBUG: {
                // A --break-eip address: log where the guest is and drop every breakpoint so it runs on.
                Regs r{};
                lock.unlock();
                read_regs(r);
                trace::log(trace::Cat::Boot, "break: eip {:08x} reached\n{}", r.eip, debug::describe_stack(machine_->bus(), r));
                kvm_guest_debug off{};
                ioctl(vcpu_fd_, KVM_SET_GUEST_DEBUG, &off);
                breakpoints_armed_ = false;
                continue;
            }
            case KVM_EXIT_SHUTDOWN: {
                Regs r{};
                lock.unlock();
                read_regs(r);
                fail(Stop::TripleFault, std::format("triple fault at cs {:04x} eip {:08x}", r.cs, r.eip));
                continue;
            }
            case KVM_EXIT_FAIL_ENTRY:
                fail(Stop::InternalError,
                     std::format("KVM_EXIT_FAIL_ENTRY reason {:#x}", static_cast<u64>(run_->fail_entry.hardware_entry_failure_reason)));
                break;
            case KVM_EXIT_INTERNAL_ERROR:
                fail(Stop::InternalError, std::format("KVM internal error {}", run_->internal.suberror));
                break;
            default:
                fail(Stop::InternalError, std::format("unhandled KVM exit {}", run_->exit_reason));
                break;
        }
    }
    running_ = false;
    machine_->wake();
}

// Hardware execution breakpoints (DR0-DR3) on the --break-eip addresses; the debug exit logs and disarms them.
void KvmBackend::arm_breakpoints() {
    if (break_eips_.empty()) return;
    kvm_guest_debug dbg{};
    dbg.control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW_BP;
    dbg.arch.debugreg[7] = 1u << 10;  // the bit that is always set in DR7
    for (std::size_t i = 0; i < break_eips_.size() && i < 4; ++i) {
        dbg.arch.debugreg[i] = break_eips_[i];
        dbg.arch.debugreg[7] |= 1u << (i * 2);  // Li enabled; R/Wi 00 = execution, LENi 00 = one byte
    }
    if (ioctl(vcpu_fd_, KVM_SET_GUEST_DEBUG, &dbg) < 0) {
        trace::log(trace::Cat::Boot, "break: KVM_SET_GUEST_DEBUG failed: {}", std::strerror(errno));
        return;
    }
    breakpoints_armed_ = true;
}

void KvmBackend::take_dirty_pages(std::vector<u32>& pages) {
    if (!dirty_tracking_ || vm_fd_ < 0) return;
    // Both RAM slots (direct and the GPU-side alias) back the same pages; the union of their logs is what changed.
    std::vector<u64> merged(dirty_bitmap_.size(), 0);
    for (u32 slot : {kSlotRam, kSlotRamAlias}) {
        std::fill(dirty_bitmap_.begin(), dirty_bitmap_.end(), 0);
        kvm_dirty_log log{};
        log.slot = slot;
        log.dirty_bitmap = dirty_bitmap_.data();
        if (ioctl(vm_fd_, KVM_GET_DIRTY_LOG, &log) < 0) continue;
        for (std::size_t i = 0; i < merged.size(); ++i) merged[i] |= dirty_bitmap_[i];
    }
    for (std::size_t w = 0; w < merged.size(); ++w) {
        u64 bits = merged[w];
        while (bits) {
            const unsigned bit = static_cast<unsigned>(std::countr_zero(bits));
            pages.push_back(static_cast<u32>(w * 64 + bit));
            bits &= bits - 1;
        }
    }
}

bool KvmBackend::read_regs(Regs& out) {
    kvm_regs regs{};
    kvm_sregs sregs{};
    if (ioctl(vcpu_fd_, KVM_GET_REGS, &regs) < 0) return false;
    if (ioctl(vcpu_fd_, KVM_GET_SREGS, &sregs) < 0) return false;
    out.eax = static_cast<u32>(regs.rax);
    out.ebx = static_cast<u32>(regs.rbx);
    out.ecx = static_cast<u32>(regs.rcx);
    out.edx = static_cast<u32>(regs.rdx);
    out.esi = static_cast<u32>(regs.rsi);
    out.edi = static_cast<u32>(regs.rdi);
    out.ebp = static_cast<u32>(regs.rbp);
    out.esp = static_cast<u32>(regs.rsp);
    out.eip = static_cast<u32>(regs.rip);
    out.eflags = static_cast<u32>(regs.rflags);
    out.cr0 = static_cast<u32>(sregs.cr0);
    out.cr3 = static_cast<u32>(sregs.cr3);
    out.cr4 = static_cast<u32>(sregs.cr4);
    out.cs = sregs.cs.selector;
    out.ds = sregs.ds.selector;
    out.es = sregs.es.selector;
    out.fs = sregs.fs.selector;
    out.gs = sregs.gs.selector;
    out.ss = sregs.ss.selector;
    MsrsOne msrs{};
    msrs.header()->nmsrs = 1;
    msrs.entry()->index = 0x10;
    out.tsc = ioctl(vcpu_fd_, KVM_GET_MSRS, msrs.header()) == 1 ? msrs.entry()->data : 0;
    return true;
}

std::unique_ptr<CpuBackend> make_fast_backend(std::string& why_not) {
    if (access("/dev/kvm", R_OK | W_OK) != 0) {
        why_not = std::format("/dev/kvm is not accessible: {}", std::strerror(errno));
        return nullptr;
    }
    return std::make_unique<KvmBackend>();
}

}  // namespace mv
#else
#include "cpu/backend.h"

namespace mv {
std::unique_ptr<CpuBackend> make_fast_backend(std::string& why_not) {
    why_not = "no hardware virtualization backend on this platform yet (WHPX is planned)";
    return nullptr;
}
}  // namespace mv
#endif
