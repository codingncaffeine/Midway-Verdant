#if defined(__linux__)
#include "cpu/kvm/kvm_backend.h"

#include <vector>
#include <cstdlib>
#include <fcntl.h>
#include <linux/kvm.h>
#include <pthread.h>
#include <signal.h>
#include <sys/ioctl.h>

#include "cpu/kvm/kvm_ioctl.h"

// The thread a SIGEV_THREAD_ID timer signals: sigev_notify_thread_id in musl and glibc 2.35 on, only the internal
// _sigev_un._tid before that (the release core builds on glibc 2.28).
#if defined(sigev_notify_thread_id)
#define MV_SIGEV_THREAD_ID sigev_notify_thread_id
#else
#define MV_SIGEV_THREAD_ID _sigev_un._tid
#endif
#include <sys/mman.h>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>

#include "core/clock.h"
#include "cpu/kvm/coalesced_ring.h"

#include "core/machine.h"
#include "debug/guest_stack.h"
#include "debug/trace.h"

namespace mv {

namespace {
constexpr int kKickSignal = SIGUSR1;
// The kick: the handler marks the run area so a KVM_RUN entered after the signal leaves at once (the race the
// KVM documentation describes for immediate_exit), and the signal itself breaks a KVM_RUN already in progress.
std::atomic<kvm_run*> g_kick_target{nullptr};
// The flag is shared by this handler, kick() on other threads and the vCPU thread's reset: atomic stores, and the
// handler keeps errno as it found it (it can land between a failing call and the read of errno).
void set_immediate_exit(kvm_run* r, u8 v) { std::atomic_ref<u8>(r->immediate_exit).store(v, std::memory_order_relaxed); }
void kick_handler(int) {
    const int saved = errno;
    if (kvm_run* r = g_kick_target.load(std::memory_order_relaxed)) set_immediate_exit(r, 1);
    errno = saved;
}
// MV_PROFILE_EIP_US=N: a timer on the vCPU thread's CPU clock breaks KVM_RUN every N us of its CPU time, so the EIP
// samples fall evenly over the time the guest takes (the kicks alone fall where devices wake it).
constexpr int kProfileSignal = SIGUSR2;
std::atomic<bool> g_profile_tick{false};
void profile_handler(int) {
    const int saved = errno;
    g_profile_tick.store(true, std::memory_order_relaxed);
    if (kvm_run* r = g_kick_target.load(std::memory_order_relaxed)) set_immediate_exit(r, 1);
    errno = saved;
}

constexpr u32 kSlotRam = 0, kSlotRamAlias = 1, kSlotFlash = 2, kSlotTopPage = 3;
// What a device write costs the console with exact exits (D33): the accurate core's model (tp::kDeviceStore,
// kPortIo in cpu/p6/timing.h), a posted write and a port access. A read keeps the host's price (GuestClock::classify).
constexpr Cycles kDeviceStoreCycles = 8, kPortIoCycles = 300;

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

u64 clock_ns(clockid_t id) {
    timespec t{};
    clock_gettime(id, &t);
    return static_cast<u64>(t.tv_sec) * 1'000'000'000ull + static_cast<u64>(t.tv_nsec);
}
// The host's steady clock and this thread's CPU time.
u64 mono_ns() { return clock_ns(CLOCK_MONOTONIC); }
u64 thread_cpu_ns() { return clock_ns(CLOCK_THREAD_CPUTIME_ID); }
}  // namespace

KvmBackend::KvmBackend() {
    const char* p = std::getenv("MV_PROFILE_EIP");
    profile_eip_ = p && *p && *p != '0';
    if (profile_eip_ && std::strchr(p, '-')) {
        char* end = nullptr;
        profile_from_ = std::strtoull(p, &end, 10);
        profile_to_ = std::strtoull(end + 1, nullptr, 10);
    }
    debug_lead_ = std::getenv("MV_DEBUG_TSC_LEAD") != nullptr;
    if (const char* us = std::getenv("MV_PROFILE_EIP_US"); profile_eip_ && us && *us) profile_us_ = std::min(999u, static_cast<unsigned>(std::strtoul(us, nullptr, 10)));
    if (const char* s = std::getenv("MV_IDLE_SKIP"); s && *s == '0') idle_skip_ = false;
    if (const char* f = std::getenv("MV_BREAK_FROM"); f && *f) break_from_ = std::strtoull(f, nullptr, 10);
    if (const char* w = std::getenv("MV_WATCH_AT_BREAK"); w && *w) {
        const std::string s = w;
        const auto plus = s.find('+'), star = s.find('*');
        watch_reg_ = s.substr(0, plus);
        if (plus != std::string::npos) watch_offset_ = static_cast<u32>(std::strtoul(s.c_str() + plus + 1, nullptr, 16));
        if (star != std::string::npos) watch_max_ = static_cast<u32>(std::strtoul(s.c_str() + star + 1, nullptr, 10));
    }
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
    if (kvm_ioctl(vm_fd_, KVM_SET_USER_MEMORY_REGION, &region) < 0) {
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
    if (kvm_ioctl(vcpu_fd_, KVM_SET_CPUID2, cpuid.header()) < 0) {
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
    const int version = kvm_ioctl(kvm_fd_, KVM_GET_API_VERSION, 0);
    if (version != KVM_API_VERSION) {
        error = std::format("KVM API version {} (expected {})", version, KVM_API_VERSION);
        return false;
    }
    if (kvm_ioctl(kvm_fd_, KVM_CHECK_EXTENSION, KVM_CAP_READONLY_MEM) <= 0) {
        error = "KVM lacks read-only memory slots";
        return false;
    }
    vm_fd_ = kvm_ioctl(kvm_fd_, KVM_CREATE_VM, 0);
    if (vm_fd_ < 0) {
        error = std::format("KVM_CREATE_VM failed: {}", std::strerror(errno));
        return false;
    }
    // Intel hosts need these for real-mode guests; harmless elsewhere.
    kvm_ioctl(vm_fd_, KVM_SET_TSS_ADDR, 0xFFFBD000);
    u64 identity = 0xFFFBC000;
    kvm_ioctl(vm_fd_, KVM_SET_IDENTITY_MAP_ADDR, &identity);

    // Selective dirty tracking: the log starts all set and KVM protects a page only when we clear its bit, so the RAM
    // keeps large pages except where a texture lies (KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2).
    if (dirty_tracking_ && selective_requested_) {
        const int caps = kvm_ioctl(kvm_fd_, KVM_CHECK_EXTENSION, KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2);
        if (caps > 0 && (caps & KVM_DIRTY_LOG_MANUAL_PROTECT_ENABLE)) {
            kvm_enable_cap cap{};
            cap.cap = KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2;
            cap.args[0] = static_cast<__u64>(static_cast<unsigned>(KVM_DIRTY_LOG_MANUAL_PROTECT_ENABLE) | (static_cast<unsigned>(caps) & static_cast<unsigned>(KVM_DIRTY_LOG_INITIALLY_SET)));
            selective_ = kvm_ioctl(vm_fd_, KVM_ENABLE_CAP, &cap) == 0;
        }
    }
    Bus& bus = machine.bus();
    Flash& flash = machine.flash();
    if (!set_slot(kSlotRam, 0, bus.ram_size(), bus.ram(), false, error, dirty_tracking_)) return false;
    if (!set_slot(kSlotRamAlias, Bus::kRamAliasBase, bus.ram_size(), bus.ram(), false, error, dirty_tracking_)) return false;
    if (dirty_tracking_) dirty_bitmap_.assign((bus.ram_size() / 4096 + 63) / 64, 0);
    if (selective_) watched_.assign(dirty_bitmap_.size(), 0);
    if (!set_slot(kSlotFlash, Flash::kBase, Flash::kWindow - 4096, const_cast<u8*>(flash.window()), true, error)) return false;
    const void* top = flash.rom_visible() ? flash.overlay_page() : flash.window() + Flash::kWindow - 4096;
    if (!set_slot(kSlotTopPage, 0xFFFFF000u, 4096, const_cast<void*>(top), true, error)) return false;

    vcpu_fd_ = kvm_ioctl(vm_fd_, KVM_CREATE_VCPU, 0);
    if (vcpu_fd_ < 0) {
        error = std::format("KVM_CREATE_VCPU failed: {}", std::strerror(errno));
        return false;
    }
    const int mmap_size = kvm_ioctl(kvm_fd_, KVM_GET_VCPU_MMAP_SIZE, 0);
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

    // Device writes without an exit each where a device asks for it (Bus::coalesce_writes): KVM's coalesced MMIO
    // ring, in the run area's page the capability names, taken whenever the machine advances time (either thread,
    // under its lock: before the access an exit stopped for, and before any device event). MV_COALESCE=0 keeps an
    // exit per write (the control).
    if (const char* c = std::getenv("MV_COALESCE"); !(c && *c == '0')) {
        const int page = kvm_ioctl(kvm_fd_, KVM_CHECK_EXTENSION, KVM_CAP_COALESCED_MMIO);
        const auto page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        if (page > 0 && (static_cast<std::size_t>(page) + 1) * page_size <= run_size_) {
            coalesced_ = reinterpret_cast<kvm_coalesced_mmio_ring*>(reinterpret_cast<u8*>(run_) + static_cast<std::size_t>(page) * page_size);
            coalesced_slots_ = static_cast<u32>((page_size - sizeof(kvm_coalesced_mmio_ring)) / sizeof(kvm_coalesced_mmio));
            machine.set_before_events([this] { drain_coalesced_writes(); });
            machine.bus().set_coalesce_hook([this](PhysAddr base, u32 size, bool on) {
                if (!on) drain_coalesced_writes();  // what the zone took lands first
                kvm_coalesced_mmio_zone zone{};
                zone.addr = base;
                zone.size = size;
                if (kvm_ioctl(vm_fd_, on ? KVM_REGISTER_COALESCED_MMIO : KVM_UNREGISTER_COALESCED_MMIO, &zone) < 0) ++coalesce_failures_;
                else coalesced_zones_ += on ? 1 : -1;
            });
        }
    }

    // The timeline: the guest clock unless the configuration or MV_GUEST_CLOCK asks for the host clock.
    std::string mode = machine.config().guest_clock;
    if (const char* e = std::getenv("MV_GUEST_CLOCK"); e && *e) mode = e;
    if (mode == "wall") {
        guest_clock_on_ = false;
    } else if (mode != "cputime") {
        error = std::format("unknown guest clock '{}' (cputime or wall)", mode);
        return false;
    }
    double scale = machine.config().fast_cpu_scale;
    if (const char* e = std::getenv("MV_FAST_CPU_SCALE"); e && *e) scale = std::atof(e);
    if (!(scale >= 0.25 && scale <= 64.0)) {
        error = std::format("fast CPU scale {} out of range (0.25 to 64)", scale);
        return false;
    }
    clock_.set_scale_milli(static_cast<u32>(std::lround(scale * 1000.0)));
    // The TSC counts 733.333 MHz of guest time; with the guest clock, guest time inside the guest runs `scale` times
    // faster than the host's, and so must the TSC.
    if (kvm_ioctl(kvm_fd_, KVM_CHECK_EXTENSION, KVM_CAP_TSC_CONTROL) > 0) {
        if (kvm_ioctl(vcpu_fd_, KVM_SET_TSC_KHZ, guest_clock_on_ ? clock_.tsc_khz() : 733333u) == 0) tsc_scaled_ = true;
    }
    if (!tsc_scaled_) trace::log(trace::Cat::Cpu, "TSC scaling unavailable: the guest TSC runs at host rate");
    {
        kvm_device_attr a{};
        a.group = KVM_VCPU_TSC_CTRL;
        a.attr = KVM_VCPU_TSC_OFFSET;
        tsc_offset_ctl_ = kvm_ioctl(vcpu_fd_, KVM_HAS_DEVICE_ATTR, &a) == 0;
    }
    // Exact exits (D33) need the rewinds; MV_EXIT_COST=host keeps the old account (exits cost guest time at the
    // host's price), the control.
    {
        const char* ec = std::getenv("MV_EXIT_COST");
        clock_.set_exact_exits(guest_clock_on_ && tsc_offset_ctl_ && !(ec && std::strcmp(ec, "host") == 0));
    }
    if (guest_clock_on_ && tsc_offset_ctl_) {
        // The guest's own writes of the TSC come out here, so the clock keeps its account of where the TSC should be.
        kvm_enable_cap cap{};
        cap.cap = KVM_CAP_X86_USER_SPACE_MSR;
        cap.args[0] = KVM_MSR_EXIT_REASON_FILTER;
        u8 deny = 0;  // the bitmap is copied: no bit set = the write leaves the guest
        kvm_msr_filter filter{};
        filter.flags = KVM_MSR_FILTER_DEFAULT_ALLOW;
        filter.ranges[0].flags = KVM_MSR_FILTER_WRITE;
        filter.ranges[0].base = 0x10;
        filter.ranges[0].nmsrs = 1;
        filter.ranges[0].bitmap = &deny;
        tsc_writes_trapped_ = kvm_ioctl(vm_fd_, KVM_ENABLE_CAP, &cap) == 0 && kvm_ioctl(vm_fd_, KVM_X86_SET_MSR_FILTER, &filter) == 0;
    }

    // The kick signal stays deliverable: its handler sets immediate_exit on the run area.
    g_kick_target.store(run_, std::memory_order_relaxed);
    struct sigaction sa{};
    sa.sa_handler = kick_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(kKickSignal, &sa, nullptr);
    if (profile_us_) {
        struct sigaction ps{};
        ps.sa_handler = profile_handler;
        sigemptyset(&ps.sa_mask);
        sigaction(kProfileSignal, &ps, nullptr);
    }

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
    kvm_ioctl(vm_fd_, KVM_SET_USER_MEMORY_REGION, &region);
    if (!set_slot(kSlotTopPage, 0xFFFFF000u, 4096, const_cast<void*>(backing), true, error)) fail(Stop::InternalError, error);
}

void KvmBackend::on_rom_hidden() { set_top_page(machine_->flash().window() + Flash::kWindow - 4096); }

void KvmBackend::on_rom_shown() { set_top_page(machine_->flash().overlay_page()); }

void KvmBackend::reset() {
    // An interrupt injected just before the stop stays queued in KVM until the next entry (the stop's kick returns
    // from KVM_RUN before it is delivered): after the reset it would reach the boot code in real mode, through the old
    // kernel's vector table still in RAM, and the console never came back. A reset clears pending events.
    kvm_vcpu_events events{};
    if (kvm_ioctl(vcpu_fd_, KVM_GET_VCPU_EVENTS, &events) == 0 &&
        (events.interrupt.injected || events.exception.injected || events.exception.pending || events.nmi.injected || events.nmi.pending))
        trace::log(trace::Cat::Cpu, "reset: dropped a pending event (interrupt {} vector {:02x}, exception {}, nmi {})", events.interrupt.injected, events.interrupt.nr,
                   events.exception.injected | events.exception.pending, events.nmi.injected | events.nmi.pending);
    events = {};
    events.flags = KVM_VCPUEVENT_VALID_NMI_PENDING | KVM_VCPUEVENT_VALID_SHADOW;
    kvm_ioctl(vcpu_fd_, KVM_SET_VCPU_EVENTS, &events);
    if (run_) run_->request_interrupt_window = 0;
    kvm_sregs sregs{};
    kvm_ioctl(vcpu_fd_, KVM_GET_SREGS, &sregs);
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
    kvm_ioctl(vcpu_fd_, KVM_SET_SREGS, &sregs);

    kvm_regs regs{};
    regs.rip = 0xFFF0;
    regs.rflags = 0x2;
    regs.rdx = cpuid_signature_;
    kvm_ioctl(vcpu_fd_, KVM_SET_REGS, &regs);

    // TSC starts at zero at power-on; the time until the first entry is not guest time.
    set_tsc(0);
    tsc_epoch_ = clock_.now();
    clock_.mark_out(mono_ns());
    halted_ = false;
    trace::log(trace::Cat::Cpu, "reset: real mode, CS f000:fff0, signature {:08x}", cpuid_signature_);
}

u64 KvmBackend::read_tsc() {
    MsrsOne msrs{};
    msrs.header()->nmsrs = 1;
    msrs.entry()->index = 0x10;  // IA32_TIME_STAMP_COUNTER
    return kvm_ioctl(vcpu_fd_, KVM_GET_MSRS, msrs.header()) == 1 ? msrs.entry()->data : 0;
}

bool KvmBackend::tsc_offset_add(i64 delta) {
    if (!tsc_offset_ctl_) return false;
    kvm_device_attr a{};
    a.group = KVM_VCPU_TSC_CTRL;
    a.attr = KVM_VCPU_TSC_OFFSET;
    // The offset is read once and kept: only this thread moves it (exact exits rewind it at nearly every exit).
    a.addr = reinterpret_cast<u64>(&tsc_offset_);
    if (!tsc_offset_known_) {
        if (kvm_ioctl(vcpu_fd_, KVM_GET_DEVICE_ATTR, &a) < 0) {
            ++tsc_adjust_failures_;
            return false;
        }
        tsc_offset_known_ = true;
    }
    const u64 offset = tsc_offset_ + static_cast<u64>(delta);
    a.addr = reinterpret_cast<u64>(&offset);
    if (kvm_ioctl(vcpu_fd_, KVM_SET_DEVICE_ATTR, &a) < 0) {
        ++tsc_adjust_failures_;
        return false;
    }
    tsc_offset_ = offset;
    return true;
}

void KvmBackend::set_tsc(u64 value) {
    // Through the offset when the host has it: KVM takes a host write of the MSR within a second of the value it
    // expects as a synchronization and keeps the old offset, so a warm reset's zero would be ignored.
    if (tsc_offset_add(static_cast<i64>(value - read_tsc()))) return;
    MsrsOne msrs{};
    msrs.header()->nmsrs = 1;
    msrs.entry()->index = 0x10;
    msrs.entry()->data = value;
    kvm_ioctl(vcpu_fd_, KVM_SET_MSRS, msrs.header());
    tsc_offset_known_ = false;  // KVM moved the offset itself
}

Cycles KvmBackend::timeline_cycles() const {
    // On the vCPU thread (at an exit) the bracket is closed and this is the clock's own count; another thread adds
    // the vCPU's CPU time in the guest so far.
    return clock_.estimate([this] { return vcpu_clock_known_.load(std::memory_order_acquire) ? clock_ns(vcpu_cpu_clock_.load()) : 0; });
}

double KvmBackend::timeline_rate() const { return guest_clock_on_ ? static_cast<double>(clock_.scale_milli()) / 1000.0 : 0.0; }

void KvmBackend::timeline_advanced() { arm_backstop(true); }

void KvmBackend::arm_backstop(bool always) {
    if (!backstop_ok_) return;
    // Due this long after the event: the io thread fires it first whenever it runs on time.
    // 250 us: measured against 100 us (the same frames, 7% slower) and 500 us (frames lost, 0.9 ms latency).
    constexpr Cycles kSlack = 183'333;
    const Scheduler& sched = machine_->scheduler();
    if (!sched.has_pending()) return;
    const Cycles due = sched.next_deadline() + kSlack, now = sched.now();
    const u64 now_ns = mono_ns();
    const u64 at = now_ns + std::max<u64>(due > now ? ns_from_cycles(due - now) * 1000u / clock_.scale_milli() : 0, 1);
    if (!always && backstop_at_ns_ > now_ns && backstop_at_ns_ <= at) return;  // an earlier one is on its way
    itimerspec its{};
    its.it_value.tv_sec = static_cast<time_t>(at / 1'000'000'000ull);
    its.it_value.tv_nsec = static_cast<long>(at % 1'000'000'000ull);
    if (timer_settime(backstop_, TIMER_ABSTIME, &its, nullptr) == 0) {
        backstop_at_ns_ = at;
        ++backstop_shots_;
    }
}

void KvmBackend::note_frame_tsc() {
    // The TSC against the guest clock at a frame's end: ahead by the time the host ran something else while the
    // vCPU was in the guest (the one stretch a rewind cannot take back: the guest may have read the TSC after it).
    const i64 lead = static_cast<i64>(read_tsc()) - static_cast<i64>(clock_.now() - tsc_epoch_);
    tsc_lead_last_ = lead;
    if (tsc_lead_samples_ == 0 || lead > tsc_lead_max_) tsc_lead_max_ = lead;
    if (tsc_lead_samples_ == 0 || lead < tsc_lead_min_) tsc_lead_min_ = lead;
    tsc_lead_sum_ += static_cast<double>(lead);
    ++tsc_lead_samples_;
}

std::string KvmBackend::clock_name() const {
    if (!guest_clock_on_) return "the host wall clock (MV_GUEST_CLOCK=wall)";
    return std::format("the guest clock: the vCPU's time in the guest x{:g}", static_cast<double>(clock_.scale_milli()) / 1000.0);
}

std::string KvmBackend::report() const {
    const GuestClock::Stats& s = clock_.stats();
    const double scale = static_cast<double>(clock_.scale_milli()) / 1000.0;
    const auto sec = [](u64 ns) { return static_cast<double>(ns) / 1e9; };
    const auto us = [](i64 cycles) { return static_cast<double>(cycles) / 733.333; };
    std::string r = !guest_clock_on_ ? std::string("guest clock: off, the host wall clock drives the timeline (MV_GUEST_CLOCK=wall)") : std::format(
        "guest clock: x{:g}, {:.3f} s of guest time = {:.3f} s in the guest + {:.3f} s of {} short exits (host time x{:g}); not counted: "
        "{:.3f} s in {} long exits (longest {:.1f} ms), {:.3f} s preempted in the guest, {:.3f} s of short exits off the CPU (both run the TSC ahead), "
        "{} cycles skipped at HLT",
        scale, static_cast<double>(clock_.now()) / 733'333'333.0, sec(s.in_guest_ns), sec(s.out_counted_ns), s.short_exits, scale, sec(s.long_ns),
        s.long_exits, static_cast<double>(s.longest_exit_ns) / 1e6, sec(s.preempted_ns), sec(s.short_excess_ns), s.warped_cycles);
    r += std::format("\nkvm: {} exits (mmio {}, pio {}, hlt {}, irq window {}, interrupted {}), {} kicks, {} interrupts injected",
                     stats_.exits, stats_.mmio_exits, stats_.pio_exits, stats_.hlt_exits, stats_.irq_window_exits, stats_.intr_exits, stats_.kicks,
                     stats_.irq_injections);
    if (coalesced_ || coalesce_failures_)
        r += std::format("\nkvm: {} device writes taken without an exit (coalesced MMIO, {} zones; {} zone changes refused)", coalesced_writes_, coalesced_zones_, coalesce_failures_);
    r += std::format("\nkvm: host time on the vCPU thread: firing device events {:.3f} s, exit handlers {:.3f} s, held by pacing {:.3f} s; backstop timer armed {} times",
                     sec(event_ns_), sec(handler_ns_), sec(pause_ns_), backstop_shots_);
    r += std::format("\nkvm: interrupt latency (guest us) <10 {}, <50 {}, <100 {}, <250 {}, <500 {}, <1000 {}, more {}; max {:.0f}", irq_latency_buckets_[0],
                     irq_latency_buckets_[1], irq_latency_buckets_[2], irq_latency_buckets_[3], irq_latency_buckets_[4], irq_latency_buckets_[5],
                     irq_latency_buckets_[6], us(static_cast<i64>(irq_latency_max_)));
    {
        std::vector<std::pair<u32, u64>> top(exit_sources_.begin(), exit_sources_.end());
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        r += "\nkvm: busiest exits:";
        for (std::size_t i = 0; i < top.size() && i < 8; ++i)
            r += (top[i].first & 1u) ? std::format(" port {:04x} x{}", top[i].first >> 16, top[i].second)
                                           : std::format(" mmio {:08x} x{}", top[i].first, top[i].second);
    }
    if (!stopped_at_.empty()) r += "\nkvm: the guest stopped: " + stopped_at_;
    if (idle_eip_) r += std::format("\nkvm: the kernel idle loops at {:08x} {:08x}: {} hits, {} turned into halts", idle_eips_[0], idle_eips_[1], idle_hits_, idle_halts_);
    else if (idle_skip_ && guest_clock_on_) r += std::format("\nkvm: the kernel idle loop was not found ({} scans): it spins", idle_scans_);
    if (!guest_clock_on_) return r;
    r += std::format("\nguest clock: TSC {} kHz on the host clock{}, {} rewinds ({:.3f} s of guest time), {} guest writes{}{}; TSC lead at frame ends (us): last {:.1f}, mean {:.1f}, min {:.1f}, max {:.1f} over {}",
                     clock_.tsc_khz(), tsc_scaled_ ? "" : " (NOT SCALED: the host has no TSC control)", tsc_rewinds_,
                     static_cast<double>(tsc_rewound_) / 733'333'333.0, guest_tsc_writes_,
                     tsc_writes_trapped_ ? "" : " (not trapped)",
                     tsc_offset_ctl_ ? (tsc_adjust_failures_ ? std::format(" ({} failed)", tsc_adjust_failures_) : std::string()) : std::string(" (no TSC offset control: rewinds off)"),
                     us(tsc_lead_last_), tsc_lead_samples_ ? us(static_cast<i64>(tsc_lead_sum_ / static_cast<double>(tsc_lead_samples_))) : 0.0,
                     us(tsc_lead_min_), us(tsc_lead_max_), tsc_lead_samples_);
    if (debug_lead_) {
        r += "\nguest clock: MMIO runs by CPU time (100 ns buckets):";
        for (std::size_t i = 0; i < run_hist_.size(); ++i) if (run_hist_[i]) r += std::format(" {}:{}", i, run_hist_[i]);
        r += std::format("\nguest clock: TSC lead change (cycles) inside runs {}; across exits by reason:", lead_in_);
        for (const auto& [reason, b] : lead_out_) r += std::format(" {}: x{} {} ({:.1f} a time);", reason, b.first, b.second, static_cast<double>(b.second) / static_cast<double>(b.first));
    }
    if (clock_.exact_exits())
        r += std::format("\nguest clock: exact exits (D33, MV_EXIT_COST=host for the old account): {:.3f} s of exit cost inside KVM_RUN not counted "
                         "({} ns a run, 1st percentile of MMIO runs about {} ns, {} device exits), {:.3f} s of short stretches outside not counted, device "
                         "accesses {:.3f} s of guest time",
                         sec(s.overhead_ns), exit_overhead_ns_, exit_overhead_ns_ * 5 / 4 + 100, device_exits_, sec(s.excluded_short_ns),
                         static_cast<double>(s.device_cycles) / 733'333'333.0, poll_exits_);
    return r;
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
    if (profile_timer_ok_) {
        timer_delete(profile_timer_);
        profile_timer_ok_ = false;
    }
    if (profile_eip_ && !eip_samples_.empty()) {
        std::vector<std::pair<u32, u64>> top(eip_samples_.begin(), eip_samples_.end());
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        u64 total = 0;
        for (const auto& [eip, n] : top) total += n;
        std::fprintf(stderr, "eip profile (%llu samples):\n", static_cast<unsigned long long>(total));
        // MV_PROFILE_EIP_TOP=N: list N addresses (default 20).
        static const std::size_t shown = std::getenv("MV_PROFILE_EIP_TOP") ? std::strtoull(std::getenv("MV_PROFILE_EIP_TOP"), nullptr, 10) : 20;
        for (std::size_t i = 0; i < top.size() && i < shown; ++i)
            std::fprintf(stderr, "  %08x  %6llu  %5.1f%%\n", top[i].first, static_cast<unsigned long long>(top[i].second), 100.0 * static_cast<double>(top[i].second) / static_cast<double>(total));
        eip_samples_.clear();
        std::vector<std::pair<u32, u64>> busy(eip_busy_samples_.begin(), eip_busy_samples_.end());
        std::sort(busy.begin(), busy.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::fprintf(stderr, "eip profile while the GPU thread had work (%.3f s of vCPU time in the guest):\n", static_cast<double>(busy_guest_ns_) / 1e9);
        for (std::size_t i = 0; i < busy.size() && i < 20; ++i) std::fprintf(stderr, "  %08x  %6llu\n", busy[i].first, static_cast<unsigned long long>(busy[i].second));
    }
    running_ = false;
}

void KvmBackend::kick() {
    std::atomic_ref<u64>(stats_.kicks).fetch_add(1, std::memory_order_relaxed);  // also from the GPU's FIFO thread
    if (run_) set_immediate_exit(run_, 1);
    if (thread_.joinable()) pthread_kill(thread_.native_handle(), kKickSignal);
    machine_->wake_cpu();
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
        if (kvm_ioctl(vcpu_fd_, KVM_INTERRUPT, &irq) == 0) {
            ++stats_.irq_injections;
            at_idle_ = false;
            ++stats_.by_vector[irq.irq & 0xFF];
            trace::log(trace::Cat::Irq, "injected vector {:02x}", irq.irq);
            const Cycles now = machine_->scheduler().now(), rose = machine_->intr_rose_at();
            const Cycles late = now > rose ? now - rose : 0;
            static constexpr Cycles kBounds[] = {7'333, 36'667, 73'333, 183'333, 366'667, 733'333};
            unsigned b = 0;
            while (b < 6 && late >= kBounds[b]) ++b;
            ++irq_latency_buckets_[b];
            irq_latency_max_ = std::max(irq_latency_max_, late);
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

bool KvmBackend::counted_device_exit() {
    // Guests poll by reading, and some by a whole call made of writes and a read: a title asks its sound library for a
    // stream's position over and over (a free-count read, voice select, lock, unlock, the same values every pass). At
    // the bus's price such a wait took 1.6 times the host time (an intro 95 -> 61 fps). A device access sequence that
    // repeats exactly (address, value, direction) with a period of up to 8 accesses, two periods running, is a poll;
    // a loop that writes something new each pass (updating voice after voice) is work.
    u64 key;
    if (run_->exit_reason == KVM_EXIT_MMIO) {
        if (!run_->mmio.is_write) return true;
        u64 v = 0;
        std::memcpy(&v, run_->mmio.data, std::min<u32>(run_->mmio.len, 8));
        key = (run_->mmio.phys_addr << 4) ^ (v * 0x9E3779B97F4A7C15ull) ^ run_->mmio.len;
    } else if (run_->exit_reason == KVM_EXIT_IO) {
        if (run_->io.direction == KVM_EXIT_IO_IN) return true;
        u32 v = 0;
        std::memcpy(&v, reinterpret_cast<const u8*>(run_) + run_->io.data_offset, std::min<u32>(run_->io.size, 4));
        key = (u64{run_->io.port} << 40) ^ (u64{v} * 0x9E3779B97F4A7C15ull) ^ run_->io.count;
    } else {
        return false;
    }
    return poll_step(key);  // the writes alone carry the pattern
}

bool KvmBackend::poll_step(u64 key) {
    // at(1) is the last write recorded, at(2) the one before.
    const auto at = [this](unsigned back) { return poll_ring_[(poll_pos_ + 1 - back) & (poll_ring_.size() - 1)]; };
    if (poll_period_ && at(poll_period_) == key) {
        ++poll_streak_;
    } else {
        poll_period_ = 0;
        poll_streak_ = 0;
        for (unsigned p = 1; p <= 8; ++p)
            if (at(p) == key) {
                poll_period_ = p;
                poll_streak_ = 1;
                break;
            }
    }
    poll_ring_[++poll_pos_ & (poll_ring_.size() - 1)] = key;
    const bool poll = poll_period_ && poll_streak_ >= 2 * poll_period_;
    if (poll) ++poll_exits_;
    return poll;
}

void KvmBackend::device_exit(Cycles cost, bool calibrates) {
    // The exit's own cost inside KVM_RUN is no guest time with exact exits. MMIO runs (a string port access can come
    // back without entering the guest) have a hard floor, the exit itself: on this host 5.0 us, with 1.7 M of 1.7 M
    // runs between 5 and 6 us and a hundred stray ones below (most likely back without the guest running). Four fifths of the
    // 1st percentile comes off every run, so a rewind stays out of the time the guest ran; a host without the IBPB on
    // exits measures its own, smaller. MV_EXIT_OVERHEAD_NS fixes it; MV_DEVICE_COST=0 drops the device costs.
    static const long force_ns = [] { const char* e = std::getenv("MV_EXIT_OVERHEAD_NS"); return e && *e ? std::strtol(e, nullptr, 10) : -1L; }();
    static const bool no_cost = [] { const char* e = std::getenv("MV_DEVICE_COST"); return e && *e == '0'; }();
    if (!no_cost) device_pending_ += cost;
    ++device_exits_;
    if (!calibrates || !clock_.exact_exits()) return;
    ++run_hist_[std::min<u64>(last_run_cpu_ / 100, run_hist_.size() - 1)];
    if (++calibration_runs_ % 4096) return;
    if (force_ns >= 0) {
        exit_overhead_ns_ = static_cast<u64>(force_ns);
        return;
    }
    u64 seen = 0;
    std::size_t b = 0;
    while (b + 1 < run_hist_.size() && (seen += run_hist_[b]) * 100 < calibration_runs_) ++b;
    exit_overhead_ns_ = b * 100 * 4 / 5;
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

void KvmBackend::drain_coalesced_writes() {
    if (!coalesced_) return;
    Bus& bus = machine_->bus();
    coalesced_writes_ += kvm::drain_coalesced(coalesced_, coalesced_slots_, [&](u64 phys, u32 len, const u8* data) {
        if (len == 8) {
            u64 v;
            std::memcpy(&v, data, 8);
            bus.write64(static_cast<PhysAddr>(phys), v);
        } else {
            u32 v = 0;
            std::memcpy(&v, data, std::min<u32>(len, 4));
            bus.write(static_cast<PhysAddr>(phys), len, v);
        }
    });
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
            if (const auto& b = machine_->config().break_mmio_write; b && b->address == addr && (!b->value || *b->value == v) && mmio_breaks_ < b->count) {
                ++mmio_breaks_;
                Regs r{};
                if (read_regs(r))
                    trace::log(trace::Cat::Boot, "break: mmio write {:08x} <- {:08x} (size {})\n{}", addr, v, len, debug::describe_stack(bus, r));
            }
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
    pthread_setname_np(pthread_self(), "mv-vcpu");  // shows in top -H and profilers
    sigset_t unblock;
    sigemptyset(&unblock);
    sigaddset(&unblock, kKickSignal);
    if (profile_us_) sigaddset(&unblock, kProfileSignal);
    pthread_sigmask(SIG_UNBLOCK, &unblock, nullptr);
    if (profile_eip_ && profile_us_) {
        sigevent sev{};
        sev.sigev_notify = SIGEV_THREAD_ID;
        sev.sigev_signo = kProfileSignal;
        sev.MV_SIGEV_THREAD_ID = static_cast<pid_t>(syscall(SYS_gettid));  // gettid() itself is glibc 2.30 on
        if (timer_create(CLOCK_THREAD_CPUTIME_ID, &sev, &profile_timer_) == 0) {
            profile_timer_ok_ = true;
            itimerspec its{};
            its.it_interval.tv_nsec = its.it_value.tv_nsec = static_cast<long>(profile_us_) * 1000;
            timer_settime(profile_timer_, 0, &its, nullptr);
        }
    }
    if (guest_clock_on_) {
        // The io thread reads this thread's CPU clock to see guest time move while the guest runs.
        clockid_t id{};
        if (pthread_getcpuclockid(pthread_self(), &id) == 0) {
            vcpu_cpu_clock_.store(id);
            vcpu_clock_known_.store(true, std::memory_order_release);
        }
        // Guest time counts this thread's CPU time from here (the stretch since the reset is not guest time).
        clock_.thread_started(thread_cpu_ns());
        // The backstop timer signals this thread (the kick).
        sigevent sev{};
        sev.sigev_notify = SIGEV_THREAD_ID;
        sev.sigev_signo = kKickSignal;
        sev.MV_SIGEV_THREAD_ID = static_cast<pid_t>(syscall(SYS_gettid));  // gettid() itself is glibc 2.30 on
        std::lock_guard lock(machine_->lock());
        backstop_ok_ = timer_create(CLOCK_MONOTONIC, &sev, &backstop_) == 0;
        backstop_at_ns_ = 0;
    }

    while (!stop_requested_) {
        // From here a kick makes the coming KVM_RUN return at once (the check below runs after it, under the lock).
        set_immediate_exit(run_, 0);
        {
            std::unique_lock lock(machine_->lock());
            const u64 t0 = mono_ns();
            machine_->advance_time();
            const u64 t1 = mono_ns();
            event_ns_ += t1 - t0;
            if (guest_clock_on_) {
                if (!machine_->pace_cpu(lock, clock_.now(), stop_requested_)) {
                    pause_ns_ += mono_ns() - t1;
                    continue;
                }
                Scheduler& sched = machine_->scheduler();
                if (halted_) {
                    // HLT: no guest time passes in the guest; it skips to each next event until one raises an
                    // interrupt (the TSC moves as far), with the pacing above between the steps.
                    if (machine_->pic().pending_vector() >= 0) {
                        halted_ = false;
                    } else if (sched.has_pending()) {
                        // The GPU's FIFO thread finishes first: an idle CPU sees the GPU finish at once, as when the
                        // pusher ran inline, instead of skipping guest time past work still in flight.
                        if (machine_->nv2a().fifo_threaded()) {
                            machine_->nv2a().wait_fifo_idle();
                            // What it raised meanwhile reaches the PIC before any time is skipped.
                            machine_->advance_time();
                            if (machine_->pic().pending_vector() >= 0) {
                                halted_ = false;
                                continue;
                            }
                        }
                        if (const Cycles skipped = clock_.warp_to(sched.next_deadline())) tsc_offset_add(static_cast<i64>(skipped));
                        continue;
                    } else {
                        wait_for_interrupt();
                        continue;
                    }
                }
                if (const u64 v = machine_->nv2a().vblanks(); v != vblank_seen_) {
                    vblank_seen_ = v;
                    frame_tsc_due_ = true;
                    if (idle_skip_ && !idle_eip_ && idle_scans_ < 1200 && break_eips_.empty() && !watch_va_) find_idle_loop();
                }
                arm_backstop(false);
            }
            inject_interrupt_if_possible();
        }
        if (guest_clock_on_) {
            // The stretch since the exit: a short one is guest time and a long one is not and comes off the TSC; with exact
            // exits (D33) only a device read's counts. At every boundary the wall clock is read first and the CPU clock
            // second, so the two accounts line up.
            const GuestClock::Exit e = clock_.classify(mono_ns(), device_pending_, read_exit_);
            device_pending_ = 0;
            if (e.excluded) {
                const u64 rewind = e.tsc_rewind + clock_.excluded_until(thread_cpu_ns());
                if (rewind && tsc_offset_add(-static_cast<i64>(rewind))) {
                    ++tsc_rewinds_;
                    tsc_rewound_ += rewind;
                }
            }
            {
                const u64 wall = mono_ns();
                const u64 cpu = thread_cpu_ns();
                clock_.entered(cpu, wall);
                run_entered_cpu_ = cpu;
                if (debug_lead_) {
                    const i64 l = static_cast<i64>(read_tsc()) - static_cast<i64>(clock_.now() - tsc_epoch_);
                    if (lead_left_set_) {
                        auto& b = lead_out_[lead_reason_];
                        ++b.first;
                        b.second += l - lead_left_;
                    }
                    lead_entered_ = l;
                }
            }
            // The io thread may have backed off while this clock stood still: it re-plans on the running clock.
            if (e.long_exit) machine_->wake_io();
            if (frame_tsc_due_) {
                frame_tsc_due_ = false;
                note_frame_tsc();
            }
        }
        if (watch_pending_.load(std::memory_order_acquire)) apply_pending_watches();  // out of the guest: clears are cheap
        const bool gpu_busy = machine_->nv2a().fifo_busy_now();
        if (break_from_ && !breakpoints_armed_ && !break_eips_.empty() && machine_->nv2a().vblanks() >= break_from_) {
            vblank_seen_ = machine_->nv2a().vblanks();
            arm_breakpoints();
        }
        const bool sampling = profile_eip_ && vblank_seen_ >= profile_from_ && vblank_seen_ < profile_to_;
        const bool idle_run = at_idle_;
        const u64 guest_before = clock_.stats().in_guest_ns;
        const int rc = kvm_ioctl(vcpu_fd_, KVM_RUN, 0);
        if (guest_clock_on_) {
            const u64 wall = mono_ns();
            const u64 cpu = thread_cpu_ns();
            // A device read or a poll keeps the host's price (see GuestClock::classify): nothing comes off its run either.
            read_exit_ = rc == 0 && counted_device_exit();
            clock_.left(cpu, wall, read_exit_ ? 0 : exit_overhead_ns_);
            last_run_cpu_ = cpu - run_entered_cpu_;
            if (debug_lead_) {
                lead_left_ = static_cast<i64>(read_tsc()) - static_cast<i64>(clock_.now() - tsc_epoch_);
                lead_left_set_ = true;
                lead_in_ += lead_left_ - lead_entered_;
                lead_reason_ = rc < 0 ? 255u : (run_->exit_reason < 64 ? run_->exit_reason : 254u);
            }
            if (gpu_busy) busy_guest_ns_ += clock_.stats().in_guest_ns - guest_before;
            if (idle_run) idle_spin_ns_ += clock_.stats().in_guest_ns - guest_before;
        }
        ++stats_.exits;
        if (rc < 0 && errno == EINTR) {
            ++stats_.intr_exits;
            if (sampling && (!profile_us_ || g_profile_tick.exchange(false, std::memory_order_relaxed))) {
                kvm_regs r{};
                if (kvm_ioctl(vcpu_fd_, KVM_GET_REGS, &r) == 0) {
                    ++eip_samples_[static_cast<u32>(r.rip)];
                    if (gpu_busy) ++eip_busy_samples_[static_cast<u32>(r.rip)];
                }
            }
            continue;  // kicked: loop to inject or stop
        }
        if (rc < 0) {
            fail(Stop::InternalError, std::format("KVM_RUN failed: {}", std::strerror(errno)));
            break;
        }
        std::unique_lock lock(machine_->lock());
        const u64 h0 = mono_ns();
        machine_->advance_time();
        const u64 h1 = mono_ns();
        event_ns_ += h1 - h0;
        struct HandlerTime {
            u64& acc;
            u64 t0;
            ~HandlerTime() { acc += mono_ns() - t0; }
        } handler_time{handler_ns_, h1};
        // Render targets written back in the background land here, on this thread (it alone can name the pages the guest
        // wrote meanwhile), at most every 200 us while some are in flight.
        if (h1 >= next_write_back_poll_ns_ && machine_->nv2a().write_backs_in_flight()) {
            next_write_back_poll_ns_ = h1 + 200'000;
            machine_->nv2a().poll_write_backs();
        }
        if (run_->exit_reason < 64) ++stats_.by_reason[run_->exit_reason];
        if (run_->exit_reason == KVM_EXIT_IO) ++exit_sources_[(u32{run_->io.port} << 16) | 1u];  // a port: bit 0 set (an MMIO key is a page)
        else if (run_->exit_reason == KVM_EXIT_MMIO) ++exit_sources_[static_cast<u32>(run_->mmio.phys_addr) & ~0xFFFu];
        switch (run_->exit_reason) {
            case KVM_EXIT_IO:
                handle_io();
                device_exit(run_->io.direction == KVM_EXIT_IO_OUT ? kPortIoCycles * run_->io.count : 0, false);
                break;
            case KVM_EXIT_MMIO:
                handle_mmio();
                device_exit(run_->mmio.is_write ? kDeviceStoreCycles : 0, true);
                break;
            case KVM_EXIT_HLT:
                ++stats_.hlt_exits;
                // With the guest clock, an idle CPU skips guest time (the loop above); with interrupts off it has
                // stopped for good, and only a device or a stop wakes it, as with the host clock.
                if (guest_clock_on_ && run_->if_flag) halted_ = true;
                else {
                    // Interrupts off: the CPU stops until a reset (the kernel ends a bug check this way). The first
                    // such stop is kept for the unload report, with the stack it stopped on.
                    if (stopped_at_.empty()) {
                        Regs r{};
                        if (read_regs(r)) stopped_at_ = std::format("HLT with interrupts off at eip {:08x}\n{}", r.eip, debug::describe_stack(machine_->bus(), r, 32));
                        else stopped_at_ = "HLT with interrupts off (registers unreadable)";
                        trace::log(trace::Cat::Cpu, "the guest stopped: {}", stopped_at_);
                    }
                    wait_for_interrupt();
                }
                break;
            case KVM_EXIT_IRQ_WINDOW_OPEN: ++stats_.irq_window_exits; break;
            case KVM_EXIT_X86_WRMSR:
                // The guest set its TSC (the only MSR filtered): do it, and count TSC time from here.
                if (run_->msr.index == 0x10) {
                    set_tsc(run_->msr.data);
                    tsc_epoch_ = clock_.now() - std::min<Cycles>(clock_.now(), run_->msr.data);
                    ++guest_tsc_writes_;
                    trace::log(trace::Cat::Cpu, "guest wrote the TSC: {:#x}", static_cast<u64>(run_->msr.data));
                }
                run_->msr.error = 0;
                break;
            case KVM_EXIT_INTR: break;
            case KVM_EXIT_DEBUG: {
                if (idle_eip_ && (run_->debug.arch.pc == idle_eips_[0] || run_->debug.arch.pc == idle_eips_[1]) && break_eips_.empty() && !watch_va_) {
                    // The idle loop. A turn of it with no interrupt injected since the last hit found no work: halt.
                    ++idle_hits_;
                    at_idle_ = true;
                    const bool quiet = stats_.irq_injections == idle_injections_at_hit_;
                    idle_injections_at_hit_ = stats_.irq_injections;
                    if (quiet && run_->if_flag && guest_clock_on_) {
                        halted_ = true;
                        ++idle_halts_;
                    }
                    // Step over the breakpoint: the resume flag suppresses it for one instruction.
                    kvm_regs regs{};
                    if (kvm_ioctl(vcpu_fd_, KVM_GET_REGS, &regs) == 0) {
                        regs.rflags |= 1u << 16;
                        kvm_ioctl(vcpu_fd_, KVM_SET_REGS, &regs);
                    }
                    break;
                }
                // A --break-eip address: log where the guest is, then drop that breakpoint and keep the others.
                Regs r{};
                lock.unlock();
                read_regs(r);
                if (watch_va_ && (run_->debug.arch.dr6 & (1u << 3))) {
                    // The write watch: the instruction has written; log the writer and the word as it is now.
                    u32 now_value = 0;
                    if (const auto pa = debug::virt_to_phys(machine_->bus(), r, watch_va_)) machine_->bus().read_ram(*pa, &now_value, 4);
                    trace::log(trace::Cat::Boot, "watch: write to {:08x} (now {:08x}), eip after it {:08x}\n{}", watch_va_, now_value, r.eip, debug::describe_stack(machine_->bus(), r, 32));
                    if (++watch_hits_ >= watch_max_) {
                        watch_va_ = 0;
                        kvm_guest_debug off{};
                        kvm_ioctl(vcpu_fd_, KVM_SET_GUEST_DEBUG, &off);
                        breakpoints_armed_ = false;
                        arm_breakpoints();
                    }
                    continue;
                }
                // MV_BREAK_KEEP=N: the breakpoints stay armed for N hits, each a short line: the caller and, at a file
                // call's entry, the name its OBJECT_ATTRIBUTES carries (the first or third stdcall argument).
                if (static const u32 keep = std::getenv("MV_BREAK_KEEP") ? static_cast<u32>(std::strtoul(std::getenv("MV_BREAK_KEEP"), nullptr, 10)) : 0u;
                    keep && break_hits_ < keep) {
                    ++break_hits_;
                    const auto rd32 = [&](u32 va) -> u32 {
                        u32 v = 0;
                        if (const auto pa = debug::virt_to_phys(machine_->bus(), r, va)) machine_->bus().read_ram(*pa, &v, 4);
                        return v;
                    };
                    const auto name_of = [&](u32 attributes) -> std::string {
                        const u32 string = attributes ? rd32(attributes + 4) : 0;
                        if (!string) return {};
                        const u32 len = rd32(string) & 0xFFFFu, buf = rd32(string + 4);
                        std::string s;
                        for (u32 i = 0; i < len && i < 160; ++i) {
                            const u32 ch = rd32(buf + i) & 0xFFu;
                            s += (ch >= 0x20 && ch < 0x7F) ? static_cast<char>(ch) : '.';
                        }
                        return s;
                    };
                    std::string up;  // the code addresses further up the stack: the callers
                    for (u32 i = 4, found = 0; i < 64 && found < 6; ++i)
                        if (const u32 w = rd32(r.esp + i * 4); w >= 0x10000 && w < 0x400000) {
                            up += std::format(" {:08x}", w);
                            ++found;
                        }
                    std::string text;  // MV_BREAK_STR=[*]REG+HEX: the text there too (with *, at the pointer stored there)
                    if (const char* bs = std::getenv("MV_BREAK_STR"); bs && *bs) {
                        const bool deref = *bs == '*';
                        const std::string spec = bs + (deref ? 1 : 0);
                        const std::string g = spec.substr(0, spec.find('+'));
                        const u32 off = spec.find('+') != std::string::npos ? static_cast<u32>(std::strtoul(spec.c_str() + spec.find('+') + 1, nullptr, 16)) : 0u;
                        const u32 base = g == "eax" ? r.eax : g == "ebx" ? r.ebx : g == "ecx" ? r.ecx : g == "edx" ? r.edx : g == "esi" ? r.esi : g == "edi" ? r.edi : g == "esp" ? r.esp : r.ebp;
                        const u32 at = deref ? rd32(base + off) : base + off;
                        for (u32 i = 0; i < 160; ++i) {
                            const u32 c = rd32(at + i) & 0xFFu;
                            if (!c) break;
                            text += static_cast<char>(c);
                        }
                    }
                    trace::log(trace::Cat::Boot, "break: eip {:08x} from {:08x} eax {:08x} args {:08x} {:08x} {:08x} names [{}] [{}] up{} esp {:08x} text [{}]", r.eip, rd32(r.esp),
                               r.eax, rd32(r.esp + 4), rd32(r.esp + 8), rd32(r.esp + 12), name_of(rd32(r.esp + 4)), name_of(rd32(r.esp + 12)), up, r.esp, text);
                    kvm_regs regs{};
                    if (kvm_ioctl(vcpu_fd_, KVM_GET_REGS, &regs) == 0) {
                        regs.rflags |= 1u << 16;
                        kvm_ioctl(vcpu_fd_, KVM_SET_REGS, &regs);
                    }
                    continue;
                }
                // MV_BREAK_STR=REG+HEX: also the text at that address (a name the code is working on).
                if (const char* bs = std::getenv("MV_BREAK_STR"); bs && *bs) {
                    const std::string spec = bs;
                    const std::string g = spec.substr(0, spec.find('+'));
                    const u32 off = spec.find('+') != std::string::npos ? static_cast<u32>(std::strtoul(spec.c_str() + spec.find('+') + 1, nullptr, 16)) : 0u;
                    const u32 base = g == "eax" ? r.eax : g == "ebx" ? r.ebx : g == "ecx" ? r.ecx : g == "edx" ? r.edx : g == "esi" ? r.esi : g == "edi" ? r.edi : g == "esp" ? r.esp : r.ebp;
                    std::string text;
                    for (u32 i = 0; i < 160; ++i) {
                        u8 c = 0;
                        if (const auto pa = debug::virt_to_phys(machine_->bus(), r, base + off + i)) machine_->bus().read_ram(*pa, &c, 1);
                        if (!c) break;
                        text += static_cast<char>(c);
                    }
                    trace::log(trace::Cat::Boot, "break: text at {}: [{}]", spec, text);
                }
                // MV_BREAK_EBPWALK=1: the callers by the frame-pointer chain (code built with frame pointers).
                if (const char* w = std::getenv("MV_BREAK_EBPWALK"); w && *w == '1') {
                    std::string chain;
                    u32 bp = r.ebp;
                    for (int i = 0; i < 24 && bp; ++i) {
                        u32 frame[2]{};
                        const auto pa = debug::virt_to_phys(machine_->bus(), r, bp);
                        if (!pa || !machine_->bus().read_ram(*pa, frame, 8)) break;
                        chain += std::format(" {:08x}", frame[1]);
                        if (frame[0] <= bp) break;
                        bp = frame[0];
                    }
                    trace::log(trace::Cat::Boot, "break: callers by ebp:{}", chain);
                }
                trace::log(trace::Cat::Boot, "break: eip {:08x} reached\n{}", r.eip, debug::describe_stack(machine_->bus(), r));
                if (!watch_reg_.empty() && !watch_va_ && !watch_hits_) {
                    const std::string& g = watch_reg_;
                    const u32 base = g == "eax" ? r.eax : g == "ebx" ? r.ebx : g == "ecx" ? r.ecx : g == "edx" ? r.edx : g == "esi" ? r.esi : g == "edi" ? r.edi : g == "esp" ? r.esp : r.ebp;
                    watch_va_ = base + watch_offset_;
                    trace::log(trace::Cat::Boot, "watch: writes to {:08x} ({}+{:x}) from here on, up to {}", watch_va_, g, watch_offset_, watch_max_);
                }
                std::erase(break_eips_, r.eip);
                kvm_guest_debug off{};
                kvm_ioctl(vcpu_fd_, KVM_SET_GUEST_DEBUG, &off);
                breakpoints_armed_ = false;
                arm_breakpoints();
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
    vcpu_clock_known_.store(false, std::memory_order_release);
    {
        std::lock_guard lock(machine_->lock());
        if (backstop_ok_) timer_delete(backstop_);
        backstop_ok_ = false;
    }
    running_ = false;
    machine_->wake();
}

// Hardware execution breakpoints (DR0-DR3) on the --break-eip addresses; the debug exit logs and disarms them.
void KvmBackend::arm_breakpoints() {
    const bool idle = idle_eip_ && break_eips_.empty() && !watch_va_;
    if (break_eips_.empty() && !watch_va_ && !idle) return;
    if (vblank_seen_ < break_from_ && !idle) return;
    kvm_guest_debug dbg{};
    dbg.control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW_BP;
    dbg.arch.debugreg[7] = 1u << 10;  // the bit that is always set in DR7
    for (std::size_t i = 0; i < break_eips_.size() && i < 4; ++i) {
        dbg.arch.debugreg[i] = break_eips_[i];
        dbg.arch.debugreg[7] |= 1u << (i * 2);  // Li enabled; R/Wi 00 = execution, LENi 00 = one byte
    }
    if (watch_va_) {
        dbg.arch.debugreg[3] = watch_va_;
        dbg.arch.debugreg[7] |= (1u << 6) | (1u << 28) | (3u << 30);  // L3; R/W3 01 = data writes; LEN3 11 = 4 bytes
    }
    if (idle) {
        for (std::size_t i = 0; i < idle_eips_.size(); ++i)
            if (idle_eips_[i]) {
                dbg.arch.debugreg[i] = idle_eips_[i];
                dbg.arch.debugreg[7] |= 1u << (i * 2);  // Li, execution, one byte
            }
    }
    if (kvm_ioctl(vcpu_fd_, KVM_SET_GUEST_DEBUG, &dbg) < 0) {
        trace::log(trace::Cat::Boot, "break: KVM_SET_GUEST_DEBUG failed: {}", std::strerror(errno));
        return;
    }
    breakpoints_armed_ = true;
}

void KvmBackend::take_dirty_pages(std::vector<u32>& pages) {
    if (!dirty_tracking_ || vm_fd_ < 0) return;
    // Selective tracking reads the log on the vCPU thread only (between exits); another thread's harvest waits for
    // the next one there (the log keeps its bits).
    if (selective_ && std::this_thread::get_id() != thread_.get_id()) return;
    if (selective_) {
        if (watch_pending_.load(std::memory_order_acquire)) apply_pending_watches();
        for (u32 p : watch_reported_) {
            pages.push_back(p);
            if (p / 64 < watched_.size()) watched_[p / 64] &= ~(u64{1} << (p % 64));
        }
        watch_reported_.clear();
    }
    // Both RAM slots (direct and the GPU-side alias) back the same pages; the union of their logs is what changed.
    std::vector<u64> merged(dirty_bitmap_.size(), 0);
    for (u32 slot : {kSlotRam, kSlotRamAlias}) {
        std::fill(dirty_bitmap_.begin(), dirty_bitmap_.end(), 0);
        kvm_dirty_log log{};
        log.slot = slot;
        log.dirty_bitmap = dirty_bitmap_.data();
        if (kvm_ioctl(vm_fd_, KVM_GET_DIRTY_LOG, &log) < 0) continue;
        for (std::size_t i = 0; i < merged.size(); ++i) merged[i] |= dirty_bitmap_[i];
    }
    if (selective_)
        for (std::size_t w = 0; w < merged.size(); ++w) {
            merged[w] &= watched_[w];
            watched_[w] &= ~merged[w];  // reported once; named again when the texture is decoded or checked
        }
    // Exact reports: the reported pages' bits are cleared (each write-protected again, one fault on its next write), so a
    // later watch of them reports only a write made after this report.
    if (selective_ && exact_reports_.load(std::memory_order_relaxed) && std::any_of(merged.begin(), merged.end(), [](u64 w) { return w != 0; }))
        for (u32 slot : {kSlotRam, kSlotRamAlias}) {
            kvm_clear_dirty_log clear{};
            clear.slot = slot;
            clear.first_page = 0;
            clear.num_pages = static_cast<u32>(merged.size() * 64);
            clear.dirty_bitmap = merged.data();
            kvm_ioctl(vm_fd_, KVM_CLEAR_DIRTY_LOG, &clear);
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

void KvmBackend::watch_pages(u32 first, u32 count) {
    if (!selective_ || count == 0 || vm_fd_ < 0) return;
    if (std::this_thread::get_id() != thread_.get_id()) {
        std::lock_guard g(watch_mutex_);
        watch_queue_.emplace_back(first, count);
        watch_pending_.store(true, std::memory_order_release);
        return;
    }
    apply_watch(first, count, false);
}

void KvmBackend::apply_pending_watches() {
    std::vector<std::pair<u32, u32>> queue;
    {
        std::lock_guard g(watch_mutex_);
        queue.swap(watch_queue_);
        watch_pending_.store(false, std::memory_order_relaxed);
    }
    for (const auto& [first, count] : queue) apply_watch(first, count, true);
}

void KvmBackend::apply_watch(u32 first, u32 count, bool conservative) {
    const u64 pages = static_cast<u64>(watched_.size()) * 64;
    if (first >= pages) return;
    count = static_cast<u32>(std::min<u64>(count, pages - first));
    // One clear per slot over the 64-page blocks the range touches, with bits for its own pages only: KVM
    // write-protects those pages again, and their next write sets their bits.
    const u32 block0 = first / 64, block1 = (first + count - 1) / 64;
    if (conservative) {
        // Pages the log already shows written are reported: the write may have come after the texture read them.
        for (u32 slot : {kSlotRam, kSlotRamAlias}) {
            std::fill(dirty_bitmap_.begin(), dirty_bitmap_.end(), 0);
            kvm_dirty_log log{};
            log.slot = slot;
            log.dirty_bitmap = dirty_bitmap_.data();
            if (kvm_ioctl(vm_fd_, KVM_GET_DIRTY_LOG, &log) < 0) continue;
            for (u32 p = first; p < first + count; ++p)
                if ((dirty_bitmap_[p / 64] >> (p % 64)) & 1u) watch_reported_.push_back(p);
        }
    }
    std::vector<u64> bits(block1 - block0 + 1, 0);
    for (u32 p = first; p < first + count; ++p) {
        const u64 bit = u64{1} << (p % 64);
        bits[p / 64 - block0] |= bit;
        watched_[p / 64] |= bit;
    }
    const u64 slot_pages = pages;
    for (u32 slot : {kSlotRam, kSlotRamAlias}) {
        kvm_clear_dirty_log clear{};
        clear.slot = slot;
        clear.first_page = static_cast<u64>(block0) * 64;
        clear.num_pages = static_cast<u32>(std::min<u64>(static_cast<u64>(bits.size()) * 64, slot_pages - clear.first_page));
        clear.dirty_bitmap = bits.data();
        kvm_ioctl(vm_fd_, KVM_CLEAR_DIRTY_LOG, &clear);
    }
}

bool KvmBackend::read_regs(Regs& out) {
    kvm_regs regs{};
    kvm_sregs sregs{};
    if (kvm_ioctl(vcpu_fd_, KVM_GET_REGS, &regs) < 0) return false;
    if (kvm_ioctl(vcpu_fd_, KVM_GET_SREGS, &sregs) < 0) return false;
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
    out.tsc = kvm_ioctl(vcpu_fd_, KVM_GET_MSRS, msrs.header()) == 1 ? msrs.entry()->data : 0;
    return true;
}

std::unique_ptr<CpuBackend> make_fast_backend(std::string& why_not) {
    if (access("/dev/kvm", R_OK | W_OK) != 0) {
        // Said so a player knows what to do: the frontend shows it on screen.
        if (errno == ENOENT)
            why_not = "hardware virtualization is off or missing (no /dev/kvm). Switch on Intel VT-x or AMD-V (SVM) in your PC's firmware setup, then restart.";
        else if (errno == EACCES || errno == EPERM)
            why_not = "no permission to use /dev/kvm. Add yourself to the kvm group (sudo usermod -aG kvm $USER), then log out and back in.";
        else
            why_not = std::format("/dev/kvm is not accessible: {}", std::strerror(errno));
        return nullptr;
    }
    return std::make_unique<KvmBackend>();
}


// The loops the kernel spins in when it has nothing to run: STI, two NOPs, CLI, then a compare of the DPC list head
// with itself (the shape of the NT-family idle loop, found by profiling where an idle console spends its time). The
// kernel image is identity-mapped at 0x80000000; it holds two (the idle thread's, which also looks for a ready thread,
// and one that only waits for DPCs). Each found arms a breakpoint on its second NOP, where interrupts are enabled and
// the loop has not yet looked for work; either leaves only through work an interrupt queued.
void KvmBackend::find_idle_loop() {
    ++idle_scans_;
    static constexpr u8 kShape[] = {0xFB, 0x90, 0x90, 0xFA, 0x3B, 0x6D, 0x00, 0x74};
    constexpr u32 kFrom = 0x10000, kTo = 0x80000;
    const Bus& bus = machine_->bus();
    if (bus.ram_size() < kTo) return;
    const u8* ram = bus.ram();
    std::array<u32, 2> found{};
    u32 matches = 0;
    for (u32 a = kFrom; a + sizeof kShape <= kTo; ++a) {
        if (ram[a] != 0xFB || std::memcmp(ram + a, kShape, sizeof kShape) != 0) continue;
        if (matches < found.size()) found[matches] = 0x80000000u + a + 2;
        ++matches;
    }
    if (matches == 0 || matches > found.size()) return;
    idle_eips_ = found;
    idle_eip_ = found[0];
    trace::log(trace::Cat::Cpu, "the kernel idle loops are at {:08x} {:08x}: their idle turns halt", found[0], found[1]);
    kvm_guest_debug off{};
    kvm_ioctl(vcpu_fd_, KVM_SET_GUEST_DEBUG, &off);
    breakpoints_armed_ = false;
    arm_breakpoints();
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
