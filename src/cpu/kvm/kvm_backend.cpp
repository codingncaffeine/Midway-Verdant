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
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>

#include "core/clock.h"

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
    if (ioctl(kvm_fd_, KVM_CHECK_EXTENSION, KVM_CAP_TSC_CONTROL) > 0) {
        if (ioctl(vcpu_fd_, KVM_SET_TSC_KHZ, guest_clock_on_ ? clock_.tsc_khz() : 733333u) == 0) tsc_scaled_ = true;
    }
    if (!tsc_scaled_) trace::log(trace::Cat::Cpu, "TSC scaling unavailable: the guest TSC runs at host rate");
    {
        kvm_device_attr a{};
        a.group = KVM_VCPU_TSC_CTRL;
        a.attr = KVM_VCPU_TSC_OFFSET;
        tsc_offset_ctl_ = ioctl(vcpu_fd_, KVM_HAS_DEVICE_ATTR, &a) == 0;
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
        tsc_writes_trapped_ = ioctl(vm_fd_, KVM_ENABLE_CAP, &cap) == 0 && ioctl(vm_fd_, KVM_X86_SET_MSR_FILTER, &filter) == 0;
    }

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
    return ioctl(vcpu_fd_, KVM_GET_MSRS, msrs.header()) == 1 ? msrs.entry()->data : 0;
}

bool KvmBackend::tsc_offset_add(i64 delta) {
    if (!tsc_offset_ctl_) return false;
    u64 offset = 0;
    kvm_device_attr a{};
    a.group = KVM_VCPU_TSC_CTRL;
    a.attr = KVM_VCPU_TSC_OFFSET;
    a.addr = reinterpret_cast<u64>(&offset);
    if (ioctl(vcpu_fd_, KVM_GET_DEVICE_ATTR, &a) < 0) {
        ++tsc_adjust_failures_;
        return false;
    }
    offset += static_cast<u64>(delta);
    if (ioctl(vcpu_fd_, KVM_SET_DEVICE_ATTR, &a) < 0) {
        ++tsc_adjust_failures_;
        return false;
    }
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
    ioctl(vcpu_fd_, KVM_SET_MSRS, msrs.header());
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
        "{:.3f} s in {} long exits (longest {:.1f} ms), {:.3f} s preempted in the guest, {} cycles skipped at HLT",
        scale, static_cast<double>(clock_.now()) / 733'333'333.0, sec(s.in_guest_ns), sec(s.out_counted_ns), s.short_exits, scale, sec(s.long_ns),
        s.long_exits, static_cast<double>(s.longest_exit_ns) / 1e6, sec(s.preempted_ns), s.warped_cycles);
    r += std::format("\nkvm: {} exits (mmio {}, pio {}, hlt {}, irq window {}, interrupted {}), {} kicks, {} interrupts injected",
                     stats_.exits, stats_.mmio_exits, stats_.pio_exits, stats_.hlt_exits, stats_.irq_window_exits, stats_.intr_exits, stats_.kicks,
                     stats_.irq_injections);
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
            r += (top[i].first & 0x10000u) ? std::format(" port {:04x} x{}", top[i].first & 0xFFFFu, top[i].second)
                                           : std::format(" mmio {:08x} x{}", top[i].first, top[i].second);
    }
    if (!guest_clock_on_) return r;
    r += std::format("\nguest clock: TSC {} kHz on the host clock{}, {} rewinds ({:.3f} s of guest time), {} guest writes{}{}; TSC lead at frame ends (us): last {:.1f}, mean {:.1f}, min {:.1f}, max {:.1f} over {}",
                     clock_.tsc_khz(), tsc_scaled_ ? "" : " (NOT SCALED: the host has no TSC control)", tsc_rewinds_,
                     static_cast<double>(tsc_rewound_) / 733'333'333.0, guest_tsc_writes_,
                     tsc_writes_trapped_ ? "" : " (not trapped)",
                     tsc_offset_ctl_ ? (tsc_adjust_failures_ ? std::format(" ({} failed)", tsc_adjust_failures_) : std::string()) : std::string(" (no TSC offset control: rewinds off)"),
                     us(tsc_lead_last_), tsc_lead_samples_ ? us(static_cast<i64>(tsc_lead_sum_ / static_cast<double>(tsc_lead_samples_))) : 0.0,
                     us(tsc_lead_min_), us(tsc_lead_max_), tsc_lead_samples_);
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
        if (ioctl(vcpu_fd_, KVM_INTERRUPT, &irq) == 0) {
            ++stats_.irq_injections;
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
    pthread_sigmask(SIG_UNBLOCK, &unblock, nullptr);
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
        sev._sigev_un._tid = gettid();
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
                }
                arm_backstop(false);
            }
            inject_interrupt_if_possible();
        }
        if (guest_clock_on_) {
            // The stretch since the exit: a short one is guest time, a long one is not and comes off the TSC. At every
            // boundary the wall clock is read first and the CPU clock second, so the two accounts line up.
            const GuestClock::Exit e = clock_.classify(mono_ns());
            if (e.long_exit) {
                clock_.excluded_until(thread_cpu_ns());
                if (e.tsc_rewind && tsc_offset_add(-static_cast<i64>(e.tsc_rewind))) {
                    ++tsc_rewinds_;
                    tsc_rewound_ += e.tsc_rewind;
                }
            }
            {
                const u64 wall = mono_ns();
                const u64 cpu = thread_cpu_ns();
                clock_.entered(cpu, wall);
            }
            // The io thread may have backed off while this clock stood still: it re-plans on the running clock.
            if (e.long_exit) machine_->wake_io();
            if (frame_tsc_due_) {
                frame_tsc_due_ = false;
                note_frame_tsc();
            }
        }
        const int rc = ioctl(vcpu_fd_, KVM_RUN, 0);
        if (guest_clock_on_) {
            const u64 wall = mono_ns();
            const u64 cpu = thread_cpu_ns();
            clock_.left(cpu, wall);
        }
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
        const u64 h0 = mono_ns();
        machine_->advance_time();
        const u64 h1 = mono_ns();
        event_ns_ += h1 - h0;
        struct HandlerTime {
            u64& acc;
            u64 t0;
            ~HandlerTime() { acc += mono_ns() - t0; }
        } handler_time{handler_ns_, h1};
        if (run_->exit_reason < 64) ++stats_.by_reason[run_->exit_reason];
        if (run_->exit_reason == KVM_EXIT_IO) ++exit_sources_[0x10000u | run_->io.port];
        else if (run_->exit_reason == KVM_EXIT_MMIO) ++exit_sources_[static_cast<u32>(run_->mmio.phys_addr) & ~0xFFFu];
        switch (run_->exit_reason) {
            case KVM_EXIT_IO: handle_io(); break;
            case KVM_EXIT_MMIO: handle_mmio(); break;
            case KVM_EXIT_HLT:
                ++stats_.hlt_exits;
                // With the guest clock, an idle CPU skips guest time (the loop above); with interrupts off it has
                // stopped for good, and only a device or a stop wakes it, as with the host clock.
                if (guest_clock_on_ && run_->if_flag) halted_ = true;
                else wait_for_interrupt();
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
                // A --break-eip address: log where the guest is, then drop that breakpoint and keep the others.
                Regs r{};
                lock.unlock();
                read_regs(r);
                trace::log(trace::Cat::Boot, "break: eip {:08x} reached\n{}", r.eip, debug::describe_stack(machine_->bus(), r));
                std::erase(break_eips_, r.eip);
                kvm_guest_debug off{};
                ioctl(vcpu_fd_, KVM_SET_GUEST_DEBUG, &off);
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
