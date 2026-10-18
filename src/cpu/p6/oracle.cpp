#include "cpu/p6/oracle.h"

#include <cstdio>
#include <format>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/kvm.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>

#include "core/machine.h"
#include "debug/trace.h"

namespace mv::p6 {

namespace {
constexpr u32 kSlotRam = 0, kSlotRamAlias = 1, kSlotFlash = 2, kSlotTopPage = 3;
constexpr unsigned kFullReports = 24;  // divergences printed in full before only counting

struct CpuidFour {
    alignas(8) unsigned char bytes[sizeof(kvm_cpuid2) + 4 * sizeof(kvm_cpuid_entry2)]{};
    kvm_cpuid2* header() { return reinterpret_cast<kvm_cpuid2*>(bytes); }
    kvm_cpuid_entry2* entries() { return reinterpret_cast<kvm_cpuid_entry2*>(bytes + sizeof(kvm_cpuid2)); }
};

kvm_segment to_kvm(const Segment& s) {
    kvm_segment k{};
    k.selector = s.selector;
    k.base = s.base;
    k.limit = s.limit;
    k.type = static_cast<u8>(s.ar & 0xF);
    k.s = static_cast<u8>((s.ar >> 4) & 1);
    k.dpl = static_cast<u8>((s.ar >> 5) & 3);
    k.present = static_cast<u8>((s.ar >> 7) & 1);
    k.avl = static_cast<u8>((s.ar >> 12) & 1);
    k.l = static_cast<u8>((s.ar >> 13) & 1);
    k.db = static_cast<u8>((s.ar >> 14) & 1);
    k.g = static_cast<u8>((s.ar >> 15) & 1);
    k.unusable = static_cast<u8>((s.ar >> 16) & 1);
    return k;
}

Segment from_kvm(const kvm_segment& k) {
    Segment s;
    s.selector = k.selector;
    s.base = static_cast<u32>(k.base);
    s.limit = k.limit;
    s.ar = static_cast<u32>(k.type) | (static_cast<u32>(k.s) << 4) | (static_cast<u32>(k.dpl) << 5) | (static_cast<u32>(k.present) << 7) |
           (static_cast<u32>(k.avl) << 12) | (static_cast<u32>(k.l) << 13) | (static_cast<u32>(k.db) << 14) | (static_cast<u32>(k.g) << 15) |
           (k.unusable ? Segment::kUnusable : 0u);
    return s;
}

const char* seg_name(unsigned i) {
    static const char* const names[] = {"es", "cs", "ss", "ds", "fs", "gs"};
    return names[i];
}
const char* gpr_name(unsigned i) {
    static const char* const names[] = {"eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"};
    return names[i];
}
}  // namespace

struct Oracle::Impl {
    Machine* machine{nullptr};
    Cpu* cpu{nullptr};
    int kvm_fd{-1}, vm_fd{-1}, vcpu_fd{-1};
    kvm_run* run{nullptr};
    std::size_t run_size{0};
    bool sync_regs{false};
    u64 full_reports{0};
    std::string error;

    ~Impl() {
        if (run) munmap(run, run_size);
        if (vcpu_fd >= 0) close(vcpu_fd);
        if (vm_fd >= 0) close(vm_fd);
        if (kvm_fd >= 0) close(kvm_fd);
    }

    bool set_slot(u32 slot, u64 guest_phys, u64 size, const void* host, bool readonly, std::string& err) {
        kvm_userspace_memory_region region{};
        region.slot = slot;
        region.flags = readonly ? KVM_MEM_READONLY : 0u;
        region.guest_phys_addr = guest_phys;
        region.memory_size = size;
        region.userspace_addr = reinterpret_cast<u64>(host);
        if (ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region) < 0) {
            err = std::format("oracle: KVM_SET_USER_MEMORY_REGION slot {} failed: {}", slot, std::strerror(errno));
            return false;
        }
        return true;
    }

    void set_top_page(const void* backing) {
        kvm_userspace_memory_region region{};
        region.slot = kSlotTopPage;
        region.memory_size = 0;
        region.guest_phys_addr = 0xFFFFF000u;
        ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region);
        std::string err;
        if (!set_slot(kSlotTopPage, 0xFFFFF000u, 4096, backing, true, err)) error = err;
    }

    bool init(Machine& m, Cpu& c, std::string& err) {
        machine = &m;
        cpu = &c;
        kvm_fd = open("/dev/kvm", O_RDWR | O_CLOEXEC);
        if (kvm_fd < 0) {
            err = std::format("oracle: cannot open /dev/kvm: {}", std::strerror(errno));
            return false;
        }
        vm_fd = ioctl(kvm_fd, KVM_CREATE_VM, 0);
        if (vm_fd < 0) {
            err = std::format("oracle: KVM_CREATE_VM failed: {}", std::strerror(errno));
            return false;
        }
        ioctl(vm_fd, KVM_SET_TSS_ADDR, 0xFFFBD000);
        u64 identity = 0xFFFBC000;
        ioctl(vm_fd, KVM_SET_IDENTITY_MAP_ADDR, &identity);
        Bus& bus = m.bus();
        Flash& flash = m.flash();
        if (!set_slot(kSlotRam, 0, bus.ram_size(), bus.ram(), false, err)) return false;
        if (!set_slot(kSlotRamAlias, Bus::kRamAliasBase, bus.ram_size(), bus.ram(), false, err)) return false;
        if (!set_slot(kSlotFlash, Flash::kBase, Flash::kWindow - 4096, flash.window(), true, err)) return false;
        const void* top = flash.rom_visible() ? flash.overlay_page() : flash.window() + Flash::kWindow - 4096;
        if (!set_slot(kSlotTopPage, 0xFFFFF000u, 4096, top, true, err)) return false;
        vcpu_fd = ioctl(vm_fd, KVM_CREATE_VCPU, 0);
        if (vcpu_fd < 0) {
            err = std::format("oracle: KVM_CREATE_VCPU failed: {}", std::strerror(errno));
            return false;
        }
        const int mmap_size = ioctl(kvm_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
        if (mmap_size <= 0) {
            err = "oracle: KVM_GET_VCPU_MMAP_SIZE failed";
            return false;
        }
        run_size = static_cast<std::size_t>(mmap_size);
        void* p = mmap(nullptr, run_size, PROT_READ | PROT_WRITE, MAP_SHARED, vcpu_fd, 0);
        if (p == MAP_FAILED) {
            err = std::format("oracle: mmap of the run area failed: {}", std::strerror(errno));
            return false;
        }
        run = static_cast<kvm_run*>(p);
        // The same Coppermine the fast mode presents.
        CpuidFour cpuid{};
        cpuid.header()->nent = 4;
        kvm_cpuid_entry2* e = cpuid.entries();
        e[0].function = 0; e[0].eax = 2; e[0].ebx = 0x756E6547; e[0].edx = 0x49656E69; e[0].ecx = 0x6C65746E;
        e[1].function = 1; e[1].eax = 0x0000068A; e[1].edx = 0x0383F9FF;
        e[2].function = 2; e[2].eax = 0x03020101; e[2].edx = 0x0C040841;
        e[3].function = 0x80000000;
        if (ioctl(vcpu_fd, KVM_SET_CPUID2, cpuid.header()) < 0) {
            err = std::format("oracle: KVM_SET_CPUID2 failed: {}", std::strerror(errno));
            return false;
        }
        kvm_guest_debug dbg{};
        dbg.control = KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_SINGLESTEP;
        if (ioctl(vcpu_fd, KVM_SET_GUEST_DEBUG, &dbg) < 0) {
            err = std::format("oracle: KVM_SET_GUEST_DEBUG (single-step) failed: {}", std::strerror(errno));
            return false;
        }
#if defined(KVM_CAP_SYNC_REGS) && defined(KVM_SYNC_X86_REGS)
        const int caps = ioctl(kvm_fd, KVM_CHECK_EXTENSION, KVM_CAP_SYNC_REGS);
        const u64 want = static_cast<u64>(KVM_SYNC_X86_REGS | KVM_SYNC_X86_SREGS);
        sync_regs = caps > 0 && (static_cast<u64>(caps) & want) == want;
#endif
        flash.add_on_rom_hidden([this] { set_top_page(machine->flash().window() + Flash::kWindow - 4096); });
        flash.add_on_rom_shown([this] { set_top_page(machine->flash().overlay_page()); });
        std::fprintf(stderr, "oracle: KVM lock-step oracle armed (%s)\n", sync_regs ? "sync_regs" : "ioctl regs");
        return true;
    }

    void fill_regs(const CpuState& s, kvm_regs& r, kvm_sregs& sr) {
        r = kvm_regs{};
        r.rax = s.gpr[kEax]; r.rcx = s.gpr[kEcx]; r.rdx = s.gpr[kEdx]; r.rbx = s.gpr[kEbx];
        r.rsp = s.gpr[kEsp]; r.rbp = s.gpr[kEbp]; r.rsi = s.gpr[kEsi]; r.rdi = s.gpr[kEdi];
        r.rip = s.eip;
        // KVM single-steps with the trap flag and re-arms it only when the entry RIP is the one it recorded
        // when single-stepping was switched on; every replay carries TF itself.
        r.rflags = s.eflags() | fl::TF;
        sr = kvm_sregs{};
        sr.es = to_kvm(s.seg[kEs]); sr.cs = to_kvm(s.seg[kCs]); sr.ss = to_kvm(s.seg[kSs]);
        sr.ds = to_kvm(s.seg[kDs]); sr.fs = to_kvm(s.seg[kFs]); sr.gs = to_kvm(s.seg[kGs]);
        sr.tr = to_kvm(s.tr);
        sr.ldt = to_kvm(s.ldtr);
        sr.gdt.base = s.gdtr.base; sr.gdt.limit = static_cast<u16>(s.gdtr.limit);
        sr.idt.base = s.idtr.base; sr.idt.limit = static_cast<u16>(s.idtr.limit);
        sr.cr0 = s.cr0; sr.cr2 = s.cr2; sr.cr3 = s.cr3; sr.cr4 = s.cr4;
        sr.efer = 0;
        sr.apic_base = 0;
    }

    bool load(const CpuState& s) {
        kvm_regs r;
        kvm_sregs sr;
        fill_regs(s, r, sr);
        if (sync_regs) {
#if defined(KVM_SYNC_X86_REGS)
            run->s.regs.regs = r;
            run->s.regs.sregs = sr;
            run->kvm_valid_regs = KVM_SYNC_X86_REGS | KVM_SYNC_X86_SREGS;
            run->kvm_dirty_regs = KVM_SYNC_X86_REGS | KVM_SYNC_X86_SREGS;
            return true;
#endif
        }
        if (ioctl(vcpu_fd, KVM_SET_SREGS, &sr) < 0) {
            error = std::format("oracle: KVM_SET_SREGS failed: {}", std::strerror(errno));
            return false;
        }
        if (ioctl(vcpu_fd, KVM_SET_REGS, &r) < 0) {
            error = std::format("oracle: KVM_SET_REGS failed: {}", std::strerror(errno));
            return false;
        }
        return true;
    }

    bool fetch(kvm_regs& r, kvm_sregs& sr) {
        if (sync_regs) {
#if defined(KVM_SYNC_X86_REGS)
            r = run->s.regs.regs;
            sr = run->s.regs.sregs;
            return true;
#endif
        }
        if (ioctl(vcpu_fd, KVM_GET_REGS, &r) < 0 || ioctl(vcpu_fd, KVM_GET_SREGS, &sr) < 0) {
            error = std::format("oracle: KVM_GET_REGS failed: {}", std::strerror(errno));
            return false;
        }
        return true;
    }

    // The x87/SSE state crosses only around the instructions that touch it (the hypervisor's own FPU state is
    // otherwise unrelated to the interpreter's).
    static bool touches_fpu(const Insn& in) {
        switch (in.op) {
            case Op::X87: case Op::Simd: case Op::Fxsave: case Op::Fxrstor: case Op::Ldmxcsr: case Op::Stmxcsr: case Op::Wait: return true;
            default: return false;
        }
    }
    void fill_fpu(const FpuState& f, kvm_fpu& k) {
        k = kvm_fpu{};
        for (unsigned i = 0; i < 8; ++i) std::memcpy(k.fpr[i], &f.st[(f.top() + i) & 7], 10);
        k.fcw = f.fcw;
        k.fsw = f.fsw;
        u8 abridged = 0;
        for (unsigned p = 0; p < 8; ++p)
            if (((f.ftw >> (2 * p)) & 3) != 3) abridged = static_cast<u8>(abridged | (1u << p));
        k.ftwx = abridged;
        k.last_opcode = f.fop;
        k.last_ip = f.fip;
        k.last_dp = f.fdp;
        for (unsigned i = 0; i < 8; ++i) std::memcpy(k.xmm[i], f.xmm[i].data(), 16);
        k.mxcsr = f.mxcsr;
    }
    bool set_fpu(const FpuState& f) {
        kvm_fpu k;
        fill_fpu(f, k);
        if (ioctl(vcpu_fd, KVM_SET_FPU, &k) < 0) {
            error = std::format("oracle: KVM_SET_FPU failed: {}", std::strerror(errno));
            return false;
        }
        return true;
    }
    bool get_fpu(kvm_fpu& k) {
        if (ioctl(vcpu_fd, KVM_GET_FPU, &k) < 0) {
            error = std::format("oracle: KVM_GET_FPU failed: {}", std::strerror(errno));
            return false;
        }
        // KVM_GET_FPU leaves MXCSR out; the XSAVE image carries it at byte 24 of the legacy region.
        kvm_xsave xs{};
        if (ioctl(vcpu_fd, KVM_GET_XSAVE, &xs) == 0) k.mxcsr = xs.region[6];
        return true;
    }
    // RCPPS/RCPSS/RSQRTPS/RSQRTSS are approximations whose tables differ between processor models (the hypervisor
    // runs on the host's): two results that each lie within the architectural bound of the exact value
    // (1.5 x 2^-12 relative) agree.
    static bool approximately_equal(const u8* a, const u8* b) {
        for (unsigned lane = 0; lane < 4; ++lane) {
            float x, y;
            std::memcpy(&x, a + 4 * lane, 4);
            std::memcpy(&y, b + 4 * lane, 4);
            if (std::memcmp(&x, &y, 4) == 0) continue;
            if (!std::isfinite(x) || !std::isfinite(y) || std::fabs(x - y) > 3.0f * 0x1p-12f * std::fabs(y)) return false;
        }
        return true;
    }
    // Compare the interpreter's FPU state with the hypervisor's; the pointers (FIP/FDP/FOP) are reported apart.
    static void compare_fpu(const FpuState& f, const kvm_fpu& k, bool approximation, std::vector<std::string>& diffs, u64& pointer_diffs,
                            u64& approximation_diffs) {
        if (f.fcw != k.fcw) diffs.push_back(std::format("fcw ours {:04x} kvm {:04x}", f.fcw, k.fcw));
        if (f.fsw != k.fsw) diffs.push_back(std::format("fsw ours {:04x} kvm {:04x}", f.fsw, k.fsw));
        u8 abridged = 0;
        for (unsigned p = 0; p < 8; ++p)
            if (((f.ftw >> (2 * p)) & 3) != 3) abridged = static_cast<u8>(abridged | (1u << p));
        if (abridged != k.ftwx) diffs.push_back(std::format("ftw ours {:02x} kvm {:02x} (abridged)", abridged, k.ftwx));
        for (unsigned i = 0; i < 8; ++i) {
            if (!((abridged >> ((f.top() + i) & 7)) & 1)) continue;  // empty registers are not compared
            u8 ours[10];
            std::memcpy(ours, &f.st[(f.top() + i) & 7], 10);
            if (std::memcmp(ours, k.fpr[i], 10) != 0) {
                std::string a, b;
                for (int j = 9; j >= 0; --j) {
                    a += std::format("{:02x}", ours[j]);
                    b += std::format("{:02x}", k.fpr[i][j]);
                }
                diffs.push_back(std::format("st({}) ours {} kvm {}", i, a, b));
            }
        }
        for (unsigned i = 0; i < 8; ++i) {
            if (std::memcmp(f.xmm[i].data(), k.xmm[i], 16) != 0) {
                if (approximation && approximately_equal(f.xmm[i].data(), k.xmm[i])) {
                    ++approximation_diffs;
                    continue;
                }
                std::string a, b;
                for (int j = 15; j >= 0; --j) {
                    a += std::format("{:02x}", f.xmm[i][static_cast<std::size_t>(j)]);
                    b += std::format("{:02x}", k.xmm[i][j]);
                }
                diffs.push_back(std::format("xmm{} ours {} kvm {}", i, a, b));
            }
        }
        if (f.mxcsr != k.mxcsr) diffs.push_back(std::format("mxcsr ours {:08x} kvm {:08x}", f.mxcsr, k.mxcsr));
        if (f.fip != static_cast<u32>(k.last_ip) || f.fdp != static_cast<u32>(k.last_dp) || (f.fop & 0x7FF) != (k.last_opcode & 0x7FF)) ++pointer_diffs;
    }
    static void adopt_fpu(FpuState& f, const kvm_fpu& k) {
        f.fcw = k.fcw;
        f.fsw = k.fsw;
        for (unsigned i = 0; i < 8; ++i) std::memcpy(&f.st[(f.top() + i) & 7], k.fpr[i], 10);
        u16 ftw = 0;
        for (unsigned p = 0; p < 8; ++p) ftw = static_cast<u16>(ftw | ((((k.ftwx >> p) & 1) ? 0u : 3u) << (2 * p)));
        f.ftw = ftw;  // the tags are re-derived by the next FXSAVE/FXRSTOR; valid/zero/special are not distinguished here
        for (unsigned i = 0; i < 8; ++i) std::memcpy(f.xmm[i].data(), k.xmm[i], 16);
        f.mxcsr = k.mxcsr;
    }

    static void adopt(CpuState& s, const kvm_regs& r, const kvm_sregs& sr) {
        s.gpr[kEax] = static_cast<u32>(r.rax); s.gpr[kEcx] = static_cast<u32>(r.rcx); s.gpr[kEdx] = static_cast<u32>(r.rdx);
        s.gpr[kEbx] = static_cast<u32>(r.rbx); s.gpr[kEsp] = static_cast<u32>(r.rsp); s.gpr[kEbp] = static_cast<u32>(r.rbp);
        s.gpr[kEsi] = static_cast<u32>(r.rsi); s.gpr[kEdi] = static_cast<u32>(r.rdi);
        s.eip = static_cast<u32>(r.rip);
        s.set_eflags(static_cast<u32>(r.rflags) & ~fl::TF);
        s.seg[kEs] = from_kvm(sr.es); s.seg[kCs] = from_kvm(sr.cs); s.seg[kSs] = from_kvm(sr.ss);
        s.seg[kDs] = from_kvm(sr.ds); s.seg[kFs] = from_kvm(sr.fs); s.seg[kGs] = from_kvm(sr.gs);
        s.tr = from_kvm(sr.tr);
        s.ldtr = from_kvm(sr.ldt);
        s.gdtr.base = static_cast<u32>(sr.gdt.base); s.gdtr.limit = sr.gdt.limit;
        s.idtr.base = static_cast<u32>(sr.idt.base); s.idtr.limit = sr.idt.limit;
        s.cr0 = static_cast<u32>(sr.cr0); s.cr2 = static_cast<u32>(sr.cr2); s.cr3 = static_cast<u32>(sr.cr3); s.cr4 = static_cast<u32>(sr.cr4);
    }
};

Oracle::Oracle() : impl_(new Impl) {}
Oracle::~Oracle() { delete impl_; }

bool Oracle::init(Machine& machine, Cpu& cpu, std::string& error) { return impl_->init(machine, cpu, error); }

void Oracle::reset() {}

std::string Oracle::summary() const {
    return std::format("oracle: {} steps, {} compared, {} divergences ({} resyncs), {} skipped, {} lock-step breaks, {} device-log mismatches, {} memory mismatches, {} undefined-flag differences, {} x87 pointer differences, {} RCP/RSQRT approximation differences",
                       counts_.steps, counts_.compared, counts_.divergences, counts_.resyncs, counts_.skipped, counts_.lockstep_breaks, counts_.io_mismatches,
                       counts_.memory_mismatches, counts_.undefined_flag_differences, counts_.fpu_pointer_differences, counts_.approximation_differences);
}

bool Oracle::step(Cpu& cpu) {
    Impl& im = *impl_;
    Memory& mem = cpu.memory();
    const CpuState before = cpu.state();
    mem.set_logging(true);
    try {
        cpu.step();
    } catch (...) {
        mem.set_logging(false);
        throw;
    }
    mem.set_logging(false);
    ++counts_.steps;
    const Insn& in = cpu.last_insn();
    const CpuState& after = cpu.state();

    // Steps the hypervisor cannot replay one-for-one: a fault (its handler would run as one step), an external
    // delivery, software interrupts, an interrupt shadow, MSR writes (the hypervisor's MSRs are not ours).
    bool skip = cpu.last_faulted() || cpu.last_delivered_interrupt() || after.inhibit_interrupts || cpu.triple_fault();
    switch (in.op) {
        case Op::Int: case Op::Int3: case Op::Into: case Op::Int1: case Op::Wrmsr: case Op::Rsm: case Op::Ud2: skip = true; break;
        // String output: KVM's emulator retires it at the exit without the trap flag and without advancing RIP
        // until the next entry (a FIXME in KVM), which cannot be replayed one-for-one; the device traffic
        // it produces is checked by the devices themselves.
        case Op::Outs: skip = true; break;
        default: break;
    }
    if (skip) {
        mem.commit_writes();
        ++counts_.skipped;
        return true;
    }

    // Replay on the hypervisor from the pre-state; its device exits are answered from the interpreter's log.
    const bool fpu_insn = Impl::touches_fpu(in);
    if (fpu_insn && !im.set_fpu(before.fpu)) {
        error_ = im.error;
        return false;
    }
    if (!im.load(before)) {
        error_ = im.error;
        return false;
    }
    const auto& io = mem.io_log();
    std::size_t io_next = 0;
    bool halted = false, shutdown = false;
    bool rep_ins_retired = false;
run_again:
    for (;;) {
        im.run->immediate_exit = 0;
        const int rc = ioctl(im.vcpu_fd, KVM_RUN, 0);
        if (rc < 0 && errno == EINTR) continue;
        if (rc < 0) {
            error_ = std::format("oracle: KVM_RUN failed: {}", std::strerror(errno));
            return false;
        }
        bool done = false;
        switch (im.run->exit_reason) {
            case KVM_EXIT_DEBUG: done = true; break;
            case KVM_EXIT_HLT: halted = true; done = true; break;
            case KVM_EXIT_SHUTDOWN: shutdown = true; done = true; break;
            case KVM_EXIT_INTR: break;
            case KVM_EXIT_IO: {
                u8* data = reinterpret_cast<u8*>(im.run) + im.run->io.data_offset;
                const unsigned size = im.run->io.size;
                for (u32 i = 0; i < im.run->io.count; ++i, data += size) {
                    const bool is_write = im.run->io.direction == KVM_EXIT_IO_OUT;
                    u32 v = 0;
                    if (is_write) std::memcpy(&v, data, size);
                    bool matched = false;
                    if (io_next < io.size()) {
                        const IoRecord& rec = io[io_next];
                        if (!rec.is_mmio && rec.is_write == is_write && rec.addr == im.run->io.port && rec.size == size && (!is_write || rec.value == v)) {
                            matched = true;
                            ++io_next;
                            if (!is_write) std::memcpy(data, &rec.value, size);
                        }
                    }
                    if (!matched) {
                        ++counts_.io_mismatches;
                        if (im.full_reports < kFullReports)
                            std::fprintf(stderr, "oracle: port %s %04x size %u%s not in the interpreter's log (record %zu of %zu) at %04x:%08x\n",
                                         is_write ? "write" : "read", im.run->io.port, size, is_write ? std::format(" value {:08x}", v).c_str() : "", io_next,
                                         io.size(), cpu.last_cs(), cpu.last_eip());
                        // Keep the hypervisor coherent with the machine anyway.
                        if (is_write) im.machine->bus().pio_write(im.run->io.port, size, v);
                        else {
                            const u32 r = im.machine->bus().pio_read(im.run->io.port, size);
                            std::memcpy(data, &r, size);
                        }
                    }
                }
                break;
            }
            case KVM_EXIT_MMIO: {
                const PhysAddr addr = static_cast<PhysAddr>(im.run->mmio.phys_addr);
                const unsigned len = im.run->mmio.len;
                const bool is_write = im.run->mmio.is_write != 0;
                u32 v = 0;
                if (is_write) std::memcpy(&v, im.run->mmio.data, len < 4 ? len : 4);
                bool matched = false;
                if (len <= 4 && io_next < io.size()) {
                    const IoRecord& rec = io[io_next];
                    if (rec.is_mmio && rec.is_write == is_write && rec.addr == addr && rec.size == len && (!is_write || rec.value == v)) {
                        matched = true;
                        ++io_next;
                        if (!is_write) std::memcpy(im.run->mmio.data, &rec.value, len);
                    }
                }
                if (!matched) {
                    ++counts_.io_mismatches;
                    if (im.full_reports < kFullReports)
                        std::fprintf(stderr, "oracle: mmio %s %08x len %u%s not in the interpreter's log (record %zu of %zu) at %04x:%08x\n",
                                     is_write ? "write" : "read", addr, len, is_write ? std::format(" value {:08x}", v).c_str() : "", io_next, io.size(),
                                     cpu.last_cs(), cpu.last_eip());
                    Bus& bus = im.machine->bus();
                    if (len == 8) {
                        if (is_write) {
                            u64 w;
                            std::memcpy(&w, im.run->mmio.data, 8);
                            bus.write64(addr, w);
                        } else {
                            const u64 r = bus.read64(addr);
                            std::memcpy(im.run->mmio.data, &r, 8);
                        }
                    } else if (is_write) {
                        bus.write(addr, len, v);
                    } else {
                        const u32 r = bus.read(addr, len);
                        std::memcpy(im.run->mmio.data, &r, len);
                    }
                }
                // KVM retires an instruction whose last operation is an MMIO write at this exit and its
                // emulator's write-back drops the trap flag, so running on would run free: the step ends here
                // once the interpreter's log holds nothing more for it.
                if (is_write && (!matched || io_next >= io.size())) done = true;
                break;
            }
            case KVM_EXIT_FAIL_ENTRY:
                error_ = std::format("oracle: KVM_EXIT_FAIL_ENTRY {:#x}", static_cast<u64>(im.run->fail_entry.hardware_entry_failure_reason));
                return false;
            case KVM_EXIT_INTERNAL_ERROR: error_ = std::format("oracle: KVM internal error {}", im.run->internal.suberror); return false;
            default: error_ = std::format("oracle: unhandled KVM exit {}", im.run->exit_reason); return false;
        }
        if (done) break;
    }
    kvm_regs r;
    kvm_sregs sr;
    if (!im.fetch(r, sr)) {
        error_ = im.error;
        return false;
    }
    // A batched REP INS: KVM copies the data in on the entry after the exit but skips the RIP write-back, leaving
    // ECX at zero; the zero-count repetition that retires the instruction runs on the next entry.
    if (in.op == Op::Ins && in.rep && !rep_ins_retired && !halted && !shutdown && r.rip == before.eip &&
        (r.rcx & (in.adsize == 2 ? 0xFFFFu : 0xFFFFFFFFu)) == 0) {
        rep_ins_retired = true;
        goto run_again;
    }
    ++counts_.compared;

    // Compare.
    std::vector<std::string> diffs;
    u32 flag_mask = fl::TF | fl::RF | after.undefined_flags;
    bool mask_eax_edx = cpu.last_was_rdtsc() || in.op == Op::Rdmsr;
    if (cpu.last_was_cpuid()) mask_eax_edx = true;
    for (unsigned i = 0; i < 8; ++i) {
        const u32 k = static_cast<u32>(i == kEax ? r.rax : i == kEcx ? r.rcx : i == kEdx ? r.rdx : i == kEbx ? r.rbx : i == kEsp ? r.rsp : i == kEbp ? r.rbp : i == kEsi ? r.rsi : r.rdi);
        if (mask_eax_edx && (i == kEax || i == kEdx)) continue;
        if (cpu.last_was_cpuid() && (i == kEbx || i == kEcx)) continue;
        if (after.gpr[i] != k) diffs.push_back(std::format("{} ours {:08x} kvm {:08x}", gpr_name(i), after.gpr[i], k));
    }
    if (after.eip != static_cast<u32>(r.rip)) diffs.push_back(std::format("eip ours {:08x} kvm {:08x}", after.eip, static_cast<u32>(r.rip)));
    {
        const u32 ours = after.eflags();
        const u32 kvm = static_cast<u32>(r.rflags);
        if (((ours ^ kvm) & ~flag_mask) != 0) diffs.push_back(std::format("eflags ours {:08x} kvm {:08x} (mask {:08x})", ours, kvm, flag_mask));
        else if (((ours ^ kvm) & after.undefined_flags) != 0) ++counts_.undefined_flag_differences;
    }
    const kvm_segment* ks[6] = {&sr.es, &sr.cs, &sr.ss, &sr.ds, &sr.fs, &sr.gs};
    for (unsigned i = 0; i < 6; ++i) {
        const Segment& o = after.seg[i];
        const Segment k = from_kvm(*ks[i]);
        if (o.unusable() || k.unusable()) {
            if (o.unusable() != k.unusable()) diffs.push_back(std::format("{} usable ours {} kvm {}", seg_name(i), !o.unusable(), !k.unusable()));
            continue;
        }
        if (o.selector != k.selector || o.base != k.base || o.limit != k.limit)
            diffs.push_back(std::format("{} ours {:04x} base {:08x} limit {:08x} kvm {:04x} base {:08x} limit {:08x}", seg_name(i), o.selector, o.base, o.limit,
                                        k.selector, k.base, k.limit));
        else if (after.protected_mode() && (o.ar & 0xF0FFu) != (k.ar & 0xF0FFu))
            diffs.push_back(std::format("{} rights ours {:04x} kvm {:04x}", seg_name(i), o.ar & 0xF0FFu, k.ar & 0xF0FFu));
    }
    if (after.gdtr.base != static_cast<u32>(sr.gdt.base) || after.gdtr.limit != sr.gdt.limit)
        diffs.push_back(std::format("gdtr ours {:08x}/{:04x} kvm {:08x}/{:04x}", after.gdtr.base, after.gdtr.limit, static_cast<u32>(sr.gdt.base), sr.gdt.limit));
    if (after.idtr.base != static_cast<u32>(sr.idt.base) || after.idtr.limit != sr.idt.limit)
        diffs.push_back(std::format("idtr ours {:08x}/{:04x} kvm {:08x}/{:04x}", after.idtr.base, after.idtr.limit, static_cast<u32>(sr.idt.base), sr.idt.limit));
    if (after.cr0 != static_cast<u32>(sr.cr0)) diffs.push_back(std::format("cr0 ours {:08x} kvm {:08x}", after.cr0, static_cast<u32>(sr.cr0)));
    if (after.cr3 != static_cast<u32>(sr.cr3)) diffs.push_back(std::format("cr3 ours {:08x} kvm {:08x}", after.cr3, static_cast<u32>(sr.cr3)));
    if (after.cr4 != static_cast<u32>(sr.cr4)) diffs.push_back(std::format("cr4 ours {:08x} kvm {:08x}", after.cr4, static_cast<u32>(sr.cr4)));
    if (after.protected_mode()) {
        const Segment kt = from_kvm(sr.tr), kl = from_kvm(sr.ldt);
        if (after.tr.selector != kt.selector || (!after.tr.unusable() && (after.tr.base != kt.base || after.tr.limit != kt.limit)))
            diffs.push_back(std::format("tr ours {:04x} base {:08x} kvm {:04x} base {:08x}", after.tr.selector, after.tr.base, kt.selector, kt.base));
        if (after.ldtr.selector != kl.selector) diffs.push_back(std::format("ldtr ours {:04x} kvm {:04x}", after.ldtr.selector, kl.selector));
    }
    if (after.halted != halted) diffs.push_back(std::format("halted ours {} kvm {}", after.halted, halted));
    if (shutdown) diffs.push_back("kvm: shutdown (triple fault)");
    kvm_fpu kf{};
    if (fpu_insn) {
        if (!im.get_fpu(kf)) {
            error_ = im.error;
            return false;
        }
        const bool approximation = in.op == Op::Simd && in.two_byte && (in.opcode == 0x52 || in.opcode == 0x53);
        Impl::compare_fpu(after.fpu, kf, approximation, diffs, counts_.fpu_pointer_differences, counts_.approximation_differences);
    }
    if (io_next != io.size()) {
        ++counts_.io_mismatches;
        diffs.push_back(std::format("device log: {} of {} records consumed by the hypervisor", io_next, io.size()));
    }
    // Memory: what the interpreter wrote against what the hypervisor left in RAM.
    Bus& bus = im.machine->bus();
    unsigned record_index = 0;
    for (const auto& w : mem.writes()) {
        u32 actual = 0;
        if (!bus.read_ram(w.phys, &actual, w.size)) continue;
        u32 mask = size_mask(w.size);
        // A pushed EFLAGS image carries the hypervisor's trap flag.
        if (in.op == Op::Pushf && record_index == 0) mask &= ~fl::TF;
        // An FXSAVE image's pointer fields and MXCSR_MASK are the host's; FNSAVE/FNSTENV's pointers likewise.
        if (in.op == Op::Fxsave) {
            const u32 off = w.phys - mem.writes().front().phys;
            if ((off >= 6 && off < 24) || (off >= 28 && off < 32)) mask = 0;
        }
        if (in.op == Op::X87 && (in.opcode == 0xDD || in.opcode == 0xD9) && in.mem && (in.reg == 6)) {
            const u32 off = w.phys - mem.writes().front().phys;
            const u32 env_lo = in.size == 4 ? 12u : 6u, env_hi = in.size == 4 ? 28u : 14u;
            if (off >= env_lo && off < env_hi) mask = 0;
        }
        if (((actual ^ w.value) & mask) != 0) {
            ++counts_.memory_mismatches;
            diffs.push_back(std::format("mem {:08x} size {} ours {:08x} kvm {:08x}", w.phys, w.size, w.value & mask, actual & mask));
        }
        ++record_index;
    }
    if (!diffs.empty()) {
        ++counts_.divergences;
        const bool big = diffs.size() > 6 || after.eip != static_cast<u32>(r.rip);
        if (big) ++counts_.lockstep_breaks;
        if (im.full_reports < kFullReports) {
            ++im.full_reports;
            std::fprintf(stderr, "oracle: divergence #%llu after %llu steps at %04x:%08x [%s] %s\n", static_cast<unsigned long long>(counts_.divergences),
                         static_cast<unsigned long long>(counts_.steps), cpu.last_cs(), cpu.last_eip(), cpu.last_bytes().c_str(), op_name(in.op));
            for (const auto& d : diffs) std::fprintf(stderr, "        %s\n", d.c_str());
            std::fprintf(stderr, "        before: %s\n", [&] {
                CpuState b = before;
                return std::format("eax {:08x} ebx {:08x} ecx {:08x} edx {:08x} esi {:08x} edi {:08x} ebp {:08x} esp {:08x} eflags {:08x}", b.gpr[kEax], b.gpr[kEbx],
                                   b.gpr[kEcx], b.gpr[kEdx], b.gpr[kEsi], b.gpr[kEdi], b.gpr[kEbp], b.gpr[kEsp], b.eflags());
            }().c_str());
            if (im.full_reports == kFullReports) std::fprintf(stderr, "oracle: further divergences are counted, not printed\n");
        }
        // The hardware's view wins: adopt it so the run goes on finding the next bug.
        Impl::adopt(cpu.state(), r, sr);
        if (fpu_insn) Impl::adopt_fpu(cpu.state().fpu, kf);
        cpu.state().halted = halted;
        mem.flush_tlb(false);
        ++counts_.resyncs;
    }
    // RAM already holds the hypervisor's writes; the capture's dirty record still wants the pages.
    for (const auto& w : mem.writes())
        if (!w.mmio) bus.mark_dirty(w.phys, w.size);
    mem.begin_instruction();
    return true;
}

}  // namespace mv::p6

#else

namespace mv::p6 {
struct Oracle::Impl {};
Oracle::Oracle() = default;
Oracle::~Oracle() = default;
bool Oracle::init(Machine&, Cpu&, std::string& error) {
    error = "the KVM oracle exists on Linux only";
    return false;
}
void Oracle::reset() {}
bool Oracle::step(Cpu& cpu) {
    cpu.step();
    return true;
}
std::string Oracle::summary() const { return {}; }
}  // namespace mv::p6

#endif
