// The headless runner: boots the machine from the user's dumps and reports what the boot chain did.
// The window shell arrives with the first picture (Phase 3); until then this is the instrument.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>

#include "core/clock.h"
#include "core/machine.h"
#include "core/version.h"
#include "debug/trace.h"

namespace {

void print_usage() {
    std::printf("%s %s\n\n", mv::product_name().data(), mv::version_string().data());
    std::printf(
        "Usage: midway-verdant --mcpx FILE --bios FILE [options]\n\n"
        "  --mcpx FILE        the 512-byte MCPX boot ROM dump\n"
        "  --bios FILE        the flash image (256 KiB, 512 KiB or 1 MiB)\n"
        "  --eeprom FILE      a 256-byte EEPROM image; generated there from the kernel's key when missing\n"
        "  --hdd FILE         a raw hard-disk image (8 GiB retail layout); no drive without it\n"
        "  --hdd-unlocked     present the disk unlocked instead of locked with the EEPROM's key\n"
        "  --smc-scratch N    the SMC scratch flags the kernel reads at boot (4 = short animation, 8 = dashboard)\n"
        "  --apu-irq-bits M   development probe: raise these APU interrupt status bits every audio frame\n"
        "  --seconds N        run for N seconds of host time (default 5)\n"
        "  --until EVENT      stop early at a milestone: rom-hidden, bootloader, kernel\n"
        "  --trace LIST       comma-separated: post,pci,smbus,mmio,pio,irq,flash,cpu,boot,timer,gpu,all\n"
        "  --trace-file FILE  write the trace there instead of stderr\n"
        "  --rtc-time UNIX    fix the clock's start for a deterministic run\n"
        "  --dump-ram FILE    write the guest RAM to FILE when the run ends\n"
        "  --version, --help\n");
}

const char* reason_name(mv::Machine::StopReason r) {
    switch (r) {
        case mv::Machine::StopReason::Duration: return "time limit";
        case mv::Machine::StopReason::Condition: return "milestone reached";
        case mv::Machine::StopReason::CpuStopped: return "CPU stopped";
        case mv::Machine::StopReason::PowerOff: return "power off";
    }
    return "?";
}

std::string ms(std::optional<mv::Cycles> c) {
    if (!c) return "-";
    return std::format("{:.3f} ms", static_cast<double>(mv::ns_from_cycles(*c)) / 1e6);
}

}  // namespace

int main(int argc, char** argv) {
    mv::MachineConfig cfg;
    double seconds = 5.0;
    std::string until;
    std::FILE* trace_file = nullptr;
    std::string dump_ram;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "midway-verdant: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--version") {
            std::printf("%s %s\n", mv::product_name().data(), mv::version_string().data());
            return 0;
        } else if (a == "--help" || a == "-h") {
            print_usage();
            return 0;
        } else if (a == "--mcpx") {
            cfg.mcpx_rom = need("--mcpx");
        } else if (a == "--bios") {
            cfg.flash_image = need("--bios");
        } else if (a == "--eeprom") {
            cfg.eeprom_image = need("--eeprom");
        } else if (a == "--hdd") {
            cfg.hdd_image = need("--hdd");
        } else if (a == "--hdd-unlocked") {
            cfg.hdd_locked = false;
        } else if (a == "--apu-irq-read-clears") {
            cfg.apu_probe_read_clears = true;
        } else if (a == "--apu-irq-bits") {
            cfg.apu_probe_irq_bits = static_cast<mv::u32>(std::strtoul(need("--apu-irq-bits"), nullptr, 0));
        } else if (a == "--smc-scratch") {
            cfg.smc_scratch = static_cast<mv::u8>(std::strtoul(need("--smc-scratch"), nullptr, 0));
        } else if (a == "--seconds") {
            seconds = std::atof(need("--seconds"));
        } else if (a == "--until") {
            until = need("--until");
        } else if (a == "--trace") {
            mv::u32 mask = 0;
            const char* list = need("--trace");
            if (!mv::trace::parse_list(list, mask)) {
                std::fprintf(stderr, "midway-verdant: unknown trace category in %s\n", list);
                return 2;
            }
            mv::trace::enable(mask);
        } else if (a == "--trace-file") {
            trace_file = std::fopen(need("--trace-file"), "w");
            if (!trace_file) {
                std::fprintf(stderr, "midway-verdant: cannot open the trace file\n");
                return 2;
            }
            mv::trace::set_sink(trace_file);
        } else if (a == "--dump-ram") {
            dump_ram = need("--dump-ram");
        } else if (a == "--rtc-time") {
            cfg.rtc_base_time = std::atoll(need("--rtc-time"));
        } else {
            std::fprintf(stderr, "midway-verdant: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (cfg.mcpx_rom.empty() || cfg.flash_image.empty()) {
        print_usage();
        return 2;
    }

    mv::Machine machine(cfg);
    std::string error;
    if (!machine.init(error)) {
        std::fprintf(stderr, "midway-verdant: %s\n", error.c_str());
        return 1;
    }
    std::function<bool(mv::Machine&)> condition;
    if (until == "rom-hidden") condition = [](mv::Machine& m) { return m.milestones().rom_hidden.has_value(); };
    else if (until == "bootloader") condition = [](mv::Machine& m) { return m.milestones().bootloader_magic.has_value(); };
    else if (until == "kernel") condition = [](mv::Machine& m) { return m.milestones().kernel_header.has_value(); };
    else if (!until.empty()) {
        std::fprintf(stderr, "midway-verdant: unknown milestone %s\n", until.c_str());
        return 2;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto reason = machine.run(std::chrono::nanoseconds(static_cast<long long>(seconds * 1e9)), condition);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("stopped: %s after %.3f s host, %.3f ms guest\n", reason_name(reason), wall,
                static_cast<double>(mv::ns_from_cycles(machine.scheduler().now())) / 1e6);
    if (machine.cpu()->stop_reason() != mv::CpuBackend::Stop::None)
        std::printf("cpu: %s\n", machine.cpu()->stop_detail().c_str());
    const auto& m = machine.milestones();
    std::printf("milestones: rom hidden %s | smc challenge %s | bootloader %s | kernel %s | power off %s\n", ms(m.rom_hidden).c_str(),
                ms(m.smc_challenge_answered).c_str(), ms(m.bootloader_magic).c_str(), ms(m.kernel_header).c_str(), ms(m.power_off).c_str());
    if (!machine.power_off_reason().empty()) std::printf("power off: %s\n", machine.power_off_reason().c_str());
    if (const auto v = machine.kernel().version())
        std::printf("kernel: %u.%u.%u.%u%s\n", v->major, v->minor, v->build, v->qfe, machine.eeprom_generated() ? " (EEPROM generated from its key)" : "");

    std::string post;
    for (const auto& e : machine.post().history()) post += std::format("{:02x} ", e.code);
    std::printf("post codes (%zu): %s\n", machine.post().history().size(), post.c_str());

    const auto stats = machine.cpu()->stats();
    std::printf("cpu exits: %llu (mmio %llu, pio %llu, hlt %llu, irq window %llu), interrupts injected %llu\n",
                static_cast<unsigned long long>(stats.exits), static_cast<unsigned long long>(stats.mmio_exits),
                static_cast<unsigned long long>(stats.pio_exits), static_cast<unsigned long long>(stats.hlt_exits),
                static_cast<unsigned long long>(stats.irq_window_exits), static_cast<unsigned long long>(stats.irq_injections));
    {
        const mv::u8 base = machine.pic().master().vector_base;
        std::printf("timer interrupts (irq0, vector %02x): %llu; injections by vector:", base,
                    static_cast<unsigned long long>(stats.by_vector[base]));
        for (int v = 0; v < 256; ++v)
            if (stats.by_vector[v]) std::printf(" %02x×%llu", v, static_cast<unsigned long long>(stats.by_vector[v]));
        std::printf("\n");
    }
    std::printf("smbus transactions: %llu; gpu register reads %llu writes %llu, vblanks %llu; apu r/w %llu/%llu; aci r/w %llu/%llu\n",
                static_cast<unsigned long long>(machine.smbus().transactions()), static_cast<unsigned long long>(machine.nv2a().reads()),
                static_cast<unsigned long long>(machine.nv2a().writes()), static_cast<unsigned long long>(machine.nv2a().vblanks()),
                static_cast<unsigned long long>(machine.apu().reads()), static_cast<unsigned long long>(machine.apu().writes()),
                static_cast<unsigned long long>(machine.aci().reads()), static_cast<unsigned long long>(machine.aci().writes()));
    std::printf("apu frames %llu; apu irq %s\n", static_cast<unsigned long long>(machine.apu_frames()), machine.apu().irq_level() ? "high" : "low");
    std::printf("gpu fifo: %llu methods, %llu semaphore releases, %llu notifies%s\n",
                static_cast<unsigned long long>(machine.nv2a().pushbuffer().methods_seen()),
                static_cast<unsigned long long>(machine.nv2a().semaphore_releases()), static_cast<unsigned long long>(machine.nv2a().notifies()),
                machine.nv2a().pushbuffer().errored() ? (" ERROR: " + machine.nv2a().pushbuffer().last_error()).c_str() : "");
    std::printf("ide: %llu commands, %llu sectors read, %llu written; hdd %s, unlock attempts %llu; dvd packets %llu\n",
                static_cast<unsigned long long>(machine.ide().primary().commands()),
                static_cast<unsigned long long>(machine.ide().primary().sectors_read()),
                static_cast<unsigned long long>(machine.ide().primary().sectors_written()),
                machine.hdd() ? (machine.hdd()->locked() ? "locked" : "unlocked") : "absent",
                static_cast<unsigned long long>(machine.hdd() ? machine.hdd()->unlock_attempts() : 0),
                static_cast<unsigned long long>(static_cast<const mv::AtapiCdrom*>(machine.ide().primary().device(1)) ? static_cast<const mv::AtapiCdrom*>(machine.ide().primary().device(1))->packets() : 0));
    mv::CpuBackend::Regs regs{};
    if (machine.cpu()->read_regs(regs)) {
        std::printf("cpu: cs %04x eip %08x eflags %08x cr0 %08x cr3 %08x esp %08x tsc %llu\n", regs.cs, regs.eip, regs.eflags, regs.cr0,
                    regs.cr3, regs.esp, static_cast<unsigned long long>(regs.tsc));
        std::printf("     eax %08x ebx %08x ecx %08x edx %08x esi %08x edi %08x ebp %08x\n", regs.eax, regs.ebx, regs.ecx, regs.edx, regs.esi,
                    regs.edi, regs.ebp);
    }
    const auto unknown_mmio = machine.bus().unknown_mmio();
    const auto unknown_pio = machine.bus().unknown_pio();
    if (!unknown_mmio.empty()) {
        std::printf("unmapped mmio pages (%zu):", unknown_mmio.size());
        std::size_t shown = 0;
        for (const auto& u : unknown_mmio) {
            if (shown++ >= 24) {
                std::printf(" ...");
                break;
            }
            std::printf(" %08x%s×%llu", u.page, u.is_write ? "w" : "r", static_cast<unsigned long long>(u.count));
        }
        std::printf("\n");
    }
    if (!unknown_pio.empty()) {
        std::printf("unmapped ports (%zu):", unknown_pio.size());
        std::size_t shown = 0;
        for (const auto& u : unknown_pio) {
            if (shown++ >= 24) {
                std::printf(" ...");
                break;
            }
            std::printf(" %04x%s×%llu", u.page, u.is_write ? "w" : "r", static_cast<unsigned long long>(u.count));
        }
        std::printf("\n");
    }
    std::printf("kvm: kicks %llu, EINTR exits %llu; exits by reason:", static_cast<unsigned long long>(stats.kicks),
                static_cast<unsigned long long>(stats.intr_exits));
    for (int r = 0; r < 64; ++r)
        if (stats.by_reason[r]) std::printf(" %d×%llu", r, static_cast<unsigned long long>(stats.by_reason[r]));
    std::printf("\n");
    std::printf("pic: master irr %02x isr %02x imr %02x elcr %02x | slave irr %02x isr %02x imr %02x elcr %02x | intr %d\n",
                machine.pic().master().irr, machine.pic().master().isr, machine.pic().master().imr, machine.pic().master().elcr,
                machine.pic().slave().irr, machine.pic().slave().isr, machine.pic().slave().imr, machine.pic().slave().elcr,
                machine.pic().intr_asserted() ? 1 : 0);
    std::printf("pit: ch0 mode %u reload %u counting %d\n", machine.pit().channel(0).mode, machine.pit().channel(0).reload,
                machine.pit().channel(0).counting ? 1 : 0);
    if (!dump_ram.empty()) {
        if (std::FILE* f = std::fopen(dump_ram.c_str(), "wb")) {
            std::fwrite(machine.bus().ram(), 1, machine.bus().ram_size(), f);
            std::fclose(f);
            std::printf("ram written to %s\n", dump_ram.c_str());
        }
    }
    if (trace_file) std::fclose(trace_file);
    return 0;
}
