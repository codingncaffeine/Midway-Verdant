// The shell: boots the machine from the user's dumps, shows its scanout in a window when asked, and reports
// what the boot chain did.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "core/clock.h"
#include "core/machine.h"
#include "core/version.h"
#include "debug/png.h"
#include "ide/hdd_format.h"
#include "debug/trace.h"
#include "nv2a/vulkan/vk_renderer.h"
#include "ui/window.h"

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
        "  --dvd FILE         a game disc image (XISO or a full dump) in the DVD tray\n"
        "  --create-hdd FILE  write a blank 8 GB retail hard-disk image (sparse, five FATX partitions) and exit\n"
        "  --smc-scratch N    the SMC scratch flags the kernel reads at boot (4 = short animation, 8 = dashboard)\n"
        "  --seconds N        run for N seconds of host time (default 5)\n"
        "  --until EVENT      stop early at a milestone: rom-hidden, bootloader, kernel\n"
        "  --trace LIST       comma-separated: post,pci,smbus,mmio,pio,irq,flash,cpu,boot,timer,gpu,apu,aci,all\n"
        "  --trace-file FILE  write the trace there instead of stderr\n"
        "  --rtc-time UNIX    fix the clock's start for a deterministic run\n"
        "  --dump-ram FILE    write the guest RAM to FILE when the run ends\n"
        "  --capture FILE     write a pushbuffer capture (GPU methods, RAM pages, objects) for offline replay\n"
        "  --watch HEX,...    log every change of these physical RAM words (polled each tick) under the boot trace\n"
        "  --break-smbus AA:CC  log the CPU registers and stack (boot trace) when SMBus address AA gets command CC written\n"
        "  --break-eip HEX,...  log the CPU registers and stack (boot trace) the first time execution reaches each address (up to 4)\n"
        "  --break-mmio ADDR[=VALUE][*N]  log the CPU registers and stack (boot trace) at a write to physical ADDR (hex), optionally only of VALUE, up to N times\n"
        "  --reset-at S       press the front-panel reset S seconds into the run (a warm reboot)\n"
        "  --window           show the console's scanout in a window (Escape or closing it ends the run)\n"
        "  --gpu              render the console's graphics live on the host's Vulkan device (frames land in guest RAM)\n"
        "  --snapshots DIR    with --window: save what the window shows as DIR/shot-SSS.bmp every 2 seconds\n"
        "  --luma-log FILE    write the scanout's guest time and mean luma (BT.601, 0-255) at every vblank\n"
        "  --scanout-pngs DIR save the scanout as DIR/scan-MMMMMM.png (guest milliseconds) every 15th vblank\n"
        "  --press B@F-T     hold button B on pad 1 from vblank F to vblank T (a b x y black white lt rt start back up down left right)\n"
        "  --cpu kvm|p6       the CPU: the hardware-virtualized fast mode (default) or our cycle-counted interpreter\n"
        "  --oracle           with --cpu p6: check every instruction against a KVM vCPU and report divergences\n"
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

// --press: a button held over a range of vblanks. Buttons 0-7 are the analog ones (PadState::Analog), 8 and up the
// digital bits.
struct Press {
    int button;
    unsigned long long from, to;
};
int press_button(const std::string& name) {
    static const std::pair<const char*, int> kNames[] = {{"a", 0}, {"b", 1}, {"x", 2}, {"y", 3}, {"black", 4}, {"white", 5}, {"lt", 6}, {"rt", 7},
                                                         {"up", 8}, {"down", 9}, {"left", 10}, {"right", 11}, {"start", 12}, {"back", 13}};
    for (const auto& [n, b] : kNames)
        if (name == n) return b;
    return -1;
}

int main(int argc, char** argv) {
    mv::MachineConfig cfg;
    double seconds = 5.0;
    std::string until;
    std::vector<Press> presses;
    std::FILE* trace_file = nullptr;
    std::string dump_ram;
    bool want_window = false;
    std::string snapshot_dir;
    std::FILE* luma_log = nullptr;
    std::string scanout_png_dir;
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
        } else if (a == "--create-hdd") {
            std::string create_error;
            const char* target = need("--create-hdd");
            if (!mv::create_blank_hdd(target, static_cast<mv::u32>(std::time(nullptr)), create_error)) {
                std::fprintf(stderr, "midway-verdant: %s\n", create_error.c_str());
                return 1;
            }
            std::printf("created a blank 8 GB hard disk at %s\n", target);
            return 0;
        } else if (a == "--dvd") {
            cfg.dvd_image = need("--dvd");
        } else if (a == "--hdd") {
            cfg.hdd_image = need("--hdd");
        } else if (a == "--hdd-unlocked") {
            cfg.hdd_locked = false;
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
        } else if (a == "--press") {
            const std::string spec = need("--press");
            const auto at = spec.find('@'), dash = spec.find('-', at == std::string::npos ? 0 : at);
            const int b = at == std::string::npos ? -1 : press_button(spec.substr(0, at));
            if (b < 0 || dash == std::string::npos) {
                std::fprintf(stderr, "midway-verdant: bad --press %s\n", spec.c_str());
                return 2;
            }
            presses.push_back({b, std::strtoull(spec.c_str() + at + 1, nullptr, 10), std::strtoull(spec.c_str() + dash + 1, nullptr, 10)});
        } else if (a == "--capture") {
            cfg.capture_path = need("--capture");
        } else if (a == "--watch") {
            std::string list = need("--watch");
            std::size_t pos = 0;
            while (pos < list.size()) {
                const auto comma = list.find(',', pos);
                const std::string item = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!item.empty()) cfg.watch.push_back(static_cast<mv::PhysAddr>(std::strtoul(item.c_str(), nullptr, 16)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else if (a == "--break-smbus") {
            const std::string spec = need("--break-smbus");
            const auto colon = spec.find(':');
            if (colon == std::string::npos) {
                std::fprintf(stderr, "midway-verdant: --break-smbus wants AA:CC (hex)\n");
                return 2;
            }
            cfg.break_smbus_write = std::make_pair(static_cast<mv::u8>(std::strtoul(spec.substr(0, colon).c_str(), nullptr, 16)),
                                                   static_cast<mv::u8>(std::strtoul(spec.substr(colon + 1).c_str(), nullptr, 16)));
        } else if (a == "--break-mmio") {
            std::string spec = need("--break-mmio");
            mv::MachineConfig::MmioBreak b;
            if (const auto star = spec.find('*'); star != std::string::npos) {
                b.count = static_cast<mv::u32>(std::strtoul(spec.c_str() + star + 1, nullptr, 10));
                spec.resize(star);
            }
            const auto eq = spec.find('=');
            b.address = static_cast<mv::u32>(std::strtoul(spec.substr(0, eq).c_str(), nullptr, 16));
            if (eq != std::string::npos) b.value = static_cast<mv::u32>(std::strtoul(spec.c_str() + eq + 1, nullptr, 16));
            cfg.break_mmio_write = b;
        } else if (a == "--break-eip") {
            const std::string list = need("--break-eip");
            std::size_t pos = 0;
            while (pos < list.size()) {
                const auto comma = list.find(',', pos);
                const std::string item = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!item.empty()) cfg.break_eips.push_back(static_cast<mv::u32>(std::strtoul(item.c_str(), nullptr, 16)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else if (a == "--window") {
            want_window = true;
        } else if (a == "--gpu") {
            cfg.gpu = true;
        } else if (a == "--scanout-pngs") {
            scanout_png_dir = need("--scanout-pngs");
        } else if (a == "--luma-log") {
            luma_log = std::fopen(need("--luma-log"), "w");
            if (!luma_log) {
                std::fprintf(stderr, "midway-verdant: cannot open the luma log\n");
                return 2;
            }
        } else if (a == "--snapshots") {
            snapshot_dir = need("--snapshots");
        } else if (a == "--reset-at") {
            cfg.reset_at_seconds = std::atof(need("--reset-at"));
        } else if (a == "--rtc-time") {
            cfg.rtc_base_time = std::atoll(need("--rtc-time"));
        } else if (a == "--cpu") {
            cfg.cpu = need("--cpu");
        } else if (a == "--oracle") {
            cfg.oracle = true;
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

    if (luma_log || !scanout_png_dir.empty() || !presses.empty()) {
        // Instruments on the scanout at every vblank, in guest time: the picture's mean luma (every fourth pixel of
        // every fourth line) to line the boot animation up with a recording of the real console, and a picture of
        // every fifteenth frame for looking at what a headless run showed.
        if (!scanout_png_dir.empty()) std::filesystem::create_directories(scanout_png_dir);
        machine.nv2a().set_vblank_hook([&machine, luma_log, scanout_png_dir, presses](mv::Cycles now) {
            mv::Nv2aStub& gpu = machine.nv2a();
            if (!presses.empty()) {
                // Scripted input: the buttons held at this vblank, on pad 1.
                mv::usb::PadState s;
                const mv::u64 v = gpu.vblanks();
                for (const Press& p : presses)
                    if (v >= p.from && v <= p.to) {
                        if (p.button < 8) s.analog[static_cast<std::size_t>(p.button)] = 255;
                        else s.buttons = static_cast<mv::u16>(s.buttons | (1u << (p.button - 8)));
                    }
                machine.pad(0).set_state(s);
            }
            if (!luma_log && scanout_png_dir.empty()) return;
            const mv::u32 start = gpu.framebuffer_start();
            const mv::u32 w = gpu.scanout_programmed() ? gpu.scanout_width() : 640u;
            const mv::u32 h = gpu.scanout_programmed() ? gpu.scanout_height() : 480u;
            mv::u32 pitch = gpu.scanout_programmed() ? gpu.scanout_pitch() : 640u * 4u;
            const double t = static_cast<double>(now) / mv::clk::kCpuHz;
            if (start == 0 || w < 64 || h < 64 || pitch < w * 4 || static_cast<mv::u64>(start) + static_cast<mv::u64>(pitch) * h > machine.bus().ram_size()) {
                if (luma_log) std::fprintf(luma_log, "%.4f - %08x\n", t, start);
                return;
            }
            const mv::u8* fb = machine.bus().ram() + start;
            // The video overlay goes over a copy of the frame, as the display shows it.
            static std::vector<mv::u8> composed;
            const auto ov = gpu.overlay();
            const auto lut = gpu.gamma_lut();
            if (ov || lut) {
                composed.resize(static_cast<std::size_t>(w) * h * 4);
                for (mv::u32 y = 0; y < h; ++y) std::memcpy(composed.data() + static_cast<std::size_t>(y) * w * 4, fb + static_cast<std::size_t>(y) * pitch, static_cast<std::size_t>(w) * 4);
                if (lut) mv::apply_gamma(*lut, composed.data(), w * 4, w, h);
                if (ov) mv::draw_overlay(*ov, machine.bus().ram(), machine.bus().ram_size(), composed.data(), w * 4, w, h);
                fb = composed.data();
                pitch = w * 4;
            }
            if (luma_log) {
                double sum = 0;
                mv::u64 n = 0;
                for (mv::u32 y = 0; y < h; y += 4) {
                    const mv::u8* row = fb + static_cast<std::size_t>(y) * pitch;
                    for (mv::u32 x = 0; x < w; x += 4, ++n) sum += 0.114 * row[x * 4] + 0.587 * row[x * 4 + 1] + 0.299 * row[x * 4 + 2];
                }
                std::fprintf(luma_log, "%.4f %.1f %08x\n", t, sum / static_cast<double>(n), start);
            }
            if (!scanout_png_dir.empty() && gpu.vblanks() % 15 == 0) {
                std::vector<mv::u8> rgba(static_cast<std::size_t>(w) * h * 4);
                for (mv::u32 y = 0; y < h; ++y) {
                    const mv::u8* row = fb + static_cast<std::size_t>(y) * pitch;
                    mv::u8* out = rgba.data() + static_cast<std::size_t>(y) * w * 4;
                    for (mv::u32 x = 0; x < w; ++x) {
                        out[x * 4] = row[x * 4 + 2];
                        out[x * 4 + 1] = row[x * 4 + 1];
                        out[x * 4 + 2] = row[x * 4];
                        out[x * 4 + 3] = 255;
                    }
                }
                std::string png_error;
                const auto path = std::filesystem::path(scanout_png_dir) / std::format("scan-{:06}.png", static_cast<unsigned>(t * 1000.0));
                if (!mv::png::write(path, w, h, rgba.data(), w * 4, png_error)) std::fprintf(stderr, "midway-verdant: %s\n", png_error.c_str());
            }
        });
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto duration = std::chrono::nanoseconds(static_cast<long long>(seconds * 1e9));
    mv::Machine::StopReason reason = mv::Machine::StopReason::Duration;
    if (cfg.gpu) std::printf("gpu: live Vulkan renderer on %s\n", machine.gpu_name().c_str());
    if (want_window && !cfg.gpu)
        std::fprintf(stderr, "window: without --gpu the console's graphics are not rendered; the window shows only what the CPU writes into the frame buffer\n");
    if (want_window) {
        mv::ui::ScanoutWindow window;
        if (!window.open(error)) {
            std::fprintf(stderr, "midway-verdant: %s\n", error.c_str());
            return 1;
        }
        if (!snapshot_dir.empty()) window.set_snapshots(snapshot_dir, 2.0);
        // The machine runs on its own io thread; this thread belongs to the window (SDL wants the main thread).
        std::atomic<bool> stop{false};
        std::thread io([&] {
            reason = machine.run(duration, [&](mv::Machine& m) { return stop.load() || (condition && condition(m)); });
            stop = true;
        });
        const unsigned frames = window.run(machine, stop);
        io.join();
        std::printf("window: %u frames presented\n", frames);
    } else {
        reason = machine.run(duration, condition);
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("stopped: %s after %.3f s host, %.3f ms guest\n", reason_name(reason), wall,
                static_cast<double>(mv::ns_from_cycles(machine.scheduler().now())) / 1e6);
    if (machine.cpu()->stop_reason() != mv::CpuBackend::Stop::None)
        std::printf("cpu: %s\n", machine.cpu()->stop_detail().c_str());
    const auto& m = machine.milestones();
    if (machine.boots() > 0) std::printf("reboots: %u (the milestones below are the last boot's)\n", machine.boots());
    if (const auto* r = machine.renderer()) {
        const auto& st = r->stats();
        std::printf("gpu: %llu frames rendered; %llu draws, %llu clears, %llu skipped, %llu pipelines, %llu shaders, %llu textures, %llu write-backs\n",
                    static_cast<unsigned long long>(machine.nv2a().rendered_frames()), static_cast<unsigned long long>(st.draws),
                    static_cast<unsigned long long>(st.clears), static_cast<unsigned long long>(st.skipped), static_cast<unsigned long long>(st.pipelines),
                    static_cast<unsigned long long>(st.shaders), static_cast<unsigned long long>(st.textures_decoded),
                    static_cast<unsigned long long>(st.writebacks));
        std::printf("gpu time: draw %.0f ms (fetch %.0f, textures %.0f, pipelines %.0f, shader compiles %.0f), flush %.0f, write-back %.0f (device copies %.0f)\n",
                    static_cast<double>(st.ns_draw) / 1e6, static_cast<double>(st.ns_fetch) / 1e6, static_cast<double>(st.ns_texture) / 1e6,
                    static_cast<double>(st.ns_pipeline) / 1e6, static_cast<double>(st.ns_vs + st.ns_fs) / 1e6, static_cast<double>(st.ns_flush) / 1e6,
                    static_cast<double>(st.ns_write_back) / 1e6, static_cast<double>(st.ns_write_back_gpu) / 1e6);
        if (st.batch_flushes) std::printf("gpu: %llu mid-frame batch submits (rings full)\n", static_cast<unsigned long long>(st.batch_flushes));
        for (const auto& e : st.errors) std::printf("gpu skipped a draw: %s\n", e.c_str());
    }
    std::printf("milestones: rom hidden %s | smc challenge %s | bootloader %s | kernel %s | power off %s\n", ms(m.rom_hidden).c_str(),
                ms(m.smc_challenge_answered).c_str(), ms(m.bootloader_magic).c_str(), ms(m.kernel_header).c_str(), ms(m.power_off).c_str());
    if (!machine.power_off_reason().empty()) std::printf("power off: %s\n", machine.power_off_reason().c_str());
    if (const auto v = machine.kernel().version())
        std::printf("kernel: %u.%u.%u.%u%s\n", v->major, v->minor, v->build, v->qfe, machine.eeprom_generated() ? " (EEPROM generated from its key)" : "");

    std::string post;
    for (const auto& e : machine.post().history()) post += std::format("{:02x} ", e.code);
    std::printf("post codes (%zu): %s\n", machine.post().history().size(), post.c_str());

    const auto stats = machine.cpu()->stats();
    std::printf("cpu backend: %s\n", std::string(machine.cpu()->name()).c_str());
    if (stats.instructions) {
        const double guest_s = static_cast<double>(mv::ns_from_cycles(stats.cycles)) / 1e9;
        std::printf("cpu p6: %llu instructions, %llu cycles (%.3f ms guest), %.1f M instructions/s host, %.3f guest-s per host-s; halts %llu, interrupts taken %llu\n",
                    static_cast<unsigned long long>(stats.instructions), static_cast<unsigned long long>(stats.cycles), guest_s * 1e3,
                    static_cast<double>(stats.instructions) / wall / 1e6, guest_s / wall, static_cast<unsigned long long>(stats.hlt_exits),
                    static_cast<unsigned long long>(stats.irq_injections));
    }
    if (const std::string r = machine.cpu()->report(); !r.empty()) std::printf("%s\n", r.c_str());
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
    std::printf("smbus transactions: %llu; gpu register reads %llu writes %llu, vblanks %llu, scanout %08x; apu r/w %llu/%llu; aci r/w %llu/%llu\n",
                static_cast<unsigned long long>(machine.smbus().transactions()), static_cast<unsigned long long>(machine.nv2a().reads()),
                static_cast<unsigned long long>(machine.nv2a().writes()), static_cast<unsigned long long>(machine.nv2a().vblanks()),
                machine.nv2a().framebuffer_start(), static_cast<unsigned long long>(machine.apu().reads()),
                static_cast<unsigned long long>(machine.apu().writes()), static_cast<unsigned long long>(machine.aci().reads()),
                static_cast<unsigned long long>(machine.aci().writes()));
    {
        auto& gp = machine.apu().gp();
        const auto& st = machine.apu().gp_stats();
        std::printf("apu frames %llu; irq %s; gp %s, %llu frames run, last %llu cycles, max %llu; dma %llu transfers (%llu bytes); fifo0 cur %08x%s%s%s%s\n",
                    static_cast<unsigned long long>(machine.apu_frames()), machine.apu().irq_level() ? "high" : "low",
                    gp.running() ? "running" : "in reset", static_cast<unsigned long long>(st.frames_run),
                    static_cast<unsigned long long>(st.last_cycles), static_cast<unsigned long long>(st.max_cycles),
                    static_cast<unsigned long long>(gp.dma().transfers()), static_cast<unsigned long long>(gp.dma().bytes_moved()),
                    machine.apu().reg(mv::Apu::kGpFifoRegs + mv::Apu::kFifoCurrent), gp.last_problem().empty() ? "" : " PROBLEM: ",
                    gp.last_problem().c_str(), gp.dma().last_error().empty() ? "" : " DMA: ", gp.dma().last_error().c_str());
        {
            auto& ep = machine.apu().ep();
            const auto& es = machine.apu().ep_stats();
            std::printf("ep %s, %llu frames run, last %llu cycles, max %llu; dma %llu transfers (%llu bytes)%s%s%s%s\n", ep.running() ? "running" : "in reset",
                        static_cast<unsigned long long>(es.frames_run), static_cast<unsigned long long>(es.last_cycles),
                        static_cast<unsigned long long>(es.max_cycles), static_cast<unsigned long long>(ep.dma().transfers()),
                        static_cast<unsigned long long>(ep.dma().bytes_moved()), ep.last_problem().empty() ? "" : " PROBLEM: ", ep.last_problem().c_str(),
                        ep.dma().last_error().empty() ? "" : " DMA: ", ep.dma().last_error().c_str());
            // Bytes each processor moved through its FIFOs, and where each output ring sits in the FIFO space.
            for (const bool is_ep : {false, true}) {
                const mv::u32 regs = is_ep ? mv::Apu::kEpFifoRegs : mv::Apu::kGpFifoRegs;
                std::printf("%s fifos:", is_ep ? "ep" : "gp");
                for (mv::u32 f = 0; f < 4; ++f)
                    std::printf(" out%u %llu B [%x-%x cur %x]", f, static_cast<unsigned long long>(machine.apu().fifo_out_bytes(is_ep, f)),
                                machine.apu().reg(regs + f * mv::Apu::kFifoStride + mv::Apu::kFifoBase), machine.apu().reg(regs + f * mv::Apu::kFifoStride + mv::Apu::kFifoEnd),
                                machine.apu().reg(regs + f * mv::Apu::kFifoStride + mv::Apu::kFifoCurrent));
                std::printf(" in0 %llu B in1 %llu B; table %08x max %u\n", static_cast<unsigned long long>(machine.apu().fifo_in_bytes(is_ep, 0)),
                            static_cast<unsigned long long>(machine.apu().fifo_in_bytes(is_ep, 1)),
                            machine.apu().reg(is_ep ? mv::Apu::kEpFifoTable : mv::Apu::kGpFifoTable), machine.apu().reg(is_ep ? mv::Apu::kEpFifoMaxSge : mv::Apu::kGpFifoMaxSge));
                const mv::u32 table = machine.apu().reg(is_ep ? mv::Apu::kEpFifoTable : mv::Apu::kGpFifoTable);
                const mv::u32 last = std::min<mv::u32>(machine.apu().reg(is_ep ? mv::Apu::kEpFifoMaxSge : mv::Apu::kGpFifoMaxSge), 15);
                std::printf("  %s fifo pages:", is_ep ? "ep" : "gp");
                for (mv::u32 i = 0; i <= last; ++i) {
                    mv::u32 entry = 0;
                    machine.bus().read_ram(table + i * 8, &entry, 4);
                    std::printf(" %08x", entry);
                }
                std::printf("\n");
            }
        }
        std::printf("vp synthesis: %llu voice-frames mixed, peak %.4f of full scale\n", static_cast<unsigned long long>(machine.apu().setup_engine().frames_mixed()),
                    static_cast<double>(machine.apu().setup_engine().peak()));
        std::printf("vp: %llu voices on, %llu retired, %llu front-end traps%s, %llu stream notifications; fectl %08x ists %08x ien %08x\n",
                    static_cast<unsigned long long>(machine.apu().setup_engine().voices_started()),
                    static_cast<unsigned long long>(machine.apu().setup_engine().voices_retired()),
                    static_cast<unsigned long long>(machine.apu().traps()), machine.apu().front_end_trapped() ? " (trapped now)" : "",
                    static_cast<unsigned long long>(machine.apu().setup_engine().notifications()),
                    machine.apu().reg(mv::Apu::kFectl), machine.apu().reg(mv::Apu::kIsts), machine.apu().reg(mv::Apu::kIen));
        std::printf("gp peripherals (reads/writes):");
        for (const auto& [addr, rw] : gp.peripheral_traffic())
            std::printf(" %06x %llu/%llu", addr, static_cast<unsigned long long>(rw.first), static_cast<unsigned long long>(rw.second));
        std::printf("\n");
    }
    std::printf("gpu fifo: %llu methods, %llu semaphore releases, %llu notifies%s\n",
                static_cast<unsigned long long>(machine.nv2a().pushbuffer().methods_seen()),
                static_cast<unsigned long long>(machine.nv2a().semaphore_releases()), static_cast<unsigned long long>(machine.nv2a().notifies()),
                machine.nv2a().pushbuffer().errored() ? (" ERROR: " + machine.nv2a().pushbuffer().last_error()).c_str() : "");
    std::printf("gpu software methods %llu%s\n", static_cast<unsigned long long>(machine.nv2a().software_methods()),
                machine.nv2a().pgraph_stalled() ? " (PGRAPH held now)" : "");
    std::printf("ide: %llu commands, %llu sectors read, %llu written; hdd %s, unlock attempts %llu; dvd packets %llu\n",
                static_cast<unsigned long long>(machine.ide().primary().commands()),
                static_cast<unsigned long long>(machine.ide().primary().sectors_read()),
                static_cast<unsigned long long>(machine.ide().primary().sectors_written()),
                machine.hdd() ? (machine.hdd()->locked() ? "locked" : "unlocked") : "absent",
                static_cast<unsigned long long>(machine.hdd() ? machine.hdd()->unlock_attempts() : 0),
                static_cast<unsigned long long>(static_cast<const mv::AtapiCdrom*>(machine.ide().primary().device(1)) ? static_cast<const mv::AtapiCdrom*>(machine.ide().primary().device(1))->packets() : 0));
    // The drives at the end of the run: a command still in flight shows as BSY or DRQ set.
    for (int d = 0; d < 2; ++d) {
        if (const mv::IdeDevice* dev = machine.ide().primary().device(d)) {
            std::printf("ide %s: status %02x error %02x last command %02x%s\n", d == 0 ? "hdd" : "dvd", dev->tf().status, dev->tf().error,
                        machine.ide().primary().last_command(d), (dev->tf().status & (mv::ata::kStatusBsy | mv::ata::kStatusDrq)) ? "  <-- IN FLIGHT" : "");
        }
    }
    std::printf("ide bus master: command %02x status %02x%s\n", machine.ide().primary().bm_command(), machine.ide().primary().bm_status(),
                machine.ide().primary().dma_active() ? "  <-- DMA ACTIVE" : "");
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
    std::printf("pit: ch0 mode %u reload %u counting %d; late ticks delivered %llu, dropped %llu, backlog %u\n", machine.pit().channel(0).mode,
                machine.pit().channel(0).reload, machine.pit().channel(0).counting ? 1 : 0, static_cast<unsigned long long>(machine.pit().ticks_delayed()),
                static_cast<unsigned long long>(machine.pit().ticks_dropped()), machine.pit().tick_backlog());
    if (machine.capture())
        std::printf("capture: %llu methods, %llu pages, %llu bytes to %s\n", static_cast<unsigned long long>(machine.capture()->methods_written()),
                    static_cast<unsigned long long>(machine.capture()->pages_written()),
                    static_cast<unsigned long long>(machine.capture()->bytes_written()), cfg.capture_path->string().c_str());
    if (!dump_ram.empty()) {
        if (std::FILE* f = std::fopen(dump_ram.c_str(), "wb")) {
            std::fwrite(machine.bus().ram(), 1, machine.bus().ram_size(), f);
            std::fclose(f);
            std::printf("ram written to %s\n", dump_ram.c_str());
        }
    }
    if (trace_file) std::fclose(trace_file);
    if (luma_log) {
        machine.nv2a().set_vblank_hook(nullptr);
        std::fclose(luma_log);
    }
    return 0;
}
