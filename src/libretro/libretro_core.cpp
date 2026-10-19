// Midway Verdant as a libretro core: the machine runs on its own threads, and each retro_run lets the CPU run
// exactly one frame of the console's time (frame steps on the guest clock, D16: it holds at every vblank until the
// next call), then hands the frontend the scanout (with the video overlay), the audio the AC'97 bus master played
// in that frame, and the pads' state from the RetroPads. The console's own files (the MCPX boot ROM, the flash
// image, the EEPROM and the hard-disk image) come from <system>/midway-verdant/.
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/machine.h"
#include "debug/trace.h"
#include "ide/disk_image.h"
#include "ide/dvd_image.h"
#include "ide/hdd_format.h"
#include "mv/version.h"
#include "libretro/libretro.h"
#include "libretro/retropad_map.h"
#include "nv2a/overlay.h"
#include "nv2a/scanout.h"
#include "nv2a/vulkan/vk_renderer.h"

namespace {

retro_environment_t env_cb;
retro_video_refresh_t video_cb;
retro_audio_sample_t audio_sample_cb;
retro_audio_sample_batch_t audio_batch_cb;
retro_input_poll_t input_poll_cb;
retro_input_state_t input_state_cb;
retro_log_printf_t log_cb;
retro_rumble_interface rumble{};
bool have_rumble = false;

std::unique_ptr<mv::Machine> machine;
std::vector<mv::u32> frame(640 * 480, 0);
mv::u32 frame_w = 640, frame_h = 480;
std::vector<mv::u8> composed;

std::FILE* trace_file = nullptr;  // MV_TRACE_FILE

std::mutex audio_lock;
std::vector<mv::i16> audio;  // interleaved stereo at 48 kHz, filled on the machine's thread
constexpr unsigned kPads = 4;
unsigned pads_plugged = 1;
// Frame pipelining: the CPU is let into the next frame before retro_run returns, up to the vblank it holds at.
bool pipeline = true;
bool cpu_released = false;
mv::u64 released_to = 0;

constexpr double kFps = 60000.0 / 1001.0;
constexpr unsigned kSampleRate = 48000;

// retro_run's host time: how long the frames took, for the unload report.
unsigned long long frames_run = 0, frames_capped = 0;
double frame_ms_sum = 0, frame_ms_max = 0;
// Each frame the CPU waits at its vblank until the picture is taken: how long the frontend's thread took to notice,
// and the grab itself (with pipelining the CPU is released after it).
double hold_wake_ms_sum = 0, hold_grab_ms_sum = 0, hold_wake_ms_max = 0;
// MV_FRAME_LOG=FILE: a line per frame with where its host time went (deltas of the run's accounts, sampled while
// the CPU holds at the vblank), so a slow frame can be traced to the part that made it slow.
struct FrameSample {
    std::chrono::steady_clock::time_point at{};
    mv::u64 guest{0}, handlers{0}, events{0}, held{0}, pusher{0}, draw{0}, fetch{0}, texture{0}, upload{0}, flush{0}, write_back{0};
    mv::u64 setup_engine{0}, dsp_wait{0}, draws{0}, decodes{0}, write_backs{0}, waits{0}, flips{0}, io{0};
};
std::FILE* frame_log = nullptr;
FrameSample frame_last;
unsigned long long frame_log_index = 0;

void log(retro_log_level level, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (log_cb)
        log_cb(level, "[midway-verdant] %s\n", buf);
    else
        std::fprintf(stderr, "[midway-verdant] %s\n", buf);
}

void message(const char* text) {
    retro_message m{text, 600};
    if (env_cb) env_cb(RETRO_ENVIRONMENT_SET_MESSAGE, &m);
    log(RETRO_LOG_ERROR, "%s", text);
}

const char* option(const char* key) {
    retro_variable var{key, nullptr};
    if (env_cb && env_cb(RETRO_ENVIRONMENT_GET_VARIABLE, &var) && var.value) return var.value;
    return nullptr;
}

// The console's files in <system>/midway-verdant: the MCPX ROM is the 512-byte file, the flash image the
// 256 KiB / 512 KiB / 1 MiB one; the EEPROM is eeprom.bin (generated
// from the kernel's key on first use when missing).
struct ConsoleFiles {
    std::filesystem::path mcpx, flash, eeprom, hdd;
};

// The hard disk for a drive slot: Drive 1 is xbox_hdd.raw (or, when that is absent, any raw image of 8 GB or more
// already in the folder), Drives 2 and 3 are xbox_hdd_2.raw and xbox_hdd_3.raw. A slot with no file gets a blank
// retail drive, created here: games boot from it and keep their saves on it.
bool resolve_hdd(const std::filesystem::path& dir, unsigned slot, std::filesystem::path& out, std::string& error) {
    const std::filesystem::path named = dir / (slot <= 1 ? std::string("xbox_hdd.raw") : std::format("xbox_hdd_{}.raw", slot));
    std::error_code ec;
    if (std::filesystem::exists(named, ec)) {
        out = named;
        return true;
    }
    if (slot <= 1) {
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            if (!e.is_regular_file(ec)) continue;
            const std::string name = e.path().filename().string();
            if (e.file_size(ec) >= 8'000'000'000ull && name.find(".qcow2") == std::string::npos && name.rfind("xbox_hdd_", 0) != 0) {
                out = e.path();
                return true;
            }
        }
    }
    if (!mv::create_blank_hdd(named, static_cast<mv::u32>(std::time(nullptr)), error)) return false;
    log(RETRO_LOG_INFO, "created a blank hard disk for drive %u: %s", slot, named.string().c_str());
    out = named;
    return true;
}

bool find_console_files(const std::filesystem::path& dir, ConsoleFiles& out, std::string& error) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        error = "Put the console's files in " + dir.string() + " (MCPX ROM, flash image, hard-disk image)";
        return false;
    }
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        const auto size = e.file_size(ec);
        if (size == 512 && out.mcpx.empty()) out.mcpx = e.path();
        else if ((size == 256 * 1024 || size == 512 * 1024 || size == 1024 * 1024) && out.flash.empty()) out.flash = e.path();
    }
    out.eeprom = dir / "eeprom.bin";
    if (out.mcpx.empty()) error = "No MCPX boot ROM (a 512-byte file) in " + dir.string();
    else if (out.flash.empty()) error = "No flash image (256 KiB, 512 KiB or 1 MiB) in " + dir.string();
    return error.empty();
}

// The scanout as the display shows it, with the video overlay, into `frame`.
void grab_frame() {
    std::lock_guard g(machine->lock());
    mv::Nv2aStub& gpu = machine->nv2a();
    const mv::u32 start = gpu.framebuffer_start();
    const mv::u32 w = gpu.scanout_programmed() ? gpu.scanout_width() : 640u;
    const mv::u32 h = gpu.scanout_programmed() ? gpu.scanout_height() : 480u;
    const mv::u32 pitch = gpu.scanout_programmed() ? gpu.scanout_pitch() : 640u * 4u;
    const mv::u32 bpp = gpu.scanout_bytes_per_pixel();
    mv::Bus& bus = machine->bus();
    if (start == 0 || w < 64 || h < 64 || w > 1920 || h > 1080 || pitch < w * bpp ||
        static_cast<mv::u64>(start) + static_cast<mv::u64>(pitch) * h > bus.ram_size()) {
        std::fill(frame.begin(), frame.end(), 0u);
        return;
    }
    frame_w = w;
    frame_h = h;
    frame.resize(static_cast<std::size_t>(w) * h);
    const mv::u8* fb = bus.ram() + start;
    mv::scanout_to_xrgb(fb, pitch, bpp, gpu.scanout_rgb565(), w, h, reinterpret_cast<mv::u8*>(frame.data()));
    if (const auto lut = gpu.gamma_lut()) mv::apply_gamma(*lut, reinterpret_cast<mv::u8*>(frame.data()), w * 4, w, h);
    if (const auto ov = gpu.overlay())
        mv::draw_overlay(*ov, bus.ram(), bus.ram_size(), reinterpret_cast<mv::u8*>(frame.data()), w * 4, w, h);
    // The scanout ignores the alpha byte; the frontend's XRGB8888 wants it clear of meaning too.
    for (auto& px : frame) px &= 0x00FFFFFFu;
}

}  // namespace

extern "C" {

RETRO_API unsigned retro_api_version(void) { return RETRO_API_VERSION; }

RETRO_API void retro_set_environment(retro_environment_t cb) {
    env_cb = cb;
    bool no_game = true;
    cb(RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME, &no_game);  // no disc: the console boots to its dashboard
    static const retro_variable vars[] = {
        {"midway_verdant_pads", "Controllers plugged in; 1|2|3|4"},
        {"midway_verdant_drive", "Hard drive; Drive 1|Drive 2|Drive 3"},
        {"midway_verdant_pipeline", "Run the next frame while the frontend shows this one (faster, up to a frame more input lag); enabled|disabled"},
        {"midway_verdant_cpu", "CPU; fast (hardware virtualization)|accurate (cycle-counted, recompiled)"},
        {nullptr, nullptr},
    };
    cb(RETRO_ENVIRONMENT_SET_VARIABLES, const_cast<retro_variable*>(vars));
    static const retro_controller_description pad{"Xbox Controller", RETRO_DEVICE_JOYPAD};
    static const retro_controller_info ports[] = {{&pad, 1}, {&pad, 1}, {&pad, 1}, {&pad, 1}, {nullptr, 0}};
    cb(RETRO_ENVIRONMENT_SET_CONTROLLER_INFO, const_cast<retro_controller_info*>(ports));
    retro_log_callback logging{};
    if (cb(RETRO_ENVIRONMENT_GET_LOG_INTERFACE, &logging)) log_cb = logging.log;
}

RETRO_API void retro_set_video_refresh(retro_video_refresh_t cb) { video_cb = cb; }
RETRO_API void retro_set_audio_sample(retro_audio_sample_t cb) { audio_sample_cb = cb; }
RETRO_API void retro_set_audio_sample_batch(retro_audio_sample_batch_t cb) { audio_batch_cb = cb; }
RETRO_API void retro_set_input_poll(retro_input_poll_t cb) { input_poll_cb = cb; }
RETRO_API void retro_set_input_state(retro_input_state_t cb) { input_state_cb = cb; }

RETRO_API void retro_init(void) {}
RETRO_API void retro_deinit(void) {}

RETRO_API void retro_get_system_info(retro_system_info* info) {
    std::memset(info, 0, sizeof *info);
    info->library_name = "Midway Verdant";
    info->library_version = MV_VERSION_STRING;
    info->valid_extensions = "iso|xiso";
    info->need_fullpath = true;
    info->block_extract = true;
}

RETRO_API void retro_get_system_av_info(retro_system_av_info* info) {
    std::memset(info, 0, sizeof *info);
    info->geometry.base_width = 640;
    info->geometry.base_height = 480;
    info->geometry.max_width = 1920;
    info->geometry.max_height = 1080;
    info->geometry.aspect_ratio = 4.0f / 3.0f;
    info->timing.fps = kFps;
    info->timing.sample_rate = kSampleRate;
}

RETRO_API void retro_set_controller_port_device(unsigned, unsigned) {}

RETRO_API bool retro_load_game(const retro_game_info* game) {
    retro_pixel_format fmt = RETRO_PIXEL_FORMAT_XRGB8888;
    if (!env_cb(RETRO_ENVIRONMENT_SET_PIXEL_FORMAT, &fmt)) {
        log(RETRO_LOG_ERROR, "the frontend has no XRGB8888");
        return false;
    }
    const char* sys = nullptr;
    if (!env_cb(RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY, &sys) || !sys) {
        message("Midway Verdant needs a system directory");
        return false;
    }
    ConsoleFiles files;
    std::string error;
    if (!find_console_files(std::filesystem::path(sys) / "midway-verdant", files, error)) {
        message(error.c_str());
        return false;
    }
    unsigned slot = 1;
    if (const char* d = option("midway_verdant_drive"); d && *d) slot = std::clamp(static_cast<unsigned>(d[std::strlen(d) - 1] - '0'), 1u, 3u);
    if (!resolve_hdd(files.mcpx.parent_path(), slot, files.hdd, error)) {
        message(("Midway Verdant: " + error).c_str());
        return false;
    }
    if (const char* p = option("midway_verdant_pads")) pads_plugged = std::clamp(static_cast<unsigned>(std::atoi(p)), 1u, kPads);
    if (const char* p = option("midway_verdant_pipeline")) pipeline = std::strcmp(p, "disabled") != 0;
    cpu_released = false;
    have_rumble = env_cb(RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE, &rumble) && rumble.set_rumble_state;

    // Instruments for a core run under a frontend: MV_TRACE=apu,boot,... (the shell's --trace list) into
    // MV_TRACE_FILE (default stderr).
    if (const char* f = std::getenv("MV_FRAME_LOG"); f && *f && !frame_log) {
        frame_log = std::fopen(f, "w");
        frame_log_index = 0;
        if (frame_log) std::fprintf(frame_log, "frame\tperiod\twait\tguest\thandlers\tevents\theld\tpusher\tdraw\tfetch\ttexture\tupload\tflush\twriteback\tsetup\tdspwait\tdraws\tdecodes\twritebacks\twaits\tflips\tio\n");
    }
    if (const char* t = std::getenv("MV_TRACE"); t && *t) {
        mv::u32 mask = 0;
        if (mv::trace::parse_list(t, mask)) {
            mv::trace::enable(mask);
            if (const char* f = std::getenv("MV_TRACE_FILE"); f && *f && !trace_file) {
                trace_file = std::fopen(f, "w");
                if (trace_file) mv::trace::set_sink(trace_file);
            }
            log(RETRO_LOG_INFO, "tracing %s to %s", t, trace_file ? std::getenv("MV_TRACE_FILE") : "stderr");
        } else {
            log(RETRO_LOG_WARN, "MV_TRACE: unknown category in %s", t);
        }
    }
    mv::MachineConfig cfg;
    cfg.mcpx_rom = files.mcpx;
    cfg.flash_image = files.flash;
    cfg.eeprom_image = files.eeprom;
    cfg.hdd_image = files.hdd;
    cfg.gpu = true;
    // Compiled shaders persist beside the console's files, so a scene seen in an earlier session does not stall again.
    cfg.shader_cache_dir = files.eeprom.parent_path() / "shader-cache";
    cfg.cpu = "kvm";
    if (const char* c = option("midway_verdant_cpu"); c && std::strncmp(c, "accurate", 8) == 0) cfg.cpu = "p6";
    cfg.frame_step = true;
    // Instrument: MV_RTC_TIME=UNIX fixes the clock's start, so two accurate-core runs of a title can be compared.
    if (const char* t = std::getenv("MV_RTC_TIME"); t && *t) cfg.rtc_base_time = std::strtoll(t, nullptr, 10);
    cfg.pads = pads_plugged;
    if (game && game->path && *game->path) cfg.dvd_image = game->path;
    machine = std::make_unique<mv::Machine>(cfg);
    if (!machine->init(error)) {
        message(("Midway Verdant: " + error).c_str());
        machine.reset();
        return false;
    }
    machine->aci().set_sample_sink([](mv::Aci::Channel c, const mv::u8* data, mv::u32 bytes) {
        if (c != mv::Aci::PcmOut) return;
        std::lock_guard g(audio_lock);
        const std::size_t n = bytes / 2;
        const std::size_t at = audio.size();
        audio.resize(at + n);
        std::memcpy(audio.data() + at, data, n * 2);
        // A frontend that stops calling retro_run must not build up latency: keep at most a quarter second.
        constexpr std::size_t kMaxBacklog = kSampleRate / 4 * 2;
        if (audio.size() > kMaxBacklog) audio.erase(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(audio.size() - kMaxBacklog));
    });
    if (have_rumble) {
        for (unsigned p = 0; p < pads_plugged; ++p)
            machine->pad(p).set_rumble([p](mv::u16 left, mv::u16 right) {
                rumble.set_rumble_state(p, RETRO_RUMBLE_STRONG, left);
                rumble.set_rumble_state(p, RETRO_RUMBLE_WEAK, right);
            });
    }
    log(RETRO_LOG_INFO, "booting %s with %u pad(s); files from %s", cfg.dvd_image ? cfg.dvd_image->string().c_str() : "no disc", pads_plugged,
        files.mcpx.parent_path().string().c_str());
    log(RETRO_LOG_INFO, "timeline: %s; one console frame per retro_run", machine->cpu()->clock_name().c_str());
    frames_run = frames_capped = 0;
    frame_ms_sum = frame_ms_max = 0;
    hold_wake_ms_sum = hold_grab_ms_sum = hold_wake_ms_max = 0;
    machine->start();
    return true;
}

RETRO_API bool retro_load_game_special(unsigned, const retro_game_info*, size_t) { return false; }

RETRO_API void retro_unload_game(void) {
    if (!machine) return;
    // What the game did with the pads: polls of the report endpoint, reports delivered, class requests.
    for (unsigned p = 0; p < pads_plugged; ++p)
        log(RETRO_LOG_INFO, "pad %u: %llu polls, %llu reports, %llu class requests; usb frames %llu/%llu, transfers %llu/%llu", p + 1,
            static_cast<unsigned long long>(machine->pad(p).polls()), static_cast<unsigned long long>(machine->pad(p).reports()),
            static_cast<unsigned long long>(machine->pad(p).control_requests()), static_cast<unsigned long long>(machine->usb0().stats().frames),
            static_cast<unsigned long long>(machine->usb1().stats().frames), static_cast<unsigned long long>(machine->usb0().stats().transfers),
            static_cast<unsigned long long>(machine->usb1().stats().transfers));
    // What the console did: its own time, the frames it rendered and the voices its sound started (a cue the console
    // never reached shows as a missing voice).
    {
        std::lock_guard g(machine->lock());
        std::string voices;
        for (const auto& s : machine->apu().setup_engine().handles_started())
            voices += std::format(" {:x}@{:.3f}", s.handle, static_cast<double>(s.at) / mv::clk::kCpuHz);
        log(RETRO_LOG_INFO, "ran %.1f s of console time; %llu frames rendered, %llu displayed; voices started:%s", static_cast<double>(machine->scheduler().now()) / mv::clk::kCpuHz,
            static_cast<unsigned long long>(machine->nv2a().semaphore_releases()), static_cast<unsigned long long>(machine->nv2a().display_flips()), voices.empty() ? " none" : voices.c_str());
        log(RETRO_LOG_INFO, "retro_run: %llu frames, host time mean %.2f ms, max %.1f ms%s; pit late ticks delivered %llu, dropped %llu", frames_run,
            frames_run ? frame_ms_sum / static_cast<double>(frames_run) : 0.0, frame_ms_max,
            frames_capped ? std::format(", {} cut at 1 s without a vblank", frames_capped).c_str() : "",
            static_cast<unsigned long long>(machine->pit().ticks_delayed()), static_cast<unsigned long long>(machine->pit().ticks_dropped()));
        log(RETRO_LOG_INFO, "retro_run: the CPU held at each vblank for the picture: noticed after %.3f ms (max %.1f), grabbed in %.3f ms (means)",
            frames_run ? hold_wake_ms_sum / static_cast<double>(frames_run) : 0.0, hold_wake_ms_max, frames_run ? hold_grab_ms_sum / static_cast<double>(frames_run) : 0.0);
        log(RETRO_LOG_INFO, "gpu: busiest register reads:%s", machine->nv2a().busiest_reads(6).c_str());
        log(RETRO_LOG_INFO, "gpu: %s; %s", machine->nv2a().harvest_report().c_str(), machine->cpu()->selective_tracking() ? "only the pages under textures watched" : "every page logged");
        if (machine->nv2a().fifo_threaded()) log(RETRO_LOG_INFO, "gpu: register reads while the FIFO thread had work:%s", machine->nv2a().busiest_reads(10, true).c_str());
        log(RETRO_LOG_INFO, "gpu: a FIFO thread could have run %.3f s of the pusher's %.3f s alongside the guest (%llu points where the guest needed its result)",
            static_cast<double>(machine->nv2a().pusher_overlap_ns()) / 1e9, static_cast<double>(machine->nv2a().pusher_wall_ns()) / 1e9,
            static_cast<unsigned long long>(machine->nv2a().pusher_sync_points()));
        if (machine->nv2a().fifo_threaded())
            log(RETRO_LOG_INFO, "gpu: the FIFO thread ran the pusher; the CPU waited for it %llu times at the lead (%.3f s) and %llu times at a vblank (%.3f s)",
                static_cast<unsigned long long>(machine->nv2a().fifo_lead_waits()), static_cast<double>(machine->nv2a().fifo_lead_wait_ns()) / 1e9,
                static_cast<unsigned long long>(machine->nv2a().fifo_vblank_waits()), static_cast<double>(machine->nv2a().fifo_vblank_wait_ns()) / 1e9);
        if (machine->nv2a().fifo_threaded()) log(RETRO_LOG_INFO, "gpu: the CPU side waited for the FIFO thread: %s", machine->nv2a().fifo_wait_report().c_str());
        log(RETRO_LOG_INFO, "gpu: what ended those runs' overlap (pusher time not hidden):%s", machine->nv2a().sync_reasons(8).c_str());
        log(RETRO_LOG_INFO, "apu: %llu front-end memory fences written, %llu front-end traps raised", static_cast<unsigned long long>(machine->apu().fence_writes()),
            static_cast<unsigned long long>(machine->apu().traps()));
        // The renderer's share of the pusher: draws and their parts, submits and fence waits, copies back to RAM.
        if (const auto* r = machine->nv2a().renderer()) {
            const auto& st = r->stats();
            const auto ms = [](mv::u64 ns) { return static_cast<double>(ns) / 1e6; };
            log(RETRO_LOG_INFO, "gpu time: %llu draws %.0f ms (fetch %.0f, indices %.0f, textures %.0f, pipelines %.0f, shader compiles %.0f, uploads %.0f, descriptors %.0f, record %.0f); %llu submits + %llu waits %.0f ms; %llu write-backs %.0f ms (device copies %.0f); %llu textures decoded (%llu kept: pages written, bytes unchanged), %llu shaders from the disk cache",
                static_cast<unsigned long long>(st.draws), ms(st.ns_draw), ms(st.ns_fetch), ms(st.ns_indices), ms(st.ns_texture), ms(st.ns_pipeline), ms(st.ns_vs + st.ns_fs),
                ms(st.ns_upload), ms(st.ns_descriptors), ms(st.ns_record), static_cast<unsigned long long>(st.flushes), static_cast<unsigned long long>(st.waits),
                ms(st.ns_flush), static_cast<unsigned long long>(st.writebacks), ms(st.ns_write_back), ms(st.ns_write_back_gpu),
                static_cast<unsigned long long>(st.textures_decoded), static_cast<unsigned long long>(st.textures_unchanged), static_cast<unsigned long long>(st.shaders_from_disk));
            // Draws the renderer could not draw, and why (the first of each distinct reason).
            if (st.skipped) log(RETRO_LOG_INFO, "gpu: %llu draws skipped", static_cast<unsigned long long>(st.skipped));
            log(RETRO_LOG_INFO, "gpu: write-backs by cause: frame end %llu, texture %llu, scanout %llu, 2D/range %llu, relayout %llu, idle vblank %llu; %llu targets sampled in place, %llu snapshots for draws sampling their own target, %llu depth buffers converted to colour, %llu targets the CPU wrote reloaded, %llu depth buffers handed between targets, %llu loaded from RAM; %llu targets evicted by an overlapping one, %llu retired unused, %llu rectangles of a region copied for a texture",
                static_cast<unsigned long long>(st.writebacks_by[0]), static_cast<unsigned long long>(st.writebacks_by[1]), static_cast<unsigned long long>(st.writebacks_by[2]),
                static_cast<unsigned long long>(st.writebacks_by[3]), static_cast<unsigned long long>(st.writebacks_by[4]), static_cast<unsigned long long>(st.writebacks_by[5]),
                static_cast<unsigned long long>(st.surface_samples), static_cast<unsigned long long>(st.snapshots), static_cast<unsigned long long>(st.zeta_conversions), static_cast<unsigned long long>(st.cpu_written_targets),
                static_cast<unsigned long long>(st.depth_handovers), static_cast<unsigned long long>(st.depth_uploads), static_cast<unsigned long long>(st.evicted), static_cast<unsigned long long>(st.retired_unused), static_cast<unsigned long long>(st.subrect_copies));
            if (st.surface_reloads) log(RETRO_LOG_INFO, "gpu: %llu render targets reloaded from RAM after a 2D write", static_cast<unsigned long long>(st.surface_reloads));
            for (const auto& e : st.errors) log(RETRO_LOG_INFO, "gpu: renderer note: %s", e.c_str());
        }
        log(RETRO_LOG_INFO, "device time: the frontend's thread fired events for %.3f s; the gpu pusher took %.3f s on the CPU thread",
            static_cast<double>(machine->io_event_ns()) / 1e9, static_cast<double>(machine->nv2a().pusher_wall_ns()) / 1e9);
        // Where the event time went: host time inside each device event (both threads), and the APU frame's parts.
        {
            auto costs = machine->scheduler().event_costs();
            std::ranges::sort(costs, std::greater{}, &mv::Scheduler::EventCost::host_ns);
            std::string by_event;
            for (const auto& c : costs)
                by_event += std::format(" {} {:.3f} s/{}", c.name, static_cast<double>(c.host_ns) / 1e9, c.fired);
            const auto& apu = machine->apu();
            log(RETRO_LOG_INFO, "event host time (s/fired):%s; apu frame = setup engine %.3f s + gp %.3f s + ep %.3f s; the machine waited %.3f s at %llu points for the processors' thread",
                by_event.c_str(), static_cast<double>(apu.setup_engine_host_ns()) / 1e9, static_cast<double>(apu.gp_stats().host_ns) / 1e9,
                static_cast<double>(apu.ep_stats().host_ns) / 1e9, static_cast<double>(apu.dsp_wait_ns()) / 1e9, static_cast<unsigned long long>(apu.dsp_waits()));
            for (std::string r = apu.pc_histogram_report(24); !r.empty();) {
                const auto nl = r.find('\n');
                log(RETRO_LOG_INFO, "%s", r.substr(0, nl).c_str());
                r = nl == std::string::npos ? std::string() : r.substr(nl + 1);
            }
        }
    }
    if (frame_log) {
        std::fclose(frame_log);
        frame_log = nullptr;
    }
    machine->finish();
    if (trace_file) {
        mv::trace::set_sink(nullptr);  // back to stderr, whose buffering a file sink's setvbuf must not touch
        std::fclose(trace_file);
        trace_file = nullptr;
    }
    // The CPU's own report (the guest clock's accounts), once its thread has stopped.
    for (std::string r = machine->cpu()->report(); !r.empty();) {
        const auto nl = r.find('\n');
        log(RETRO_LOG_INFO, "%s", r.substr(0, nl).c_str());
        r = nl == std::string::npos ? std::string() : r.substr(nl + 1);
    }
    machine.reset();
}

RETRO_API void retro_reset(void) {
    if (machine) {
        std::lock_guard g(machine->lock());
        machine->request_reset("frontend reset");
    }
}

// The run's accounts now (the machine lock held, the CPU at its vblank), and a frame-log line of the change since the last.
static FrameSample frame_sample() {
    FrameSample s;
    s.at = std::chrono::steady_clock::now();
    const auto h = machine->cpu()->host_accounts();
    s.guest = h.in_guest_ns, s.handlers = h.handler_ns, s.events = h.event_ns, s.held = h.pause_ns;
    s.pusher = machine->nv2a().pusher_wall_ns();
    if (const auto* r = machine->nv2a().renderer()) {
        const auto& st = r->stats();
        s.draw = st.ns_draw, s.fetch = st.ns_fetch, s.texture = st.ns_texture, s.upload = st.ns_upload, s.flush = st.ns_flush;
        s.write_back = st.ns_write_back, s.draws = st.draws, s.decodes = st.textures_decoded, s.write_backs = st.writebacks, s.waits = st.waits;
    }
    s.setup_engine = machine->apu().setup_engine_host_ns();
    s.dsp_wait = machine->apu().dsp_wait_ns();
    s.flips = machine->nv2a().display_flips();
    s.io = mv::DvdImage::host_read_ns() + mv::DiskImage::host_read_ns();
    return s;
}

static void frame_log_line(double wait_ms) {
    const FrameSample s = frame_sample();
    const auto ms = [](mv::u64 a, mv::u64 b) { return static_cast<double>(a - b) / 1e6; };
    const double period = frame_log_index ? std::chrono::duration<double, std::milli>(s.at - frame_last.at).count() : 0.0;
    const std::string line = std::format("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}\t{}\t{:.2f}\n",
        frame_log_index, period, wait_ms, ms(s.guest, frame_last.guest), ms(s.handlers, frame_last.handlers), ms(s.events, frame_last.events),
        ms(s.held, frame_last.held), ms(s.pusher, frame_last.pusher), ms(s.draw, frame_last.draw), ms(s.fetch, frame_last.fetch),
        ms(s.texture, frame_last.texture), ms(s.upload, frame_last.upload), ms(s.flush, frame_last.flush), ms(s.write_back, frame_last.write_back),
        ms(s.setup_engine, frame_last.setup_engine), ms(s.dsp_wait, frame_last.dsp_wait), s.draws - frame_last.draws, s.decodes - frame_last.decodes,
        s.write_backs - frame_last.write_backs, s.waits - frame_last.waits, s.flips - frame_last.flips, ms(s.io, frame_last.io));
    std::fputs(line.c_str(), frame_log);
    frame_last = s;
    ++frame_log_index;
}

RETRO_API void retro_run(void) {
    if (!machine) return;
    input_poll_cb();
    {
        // The guest sees the new state where the CPU holds for a frame (with pipelining, the end of the frame it runs).
        std::lock_guard g(machine->lock());
        for (unsigned p = 0; p < pads_plugged; ++p)
            machine->set_pad_input(p, mv::libretro::map_retropad([p](unsigned device, unsigned index, unsigned id) { return input_state_cb(p, device, index, id); }));
    }
    // One frame of the console's time: the CPU runs to the next vblank and holds there, however long the host takes
    // (a hung guest or a stalled host returns after a second without the vblank, and the report counts it). With
    // pipelining the CPU was let into this frame when the last call returned, so it may be done already.
    mv::u64 target;
    {
        std::lock_guard g(machine->lock());
        target = cpu_released ? released_to : machine->nv2a().vblanks() + 1;
        machine->set_frame_limit(target);
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (machine->pump(t0 + std::chrono::seconds(1), [target](mv::Machine& m) { return m.nv2a().vblanks() >= target; }) ==
        mv::Machine::StopReason::Duration) {
        if (frames_capped++ == 0) log(RETRO_LOG_WARN, "a frame took over a second of host time; returned without its vblank");
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ++frames_run;
    frame_ms_sum += ms;
    frame_ms_max = std::max(frame_ms_max, ms);
    {
        std::lock_guard g(machine->lock());
        if (machine->cpu_paused() && machine->frame_hold_at() > t0) {
            const double wake = std::chrono::duration<double, std::milli>(t1 - machine->frame_hold_at()).count();
            hold_wake_ms_sum += wake;
            hold_wake_ms_max = std::max(hold_wake_ms_max, wake);
        }
    }
    if (frame_log) {
        std::lock_guard g(machine->lock());
        frame_log_line(ms);
    }
    grab_frame();
    hold_grab_ms_sum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    if (pipeline) {
        // The picture is taken while the CPU holds at the vblank; now it runs the next frame while the frontend
        // shows this one and the input for it arrives at the next call.
        std::lock_guard g(machine->lock());
        released_to = machine->nv2a().vblanks() + 1;
        machine->set_frame_limit(released_to);
        cpu_released = true;
    }
    video_cb(frame.data(), frame_w, frame_h, frame_w * 4);
    // The audio the console played since the last frame, as it played it: the AC'97 output runs on the console's
    // own clock, so a frame carries about 800 stereo samples, a few more or fewer as the reads fall. Padding a short
    // frame with silence would put a click in the sound, so the frontend's queue absorbs the jitter instead. While
    // the console's PCM output is halted (before the kernel starts it) a frame of silence keeps the stream going.
    std::vector<mv::i16> out;
    {
        std::lock_guard g(audio_lock);
        const auto whole = static_cast<std::ptrdiff_t>(audio.size() & ~std::size_t{1});  // whole stereo frames only
        out.assign(audio.begin(), audio.begin() + whole);
        audio.erase(audio.begin(), audio.begin() + whole);
    }
    if (out.empty()) {
        bool halted;
        {
            std::lock_guard g(machine->lock());
            halted = machine->aci().channel(mv::Aci::PcmOut).sr & mv::Aci::kSrDch;
        }
        if (halted) out.assign(static_cast<std::size_t>(kSampleRate * 1001 / 60000) * 2, 0);
    }
    if (!out.empty()) audio_batch_cb(out.data(), out.size() / 2);
}

RETRO_API size_t retro_serialize_size(void) { return 0; }
RETRO_API bool retro_serialize(void*, size_t) { return false; }
RETRO_API bool retro_unserialize(const void*, size_t) { return false; }
RETRO_API void retro_cheat_reset(void) {}
RETRO_API void retro_cheat_set(unsigned, bool, const char*) {}
RETRO_API unsigned retro_get_region(void) { return RETRO_REGION_NTSC; }
RETRO_API void* retro_get_memory_data(unsigned) { return nullptr; }
RETRO_API size_t retro_get_memory_size(unsigned) { return 0; }

}  // extern "C"
