// Midway Verdant as a libretro core: the machine runs on its own threads (the CPU free-running in the fast mode),
// each retro_run pumps it to the next vblank and hands the frontend the scanout (with the video overlay), the
// audio the AC'97 bus master played, and the pads' state from the RetroPads. The console's own files (the MCPX
// boot ROM, the flash image, the EEPROM and the hard-disk image) come from <system>/midway-verdant/.
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/machine.h"
#include "ide/hdd_format.h"
#include "mv/version.h"
#include "libretro/libretro.h"
#include "libretro/retropad_map.h"
#include "nv2a/overlay.h"

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

std::mutex audio_lock;
std::vector<mv::i16> audio;  // interleaved stereo at 48 kHz, filled on the machine's thread
constexpr unsigned kPads = 4;
unsigned pads_plugged = 1;

constexpr double kFps = 60000.0 / 1001.0;
constexpr unsigned kSampleRate = 48000;

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
    mv::Bus& bus = machine->bus();
    if (start == 0 || w < 64 || h < 64 || w > 1920 || h > 1080 || pitch < w * 4 ||
        static_cast<mv::u64>(start) + static_cast<mv::u64>(pitch) * h > bus.ram_size()) {
        std::fill(frame.begin(), frame.end(), 0u);
        return;
    }
    frame_w = w;
    frame_h = h;
    frame.resize(static_cast<std::size_t>(w) * h);
    const mv::u8* fb = bus.ram() + start;
    for (mv::u32 y = 0; y < h; ++y) std::memcpy(frame.data() + static_cast<std::size_t>(y) * w, fb + static_cast<std::size_t>(y) * pitch, w * 4u);
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
    have_rumble = env_cb(RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE, &rumble) && rumble.set_rumble_state;

    mv::MachineConfig cfg;
    cfg.mcpx_rom = files.mcpx;
    cfg.flash_image = files.flash;
    cfg.eeprom_image = files.eeprom;
    cfg.hdd_image = files.hdd;
    cfg.gpu = true;
    cfg.cpu = "kvm";
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
        if (audio.size() > kSampleRate) audio.erase(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(audio.size() - kSampleRate));
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
    machine->start();
    return true;
}

RETRO_API bool retro_load_game_special(unsigned, const retro_game_info*, size_t) { return false; }

RETRO_API void retro_unload_game(void) {
    if (!machine) return;
    machine->finish();
    machine.reset();
}

RETRO_API void retro_reset(void) {
    if (machine) {
        std::lock_guard g(machine->lock());
        machine->request_reset("frontend reset");
    }
}

RETRO_API void retro_run(void) {
    if (!machine) return;
    input_poll_cb();
    for (unsigned p = 0; p < pads_plugged; ++p)
        machine->pad(p).set_state(mv::libretro::map_retropad([p](unsigned device, unsigned index, unsigned id) { return input_state_cb(p, device, index, id); }));
    // One frame: until the next vblank, at most two frames of wall time.
    mv::u64 vblanks;
    {
        std::lock_guard g(machine->lock());
        vblanks = machine->nv2a().vblanks();
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(33'000);
    machine->pump(deadline, [vblanks](mv::Machine& m) { return m.nv2a().vblanks() != vblanks; });
    grab_frame();
    video_cb(frame.data(), frame_w, frame_h, frame_w * 4);
    // A frame's worth of audio (800 stereo samples at 48 kHz / 59.94 Hz), silence where the console made none.
    constexpr std::size_t kFrameSamples = kSampleRate * 1001 / 60000;
    std::vector<mv::i16> out(kFrameSamples * 2, 0);
    {
        std::lock_guard g(audio_lock);
        const std::size_t n = std::min(out.size(), audio.size());
        std::copy(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(n), out.begin());
        audio.erase(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(n));
    }
    audio_batch_cb(out.data(), kFrameSamples);
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
