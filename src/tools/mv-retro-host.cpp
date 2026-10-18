// mv-retro-host: a headless libretro frontend for testing the core the way Emutastic drives it. It loads the core
// library, answers the environment calls (system directory, pixel format, options, log), runs N frames, saves
// pictures of chosen frames and holds RetroPad buttons over chosen frame ranges.
//   mv-retro-host CORE.so SYSTEM_DIR [GAME] --frames N [--png-every K DIR] [--press BUTTON@FROM-TO ...]
// Buttons: a b x y l r l2 r2 start select up down left right.
#include <dlfcn.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>
#include <vector>

#include "debug/png.h"
#include "libretro/libretro.h"

namespace {

std::string system_dir;
unsigned pixel_format = RETRO_PIXEL_FORMAT_0RGB1555;
std::vector<unsigned char> last_frame;
unsigned last_w = 0, last_h = 0;
std::size_t last_pitch = 0;
unsigned long long frames_seen = 0, audio_frames = 0, audio_nonzero = 0;

struct Press {
    unsigned id;
    unsigned from, to;
};
std::vector<Press> presses;
unsigned current_frame = 0;

void core_log(retro_log_level, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
}

bool environment(unsigned cmd, void* data) {
    switch (cmd) {
        case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
        case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
            *static_cast<const char**>(data) = system_dir.c_str();
            return true;
        case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
            pixel_format = *static_cast<const unsigned*>(data);
            return pixel_format == RETRO_PIXEL_FORMAT_XRGB8888;
        case RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
            static_cast<retro_log_callback*>(data)->log = core_log;
            return true;
        case RETRO_ENVIRONMENT_SET_MESSAGE:
            std::fprintf(stderr, "core message: %s\n", static_cast<const retro_message*>(data)->msg);
            return true;
        case RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME:
        case RETRO_ENVIRONMENT_SET_VARIABLES:
        case RETRO_ENVIRONMENT_SET_CONTROLLER_INFO:
            return true;
        case RETRO_ENVIRONMENT_GET_VARIABLE:
            static_cast<retro_variable*>(data)->value = nullptr;  // the core's defaults
            return false;
        default:
            return false;
    }
}

void video(const void* data, unsigned w, unsigned h, size_t pitch) {
    ++frames_seen;
    if (!data) return;
    last_w = w;
    last_h = h;
    last_pitch = pitch;
    last_frame.assign(static_cast<const unsigned char*>(data), static_cast<const unsigned char*>(data) + pitch * h);
}

size_t audio_batch(const int16_t* data, size_t frames) {
    audio_frames += frames;
    for (size_t i = 0; i < frames * 2; ++i)
        if (data[i]) ++audio_nonzero;
    return frames;
}
void audio_sample(int16_t, int16_t) {}
void input_poll() {}
int16_t input_state(unsigned port, unsigned device, unsigned, unsigned id) {
    if (port != 0 || device != RETRO_DEVICE_JOYPAD) return 0;
    for (const Press& p : presses)
        if (p.id == id && current_frame >= p.from && current_frame <= p.to) return 1;
    return 0;
}

bool save_png(const std::filesystem::path& path) {
    if (last_frame.empty()) return false;
    std::vector<unsigned char> rgba(static_cast<std::size_t>(last_w) * last_h * 4);
    for (unsigned y = 0; y < last_h; ++y)
        for (unsigned x = 0; x < last_w; ++x) {
            const unsigned char* s = &last_frame[y * last_pitch + x * 4];
            unsigned char* d = &rgba[(static_cast<std::size_t>(y) * last_w + x) * 4];
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = 255;
        }
    std::string error;
    return mv::png::write(path, last_w, last_h, rgba.data(), last_w * 4, error);
}

unsigned button_id(const std::string& name) {
    static const std::pair<const char*, unsigned> kNames[] = {
        {"a", RETRO_DEVICE_ID_JOYPAD_A}, {"b", RETRO_DEVICE_ID_JOYPAD_B}, {"x", RETRO_DEVICE_ID_JOYPAD_X}, {"y", RETRO_DEVICE_ID_JOYPAD_Y},
        {"l", RETRO_DEVICE_ID_JOYPAD_L}, {"r", RETRO_DEVICE_ID_JOYPAD_R}, {"l2", RETRO_DEVICE_ID_JOYPAD_L2}, {"r2", RETRO_DEVICE_ID_JOYPAD_R2},
        {"start", RETRO_DEVICE_ID_JOYPAD_START}, {"select", RETRO_DEVICE_ID_JOYPAD_SELECT}, {"up", RETRO_DEVICE_ID_JOYPAD_UP},
        {"down", RETRO_DEVICE_ID_JOYPAD_DOWN}, {"left", RETRO_DEVICE_ID_JOYPAD_LEFT}, {"right", RETRO_DEVICE_ID_JOYPAD_RIGHT}};
    for (const auto& [n, id] : kNames)
        if (name == n) return id;
    return ~0u;
}

template <typename T>
T sym(void* lib, const char* name) {
    void* p = dlsym(lib, name);
    if (!p) {
        std::fprintf(stderr, "mv-retro-host: the core has no %s\n", name);
        std::exit(1);
    }
    return reinterpret_cast<T>(p);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: mv-retro-host CORE.so SYSTEM_DIR [GAME] --frames N [--png-every K DIR] [--press BUTTON@FROM-TO ...]\n");
        return 2;
    }
    const char* core_path = argv[1];
    system_dir = argv[2];
    std::string game;
    unsigned frames = 600, png_every = 0;
    std::filesystem::path png_dir;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--frames" && i + 1 < argc) frames = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (a == "--png-every" && i + 2 < argc) {
            png_every = static_cast<unsigned>(std::atoi(argv[++i]));
            png_dir = argv[++i];
        } else if (a == "--press" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const auto at = spec.find('@'), dash = spec.find('-', at);
            if (at == std::string::npos || dash == std::string::npos || button_id(spec.substr(0, at)) == ~0u) {
                std::fprintf(stderr, "mv-retro-host: bad --press %s\n", spec.c_str());
                return 2;
            }
            presses.push_back({button_id(spec.substr(0, at)), static_cast<unsigned>(std::atoi(spec.c_str() + at + 1)),
                               static_cast<unsigned>(std::atoi(spec.c_str() + dash + 1))});
        } else if (a.rfind("--", 0) != 0) {
            game = a;
        }
    }
    void* lib = dlopen(core_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "mv-retro-host: %s\n", dlerror());
        return 1;
    }
    sym<void (*)(retro_environment_t)>(lib, "retro_set_environment")(environment);
    sym<void (*)(retro_video_refresh_t)>(lib, "retro_set_video_refresh")(video);
    sym<void (*)(retro_audio_sample_t)>(lib, "retro_set_audio_sample")(audio_sample);
    sym<void (*)(retro_audio_sample_batch_t)>(lib, "retro_set_audio_sample_batch")(audio_batch);
    sym<void (*)(retro_input_poll_t)>(lib, "retro_set_input_poll")(input_poll);
    sym<void (*)(retro_input_state_t)>(lib, "retro_set_input_state")(input_state);
    sym<void (*)()>(lib, "retro_init")();
    retro_system_info info{};
    sym<void (*)(retro_system_info*)>(lib, "retro_get_system_info")(&info);
    std::printf("core: %s %s, extensions %s\n", info.library_name, info.library_version, info.valid_extensions);
    retro_game_info gi{};
    gi.path = game.empty() ? nullptr : game.c_str();
    if (!sym<bool (*)(const retro_game_info*)>(lib, "retro_load_game")(game.empty() ? nullptr : &gi)) {
        std::fprintf(stderr, "mv-retro-host: retro_load_game failed\n");
        return 1;
    }
    retro_system_av_info av{};
    sym<void (*)(retro_system_av_info*)>(lib, "retro_get_system_av_info")(&av);
    std::printf("av: %ux%u base, %.3f fps, %.0f Hz\n", av.geometry.base_width, av.geometry.base_height, av.timing.fps, av.timing.sample_rate);
    if (png_every) std::filesystem::create_directories(png_dir);
    auto run = sym<void (*)()>(lib, "retro_run");
    const auto t0 = std::chrono::steady_clock::now();
    for (current_frame = 0; current_frame < frames; ++current_frame) {
        run();
        if (png_every && (current_frame + 1) % png_every == 0) save_png(png_dir / std::format("frame-{:06}.png", current_frame + 1));
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("ran %u frames in %.2f s (%.1f fps); %llu video callbacks, last %ux%u; %llu audio frames, %llu non-zero samples\n", frames, secs,
                frames / secs, frames_seen, last_w, last_h, audio_frames, audio_nonzero);
    sym<void (*)()>(lib, "retro_unload_game")();
    sym<void (*)()>(lib, "retro_deinit")();
    dlclose(lib);
    return 0;
}
