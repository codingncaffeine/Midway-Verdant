// Midway Verdant as a libretro core: the machine runs on its own threads, and each retro_run lets the CPU run
// exactly one frame of the console's time (frame steps on the guest clock, D16: it holds at every vblank until the
// next call), then hands the frontend the scanout (with the video overlay), the audio the AC'97 bus master played
// in that frame, and the pads' state from the RetroPads. The console's own files (the MCPX boot ROM, the flash
// image, the EEPROM and the hard-disk image) come from <system>/midway-verdant/.
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
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
#include <thread>
#include <vector>

#include "core/machine.h"
#include "debug/trace.h"
#include "ide/disk_image.h"
#include "ide/dvd_image.h"
#include "ide/hdd_format.h"
#include "mv/version.h"
#include "libretro/libretro.h"
#include "libretro/libretro_vulkan.h"
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
// What video_cb gets: `frame`, or at an internal resolution above 1x the GPU's finished picture where it lies (read back
// already converted, the colour table applied). That one is the frontend's from grab_frame to picture_done: its copy
// may still be landing (frame_pic.ready, waited for after the CPU is let go), and no vblank writes it meanwhile (D36).
const void* frame_ptr = nullptr;
std::size_t frame_pitch = 0;
mv::pgraph::VulkanRenderer::ScaledPicture frame_pic;
std::vector<mv::u8> composed;

std::FILE* trace_file = nullptr;  // MV_TRACE_FILE

std::mutex audio_lock;
std::vector<mv::i16> audio;  // interleaved stereo at 48 kHz, filled on the machine's thread
// The audio option: the console's output (the AC'97 playing what the GP and EP made) or the direct mix (every voice
// straight to stereo, no DSP effects; mv::Apu::set_direct_mix). Read at load and whenever the frontend changes it.
std::atomic<bool> direct_audio{false};
// The internal resolution option (PLAN D34): the frame is res_scale times the scanout each way. The fast core only.
unsigned res_scale = 1;
// The frontend took the Vulkan hardware-render interface (PLAN D38): frames go to it as images on its device.
bool hw_video = false;
bool fast_core = true;
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
    mv::u64 setup_engine{0}, dsp_wait{0}, draws{0}, decodes{0}, write_backs{0}, waits{0}, flips{0}, io{0}, gpu_busy{0};
    mv::u64 counted_out{0}, idle_spin{0}, warped{0}, halts{0}, exits{0}, mmio{0};
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

bool direct_mix_option() {
    const char* a = option("midway_verdant_audio");
    return a && std::strncmp(a, "direct", 6) == 0;
}

unsigned resolution_option() {
    const char* r = option("midway_verdant_resolution");
    return r && r[0] >= '1' && r[0] <= '8' ? static_cast<unsigned>(r[0] - '0') : 1u;
}

unsigned anisotropy_option() {
    const char* an = option("midway_verdant_anisotropy");
    return an && std::isdigit(static_cast<unsigned char>(an[0])) ? static_cast<unsigned>(std::atoi(an)) : 1u;
}

// Interleaved 16-bit stereo samples the console played, for the next retro_run to hand over.
void queue_audio(const void* samples, std::size_t n) {
    std::lock_guard g(audio_lock);
    const std::size_t at = audio.size();
    audio.resize(at + n);
    std::memcpy(audio.data() + at, samples, n * 2);
    // A frontend that stops calling retro_run must not build up latency: keep at most a quarter second.
    constexpr std::size_t kMaxBacklog = kSampleRate / 4 * 2;
    if (audio.size() > kMaxBacklog) audio.erase(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(audio.size() - kMaxBacklog));
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

// Rows [0, rows) split into bands on up to four threads: a picture at a high internal resolution is tens of MB a frame.
// `big`: the work is large however few the rows (each a block of rows at the internal resolution).
template <typename F>
void parallel_rows(mv::u32 rows, F&& f, bool big = false) {
    const mv::u32 n = (rows >= 1024 || big) && rows >= 4 ? 4u : 1u, step = (rows + n - 1) / n;
    std::vector<std::thread> helpers;
    for (mv::u32 i = 1; i < n; ++i)
        if (const mv::u32 a = i * step; a < rows) helpers.emplace_back([&f, a, b = std::min(rows, a + step)] { f(a, b); });
    f(0u, std::min(rows, step));
    for (std::thread& t : helpers) t.join();
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
        frame_ptr = frame.data();
        frame_pitch = static_cast<std::size_t>(frame_w) * 4;
        return;
    }
    const mv::u8* fb = bus.ram() + start;
    // At an internal resolution above 1x the picture is the scanned-out target as the GPU rendered it; one only RAM
    // holds (the CPU or a blit wrote it, nothing drew it) is widened, each pixel to a block.
    mv::u32 sc = 1;
    const mv::pgraph::VulkanRenderer* renderer = machine->renderer();
    mv::pgraph::VulkanRenderer::ScaledPicture pic;
    bool rendered = res_scale > 1 && renderer && renderer->scaled_picture(start, pitch, bpp, w, h, pic);
    if (rendered && hw_video && !(pic.xrgb && (!gpu.overlay() || pic.overlaid))) {
        // On the frontend's device the picture is in device memory: one the GPU did not finish (an overlay too
        // large for its pass) is taken from RAM instead.
        renderer->picture_done();
        rendered = false;
    }
    sc = rendered ? pic.sc : res_scale;
    // Instrument: MV_DEBUG_GRAB=1 prints each frame's path (the GPU's picture in place, copied, or RAM's widened).
    static const bool debug_grab = [] { const char* d = std::getenv("MV_DEBUG_GRAB"); return d && *d == '1'; }();
    if (debug_grab)
        std::fprintf(stderr, "grab: frame %llu start %08x %ux%u bpp %u rendered %d xrgb %d overlay %d\n", static_cast<unsigned long long>(frames_run), start, w, h, bpp, rendered ? 1 : 0,
                     rendered && pic.xrgb ? 1 : 0, gpu.overlay() ? 1 : 0);
    if (rendered && pic.xrgb && (!gpu.overlay() || pic.overlaid)) {
        // The GPU's finished picture (the video overlay drawn in too) goes to the frontend as it lies: no pass over it here.
        frame_w = w * sc;
        frame_h = h * sc;
        frame_ptr = pic.pixels;
        frame_pitch = pic.pitch;
        frame_pic = pic;  // retro_run waits for it once the CPU is let go, and lets it go after video_cb
        return;
    }
    if (rendered) renderer->picture_ready(pic);  // read here: landed first
    const mv::u32 fw = w * sc, fh = h * sc;
    frame_w = fw;
    frame_h = fh;
    frame.resize(static_cast<std::size_t>(fw) * fh);
    mv::u8* out = reinterpret_cast<mv::u8*>(frame.data());
    const auto lut = gpu.gamma_lut();
    const bool rgb565 = gpu.scanout_rgb565();
    // The colour table as three tables of finished channel bits: a pixel is three lookups OR-ed.
    std::array<mv::u32, 256> lut_b{}, lut_g{}, lut_r{};
    if (lut)
        for (mv::u32 i = 0; i < 256; ++i) {
            lut_b[i] = (*lut)[i * 3u + 2];
            lut_g[i] = static_cast<mv::u32>((*lut)[i * 3u + 1]) << 8;
            lut_r[i] = static_cast<mv::u32>((*lut)[i * 3u]) << 16;
        }
    if (rendered) {
        // The rendered picture into the frame where it lies, in bands on a few threads, the colour table applied and the
        // alpha byte cleared in the same pass (at 6x a frame is 44 MB; three passes on one thread took 8 ms).
        parallel_rows(fh, [&](mv::u32 y0, mv::u32 y1) {
            for (mv::u32 y = y0; y < y1; ++y) {
                const mv::u8* src = pic.pixels + static_cast<std::size_t>(y) * pic.pitch;
                mv::u32* dst = frame.data() + static_cast<std::size_t>(y) * fw;
                if (pic.xrgb) {
                    std::memcpy(dst, src, static_cast<std::size_t>(fw) * 4);  // finished on the GPU (an overlay follows)
                } else if (bpp != 4) {
                    mv::scanout_to_xrgb(src, pic.pitch, bpp, rgb565, fw, 1, reinterpret_cast<mv::u8*>(dst));
                    if (lut) mv::apply_gamma(*lut, reinterpret_cast<mv::u8*>(dst), fw * 4, fw, 1);
                } else if (lut) {
                    const mv::u32* row = reinterpret_cast<const mv::u32*>(src);
                    for (mv::u32 x = 0; x < fw; ++x) {
                        const mv::u32 p = row[x];
                        dst[x] = lut_b[p & 0xFFu] | lut_g[(p >> 8) & 0xFFu] | lut_r[(p >> 16) & 0xFFu];
                    }
                } else {
                    const mv::u32* row = reinterpret_cast<const mv::u32*>(src);
                    for (mv::u32 x = 0; x < fw; ++x) dst[x] = row[x] & 0x00FFFFFFu;
                }
            }
        });
        renderer->picture_done();
    } else if (sc > 1) {
        // RAM's picture finished at its own size (colour table, the video overlay: a movie samples its nearest texel
        // either way), then each pixel widened to a block on a few threads, a block row built once and copied down.
        // Done at the internal resolution on one thread this took 50 ms a frame at 6x (a title's intro movie).
        composed.resize(static_cast<std::size_t>(w) * h * 4);
        mv::u8* c = composed.data();
        mv::scanout_to_xrgb(fb, pitch, bpp, gpu.scanout_rgb565(), w, h, c);
        if (lut) mv::apply_gamma(*lut, c, w * 4, w, h);
        if (const auto ov = gpu.overlay()) mv::draw_overlay(*ov, bus.ram(), bus.ram_size(), c, w * 4, w, h);
        const mv::u32* src = reinterpret_cast<const mv::u32*>(c);
        parallel_rows(
            h,
            [&](mv::u32 y0, mv::u32 y1) {
                for (mv::u32 y = y0; y < y1; ++y) {
                    const mv::u32* row = src + static_cast<std::size_t>(y) * w;
                    mv::u32* dst = frame.data() + static_cast<std::size_t>(y) * sc * fw;
                    for (mv::u32 x = 0; x < w; ++x) std::fill_n(dst + static_cast<std::size_t>(x) * sc, sc, row[x] & 0x00FFFFFFu);
                    for (mv::u32 k = 1; k < sc; ++k) std::memcpy(dst + static_cast<std::size_t>(k) * fw, dst, static_cast<std::size_t>(fw) * 4);
                }
            },
            true);
        frame_ptr = frame.data();
        frame_pitch = static_cast<std::size_t>(fw) * 4;
        return;
    } else {
        mv::scanout_to_xrgb(fb, pitch, bpp, gpu.scanout_rgb565(), w, h, out);
    }
    if (lut && !rendered) mv::apply_gamma(*lut, out, fw * 4, fw, fh);
    const auto ov = gpu.overlay();
    // The scanout ignores the alpha byte; the frontend's XRGB8888 wants it clear of meaning too (a rendered 32-bit
    // picture came with it cleared).
    if (ov) {
        // The overlay's rectangle and steps are in scanout pixels: sc times as many at the internal resolution; drawn
        // over the rendered picture in bands.
        mv::Overlay o = *ov;
        o.out_x *= sc;
        o.out_y *= sc;
        o.out_w *= sc;
        o.out_h *= sc;
        o.dsdx /= sc;
        o.dtdy /= sc;
        parallel_rows(fh, [&](mv::u32 y0, mv::u32 y1) {
            mv::draw_overlay(o, bus.ram(), bus.ram_size(), out, fw * 4, fw, fh, y0, y1);
            for (std::size_t i = static_cast<std::size_t>(y0) * fw; i < static_cast<std::size_t>(y1) * fw; ++i) frame[i] &= 0x00FFFFFFu;
        });
    } else if (!rendered) {
        for (auto& px : frame) px &= 0x00FFFFFFu;
    }
    frame_ptr = frame.data();
    frame_pitch = static_cast<std::size_t>(fw) * 4;
}

// The frame's geometry follows the internal resolution: the base is the console's 640x480 times it, the largest the
// HD modes' 1920x1080 times it.
void set_geometry(retro_game_geometry& g) {
    g.base_width = 640 * res_scale;
    g.base_height = 480 * res_scale;
    g.max_width = 1920 * res_scale;
    g.max_height = 1080 * res_scale;
    g.aspect_ratio = 4.0f / 3.0f;
}

// The visual options take effect while the game runs: the internal resolution at the next vblank (the targets come
// back from RAM at the new size; the frontend is told the new geometry), anisotropic filtering with the next sampler.
void apply_visual_options() {
    mv::pgraph::VulkanRenderer* renderer = machine->renderer();
    if (!renderer) return;
    renderer->set_min_anisotropy(anisotropy_option());
    const unsigned s = fast_core ? resolution_option() : 1u;
    if (s == res_scale) return;
    res_scale = s;
    renderer->set_resolution_scale(s);
    retro_system_av_info av{};
    set_geometry(av.geometry);
    av.timing.fps = kFps;
    av.timing.sample_rate = kSampleRate;
    env_cb(RETRO_ENVIRONMENT_SET_SYSTEM_AV_INFO, &av);
    log(RETRO_LOG_INFO, "internal resolution: %ux", s);
}

// ── The frontend's Vulkan device (the libretro hardware-render interface, PLAN D38) ─────────────────────────────────
// A frontend that takes SET_HW_RENDER for Vulkan makes the device (through our negotiation callbacks when it has
// them, so the renderer's features and extensions are on it), and each frame goes to it as an image on the GPU
// instead of being read back and copied (a 6x frame is 44 MB). The console is built once the device exists.
constexpr mv::u32 kVulkan11 = (1u << 22) | (1u << 12);  // VK_API_VERSION_1_1 (its macro's casts trip -Wold-style-cast)
retro_hw_render_callback hw_render{};
const retro_hw_render_interface_vulkan* vk_iface = nullptr;
std::optional<mv::MachineConfig> pending_cfg;  // the console to build at the next context_reset
mv::vk::Enabled negotiated_enabled;  // what create_logical enabled on the frontend's device
bool negotiated = false;

const VkApplicationInfo* hw_application_info() {
    static VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "Midway Verdant", 0, "Midway Verdant", 0, kVulkan11};
    return &app;
}

// The GPU the frontend leaves to us: a discrete one first.
VkPhysicalDevice pick_gpu(const mv::vk::Functions& fn, VkInstance instance) {
    mv::u32 n = 0;
    fn.vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> gpus(n);
    fn.vkEnumeratePhysicalDevices(instance, &n, gpus.data());
    VkPhysicalDevice best = VK_NULL_HANDLE;
    int best_score = -1;
    for (VkPhysicalDevice g : gpus) {
        VkPhysicalDeviceProperties p{};
        fn.vkGetPhysicalDeviceProperties(g, &p);
        const int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if (score > best_score) {
            best_score = score;
            best = g;
        }
    }
    return best;
}

// The device the renderer wants, made by `create` (the frontend's wrapper, or vkCreateDevice with the frontend's
// required extensions added), with its queue handed to the frontend for presenting too.
bool make_frontend_device(retro_vulkan_context* ctx, VkInstance instance, VkPhysicalDevice gpu, PFN_vkGetInstanceProcAddr gipa,
                          const std::function<VkDevice(VkPhysicalDevice, const VkDeviceCreateInfo&)>& create) {
    mv::vk::Functions fn{};
    fn.vkGetInstanceProcAddr = gipa;
    mv::vk::load_instance(fn, instance);
    if (!gpu) gpu = pick_gpu(fn, instance);
    if (!gpu) {
        log(RETRO_LOG_ERROR, "Vulkan: the frontend's instance has no GPU");
        return false;
    }
    mv::u32 family = 0;
    std::string error;
    const VkDevice device = mv::vk::Device::create_logical(fn, gpu, family, negotiated_enabled, [&](const VkDeviceCreateInfo& ci) { return create(gpu, ci); }, error);
    if (!device) {
        log(RETRO_LOG_ERROR, "Vulkan: %s", error.c_str());
        return false;
    }
    const auto get_queue = reinterpret_cast<PFN_vkGetDeviceQueue>(fn.vkGetDeviceProcAddr(device, "vkGetDeviceQueue"));
    ctx->gpu = gpu;
    ctx->device = device;
    ctx->queue_family_index = family;
    get_queue(device, family, 0, &ctx->queue);
    ctx->presentation_queue = ctx->queue;  // the graphics queue presents on every desktop driver we know
    ctx->presentation_queue_family_index = family;
    negotiated = true;
    return true;
}

bool hw_create_device2(retro_vulkan_context* ctx, VkInstance instance, VkPhysicalDevice gpu, VkSurfaceKHR, PFN_vkGetInstanceProcAddr gipa,
                       retro_vulkan_create_device_wrapper_t wrapper, void* opaque) {
    return make_frontend_device(ctx, instance, gpu, gipa, [&](VkPhysicalDevice g, const VkDeviceCreateInfo& ci) { return wrapper(g, opaque, &ci); });
}

// v1: the frontend's required extensions and layers go in with ours.
bool hw_create_device(retro_vulkan_context* ctx, VkInstance instance, VkPhysicalDevice gpu, VkSurfaceKHR, PFN_vkGetInstanceProcAddr gipa,
                      const char** required_extensions, unsigned n_extensions, const char** required_layers, unsigned n_layers, const VkPhysicalDeviceFeatures*) {
    const auto create_device = reinterpret_cast<PFN_vkCreateDevice>(gipa(instance, "vkCreateDevice"));
    return make_frontend_device(ctx, instance, gpu, gipa, [&](VkPhysicalDevice g, const VkDeviceCreateInfo& ci) {
        std::vector<const char*> ext(ci.ppEnabledExtensionNames, ci.ppEnabledExtensionNames + ci.enabledExtensionCount);
        for (unsigned i = 0; i < n_extensions; ++i)
            if (std::ranges::none_of(ext, [&](const char* e) { return std::strcmp(e, required_extensions[i]) == 0; })) ext.push_back(required_extensions[i]);
        VkDeviceCreateInfo with = ci;
        with.enabledExtensionCount = static_cast<mv::u32>(ext.size());
        with.ppEnabledExtensionNames = ext.data();
        with.enabledLayerCount = n_layers;
        with.ppEnabledLayerNames = required_layers;
        VkDevice d = VK_NULL_HANDLE;
        return create_device(g, &with, nullptr, &d) == VK_SUCCESS ? d : VK_NULL_HANDLE;
    });
}

void hw_destroy_device() {}  // nothing of ours lives outside the renderer, which goes with the console

bool start_machine(const mv::MachineConfig& cfg);

void hw_context_reset() {
    vk_iface = nullptr;
    if (!env_cb(RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE, &vk_iface) || !vk_iface || vk_iface->interface_type != RETRO_HW_RENDER_INTERFACE_VULKAN || !vk_iface->set_image) {
        message("Midway Verdant: the frontend gave no Vulkan interface");
        vk_iface = nullptr;
        return;
    }
    if (!pending_cfg) return;  // the console runs already
    mv::MachineConfig cfg = *pending_cfg;
    mv::vk::External e;
    e.instance = vk_iface->instance;
    e.gpu = vk_iface->gpu;
    e.device = vk_iface->device;
    e.queue = vk_iface->queue;
    e.queue_family = vk_iface->queue_index;
    e.get_instance_proc_addr = vk_iface->get_instance_proc_addr;
    e.lock_queue = vk_iface->lock_queue;
    e.unlock_queue = vk_iface->unlock_queue;
    e.handle = vk_iface->handle;
    e.negotiated = negotiated;
    e.enabled = negotiated_enabled;
    cfg.gpu_external = e;
    if (!negotiated) log(RETRO_LOG_WARN, "Vulkan: the frontend made the device without our negotiation; optional features (border colours, anisotropy) are off");
    if (start_machine(cfg)) pending_cfg.reset();
}

// The frontend takes its device away (it is going, or remaking its video): the console cannot outlive the renderer on
// it, so it stops; the next context_reset builds it again from the start.
void hw_context_destroy() {
    if (machine) {
        log(RETRO_LOG_WARN, "Vulkan: the frontend took its device away while the console ran; it starts again with the next one");
        std::lock_guard g(audio_lock);
        audio.clear();
    }
    if (machine) {
        pending_cfg = machine->config();
        machine.reset();  // its CPU stops and its renderer goes before the device does
    }
    vk_iface = nullptr;
}

// Asks the frontend for Vulkan hardware rendering and offers our device negotiation; false (software frames, as
// before) when it has no Vulkan or MV_VIDEO=software asks for that.
bool offer_hw_render() {
    hw_video = false;
    negotiated = false;
    if (const char* v = std::getenv("MV_VIDEO"); v && std::strcmp(v, "software") == 0) return false;
    hw_render = {};
    hw_render.context_type = RETRO_HW_CONTEXT_VULKAN;
    hw_render.version_major = kVulkan11;
    hw_render.context_reset = hw_context_reset;
    hw_render.context_destroy = hw_context_destroy;
    if (!env_cb(RETRO_ENVIRONMENT_SET_HW_RENDER, &hw_render)) return false;
    static retro_hw_render_context_negotiation_interface_vulkan iface{};
    iface.interface_type = RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN;
    iface.interface_version = RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN_VERSION;
    iface.get_application_info = hw_application_info;
    iface.create_device = hw_create_device;
    iface.destroy_device = hw_destroy_device;
    iface.create_instance = nullptr;  // the frontend's instance is fine: we need nothing of it
    iface.create_device2 = hw_create_device2;
    env_cb(RETRO_ENVIRONMENT_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE, &iface);  // optional: without it the frontend makes the device alone
    hw_video = true;
    return true;
}

// Builds the console and lets it run: at load, or once the frontend made its Vulkan device (hw_context_reset).
bool start_machine(const mv::MachineConfig& cfg) {
    std::string error;
    machine = std::make_unique<mv::Machine>(cfg);
    if (!machine->init(error)) {
        message(("Midway Verdant: " + error).c_str());
        machine.reset();
        return false;
    }
    direct_audio = direct_mix_option();
    machine->apu().set_direct_mix(direct_audio);
    machine->apu().set_direct_sink([](const mv::i16* stereo, mv::u32 frames) {
        if (direct_audio) queue_audio(stereo, frames * 2);
    });
    machine->aci().set_sample_sink([](mv::Aci::Channel c, const mv::u8* data, mv::u32 bytes) {
        if (c != mv::Aci::PcmOut || direct_audio) return;
        queue_audio(data, bytes / 2);
    });
    if (have_rumble) {
        for (unsigned p = 0; p < pads_plugged; ++p)
            machine->pad(p).set_rumble([p](mv::u16 left, mv::u16 right) {
                rumble.set_rumble_state(p, RETRO_RUMBLE_STRONG, left);
                rumble.set_rumble_state(p, RETRO_RUMBLE_WEAK, right);
            });
    }
    log(RETRO_LOG_INFO, "booting %s with %u pad(s); files from %s", cfg.dvd_image ? cfg.dvd_image->string().c_str() : "no disc", pads_plugged,
        cfg.mcpx_rom.parent_path().string().c_str());
    log(RETRO_LOG_INFO, "timeline: %s; one console frame per retro_run", machine->cpu()->clock_name().c_str());
    log(RETRO_LOG_INFO, "audio: %s", direct_audio ? "direct mix" : "console");
    if (hw_video && machine->renderer()) machine->renderer()->set_handoff(true);  // captures stay on the device (D38)
    frames_run = frames_capped = 0;
    frame_ms_sum = frame_ms_max = 0;
    hold_wake_ms_sum = hold_grab_ms_sum = hold_wake_ms_max = 0;
    machine->start();
    return true;
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
        {"midway_verdant_disc_speed", "Disc loading; fast (8x the console's drive)|console speed"},
        {"midway_verdant_resolution", "Internal resolution (the fast CPU only); 1x (native)|2x|4x|6x"},
        {"midway_verdant_anisotropy", "Anisotropic filtering; off (as the game asks)|2x|4x|8x|16x"},
        {"midway_verdant_shaders", "Shader compiling; in the background (no stutter; a new effect can appear a frame late)|in the frame (stutters the first time a scene is seen)"},
        {"midway_verdant_audio", "Audio; console (DSP effects such as reverb)|direct mix (every voice straight to stereo, no DSP effects)"},
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
    set_geometry(info->geometry);
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
        // A folder: one log per game and start, named after the disc (a frontend session runs several games).
        std::filesystem::path path = f;
        if (std::error_code ec; std::filesystem::is_directory(path, ec)) {
            const std::time_t now = std::time(nullptr);
            char stamp[32];
            std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&now));
            const std::string disc = game && game->path ? std::filesystem::path(game->path).stem().string() : "dashboard";
            path /= disc + "-" + stamp + ".tsv";
        }
        frame_log = std::fopen(path.string().c_str(), "w");
        frame_log_index = 0;
        if (frame_log) std::fprintf(frame_log, "frame\tperiod\twait\tguest\thandlers\tevents\theld\tpusher\tdraw\tfetch\ttexture\tupload\tflush\twriteback\tsetup\tdspwait\tdraws\tdecodes\twritebacks\twaits\tflips\tio\tgpubusy\tcounted\tidlespin\twarp\thalts\texits\tmmio\n");
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
    cfg.disc_speed = 8;
    if (const char* d = option("midway_verdant_disc_speed"); d && std::strncmp(d, "console", 7) == 0) cfg.disc_speed = 1;
    { const char* sh = option("midway_verdant_shaders"); cfg.async_shaders = !(sh && std::strncmp(sh, "in the frame", 12) == 0); }
    cfg.anisotropy = anisotropy_option();
    // The internal resolution needs the fast core (the accurate core's pictures and pixel counts stay the console's).
    fast_core = cfg.cpu == "kvm";
    res_scale = fast_core ? resolution_option() : 1u;
    cfg.resolution_scale = res_scale;
    cfg.frame_step = true;
    // Instrument: MV_RTC_TIME=UNIX fixes the clock's start, so two accurate-core runs of a title can be compared.
    if (const char* t = std::getenv("MV_RTC_TIME"); t && *t) cfg.rtc_base_time = std::strtoll(t, nullptr, 10);
    // Instrument: MV_BREAK_EIP=HEX,... is the shell's --break-eip (registers and stack into the boot trace, MV_TRACE=boot).
    if (const char* b = std::getenv("MV_BREAK_EIP"); b && *b)
        for (const char* p = b; *p;) {
            char* end = nullptr;
            const unsigned long eip = std::strtoul(p, &end, 16);
            if (end == p) break;
            cfg.break_eips.push_back(static_cast<mv::u32>(eip));
            p = *end ? end + 1 : end;
        }
    // Instrument: MV_BREAK_MMIO=ADDR[=VALUE][*N] is the shell's --break-mmio (a write to physical ADDR).
    if (const char* b = std::getenv("MV_BREAK_MMIO"); b && *b) {
        std::string spec = b;
        mv::MachineConfig::MmioBreak brk;
        if (const auto star = spec.find('*'); star != std::string::npos) {
            brk.count = static_cast<mv::u32>(std::strtoul(spec.c_str() + star + 1, nullptr, 10));
            spec.resize(star);
        }
        const auto eq = spec.find('=');
        brk.address = static_cast<mv::u32>(std::strtoul(spec.substr(0, eq).c_str(), nullptr, 16));
        if (eq != std::string::npos) brk.value = static_cast<mv::u32>(std::strtoul(spec.c_str() + eq + 1, nullptr, 16));
        cfg.break_mmio_write = brk;
    }
    cfg.pads = pads_plugged;
    if (game && game->path && *game->path) cfg.dvd_image = game->path;
    // MV_CAPTURE=FILE: a pushbuffer capture of the whole run (GPU methods, RAM pages, objects) for offline replay with
    // mv-pbcap, as the shell's --capture (a picture fault seen under a frontend, replayed draw by draw). Large: GBs a minute.
    if (const char* c = std::getenv("MV_CAPTURE"); c && *c) cfg.capture_path = c;
    if (offer_hw_render()) {
        pending_cfg = cfg;  // started on the frontend's device (hw_context_reset)
        log(RETRO_LOG_INFO, "video: the frontend's Vulkan device, frames handed over as images");
        return true;
    }
    return start_machine(cfg);
}

RETRO_API bool retro_load_game_special(unsigned, const retro_game_info*, size_t) { return false; }

RETRO_API void retro_unload_game(void) {
    pending_cfg.reset();  // a console the frontend's device never came for
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
        // MV_DUMP_AT_EXIT=FILE: the guest RAM to FILE and the CPU's registers to FILE.regs as the game is closed (a game
        // stuck in a loop: what the loop reads, and CR3 to find its code).
        if (const char* e = std::getenv("MV_DUMP_AT_EXIT"); e && *e) {
            if (std::FILE* f = std::fopen(e, "wb")) {
                std::fwrite(machine->bus().ram(), 1, machine->bus().ram_size(), f);
                std::fclose(f);
            }
            mv::CpuBackend::Regs r{};
            if (machine->cpu() && machine->cpu()->read_regs(r))
                if (std::FILE* f = std::fopen(std::format("{}.regs", e).c_str(), "w")) {
                    std::fprintf(f, "eax %08x ebx %08x ecx %08x edx %08x esi %08x edi %08x ebp %08x esp %08x eip %08x eflags %08x cr0 %08x cr3 %08x cr4 %08x\n", r.eax, r.ebx, r.ecx,
                                 r.edx, r.esi, r.edi, r.ebp, r.esp, r.eip, r.eflags, r.cr0, r.cr3, r.cr4);
                    std::fclose(f);
                }
            log(RETRO_LOG_INFO, "guest RAM and registers written to %s", e);
        }
        if (machine->nv2a().timed())
            log(RETRO_LOG_INFO, "gpu timing: %.3f s of GPU time for %.3f s of console time; %llu pixels drawn costing %.3f s; their counts waited for %llu times; %llu z-pass reports",
                static_cast<double>(machine->nv2a().gpu_clocks()) / (static_cast<double>(mv::clk::kGpuHzNum) / mv::clk::kGpuHzDen),
                static_cast<double>(mv::ns_from_cycles(machine->scheduler().now())) / 1e9, static_cast<unsigned long long>(machine->nv2a().timed_pixels()),
                static_cast<double>(machine->nv2a().pixel_clocks()) / (static_cast<double>(mv::clk::kGpuHzNum) / mv::clk::kGpuHzDen),
                static_cast<unsigned long long>(machine->nv2a().pixel_resolves()), static_cast<unsigned long long>(machine->nv2a().reports()));
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
            log(RETRO_LOG_INFO, "gpu time: %llu draws %.0f ms (fetch %.0f, indices %.0f, textures %.0f, pipelines %.0f, shader compiles %.0f, uploads %.0f, descriptors %.0f, record %.0f); %llu submits + %llu waits %.0f ms; %llu write-backs %.0f ms (device copies %.0f); %llu textures decoded (%llu as DXT blocks; %llu kept: pages written, bytes unchanged), %llu shaders from the disk cache; background compiles: %llu shaders + %llu pipelines, %llu draws skipped waiting (%llu short waits first); %llu pipelines made from the cache at once",
                static_cast<unsigned long long>(st.draws), ms(st.ns_draw), ms(st.ns_fetch), ms(st.ns_indices), ms(st.ns_texture), ms(st.ns_pipeline), ms(st.ns_vs + st.ns_fs),
                ms(st.ns_upload), ms(st.ns_descriptors), ms(st.ns_record), static_cast<unsigned long long>(st.flushes), static_cast<unsigned long long>(st.waits),
                ms(st.ns_flush), static_cast<unsigned long long>(st.writebacks), ms(st.ns_write_back), ms(st.ns_write_back_gpu),
                static_cast<unsigned long long>(st.textures_decoded), static_cast<unsigned long long>(st.textures_native_dxt), static_cast<unsigned long long>(st.textures_unchanged), static_cast<unsigned long long>(st.shaders_from_disk),
                static_cast<unsigned long long>(st.compiles_queued), static_cast<unsigned long long>(st.pipelines_queued), static_cast<unsigned long long>(st.pending_skips), static_cast<unsigned long long>(st.pending_waits), static_cast<unsigned long long>(st.pipeline_cache_hits));
            // Draws the renderer could not draw, and why (the first of each distinct reason).
            if (st.skipped) log(RETRO_LOG_INFO, "gpu: %llu draws skipped", static_cast<unsigned long long>(st.skipped));
            log(RETRO_LOG_INFO, "gpu: write-backs by cause: frame end %llu, texture %llu, scanout %llu, 2D/range %llu, relayout %llu, idle vblank %llu; %llu targets sampled in place, %llu snapshots for draws sampling their own target (%llu of depth), %llu depth buffers converted to colour, %llu targets the CPU wrote reloaded, %llu depth buffers handed between targets, %llu loaded from RAM; %llu targets evicted by an overlapping one, %llu retired unused, %llu rectangles of a region copied for a texture",
                static_cast<unsigned long long>(st.writebacks_by[0]), static_cast<unsigned long long>(st.writebacks_by[1]), static_cast<unsigned long long>(st.writebacks_by[2]),
                static_cast<unsigned long long>(st.writebacks_by[3]), static_cast<unsigned long long>(st.writebacks_by[4]), static_cast<unsigned long long>(st.writebacks_by[5]),
                static_cast<unsigned long long>(st.surface_samples), static_cast<unsigned long long>(st.snapshots), static_cast<unsigned long long>(st.depth_snapshots), static_cast<unsigned long long>(st.zeta_conversions), static_cast<unsigned long long>(st.cpu_written_targets),
                static_cast<unsigned long long>(st.depth_handovers), static_cast<unsigned long long>(st.depth_uploads), static_cast<unsigned long long>(st.evicted), static_cast<unsigned long long>(st.retired_unused), static_cast<unsigned long long>(st.subrect_copies));
            log(RETRO_LOG_INFO, "gpu: waits for a frame slot %.3f s, for everything submitted %.3f s", static_cast<double>(st.ns_wait_slot) / 1e9, static_cast<double>(st.ns_wait_finish) / 1e9);
            if (st.cpu_written_depth) log(RETRO_LOG_INFO, "gpu: of the targets the CPU wrote, %llu were depth buffers let go", static_cast<unsigned long long>(st.cpu_written_depth));
            if (st.surface_reloads) log(RETRO_LOG_INFO, "gpu: %llu render targets reloaded from RAM after a 2D write", static_cast<unsigned long long>(st.surface_reloads));
            if (st.cube_assemblies) log(RETRO_LOG_INFO, "gpu: %llu cube maps put together on the GPU from their face targets", static_cast<unsigned long long>(st.cube_assemblies));
            if (st.depth_unparks || st.parked_samples || st.depth_loads_skipped || st.depth_borrows)
                log(RETRO_LOG_INFO, "gpu: depth buffers: %llu parked images bound again, %llu sampled in place, %llu loads skipped for a whole clear, %llu drawn with a bigger target's image",
                    static_cast<unsigned long long>(st.depth_unparks), static_cast<unsigned long long>(st.parked_samples), static_cast<unsigned long long>(st.depth_loads_skipped), static_cast<unsigned long long>(st.depth_borrows));
            if (st.mip_assemblies) log(RETRO_LOG_INFO, "gpu: %llu mip chains put together on the GPU from their level targets", static_cast<unsigned long long>(st.mip_assemblies));
            if (!st.forced_by.empty()) {
                // The write-backs a texture forced, most first: what is sampled through RAM instead of in place.
                std::vector<std::pair<std::string, mv::u64>> forced(st.forced_by.begin(), st.forced_by.end());
                std::sort(forced.begin(), forced.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
                for (std::size_t i = 0; i < forced.size() && i < 8; ++i) log(RETRO_LOG_INFO, "gpu: forced by a texture x%llu: %s", static_cast<unsigned long long>(forced[i].second), forced[i].first.c_str());
            }
            if (!st.blend_modes.empty()) {
                // The blended draws, most first: which blend each effect uses (a white patch, a missing glow).
                std::vector<std::pair<mv::u64, mv::u64>> modes(st.blend_modes.begin(), st.blend_modes.end());
                std::sort(modes.begin(), modes.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
                std::string line;
                for (std::size_t i = 0; i < modes.size() && i < 16; ++i) {
                    const mv::u64 k = modes[i].first;
                    line += std::format(" [sf {:x} df {:x} eq {:x}{} tex0 fmt {:02x} wrap {}/{}] x{}", (k >> 48) & 0xFFFFu, (k >> 32) & 0xFFFFu, (k >> 16) & 0xFFFFu,
                                        (k & 0x8000u) ? " w" : "", (k >> 8) & 0x7Fu, (k >> 4) & 0xFu, k & 0xFu, modes[i].second);
                }
                log(RETRO_LOG_INFO, "gpu: blended draws:%s", line.c_str());
            }
            if (!st.store_cost.empty()) {
                // Where storing write-backs in RAM went, by target, most first.
                std::vector<std::pair<std::string, std::pair<mv::u64, mv::u64>>> costs(st.store_cost.begin(), st.store_cost.end());
                std::sort(costs.begin(), costs.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
                for (std::size_t i = 0; i < costs.size() && i < 8; ++i)
                    log(RETRO_LOG_INFO, "gpu: stored in RAM x%llu, %.3f s: %s", static_cast<unsigned long long>(costs[i].second.first), static_cast<double>(costs[i].second.second) / 1e9, costs[i].first.c_str());
            }
            if (!st.relayouts.empty()) {
                // Targets given up for another at their memory (each written back with a wait when drawn into), most first.
                std::vector<std::pair<std::string, mv::u64>> given_up(st.relayouts.begin(), st.relayouts.end());
                std::sort(given_up.begin(), given_up.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
                for (std::size_t i = 0; i < given_up.size() && i < 12; ++i) log(RETRO_LOG_INFO, "gpu: x%llu %s", static_cast<unsigned long long>(given_up[i].second), given_up[i].first.c_str());
            }
            log(RETRO_LOG_INFO, "gpu: submits mid-frame: %llu with the draw rings full, %llu with the upload staging full; %llu 2D copies made on the GPU", static_cast<unsigned long long>(st.batch_flushes), static_cast<unsigned long long>(st.staging_flushes), static_cast<unsigned long long>(st.gpu_copies));
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
    s.gpu_busy = h.gpu_busy_guest_ns;
    s.counted_out = h.counted_out_ns, s.idle_spin = h.idle_spin_ns, s.warped = h.warped_cycles, s.halts = h.halts;
    s.exits = h.exits, s.mmio = h.mmio_exits;
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
    const std::string line = std::format("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\n",
        frame_log_index, period, wait_ms, ms(s.guest, frame_last.guest), ms(s.handlers, frame_last.handlers), ms(s.events, frame_last.events),
        ms(s.held, frame_last.held), ms(s.pusher, frame_last.pusher), ms(s.draw, frame_last.draw), ms(s.fetch, frame_last.fetch),
        ms(s.texture, frame_last.texture), ms(s.upload, frame_last.upload), ms(s.flush, frame_last.flush), ms(s.write_back, frame_last.write_back),
        ms(s.setup_engine, frame_last.setup_engine), ms(s.dsp_wait, frame_last.dsp_wait), s.draws - frame_last.draws, s.decodes - frame_last.decodes,
        s.write_backs - frame_last.write_backs, s.waits - frame_last.waits, s.flips - frame_last.flips, ms(s.io, frame_last.io), ms(s.gpu_busy, frame_last.gpu_busy),
        ms(s.counted_out, frame_last.counted_out), ms(s.idle_spin, frame_last.idle_spin), static_cast<double>(s.warped - frame_last.warped) / 733333.333,
        s.halts - frame_last.halts, s.exits - frame_last.exits, s.mmio - frame_last.mmio);
    std::fputs(line.c_str(), frame_log);
    frame_last = s;
    ++frame_log_index;
}

RETRO_API void retro_run(void) {
    if (!machine) return;
    input_poll_cb();
    if (bool changed = false; env_cb(RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE, &changed) && changed) {
        // The audio option switches while the game runs: what was queued from the other path is dropped.
        if (direct_mix_option() != direct_audio) {
            std::lock_guard g(machine->lock());
            direct_audio = !direct_audio;
            machine->apu().set_direct_mix(direct_audio);
            std::lock_guard a(audio_lock);
            audio.clear();
            log(RETRO_LOG_INFO, "audio: %s", direct_audio ? "direct mix" : "console");
        }
        apply_visual_options();
    }
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
    // MV_FRAME_WAIT_MS=0 waits for the vblank however long it takes (oracle and benchmark runs: a call returned
    // without its vblank shifts the front end's frame count, and with it scripted input, against the guest's).
    static const long wait_ms = [] {
        const char* w = std::getenv("MV_FRAME_WAIT_MS");
        return w && *w ? std::strtol(w, nullptr, 10) : 1000L;
    }();
    const auto cap = wait_ms > 0 ? std::chrono::milliseconds(wait_ms) : std::chrono::hours(24 * 365);
    // The vblank alone is not the frame's end: the CPU holds at the first boundary after it (Machine::pace_cpu). Released
    // before it got there (the pipelined release below), it would run through the hold and take this frame's input a
    // frame late — a race with the host's timing (one hold in thousands went missing in an oracle run).
    if (machine->pump(t0 + cap, [target](mv::Machine& m) { return m.nv2a().vblanks() >= target && m.cpu_paused(); }) ==
        mv::Machine::StopReason::Duration) {
        if (frames_capped++ == 0) log(RETRO_LOG_WARN, "a frame took longer than the wait cap (MV_FRAME_WAIT_MS, default 1000) of host time; returned without its vblank");
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
    // MV_DUMP_RAM=FRAME:FILE[,FRAME:FILE...]: the guest RAM to FILE when that frame holds (with MV_RENDER_DEBUG=wball every
    // GPU target is in it), for finding what the guest copied from where.
    static const std::vector<std::pair<unsigned long long, std::string>> ram_dumps = [] {
        std::vector<std::pair<unsigned long long, std::string>> v;
        if (const char* e = std::getenv("MV_DUMP_RAM")) {
            std::string s = e;
            for (std::size_t at = 0; at < s.size();) {
                const std::size_t comma = std::min(s.find(',', at), s.size()), colon = s.find(':', at);
                if (colon < comma) v.emplace_back(std::strtoull(s.c_str() + at, nullptr, 10), s.substr(colon + 1, comma - colon - 1));
                at = comma + 1;
            }
        }
        return v;
    }();
    for (const auto& [at, path] : ram_dumps) {
        if (at != frames_run) continue;
        std::lock_guard g(machine->lock());
        if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(machine->bus().ram(), 1, machine->bus().ram_size(), f);
            std::fclose(f);
            log(RETRO_LOG_INFO, "frame %llu: guest RAM written to %s", at, path.c_str());
        }
    }
    grab_frame();
    hold_grab_ms_sum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    // The picture is taken while the CPU holds at the vblank; then it runs the next frame while the frontend shows this
    // one and the input for it arrives at the next call. A picture handed over where it lies (the GPU's read-back
    // buffer) is the frontend's until picture_done, the next vblank writing the target's other buffer (D36): the CPU
    // runs on while that copy lands and the frontend takes it.
    const auto release = [] {
        std::lock_guard g(machine->lock());
        released_to = machine->nv2a().vblanks() + 1;
        machine->set_frame_limit(released_to);
        cpu_released = true;
    };
    const bool in_place = frame_ptr != frame.data();
    if (pipeline) release();
    if (hw_video && vk_iface) {
        // The frame into the image of the frontend's sync index, on the GPU (D38): the GPU's picture where it lies, or
        // the CPU's through a staging copy. The index's last image is the frontend's until wait_sync_index returns.
        const mv::u32 index = vk_iface->get_sync_index(vk_iface->handle);
        vk_iface->wait_sync_index(vk_iface->handle);
        mv::pgraph::VulkanRenderer::HandedOver h;
        const bool ok = machine->renderer()->hand_over(index, in_place ? &frame_pic : nullptr, frame_w, frame_h, in_place ? nullptr : frame.data(), frame_pitch, h);
        if (in_place) machine->renderer()->picture_done();
        if (ok) {
            static retro_vulkan_image image;  // the frontend reads it until video_cb returns
            image.image_view = h.view;
            image.image_layout = VK_IMAGE_LAYOUT_GENERAL;
            image.create_info = *h.info;
            vk_iface->set_image(vk_iface->handle, &image, 0, nullptr, VK_QUEUE_FAMILY_IGNORED);
            video_cb(RETRO_HW_FRAME_BUFFER_VALID, h.width, h.height, 0);
        } else {
            video_cb(nullptr, frame_w, frame_h, 0);  // the last picture again
        }
    } else {
        if (in_place) machine->renderer()->picture_ready(frame_pic);
        video_cb(frame_ptr, frame_w, frame_h, frame_pitch);
        if (in_place) machine->renderer()->picture_done();
    }
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
            halted = direct_audio ? !machine->apu().frame_clock_running() : (machine->aci().channel(mv::Aci::PcmOut).sr & mv::Aci::kSrDch) != 0;
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
