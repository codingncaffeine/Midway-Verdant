// mv-retro-host: a headless libretro frontend for testing the core the way Emutastic drives it. It loads the core
// library, answers the environment calls (system directory, pixel format, options, log), runs N frames, saves
// pictures of chosen frames and holds RetroPad buttons over chosen frame ranges.
//   mv-retro-host CORE.so SYSTEM_DIR [GAME] --frames N [--png-every K DIR [--png-from N --png-to M]] [--press BUTTON@FROM-TO ...]
//     [--reset@FRAME ...]: retro_reset before that frame, as a frontend's restart button does (Emutastic calls it on its loop).
//     [--option@FRAME KEY=VALUE ...]: the option changes before that frame, as a frontend's live options menu does (the next
//     GET_VARIABLE_UPDATE reports it).
//     [--hw-video]: offer the Vulkan hardware-render interface; the core hands frames over as images (read back for pictures).
// Buttons: a b x y l r l2 r2 start select up down left right.
#include <dlfcn.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <filesystem>
#include <format>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "debug/png.h"
#include "libretro/libretro.h"
#include "libretro/libretro_vulkan.h"
#include "nv2a/vulkan/vk_device.h"

namespace {

std::string system_dir;
unsigned pixel_format = RETRO_PIXEL_FORMAT_0RGB1555;
std::vector<unsigned char> last_frame;
unsigned last_w = 0, last_h = 0;
std::size_t last_pitch = 0;
unsigned long long frames_seen = 0, audio_frames = 0, audio_nonzero = 0;
std::FILE* wav_out = nullptr;  // --wav FILE: the core's audio as raw interleaved stereo 16-bit (a WAV header is written at the end)
unsigned long long wav_frames = 0;
// Audio frames per retro_run: a core on the console's clock hands over 800 or 801 (48 kHz over 59.94 Hz).
unsigned long long audio_this_call = 0;
std::map<unsigned long long, unsigned long long> audio_per_call;

struct Press {
    unsigned id;
    unsigned from, to;
};
std::vector<Press> presses;
std::vector<std::pair<std::string, std::string>> options;  // --option key=value
bool options_changed = false;  // since the core last asked (GET_VARIABLE_UPDATE)
unsigned current_frame = 0;

// ── --hw-video: the frontend side of the libretro Vulkan hardware-render interface (PLAN D38) ─────────────────────
// The core makes the device through the negotiation interface (create_device2 through our wrapper); each frame it
// hands over arrives as an image, read back here only for the frames a picture is saved of. Two sync indices, one
// fence each, as a frontend with a double-buffered swapchain has.
bool hw_requested = false, hw_accepted = false, hw_ready = false;
retro_hw_render_callback hw_cb{};
const retro_hw_render_context_negotiation_interface_vulkan* negotiation = nullptr;
mv::vk::Functions vkf{};
VkInstance vk_instance = VK_NULL_HANDLE;
VkDevice vk_device = VK_NULL_HANDLE;
VkQueue vk_queue = VK_NULL_HANDLE;
std::mutex queue_mutex;
retro_hw_render_interface_vulkan vk_iface{};
constexpr unsigned kSyncImages = 2;
std::array<VkFence, kSyncImages> sync_fences{};
std::array<bool, kSyncImages> sync_pending{};
std::array<VkCommandBuffer, kSyncImages> sync_cbs{};
unsigned sync_index = 0;
VkCommandPool hw_pool = VK_NULL_HANDLE;
VkImage handed_image = VK_NULL_HANDLE;  // set_image's, for this frame
std::unique_ptr<mv::vk::Device> hw_dev;  // the same device adopted, for the read-back buffer
mv::vk::Buffer readback;
bool readback_this_frame = false;
unsigned long long hw_frames = 0, hw_dupes = 0;

void hw_lock(void*) { queue_mutex.lock(); }
void hw_unlock(void*) { queue_mutex.unlock(); }
void hw_wait_index(unsigned i) {
    if (!sync_pending[i]) return;
    vkf.vkWaitForFences(vk_device, 1, &sync_fences[i], VK_TRUE, ~0ull);
    vkf.vkResetFences(vk_device, 1, &sync_fences[i]);
    sync_pending[i] = false;
}
void hw_set_image(void*, const retro_vulkan_image* image, uint32_t, const VkSemaphore*, uint32_t) { handed_image = image ? image->create_info.image : VK_NULL_HANDLE; }
uint32_t hw_get_sync_index(void*) { return sync_index; }
uint32_t hw_get_sync_index_mask(void*) { return (1u << kSyncImages) - 1; }
void hw_set_command_buffers(void*, uint32_t, const VkCommandBuffer*) {}
void hw_wait_sync_index(void*) { hw_wait_index(sync_index); }
void hw_set_signal_semaphore(void*, VkSemaphore) {}

VkDevice hw_create_device_wrapper(VkPhysicalDevice gpu, void*, const VkDeviceCreateInfo* ci) {
    VkDevice d = VK_NULL_HANDLE;
    return vkf.vkCreateDevice(gpu, ci, nullptr, &d) == VK_SUCCESS ? d : VK_NULL_HANDLE;
}

bool hw_create(std::string& error) {
    if (!mv::vk::load_global(vkf, error)) return false;
    static const VkApplicationInfo fallback{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "mv-retro-host", 0, "mv-retro-host", 0, (1u << 22) | (1u << 12)};
    const VkApplicationInfo* app = negotiation && negotiation->get_application_info ? negotiation->get_application_info() : &fallback;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = app ? app : &fallback;
    if (vkf.vkCreateInstance(&ici, nullptr, &vk_instance) != VK_SUCCESS) {
        error = "vkCreateInstance failed";
        return false;
    }
    mv::vk::load_instance(vkf, vk_instance);
    retro_vulkan_context ctx{};
    bool ok = false;
    if (negotiation && negotiation->interface_version >= 2 && negotiation->create_device2)
        ok = negotiation->create_device2(&ctx, vk_instance, VK_NULL_HANDLE, VK_NULL_HANDLE, vkf.vkGetInstanceProcAddr, hw_create_device_wrapper, nullptr);
    else if (negotiation && negotiation->create_device)
        ok = negotiation->create_device(&ctx, vk_instance, VK_NULL_HANDLE, VK_NULL_HANDLE, vkf.vkGetInstanceProcAddr, nullptr, 0, nullptr, 0, nullptr);
    if (!ok) {
        error = "the core made no device through the negotiation interface";
        return false;
    }
    vk_device = ctx.device;
    vk_queue = ctx.queue;
    mv::vk::load_device(vkf, vk_device);
    mv::vk::External e;
    e.instance = vk_instance;
    e.gpu = ctx.gpu;
    e.device = vk_device;
    e.queue = vk_queue;
    e.queue_family = ctx.queue_family_index;
    e.get_instance_proc_addr = vkf.vkGetInstanceProcAddr;
    e.lock_queue = hw_lock;
    e.unlock_queue = hw_unlock;
    hw_dev = std::make_unique<mv::vk::Device>();
    if (!hw_dev->adopt(e, error)) return false;
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx.queue_family_index;
    vkf.vkCreateCommandPool(vk_device, &pci, nullptr, &hw_pool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = hw_pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = kSyncImages;
    vkf.vkAllocateCommandBuffers(vk_device, &cai, sync_cbs.data());
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (VkFence& fence : sync_fences) vkf.vkCreateFence(vk_device, &fci, nullptr, &fence);
    vk_iface = {};
    vk_iface.interface_type = RETRO_HW_RENDER_INTERFACE_VULKAN;
    vk_iface.interface_version = RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION;
    vk_iface.instance = vk_instance;
    vk_iface.gpu = ctx.gpu;
    vk_iface.device = vk_device;
    vk_iface.get_device_proc_addr = vkf.vkGetDeviceProcAddr;
    vk_iface.get_instance_proc_addr = vkf.vkGetInstanceProcAddr;
    vk_iface.queue = vk_queue;
    vk_iface.queue_index = ctx.queue_family_index;
    vk_iface.set_image = hw_set_image;
    vk_iface.get_sync_index = hw_get_sync_index;
    vk_iface.get_sync_index_mask = hw_get_sync_index_mask;
    vk_iface.set_command_buffers = hw_set_command_buffers;
    vk_iface.wait_sync_index = hw_wait_sync_index;
    vk_iface.lock_queue = hw_lock;
    vk_iface.unlock_queue = hw_unlock;
    vk_iface.set_signal_semaphore = hw_set_signal_semaphore;
    hw_ready = true;
    hw_cb.context_reset();
    return true;
}

// A handed-over frame: what the frontend's own work would be (a submission after the core's, with this index's
// fence), and the image read back into last_frame when a picture of this frame is saved.
void hw_frame(unsigned w, unsigned h) {
    ++hw_frames;
    hw_wait_index(sync_index);
    VkCommandBuffer cb = sync_cbs[sync_index];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkf.vkBeginCommandBuffer(cb, &bi);
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;
    const bool read = readback_this_frame && handed_image;
    if (read && readback.size < bytes) {
        hw_dev->destroy_buffer(readback);
        std::string error;
        if (!hw_dev->create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, mv::vk::BufferMemory::HostRead, readback, error)) std::fprintf(stderr, "mv-retro-host: %s\n", error.c_str());
    }
    if (read && readback.buffer) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkf.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {w, h, 1};
        vkf.vkCmdCopyImageToBuffer(cb, handed_image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer, 1, &r);
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkf.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    vkf.vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    {
        std::lock_guard g(queue_mutex);
        vkf.vkQueueSubmit(vk_queue, 1, &si, sync_fences[sync_index]);
    }
    sync_pending[sync_index] = true;
    if (read && readback.buffer) {
        hw_wait_index(sync_index);
        last_w = w;
        last_h = h;
        last_pitch = static_cast<std::size_t>(w) * 4;
        const auto* p = static_cast<const unsigned char*>(readback.mapped);
        last_frame.assign(p, p + bytes);
    }
}

void hw_destroy() {
    if (!hw_ready) return;
    {
        std::lock_guard g(queue_mutex);
        vkf.vkQueueWaitIdle(vk_queue);
    }
    if (hw_cb.context_destroy) hw_cb.context_destroy();
    if (negotiation && negotiation->destroy_device) negotiation->destroy_device();
    hw_dev->destroy_buffer(readback);
    hw_dev.reset();
    for (VkFence f : sync_fences) vkf.vkDestroyFence(vk_device, f, nullptr);
    vkf.vkDestroyCommandPool(vk_device, hw_pool, nullptr);
    vkf.vkDestroyDevice(vk_device, nullptr);
    vkf.vkDestroyInstance(vk_instance, nullptr);
    hw_ready = false;
}

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
        case RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
            *static_cast<bool*>(data) = options_changed;
            options_changed = false;
            return true;
        case RETRO_ENVIRONMENT_SET_HW_RENDER: {
            auto* cb = static_cast<retro_hw_render_callback*>(data);
            if (!hw_requested || cb->context_type != RETRO_HW_CONTEXT_VULKAN) return false;
            hw_cb = *cb;
            hw_accepted = true;
            return true;
        }
        case RETRO_ENVIRONMENT_GET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_SUPPORT: {
            if (!hw_requested) return false;
            auto* n = static_cast<retro_hw_render_context_negotiation_interface*>(data);
            n->interface_version = n->interface_type == RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN ? 2 : 0;
            return true;
        }
        case RETRO_ENVIRONMENT_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE:
            negotiation = static_cast<const retro_hw_render_context_negotiation_interface_vulkan*>(data);
            return hw_requested;
        case RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE:
            if (!hw_ready) return false;
            *static_cast<const retro_hw_render_interface**>(data) = reinterpret_cast<const retro_hw_render_interface*>(&vk_iface);
            return true;
        case RETRO_ENVIRONMENT_GET_VARIABLE:
            for (const auto& [k, v] : options)
                if (k == static_cast<retro_variable*>(data)->key) {
                    static_cast<retro_variable*>(data)->value = v.c_str();
                    return true;
                }
            static_cast<retro_variable*>(data)->value = nullptr;  // the core's defaults
            return false;
        default:
            return false;
    }
}

void video(const void* data, unsigned w, unsigned h, size_t pitch) {
    ++frames_seen;
    if (data == RETRO_HW_FRAME_BUFFER_VALID) {
        if (hw_ready) hw_frame(w, h);
        return;
    }
    if (!data) {
        ++hw_dupes;
        return;
    }
    last_w = w;
    last_h = h;
    last_pitch = pitch;
    last_frame.assign(static_cast<const unsigned char*>(data), static_cast<const unsigned char*>(data) + pitch * h);
}

size_t audio_batch(const int16_t* data, size_t frames) {
    audio_frames += frames;
    audio_this_call += frames;
    for (size_t i = 0; i < frames * 2; ++i)
        if (data[i]) ++audio_nonzero;
    if (wav_out) {
        std::fwrite(data, 4, frames, wav_out);
        wav_frames += frames;
    }
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
    // MV_ALLOW_PROFILER=1 lets a sampling profiler that is not our parent (eu-stack, gdb -p) attach under Yama's
    // ptrace_scope 1, which otherwise allows only ancestors.
#if defined(__linux__)
    if (const char* p = std::getenv("MV_ALLOW_PROFILER"); p && *p == '1') prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
#endif
    if (argc < 3) {
        std::fprintf(stderr, "usage: mv-retro-host CORE.so SYSTEM_DIR [GAME] --frames N [--png-every K DIR [--png-from N --png-to M]] [--press BUTTON@FROM-TO ...] [--option KEY=VALUE ...] [--option@FRAME KEY=VALUE ...] [--wav FILE] [--frontend-ms N] [--realtime] [--reset@FRAME ...]\n");
        return 2;
    }
    const char* core_path = argv[1];
    system_dir = argv[2];
    std::string game;
    unsigned frames = 600, png_every = 0, png_from = 0, png_to = ~0u;  // --png-from/--png-to: every frame in [from, to] too
    double frontend_ms = 0;  // --frontend-ms N: time a frontend spends presenting each frame, spent here as a sleep
    bool realtime = false;  // --realtime: call retro_run no faster than the core's frame rate, as a frontend synced to the display does
    std::filesystem::path png_dir;
    std::vector<unsigned> resets;  // --reset@F
    std::vector<std::pair<unsigned, std::string>> option_changes;  // --option@F KEY=VALUE
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--frames" && i + 1 < argc) frames = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (a == "--png-from" && i + 1 < argc) png_from = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (a == "--png-to" && i + 1 < argc) png_to = static_cast<unsigned>(std::atoi(argv[++i]));
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
        } else if (a.rfind("--option@", 0) == 0 && i + 1 < argc) {
            option_changes.emplace_back(static_cast<unsigned>(std::atoi(a.c_str() + 9)), argv[++i]);
        } else if (a.rfind("--reset@", 0) == 0) {
            resets.push_back(static_cast<unsigned>(std::atoi(a.c_str() + 8)));
        } else if (a == "--frontend-ms" && i + 1 < argc) {
            frontend_ms = std::atof(argv[++i]);
        } else if (a == "--realtime") {
            realtime = true;
        } else if (a == "--hw-video") {
            hw_requested = true;  // offer the core the Vulkan hardware-render interface (PLAN D38)
        } else if (a == "--wav" && i + 1 < argc) {
            wav_out = std::fopen(argv[++i], "wb");
            if (!wav_out) {
                std::fprintf(stderr, "mv-retro-host: cannot write %s\n", argv[i]);
                return 2;
            }
            const unsigned char blank[44] = {};
            std::fwrite(blank, 1, sizeof blank, wav_out);
        } else if (a == "--option" && i + 1 < argc) {
            const std::string kv = argv[++i];
            const auto eq = kv.find('=');
            if (eq != std::string::npos) options.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
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
    if (hw_accepted) {
        std::string error;
        if (!hw_create(error)) {
            std::fprintf(stderr, "mv-retro-host: --hw-video: %s\n", error.c_str());
            return 1;
        }
        std::printf("hw video: the core's Vulkan device, frames handed over as images\n");
    }
    retro_system_av_info av{};
    sym<void (*)(retro_system_av_info*)>(lib, "retro_get_system_av_info")(&av);
    std::printf("av: %ux%u base, %.3f fps, %.0f Hz\n", av.geometry.base_width, av.geometry.base_height, av.timing.fps, av.timing.sample_rate);
    if (png_every) std::filesystem::create_directories(png_dir);
    auto run = sym<void (*)()>(lib, "retro_run");
    auto reset = sym<void (*)()>(lib, "retro_reset");
    const auto t0 = std::chrono::steady_clock::now();
    double call_ms_max = 0;
    unsigned long long over17 = 0, over33 = 0;  // calls slower than a frame and than two (a frontend shows them as hitches)
    for (current_frame = 0; current_frame < frames; ++current_frame) {
        if (std::find(resets.begin(), resets.end(), current_frame) != resets.end()) {
            std::printf("frame %u: retro_reset\n", current_frame);
            reset();
        }
        for (const auto& [at, kv] : option_changes) {
            if (at != current_frame) continue;
            const auto eq = kv.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = kv.substr(0, eq), value = kv.substr(eq + 1);
            auto it = std::find_if(options.begin(), options.end(), [&](const auto& o) { return o.first == key; });
            if (it != options.end()) it->second = value;
            else options.emplace_back(key, value);
            options_changed = true;
            std::printf("frame %u: option %s = %s\n", current_frame, key.c_str(), value.c_str());
        }
        audio_this_call = 0;
        const bool save = png_every && ((current_frame + 1) % png_every == 0 || (png_from && current_frame + 1 >= png_from && current_frame + 1 <= png_to));
        readback_this_frame = save;
        const auto c0 = std::chrono::steady_clock::now();
        run();
        if (hw_ready) sync_index = (sync_index + 1) % kSyncImages;
        const double call_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
        call_ms_max = std::max(call_ms_max, call_ms);
        over17 += call_ms > 1000.0 / 60.0;
        over33 += call_ms > 2000.0 / 60.0;
        ++audio_per_call[audio_this_call];
        if (save) save_png(png_dir / std::format("frame-{:06}.png", current_frame + 1));
        if (frontend_ms > 0) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(frontend_ms));
        if (realtime) std::this_thread::sleep_until(t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>((current_frame + 1) / av.timing.fps)));
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("calls over one frame %llu, over two %llu\n", over17, over33);
    std::printf("ran %u frames in %.2f s (%.1f fps, slowest call %.1f ms); %llu video callbacks, last %ux%u; %llu audio frames, %llu non-zero samples\n", frames,
                secs, frames / secs, call_ms_max, frames_seen, last_w, last_h, audio_frames, audio_nonzero);
    // The most common audio frame counts per call, most frequent first.
    std::vector<std::pair<unsigned long long, unsigned long long>> hist(audio_per_call.begin(), audio_per_call.end());
    std::sort(hist.begin(), hist.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string counts;
    for (std::size_t i = 0; i < hist.size() && i < 8; ++i) counts += std::format(" {}x{}", hist[i].first, hist[i].second);
    std::printf("audio frames per call (count x calls):%s%s\n", counts.c_str(), hist.size() > 8 ? " ..." : "");
    if (wav_out) {
        // RIFF header for 48 kHz stereo 16-bit PCM.
        auto le32 = [](unsigned char* p, unsigned v) { for (int k = 0; k < 4; ++k) p[k] = static_cast<unsigned char>(v >> (8 * k)); };
        unsigned char h[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 2, 0};
        const unsigned bytes = static_cast<unsigned>(wav_frames * 4);
        le32(h + 4, 36 + bytes);
        le32(h + 24, 48000);
        le32(h + 28, 48000 * 4);
        h[32] = 4;
        h[34] = 16;
        std::memcpy(h + 36, "data", 4);
        le32(h + 40, bytes);
        std::fseek(wav_out, 0, SEEK_SET);
        std::fwrite(h, 1, sizeof h, wav_out);
        std::fclose(wav_out);
    }
    sym<void (*)()>(lib, "retro_unload_game")();
    if (hw_ready) std::printf("hw video: %llu frames handed over, %llu repeated\n", hw_frames, hw_dupes);
    hw_destroy();
    sym<void (*)()>(lib, "retro_deinit")();
    dlclose(lib);
    return 0;
}
