#include "ui/window.h"

#include <SDL3/SDL.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>

#include "core/machine.h"

namespace mv::ui {

struct ScanoutWindow::Impl {
    SDL_Window* window{nullptr};
    SDL_Renderer* renderer{nullptr};
    SDL_Texture* texture{nullptr};
    u32 width{0}, height{0};
    bool sdl_up{false};
};

ScanoutWindow::ScanoutWindow() : impl_(new Impl) {}

ScanoutWindow::~ScanoutWindow() {
    if (impl_->texture) SDL_DestroyTexture(impl_->texture);
    if (impl_->renderer) SDL_DestroyRenderer(impl_->renderer);
    if (impl_->window) SDL_DestroyWindow(impl_->window);
    if (impl_->sdl_up) SDL_Quit();
    delete impl_;
}

bool ScanoutWindow::open(std::string& error) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        error = std::string("SDL_Init: ") + SDL_GetError();
        return false;
    }
    impl_->sdl_up = true;
    if (!SDL_CreateWindowAndRenderer("Midway Verdant", 1280, 960, SDL_WINDOW_RESIZABLE, &impl_->window, &impl_->renderer)) {
        error = std::string("SDL_CreateWindowAndRenderer: ") + SDL_GetError();
        return false;
    }
    SDL_SetRenderVSync(impl_->renderer, 1);
    std::fprintf(stderr, "window: video driver %s, renderer %s\n", SDL_GetCurrentVideoDriver(), SDL_GetRendererName(impl_->renderer));
    // The icon ships beside the executable's assets in a build tree and under the package's share directory.
    for (const char* candidate : {"assets/branding/icon.bmp", "../assets/branding/icon.bmp", "../../assets/branding/icon.bmp",
                                  "../share/midway-verdant/icon.bmp"}) {
        if (!std::filesystem::exists(candidate)) continue;
        if (SDL_Surface* icon = SDL_LoadBMP(candidate)) {
            SDL_SetWindowIcon(impl_->window, icon);
            SDL_DestroySurface(icon);
            break;
        }
    }
    return true;
}

unsigned ScanoutWindow::run(Machine& machine, std::atomic<bool>& stop) {
    unsigned presented = 0;
    Bus& bus = machine.bus();
    while (!stop.load()) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT || (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE)) stop = true;
        }
        // The CRTC's registers belong to the machine; read them under its lock, the pixels without (a torn frame
        // is what a real display shows too).
        u32 start, width, height, pitch;
        {
            std::lock_guard g(machine.lock());
            Nv2aStub& gpu = machine.nv2a();
            start = gpu.framebuffer_start();
            if (gpu.scanout_programmed()) {
                width = gpu.scanout_width();
                height = gpu.scanout_height();
                pitch = gpu.scanout_pitch();
            } else {
                width = 640;
                height = 480;
                pitch = 640 * 4;
            }
        }
        const bool sane = width >= 64 && width <= 2048 && height >= 64 && height <= 1080 && pitch >= width * 4;
        if (sane && (width != impl_->width || height != impl_->height)) {
            if (impl_->texture) SDL_DestroyTexture(impl_->texture);
            // The frame buffer's alpha byte means nothing to the display (the console's scanout ignores it, and the
            // kernel and programs leave it zero): an alpha-less format, and no blending, or the picture vanishes.
            impl_->texture = SDL_CreateTexture(impl_->renderer, SDL_PIXELFORMAT_XRGB8888, SDL_TEXTUREACCESS_STREAMING,
                                               static_cast<int>(width), static_cast<int>(height));
            SDL_SetTextureBlendMode(impl_->texture, SDL_BLENDMODE_NONE);
            SDL_SetTextureScaleMode(impl_->texture, SDL_SCALEMODE_NEAREST);
            SDL_SetRenderLogicalPresentation(impl_->renderer, static_cast<int>(width), static_cast<int>(height),
                                             SDL_LOGICAL_PRESENTATION_LETTERBOX);
            impl_->width = width;
            impl_->height = height;
            std::fprintf(stderr, "window: scanout %ux%u, pitch %u, frame buffer at %08x\n", width, height, pitch, start);
        }
        if (!sane && !impl_->texture) {
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "window: no usable mode yet (%ux%u pitch %u)\n", width, height, pitch);
            }
        }
        SDL_SetRenderDrawColor(impl_->renderer, 0, 0, 0, 255);
        SDL_RenderClear(impl_->renderer);
        if (impl_->texture && sane && start != 0 && static_cast<u64>(start) + static_cast<u64>(pitch) * height <= bus.ram_size()) {
            SDL_UpdateTexture(impl_->texture, nullptr, bus.ram() + start, static_cast<int>(pitch));
            SDL_RenderTexture(impl_->renderer, impl_->texture, nullptr, nullptr);
        }
        SDL_RenderPresent(impl_->renderer);
        ++presented;
    }
    return presented;
}

}  // namespace mv::ui
