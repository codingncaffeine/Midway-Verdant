#include "nv2a/vulkan/vk_loader.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mv::vk {

namespace {
void* g_library = nullptr;

void* open_library(const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}
}  // namespace

bool load_global(Functions& fn, std::string& error) {
    if (!g_library) {
#if defined(_WIN32)
        g_library = open_library("vulkan-1.dll");
#elif defined(__APPLE__)
        g_library = open_library("libvulkan.1.dylib");
        if (!g_library) g_library = open_library("libMoltenVK.dylib");
#else
        g_library = open_library("libvulkan.so.1");
        if (!g_library) g_library = open_library("libvulkan.so");
#endif
    }
    if (!g_library) {
        error = "no Vulkan loader library on this machine";
        return false;
    }
    fn.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(symbol(g_library, "vkGetInstanceProcAddr"));
    if (!fn.vkGetInstanceProcAddr) {
        error = "the Vulkan loader has no vkGetInstanceProcAddr";
        return false;
    }
#define MV_VK_LOAD(name) fn.name = reinterpret_cast<PFN_##name>(fn.vkGetInstanceProcAddr(nullptr, #name));
    MV_VK_GLOBAL_FUNCTIONS(MV_VK_LOAD)
#undef MV_VK_LOAD
    if (!fn.vkCreateInstance) {
        error = "the Vulkan loader has no vkCreateInstance";
        return false;
    }
    return true;
}

void load_instance(Functions& fn, VkInstance instance) {
#define MV_VK_LOAD(name) fn.name = reinterpret_cast<PFN_##name>(fn.vkGetInstanceProcAddr(instance, #name));
    MV_VK_INSTANCE_FUNCTIONS(MV_VK_LOAD)
#undef MV_VK_LOAD
}

void load_device(Functions& fn, VkDevice device) {
#define MV_VK_LOAD(name) fn.name = reinterpret_cast<PFN_##name>(fn.vkGetDeviceProcAddr(device, #name));
    MV_VK_DEVICE_FUNCTIONS(MV_VK_LOAD)
#undef MV_VK_LOAD
}

const char* result_name(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        default: return "VK_ERROR (other)";
    }
}

}  // namespace mv::vk
