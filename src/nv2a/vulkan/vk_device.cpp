#include "nv2a/vulkan/vk_device.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string_view>
#include <vector>

namespace mv::vk {


Device::~Device() {
    if (device_) {
        wait_queue_idle();
        if (pool_) fn_.vkDestroyCommandPool(device_, pool_, nullptr);
        if (!external_) fn_.vkDestroyDevice(device_, nullptr);
    }
    if (instance_ && !external_) fn_.vkDestroyInstance(instance_, nullptr);
}

bool Device::init(std::string& error, bool software) {
    if (!load_global(fn_, error)) return false;

    std::vector<const char*> layers;
    if (const char* v = std::getenv("MV_VK_VALIDATE"); v && *v && *v != '0') {
        u32 count = 0;
        fn_.vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> props(count);
        fn_.vkEnumerateInstanceLayerProperties(&count, props.data());
        for (const auto& p : props)
            if (std::strcmp(p.layerName, "VK_LAYER_KHRONOS_validation") == 0) layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Midway Verdant";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = static_cast<u32>(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    if (VkResult r = fn_.vkCreateInstance(&ici, nullptr, &instance_); r != VK_SUCCESS) {
        error = std::format("vkCreateInstance: {}", result_name(r));
        return false;
    }
    load_instance(fn_, instance_);

    u32 count = 0;
    fn_.vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    fn_.vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    if (devices.empty()) {
        error = "no Vulkan device (no driver installed, or none usable headless)";
        return false;
    }
    // Prefer a discrete GPU, or the CPU implementation when asked for.
    int best_score = -1;
    for (VkPhysicalDevice d : devices) {
        VkPhysicalDeviceProperties p{};
        fn_.vkGetPhysicalDeviceProperties(d, &p);
        int score = 1;
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = software ? 2 : 4;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = software ? 2 : 3;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) score = software ? 5 : 1;
        if (score > best_score) {
            best_score = score;
            physical_ = d;
        }
    }
    device_ = create_logical(fn_, physical_, queue_family_, enabled_, [&](const VkDeviceCreateInfo& dci) {
        VkDevice d = VK_NULL_HANDLE;
        if (VkResult r = fn_.vkCreateDevice(physical_, &dci, nullptr, &d); r != VK_SUCCESS) error = std::format("vkCreateDevice: {}", result_name(r));
        return d;
    }, error);
    if (!device_) return false;
    load_device(fn_, device_);
    fn_.vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    return finish_setup(error);
}

bool Device::adopt(const External& e, std::string& error) {
    if (!e.instance || !e.gpu || !e.device || !e.queue || !e.get_instance_proc_addr) {
        error = "the frontend's Vulkan context is incomplete";
        return false;
    }
    external_ = true;
    fn_.vkGetInstanceProcAddr = e.get_instance_proc_addr;
    instance_ = e.instance;
    load_instance(fn_, instance_);
    physical_ = e.gpu;
    device_ = e.device;
    queue_ = e.queue;
    queue_family_ = e.queue_family;
    lock_queue_ = e.lock_queue;
    unlock_queue_ = e.unlock_queue;
    queue_handle_ = e.handle;
    enabled_ = e.negotiated ? e.enabled : Enabled{};
    load_device(fn_, device_);
    return finish_setup(error);
}

bool Device::finish_setup(std::string& error) {
    VkPhysicalDeviceProperties p{};
    fn_.vkGetPhysicalDeviceProperties(physical_, &p);
    name_ = p.deviceName;
    limits_ = p.limits;
    fn_.vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family_;
    if (VkResult r = fn_.vkCreateCommandPool(device_, &pci, nullptr, &pool_); r != VK_SUCCESS) {
        error = std::format("vkCreateCommandPool: {}", result_name(r));
        return false;
    }
    return true;
}

VkDevice Device::create_logical(const Functions& fn, VkPhysicalDevice gpu, u32& queue_family, Enabled& enabled,
                                const std::function<VkDevice(const VkDeviceCreateInfo&)>& create, std::string& error) {
    // One queue of a family that does graphics and compute (the libretro interface asks both of the frontend's queue).
    u32 families = 0;
    fn.vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, nullptr);
    std::vector<VkQueueFamilyProperties> qf(families);
    fn.vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, qf.data());
    bool found = false;
    for (u32 i = 0; i < families && !found; ++i) {
        if ((qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
            queue_family = i;
            found = true;
        }
    }
    if (!found) {
        error = "the Vulkan device has no queue that does graphics and compute";
        return VK_NULL_HANDLE;
    }
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures features{};
    VkPhysicalDeviceFeatures available{};
    fn.vkGetPhysicalDeviceFeatures(gpu, &available);
    enabled = Enabled{};
    features.fillModeNonSolid = available.fillModeNonSolid;
    features.samplerAnisotropy = available.samplerAnisotropy;
    enabled.anisotropy = available.samplerAnisotropy;
    features.depthClamp = available.depthClamp;
    enabled.depth_clamp = available.depthClamp;
    features.independentBlend = available.independentBlend;
    features.largePoints = available.largePoints;
    features.occlusionQueryPrecise = available.occlusionQueryPrecise;
    enabled.occlusion_precise = available.occlusionQueryPrecise;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &features;
    // Custom sampler border colours, where the device offers them.
    std::vector<const char*> extensions;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT border{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    {
        u32 n_ext = 0;
        fn.vkEnumerateDeviceExtensionProperties(gpu, nullptr, &n_ext, nullptr);
        std::vector<VkExtensionProperties> props(n_ext);
        fn.vkEnumerateDeviceExtensionProperties(gpu, nullptr, &n_ext, props.data());
        const bool offered = std::ranges::any_of(props, [](const VkExtensionProperties& p) {
            return std::string_view(p.extensionName) == VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME;
        });
        if (offered && fn.vkGetPhysicalDeviceFeatures2) {
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &border;
            fn.vkGetPhysicalDeviceFeatures2(gpu, &f2);
            if (border.customBorderColors) {
                extensions.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
                border.pNext = nullptr;
                dci.pNext = &border;
                enabled.custom_border = true;
                enabled.custom_border_unformatted = border.customBorderColorWithoutFormat;
            }
        }
    }
    // Pipelines made from the cache alone or not at all (VK_EXT_pipeline_creation_cache_control): with background
    // compiles a pipeline the cache holds is made at once, and only one the driver has to compile goes to a worker
    // (draws waiting for every new pipeline left a boot animation's last effect undrawn for two seconds).
    VkPhysicalDevicePipelineCreationCacheControlFeaturesEXT cache_control{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT};
    {
        u32 n_ext = 0;
        fn.vkEnumerateDeviceExtensionProperties(gpu, nullptr, &n_ext, nullptr);
        std::vector<VkExtensionProperties> props(n_ext);
        fn.vkEnumerateDeviceExtensionProperties(gpu, nullptr, &n_ext, props.data());
        const bool offered = std::ranges::any_of(props, [](const VkExtensionProperties& p) {
            return std::string_view(p.extensionName) == VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME;
        });
        if (offered && fn.vkGetPhysicalDeviceFeatures2) {
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &cache_control;
            fn.vkGetPhysicalDeviceFeatures2(gpu, &f2);
            if (cache_control.pipelineCreationCacheControl) {
                extensions.push_back(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME);
                cache_control.pNext = const_cast<void*>(dci.pNext);
                dci.pNext = &cache_control;
                enabled.cache_control = true;
            }
        }
    }
    dci.enabledExtensionCount = static_cast<u32>(extensions.size());
    dci.ppEnabledExtensionNames = extensions.data();
    const VkDevice device = create(dci);
    if (!device) {
        if (error.empty()) error = "the Vulkan device could not be created";
        enabled = Enabled{};
    }
    return device;
}

VkResult Device::submit(u32 count, const VkSubmitInfo* infos, VkFence fence) {
    if (lock_queue_) lock_queue_(queue_handle_);
    const VkResult r = fn_.vkQueueSubmit(queue_, count, infos, fence);
    if (unlock_queue_) unlock_queue_(queue_handle_);
    return r;
}

void Device::wait_queue_idle() {
    if (!queue_) return;
    if (lock_queue_) lock_queue_(queue_handle_);
    fn_.vkQueueWaitIdle(queue_);
    if (unlock_queue_) unlock_queue_(queue_handle_);
}

u32 Device::memory_type(u32 type_bits, VkMemoryPropertyFlags flags, bool* found) const {
    for (u32 i = 0; i < memory_.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (memory_.memoryTypes[i].propertyFlags & flags) == flags) {
            if (found) *found = true;
            return i;
        }
    }
    if (found) *found = false;
    return 0;
}

bool Device::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, BufferMemory where, Buffer& out, std::string& error) {
    out = Buffer{};
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (VkResult r = fn_.vkCreateBuffer(device_, &bci, nullptr, &out.buffer); r != VK_SUCCESS) {
        error = std::format("vkCreateBuffer: {}", result_name(r));
        return false;
    }
    VkMemoryRequirements req{};
    fn_.vkGetBufferMemoryRequirements(device_, out.buffer, &req);
    bool found = false;
    const bool host_visible = where != BufferMemory::Device;
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    u32 type = 0;
    if (where == BufferMemory::HostRead) type = memory_type(req.memoryTypeBits, host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &found);
    if (!found) type = memory_type(req.memoryTypeBits, host_visible ? host : static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), &found);
    if (!found) {
        error = "no suitable memory type for a buffer";
        destroy_buffer(out);
        return false;
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (VkResult r = fn_.vkAllocateMemory(device_, &mai, nullptr, &out.memory); r != VK_SUCCESS) {
        error = std::format("vkAllocateMemory (buffer, {} bytes): {}", static_cast<unsigned long long>(req.size), result_name(r));
        destroy_buffer(out);
        return false;
    }
    fn_.vkBindBufferMemory(device_, out.buffer, out.memory, 0);
    out.size = size;
    if (host_visible) {
        if (VkResult r = fn_.vkMapMemory(device_, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped); r != VK_SUCCESS) {
            error = std::format("vkMapMemory: {}", result_name(r));
            destroy_buffer(out);
            return false;
        }
    }
    return true;
}

void Device::destroy_buffer(Buffer& b) {
    if (b.mapped) fn_.vkUnmapMemory(device_, b.memory);
    if (b.buffer) fn_.vkDestroyBuffer(device_, b.buffer, nullptr);
    if (b.memory) fn_.vkFreeMemory(device_, b.memory, nullptr);
    b = Buffer{};
}

bool Device::create_image(VkFormat format, u32 width, u32 height, u32 depth, u32 levels, u32 layers, VkImageUsageFlags usage,
                          VkImageAspectFlags aspect, bool cube, Image& out, std::string& error, VkImageCreateFlags flags) {
    out = Image{};
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.flags = flags | (cube ? static_cast<u32>(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) : 0u);
    ici.imageType = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {width, height, depth};
    ici.mipLevels = levels;
    ici.arrayLayers = layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (VkResult r = fn_.vkCreateImage(device_, &ici, nullptr, &out.image); r != VK_SUCCESS) {
        error = std::format("vkCreateImage ({}x{} format {}): {}", width, height, static_cast<int>(format), result_name(r));
        return false;
    }
    VkMemoryRequirements req{};
    fn_.vkGetImageMemoryRequirements(device_, out.image, &req);
    bool found = false;
    const u32 type = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &found);
    if (!found) {
        error = "no device-local memory type for an image";
        destroy_image(out);
        return false;
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (VkResult r = fn_.vkAllocateMemory(device_, &mai, nullptr, &out.memory); r != VK_SUCCESS) {
        error = std::format("vkAllocateMemory (image): {}", result_name(r));
        destroy_image(out);
        return false;
    }
    fn_.vkBindImageMemory(device_, out.image, out.memory, 0);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = out.image;
    vci.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : (depth > 1 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D);
    vci.format = format;
    vci.subresourceRange = {aspect, 0, levels, 0, layers};
    if (VkResult r = fn_.vkCreateImageView(device_, &vci, nullptr, &out.view); r != VK_SUCCESS) {
        error = std::format("vkCreateImageView: {}", result_name(r));
        destroy_image(out);
        return false;
    }
    out.format = format;
    out.width = width;
    out.height = height;
    out.depth = depth;
    out.levels = levels;
    out.layers = layers;
    out.aspect = aspect;
    out.cube = cube;
    return true;
}

void Device::destroy_image(Image& i) {
    if (i.view) fn_.vkDestroyImageView(device_, i.view, nullptr);
    if (i.image) fn_.vkDestroyImage(device_, i.image, nullptr);
    if (i.memory) fn_.vkFreeMemory(device_, i.memory, nullptr);
    i = Image{};
}

bool Device::format_supports(VkFormat format, VkFormatFeatureFlags features) const {
    VkFormatProperties p{};
    fn_.vkGetPhysicalDeviceFormatProperties(physical_, format, &p);
    return (p.optimalTilingFeatures & features) == features;
}

VkCommandBuffer Device::begin_once() {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    fn_.vkAllocateCommandBuffers(device_, &cai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    fn_.vkBeginCommandBuffer(cb, &bi);
    return cb;
}

void Device::finish_once(VkCommandBuffer cb) {
    fn_.vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    submit(1, &si, VK_NULL_HANDLE);
    wait_queue_idle();
    fn_.vkFreeCommandBuffers(device_, pool_, 1, &cb);
}

void Device::transition(VkCommandBuffer cb, const Image& image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image.image;
    b.subresourceRange = {image.aspect, 0, image.levels, 0, image.layers};
    fn_.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

}  // namespace mv::vk
