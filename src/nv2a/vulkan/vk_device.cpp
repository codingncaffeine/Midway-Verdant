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
        fn_.vkDeviceWaitIdle(device_);
        if (pool_) fn_.vkDestroyCommandPool(device_, pool_, nullptr);
        fn_.vkDestroyDevice(device_, nullptr);
    }
    if (instance_) fn_.vkDestroyInstance(instance_, nullptr);
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
            name_ = p.deviceName;
            limits_ = p.limits;
        }
    }
    u32 families = 0;
    fn_.vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, nullptr);
    std::vector<VkQueueFamilyProperties> qf(families);
    fn_.vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, qf.data());
    bool found = false;
    for (u32 i = 0; i < families; ++i) {
        if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queue_family_ = i;
            found = true;
            break;
        }
    }
    if (!found) {
        error = "the Vulkan device has no graphics queue";
        return false;
    }
    fn_.vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures features{};
    VkPhysicalDeviceFeatures available{};
    fn_.vkGetPhysicalDeviceFeatures(physical_, &available);
    features.fillModeNonSolid = available.fillModeNonSolid;
    features.samplerAnisotropy = available.samplerAnisotropy;
    anisotropy_ = available.samplerAnisotropy;
    features.independentBlend = available.independentBlend;
    features.largePoints = available.largePoints;
    features.occlusionQueryPrecise = available.occlusionQueryPrecise;
    occlusion_precise_ = available.occlusionQueryPrecise;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &features;
    // Custom sampler border colours, where the device offers them.
    std::vector<const char*> extensions;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT border{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    {
        u32 n_ext = 0;
        fn_.vkEnumerateDeviceExtensionProperties(physical_, nullptr, &n_ext, nullptr);
        std::vector<VkExtensionProperties> props(n_ext);
        fn_.vkEnumerateDeviceExtensionProperties(physical_, nullptr, &n_ext, props.data());
        const bool offered = std::ranges::any_of(props, [](const VkExtensionProperties& p) {
            return std::string_view(p.extensionName) == VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME;
        });
        if (offered && fn_.vkGetPhysicalDeviceFeatures2) {
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &border;
            fn_.vkGetPhysicalDeviceFeatures2(physical_, &f2);
            if (border.customBorderColors) {
                extensions.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
                border.pNext = nullptr;
                dci.pNext = &border;
                custom_border_ = true;
                custom_border_unformatted_ = border.customBorderColorWithoutFormat;
            }
        }
    }
    dci.enabledExtensionCount = static_cast<u32>(extensions.size());
    dci.ppEnabledExtensionNames = extensions.data();
    if (VkResult r = fn_.vkCreateDevice(physical_, &dci, nullptr, &device_); r != VK_SUCCESS) {
        error = std::format("vkCreateDevice: {}", result_name(r));
        return false;
    }
    load_device(fn_, device_);
    fn_.vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family_;
    if (VkResult r = fn_.vkCreateCommandPool(device_, &pci, nullptr, &pool_); r != VK_SUCCESS) {
        error = std::format("vkCreateCommandPool: {}", result_name(r));
        return false;
    }
    return true;
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
                          VkImageAspectFlags aspect, bool cube, Image& out, std::string& error) {
    out = Image{};
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.flags = cube ? static_cast<u32>(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) : 0u;
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
    fn_.vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
    fn_.vkQueueWaitIdle(queue_);
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
