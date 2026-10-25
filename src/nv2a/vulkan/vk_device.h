#pragma once
#include <string>
#include <vector>

#include "core/types.h"
#include "nv2a/vulkan/vk_loader.h"

namespace mv::vk {

// Where a buffer's memory lives. The device's own memory; host memory the CPU fills and the GPU reads (any
// host-visible type, write-combined included); host memory the CPU reads back, which wants a cached type: a load
// from write-combined memory is a bus round trip, and a 2 MiB surface read four bytes at a time took 90 ms.
enum class BufferMemory { Device, HostWrite, HostRead };

struct Buffer {
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize size{0};
    void* mapped{nullptr};
};

struct Image {
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    VkFormat format{VK_FORMAT_UNDEFINED};
    u32 width{0}, height{0}, depth{1}, levels{1}, layers{1};
    VkImageAspectFlags aspect{VK_IMAGE_ASPECT_COLOR_BIT};
    bool cube{false};
};

// One Vulkan instance, physical device, logical device and graphics queue, with the memory and command helpers
// the renderer needs. Headless: no surface, no swapchain.
class Device {
public:
    Device() = default;
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // `software` prefers a CPU implementation (lavapipe) when one is present; validation layers come from
    // the MV_VK_VALIDATE environment variable.
    bool init(std::string& error, bool software = false);
    bool ready() const { return device_ != VK_NULL_HANDLE; }
    // Exact occlusion counts, enabled when the device has them.
    bool occlusion_precise() const { return occlusion_precise_; }
    // Sampler border colours of our own (VK_EXT_custom_border_color): the NV2A's per-stage BORDER_COLOR. Without it a
    // sampler takes the nearest of Vulkan's three built-in borders; `_unformatted` = no image format needed.
    bool custom_border() const { return custom_border_; }
    bool custom_border_unformatted() const { return custom_border_unformatted_; }
    bool anisotropy() const { return anisotropy_; }  // samplerAnisotropy enabled

    const Functions& fn() const { return fn_; }
    VkDevice device() const { return device_; }
    VkPhysicalDevice physical() const { return physical_; }
    VkQueue queue() const { return queue_; }
    u32 queue_family() const { return queue_family_; }
    VkCommandPool pool() const { return pool_; }
    const std::string& name() const { return name_; }
    const VkPhysicalDeviceLimits& limits() const { return limits_; }

    u32 memory_type(u32 type_bits, VkMemoryPropertyFlags flags, bool* found) const;
    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, BufferMemory where, Buffer& out, std::string& error);
    void destroy_buffer(Buffer& b);
    bool create_image(VkFormat format, u32 width, u32 height, u32 depth, u32 levels, u32 layers, VkImageUsageFlags usage,
                      VkImageAspectFlags aspect, bool cube, Image& out, std::string& error);
    void destroy_image(Image& i);
    bool format_supports(VkFormat format, VkFormatFeatureFlags features) const;

    // One-shot command buffers: begin, record, then `finish` submits and waits.
    VkCommandBuffer begin_once();
    void finish_once(VkCommandBuffer cb);
    // Layout transitions for images that the helpers keep in GENERAL.
    void transition(VkCommandBuffer cb, const Image& image, VkImageLayout from, VkImageLayout to);

private:
    Functions fn_{};
    VkInstance instance_{VK_NULL_HANDLE};
    VkPhysicalDevice physical_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    bool occlusion_precise_{false};
    bool custom_border_{false}, custom_border_unformatted_{false};
    bool anisotropy_{false};
    VkQueue queue_{VK_NULL_HANDLE};
    u32 queue_family_{0};
    VkCommandPool pool_{VK_NULL_HANDLE};
    VkPhysicalDeviceMemoryProperties memory_{};
    VkPhysicalDeviceLimits limits_{};
    std::string name_;
};

}  // namespace mv::vk
