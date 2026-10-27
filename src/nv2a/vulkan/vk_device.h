#pragma once
#include <functional>
#include <map>
#include <mutex>
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
    VkImageUsageFlags usage{0};
    VkImageCreateFlags flags{0};
    // Placed in one of the device's shared texture blocks (create_image with pooled): which, where and how much;
    // memory is then the block's and is not freed with the image.
    u32 block{~0u};
    VkDeviceSize offset{0}, bytes{0};
};

// What the device was created with, of the optional things the renderer uses.
struct Enabled {
    bool occlusion_precise{false};
    bool custom_border{false}, custom_border_unformatted{false};
    bool anisotropy{false};
    bool depth_clamp{false};  // depthClamp: primitives are not clipped at the near and far planes (the console clamps or culls per pixel)
    bool bc_textures{false};  // textureCompressionBC: DXT textures sampled as BC1-3 blocks
    bool cache_control{false};  // pipelines can be asked of the cache alone (FAIL_ON_PIPELINE_COMPILE_REQUIRED)
};

// A device a libretro frontend owns (the Vulkan hardware-render interface, PLAN D38): its handles, the lock it
// wants around every use of its queue, and what was enabled on it (`negotiated`: create_logical made it through the
// frontend's wrapper; otherwise the frontend made it alone and none of the optional features are assumed).
struct External {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice gpu{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    u32 queue_family{0};
    PFN_vkGetInstanceProcAddr get_instance_proc_addr{nullptr};
    void (*lock_queue)(void* handle){nullptr};
    void (*unlock_queue)(void* handle){nullptr};
    void* handle{nullptr};
    bool negotiated{false};
    Enabled enabled;
};

// One Vulkan instance, physical device, logical device and graphics queue, with the memory and command helpers
// the renderer needs. Headless: no surface, no swapchain. Its own (init), or a frontend's (adopt).
class Device {
public:
    Device() = default;
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // `software` prefers a CPU implementation (lavapipe) when one is present; validation layers come from
    // the MV_VK_VALIDATE environment variable.
    bool init(std::string& error, bool software = false);
    // The frontend's device: nothing of it is destroyed here, and every submit takes the frontend's queue lock.
    bool adopt(const External& e, std::string& error);
    // The logical device on `gpu` with the queue family, features and extensions the renderer uses, made through
    // `create` (vkCreateDevice, or a frontend's wrapper that may add its own). `fn` holds the instance's functions.
    static VkDevice create_logical(const Functions& fn, VkPhysicalDevice gpu, u32& queue_family, Enabled& enabled,
                                   const std::function<VkDevice(const VkDeviceCreateInfo&)>& create, std::string& error);
    // Submits under the frontend's queue lock (when there is one).
    VkResult submit(u32 count, const VkSubmitInfo* infos, VkFence fence);
    void wait_queue_idle();
    bool ready() const { return device_ != VK_NULL_HANDLE; }
    // Exact occlusion counts, enabled when the device has them.
    bool occlusion_precise() const { return enabled_.occlusion_precise; }
    // Sampler border colours of our own (VK_EXT_custom_border_color): the NV2A's per-stage BORDER_COLOR. Without it a
    // sampler takes the nearest of Vulkan's three built-in borders; `_unformatted` = no image format needed.
    bool custom_border() const { return enabled_.custom_border; }
    bool custom_border_unformatted() const { return enabled_.custom_border_unformatted; }
    bool anisotropy() const { return enabled_.anisotropy; }  // samplerAnisotropy enabled
    bool depth_clamp() const { return enabled_.depth_clamp; }
    bool bc_textures() const { return enabled_.bc_textures; }
    bool cache_control() const { return enabled_.cache_control; }  // pipeline creation from the cache alone
    bool external() const { return external_; }  // a frontend's device (adopt)

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
                      VkImageAspectFlags aspect, bool cube, Image& out, std::string& error, VkImageCreateFlags flags = 0, bool pooled = false);
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
    Enabled enabled_;
    bool external_{false};
    void (*lock_queue_)(void*){nullptr};
    void (*unlock_queue_)(void*){nullptr};
    void* queue_handle_{nullptr};
    bool finish_setup(std::string& error);  // the device's name, memory, limits and command pool
    VkQueue queue_{VK_NULL_HANDLE};
    u32 queue_family_{0};
    VkCommandPool pool_{VK_NULL_HANDLE};
    VkPhysicalDeviceMemoryProperties memory_{};
    VkPhysicalDeviceLimits limits_{};
    std::string name_;
    // Shared blocks of device memory that sampled textures are placed in (a vkAllocateMemory each was ~0.03-0.07 ms,
    // most of what a new texture cost: Crimson Skies met 176 in one frame). First fit, freed ranges merged.
    struct Block {
        VkDeviceMemory memory{VK_NULL_HANDLE};
        u32 type{0};
        std::map<VkDeviceSize, VkDeviceSize> free;  // offset -> size
    };
    static constexpr VkDeviceSize kBlockBytes = 64ull << 20;
    std::vector<Block> blocks_;
    std::mutex blocks_mutex_;
    bool place(u32 type, VkDeviceSize bytes, VkDeviceSize alignment, u32& block, VkDeviceSize& offset);
    void release(u32 block, VkDeviceSize offset, VkDeviceSize bytes);
};

}  // namespace mv::vk
