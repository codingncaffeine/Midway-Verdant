#pragma once
// The Vulkan loader is opened at run time and every entry point we use is fetched through it: no link-time
// dependency on a Vulkan SDK, so the build is the same on every platform and a machine without a driver just
// reports "no Vulkan" instead of failing to start.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <string>

namespace mv::vk {

#define MV_VK_GLOBAL_FUNCTIONS(X) \
    X(vkCreateInstance)           \
    X(vkEnumerateInstanceExtensionProperties) \
    X(vkEnumerateInstanceLayerProperties)

#define MV_VK_INSTANCE_FUNCTIONS(X)       \
    X(vkDestroyInstance)                  \
    X(vkEnumeratePhysicalDevices)         \
    X(vkGetPhysicalDeviceProperties)      \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceFormatProperties) \
    X(vkGetPhysicalDeviceFeatures)        \
    X(vkGetPhysicalDeviceFeatures2)       \
    X(vkCreateDevice)                     \
    X(vkGetDeviceProcAddr)                \
    X(vkEnumerateDeviceExtensionProperties)

#define MV_VK_DEVICE_FUNCTIONS(X)   \
    X(vkDestroyDevice)              \
    X(vkGetDeviceQueue)             \
    X(vkQueueSubmit)                \
    X(vkQueueWaitIdle)              \
    X(vkDeviceWaitIdle)             \
    X(vkCreateCommandPool)          \
    X(vkDestroyCommandPool)         \
    X(vkResetCommandPool)           \
    X(vkAllocateCommandBuffers)     \
    X(vkFreeCommandBuffers)         \
    X(vkBeginCommandBuffer)         \
    X(vkEndCommandBuffer)           \
    X(vkResetCommandBuffer)         \
    X(vkCreateFence)                \
    X(vkDestroyFence)               \
    X(vkWaitForFences)              \
    X(vkResetFences)                \
    X(vkCreateBuffer)               \
    X(vkDestroyBuffer)              \
    X(vkGetBufferMemoryRequirements) \
    X(vkCreateImage)                \
    X(vkDestroyImage)               \
    X(vkGetImageMemoryRequirements) \
    X(vkAllocateMemory)             \
    X(vkFreeMemory)                 \
    X(vkBindBufferMemory)           \
    X(vkBindImageMemory)            \
    X(vkMapMemory)                  \
    X(vkUnmapMemory)                \
    X(vkFlushMappedMemoryRanges)    \
    X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateImageView)            \
    X(vkDestroyImageView)           \
    X(vkCreateSampler)              \
    X(vkDestroySampler)             \
    X(vkCreateShaderModule)         \
    X(vkDestroyShaderModule)        \
    X(vkCreateDescriptorSetLayout)  \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout)       \
    X(vkDestroyPipelineLayout)      \
    X(vkCreateDescriptorPool)       \
    X(vkDestroyDescriptorPool)      \
    X(vkResetDescriptorPool)        \
    X(vkAllocateDescriptorSets)     \
    X(vkUpdateDescriptorSets)       \
    X(vkCreateRenderPass)           \
    X(vkDestroyRenderPass)          \
    X(vkCreateFramebuffer)          \
    X(vkDestroyFramebuffer)         \
    X(vkCreateGraphicsPipelines)    \
    X(vkCreateComputePipelines)     \
    X(vkCmdDispatch)                \
    X(vkCmdPushConstants)           \
    X(vkDestroyPipeline)            \
    X(vkCreatePipelineCache)        \
    X(vkGetPipelineCacheData)       \
    X(vkDestroyPipelineCache)       \
    X(vkCmdBeginRenderPass)         \
    X(vkCmdEndRenderPass)           \
    X(vkCmdBindPipeline)            \
    X(vkCmdBindDescriptorSets)      \
    X(vkCmdBindVertexBuffers)       \
    X(vkCmdBindIndexBuffer)         \
    X(vkCmdDraw)                    \
    X(vkCmdDrawIndexed)             \
    X(vkCreateQueryPool)            \
    X(vkDestroyQueryPool)           \
    X(vkCmdResetQueryPool)          \
    X(vkCmdBeginQuery)              \
    X(vkCmdEndQuery)                \
    X(vkGetQueryPoolResults)        \
    X(vkCmdSetViewport)             \
    X(vkCmdSetScissor)              \
    X(vkCmdClearAttachments)        \
    X(vkCmdPipelineBarrier)         \
    X(vkCmdCopyBufferToImage)       \
    X(vkCmdCopyImageToBuffer)       \
    X(vkCmdCopyBuffer)              \
    X(vkCmdCopyImage)               \
    X(vkCmdBlitImage)               \
    X(vkCmdSetStencilReference)     \
    X(vkCmdSetBlendConstants)       \
    X(vkCmdSetDepthBias)

struct Functions {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr{nullptr};
#define MV_VK_DECLARE(name) PFN_##name name{nullptr};
    MV_VK_GLOBAL_FUNCTIONS(MV_VK_DECLARE)
    MV_VK_INSTANCE_FUNCTIONS(MV_VK_DECLARE)
    MV_VK_DEVICE_FUNCTIONS(MV_VK_DECLARE)
#undef MV_VK_DECLARE
};

// Opens the loader library and resolves the global entry points; false with a reason when there is no loader.
bool load_global(Functions& fn, std::string& error);
void load_instance(Functions& fn, VkInstance instance);
void load_device(Functions& fn, VkDevice device);
const char* result_name(VkResult r);

}  // namespace mv::vk
