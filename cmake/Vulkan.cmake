# The Vulkan headers (Khronos, Apache-2.0) come from a pinned fetch so the build is identical everywhere; the
# loader library itself is opened at run time (see src/nv2a/vulkan/vk_loader.cpp), never linked.
include(FetchContent)
FetchContent_Declare(VulkanHeaders
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG vulkan-sdk-1.4.363.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(VulkanHeaders)
