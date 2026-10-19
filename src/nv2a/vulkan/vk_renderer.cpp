#include "nv2a/vulkan/vk_renderer.h"

#include <algorithm>
#include <chrono>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <type_traits>

#include "debug/png.h"
#include "nv2a/shaders/glsl_compiler.h"

namespace mv::pgraph {

namespace {

constexpr VkDeviceSize kVertexRingBytes = 32u << 20, kIndexRingBytes = 8u << 20, kUniformRingBytes = 8u << 20, kStagingBytes = 16u << 20;
constexpr u32 kMaxDescriptorSets = 4096;

VkFormat color_vk_format(u32 nv_format) {
    switch (nv_format) {
        case 1: case 2: return VK_FORMAT_A1R5G5B5_UNORM_PACK16;
        case 3: return VK_FORMAT_R5G6B5_UNORM_PACK16;
        case 9: return VK_FORMAT_R8_UNORM;
        case 10: return VK_FORMAT_R8G8_UNORM;  // G8B8: two bytes, B then G
        default: return VK_FORMAT_B8G8R8A8_UNORM;  // 4-8: the 32-bit layouts
    }
}

// SET_COLOR_CLEAR_VALUE as the target's colour format reads it. The 16-bit formats take their fields from the low
// half: R5G6B5's 5-6-5, X1R5G5B5's 5-5-5. The 32-bit ones take R G B from bits 23-0. The alpha follows the format's
// name: the Z variants of the X formats write their unused bits as zero, the O variants as one, X1A7 keeps seven
// bits of the value's alpha below that bit, and only A8R8G8B8 takes all eight. B8 and G8B8 are read as 8888.
void clear_color_rgba(u32 nv_format, u32 c, float rgba[4]) {
    auto unorm = [](u32 v, u32 max) { return static_cast<float>(v) / static_cast<float>(max); };
    switch (nv_format) {
        case 1: case 2:
            rgba[0] = unorm((c >> 10) & 31u, 31);
            rgba[1] = unorm((c >> 5) & 31u, 31);
            rgba[2] = unorm(c & 31u, 31);
            break;
        case 3:
            rgba[0] = unorm((c >> 11) & 31u, 31);
            rgba[1] = unorm((c >> 5) & 63u, 63);
            rgba[2] = unorm(c & 31u, 31);
            break;
        default:
            rgba[0] = unorm((c >> 16) & 0xFFu, 255);
            rgba[1] = unorm((c >> 8) & 0xFFu, 255);
            rgba[2] = unorm(c & 0xFFu, 255);
            break;
    }
    switch (nv_format) {
        case 1: case 4: rgba[3] = 0.0f; break;
        case 6: rgba[3] = unorm((c >> 24) & 0x7Fu, 255); break;
        case 7: rgba[3] = unorm(0x80u | ((c >> 24) & 0x7Fu), 255); break;
        case 8: case 9: case 10: rgba[3] = unorm(c >> 24, 255); break;
        default: rgba[3] = 1.0f; break;
    }
}

VkBlendFactor blend_factor(u32 gl) {
    switch (gl) {
        case 0x0000: return VK_BLEND_FACTOR_ZERO;
        case 0x0001: return VK_BLEND_FACTOR_ONE;
        case 0x0300: return VK_BLEND_FACTOR_SRC_COLOR;
        case 0x0301: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 0x0302: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 0x0303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 0x0304: return VK_BLEND_FACTOR_DST_ALPHA;
        case 0x0305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 0x0306: return VK_BLEND_FACTOR_DST_COLOR;
        case 0x0307: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 0x0308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}

VkBlendOp blend_op(u32 gl) {
    switch (gl) {
        case 0x8007: return VK_BLEND_OP_MIN;
        case 0x8008: return VK_BLEND_OP_MAX;
        case 0x800A: return VK_BLEND_OP_SUBTRACT;
        case 0x800B: case 0xF005: return VK_BLEND_OP_REVERSE_SUBTRACT;
        default: return VK_BLEND_OP_ADD;  // 0x8006 and the signed add
    }
}

VkCompareOp compare_op(u32 gl) {
    switch (gl & 7u) {
        case 0: return VK_COMPARE_OP_NEVER;
        case 1: return VK_COMPARE_OP_LESS;
        case 2: return VK_COMPARE_OP_EQUAL;
        case 3: return VK_COMPARE_OP_LESS_OR_EQUAL;
        case 4: return VK_COMPARE_OP_GREATER;
        case 5: return VK_COMPARE_OP_NOT_EQUAL;
        case 6: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        default: return VK_COMPARE_OP_ALWAYS;
    }
}

VkStencilOp stencil_op(u32 gl) {
    switch (gl) {
        case 0x0000: return VK_STENCIL_OP_ZERO;
        case 0x1E01: return VK_STENCIL_OP_REPLACE;
        case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 0x150A: return VK_STENCIL_OP_INVERT;
        case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;  // 0x1E00
    }
}

VkSamplerAddressMode wrap_mode(u32 nv) {
    switch (nv) {
        case 2: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case 3: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

}  // namespace

u32 VulkanRenderer::Surface::color_bytes() const {
    switch (color_format) {
        case 1: case 2: case 3: case 10: return 2;
        case 9: return 1;
        default: return 4;
    }
}

VulkanRenderer::VulkanRenderer(vk::Device& device, const HandleResolver& handles, u8* ram, u32 ram_size)
    : dev_(device), handles_(handles), ram_(ram), ram_size_(ram_size), fn_(&device.fn()) {
    if (const char* d = std::getenv("MV_RENDER_DEBUG")) {
        const std::string s = d;
        debug_flat_ = s.find("flat") != std::string::npos;
        debug_nodepth_ = s.find("nodepth") != std::string::npos;
        debug_nocull_ = s.find("nocull") != std::string::npos;
        debug_noblend_ = s.find("noblend") != std::string::npos;
        debug_shadow_pass_ = s.find("shadow1") != std::string::npos;
        debug_show_v0_ = s.find("showv0") != std::string::npos;
        debug_skip_blend_ = s.find("skipblend") != std::string::npos;
        // drawid: every draw paints its index in the frame (red = low byte, green = high byte, blue = 1) wherever it
        // writes colour: a flat white shader blended by the constant colour.
        debug_draw_id_ = s.find("drawid") != std::string::npos;
        debug_flip_front_ = s.find("flipfront") != std::string::npos;
        // ffdump: each new fixed-function shader's state words and lighting values, to stderr (the first 400).
        debug_ff_dump_ = s.find("ffdump") != std::string::npos;
        // roundtrip: a sampled render target goes through guest RAM (write-back, decode, upload) instead of being
        // sampled in place — the control for the in-place path.
        debug_roundtrip_ = s.find("roundtrip") != std::string::npos;
        // texlog: every texture decode to stderr (address, format, size, levels, faces, whether a target overlaps it).
        debug_tex_log_ = s.find("texlog") != std::string::npos;
        // nofog: every draw as if fog were off (the pixel side reads the same switch).
        debug_no_fog_ = s.find("nofog") != std::string::npos;
        debug_vsh_dump_ = s.find("vshdump") != std::string::npos;
        if (debug_draw_id_) debug_flat_ = true;
    }
}

VulkanRenderer::~VulkanRenderer() {
    if (!dev_.ready()) return;
    const auto& f = *fn_;
    if (recording_) {
        end_pass();
        f.vkEndCommandBuffer(cb_);
        recording_ = false;
    }
    f.vkDeviceWaitIdle(dev_.device());
    save_pipeline_cache();
    if (pipeline_cache_) f.vkDestroyPipelineCache(dev_.device(), pipeline_cache_, nullptr);
    for (Frame& fr : frames_) {
        for (vk::Image& i : fr.dead_images) dev_.destroy_image(i);
        for (VkImageView v : fr.dead_views) f.vkDestroyImageView(dev_.device(), v, nullptr);
        for (VkFramebuffer b : fr.dead_framebuffers) f.vkDestroyFramebuffer(dev_.device(), b, nullptr);
    }
    for (auto& [k, s] : surfaces_) {
        destroy_surface_views(*s);
        if (s->framebuffer) f.vkDestroyFramebuffer(dev_.device(), s->framebuffer, nullptr);
        dev_.destroy_image(s->color);
        dev_.destroy_image(s->depth);
        dev_.destroy_image(s->snap);
        dev_.destroy_image(s->zcolor);
    }
    for (auto& [k, p] : passes_) f.vkDestroyRenderPass(dev_.device(), p, nullptr);
    for (auto& [k, m] : vs_cache_) f.vkDestroyShaderModule(dev_.device(), m, nullptr);
    for (auto& [k, m] : fs_cache_) f.vkDestroyShaderModule(dev_.device(), m, nullptr);
    for (auto& [k, p] : pipelines_) f.vkDestroyPipeline(dev_.device(), p, nullptr);
    for (auto& [k, p] : clear_pipelines_) f.vkDestroyPipeline(dev_.device(), p, nullptr);
    if (clear_vs_) f.vkDestroyShaderModule(dev_.device(), clear_vs_, nullptr);
    if (clear_fs_) f.vkDestroyShaderModule(dev_.device(), clear_fs_, nullptr);
    for (auto& [k, t] : textures_) dev_.destroy_image(t->image);
    for (auto& [k, s] : samplers_) f.vkDestroySampler(dev_.device(), s, nullptr);
    if (dummy_sampler_) f.vkDestroySampler(dev_.device(), dummy_sampler_, nullptr);
    dev_.destroy_image(dummy2d_);
    dev_.destroy_image(dummycube_);
    dev_.destroy_image(dummy3d_);
    if (pipeline_layout_) f.vkDestroyPipelineLayout(dev_.device(), pipeline_layout_, nullptr);
    if (zc_pipeline_) f.vkDestroyPipeline(dev_.device(), zc_pipeline_, nullptr);
    if (zc_layout_) f.vkDestroyPipelineLayout(dev_.device(), zc_layout_, nullptr);
    if (zc_set_layout_) f.vkDestroyDescriptorSetLayout(dev_.device(), zc_set_layout_, nullptr);
    if (zc_pool_) f.vkDestroyDescriptorPool(dev_.device(), zc_pool_, nullptr);
    if (zc_module_) f.vkDestroyShaderModule(dev_.device(), zc_module_, nullptr);
    dev_.destroy_buffer(zc_in_);
    dev_.destroy_buffer(zc_out_);
    if (set_layout_) f.vkDestroyDescriptorSetLayout(dev_.device(), set_layout_, nullptr);
    dev_.destroy_buffer(vertex_ring_.buffer);
    dev_.destroy_buffer(index_ring_.buffer);
    dev_.destroy_buffer(uniform_ring_.buffer);
    dev_.destroy_buffer(staging_.buffer);
    dev_.destroy_buffer(readback_);
    for (Frame& fr : frames_) {
        if (fr.pool) f.vkDestroyDescriptorPool(dev_.device(), fr.pool, nullptr);
        if (fr.fence) f.vkDestroyFence(dev_.device(), fr.fence, nullptr);
        if (fr.cb) f.vkFreeCommandBuffers(dev_.device(), dev_.pool(), 1, &fr.cb);
    }
}

void VulkanRenderer::note(const std::string& message) {
    for (const auto& e : stats_.errors)
        if (e == message) return;
    if (stats_.errors.size() < 64) stats_.errors.push_back(message);
}

bool VulkanRenderer::init(std::string& error) {
    const auto& f = *fn_;
    if (!dev_.format_supports(VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) depth24_ = VK_FORMAT_D32_SFLOAT_S8_UINT;
    // A shadow map is sampled in place only where the device samples and filters its depth format.
    constexpr VkFormatFeatureFlags kSampledFiltered = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    depth16_sampled_ = dev_.format_supports(VK_FORMAT_D16_UNORM, kSampledFiltered);
    depth24_sampled_ = dev_.format_supports(depth24_, kSampledFiltered);

    if (!create_ring(vertex_ring_, kVertexRingBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, error)) return false;
    if (!create_ring(index_ring_, kIndexRingBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, error)) return false;
    if (!create_ring(uniform_ring_, kUniformRingBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, error)) return false;
    if (!create_ring(staging_, kStagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, error)) return false;
    if (!dev_.create_buffer(kStagingBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk::BufferMemory::HostRead, readback_, error)) return false;

    // Descriptor layout: the two uniform blocks and the four texture stages, in every pipeline.
    VkDescriptorSetLayoutBinding bindings[6]{};
    for (u32 i = 0; i < 6; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 6;
    dli.pBindings = bindings;
    if (f.vkCreateDescriptorSetLayout(dev_.device(), &dli, nullptr, &set_layout_) != VK_SUCCESS) {
        error = "vkCreateDescriptorSetLayout failed";
        return false;
    }
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &set_layout_;
    if (f.vkCreatePipelineLayout(dev_.device(), &pli, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        error = "vkCreatePipelineLayout failed";
        return false;
    }
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxDescriptorSets * 2}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxDescriptorSets * 4}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = kMaxDescriptorSets;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = dev_.pool();
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (Frame& fr : frames_) {
        if (f.vkCreateDescriptorPool(dev_.device(), &dpi, nullptr, &fr.pool) != VK_SUCCESS || f.vkAllocateCommandBuffers(dev_.device(), &cai, &fr.cb) != VK_SUCCESS ||
            f.vkCreateFence(dev_.device(), &fci, nullptr, &fr.fence) != VK_SUCCESS) {
            error = "frame in flight: descriptor pool, command buffer or fence creation failed";
            return false;
        }
    }

    // Placeholder textures for the stages a shader does not sample: 1x1 white.
    const VkImageUsageFlags tex_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!dev_.create_image(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1, 1, tex_usage, VK_IMAGE_ASPECT_COLOR_BIT, false, dummy2d_, error)) return false;
    if (!dev_.create_image(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1, 6, tex_usage, VK_IMAGE_ASPECT_COLOR_BIT, true, dummycube_, error)) return false;
    if (!dev_.create_image(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 2, 1, 1, tex_usage, VK_IMAGE_ASPECT_COLOR_BIT, false, dummy3d_, error)) return false;
    {
        VkCommandBuffer cb = dev_.begin_once();
        for (vk::Image* img : {&dummy2d_, &dummycube_, &dummy3d_}) dev_.transition(cb, *img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        dev_.finish_once(cb);
        // white
        std::memset(staging_.buffer.mapped, 0xFF, 64);
        cb = dev_.begin_once();
        for (vk::Image* img : {&dummy2d_, &dummycube_, &dummy3d_}) {
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, img->layers};
            region.imageExtent = {1, 1, img->depth};
            f.vkCmdCopyBufferToImage(cb, staging_.buffer.buffer, img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        }
        dev_.finish_once(cb);
    }
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = 1.0f;
    f.vkCreateSampler(dev_.device(), &sci, nullptr, &dummy_sampler_);
    begin_frame(0);
    return true;
}

bool VulkanRenderer::create_ring(Ring& ring, VkDeviceSize region, VkBufferUsageFlags usage, std::string& error) {
    ring.region = region;
    return dev_.create_buffer(region * kFramesInFlight, usage, vk::BufferMemory::HostWrite, ring.buffer, error);
}

void VulkanRenderer::begin_frame(u32 slot) {
    const auto& f = *fn_;
    Frame& fr = frames_[slot];
    if (fr.serial > completed_) {
        // The GPU may still be executing this frame's last submission (its rings and descriptor sets are in use).
        f.vkWaitForFences(dev_.device(), 1, &fr.fence, VK_TRUE, ~0ull);
        completed_ = fr.serial;
        ++stats_.waits;
    }
    for (vk::Image& i : fr.dead_images) dev_.destroy_image(i);
    for (VkImageView v : fr.dead_views) f.vkDestroyImageView(dev_.device(), v, nullptr);
    for (VkFramebuffer b : fr.dead_framebuffers) f.vkDestroyFramebuffer(dev_.device(), b, nullptr);
    fr.dead_images.clear();
    fr.dead_views.clear();
    fr.dead_framebuffers.clear();
    f.vkResetDescriptorPool(dev_.device(), fr.pool, 0);
    descriptor_sets_used_ = 0;
    for (Ring* r : {&vertex_ring_, &index_ring_, &uniform_ring_, &staging_}) r->begin = r->cursor = r->region * slot;
    last_vs_upload_.valid = last_ps_upload_.valid = false;  // they lie in another frame's region
    last_set_ = bound_set_ = VK_NULL_HANDLE;                // the pool was reset, and binding is per command buffer
    slot_ = slot;
    cb_ = fr.cb;
    f.vkResetCommandBuffer(cb_, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    f.vkBeginCommandBuffer(cb_, &bi);
    recording_ = true;
}

void VulkanRenderer::retire(vk::Image& image) {
    if (image.image) frames_[slot_].dead_images.push_back(image);
    image = vk::Image{};
}

void VulkanRenderer::retire(VkImageView& view) {
    if (view) frames_[slot_].dead_views.push_back(view);
    view = VK_NULL_HANDLE;
}

void VulkanRenderer::retire(VkFramebuffer& framebuffer) {
    if (framebuffer) frames_[slot_].dead_framebuffers.push_back(framebuffer);
    framebuffer = VK_NULL_HANDLE;
}

void VulkanRenderer::end_pass() {
    if (pass_open_) {
        fn_->vkCmdEndRenderPass(cb_);
        pass_open_ = false;
        pass_target_ = nullptr;
    }
}

namespace {
// Adds the scope's wall time to a counter.
struct Timed {
    u64& acc;
    std::chrono::steady_clock::time_point t0;
    explicit Timed(u64& a) : acc(a), t0(std::chrono::steady_clock::now()) {}
    ~Timed() { acc += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()); }
};
u64 ns_since(std::chrono::steady_clock::time_point t0) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}
}  // namespace

void VulkanRenderer::submit() {
    const auto& f = *fn_;
    if (!recording_) return;
    Timed timed(stats_.ns_flush);
    end_pass();
    f.vkEndCommandBuffer(cb_);
    recording_ = false;
    Frame& fr = frames_[slot_];
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb_;
    f.vkResetFences(dev_.device(), 1, &fr.fence);
    f.vkQueueSubmit(dev_.queue(), 1, &si, fr.fence);
    fr.serial = ++submitted_;
    ++stats_.flushes;
    begin_frame((slot_ + 1) % kFramesInFlight);
}

void VulkanRenderer::finish() {
    const u32 last = slot_;
    submit();
    Timed timed(stats_.ns_flush);
    // A fence's signal covers every submission before its own on the queue, so the latest one is all it takes.
    Frame& fr = frames_[last];
    if (fr.serial > completed_) {
        fn_->vkWaitForFences(dev_.device(), 1, &fr.fence, VK_TRUE, ~0ull);
        completed_ = fr.serial;
        ++stats_.waits;
    }
}

VkRenderPass VulkanRenderer::render_pass(VkFormat color, VkFormat depth) {
    const u64 key = (static_cast<u64>(color) << 32) | static_cast<u64>(depth);
    if (auto it = passes_.find(key); it != passes_.end()) return it->second;
    VkAttachmentDescription att[2]{};
    att[0].format = color;
    att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    att[0].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    att[1] = att[0];
    att[1].format = depth;
    att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_GENERAL};
    VkAttachmentReference depth_ref{1, VK_IMAGE_LAYOUT_GENERAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &color_ref;
    sub.pDepthStencilAttachment = &depth_ref;
    // A pass over a target can follow another over it in the same command buffer or an earlier submission still
    // running (no wait separates them any more): its attachment reads and writes come after the earlier writes.
    constexpr VkPipelineStageFlags kAttachmentStages =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = dep.dstStageMask = kAttachmentStages;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 2;
    rpi.pAttachments = att;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sub;
    rpi.dependencyCount = 1;
    rpi.pDependencies = &dep;
    VkRenderPass pass = VK_NULL_HANDLE;
    fn_->vkCreateRenderPass(dev_.device(), &rpi, nullptr, &pass);
    passes_[key] = pass;
    return pass;
}

VulkanRenderer::Surface* VulkanRenderer::target_for(const KelvinState& s) {
    const SurfaceState d = s.surface();
    const DmaObject color_dma = handles_.resolve(s.dma_handle(mthd::kDmaColor));
    const DmaObject zeta_dma = handles_.resolve(s.dma_handle(mthd::kDmaZeta));
    const u32 phys = color_dma.base + d.color_offset;
    // Anti-aliasing: the clip is in logical pixels and the hardware renders the surface 2x wide (CENTER_CORNER_2)
    // or 2x2 (SQUARE_OFFSET_4) larger, positions scaled with it; a later pass reads it back down.
    const u32 aa = d.anti_aliasing();
    const u32 aa_x = (aa == 1 || aa == 2) ? 2u : 1u, aa_y = aa == 2 ? 2u : 1u;
    const u32 width = d.swizzled() ? (1u << d.log_width()) : (d.clip_x() + d.clip_width()) * aa_x;
    const u32 height = d.swizzled() ? (1u << d.log_height()) : (d.clip_y() + d.clip_height()) * aa_y;
    if (width == 0 || height == 0 || width > 4096 || height > 4096) {
        note(std::format("surface at {:08x} has an unusable size {}x{} (format {:08x})", phys, width, height, d.format));
        return nullptr;
    }
    auto it = surfaces_.find(phys);
    if (it != surfaces_.end()) {
        Surface& s0 = *it->second;
        if (s0.width == width && s0.height == height && s0.color_format == d.color_format() && s0.zeta_format == d.zeta_format()) {
            // The same images serve both layouts: the layout only decides how RAM is read and written.
            s0.swizzled = d.swizzled();
            s0.log_w = d.log_width();
            s0.log_h = d.log_height();
            s0.format_word = d.format;
            s0.pitch = d.color_pitch();
            s0.zeta_pitch = d.zeta_pitch();
            s0.aa_x = d.swizzled() ? 1u : aa_x;
            s0.aa_y = d.swizzled() ? 1u : aa_y;
            s0.zeta_phys = zeta_dma.base + d.zeta_offset;
            return &s0;
        }
        // The guest reuses the memory with another layout: finish with the old one.
        note(std::format("surface {:08x} relaid out: {}x{} {} format {:x} -> {}x{} {} format {:x}", phys, s0.width, s0.height, s0.swizzled ? "swizzled" : "pitch",
                         s0.color_format, width, height, d.swizzled() ? "swizzled" : "pitch", d.color_format()));
        if (pass_target_ == &s0) end_pass();
        wb_why_ = Stats::kWbRelayout;
        if (s0.dirty) write_back(s0);
        wb_why_ = Stats::kWbAll;
        // Draws already recorded or still on the GPU may use the old images: they go once those are done.
        retire(s0.x8_view);
        retire(s0.depth_view);
        retire(s0.snap_x8_view);
        retire(s0.zcolor_x8_view);
        destroy_surface_views(s0);
        retire(s0.framebuffer);
        retire(s0.color);
        retire(s0.depth);
        retire(s0.snap);
        retire(s0.zcolor);
        surfaces_.erase(it);
        stage_cache_.fill({});  // a stage may point at the erased target's entries
    }
    if (debug_tex_log_)
        std::fprintf(stderr, "texlog: target %08x %ux%u %s colour %u zeta %u pitch %u\n", phys, width, height, d.swizzled() ? "swizzled" : "linear", d.color_format(), d.zeta_format(), d.color_pitch());
    auto surf = std::make_unique<Surface>();
    Surface& t = *surf;
    t.phys = phys;
    t.zeta_phys = zeta_dma.base + d.zeta_offset;
    t.format_word = d.format;
    t.width = width;
    t.height = height;
    t.aa_x = d.swizzled() ? 1u : aa_x;
    t.aa_y = d.swizzled() ? 1u : aa_y;
    t.log_w = d.log_width();
    t.log_h = d.log_height();
    t.pitch = d.color_pitch();
    t.zeta_pitch = d.zeta_pitch();
    t.swizzled = d.swizzled();
    t.color_format = d.color_format();
    t.zeta_format = d.zeta_format();
    std::string error;
    const VkFormat cf = color_vk_format(t.color_format);
    const VkFormat df = t.zeta_format == 1 ? VK_FORMAT_D16_UNORM : depth24_;
    if (!dev_.create_image(cf, width, height, 1, 1, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT, false, t.color, error) ||
        !dev_.create_image(df, width, height, 1, 1, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               ((t.zeta_format == 1 ? depth16_sampled_ : depth24_sampled_) ? static_cast<u32>(VK_IMAGE_USAGE_SAMPLED_BIT) : 0u),
                           VK_IMAGE_ASPECT_DEPTH_BIT | (t.zeta_format == 1 ? 0u : static_cast<u32>(VK_IMAGE_ASPECT_STENCIL_BIT)), false, t.depth, error)) {
        note("surface images: " + error);
        dev_.destroy_image(t.color);
        return nullptr;
    }
    // Images start in GENERAL and stay there; the colour image starts with what RAM holds (the guest may draw over
    // an existing picture). Both are recorded into the frame, ahead of the draws that follow.
    end_pass();
    fresh_image(t.color);
    fresh_image(t.depth);
    // A target that another target's dirty colour or depth covers starts from what that one rendered: written back first
    // (the memory is the truth; a target over another's depth buffer loaded stale RAM once depth read as colour stayed
    // on the GPU).
    {
        const u64 b0 = t.phys, b1 = static_cast<u64>(t.phys) + t.bytes();
        wb_why_ = Stats::kWbRelayout;
        for (auto& [k, o] : surfaces_) {
            const u64 a0 = o->phys, a1 = a0 + o->bytes(), z0 = o->zeta_phys, z1 = z0 + o->zeta_bytes();
            if (o->dirty && a0 < b1 && b0 < a1) queue_write_back(*o);
            if (o->zeta_dirty && z0 < b1 && b0 < z1) queue_write_back_zeta(*o);
        }
        finish_write_backs();
        wb_why_ = Stats::kWbAll;
    }
    upload_color_from_ram(t);  // the guest may draw over an existing picture
    t.pass = render_pass(cf, df);
    VkImageView views[2] = {t.color.view, t.depth.view};
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = t.pass;
    fbi.attachmentCount = 2;
    fbi.pAttachments = views;
    fbi.width = width;
    fbi.height = height;
    fbi.layers = 1;
    fn_->vkCreateFramebuffer(dev_.device(), &fbi, nullptr, &t.framebuffer);
    Surface* raw = surf.get();
    surfaces_[phys] = std::move(surf);
    return raw;
}

// The colour image takes what RAM holds: at the target's creation, and again before it is drawn into once another
// engine has rewritten its memory (one title resets a glare buffer each frame with a 2D blit, then draws into it
// additively; drawing on over the GPU's stale copy saturated it to white).
void VulkanRenderer::upload_color_from_ram(Surface& t) {
    ++t.draw_serial;
    const u32 bpp = t.color_bytes();
    std::vector<u8> pixels(static_cast<std::size_t>(t.width) * t.height * bpp, 0);
    for (u32 y = 0; y < t.height; ++y) {
        for (u32 x = 0; x < t.width; ++x) {
            const u32 off = t.swizzled ? swizzled_offset(x, y, 0, t.log_w, t.log_h, 0) * bpp : y * t.pitch + x * bpp;
            if (static_cast<u64>(t.phys) + off + bpp <= ram_size_) std::memcpy(&pixels[(static_cast<std::size_t>(y) * t.width + x) * bpp], ram_ + t.phys + off, bpp);
        }
    }
    VkDeviceSize offset = 0;
    if (staging_alloc(pixels.size(), offset)) {
        std::memcpy(static_cast<u8*>(staging_.buffer.mapped) + offset, pixels.data(), pixels.size());
        VkBufferImageCopy region{};
        region.bufferOffset = offset;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {t.width, t.height, 1};
        fn_->vkCmdCopyBufferToImage(cb_, staging_.buffer.buffer, t.color.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        upload_barrier();
    }
}

bool VulkanRenderer::ensure_pass(Surface& t) {
    if (pass_open_ && pass_target_ == &t) return true;
    end_pass();
    if (t.color_ram_newer) {
        upload_color_from_ram(t);
        t.color_ram_newer = false;
        ++stats_.surface_reloads;
    }
    if (sampled_since_barrier_) surface_barrier();  // draws that sampled a target finish before it is drawn into again
    t.last_rendered_frame = frame_index_;
    t.ram_newer = false;
    attachments_written_ = true;
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = t.pass;
    rbi.framebuffer = t.framebuffer;
    rbi.renderArea = {{0, 0}, {t.width, t.height}};
    fn_->vkCmdBeginRenderPass(cb_, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    pass_open_ = true;
    pass_target_ = &t;
    // A pass may write depth; whether it writes colour is the draw's or the clear's to say (the colour mask, the
    // clear bits), so a depth-only pass over an unused colour target never lands in guest RAM.
    t.zeta_dirty = true;
    t.rendered_this_frame = true;
    return true;
}

namespace {

// The Morton address of a swizzled surface separates into an x part and a y part (disjoint bits), so two small
// tables replace a bit-interleave per texel.
void swizzle_tables(u32 width, u32 height, u32 log_w, u32 log_h, std::vector<u32>& xs, std::vector<u32>& ys) {
    xs.resize(width);
    ys.resize(height);
    for (u32 x = 0; x < width; ++x) xs[x] = swizzled_offset(x, 0, 0, log_w, log_h, 0);
    for (u32 y = 0; y < height; ++y) ys[y] = swizzled_offset(0, y, 0, log_w, log_h, 0);
}

// Scatters linear rows of texels into the swizzled layout at base.
template <typename T>
void scatter_swizzled(T* base, const T* rows, u32 width, u32 height, const std::vector<u32>& xs, const std::vector<u32>& ys) {
    for (u32 y = 0; y < height; ++y) {
        const T* row = rows + static_cast<std::size_t>(y) * width;
        const u32 oy = ys[y];
        for (u32 x = 0; x < width; ++x) base[oy | xs[x]] = row[x];
    }
}

}  // namespace

void VulkanRenderer::read_back(const vk::Image& image, VkImageAspectFlags aspect, u32 width, u32 height, VkDeviceSize offset) {
    end_pass();
    // The first copy of a batch waits for the frame's attachment writes (and uploads) before it reads the image.
    if (write_backs_.empty()) {
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    }
    // Depth and stencil are two copies: the depth's words first, then one stencil byte per texel after them.
    const bool both = (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && (aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
    VkBufferImageCopy regions[2]{};
    regions[0].bufferOffset = offset;
    regions[0].imageSubresource = {both ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT) : aspect, 0, 0, 1};
    regions[0].imageExtent = {width, height, 1};
    regions[1] = regions[0];
    regions[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    regions[1].bufferOffset = offset + static_cast<VkDeviceSize>(width) * height * 4;
    fn_->vkCmdCopyImageToBuffer(cb_, image.image, VK_IMAGE_LAYOUT_GENERAL, readback_.buffer, both ? 2 : 1, regions);
}

bool VulkanRenderer::readback_alloc(VkDeviceSize bytes, VkDeviceSize& offset) {
    if (bytes > readback_.size) return false;
    // Sixteen bytes keep every copy's offset a multiple of its texel size.
    const VkDeviceSize start = (readback_cursor_ + 15) / 16 * 16;
    if (start + bytes > readback_.size) {
        finish_write_backs();  // the batch so far fills the buffer: store it, then start over
        return readback_alloc(bytes, offset);
    }
    offset = start;
    readback_cursor_ = start + bytes;
    return true;
}

void VulkanRenderer::queue_write_back(Surface& s) {
    if (!s.dirty) return;
    Timed timed(stats_.ns_write_back);
    const std::size_t bytes = static_cast<std::size_t>(s.width) * s.height * s.color_bytes();
    VkDeviceSize offset = 0;
    if (!readback_alloc(bytes, offset)) {
        note("surface too large to write back");
        return;
    }
    read_back(s.color, VK_IMAGE_ASPECT_COLOR_BIT, s.width, s.height, offset);
    ++stats_.writebacks_by[wb_why_];
    write_backs_.push_back({&s, false, offset});
}

void VulkanRenderer::queue_write_back_zeta(Surface& s) {
    if (!s.zeta_dirty) return;
    s.zeta_dirty = false;
    if (s.zeta_format == 0 || s.zeta_phys == 0) return;  // no depth surface bound (an offset of zero is never a real one)
    Timed timed(stats_.ns_write_back);
    const bool z16 = s.zeta_format == 1;
    // D16; a 24-bit or float depth copies as a 32-bit word, followed by the stencil bytes.
    const std::size_t bytes = static_cast<std::size_t>(s.width) * s.height * (z16 ? 2u : 5u);
    if (bytes > readback_.size) {
        note("depth surface too large to write back");
        return;
    }
    if (static_cast<u64>(s.zeta_phys) + s.zeta_bytes() > ram_size_) return;
    VkDeviceSize offset = 0;
    readback_alloc(bytes, offset);  // fits: checked above
    read_back(s.depth, z16 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, s.width, s.height, offset);
    ++stats_.writebacks_by[wb_why_];
    write_backs_.push_back({&s, true, offset});
}

void VulkanRenderer::finish_write_backs() {
    if (write_backs_.empty()) return;
    Timed timed(stats_.ns_write_back);
    {
        Timed timed_gpu(stats_.ns_write_back_gpu);
        // The copies are visible to the CPU once the fence signals.
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        finish();
    }
    // Into RAM in the order queued, which is the order the surfaces were written back one at a time before.
    const u8* base = static_cast<const u8*>(readback_.mapped);
    for (const PendingWriteBack& w : write_backs_) {
        if (w.zeta) store_zeta(*w.surface, base + w.offset);
        else store_color(*w.surface, base + w.offset);
    }
    write_backs_.clear();
    readback_cursor_ = 0;
}

void VulkanRenderer::write_back(Surface& s) {
    queue_write_back(s);
    finish_write_backs();
}

void VulkanRenderer::store_color(Surface& s, const u8* pixels) {
    if (debug_tex_log_) std::fprintf(stderr, "texlog: store %08x %ux%u %s\n", s.phys, s.width, s.height, s.swizzled ? "swizzled" : "linear");
    const u32 bpp = s.color_bytes();
    const std::size_t bytes = static_cast<std::size_t>(s.width) * s.height * bpp;
    if (!s.swizzled) {
        // Linear: a row at a time.
        const u32 row_bytes = s.width * bpp;
        for (u32 y = 0; y < s.height; ++y) {
            const u64 off = static_cast<u64>(y) * s.pitch;
            if (static_cast<u64>(s.phys) + off + row_bytes <= ram_size_) std::memcpy(ram_ + s.phys + off, pixels + static_cast<std::size_t>(y) * row_bytes, row_bytes);
        }
    } else if (static_cast<u64>(s.phys) + bytes <= ram_size_) {
        std::vector<u32> xs, ys;
        swizzle_tables(s.width, s.height, s.log_w, s.log_h, xs, ys);
        u8* base = ram_ + s.phys;
        if (bpp == 4) {
            scatter_swizzled(reinterpret_cast<u32*>(base), reinterpret_cast<const u32*>(pixels), s.width, s.height, xs, ys);
        } else if (bpp == 2) {
            scatter_swizzled(reinterpret_cast<u16*>(base), reinterpret_cast<const u16*>(pixels), s.width, s.height, xs, ys);
        } else {
            for (u32 y = 0; y < s.height; ++y) {
                const u8* row = pixels + static_cast<std::size_t>(y) * s.width * bpp;
                for (u32 x = 0; x < s.width; ++x) std::memcpy(base + static_cast<std::size_t>(ys[y] | xs[x]) * bpp, row + static_cast<std::size_t>(x) * bpp, bpp);
            }
        }
    }
    s.dirty = false;
    ++stats_.writebacks;
    invalidate_textures(s.phys, s.bytes());
}

void VulkanRenderer::write_back_zeta(Surface& s) {
    queue_write_back_zeta(s);
    finish_write_backs();
}

void VulkanRenderer::store_zeta(Surface& s, const u8* px) {
    const bool z16 = s.zeta_format == 1;
    const u8* stencil = px + static_cast<std::size_t>(s.width) * s.height * 4;
    std::vector<u32> xs, ys;
    if (s.swizzled) swizzle_tables(s.width, s.height, s.log_w, s.log_h, xs, ys);
    u8* base = ram_ + s.zeta_phys;
    if (z16) {
        if (s.swizzled) {
            scatter_swizzled(reinterpret_cast<u16*>(base), reinterpret_cast<const u16*>(px), s.width, s.height, xs, ys);
        } else {
            for (u32 y = 0; y < s.height; ++y) std::memcpy(base + static_cast<std::size_t>(y) * s.zeta_pitch, px + static_cast<std::size_t>(y) * s.width * 2, s.width * 2u);
        }
    } else {
        // Z24S8: the depth in bits 31-8 above the stencil byte, converted a row at a time.
        const bool float_depth = depth24_ == VK_FORMAT_D32_SFLOAT_S8_UINT;
        std::vector<u32> row(s.width);
        for (u32 y = 0; y < s.height; ++y) {
            const u8* src = px + static_cast<std::size_t>(y) * s.width * 4;
            for (u32 x = 0; x < s.width; ++x) {
                u32 depth24;
                if (float_depth) {
                    float f;
                    std::memcpy(&f, src + static_cast<std::size_t>(x) * 4, 4);
                    depth24 = static_cast<u32>(std::clamp(f, 0.0f, 1.0f) * 16777215.0f);
                } else {
                    std::memcpy(&depth24, src + static_cast<std::size_t>(x) * 4, 4);
                    depth24 &= 0xFFFFFFu;
                }
                row[x] = (depth24 << 8) | stencil[static_cast<std::size_t>(y) * s.width + x];
            }
            if (s.swizzled) {
                u32* out = reinterpret_cast<u32*>(base);
                const u32 oy = ys[y];
                for (u32 x = 0; x < s.width; ++x) out[oy | xs[x]] = row[x];
            } else {
                std::memcpy(base + static_cast<std::size_t>(y) * s.zeta_pitch, row.data(), static_cast<std::size_t>(s.width) * 4);
            }
        }
    }
    ++stats_.writebacks;
    invalidate_textures(s.zeta_phys, s.zeta_bytes());
}

void VulkanRenderer::write_back_all() {
    for (auto& [k, s] : surfaces_) {
        queue_write_back(*s);
        queue_write_back_zeta(*s);
    }
    finish_write_backs();
}

void VulkanRenderer::write_back_range(u32 phys, u32 len) {
    wb_why_ = Stats::kWbRange;
    const u64 b0 = phys, b1 = static_cast<u64>(phys) + len;
    for (auto& [k, s] : surfaces_) {
        const u64 a0 = s->phys, a1 = static_cast<u64>(s->phys) + s->bytes();
        if (s->dirty && a0 < b1 && b0 < a1) queue_write_back(*s);
        const u64 z0 = s->zeta_phys, z1 = static_cast<u64>(s->zeta_phys) + s->zeta_bytes();
        if (s->zeta_dirty && z0 < b1 && b0 < z1) queue_write_back_zeta(*s);
    }
    finish_write_backs();
    wb_why_ = Stats::kWbAll;
}

// A GPU-side write to guest memory (a blit): decoded textures over it are stale, and so is any target's image
// the guest has not drawn into since.
void VulkanRenderer::invalidate(u32 phys, u32 len) {
    invalidate_textures(phys, len);
    const u64 b0 = phys, b1 = static_cast<u64>(phys) + len;
    for (auto& [k, s] : surfaces_) {
        const u64 a0 = s->phys, a1 = a0 + s->bytes(), z0 = s->zeta_phys, z1 = z0 + s->zeta_bytes();
        if ((a0 < b1 && b0 < a1) || (z0 < b1 && b0 < z1)) s->ram_newer = true;
        if (a0 < b1 && b0 < a1) {
            s->color_ram_newer = true;
            if (pass_open_ && pass_target_ == s.get()) end_pass();  // the next draw into it reopens the pass, reloaded
        }
    }
}

void VulkanRenderer::invalidate_textures(u32 phys, u32 len) {
    for (auto& [k, t] : textures_) {
        if (!t->valid) continue;
        const u64 a0 = t->phys, a1 = static_cast<u64>(t->phys) + t->length;
        const u64 b0 = phys, b1 = static_cast<u64>(phys) + len;
        if (a0 < b1 && b0 < a1) t->valid = false;
    }
}

VkShaderModule VulkanRenderer::module(const std::vector<u32>& spirv) {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spirv.size() * 4;
    smi.pCode = spirv.data();
    VkShaderModule m = VK_NULL_HANDLE;
    fn_->vkCreateShaderModule(dev_.device(), &smi, nullptr, &m);
    ++stats_.shaders;
    return m;
}

VkShaderModule VulkanRenderer::vertex_shader(const KelvinState& s, std::string& error) {
    Timed timed(stats_.ns_vs);
    const bool vp = s.vertex_program_mode();
    const bool smooth = s.reg(mthd::kShadeModel) != 0x1D00;
    if (!vp && !vsh::fixed_function_supported(s, error)) return VK_NULL_HANDLE;
    const bool fog = !debug_no_fog_ && (s.reg(0x02A4) & 1u) != 0;
    const u32 fog_mode = s.reg(0x029C);
    // The fog mode shapes every shader's ending (the distance-to-factor curve).
    const u32 flags = (vp ? 1u : 0u) | (smooth ? 2u : 0u) | (fog ? 4u : 0u) | (fog ? (fog_mode & 0xFFFFu) << 16 : 0u);
    const u32 start = vp ? s.program_start() : 0;
    // The fixed function's shader follows from a handful of state words; a program's from its words.
    const std::vector<u32> ff_key = vp ? std::vector<u32>{} : vsh::fixed_function_key(s);
    for (const VsMemo& m : vs_memo_) {
        if (m.module != VK_NULL_HANDLE && m.flags == flags && m.start == start &&
            (vp ? std::memcmp(m.words.data(), &s.program[start][0], m.words.size() * 4) == 0 : m.words == ff_key)) {
            ++stats_.vs_memo_hits;
            return m.module;
        }
    }
    vsh::Program program;
    std::string key;
    if (vp) {
        program = vsh::extract_program(s);
        key.assign(reinterpret_cast<const char*>(&s.program[program.start]), program.instructions.size() * 16);
    } else {
        key.assign("fixed-function:");
        key.append(reinterpret_cast<const char*>(ff_key.data()), ff_key.size() * 4);
    }
    key += smooth ? 's' : 'f';
    key += std::format(":fog{:x}", flags >> 2);
    auto remember = [&](VkShaderModule m) {
        VsMemo& e = vs_memo_[vs_memo_next_];
        vs_memo_next_ = (vs_memo_next_ + 1) % kMemoEntries;
        e.start = start;
        e.flags = flags;
        e.module = m;
        e.words.clear();
        if (vp) e.words.assign(&s.program[start][0], &s.program[start][0] + program.instructions.size() * 4);
        else e.words = ff_key;
    };
    if (auto it = vs_cache_.find(key); it != vs_cache_.end()) {
        ++stats_.vs_cache_hits;
        remember(it->second);
        return it->second;
    }
    vsh::Options options;
    options.smooth_shading = s.reg(mthd::kShadeModel) != 0x1D00;
    options.fog = fog;
    options.fog_mode = fog_mode;
    const vsh::Translation t = vp ? vsh::translate(program, options) : vsh::fixed_function_translate(s, options);
    if (debug_ff_dump_ && !vp) {
        static int dumped = 0;
        if (dumped++ < 400) {
            std::string line = "ffdump: key";
            for (u32 w : ff_key) line += std::format(" {:x}", w);
            std::vector<float> u(vsh::kFixedFunctionVectors * 4);
            vsh::fixed_function_uniforms(s, u.data());
            line += std::format("\n  emission {:.3f} {:.3f} {:.3f} alpha {:.3f} | scene ambient {:.3f} {:.3f} {:.3f} | specular curve {:.3f} {:.3f} {:.3f}", u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[8], u[9], u[10]);
            for (u32 l = 0; l < vsh::kLights; ++l) {
                if (!((s.reg(mthd::kLightEnableMask) >> (2 * l)) & 3u)) continue;
                const float* v = u.data() + 16 + l * vsh::kLightVectors * 4;
                line += std::format("\n  light {} type {}: amb {:.3f} {:.3f} {:.3f} dif {:.3f} {:.3f} {:.3f} spc {:.3f} {:.3f} {:.3f} pos {:.2f} {:.2f} {:.2f} range {:.3g} att {:.3g} {:.3g} {:.3g} dir {:.3f} {:.3f} {:.3f}", l,
                                    (s.reg(mthd::kLightEnableMask) >> (2 * l)) & 3u, v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10], v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[20], v[21], v[22]);
            }
            std::fprintf(stderr, "%s\n", line.c_str());
        }
    }
    if (!t.error.empty()) {
        error = t.error;
        return VK_NULL_HANDLE;
    }
    std::string log;
    const std::vector<u32> spirv = spirv_for(true, t.glsl, "kelvin-vp", log);
    if (spirv.empty()) {
        error = "vertex program GLSL: " + log;
        return VK_NULL_HANDLE;
    }
    VkShaderModule m = module(spirv);
    vs_cache_[key] = m;
    remember(m);
    // Instrument: MV_RENDER_DEBUG=vshdump writes each new vertex shader (its disassembly, then its GLSL) to
    // scratch/vshdump/vs-N.glsl; drawid lines name the shader each draw used.
    if (debug_draw_id_ || debug_vsh_dump_) {
        const u32 n = static_cast<u32>(vs_debug_ids_.size());
        vs_debug_ids_[m] = n;
        if (debug_vsh_dump_) {
            std::error_code ec;
            std::filesystem::create_directories("scratch/vshdump", ec);
            if (std::FILE* f = std::fopen(std::format("scratch/vshdump/vs-{}.glsl", n).c_str(), "w")) {
                for (const auto& line : t.disassembly) std::fprintf(f, "// %s\n", line.c_str());
                std::fprintf(f, "%s", t.glsl.c_str());
                std::fclose(f);
            }
        }
    }
    return m;
}

VkShaderModule VulkanRenderer::fragment_shader(const psh::Key& key, std::array<psh::SamplerKind, 4>& kinds, std::string& error) {
    Timed timed(stats_.ns_fs);
    for (const FsMemo& m : fs_memo_) {
        if (m.module != VK_NULL_HANDLE && m.key == key) {
            ++stats_.fs_memo_hits;
            kinds = m.kinds;
            return m.module;
        }
    }
    auto remember = [&](VkShaderModule m) {
        fs_memo_[fs_memo_next_] = {key, kinds, m};
        fs_memo_next_ = (fs_memo_next_ + 1) % kMemoEntries;
    };
    std::string k(reinterpret_cast<const char*>(&key), sizeof key);
    if (auto it = fs_cache_.find(k); it != fs_cache_.end()) {
        ++stats_.fs_cache_hits;
        kinds = fs_kinds_[k];
        remember(it->second);
        return it->second;
    }
    psh::Translation t = psh::translate(key);
    if (!t.error.empty()) {
        error = t.error;
        return VK_NULL_HANDLE;
    }
    if (debug_flat_) t.glsl = "#version 450\nlayout(location = 0) out vec4 frag_color;\nvoid main() { frag_color = vec4(1.0); }\n";
    if (debug_show_v0_) {
        // The interpolated diffuse colour in place of the combiners' result (the last statement of main).
        if (const auto end = t.glsl.rfind('}'); end != std::string::npos) t.glsl.insert(end, "    frag_color = vec4(clamp(vtx_d0.rgb, 0.0, 1.0), 1.0);\n");
    }
    std::string log;
    const std::vector<u32> spirv = spirv_for(false, t.glsl, "kelvin-combiners", log);
    if (spirv.empty()) {
        error = "combiner GLSL: " + log;
        return VK_NULL_HANDLE;
    }
    VkShaderModule m = module(spirv);
    fs_cache_[k] = m;
    fs_kinds_[k] = t.samplers;
    kinds = t.samplers;
    remember(m);
    return m;
}

VkPipeline VulkanRenderer::pipeline(const KelvinState& s, VkShaderModule vs, VkShaderModule fs, VkPrimitiveTopology topology, Surface& t, u32 vertex_layout, std::string& error) {
    Timed timed(stats_.ns_pipeline);
    struct RasterKey {
        VkShaderModule vs, fs;
        VkRenderPass pass;
        u32 topology, blend_enable, sfactor, dfactor, equation, depth_test, depth_func, depth_write, cull_enable, cull_face, front_face,
            color_mask, stencil_enable, stencil_func, stencil_mask, stencil_write_mask, op_fail, op_zfail, op_zpass;
        u32 vertex_layout;  // the attributes with per-vertex data (also fills the last eight bytes: the key is its bytes)
    } k{vs, fs, t.pass, static_cast<u32>(topology) | (debug_nocull_ ? 0x100u : 0u) | (debug_nodepth_ ? 0x200u : 0u) | (debug_noblend_ ? 0x400u : 0u), s.blend(), s.reg(mthd::kBlendSfactor), s.reg(mthd::kBlendDfactor), s.reg(mthd::kBlendEquation),
        s.depth_test(), s.reg(mthd::kDepthFunc), s.depth_write(), s.cull_face(), s.reg(mthd::kCullFace), s.reg(mthd::kFrontFace), s.color_mask(),
        s.stencil_test(), s.reg(mthd::kStencilFunc), s.reg(mthd::kStencilFuncMask), s.reg(mthd::kStencilMask), s.reg(mthd::kStencilOpFail),
        s.reg(mthd::kStencilOpZfail), s.reg(mthd::kStencilOpZpass), vertex_layout};
    static_assert(std::has_unique_object_representations_v<RasterKey>);
    std::string key(reinterpret_cast<const char*>(&k), sizeof k);
    if (auto it = pipelines_.find(key); it != pipelines_.end()) return it->second;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    // The attributes with per-vertex data packed in order in binding 0; the others read their constant from binding 1,
    // a block of all sixteen at a stride of zero.
    const u32 per_vertex = static_cast<u32>(std::popcount(vertex_layout & 0xFFFFu));
    VkVertexInputBindingDescription bindings[2] = {{0, std::max(per_vertex, 1u) * 16, VK_VERTEX_INPUT_RATE_VERTEX}, {1, 0, VK_VERTEX_INPUT_RATE_VERTEX}};
    VkVertexInputAttributeDescription attrs[kAttributes]{};
    for (u32 i = 0, packed = 0; i < kAttributes; ++i)
        attrs[i] = (vertex_layout >> i) & 1u ? VkVertexInputAttributeDescription{i, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16 * packed++}
                                              : VkVertexInputAttributeDescription{i, 1, VK_FORMAT_R32G32B32A32_SFLOAT, i * 16};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 2;
    vi.pVertexBindingDescriptions = bindings;
    vi.vertexAttributeDescriptionCount = kAttributes;
    vi.pVertexAttributeDescriptions = attrs;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = topology;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.0f;
    if (s.cull_face() && !debug_nocull_) {
        const u32 face = s.reg(mthd::kCullFace);
        rs.cullMode = face == 0x404 ? VK_CULL_MODE_FRONT_BIT : (face == 0x408 ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_BACK_BIT);
    }
    rs.frontFace = (s.reg(mthd::kFrontFace) == 0x900) != debug_flip_front_ ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.depthBiasEnable = VK_TRUE;  // the polygon offset, set per draw (zero when the draw has none)
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = (s.depth_test() && !debug_nodepth_) ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = s.depth_write() ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = compare_op(s.reg(mthd::kDepthFunc));
    ds.stencilTestEnable = s.stencil_test() ? VK_TRUE : VK_FALSE;
    VkStencilOpState st{};
    st.failOp = stencil_op(s.reg(mthd::kStencilOpFail));
    st.depthFailOp = stencil_op(s.reg(mthd::kStencilOpZfail));
    st.passOp = stencil_op(s.reg(mthd::kStencilOpZpass));
    st.compareOp = compare_op(s.reg(mthd::kStencilFunc));
    st.compareMask = s.reg(mthd::kStencilFuncMask) & 0xFFu;
    st.writeMask = s.reg(mthd::kStencilMask) & 0xFFu;
    ds.front = ds.back = st;
    VkPipelineColorBlendAttachmentState cb{};
    cb.blendEnable = (s.blend() && !debug_noblend_) ? VK_TRUE : VK_FALSE;
    cb.srcColorBlendFactor = cb.srcAlphaBlendFactor = blend_factor(s.reg(mthd::kBlendSfactor));
    cb.dstColorBlendFactor = cb.dstAlphaBlendFactor = blend_factor(s.reg(mthd::kBlendDfactor));
    cb.colorBlendOp = cb.alphaBlendOp = blend_op(s.reg(mthd::kBlendEquation));
    if (debug_draw_id_) {
        cb.blendEnable = VK_TRUE;
        cb.srcColorBlendFactor = cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
        cb.dstColorBlendFactor = cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cb.colorBlendOp = cb.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    const u32 mask = s.color_mask();
    cb.colorWriteMask = (mask & (1u << 16) ? static_cast<u32>(VK_COLOR_COMPONENT_R_BIT) : 0u) | (mask & (1u << 8) ? static_cast<u32>(VK_COLOR_COMPONENT_G_BIT) : 0u) |
                        (mask & 1u ? static_cast<u32>(VK_COLOR_COMPONENT_B_BIT) : 0u) | (mask & (1u << 24) ? static_cast<u32>(VK_COLOR_COMPONENT_A_BIT) : 0u);
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &cb;
    const VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                       VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 5;
    dyn.pDynamicStates = dynamics;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vp;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &blend;
    gpi.pDynamicState = &dyn;
    gpi.layout = pipeline_layout_;
    gpi.renderPass = t.pass;
    VkPipeline p = VK_NULL_HANDLE;
    if (VkResult r = fn_->vkCreateGraphicsPipelines(dev_.device(), pipeline_cache_, 1, &gpi, nullptr, &p); r != VK_SUCCESS) {
        error = std::format("vkCreateGraphicsPipelines: {}", vk::result_name(r));
        return VK_NULL_HANDLE;
    }
    ++stats_.pipelines;
    pipelines_[key] = p;
    return p;
}

u32 VulkanRenderer::texture_phys(const KelvinState& s, const TextureStage& ts) const {
    const DmaObject dma = handles_.resolve(s.dma_handle(ts.dma_select() == 2 ? mthd::kDmaB : mthd::kDmaA));
    return dma.base + ts.offset;
}

// The bytes a decode reads (texels and palette), hashed eight at a time: a check for change, not a cryptographic hash.
u64 VulkanRenderer::texture_content(u32 phys, u32 length, u32 palette_phys) const {
    const MemoryView ram{ram_, ram_size_};
    u64 h = 0x9E3779B97F4A7C15ull ^ length;
    auto mix = [&h](const u8* p, u32 n) {
        u32 i = 0;
        for (; i + 8 <= n; i += 8) {
            u64 w;
            std::memcpy(&w, p + i, 8);
            h = (h ^ w) * 0xFF51AFD7ED558CCDull;
            h ^= h >> 32;
        }
        for (; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
    };
    if (const u8* p = ram.at(phys, length)) mix(p, length);
    if (palette_phys)
        if (const u8* p = ram.at(palette_phys, 256 * 4)) mix(p, 256 * 4);
    return h;
}

// RAM is what the texture unit reads: an entry is stale once the CPU has rewritten its texels or its palette (a movie
// frame decoded into the texture, glyphs drawn into a font page) since it was decoded. Write tracking is per page, and
// a texture shares its pages with whatever the title keeps beside it (one title rewrote such neighbours several times
// a frame, and 70 % of its frame time went to decoding unchanged textures again), so a texture whose pages were
// written is hashed before it is decoded again, and kept when its bytes are what they were.
bool VulkanRenderer::texture_current(TextureEntry& e) {
    const u32 newest = std::max(handles_.newest_write(e.phys, e.length), e.palette_phys ? handles_.newest_write(e.palette_phys, 256 * 4) : 0u);
    if (newest <= e.generation) return true;
    if (texture_content(e.phys, e.length, e.palette_phys) != e.content) return false;
    e.generation = handles_.write_generation();
    ++stats_.textures_unchanged;
    return true;
}

const VulkanRenderer::TextureEntry* VulkanRenderer::texture(const KelvinState& s, u32 stage, std::string& error) {
    Timed timed(stats_.ns_texture);
    const TextureStage ts = s.texture(stage);
    const u32 b = mthd::kTextureBase + stage * mthd::kTextureStride;
    const std::array<u32, 11> regs{s.reg(b),        s.reg(b + 4),  s.reg(b + 8),  s.reg(b + 0xC),  s.reg(b + 0x10), s.reg(b + 0x14),
                                   s.reg(b + 0x1C), s.reg(b + 0x20), s.reg(b + 0x24), s.dma_handle(mthd::kDmaA), s.dma_handle(mthd::kDmaB)};
    StageCache& sc = stage_cache_[stage];
    const bool same = sc.entry != nullptr && sc.regs == regs;
    u32 phys, length;
    if (same) {
        phys = sc.phys;
        length = sc.length;
    } else {
        phys = texture_phys(s, ts);
        length = texture_size_in_memory(ts);
        if (length == 0) {
            error = std::format("texture stage {} format {:02x} has no known size", stage, ts.color_format());
            return nullptr;
        }
    }
    if (TextureEntry* e = surface_texture(ts, phys)) {
        sc = StageCache{regs, phys, length, e};
        return e;
    }
    // A render target the guest now samples: its pixels must be in RAM first — the colour image, or the depth
    // buffer when the texture is a shadow map.
    wb_why_ = Stats::kWbTexture;
    for (auto& [k, surf] : surfaces_) {
        const u64 a0 = surf->phys, a1 = static_cast<u64>(surf->phys) + surf->bytes();
        if (debug_tex_log_ && ((surf->dirty && a0 < static_cast<u64>(phys) + length && static_cast<u64>(phys) < a1) || (surf->zeta_dirty && surf->zeta_phys < static_cast<u64>(phys) + length && static_cast<u64>(phys) < static_cast<u64>(surf->zeta_phys) + surf->zeta_bytes())))
            std::fprintf(stderr, "texlog: forced write-back %08x/%08x for a texture at %08x fmt %02x %ux%u: open %d ram_newer %d\n", surf->phys, surf->zeta_phys, phys, ts.color_format(), surf->width, surf->height, (pass_open_ && pass_target_ == surf.get()) ? 1 : 0, surf->ram_newer ? 1 : 0);
        if (surf->dirty && a0 < static_cast<u64>(phys) + length && static_cast<u64>(phys) < a1) queue_write_back(*surf);
        const u64 z0 = surf->zeta_phys, z1 = static_cast<u64>(surf->zeta_phys) + surf->zeta_bytes();
        if (surf->zeta_dirty && z0 < static_cast<u64>(phys) + length && static_cast<u64>(phys) < z1) queue_write_back_zeta(*surf);
    }
    finish_write_backs();
    wb_why_ = Stats::kWbAll;
    if (same && sc.entry->valid && !sc.entry->surface && texture_current(*sc.entry)) return sc.entry;
    struct Ident {
        u32 phys, format, control1, rect, palette;
    } id{phys, ts.format, ts.control1, ts.image_rect, ts.palette};
    std::string key(reinterpret_cast<const char*>(&id), sizeof id);
    auto it = textures_.find(key);
    if (it != textures_.end() && it->second->valid && texture_current(*it->second)) {
        sc = StageCache{regs, phys, length, it->second.get()};
        return it->second.get();
    }

    DecodedTexture d;
    const u32 generation = handles_.write_generation();
    u32 palette_phys = 0;
    const u8* palette = nullptr;
    if (ts.color_format() == 0x0B) {
        const DmaObject pdma = handles_.resolve(s.dma_handle((ts.palette & 1u) ? mthd::kDmaB : mthd::kDmaA));
        palette = MemoryView{ram_, ram_size_}.at(pdma.base + (ts.palette & 0xFFFFFFC0u), 256 * 4);
        palette_phys = pdma.base + (ts.palette & 0xFFFFFFC0u);
    }
    if (!decode_texture(ts, phys, MemoryView{ram_, ram_size_}, palette, d)) {
        error = d.error;
        return nullptr;
    }
    ++stats_.textures_decoded;
    if (debug_tex_log_) {
        bool target = false;
        for (auto& [k2, surf] : surfaces_)
            target = target || (surf->phys < phys + length && phys < surf->phys + surf->bytes()) || (surf->zeta_phys < phys + length && phys < surf->zeta_phys + surf->zeta_bytes());
        std::fprintf(stderr, "texlog: decode %08x format %02x %ux%u levels %u faces %u filter %08x%s\n", phys, ts.color_format(), d.width, d.height, d.levels, d.faces, ts.filter, target ? " (a render target)" : "");
        // MV_RENDER_DEBUG=texlog,texdump: the first decode of each address, format and size as scratch/texdump/*.png.
        static const bool dump_all = std::getenv("MV_RENDER_DEBUG") && std::string_view(std::getenv("MV_RENDER_DEBUG")).find("texdump") != std::string_view::npos;
        // MV_TEXDUMP_EVERY=N: every Nth decode of each texture instead of only the first (named -N in decodes).
        static const u32 every = std::getenv("MV_TEXDUMP_EVERY") ? static_cast<u32>(std::atoi(std::getenv("MV_TEXDUMP_EVERY"))) : 0u;
        static std::unordered_map<u64, u32> dumped;
        const u64 dump_key = (static_cast<u64>(phys) << 32) ^ (static_cast<u64>(ts.color_format()) << 24) ^ (static_cast<u64>(d.width) << 12) ^ d.height;
        const u32 nth = dumped[dump_key]++;
        if (dump_all && d.pixel == HostPixel::Rgba8 && d.depth == 1 && (nth == 0 || (every && nth % every == 0))) {
            std::error_code ec;
            std::filesystem::create_directories("scratch/texdump", ec);
            std::vector<u8> img(d.image(0, 0));
            for (std::size_t i = 3; i < img.size(); i += 4) img[i] = 255;
            std::string e;
            png::write(every ? std::format("scratch/texdump/{:08x}-{:02x}-{}x{}-{:06}.png", phys, ts.color_format(), d.width, d.height, nth)
                             : std::format("scratch/texdump/{:08x}-{:02x}-{}x{}.png", phys, ts.color_format(), d.width, d.height),
                       d.width, d.height, img.data(), d.width * 4, e);
        }
        // And every 50th cube map decoded, its six faces side by side, to scratch/texlog-cube-PHYS-N.png.
        static u32 cubes = 0;
        if (d.faces == 6 && d.pixel == HostPixel::Rgba8 && cubes++ % 50 == 0) {
            std::vector<u8> strip(static_cast<std::size_t>(d.width) * 6 * d.height * 4);
            for (u32 face = 0; face < 6; ++face)
                for (u32 y = 0; y < d.height; ++y)
                    std::memcpy(&strip[(static_cast<std::size_t>(y) * d.width * 6 + face * d.width) * 4], &d.image(face, 0)[static_cast<std::size_t>(y) * d.width * 4], d.width * 4);
            for (std::size_t i = 3; i < strip.size(); i += 4) strip[i] = 255;
            std::string e;
            png::write(std::format("scratch/texlog-cube-{:08x}-{}.png", phys, cubes), d.width * 6, d.height, strip.data(), d.width * 6 * 4, e);
        }
    }
    const VkFormat format = d.pixel == HostPixel::Rgba8 ? VK_FORMAT_R8G8B8A8_UNORM : (d.pixel == HostPixel::R16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT);
    auto entry = std::make_unique<TextureEntry>();
    if (it != textures_.end()) {
        entry = std::move(it->second);
        textures_.erase(it);
        retire(entry->image);  // draws recorded or in flight may still sample it
    }
    for (u32 face = 0; face < d.faces; ++face) {
        for (u32 level = 0; level < d.levels; ++level) {
            if (d.image(face, level).size() > staging_.region) {
                error = "texture level too large for the staging buffer";
                return nullptr;
            }
        }
    }
    if (!dev_.create_image(format, d.width, d.height, d.depth, d.levels, d.faces, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT, d.faces == 6, entry->image, error))
        return nullptr;
    // Upload each face and level through the staging ring, recorded into the frame ahead of the draw that samples
    // the texture (copies cannot sit inside a render pass).
    end_pass();
    fresh_image(entry->image);
    for (u32 face = 0; face < d.faces; ++face) {
        u32 w = d.width, h = d.height, dp = d.depth;
        for (u32 level = 0; level < d.levels; ++level) {
            const auto& img = d.image(face, level);
            VkDeviceSize offset = 0;
            staging_alloc(img.size(), offset);  // fits: checked above
            std::memcpy(static_cast<u8*>(staging_.buffer.mapped) + offset, img.data(), img.size());
            VkBufferImageCopy region{};
            region.bufferOffset = offset;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, face, 1};
            region.imageExtent = {w, h, dp};
            fn_->vkCmdCopyBufferToImage(cb_, staging_.buffer.buffer, entry->image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            w = std::max(w >> 1, 1u);
            h = std::max(h >> 1, 1u);
            if (d.depth > 1) dp = std::max(dp >> 1, 1u);
        }
    }
    upload_barrier();
    entry->phys = phys;
    entry->length = length;
    entry->valid = true;
    entry->width = d.width;
    entry->height = d.height;
    entry->linear = format_info(ts.color_format()).linear;
    entry->generation = generation;
    entry->palette_phys = palette_phys;
    entry->content = texture_content(phys, length, palette_phys);
    TextureEntry* raw = entry.get();
    textures_[key] = std::move(entry);
    sc = StageCache{regs, phys, length, raw};
    return raw;
}

// The guest often samples a target it has just drawn (the boot animation's environment faces, its shadow maps and
// the composite it then downsamples). While the target is dirty its device image is newer than RAM; just after a
// write-back (every frame end) it equals RAM. A texture with the target's address and size and a format of the same
// bytes then reads the pixels the hardware would: sample the image in place instead of writing it back, decoding it
// and uploading it again (a full flush and a round trip each time, which held the CPU thread long enough to slow
// the animation, and with it the boot sound's cues, to half speed). A clean target qualifies only if it was drawn
// this frame or the last and no blit wrote over it since; an older one goes through RAM.
VulkanRenderer::TextureEntry* VulkanRenderer::surface_texture(const TextureStage& ts, u32 phys) {
    if (debug_roundtrip_ || ts.cubemap() || ts.dimensionality() != 2 || ts.mip_levels() > 1) return nullptr;
    const u32 fmt = ts.color_format();
    const bool color = fmt == 0x06 || fmt == 0x12 || fmt == 0x07 || fmt == 0x1E;  // A8R8G8B8 and X8R8G8B8
    const bool x8 = fmt == 0x07 || fmt == 0x1E;
    const bool depth16 = fmt == 0x2C || fmt == 0x30, depth24 = fmt == 0x2A || fmt == 0x2E;  // the fixed-point depths
    if (!color && !depth16 && !depth24) return nullptr;
    const bool linear = format_info(fmt).linear;
    const u32 w = linear ? ts.rect_width() : 1u << ts.log_width(), h = linear ? ts.rect_height() : 1u << ts.log_height();
    for (auto& [k, sp] : surfaces_) {
        Surface& s = *sp;
        if (debug_tex_log_ && (s.phys == phys || s.zeta_phys == phys))
            std::fprintf(stderr, "texlog: stage %08x fmt %02x %ux%u %s pitch %u vs target %08x/%08x %ux%u %s pitch %u/%u colour %u zeta %u dirty %d/%d aa %ux%u open %d\n", phys, fmt, w, h,
                         linear ? "linear" : "swizzled", ts.pitch(), s.phys, s.zeta_phys, s.width, s.height, s.swizzled ? "swizzled" : "linear", s.pitch, s.zeta_pitch,
                         s.color_format, s.zeta_format, s.dirty ? 1 : 0, s.zeta_dirty ? 1 : 0, s.aa_x, s.aa_y, (pass_open_ && pass_target_ == &s) ? 1 : 0);
        // The layout the target was last set up with does not matter: its image holds pixels, and the guest clears in
        // pitch layout (one colour, the same either way) and draws in the layout it samples.
        if (s.width != w || s.height != h) continue;
        const bool recent = !s.ram_newer && s.last_rendered_frame + 1 >= frame_index_;
        // The depth buffer read as colour at its own address: converted on the GPU (Z24S8 on a device with D24S8).
        if (color && s.zeta_phys == phys && s.phys != phys && s.zeta_format == 2 && depth24_ == VK_FORMAT_D24_UNORM_S8_UINT && s.aa_x == 1 && s.aa_y == 1 &&
            (s.zeta_dirty || recent) && linear == !s.swizzled && (!linear || ts.pitch() == s.zeta_pitch))
            return zeta_color_texture(s, ts, phys, w, h, linear, x8);
        if (color) {
            if (s.phys != phys || !(s.dirty || recent) || s.color_bytes() != 4 || color_vk_format(s.color_format) != VK_FORMAT_B8G8R8A8_UNORM) continue;
            if (linear && ts.pitch() != s.pitch) continue;
        } else {
            if (s.zeta_phys != phys || !(s.zeta_dirty || recent) || (s.zeta_format == 1) != depth16) continue;
            if (!(depth16 ? depth16_sampled_ : depth24_sampled_) || (linear && ts.pitch() != s.zeta_pitch)) continue;
        }
        if (&s == draw_target_) {
            if (!color) return nullptr;  // a draw sampling its own depth buffer: through RAM
            return snapshot_texture(s, ts, phys, w, h, linear, x8);
        }
        TextureEntry& e = color ? (x8 ? s.color_x8_tex : s.color_tex) : s.depth_tex;
        if (!e.valid) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            VkImageView view = color ? s.color.view : VK_NULL_HANDLE;
            if (x8 || !color) {
                vci.image = color ? s.color.image : s.depth.image;
                vci.format = color ? s.color.format : s.depth.format;
                vci.subresourceRange = {static_cast<VkImageAspectFlags>(color ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT), 0, 1, 0, 1};
                if (x8) vci.components.a = VK_COMPONENT_SWIZZLE_ONE;
                if (fn_->vkCreateImageView(dev_.device(), &vci, nullptr, &view) != VK_SUCCESS) return nullptr;
                (color ? s.x8_view : s.depth_view) = view;
            }
            e.image = color ? s.color : s.depth;
            e.image.view = view;
            e.image.levels = 1;
            e.phys = phys;
            e.width = w;
            e.height = h;
            e.linear = linear;
            e.surface = true;
            e.valid = true;
        }
        e.length = texture_size_in_memory(ts);
        if (attachments_written_) surface_barrier();
        sampled_since_barrier_ = true;
        ++stats_.surface_samples;
        return &e;
    }
    return nullptr;
}

// A draw samples the target it draws into: on the console the texture unit reads RAM, which holds what the draws
// before this one left. A copy of the image taken now (on the GPU, outside the pass) serves it, and is retaken only
// after the target has been drawn into or cleared since; before, each such draw wrote the target back to RAM,
// waited for the GPU and decoded the texture again (a glare buffer blurred in place did that many times a frame).
VulkanRenderer::TextureEntry* VulkanRenderer::snapshot_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear, bool x8) {
    if (!s.snap.image) {
        std::string error;
        if (!dev_.create_image(s.color.format, s.color.width, s.color.height, 1, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT, false, s.snap, error)) {
            note("snapshot image: " + error);
            return nullptr;
        }
        end_pass();
        fresh_image(s.snap);
        s.snap_serial = ~0ull;
    }
    if (s.snap_serial != s.draw_serial) {
        end_pass();
        // The target's attachment writes before the copy reads it; earlier draws' reads of the copy before it is
        // overwritten.
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0,
                                  nullptr);
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.extent = {s.color.width, s.color.height, 1};
        fn_->vkCmdCopyImage(cb_, s.color.image, VK_IMAGE_LAYOUT_GENERAL, s.snap.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        // The copy before the draws that sample it, and before the target is drawn into again.
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        s.snap_serial = s.draw_serial;
        ++stats_.snapshots;
    }
    TextureEntry& e = x8 ? s.snap_x8_tex : s.snap_tex;
    if (!e.valid) {
        VkImageView view = s.snap.view;
        if (x8) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.image = s.snap.image;
            vci.format = s.snap.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vci.components.a = VK_COMPONENT_SWIZZLE_ONE;
            if (fn_->vkCreateImageView(dev_.device(), &vci, nullptr, &view) != VK_SUCCESS) return nullptr;
            s.snap_x8_view = view;
        }
        e.image = s.snap;
        e.image.view = view;
        e.image.levels = 1;
        e.phys = phys;
        e.width = w;
        e.height = h;
        e.linear = linear;
        e.surface = true;
        e.valid = true;
    }
    e.length = texture_size_in_memory(ts);
    return &e;
}

// The conversion's pieces, made the first time a draw reads a depth buffer as colour.
bool VulkanRenderer::init_zeta_convert() {
    if (zc_pipeline_) return true;
    if (zc_failed_) return false;
    zc_failed_ = true;
    const auto& f = *fn_;
    std::string error;
    const VkDeviceSize n = kZetaConvertMaxTexels;
    if (!dev_.create_buffer(n * 5, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk::BufferMemory::Device, zc_in_, error) ||
        !dev_.create_buffer(n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk::BufferMemory::Device, zc_out_, error)) {
        note("depth conversion buffers: " + error);
        return false;
    }
    // Word i of the depth copy holds the depth in its low 24 bits; the stencil bytes follow the words, packed.
    const CompiledShader cs = compile_glsl(ShaderStage::Compute,
                                           "#version 450\n"
                                           "layout(local_size_x = 64) in;\n"
                                           "layout(std430, binding = 0) readonly buffer In { uint words[]; } src;\n"
                                           "layout(std430, binding = 1) writeonly buffer Out { uint words[]; } dst;\n"
                                           "layout(push_constant) uniform P { uint count; uint stencil_word; } pc;\n"
                                           "void main() {\n"
                                           "    uint i = gl_GlobalInvocationID.x;\n"
                                           "    if (i >= pc.count) return;\n"
                                           "    uint s = (src.words[pc.stencil_word + (i >> 2)] >> ((i & 3u) * 8u)) & 0xFFu;\n"
                                           "    dst.words[i] = ((src.words[i] & 0xFFFFFFu) << 8) | s;\n"
                                           "}\n",
                                           "zeta-to-color");
    if (!cs.ok()) {
        note("depth conversion shader: " + cs.log);
        return false;
    }
    zc_module_ = module(cs.spirv);
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (u32 i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 2;
    dli.pBindings = bindings;
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &zc_set_layout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;
    if (f.vkCreateDescriptorSetLayout(dev_.device(), &dli, nullptr, &zc_set_layout_) != VK_SUCCESS ||
        f.vkCreatePipelineLayout(dev_.device(), &pli, nullptr, &zc_layout_) != VK_SUCCESS) {
        note("depth conversion layouts failed");
        return false;
    }
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = zc_module_;
    cpi.stage.pName = "main";
    cpi.layout = zc_layout_;
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    if (f.vkCreateComputePipelines(dev_.device(), pipeline_cache_, 1, &cpi, nullptr, &zc_pipeline_) != VK_SUCCESS ||
        f.vkCreateDescriptorPool(dev_.device(), &dpi, nullptr, &zc_pool_) != VK_SUCCESS) {
        note("depth conversion pipeline failed");
        return false;
    }
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = zc_pool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &zc_set_layout_;
    if (f.vkAllocateDescriptorSets(dev_.device(), &dai, &zc_set_) != VK_SUCCESS) {
        note("depth conversion descriptor set failed");
        return false;
    }
    VkDescriptorBufferInfo infos[2] = {{zc_in_.buffer, 0, VK_WHOLE_SIZE}, {zc_out_.buffer, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[2]{};
    for (u32 i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = zc_set_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    f.vkUpdateDescriptorSets(dev_.device(), 2, writes, 0, nullptr);
    zc_failed_ = false;
    return true;
}

// A draw reads a Z24S8 depth buffer as an A8R8G8B8 texture at the same address (a title masking its glare by
// depth): the texels are the words RAM would hold, the depth in bits 31-8 above the stencil byte. They are made on
// the GPU — depth and stencil copied into a buffer, packed by a compute pass, copied into a colour image — where
// before each read wrote the whole depth buffer back to RAM, waited for the GPU and decoded it again.
VulkanRenderer::TextureEntry* VulkanRenderer::zeta_color_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear, bool x8) {
    const u32 n = s.width * s.height;
    if (n > kZetaConvertMaxTexels || !init_zeta_convert()) return nullptr;
    if (!s.zcolor.image) {
        std::string error;
        if (!dev_.create_image(VK_FORMAT_B8G8R8A8_UNORM, s.width, s.height, 1, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT, false, s.zcolor, error)) {
            note("depth-as-colour image: " + error);
            return nullptr;
        }
        end_pass();
        fresh_image(s.zcolor);
        s.zcolor_serial = ~0ull;
    }
    if (s.zcolor_serial != s.draw_serial) {
        end_pass();
        const auto barrier = [this](VkPipelineStageFlags from, VkAccessFlags src, VkPipelineStageFlags to, VkAccessFlags dst) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = src;
            mb.dstAccessMask = dst;
            fn_->vkCmdPipelineBarrier(cb_, from, to, 0, 1, &mb, 0, nullptr, 0, nullptr);
        };
        // The depth writes (and an earlier conversion's reads of the buffers) before the copies.
        barrier(VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy regions[2]{};
        regions[0].imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        regions[0].imageExtent = {s.width, s.height, 1};
        regions[1] = regions[0];
        regions[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        regions[1].bufferOffset = static_cast<VkDeviceSize>(n) * 4;
        fn_->vkCmdCopyImageToBuffer(cb_, s.depth.image, VK_IMAGE_LAYOUT_GENERAL, zc_in_.buffer, 2, regions);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        fn_->vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, zc_pipeline_);
        fn_->vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, zc_layout_, 0, 1, &zc_set_, 0, nullptr);
        const u32 push[2] = {n, n};
        fn_->vkCmdPushConstants(cb_, zc_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, push);
        fn_->vkCmdDispatch(cb_, (n + 63) / 64, 1, 1);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy out{};
        out.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        out.imageExtent = {s.width, s.height, 1};
        fn_->vkCmdCopyBufferToImage(cb_, zc_out_.buffer, s.zcolor.image, VK_IMAGE_LAYOUT_GENERAL, 1, &out);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        s.zcolor_serial = s.draw_serial;
        ++stats_.zeta_conversions;
    }
    TextureEntry& e = x8 ? s.zcolor_x8_tex : s.zcolor_tex;
    if (!e.valid) {
        VkImageView view = s.zcolor.view;
        if (x8) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.image = s.zcolor.image;
            vci.format = s.zcolor.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vci.components.a = VK_COMPONENT_SWIZZLE_ONE;
            if (fn_->vkCreateImageView(dev_.device(), &vci, nullptr, &view) != VK_SUCCESS) return nullptr;
            s.zcolor_x8_view = view;
        }
        e.image = s.zcolor;
        e.image.view = view;
        e.image.levels = 1;
        e.phys = phys;
        e.width = w;
        e.height = h;
        e.linear = linear;
        e.surface = true;
        e.valid = true;
    }
    e.length = texture_size_in_memory(ts);
    return &e;
}

void VulkanRenderer::surface_barrier() {
    end_pass();
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    attachments_written_ = false;
    sampled_since_barrier_ = false;
}

void VulkanRenderer::destroy_surface_views(Surface& s) {
    if (s.x8_view) fn_->vkDestroyImageView(dev_.device(), s.x8_view, nullptr);
    if (s.depth_view) fn_->vkDestroyImageView(dev_.device(), s.depth_view, nullptr);
    if (s.snap_x8_view) fn_->vkDestroyImageView(dev_.device(), s.snap_x8_view, nullptr);
    if (s.zcolor_x8_view) fn_->vkDestroyImageView(dev_.device(), s.zcolor_x8_view, nullptr);
    s.x8_view = s.depth_view = s.snap_x8_view = s.zcolor_x8_view = VK_NULL_HANDLE;
    s.color_tex = s.color_x8_tex = s.depth_tex = s.snap_tex = s.snap_x8_tex = s.zcolor_tex = s.zcolor_x8_tex = TextureEntry{};
}

VkSampler VulkanRenderer::sampler(const TextureStage& ts, u32 levels) {
    const u64 key = (static_cast<u64>(ts.filter) << 32) | ts.address | (static_cast<u64>(levels) << 28);
    if (auto it = samplers_.find(key); it != samplers_.end()) return it->second;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    const u32 min = ts.min_filter(), mag = ts.mag_filter();
    sci.magFilter = (mag == 1) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    sci.minFilter = (min == 1 || min == 3 || min == 5) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    sci.mipmapMode = (min == 5 || min == 6) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = wrap_mode(ts.wrap_u());
    sci.addressModeV = wrap_mode(ts.wrap_v());
    sci.addressModeW = wrap_mode(ts.wrap_p());
    sci.maxLod = (min >= 3 && min <= 6) ? static_cast<float>(levels) : 0.25f;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    VkSampler sm = VK_NULL_HANDLE;
    fn_->vkCreateSampler(dev_.device(), &sci, nullptr, &sm);
    samplers_[key] = sm;
    return sm;
}

bool VulkanRenderer::fits_in_batch(VkDeviceSize vertex_bytes, VkDeviceSize index_bytes) const {
    auto fits = [](const Ring& ring, VkDeviceSize bytes, VkDeviceSize alignment) {
        return (ring.cursor + alignment - 1) / alignment * alignment + bytes <= ring.end();
    };
    const VkDeviceSize ualign = std::max<VkDeviceSize>(dev_.limits().minUniformBufferOffsetAlignment, 16);
    const VkDeviceSize uniforms = (vsh::kUniformBytes + ualign - 1) / ualign * ualign + sizeof(psh::Uniforms);
    return fits(vertex_ring_, vertex_bytes, 16) && (index_bytes == 0 || fits(index_ring_, index_bytes, 4)) && fits(uniform_ring_, uniforms, ualign) &&
           descriptor_sets_used_ < kMaxDescriptorSets;
}

bool VulkanRenderer::ring_alloc(Ring& ring, VkDeviceSize bytes, VkDeviceSize alignment, VkDeviceSize& offset) {
    const VkDeviceSize start = (ring.cursor + alignment - 1) / alignment * alignment;
    if (start + bytes > ring.end()) return false;
    offset = start;
    ring.cursor = start + bytes;
    return true;
}

bool VulkanRenderer::staging_alloc(VkDeviceSize bytes, VkDeviceSize& offset) {
    // Sixteen bytes keep every copy's buffer offset a multiple of its texel size.
    if (bytes > staging_.region) return false;
    if (ring_alloc(staging_, bytes, 16, offset)) return true;
    // The frame's region is full: the copies so far go with its submission (still ahead, in queue order, of the
    // draws recorded after them), and the next frame's region is free once its last submission has run.
    submit();
    return ring_alloc(staging_, bytes, 16, offset);
}

void VulkanRenderer::fresh_image(const vk::Image& image) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image.image;
    b.subresourceRange = {image.aspect, 0, image.levels, 0, image.layers};
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void VulkanRenderer::upload_barrier() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void VulkanRenderer::clear(const KelvinState& s, u32 bits) {
    ++stats_.clears;
    Surface* t = target_for(s);
    if (!t) {
        ++stats_.skipped;
        return;
    }
    ensure_pass(*t);
    if (bits & 0xF0u) t->dirty = true;
    ++t->clears_this_frame;
    ++t->draw_serial;
    VkClearAttachment att[2]{};
    u32 n = 0;
    const u32 rect_h = s.reg(mthd::kClearRectH), rect_v = s.reg(mthd::kClearRectV);
    u32 x0 = rect_h & 0xFFFFu, x1 = rect_h >> 16, y0 = rect_v & 0xFFFFu, y1 = rect_v >> 16;
    if (x1 < x0 || y1 < y0 || (x0 == 0 && x1 == 0 && y0 == 0 && y1 == 0)) {
        x0 = y0 = 0;
        x1 = t->width - 1;
        y1 = t->height - 1;
    } else {
        // The clear rectangle is in logical pixels too.
        x0 *= t->aa_x;
        x1 = (x1 + 1) * t->aa_x - 1;
        y0 *= t->aa_y;
        y1 = (y1 + 1) * t->aa_y - 1;
    }
    x1 = std::min(x1, t->width - 1);
    y1 = std::min(y1, t->height - 1);
    if (debug_draw_id_)
        std::fprintf(stderr, "drawid clear: surface %08x bits %02x colour %08x zstencil %08x rect %u,%u..%u,%u of %ux%u (registers %08x %08x)\n", t->phys, bits, s.clear_color(),
                     s.clear_zstencil(), x0, y0, x1, y1, t->width, t->height, rect_h, rect_v);
    float color[4];
    clear_color_rgba(t->color_format, s.clear_color(), color);
    // CLEAR_SURFACE bits 4-7 select R, G, B and A. All four is an attachment clear; fewer is a full-screen
    // triangle that writes only those channels (an attachment clear ignores the write mask).
    const u32 channels = (bits >> 4) & 0xFu;
    if (channels == 0xFu) {
        att[n].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        att[n].colorAttachment = 0;
        std::memcpy(att[n].clearValue.color.float32, color, sizeof color);
        ++n;
    }
    if (bits & 3u) {
        const u32 z = s.clear_zstencil();
        att[n].aspectMask = (bits & 1u ? static_cast<u32>(VK_IMAGE_ASPECT_DEPTH_BIT) : 0u) | ((bits & 2u) && t->zeta_format != 1 ? static_cast<u32>(VK_IMAGE_ASPECT_STENCIL_BIT) : 0u);
        if (att[n].aspectMask) {
            att[n].clearValue.depthStencil.depth = t->zeta_format == 1 ? static_cast<float>(z & 0xFFFFu) / 65535.0f : static_cast<float>(z >> 8) / 16777215.0f;
            att[n].clearValue.depthStencil.stencil = z & 0xFFu;
            ++n;
        }
    }
    VkClearRect rect{};
    rect.rect = {{static_cast<i32>(x0), static_cast<i32>(y0)}, {x1 - x0 + 1, y1 - y0 + 1}};
    rect.layerCount = 1;
    if (n) fn_->vkCmdClearAttachments(cb_, n, att, 1, &rect);
    if (channels != 0 && channels != 0xFu) {
        const VkColorComponentFlags mask = ((channels & 1u) ? static_cast<u32>(VK_COLOR_COMPONENT_R_BIT) : 0u) |
                                           ((channels & 2u) ? static_cast<u32>(VK_COLOR_COMPONENT_G_BIT) : 0u) |
                                           ((channels & 4u) ? static_cast<u32>(VK_COLOR_COMPONENT_B_BIT) : 0u) |
                                           ((channels & 8u) ? static_cast<u32>(VK_COLOR_COMPONENT_A_BIT) : 0u);
        std::string error;
        VkPipeline p = clear_pipeline(t->pass, mask, error);
        if (!p) {
            note(error);
            return;
        }
        fn_->vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(t->width), static_cast<float>(t->height), 0.0f, 1.0f};
        fn_->vkCmdSetViewport(cb_, 0, 1, &viewport);
        fn_->vkCmdSetScissor(cb_, 0, 1, &rect.rect);
        fn_->vkCmdSetBlendConstants(cb_, color);
        fn_->vkCmdDraw(cb_, 3, 1, 0, 0);
    }
}

VkPipeline VulkanRenderer::clear_pipeline(VkRenderPass pass, VkColorComponentFlags mask, std::string& error) {
    const u64 key = (reinterpret_cast<u64>(pass) << 4) ^ mask;
    if (auto it = clear_pipelines_.find(key); it != clear_pipelines_.end()) return it->second;
    if (!clear_vs_) {
        // A triangle over the whole viewport, and a fragment of 1.0 that the blend scales by the clear colour.
        const CompiledShader vs = compile_glsl(ShaderStage::Vertex,
                                               "#version 450\n"
                                               "void main() {\n"
                                               "    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);\n"
                                               "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
                                               "}\n",
                                               "clear-vs");
        const CompiledShader fs = compile_glsl(ShaderStage::Fragment,
                                               "#version 450\n"
                                               "layout(location = 0) out vec4 frag_color;\n"
                                               "void main() { frag_color = vec4(1.0); }\n",
                                               "clear-fs");
        if (!vs.ok() || !fs.ok()) {
            error = "clear shaders: " + vs.log + fs.log;
            return VK_NULL_HANDLE;
        }
        clear_vs_ = module(vs.spirv);
        clear_fs_ = module(fs.spirv);
    }
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = clear_vs_;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = clear_fs_;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    // The colour is the blend constant times one: exact in every unsigned normalised format.
    VkPipelineColorBlendAttachmentState cb{};
    cb.blendEnable = VK_TRUE;
    cb.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
    cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
    cb.dstColorBlendFactor = cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cb.colorBlendOp = cb.alphaBlendOp = VK_BLEND_OP_ADD;
    cb.colorWriteMask = mask;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &cb;
    const VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 3;
    dyn.pDynamicStates = dynamics;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vp;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &blend;
    gpi.pDynamicState = &dyn;
    gpi.layout = pipeline_layout_;
    gpi.renderPass = pass;
    VkPipeline p = VK_NULL_HANDLE;
    if (VkResult r = fn_->vkCreateGraphicsPipelines(dev_.device(), VK_NULL_HANDLE, 1, &gpi, nullptr, &p); r != VK_SUCCESS) {
        error = std::format("clear pipeline: vkCreateGraphicsPipelines: {}", vk::result_name(r));
        return VK_NULL_HANDLE;
    }
    ++stats_.pipelines;
    clear_pipelines_[key] = p;
    return p;
}

void VulkanRenderer::draw(const KelvinState& s, const DrawBatch& batch) {
    Timed timed(stats_.ns_draw);
    ++stats_.draws;
    const u32 draw_id = frame_draws_++;
    if (debug_draw_id_)
        std::fprintf(stderr,
                     "drawid %u: surface %08x, stages %05x, blend %u, %zu indices, %zu ranges, %zu inline words, %u immediate; program %u, depth test %u func %x write %u, "
                     "colour mask %08x, stencil %u, clip %g..%g, zeta %08x format %08x control0 %08x\n",
                     draw_id, s.surface().color_offset, s.reg(mthd::kShaderStageProgram), s.blend() ? 1u : 0u, batch.indices.size(), batch.ranges.size(),
                     batch.inline_words.size(), batch.immediate_count, s.vertex_program_mode() ? 1u : 0u, s.depth_test() ? 1u : 0u, s.reg(mthd::kDepthFunc),
                     s.depth_write() ? 1u : 0u, s.reg(mthd::kColorMask), s.stencil_test() ? 1u : 0u, static_cast<double>(std::bit_cast<float>(s.reg(mthd::kClipMin))),
                     static_cast<double>(std::bit_cast<float>(s.reg(mthd::kClipMax))), s.reg(mthd::kSurfaceZetaOffset), s.reg(mthd::kSurfaceFormat), s.reg(mthd::kControl0));
    if (debug_draw_id_ && (s.blend() || s.surface().color_offset >= 0x03f00000u))
        std::fprintf(stderr, "drawid %u: blend sfactor %x dfactor %x equation %x colour %08x; texture 0 at %08x format %08x, alpha test %u, combiner control %08x\n", draw_id,
                     s.reg(mthd::kBlendSfactor), s.reg(mthd::kBlendDfactor), s.reg(mthd::kBlendEquation), s.reg(mthd::kBlendColor), texture_phys(s, s.texture(0)), s.texture(0).format,
                     s.alpha_test() ? 1u : 0u, s.reg(mthd::kCombinerControl));
    if (debug_draw_id_ && (s.surface().color_offset >= 0x03f00000u || s.reg(mthd::kShaderStageProgram) == 0x08421u))
        for (u32 st = 1; st < (s.reg(mthd::kCombinerControl) & 0xFFu); ++st)
            std::fprintf(stderr, "drawid %u: combiner stage %u colour in %08x out %08x alpha in %08x out %08x factors %08x %08x\n", draw_id, st, s.reg(mthd::kCombinerColorIcw + 4 * st),
                         s.reg(mthd::kCombinerColorOcw + 4 * st), s.reg(mthd::kCombinerAlphaIcw + 4 * st), s.reg(mthd::kCombinerAlphaOcw + 4 * st), s.reg(mthd::kCombinerFactor0 + 4 * st),
                         s.reg(mthd::kCombinerFactor1 + 4 * st));
    if (debug_draw_id_ && (s.surface().color_offset >= 0x03f00000u || s.reg(mthd::kShaderStageProgram) == 0x08421u))
        std::fprintf(stderr, "drawid %u: combiner stage 0 colour in %08x out %08x alpha in %08x out %08x factors %08x %08x; final %08x %08x; texture 0 rect %08x control0 %08x control1 %08x filter %08x address %08x\n",
                     draw_id, s.reg(mthd::kCombinerColorIcw), s.reg(mthd::kCombinerColorOcw), s.reg(mthd::kCombinerAlphaIcw), s.reg(mthd::kCombinerAlphaOcw),
                     s.reg(mthd::kCombinerFactor0), s.reg(mthd::kCombinerFactor1), s.reg(mthd::kCombinerSpecFogCw0), s.reg(mthd::kCombinerSpecFogCw1),
                     s.reg(mthd::kTextureBase + 0x18), s.reg(mthd::kTextureBase + 0x0C), s.reg(mthd::kTextureBase + 0x10), s.reg(mthd::kTextureBase + 0x14), s.reg(mthd::kTextureBase + 0x08));
    if (debug_skip_blend_ && s.blend()) return;  // MV_RENDER_DEBUG=skipblend: the opaque passes alone
    std::string error;
    Surface* t = target_for(s);
    if (!t) {
        ++stats_.skipped;
        return;
    }
    if (debug_tex_log_) std::fprintf(stderr, "texlog: draw into %08x %s\n", t->phys, s.surface().swizzled() ? "swizzled" : "linear");
    if (debug_draw_id_) {
        // Has anything but this renderer written the target's RAM since the last draw into it?
        static std::unordered_map<u32, u32> seen;
        const u32 newest = handles_.newest_write(t->phys, t->bytes());
        u32& last = seen[t->phys];
        if (last && newest > last) std::fprintf(stderr, "drawid %u: target %08x RAM written by the CPU or a device since its last draw (generation %u -> %u)\n", draw_id, t->phys, last, newest);
        last = std::max(newest, handles_.write_generation());
    }
    FetchedVertices v;
    bool fetched;
    {
        Timed timed_fetch(stats_.ns_fetch);
        fetched = fetch_vertices(s, handles_, MemoryView{ram_, ram_size_}, batch, v);
    }
    if (!fetched) {
        note("vertex fetch: " + v.error);
        ++stats_.skipped;
        return;
    }
    if (v.count == 0) return;

    // Shaders.
    VkShaderModule vs = vertex_shader(s, error);
    if (!vs) {
        note(error);
        ++stats_.skipped;
        return;
    }
    if (debug_draw_id_)
        if (auto it = vs_debug_ids_.find(vs); it != vs_debug_ids_.end()) std::fprintf(stderr, "drawid %u: vertex shader vs-%u\n", draw_id, it->second);
    psh::Key pkey = psh::key_from_state(s);
    if (debug_shadow_pass_) pkey.shadow_compare_func = 7;  // every shadow lookup lit
    std::array<psh::SamplerKind, 4> kinds{};
    VkShaderModule fs = fragment_shader(pkey, kinds, error);
    if (!fs) {
        note(error);
        ++stats_.skipped;
        return;
    }

    // Textures (may write surfaces back and flush: before the pass for this draw begins).
    draw_target_ = t;
    std::array<const TextureEntry*, 4> tex{};
    std::array<std::array<float, 2>, 4> scales{{{1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}}};
    for (u32 i = 0; i < 4; ++i) {
        if (kinds[i] == psh::SamplerKind::None) continue;
        tex[i] = texture(s, i, error);
        if (!tex[i]) {
            note(std::format("texture stage {}: {}", i, error));
            continue;
        }
        if (tex[i]->linear) scales[i] = {1.0f / static_cast<float>(tex[i]->width), 1.0f / static_cast<float>(tex[i]->height)};
    }

    draw_target_ = nullptr;

    // Topology and the index list the host understands.
    auto t_part = std::chrono::steady_clock::now();
    std::vector<u32> indices = v.indices;
    auto sequential = [&]() {
        if (indices.empty()) {
            indices.resize(v.count);
            for (u32 i = 0; i < v.count; ++i) indices[i] = i;
        }
    };
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    switch (v.primitive) {
        case Primitive::Points: topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case Primitive::Lines: topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case Primitive::LineStrip: topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        case Primitive::LineLoop:
            topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            sequential();
            indices.push_back(indices.front());
            break;
        case Primitive::Triangles: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
        case Primitive::TriangleStrip: case Primitive::QuadStrip: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case Primitive::TriangleFan: case Primitive::Polygon: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case Primitive::Quads: {
            sequential();
            std::vector<u32> tris;
            tris.reserve(indices.size() / 4 * 6);
            for (std::size_t i = 0; i + 3 < indices.size(); i += 4) {
                tris.insert(tris.end(), {indices[i], indices[i + 1], indices[i + 2], indices[i], indices[i + 2], indices[i + 3]});
            }
            indices = std::move(tris);
            topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            break;
        }
        case Primitive::End: return;
    }

    stats_.ns_indices += ns_since(t_part);
    VkPipeline p = pipeline(s, vs, fs, topology, *t, v.enabled, error);
    if (!p) {
        note(error);
        ++stats_.skipped;
        return;
    }

    // Vertex data: the attributes with per-vertex data as vec4s packed in order, then the sixteen constants the others
    // read (binding 1, a stride of zero). Writing every attribute for every vertex cost a draw-heavy title 3 ms a frame.
    t_part = std::chrono::steady_clock::now();
    const u32 per_vertex = static_cast<u32>(std::popcount(static_cast<u32>(v.enabled)));
    const VkDeviceSize vertex_bytes = static_cast<VkDeviceSize>(v.count) * per_vertex * 16;
    VkDeviceSize vbytes = vertex_bytes + kAttributes * 16, voffset = 0;
    // The rings and the descriptor pool hold one batch of draws. A draw that no longer fits submits the batch so far
    // and moves to the next frame in flight's regions, so a frame can hold any number of draws (only a single draw
    // larger than a ring region is refused below).
    if (!fits_in_batch(vbytes, indices.size() * 4)) {
        submit();
        ++stats_.batch_flushes;
    }
    if (!ring_alloc(vertex_ring_, vbytes, 16, voffset)) {
        note("vertex ring full for this frame");
        ++stats_.skipped;
        return;
    }
    float* vdst = reinterpret_cast<float*>(static_cast<u8*>(vertex_ring_.buffer.mapped) + voffset);
    for (u32 n = 0; n < v.count; ++n) {
        for (u32 a = 0; a < kAttributes; ++a) {
            float* d = vdst + (static_cast<std::size_t>(n) * kAttributes + a) * 4;
            if (v.enabled & (1u << a)) std::memcpy(d, &v.attribute[a][static_cast<std::size_t>(n) * 4], 16);
            else std::memcpy(d, v.constant[a].data(), 16);
        }
    }
    VkDeviceSize ioffset = 0;
    if (!indices.empty()) {
        if (!ring_alloc(index_ring_, indices.size() * 4, 4, ioffset)) {
            note("index ring full for this frame");
            ++stats_.skipped;
            return;
        }
        std::memcpy(static_cast<u8*>(index_ring_.buffer.mapped) + ioffset, indices.data(), indices.size() * 4);
    }

    // Uniforms: built on the CPU, and uploaded only when they differ from the block the frame last uploaded (a
    // run of draws under the same transforms and combiners binds the copy already in the ring).
    VkDeviceSize uoffset_vs = 0, uoffset_ps = 0;
    {
        float* u = vs_uniforms_.data();
        std::memcpy(u, s.constants.data(), kConstants * 16);
        u[kConstants * 4 + 0] = static_cast<float>(t->width / t->aa_x);  // the logical size the program maps to
        u[kConstants * 4 + 1] = static_cast<float>(t->height / t->aa_y);
        u[kConstants * 4 + 2] = 0.0f;
        u[kConstants * 4 + 3] = 0.0f;
        u[kConstants * 4 + 4] = 0.0f;
        u[kConstants * 4 + 5] = t->zeta_format == 1 ? 65535.0f : 16777215.0f;
        u[kConstants * 4 + 6] = 0.0f;
        u[kConstants * 4 + 7] = 0.0f;
        // The fixed function's material, lights and planes.
        vsh::fixed_function_uniforms(s, u + kConstants * 4 + 8);
        psh::Uniforms pu{};
        psh::uniforms_from_state(s, scales, pu);
        const VkDeviceSize ualign = std::max<VkDeviceSize>(dev_.limits().minUniformBufferOffsetAlignment, 16);
        auto upload = [&](UniformUpload& last, const void* block, std::size_t bytes, VkDeviceSize& offset) {
            if (last.valid && std::memcmp(last.bytes.data(), block, bytes) == 0) {
                offset = last.offset;
                ++stats_.uniform_reuses;
                return true;
            }
            if (!ring_alloc(uniform_ring_, bytes, ualign, offset)) return false;
            std::memcpy(static_cast<u8*>(uniform_ring_.buffer.mapped) + offset, block, bytes);
            last.bytes.assign(static_cast<const u8*>(block), static_cast<const u8*>(block) + bytes);
            last.offset = offset;
            last.valid = true;
            return true;
        };
        if (!upload(last_vs_upload_, u, vsh::kUniformBytes, uoffset_vs) || !upload(last_ps_upload_, &pu, sizeof pu, uoffset_ps)) {
            note("uniform ring full for this frame");
            ++stats_.skipped;
            return;
        }
    }

    stats_.ns_upload += ns_since(t_part);
    t_part = std::chrono::steady_clock::now();
    // Descriptors: the frame's last set serves again when it binds the same uniform offsets, views and samplers.
    VkDescriptorImageInfo imgs[4]{};
    for (u32 i = 0; i < 4; ++i) {
        const TextureStage ts = s.texture(i);
        imgs[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        if (tex[i]) {
            imgs[i].imageView = tex[i]->image.view;
            imgs[i].sampler = sampler(ts, tex[i]->image.levels);
        } else {
            imgs[i].imageView = kinds[i] == psh::SamplerKind::SamplerCube ? dummycube_.view : (kinds[i] == psh::SamplerKind::Sampler3D ? dummy3d_.view : dummy2d_.view);
            imgs[i].sampler = dummy_sampler_;
        }
    }
    DescriptorKey dkey{uoffset_vs, uoffset_ps, {}, {}};
    for (u32 i = 0; i < 4; ++i) {
        dkey.views[i] = imgs[i].imageView;
        dkey.samplers[i] = imgs[i].sampler;
    }
    VkDescriptorSet set = last_set_;
    if (set != VK_NULL_HANDLE && dkey == last_set_key_) {
        ++stats_.descriptor_reuses;
    } else {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = frames_[slot_].pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &set_layout_;
        if (fn_->vkAllocateDescriptorSets(dev_.device(), &dai, &set) != VK_SUCCESS) {
            note("descriptor pool exhausted for this frame");
            ++stats_.skipped;
            return;
        }
        ++descriptor_sets_used_;
        write_descriptors(set, uoffset_vs, uoffset_ps, imgs);
        last_set_ = set;
        last_set_key_ = dkey;
    }

    stats_.ns_descriptors += ns_since(t_part);
    t_part = std::chrono::steady_clock::now();
    // Record.
    ensure_pass(*t);
    if (s.color_mask() != 0) t->dirty = true;
    ++t->draws_this_frame;
    ++t->draw_serial;
    fn_->vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    // A bound set stays bound across pipelines (they share one layout) and passes, for the whole command buffer.
    if (set != bound_set_) {
        fn_->vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &set, 0, nullptr);
        bound_set_ = set;
    }
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(t->width), static_cast<float>(t->height), 0.0f, 1.0f};
    fn_->vkCmdSetViewport(cb_, 0, 1, &viewport);
    const SurfaceState sd = s.surface();
    const u32 cx = sd.clip_x() * t->aa_x, cy = sd.clip_y() * t->aa_y, cw = sd.clip_width() * t->aa_x, ch = sd.clip_height() * t->aa_y;
    VkRect2D scissor{{static_cast<i32>(std::min(cx, t->width - 1)), static_cast<i32>(std::min(cy, t->height - 1))},
                     {std::max(1u, std::min(cw, t->width - std::min(cx, t->width - 1))), std::max(1u, std::min(ch, t->height - std::min(cy, t->height - 1)))}};
    if (sd.clip_width() == 0 || sd.clip_height() == 0) scissor = {{0, 0}, {t->width, t->height}};
    fn_->vkCmdSetScissor(cb_, 0, 1, &scissor);
    const u32 bc = s.reg(mthd::kBlendColor);
    float constants[4] = {static_cast<float>((bc >> 16) & 0xFFu) / 255.0f, static_cast<float>((bc >> 8) & 0xFFu) / 255.0f,
                                static_cast<float>(bc & 0xFFu) / 255.0f, static_cast<float>(bc >> 24) / 255.0f};
    if (debug_draw_id_) {
        constants[0] = static_cast<float>(draw_id & 0xFFu) / 255.0f;
        constants[1] = static_cast<float>((draw_id >> 8) & 0xFFu) / 255.0f;
        constants[2] = constants[3] = 1.0f;
    }
    fn_->vkCmdSetBlendConstants(cb_, constants);
    fn_->vkCmdSetStencilReference(cb_, VK_STENCIL_FACE_FRONT_AND_BACK, s.reg(mthd::kStencilRef) & 0xFFu);
    {
        // The polygon offset: bias + factor x the depth slope, in depth-buffer units, which are the attachment's own steps
        // (depth reaches the host as z / the zeta format's maximum). It applies to triangle-class primitives, with the
        // enable of the front polygon mode (we fill every mode). Per the method definitions and the reference emulator.
        const u32 mode = s.reg(mthd::kFrontPolygonMode);
        const u32 enable = mode == 0x1B00 ? mthd::kPolyOffsetPointEnable : mode == 0x1B01 ? mthd::kPolyOffsetLineEnable : mthd::kPolyOffsetFillEnable;
        const bool triangles = v.primitive >= Primitive::Triangles;
        const bool on = triangles && s.reg(enable) != 0;
        fn_->vkCmdSetDepthBias(cb_, on ? std::bit_cast<float>(s.reg(mthd::kPolygonOffsetBias)) : 0.0f, 0.0f,
                               on ? std::bit_cast<float>(s.reg(mthd::kPolygonOffsetFactor)) : 0.0f);
    }
    const VkDeviceSize vb_offsets[2] = {voffset, voffset + vertex_bytes};
    const VkBuffer vbs[2] = {vertex_ring_.buffer.buffer, vertex_ring_.buffer.buffer};
    fn_->vkCmdBindVertexBuffers(cb_, 0, 2, vbs, vb_offsets);
    if (!indices.empty()) {
        fn_->vkCmdBindIndexBuffer(cb_, index_ring_.buffer.buffer, ioffset, VK_INDEX_TYPE_UINT32);
        fn_->vkCmdDrawIndexed(cb_, static_cast<u32>(indices.size()), 1, 0, 0, 0);
    } else {
        fn_->vkCmdDraw(cb_, v.count, 1, 0, 0);
    }
    stats_.ns_record += ns_since(t_part);
}

void VulkanRenderer::write_descriptors(VkDescriptorSet set, VkDeviceSize uoffset_vs, VkDeviceSize uoffset_ps, const VkDescriptorImageInfo* imgs) {
    VkDescriptorBufferInfo bufs[2] = {{uniform_ring_.buffer.buffer, uoffset_vs, vsh::kUniformBytes}, {uniform_ring_.buffer.buffer, uoffset_ps, sizeof(psh::Uniforms)}};
    VkWriteDescriptorSet writes[6]{};
    for (u32 i = 0; i < 6; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        if (i < 2) {
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[i].pBufferInfo = &bufs[i];
        } else {
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &imgs[i - 2];
        }
    }
    fn_->vkUpdateDescriptorSets(dev_.device(), 6, writes, 0, nullptr);
}

// Nothing on the CPU reads the frame's results here: the semaphore's value is the guest's, and the pixels reach RAM
// only through a write-back, which waits for the work it copies. So the frame is submitted and left to run.
void VulkanRenderer::flush_frame() { submit(); }

bool VulkanRenderer::write_back_at(u32 phys) {
    wb_why_ = Stats::kWbScanout;
    for (auto& [k, s] : surfaces_) {
        if (s->phys != phys) continue;
        if (s->dirty) write_back(*s);
        wb_why_ = Stats::kWbAll;
        return true;
    }
    wb_why_ = Stats::kWbAll;
    return false;
}

void VulkanRenderer::end_frame() {
    frame_draws_ = 0;
    ++frame_index_;
    // Colour reaches RAM every frame (the scanout and the CPU read it); depth only when a texture samples it.
    wb_why_ = Stats::kWbFrameEnd;
    for (auto& [k, s] : surfaces_) queue_write_back(*s);
    if (write_backs_.empty()) submit();  // nothing to copy: the frame runs on without a wait
    else finish_write_backs();
    wb_why_ = Stats::kWbAll;
}

std::vector<VulkanRenderer::SurfaceInfo> VulkanRenderer::surfaces() const {
    std::vector<SurfaceInfo> out;
    for (const auto& [k, s] : surfaces_) {
        out.push_back({s->phys, s->width, s->height, s->swizzled ? s->width * s->color_bytes() : s->pitch, s->color_bytes(), s->color_format, s->swizzled,
                       s->rendered_this_frame, s->draws_this_frame, s->clears_this_frame});
        s->rendered_this_frame = false;
        s->draws_this_frame = s->clears_this_frame = 0;
    }
    std::sort(out.begin(), out.end(), [](const SurfaceInfo& a, const SurfaceInfo& b) { return a.phys < b.phys; });
    return out;
}

bool VulkanRenderer::surface_rgba(const SurfaceInfo& s, std::vector<u8>& rgba) const {
    rgba.assign(static_cast<std::size_t>(s.width) * s.height * 4, 0);
    const u32 log_w = static_cast<u32>(std::countr_zero(s.width)), log_h = static_cast<u32>(std::countr_zero(s.height));
    for (u32 y = 0; y < s.height; ++y) {
        for (u32 x = 0; x < s.width; ++x) {
            const u32 off = s.swizzled ? swizzled_offset(x, y, 0, log_w, log_h, 0) * s.bytes_per_pixel : y * s.pitch + x * s.bytes_per_pixel;
            if (static_cast<u64>(s.phys) + off + s.bytes_per_pixel > ram_size_) return false;
            const u8* p = ram_ + s.phys + off;
            u8* d = &rgba[(static_cast<std::size_t>(y) * s.width + x) * 4];
            if (s.bytes_per_pixel == 4) {
                d[0] = p[2];
                d[1] = p[1];
                d[2] = p[0];
                d[3] = 255;
            } else if (s.bytes_per_pixel == 2) {
                u16 h;
                std::memcpy(&h, p, 2);
                if (s.color_format == 10) {  // G8B8
                    d[0] = 0;
                    d[1] = p[1];
                    d[2] = p[0];
                } else if (s.color_format == 3) {
                    d[0] = static_cast<u8>(((h >> 11) & 31) * 255 / 31);
                    d[1] = static_cast<u8>(((h >> 5) & 63) * 255 / 63);
                    d[2] = static_cast<u8>((h & 31) * 255 / 31);
                } else {
                    d[0] = static_cast<u8>(((h >> 10) & 31) * 255 / 31);
                    d[1] = static_cast<u8>(((h >> 5) & 31) * 255 / 31);
                    d[2] = static_cast<u8>((h & 31) * 255 / 31);
                }
                d[3] = 255;
            } else {
                d[0] = d[1] = d[2] = p[0];
                d[3] = 255;
            }
        }
    }
    return true;
}

}  // namespace mv::pgraph

namespace mv::pgraph {

// ── the disk cache ───────────────────────────────────────────────────────────────────────────────────────────────

namespace {
// FNV-1a over the bytes, from a seed: two seeds give the file name and an independent check stored inside it.
u64 fnv1a(std::string_view bytes, u64 seed) {
    u64 h = seed;
    for (char c : bytes) {
        h ^= static_cast<u8>(c);
        h *= 0x100000001B3ull;
    }
    return h;
}
constexpr u32 kSpvFileMagic = 0x4353564Du;  // "MVSC"
constexpr u32 kSpvFileVersion = 1;
constexpr u64 kNameSeed = 0xCBF29CE484222325ull, kCheckSeed = 0x84222325CBF29CE4ull;

bool read_file(const std::filesystem::path& p, std::vector<char>& out) {
    std::FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) return false;
    out.clear();
    char buf[65536];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.insert(out.end(), buf, buf + n);
    std::fclose(f);
    return true;
}

// Written to a temporary name and renamed over the old one, so a run that dies mid-write leaves no torn file.
void write_file(const std::filesystem::path& p, const void* data, std::size_t len) {
    const std::filesystem::path tmp = p.string() + ".tmp";
    std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
    if (!f) return;
    const bool ok = std::fwrite(data, 1, len, f) == len;
    if (std::fclose(f) != 0 || !ok) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, p, ec);
}
}  // namespace

void VulkanRenderer::open_disk_cache(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir / "spirv", ec);
    if (ec) return;
    cache_dir_ = dir;
    std::vector<char> data;
    read_file(dir / "pipelines.bin", data);
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    // The driver checks the header (vendor, device, cache UUID) and starts empty on data from another driver.
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (fn_->vkCreatePipelineCache(dev_.device(), &ci, nullptr, &pipeline_cache_) != VK_SUCCESS) {
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        if (fn_->vkCreatePipelineCache(dev_.device(), &ci, nullptr, &pipeline_cache_) != VK_SUCCESS) pipeline_cache_ = VK_NULL_HANDLE;
    }
}

void VulkanRenderer::save_pipeline_cache() {
    if (!pipeline_cache_ || cache_dir_.empty()) return;
    std::size_t size = 0;
    if (fn_->vkGetPipelineCacheData(dev_.device(), pipeline_cache_, &size, nullptr) != VK_SUCCESS || size == 0) return;
    std::vector<char> data(size);
    if (fn_->vkGetPipelineCacheData(dev_.device(), pipeline_cache_, &size, data.data()) != VK_SUCCESS) return;
    write_file(cache_dir_ / "pipelines.bin", data.data(), size);
}

std::vector<u32> VulkanRenderer::spirv_for(bool vertex, const std::string& glsl, const char* name, std::string& log) {
    const std::string keyed = std::string(vertex ? "vs\n" : "fs\n") + glsl;
    std::filesystem::path file;
    const u64 check = fnv1a(keyed, kCheckSeed);
    if (!cache_dir_.empty()) {
        file = cache_dir_ / "spirv" / std::format("{:016x}.spv", fnv1a(keyed, kNameSeed));
        std::vector<char> data;
        if (read_file(file, data) && data.size() > 20 && data.size() % 4 == 0) {
            u32 head[4];
            std::memcpy(head, data.data(), sizeof head);
            u32 first = 0;
            std::memcpy(&first, data.data() + sizeof head, 4);
            if (head[0] == kSpvFileMagic && head[1] == kSpvFileVersion && (static_cast<u64>(head[3]) << 32 | head[2]) == check && first == 0x07230203u) {
                std::vector<u32> spirv((data.size() - sizeof head) / 4);
                std::memcpy(spirv.data(), data.data() + sizeof head, spirv.size() * 4);
                ++stats_.shaders_from_disk;
                return spirv;
            }
        }
    }
    CompiledShader c = compile_glsl(vertex ? ShaderStage::Vertex : ShaderStage::Fragment, glsl, name);
    log = c.log;
    if (c.ok() && !file.empty()) {
        std::vector<u32> out = {kSpvFileMagic, kSpvFileVersion, static_cast<u32>(check), static_cast<u32>(check >> 32)};
        out.insert(out.end(), c.spirv.begin(), c.spirv.end());
        write_file(file, out.data(), out.size() * 4);
    }
    return std::move(c.spirv);
}

}  // namespace mv::pgraph
