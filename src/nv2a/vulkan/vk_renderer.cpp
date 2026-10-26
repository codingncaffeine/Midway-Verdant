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
#include "nv2a/shaders/vsh_interpreter.h"
#include "nv2a/scanout.h"

namespace mv::pgraph {

namespace {

constexpr VkDeviceSize kVertexRingBytes = 32u << 20, kIndexRingBytes = 8u << 20, kUniformRingBytes = 8u << 20, kStagingBytes = 16u << 20;
constexpr u32 kMaxDescriptorSets = 4096;
constexpr u32 kZpassEnable = 0x17CC;  // Kelvin SET_ZPASS_PIXEL_COUNT_ENABLE
constexpr u32 kZminMaxControl = 0x1D78;  // Kelvin SET_ZMIN_MAX_CONTROL

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
    // MV_ASYNC_WRITEBACK=0: every write-back waits for the GPU where it is asked for (the control for the background ones).
    if (const char* a = std::getenv("MV_ASYNC_WRITEBACK"); a && *a == '0') async_wb_on_ = false;
    // MV_NATIVE_DXT=0: DXT textures decoded to RGBA8 on the CPU as before (the control).
    if (const char* n = std::getenv("MV_NATIVE_DXT"); n && *n == '0') native_dxt_requested_ = false;
    if (const char* w = std::getenv("MV_PENDING_WAIT_MS"); w && *w) pending_wait_per_vblank_ns_ = std::atoll(w) * 1'000'000;
    if (const char* d = std::getenv("MV_RENDER_DEBUG")) {
        const std::string s = d;
        debug_flat_ = s.find("flat") != std::string::npos;
        debug_nodepth_ = s.find("nodepth") != std::string::npos;
        debug_nocull_ = s.find("nocull") != std::string::npos;
        debug_noblend_ = s.find("noblend") != std::string::npos;
        debug_shadow_pass_ = s.find("shadow1") != std::string::npos;
        debug_show_v0_ = s.find("showv0") != std::string::npos;
        debug_skip_blend_ = s.find("skipblend") != std::string::npos;
        // drawid: every draw paints its number (the renderer's draw count, logged with the draw: red = bits 0-7,
        // green = 8-15, blue = 16-23) wherever it writes colour: a flat white shader blended by the constant colour.
        debug_draw_id_ = s.find("drawid") != std::string::npos;
        debug_flip_front_ = s.find("flipfront") != std::string::npos;
        // wball: every dirty target reaches RAM at each vblank (the control for write-back timing).
        debug_wb_all_ = s.find("wball") != std::string::npos;
        // ffdump: each new fixed-function shader's state words and lighting values, to stderr (the first 400).
        debug_ff_dump_ = s.find("ffdump") != std::string::npos;
        // roundtrip: a sampled render target goes through guest RAM (write-back, decode, upload) instead of being
        // sampled in place — the control for the in-place path.
        debug_roundtrip_ = s.find("roundtrip") != std::string::npos;
        // depthrows: a scaled depth buffer reaches RAM a row per copy region, thinned on the CPU, as before (the control for
        // the stored-size blit).
        debug_depth_rows_ = s.find("depthrows") != std::string::npos;
        // texlog: every texture decode to stderr (address, format, size, levels, faces, whether a target overlaps it).
        debug_tex_log_ = s.find("texlog") != std::string::npos;
        // nofog: every draw as if fog were off (the pixel side reads the same switch).
        debug_no_fog_ = s.find("nofog") != std::string::npos;
        debug_vsh_dump_ = s.find("vshdump") != std::string::npos;
        if (debug_draw_id_) debug_flat_ = true;
    }
    // MV_DRAWID_VBLANKS=FROM-TO: the drawid lines of those vblanks only, with the real shading (not flat).
    if (const char* w = std::getenv("MV_DRAWID_VBLANKS"); w && *w) {
        char* end = nullptr;
        drawid_from_ = std::strtoull(w, &end, 10);
        drawid_to_ = end && *end == '-' ? std::strtoull(end + 1, nullptr, 10) : drawid_from_;
        debug_draw_id_ = true;
    }
}

// A pipeline's create-info and everything it points at, kept together so a worker can create it after pipeline()
// returns (background compiles).
struct VulkanRenderer::PipelineBuild {
    VkPipelineShaderStageCreateInfo stages[2]{};
    VkVertexInputBindingDescription bindings[2]{};
    VkVertexInputAttributeDescription attrs[kAttributes]{};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cb{};
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dynamics[5]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                               VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    std::atomic<bool> done{false};
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkResult result{VK_SUCCESS};
};

VulkanRenderer::~VulkanRenderer() {
    stop_workers();  // a job in flight finishes first; queued ones are dropped
    if (!dev_.ready()) return;
    const auto& f = *fn_;
    for (auto& [k, b] : pipelines_pending_)
        if (b->done.load(std::memory_order_acquire) && b->pipeline) f.vkDestroyPipeline(dev_.device(), b->pipeline, nullptr);
    if (recording_) {
        end_pass();
        f.vkEndCommandBuffer(cb_);
        recording_ = false;
    }
    dev_.wait_queue_idle();  // the one queue the renderer uses (a frontend's device: its lock, not a device-wide wait)
    for (Handoff& h : handoffs_) {
        destroy_handoff(h);
        if (h.fence) f.vkDestroyFence(dev_.device(), h.fence, nullptr);
    }
    if (handoff_pool_) f.vkDestroyCommandPool(dev_.device(), handoff_pool_, nullptr);  // its command buffers with it
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
        dev_.destroy_image(s->zsnap);
        dev_.destroy_image(s->snap);
        dev_.destroy_image(s->zcolor);
        dev_.destroy_image(s->lo);
        dev_.destroy_image(s->zeta_lo);
        for (auto& [k2, c] : s->subs) {
            if (c.x8_view) f.vkDestroyImageView(dev_.device(), c.x8_view, nullptr);
            dev_.destroy_image(c.image);
        }
    }
    for (auto& [k, p] : passes_) f.vkDestroyRenderPass(dev_.device(), p, nullptr);
    for (auto& [k, m] : vs_cache_) f.vkDestroyShaderModule(dev_.device(), m, nullptr);
    for (auto& [k, m] : fs_cache_) f.vkDestroyShaderModule(dev_.device(), m, nullptr);
    for (auto& [k, p] : pipelines_) f.vkDestroyPipeline(dev_.device(), p, nullptr);
    for (auto& [k, p] : clear_pipelines_) f.vkDestroyPipeline(dev_.device(), p, nullptr);
    if (clear_vs_) f.vkDestroyShaderModule(dev_.device(), clear_vs_, nullptr);
    if (clear_fs_) f.vkDestroyShaderModule(dev_.device(), clear_fs_, nullptr);
    for (auto& [k, t] : textures_) dev_.destroy_image(t->image);
    for (auto& [k, m] : mips_) {
        if (m.x8_view) f.vkDestroyImageView(dev_.device(), m.x8_view, nullptr);
        dev_.destroy_image(m.image);
    }
    for (auto& [k, c] : cubes_) {
        if (c.x8_view) f.vkDestroyImageView(dev_.device(), c.x8_view, nullptr);
        dev_.destroy_image(c.image);
    }
    for (auto& [k, s] : samplers_) f.vkDestroySampler(dev_.device(), s, nullptr);
    for (auto& [k, s] : border_samplers_) f.vkDestroySampler(dev_.device(), s, nullptr);
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
    if (dc_pipeline_) f.vkDestroyPipeline(dev_.device(), dc_pipeline_, nullptr);
    if (dc_layout_) f.vkDestroyPipelineLayout(dev_.device(), dc_layout_, nullptr);
    if (dc_set_layout_) f.vkDestroyDescriptorSetLayout(dev_.device(), dc_set_layout_, nullptr);
    if (dc_pool_) f.vkDestroyDescriptorPool(dev_.device(), dc_pool_, nullptr);
    if (dc_sampler_) f.vkDestroySampler(dev_.device(), dc_sampler_, nullptr);
    dev_.destroy_buffer(dc_lut_);
    dev_.destroy_buffer(dc_ov_);
    dev_.destroy_buffer(zc_in_);
    dev_.destroy_buffer(zc_out_);
    if (set_layout_) f.vkDestroyDescriptorSetLayout(dev_.device(), set_layout_, nullptr);
    dev_.destroy_buffer(vertex_ring_.buffer);
    dev_.destroy_buffer(index_ring_.buffer);
    dev_.destroy_buffer(uniform_ring_.buffer);
    dev_.destroy_buffer(staging_.buffer);
    dev_.destroy_buffer(readback_);
    for (auto& c : captures_) destroy_capture(*c);
    for (Frame& fr : frames_) {
        if (fr.pool) f.vkDestroyDescriptorPool(dev_.device(), fr.pool, nullptr);
        for (VkFence fence : fr.fences) f.vkDestroyFence(dev_.device(), fence, nullptr);
        if (fr.occlusion) f.vkDestroyQueryPool(dev_.device(), fr.occlusion, nullptr);
        if (!fr.cbs.empty()) f.vkFreeCommandBuffers(dev_.device(), dev_.pool(), static_cast<u32>(fr.cbs.size()), fr.cbs.data());
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
    native_dxt_ = native_dxt_requested_ && dev_.bc_textures() && dev_.format_supports(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, kSampledFiltered) && dev_.format_supports(VK_FORMAT_BC2_UNORM_BLOCK, kSampledFiltered) &&
                  dev_.format_supports(VK_FORMAT_BC3_UNORM_BLOCK, kSampledFiltered);

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
    for (Frame& fr : frames_) {
        if (f.vkCreateDescriptorPool(dev_.device(), &dpi, nullptr, &fr.pool) != VK_SUCCESS) {
            error = "frame in flight: descriptor pool creation failed";
            return false;
        }
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_OCCLUSION;
        qpi.queryCount = kQueriesPerFrame;
        if (f.vkCreateQueryPool(dev_.device(), &qpi, nullptr, &fr.occlusion) != VK_SUCCESS) {
            error = "frame in flight: occlusion query pool creation failed";
            return false;
        }
        fr.aa_scale.assign(kQueriesPerFrame, 1);
        fr.res_scale.assign(kQueriesPerFrame, 1);
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
        f.vkWaitForFences(dev_.device(), 1, &fr.last, VK_TRUE, ~0ull);
        completed_ = fr.serial;
        ++stats_.waits;
    }
    for (vk::Image& i : fr.dead_images) dev_.destroy_image(i);
    for (VkImageView v : fr.dead_views) f.vkDestroyImageView(dev_.device(), v, nullptr);
    for (VkFramebuffer b : fr.dead_framebuffers) f.vkDestroyFramebuffer(dev_.device(), b, nullptr);
    fr.dead_images.clear();
    fr.dead_views.clear();
    fr.dead_framebuffers.clear();
    if (fr.used) f.vkResetFences(dev_.device(), fr.used, fr.fences.data());  // all signalled: the last covers the rest
    fr.used = 0;
    f.vkResetDescriptorPool(dev_.device(), fr.pool, 0);
    descriptor_sets_used_ = 0;
    for (Ring* r : {&vertex_ring_, &index_ring_, &uniform_ring_, &staging_}) r->begin = r->cursor = r->region * slot;
    last_vs_upload_.valid = last_ps_upload_.valid = false;  // they lie in another frame's region
    last_set_ = VK_NULL_HANDLE;                             // the pool was reset
    slot_ = slot;
    begin_commands();
    // The frame's queries: what its last use counted is read first (its fence has signalled), then the pools reset.
    harvest(fr);
    f.vkCmdResetQueryPool(cb_, fr.occlusion, 0, kQueriesPerFrame);  // always: counting may be switched on mid-frame
    fr.queries = 0;
    fr.first_ticket = next_ticket_;
    fr.harvested = false;
}

void VulkanRenderer::begin_commands() {
    const auto& f = *fn_;
    Frame& fr = frames_[slot_];
    if (fr.used == fr.cbs.size()) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = dev_.pool();
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        f.vkAllocateCommandBuffers(dev_.device(), &cai, &cb);
        f.vkCreateFence(dev_.device(), &fci, nullptr, &fence);
        fr.cbs.push_back(cb);
        fr.fences.push_back(fence);
    }
    cb_ = fr.cbs[fr.used];
    f.vkResetCommandBuffer(cb_, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    f.vkBeginCommandBuffer(cb_, &bi);
    bound_set_ = VK_NULL_HANDLE;  // binding is per command buffer
    recording_ = true;
}

// A finished frame's query results into counts_, by ticket (once).
void VulkanRenderer::harvest(Frame& fr) {
    if (fr.harvested) return;
    fr.harvested = true;
    if (!fr.queries) return;
    std::vector<u64> occlusion(fr.queries);
    fn_->vkGetQueryPoolResults(dev_.device(), fr.occlusion, 0, fr.queries, occlusion.size() * 8, occlusion.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    for (u32 q = 0; q < fr.queries; ++q) {
        // Pixels: the samples that passed, in the target's logical pixels (the GeForce 3's Z occlusion culling makes a
        // failed pixel cheap; a shaded-fragment count from the pipeline statistics query is not exact on every driver
        // and broke the core's determinism). Samples as stored, anti-aliasing included, for the z-pass reports (an
        // occlusion count is in samples on NVIDIA hardware [verify on a console]). The internal resolution's samples are
        // not the console's: both counts are divided by them.
        const u64 res = std::max<u32>(fr.res_scale[q], 1);
        counts_[fr.first_ticket + q] = {occlusion[q] / (std::max<u32>(fr.aa_scale[q], 1) * res), occlusion[q] / res};
    }
}

VulkanRenderer::DrawCounts VulkanRenderer::draw_counts(u64 ticket) {
    if (auto it = counts_.find(ticket); it != counts_.end()) {
        const DrawCounts c = it->second;
        counts_.erase(it);
        return c;
    }
    for (u32 i = 0; i < kFramesInFlight; ++i) {
        Frame& fr = frames_[i];
        if (fr.harvested || ticket < fr.first_ticket || ticket >= fr.first_ticket + fr.queries) continue;
        if (i == slot_ && recording_) submit();  // still being recorded: it goes to the GPU now
        if (fr.serial > completed_) {
            fn_->vkWaitForFences(dev_.device(), 1, &fr.last, VK_TRUE, ~0ull);
            completed_ = std::max(completed_, fr.serial);
            ++stats_.waits;
        }
        harvest(fr);
        return draw_counts(ticket);
    }
    return {};  // a ticket no frame holds (already read)
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
        pass_fb_ = VK_NULL_HANDLE;
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
// A draw's parts on one running clock: each lap adds the time since the last mark to its counter, and the scope's whole
// time goes to `total`. One clock read a part instead of a pair per part and per helper (two dozen reads a draw were a
// tenth of the GPU thread's time in a draw-heavy race).
struct LapClock {
    u64& total;
    std::chrono::steady_clock::time_point start, mark;
    explicit LapClock(u64& t) : total(t), start(std::chrono::steady_clock::now()), mark(start) {}
    ~LapClock() { total += ns_since(start); }
    void lap(u64& acc) {
        const auto now = std::chrono::steady_clock::now();
        acc += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - mark).count());
        mark = now;
    }
    void skip() { mark = std::chrono::steady_clock::now(); }
};
}  // namespace

void VulkanRenderer::submit(bool rotate) {
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
    fr.last = fr.fences[fr.used++];
    dev_.submit(1, &si, fr.last);
    fr.serial = ++submitted_;
    ++stats_.flushes;
    // Draw counting reads a frame's queries once its fence signals, all of them at once: a submit with queries moves on.
    if (rotate || rotate_pending_ || count_draws_ || fr.queries > 0) {
        rotate_pending_ = false;
        begin_frame((slot_ + 1) % kFramesInFlight);
    } else {
        begin_commands();
    }
}

void VulkanRenderer::finish() {
    const u32 last = slot_;
    submit();
    Timed timed(stats_.ns_flush);
    // A fence's signal covers every submission before its own on the queue, so the latest one is all it takes.
    Frame& fr = frames_[last];
    if (fr.serial > completed_) {
        fn_->vkWaitForFences(dev_.device(), 1, &fr.last, VK_TRUE, ~0ull);
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
    if (surfaces_.empty() && requested_scale_.load(std::memory_order_relaxed) != scale_) apply_scale();
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
    view_ = {0, 0, width, height};
    // A pitch-linear target inside an existing one of the same pitch, formats and anti-aliasing is a rectangle of its
    // image (one title drew 640x480, 320x240 and 184x169 at one address and a 320x120 band 240 rows down each frame:
    // as separate targets each was relaid out, written back and reloaded through RAM every time).
    if (!d.swizzled()) {
        const u32 pitch = d.color_pitch(), zeta = zeta_dma.base + d.zeta_offset;
        for (auto& [k, sp] : surfaces_) {
            Surface& r = *sp;
            if (r.swizzled || r.pitch != pitch || r.color_format != d.color_format() || r.zeta_format != d.zeta_format() || r.aa_x != aa_x || r.aa_y != aa_y) continue;
            if (phys < r.phys || phys >= static_cast<u64>(r.phys) + r.bytes()) continue;
            const u32 off = phys - r.phys, bpp = r.color_bytes();
            if ((off % pitch) % bpp) continue;
            const u32 x0 = (off % pitch) / bpp, y0 = off / pitch;
            if (x0 + width > r.width || y0 + height > r.height) continue;
            if (x0 == 0 && y0 == 0) {
                r.zeta_phys = zeta;  // at the base: the zeta it binds is its own (bind_depth follows a change)
                r.zeta_pitch = d.zeta_pitch();
            } else if (d.zeta_format() != 0 && zeta != 0 && (d.zeta_pitch() != r.zeta_pitch || zeta != r.zeta_phys + y0 * r.zeta_pitch + x0 * (r.zeta_format == 1 ? 2u : 4u))) {
                continue;  // its depth buffer is not the matching rectangle of the region's
            }
            r.format_word = d.format;
            view_ = {x0, y0, width, height};
            return &r;
        }
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
        erase_surface(it);
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
    t.sc = scale_for(cf, width, height);
    // Instrument: MV_DEBUG_NOSCALE=swizzled,aa,16bit keeps those kinds of target at 1x (which kind a scaled-only fault
    // follows).
    static const std::string noscale = [] { const char* e = std::getenv("MV_DEBUG_NOSCALE"); return std::string(e ? e : ""); }();
    if (!noscale.empty() && ((t.swizzled && noscale.find("swizzled") != std::string::npos) || ((t.aa_x > 1 || t.aa_y > 1) && noscale.find("aa") != std::string::npos) ||
                             (t.color_bytes() == 2 && noscale.find("16bit") != std::string::npos)))
        t.sc = 1;
    if (!dev_.create_image(cf, t.iw(), t.ih(), 1, 1, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT, false, t.color, error) ||
        !dev_.create_image(df, t.iw(), t.ih(), 1, 1, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
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
    // A target that another target's colour or depth covers starts from what that one rendered: written back first (the
    // memory is the truth; a target over another's depth buffer loaded stale RAM once depth read as colour stayed on
    // the GPU). A target whose colour it covers is finished with, as the reference emulator evicts the surfaces a new
    // one overlaps: the boot animation's targets lingered into the games that reused their memory.
    {
        const u64 b0 = t.phys, b1 = static_cast<u64>(t.phys) + t.bytes();
        for (auto o = surfaces_.begin(); o != surfaces_.end();) {
            Surface& other = *o->second;
            if (other.depth_holds && other.depth_holds < b1 && b0 < static_cast<u64>(other.depth_holds) + other.zeta_bytes()) release_depth(other);
            drop_parked(other, b0, b1, true);
            if (other.phys < b1 && b0 < static_cast<u64>(other.phys) + other.bytes()) {
                ++stats_.evicted;
                o = erase_surface(o);
            } else {
                ++o;
            }
        }
    }
    upload_color_from_ram(t);  // the guest may draw over an existing picture
    t.pass = render_pass(cf, df);
    VkImageView views[2] = {t.color.view, t.depth.view};
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = t.pass;
    fbi.attachmentCount = 2;
    fbi.pAttachments = views;
    fbi.width = t.iw();
    fbi.height = t.ih();
    fbi.layers = 1;
    fn_->vkCreateFramebuffer(dev_.device(), &fbi, nullptr, &t.framebuffer);
    handles_.watch_writes(t.phys, t.bytes());  // a CPU write to its memory reloads it (cpu_wrote)
    Surface* raw = surf.get();
    surfaces_[phys] = std::move(surf);
    return raw;
}

// The colour image takes what RAM holds: at the target's creation, and again before it is drawn into once another
// engine has rewritten its memory (one title resets a glare buffer each frame with a 2D blit, then draws into it
// additively; drawing on over the GPU's stale copy saturated it to white).
void VulkanRenderer::upload_color_from_ram(Surface& t) {
    complete_write_backs_over(t.phys, static_cast<u64>(t.phys) + t.bytes());  // RAM must hold a copy still in flight
    if (debug_tex_log_) std::fprintf(stderr, "texlog: reload colour %08x %ux%u frame %llu\n", t.phys, t.width, t.height, static_cast<unsigned long long>(frame_index_));
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
    // A scaled target loads its stored-size copy, then widens it (NEAREST: each stored pixel keeps its value).
    const bool scaled = t.sc > 1;
    if ((!scaled || ensure_lo(t)) && staging_alloc(pixels.size(), offset)) {
        std::memcpy(static_cast<u8*>(staging_.buffer.mapped) + offset, pixels.data(), pixels.size());
        VkBufferImageCopy region{};
        region.bufferOffset = offset;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {t.width, t.height, 1};
        fn_->vkCmdCopyBufferToImage(cb_, staging_.buffer.buffer, scaled ? t.lo.image : t.color.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        upload_barrier();
        if (scaled) blit_lo(t, false);
    }
    remember_pages(t);
}

// A watch the FIFO thread asks for is applied at the CPU thread's next exit, and the log reports every page it shows
// written then, including writes from before the target existed, whose bytes the image already took from RAM (a title
// copied the last boot frame into its new buffers, cleared them on the GPU, and the late report put the copy back).
void VulkanRenderer::remember_pages(Surface& s) {
    constexpr u32 kPage = 4096;
    const u32 first = s.phys / kPage, last = static_cast<u32>((static_cast<u64>(s.phys) + s.bytes() - 1) / kPage);
    s.page_hashes.resize(last - first + 1);
    for (u32 p = first; p <= last; ++p) s.page_hashes[p - first] = texture_content(p * kPage, kPage, 0);
}

bool VulkanRenderer::ensure_pass(Surface& t, bool writes_color) {
    if (t.borrow && t.borrow->depth.image != t.borrow_image) end_borrow(t);  // the lender bound another image since
    const VkFramebuffer fb = t.borrow ? t.borrow_fb : t.framebuffer;
    if (pass_open_ && pass_target_ == &t && pass_fb_ == fb && !(writes_color && t.color_ram_newer)) return true;
    end_pass();
    if (t.color_ram_newer && writes_color) {
        upload_color_from_ram(t);
        t.color_ram_newer = false;
        ++stats_.surface_reloads;
    }
    if (sampled_since_barrier_) surface_barrier();  // draws that sampled a target finish before it is drawn into again
    t.last_rendered_frame = frame_index_;
    t.last_used_vblank = vblank_index_;
    if (!t.color_ram_newer) t.ram_newer = false;  // a colour image still to reload is not what RAM holds
    attachments_written_ = true;
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = t.pass;
    rbi.framebuffer = fb;
    rbi.renderArea = {{0, 0}, {t.iw(), t.ih()}};
    fn_->vkCmdBeginRenderPass(cb_, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    pass_open_ = true;
    pass_target_ = &t;
    pass_fb_ = fb;
    if (t.borrow) t.borrow->last_used_vblank = vblank_index_;
    // Whether the pass writes colour or depth is the draw's or the clear's to say (the colour mask, the depth and stencil
    // state, the clear bits), so a pass that never touches one never lands it in guest RAM.
    t.rendered_this_frame = true;
    return true;
}

// The depth buffer at the target's zeta address, in its depth image before a pass: on the hardware the depth buffer is
// memory, shared by every colour target that binds it; here each target has a depth image, so the contents follow the
// address. The target that last drew with it owns them; another target binding it takes them (a GPU copy when the
// shapes match, else through RAM), and a target whose image held another address hands that back to RAM first. Two
// targets sharing one depth buffer each kept their own copy, and whichever reached RAM last won.
void VulkanRenderer::bind_depth(Surface& t, bool cleared) {
    const u32 z = t.zeta_phys;
    if (t.zeta_format == 0 || z == 0) return;
    const auto found = depth_owner_.find(z);
    Surface* owner = found == depth_owner_.end() ? nullptr : found->second;
    if (owner && owner != &t && borrow_depth(t, *owner)) return;
    if (t.borrow) end_borrow(t);
    if (owner == &t && t.depth_holds == z) return;
    end_pass();
    // Another target holds the address in a parked image: back to RAM with it, and this one loads it from there.
    if (owner && owner != &t && owner->depth_holds != z && drop_parked(*owner, z, static_cast<u64>(z) + 1, true)) owner = nullptr;
    // This target parked the address's image earlier, or binds a new one while its image holds another address: the
    // image parks (D35) and the address's own comes back, or a spare takes the new address.
    if (park_for(t, z)) {
        depth_owner_[z] = &t;
        t.holds_pitch = t.zeta_pitch;
        handles_.watch_writes(z, t.zeta_bytes());
        return;
    }
    if (t.depth_holds && t.depth_holds != z) {
        // The image held another address's depth: it goes back to RAM when newer there.
        const auto o = depth_owner_.find(t.depth_holds);
        if (o != depth_owner_.end() && o->second == &t) {
            wb_why_ = Stats::kWbRange;
            write_back_held(t);
            wb_why_ = Stats::kWbAll;
            depth_owner_.erase(o);
        }
        t.depth_holds = 0;
        t.zeta_dirty = false;
    }
    // Cleared whole before anything reads it: what the address held is dead. The owner's contents need not reach RAM
    // when the clear covers every byte of them (the hardware's clear overwrites that memory), and nothing is loaded.
    if (cleared && owner && owner != &t && owner->depth_holds == z) {
        const u64 held = owner->swizzled ? static_cast<u64>(owner->zeta_format == 1 ? 2u : 4u) * owner->width * owner->height : static_cast<u64>(owner->holds_pitch) * owner->height;
        if (held <= t.zeta_bytes()) {
            owner->zeta_dirty = false;
        } else {
            wb_why_ = Stats::kWbRange;
            write_back_held(*owner);
            wb_why_ = Stats::kWbAll;
        }
        owner->depth_holds = 0;
        ++stats_.depth_handovers;
        owner = nullptr;
    }
    if (cleared && (!owner || owner == &t)) {
        ++stats_.depth_loads_skipped;
    } else if (owner && owner != &t && owner->depth_holds == z && owner->width == t.width && owner->height == t.height && owner->sc == t.sc && owner->zeta_format == t.zeta_format) {
        const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (t.zeta_format == 1 ? 0u : static_cast<u32>(VK_IMAGE_ASPECT_STENCIL_BIT));
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                                  &before, 0, nullptr, 0, nullptr);
        VkImageCopy region{};
        region.srcSubresource = {aspects, 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.extent = {t.iw(), t.ih(), 1};
        fn_->vkCmdCopyImage(cb_, owner->depth.image, VK_IMAGE_LAYOUT_GENERAL, t.depth.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        after.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &after,
                                  0, nullptr, 0, nullptr);
        t.zeta_dirty = owner->zeta_dirty;
        owner->zeta_dirty = false;
        owner->depth_holds = 0;
        ++t.draw_serial;  // a depth-as-colour conversion of it is stale
        ++stats_.depth_handovers;
    } else {
        if (owner && owner != &t) {
            if (owner->zeta_dirty && owner->depth_holds == z) {
                wb_why_ = Stats::kWbRange;
                write_back_held(*owner);  // the owner may bind another address by now
                wb_why_ = Stats::kWbAll;
            }
            owner->depth_holds = 0;
            ++stats_.depth_handovers;
        }
        upload_depth_from_ram(t);
    }
    depth_owner_[z] = &t;
    t.depth_holds = z;
    t.holds_pitch = t.zeta_pitch;
    handles_.watch_writes(z, t.zeta_bytes());  // a CPU write to the depth buffer's memory reloads it
}

// D37: the owner's bound image holds t's zeta address at t's pitch, in the same format and scale, and is at least t's
// size: t's passes draw into the image's top-left rectangle, which is the memory t's buffer covers. The owner keeps the
// address (its write-backs, parking and sampling see t's depth writes); nothing is copied or loaded.
bool VulkanRenderer::borrow_depth(Surface& t, Surface& owner) {
    if (owner.depth_holds != t.zeta_phys || owner.swizzled || t.swizzled || owner.holds_pitch != t.zeta_pitch || owner.zeta_format != t.zeta_format || owner.sc != t.sc ||
        owner.width < t.width || owner.height < t.height)
        return false;
    if (t.borrow == &owner && t.borrow_image == owner.depth.image) return true;
    if (t.borrow) end_borrow(t);
    VkImageView views[2] = {t.color.view, owner.depth.view};
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = t.pass;
    fbi.attachmentCount = 2;
    fbi.pAttachments = views;
    fbi.width = t.iw();  // the depth image may be bigger: a framebuffer's attachments only have to cover it
    fbi.height = t.ih();
    fbi.layers = 1;
    if (fn_->vkCreateFramebuffer(dev_.device(), &fbi, nullptr, &t.borrow_fb) != VK_SUCCESS) {
        t.borrow_fb = VK_NULL_HANDLE;
        return false;
    }
    if (debug_tex_log_)
        std::fprintf(stderr, "texlog: depth %08x of target %08x %ux%u lent to target %08x %ux%u frame %llu\n", t.zeta_phys, owner.phys, owner.width, owner.height, t.phys, t.width, t.height,
                     static_cast<unsigned long long>(frame_index_));
    t.borrow = &owner;
    t.borrow_image = owner.depth.image;
    ++stats_.depth_borrows;
    return true;
}

void VulkanRenderer::end_borrow(Surface& t) {
    if (pass_open_ && pass_target_ == &t && pass_fb_ == t.borrow_fb) end_pass();
    retire(t.borrow_fb);
    t.borrow = nullptr;
    t.borrow_image = VK_NULL_HANDLE;
}

void VulkanRenderer::end_borrows_from(Surface& owner) {
    for (auto& [k, sp] : surfaces_)
        if (sp->borrow == &owner) end_borrow(*sp);
}

// The bound depth image's contents to the address it holds, which the target may no longer bind, when newer there.
void VulkanRenderer::write_back_held(Surface& s) {
    if (!s.zeta_dirty || !s.depth_holds) return;
    complete_write_backs_over(s.depth_holds, static_cast<u64>(s.depth_holds) + s.held_bytes());  // an older copy lands first
    if (debug_tex_log_)
        std::fprintf(stderr, "texlog: held depth %08x of target %08x (binds %08x) to RAM, cause %u\n", s.depth_holds, s.phys, s.zeta_phys, static_cast<unsigned>(wb_why_));
    finish_write_backs();  // a queued write-back of this target is stored with its own address
    const u32 bound = s.zeta_phys, pitch = s.zeta_pitch;
    s.zeta_phys = s.depth_holds;
    if (s.holds_pitch) s.zeta_pitch = s.holds_pitch;
    write_back_zeta(s);
    s.zeta_phys = bound;
    s.zeta_pitch = pitch;
}

// ── parked depth images (D35) ────────────────────────────────────────────────────────────────────────────────────

void VulkanRenderer::swap_depth(Surface& s, Surface::ParkedDepth& p) {
    if (pass_open_ && pass_target_ == &s) end_pass();  // the open pass draws into the image bound now
    std::swap(s.depth, p.depth);
    std::swap(s.depth_view, p.depth_view);
    std::swap(s.depth_tex, p.depth_tex);
    std::swap(s.framebuffer, p.framebuffer);
    std::swap(s.zeta_dirty, p.zeta_dirty);
    std::swap(s.depth_holds, p.depth_holds);
    std::swap(s.holds_pitch, p.holds_pitch);
    ++s.draw_serial;  // a depth-as-colour conversion was of the other image
    stage_cache_.fill({});  // a stage may point at an entry that moved
}

// Binding z: true when the target's parked image for z is bound again (its contents current). Otherwise, when the
// bound image holds another address the target owns, that image parks and a spare is bound, holding nothing; the
// caller fills it (a handover or a load from RAM). False with the target as it was when nothing could park.
bool VulkanRenderer::park_for(Surface& t, u32 z) {
    const bool owns_bound = t.depth_holds && t.depth_holds != z && [&] {
        const auto o = depth_owner_.find(t.depth_holds);
        return o != depth_owner_.end() && o->second == &t;
    }();
    const auto found = depth_owner_.find(z);
    const bool owns_z = found != depth_owner_.end() && found->second == &t;
    if (owns_z) {
        for (Surface::ParkedDepth& p : t.parked) {
            if (p.depth_holds != z) continue;
            // The bound image parks in its place when it holds an address of this target's; else that slot is a spare.
            if (!owns_bound) {
                t.zeta_dirty = false;
                t.depth_holds = 0;
            }
            swap_depth(t, p);
            p.used = ++park_clock_;
            ++stats_.depth_unparks;
            return true;
        }
    }
    if (!owns_bound) return false;
    Surface::ParkedDepth* spare = nullptr;
    for (Surface::ParkedDepth& p : t.parked)
        if (p.depth_holds == 0) spare = &p;
    if (!spare && t.parked.size() < kMaxParked) {
        Surface::ParkedDepth p;
        if (spare_depth(t, p)) {
            t.parked.push_back(std::move(p));
            stage_cache_.fill({});  // the list moved
            spare = &t.parked.back();
        }
    }
    if (!spare) {
        // Full: the least recently used parked image goes back to RAM, and its image is the spare.
        Surface::ParkedDepth* lru = nullptr;
        for (Surface::ParkedDepth& p : t.parked)
            if (!lru || p.used < lru->used) lru = &p;
        if (!lru) return false;
        write_back_parked(t, *lru);
        const auto o = depth_owner_.find(lru->depth_holds);
        if (o != depth_owner_.end() && o->second == &t) depth_owner_.erase(o);
        lru->depth_holds = 0;
        lru->zeta_dirty = false;
        spare = lru;
    }
    swap_depth(t, *spare);  // the bound image (owned, at its address) parks; the spare is bound, holding nothing
    spare->used = ++park_clock_;
    return false;
}

// A depth image of the target's shape and format, with a framebuffer pairing it with the colour image.
bool VulkanRenderer::spare_depth(Surface& t, Surface::ParkedDepth& p) {
    std::string error;
    const VkFormat df = t.zeta_format == 1 ? VK_FORMAT_D16_UNORM : depth24_;
    if (!dev_.create_image(df, t.iw(), t.ih(), 1, 1, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               ((t.zeta_format == 1 ? depth16_sampled_ : depth24_sampled_) ? static_cast<u32>(VK_IMAGE_USAGE_SAMPLED_BIT) : 0u),
                           VK_IMAGE_ASPECT_DEPTH_BIT | (t.zeta_format == 1 ? 0u : static_cast<u32>(VK_IMAGE_ASPECT_STENCIL_BIT)), false, p.depth, error)) {
        note("parked depth image: " + error);
        return false;
    }
    end_pass();
    fresh_image(p.depth);
    VkImageView views[2] = {t.color.view, p.depth.view};
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = t.pass;
    fbi.attachmentCount = 2;
    fbi.pAttachments = views;
    fbi.width = t.iw();
    fbi.height = t.ih();
    fbi.layers = 1;
    if (fn_->vkCreateFramebuffer(dev_.device(), &fbi, nullptr, &p.framebuffer) != VK_SUCCESS) {
        retire(p.depth);
        return false;
    }
    return true;
}

// A parked image's contents to RAM when newer there (it stays parked, clean).
void VulkanRenderer::write_back_parked(Surface& s, Surface::ParkedDepth& p) {
    if (!p.zeta_dirty || !p.depth_holds) return;
    complete_write_backs_over(p.depth_holds, static_cast<u64>(p.depth_holds) + s.parked_bytes(p));  // an older copy lands first
    finish_write_backs();  // a queued write-back of the bound image is stored with the target's own address
    swap_depth(s, p);
    write_back_held(s);
    swap_depth(s, p);
}

void VulkanRenderer::flush_parked(u64 b0, u64 b1) {
    for (auto& [k, sp] : surfaces_)
        for (Surface::ParkedDepth& p : sp->parked)
            if (p.depth_holds && p.zeta_dirty && p.depth_holds < b1 && b0 < static_cast<u64>(p.depth_holds) + sp->parked_bytes(p)) write_back_parked(*sp, p);
}

// The parked images of s over [b0, b1) go: their contents to RAM first when `write` and newer there, their ownership
// dropped; each stays as a spare. True when one did.
bool VulkanRenderer::drop_parked(Surface& s, u64 b0, u64 b1, bool write) {
    bool any = false;
    for (Surface::ParkedDepth& p : s.parked) {
        if (!p.depth_holds || p.depth_holds >= b1 || b0 >= static_cast<u64>(p.depth_holds) + s.parked_bytes(p)) continue;
        if (write) write_back_parked(s, p);
        const auto o = depth_owner_.find(p.depth_holds);
        if (o != depth_owner_.end() && o->second == &s) depth_owner_.erase(o);
        p.depth_holds = 0;
        p.zeta_dirty = false;
        any = true;
    }
    return any;
}

void VulkanRenderer::drop_parked(u64 b0, u64 b1) {
    for (auto& [k, sp] : surfaces_) drop_parked(*sp, b0, b1, true);
}

// A texture at a parked image's address, sampled in place: the image holds what the address holds.
VulkanRenderer::TextureEntry* VulkanRenderer::parked_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear, bool depth16) {
    if (s.width != w || s.height != h || (s.zeta_format == 1) != depth16 || !(depth16 ? depth16_sampled_ : depth24_sampled_) || linear == s.swizzled) return nullptr;
    for (Surface::ParkedDepth& p : s.parked) {
        if (p.depth_holds != phys || (linear && ts.pitch() != p.holds_pitch)) continue;
        TextureEntry& e = p.depth_tex;
        if (!e.valid) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.image = p.depth.image;
            vci.format = p.depth.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            if (fn_->vkCreateImageView(dev_.device(), &vci, nullptr, &p.depth_view) != VK_SUCCESS) return nullptr;
            e.image = p.depth;
            e.image.view = p.depth_view;
            e.image.levels = 1;
            e.width = w;
            e.height = h;
            e.surface = true;
            e.valid = true;
        }
        e.phys = phys;
        e.linear = linear;
        e.length = texture_size_in_memory(ts);
        if (attachments_written_) surface_barrier();
        sampled_since_barrier_ = true;
        ++stats_.surface_samples;
        ++stats_.parked_samples;
        s.last_used_vblank = vblank_index_;
        p.used = ++park_clock_;
        return &e;
    }
    return nullptr;
}

// The depth image takes what RAM holds at the zeta address: Z16 words, or Z24S8 words (the depth in bits 31-8 above
// the stencil byte) split into the depth aspect's 24-bit words (or floats) and the stencil aspect's bytes.
void VulkanRenderer::upload_depth_from_ram(Surface& t) {
    complete_write_backs_over(t.zeta_phys, static_cast<u64>(t.zeta_phys) + t.zeta_bytes());
    if (debug_tex_log_)
        std::fprintf(stderr, "texlog: load depth %08x for target %08x %ux%u frame %llu\n", t.zeta_phys, t.phys, t.width, t.height, static_cast<unsigned long long>(frame_index_));
    ++t.draw_serial;
    const bool z16 = t.zeta_format == 1, float_depth = depth24_ == VK_FORMAT_D32_SFLOAT_S8_UINT;
    const u32 bpp = z16 ? 2u : 4u;
    if (static_cast<u64>(t.zeta_phys) + t.zeta_bytes() > ram_size_) return;
    const std::size_t n = static_cast<std::size_t>(t.width) * t.height;
    std::vector<u8> pixels(z16 ? n * 2 : n * 5, 0);
    const u8* base = ram_ + t.zeta_phys;
    for (u32 y = 0; y < t.height; ++y) {
        for (u32 x = 0; x < t.width; ++x) {
            const std::size_t off = t.swizzled ? static_cast<std::size_t>(swizzled_offset(x, y, 0, t.log_w, t.log_h, 0)) * bpp : static_cast<std::size_t>(y) * t.zeta_pitch + x * bpp;
            const std::size_t i = static_cast<std::size_t>(y) * t.width + x;
            if (z16) {
                std::memcpy(&pixels[i * 2], base + off, 2);
                continue;
            }
            u32 word;
            std::memcpy(&word, base + off, 4);
            const u32 depth = word >> 8;
            if (float_depth) {
                const float f = static_cast<float>(depth) / 16777215.0f;
                std::memcpy(&pixels[i * 4], &f, 4);
            } else {
                std::memcpy(&pixels[i * 4], &depth, 4);
            }
            pixels[n * 4 + i] = static_cast<u8>(word);
        }
    }
    if (t.sc > 1) {
        // A scaled target: each stored row widened sc times on the CPU, and the copy reads it into sc image rows (depth
        // cannot be blitted on every device; widening whole blocks on the CPU was 55 MB a load at 6x, ~1 a frame in the
        // boot animation). Bands of stored rows fit the staging ring; row strides stay multiples of four bytes.
        const u32 sc = t.sc, rw = t.iw(), dsz = z16 ? 2u : 4u;
        const VkDeviceSize drow = (static_cast<VkDeviceSize>(rw) * dsz + 3) / 4 * 4, srow = z16 ? 0 : (static_cast<VkDeviceSize>(rw) + 3) / 4 * 4;
        const u32 band = static_cast<u32>(std::max<VkDeviceSize>(1, (VkDeviceSize{4} << 20) / (drow + srow)));
        std::vector<VkBufferImageCopy> regions;
        for (u32 r0 = 0; r0 < t.height; r0 += band) {
            const u32 rows = std::min(band, t.height - r0);
            VkDeviceSize offset = 0;
            if (!staging_alloc(rows * (drow + srow), offset)) return;
            u8* dst = static_cast<u8*>(staging_.buffer.mapped) + offset;
            regions.clear();
            for (u32 r = 0; r < rows; ++r) {
                const std::size_t src_row = static_cast<std::size_t>(r0 + r) * t.width;
                u8* drow_p = dst + r * drow;
                u8* srow_p = dst + rows * drow + r * srow;
                for (u32 x = 0; x < t.width; ++x) {
                    const std::size_t i = src_row + x;
                    for (u32 k = 0; k < sc; ++k) std::memcpy(drow_p + (static_cast<std::size_t>(x) * sc + k) * dsz, &pixels[i * dsz], dsz);
                    if (!z16) std::memset(srow_p + static_cast<std::size_t>(x) * sc, pixels[n * 4 + i], sc);
                }
                for (u32 k = 0; k < sc; ++k) {
                    VkBufferImageCopy c{};
                    c.bufferOffset = offset + r * drow;
                    c.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
                    c.imageOffset = {0, static_cast<i32>((r0 + r) * sc + k), 0};
                    c.imageExtent = {rw, 1, 1};
                    regions.push_back(c);
                    if (z16) continue;
                    c.bufferOffset = offset + rows * drow + r * srow;
                    c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
                    regions.push_back(c);
                }
            }
            fn_->vkCmdCopyBufferToImage(cb_, staging_.buffer.buffer, t.depth.image, VK_IMAGE_LAYOUT_GENERAL, static_cast<u32>(regions.size()), regions.data());
        }
        upload_barrier();
        ++stats_.depth_uploads;
        return;
    }
    VkDeviceSize offset = 0;
    if (!staging_alloc(pixels.size(), offset)) return;
    std::memcpy(static_cast<u8*>(staging_.buffer.mapped) + offset, pixels.data(), pixels.size());
    VkBufferImageCopy regions[2]{};
    regions[0].bufferOffset = offset;
    regions[0].imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    regions[0].imageExtent = {t.width, t.height, 1};
    regions[1] = regions[0];
    regions[1].bufferOffset = offset + n * 4;
    regions[1].imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    fn_->vkCmdCopyBufferToImage(cb_, staging_.buffer.buffer, t.depth.image, VK_IMAGE_LAYOUT_GENERAL, z16 ? 1 : 2, regions);
    upload_barrier();
    ++stats_.depth_uploads;
}

void VulkanRenderer::release_depth(Surface& s) {
    if (!s.depth_holds) return;
    const auto o = depth_owner_.find(s.depth_holds);
    if (o != depth_owner_.end() && o->second == &s) {
        wb_why_ = Stats::kWbRelayout;
        write_back_held(s);
        wb_why_ = Stats::kWbAll;
        depth_owner_.erase(o);
    }
    s.depth_holds = 0;
}

std::unordered_map<u32, std::unique_ptr<VulkanRenderer::Surface>>::iterator VulkanRenderer::erase_surface(std::unordered_map<u32, std::unique_ptr<Surface>>::iterator it) {
    Surface& s0 = *it->second;
    complete_write_backs(true);  // a copy in flight may name this target
    if (pass_target_ == &s0) end_pass();
    end_borrows_from(s0);  // before its depth image retires
    if (s0.borrow) end_borrow(s0);
    wb_why_ = Stats::kWbRelayout;
    if (s0.dirty) write_back(s0);
    release_depth(s0);
    drop_parked(s0, 0, ~0ull, true);
    wb_why_ = Stats::kWbAll;
    for (Surface::ParkedDepth& p : s0.parked) {
        retire(p.depth_view);
        retire(p.framebuffer);
        retire(p.depth);
    }
    s0.parked.clear();
    for (auto o = depth_owner_.begin(); o != depth_owner_.end();) o = o->second == &s0 ? depth_owner_.erase(o) : std::next(o);
    if (draw_target_ == &s0) draw_target_ = nullptr;
    // Draws already recorded or still on the GPU may use the old images: they go once those are done.
    retire(s0.x8_view);
    retire(s0.depth_view);
    retire(s0.snap_x8_view);
    retire(s0.zcolor_x8_view);
    retire(s0.zcolor_abgr_view);
    retire(s0.zsnap_view);
    destroy_surface_views(s0);
    retire(s0.framebuffer);
    retire(s0.color);
    retire(s0.depth);
    retire(s0.snap);
    retire(s0.zcolor);
    retire(s0.zsnap);
    retire(s0.lo);
    retire(s0.zeta_lo);
    retire_subs(s0);
    for (auto& [k, m] : mips_)  // nor the level these came from
        for (const void*& level : m.levels)
            if (level == &s0) level = nullptr;
    for (auto& [k, c] : cubes_)  // a later target at the same place is not the face these copies came from
        for (const void*& face : c.faces)
            if (face == &s0) face = nullptr;
    stage_cache_.fill({});  // a stage may point at the erased target's entries
    return surfaces_.erase(it);
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
    if (s.sc > 1) {
        // A scaled target reaches RAM through its stored-size copy.
        if (!ensure_lo(s)) return;
        blit_lo(s, true);
        read_back(s.lo, VK_IMAGE_ASPECT_COLOR_BIT, s.width, s.height, offset);
    } else {
        read_back(s.color, VK_IMAGE_ASPECT_COLOR_BIT, s.width, s.height, offset);
    }
    ++stats_.writebacks_by[wb_why_];
    write_backs_.push_back({&s, false, offset});
    write_backs_.back().serial = s.draw_serial;
}

void VulkanRenderer::queue_write_back_zeta(Surface& s) {
    if (!s.zeta_dirty) return;
    s.zeta_dirty = false;
    if (s.zeta_format == 0 || s.zeta_phys == 0) return;  // no depth surface bound (an offset of zero is never a real one)
    if (s.depth_holds != s.zeta_phys) return;  // its image holds another address's depth, or none (bind_depth)
    Timed timed(stats_.ns_write_back);
    const bool z16 = s.zeta_format == 1;
    // D16; a 24-bit or float depth copies as a 32-bit word, followed by the stencil bytes. A scaled target's copy is
    // its stored-size copy (zeta_lo, blitted down), or where the device cannot blit the format, every sc-th row whole
    // (zeta_rows), thinned to its stored size when stored.
    if (static_cast<u64>(s.zeta_phys) + s.zeta_bytes() > ram_size_) return;
    const VkImageAspectFlags aspect = z16 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    const bool rows = s.sc > 1 && !ensure_zeta_lo(s);
    const std::size_t bytes = rows ? zeta_rows_bytes(s) : static_cast<std::size_t>(s.width) * s.height * (z16 ? 2u : 5u);
    if (rows && bytes > readback_.size) {
        write_back_zeta_banded(s);
        return;
    }
    if (bytes > readback_.size) {
        note("depth surface too large to write back");
        return;
    }
    VkDeviceSize offset = 0;
    readback_alloc(bytes, offset);  // fits: checked above
    if (rows) {
        read_back_zeta_rows(s, offset, 0, s.height);
    } else if (s.sc > 1) {
        blit_zeta_lo(s);
        read_back(s.zeta_lo, aspect, s.width, s.height, offset);
    } else {
        read_back(s.depth, aspect, s.width, s.height, offset);
    }
    ++stats_.writebacks_by[wb_why_];
    write_backs_.push_back({&s, true, offset});
    write_backs_.back().rows = rows;
    write_backs_.back().serial = s.draw_serial;
}

// Each queued copy into guest RAM, in the order queued (the order the surfaces were written back one at a time before).
void VulkanRenderer::store_write_backs(const std::vector<PendingWriteBack>& list) {
    const u8* base = static_cast<const u8*>(readback_.mapped);
    for (const PendingWriteBack& w : list) {
        Surface& s = *w.surface;
        const auto t0 = std::chrono::steady_clock::now();
        if (w.zeta && w.rows) {
            store_zeta(s, thin_zeta_rows(s, base + w.offset).data());
        } else if (w.zeta) {
            store_zeta(s, base + w.offset);
        } else {
            // Drawn into since it was queued (a write-back that waited in the background): RAM now holds the older picture
            // and the target stays dirty for the next one.
            const bool drawn_since = s.draw_serial != w.serial;
            store_color(s, base + w.offset);
            if (drawn_since) s.dirty = true;
        }
        auto& cost = stats_.store_cost[std::format("{} of {:08x} {}x{} x{} {} fmt {:x}", w.zeta ? "depth" : "colour", w.zeta ? s.zeta_phys : s.phys, s.width, s.height, s.sc, s.swizzled ? "swizzled" : "linear", w.zeta ? s.zeta_format : s.color_format)];
        ++cost.first;
        cost.second += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }
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
    // A batch still in flight is older: it lands first (the wait above covered it).
    complete_write_backs(true);
    store_write_backs(write_backs_);
    write_backs_.clear();
    readback_cursor_ = 0;
}

void VulkanRenderer::finish_write_backs_async() {
    if (write_backs_.empty()) return;
    if (!async_wb_on_) {
        finish_write_backs();
        return;
    }
    // One batch in flight at a time: an older one lands first (normally long done: it was queued a vblank ago). This
    // batch stays queued meanwhile, so the read-back buffer it occupies is not handed out again.
    complete_write_backs(true);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    const u32 slot = slot_;
    submit();
    async_fence_ = frames_[slot].last;
    async_serial_ = frames_[slot].serial;
    async_wb_.swap(write_backs_);
    write_backs_.clear();
    ++stats_.background_write_back_batches;
    async_pending_.store(true, std::memory_order_release);
}

bool VulkanRenderer::write_backs_ready() {
    if (async_wb_.empty()) return false;
    if (completed_ >= async_serial_) return true;
    if (fn_->vkGetFenceStatus(dev_.device(), async_fence_) != VK_SUCCESS) return false;
    completed_ = std::max(completed_, async_serial_);
    return true;
}

bool VulkanRenderer::complete_write_backs(bool wait, const std::vector<u32>* cpu_pages) {
    if (async_wb_.empty()) return true;
    if (!write_backs_ready()) {
        if (!wait) return false;
        Timed timed_gpu(stats_.ns_write_back_gpu);
        // A fence is reset only after the renderer has seen it signal (begin_frame), so it is still this submission's.
        fn_->vkWaitForFences(dev_.device(), 1, &async_fence_, VK_TRUE, ~0ull);
        completed_ = std::max(completed_, async_serial_);
        ++stats_.waits;
        ++stats_.background_write_back_waits;
    }
    Timed timed(stats_.ns_write_back);
    // Pages the CPU or a device wrote since the copies were queued keep what was written: the copies hold older pixels.
    constexpr u32 kPage = 4096;
    std::vector<std::pair<u32, std::vector<u8>>> kept;
    if (cpu_pages && !cpu_pages->empty())
        for (const PendingWriteBack& w : async_wb_) {
            const Surface& s = *w.surface;
            const u64 a0 = w.zeta ? s.zeta_phys : s.phys, a1 = a0 + (w.zeta ? s.zeta_bytes() : s.bytes());
            for (auto p = std::lower_bound(cpu_pages->begin(), cpu_pages->end(), static_cast<u32>(a0 / kPage)); p != cpu_pages->end() && static_cast<u64>(*p) * kPage < a1; ++p) {
                if (static_cast<u64>(*p + 1) * kPage > ram_size_) break;
                if (std::any_of(kept.begin(), kept.end(), [&](const auto& k) { return k.first == *p; })) continue;
                kept.emplace_back(*p, std::vector<u8>(ram_ + static_cast<std::size_t>(*p) * kPage, ram_ + static_cast<std::size_t>(*p + 1) * kPage));
            }
        }
    std::vector<PendingWriteBack> list;
    list.swap(async_wb_);
    store_write_backs(list);
    for (const auto& [page, bytes] : kept) std::memcpy(ram_ + static_cast<std::size_t>(page) * kPage, bytes.data(), kPage);
    stats_.background_pages_kept += kept.size();
    async_pending_.store(false, std::memory_order_release);
    if (write_backs_.empty()) readback_cursor_ = 0;
    return true;
}

void VulkanRenderer::complete_write_backs_over(u64 b0, u64 b1) {
    for (const PendingWriteBack& w : async_wb_) {
        const Surface& s = *w.surface;
        const u64 a0 = w.zeta ? s.zeta_phys : s.phys, a1 = a0 + (w.zeta ? s.zeta_bytes() : s.bytes());
        if (a0 < b1 && b0 < a1) {
            complete_write_backs(true);
            return;
        }
    }
}

void VulkanRenderer::write_back(Surface& s) {
    queue_write_back(s);
    finish_write_backs();
}

void VulkanRenderer::store_color(Surface& s, const u8* pixels) {
    if (debug_tex_log_) std::fprintf(stderr, "texlog: store %08x %ux%u %s\n", s.phys, s.width, s.height, s.swizzled ? "swizzled" : "linear");
    // Instrument: MV_DEBUG_TARGET_DUMP=PHYS,... saves the scaled image of those targets (32-bit) as it is written back, every
    // MV_DEBUG_TARGET_DUMP_EVERY-th time (default 1), to scratch/targetdump/PHYS-xSC-N.png.
    static const std::string dump_list = [] { const char* e = std::getenv("MV_DEBUG_TARGET_DUMP"); return std::string(e ? e : ""); }();
    if (!dump_list.empty() && s.color_bytes() == 4 && dump_list.find(std::format("{:08x}", s.phys)) != std::string::npos) dump_target(s);
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
    remember_pages(s);
    invalidate_textures(s.phys, s.bytes());
}

void VulkanRenderer::dump_target(Surface& s) {
    static const u32 every = [] { const char* e = std::getenv("MV_DEBUG_TARGET_DUMP_EVERY"); return e ? std::max(1u, static_cast<u32>(std::atoi(e))) : 1u; }();
    static std::unordered_map<u32, u32> counts;
    const u32 n = counts[s.phys]++;
    if (n % every) return;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(s.iw()) * s.ih() * 4;
    vk::Buffer buf;
    std::string error;
    if (!dev_.create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk::BufferMemory::HostRead, buf, error)) return;
    end_pass();
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {s.iw(), s.ih(), 1};
    fn_->vkCmdCopyImageToBuffer(cb_, s.color.image, VK_IMAGE_LAYOUT_GENERAL, buf.buffer, 1, &r);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    finish();
    std::vector<u8> rgba(bytes);
    const u8* p = static_cast<const u8*>(buf.mapped);
    for (std::size_t i = 0; i < bytes; i += 4) {
        rgba[i] = p[i + 2];
        rgba[i + 1] = p[i + 1];
        rgba[i + 2] = p[i];
        rgba[i + 3] = 255;
    }
    dev_.destroy_buffer(buf);
    std::error_code ec;
    std::filesystem::create_directories("scratch/targetdump", ec);
    png::write(std::format("scratch/targetdump/{:08x}-x{}-{:06}.png", s.phys, s.sc, n), s.iw(), s.ih(), rgba.data(), s.iw() * 4, error);
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

void VulkanRenderer::write_back_all(bool in_background) {
    for (auto& [k, s] : surfaces_) {
        queue_write_back(*s);
        if (s->depth_holds == s->zeta_phys) queue_write_back_zeta(*s);
    }
    if (in_background) finish_write_backs_async();
    else finish_write_backs();
    // Depth images holding an address their target no longer binds, and the parked ones.
    for (auto& [k, s] : surfaces_) {
        if (s->depth_holds != s->zeta_phys) write_back_held(*s);
        for (Surface::ParkedDepth& p : s->parked) write_back_parked(*s, p);
    }
}

void VulkanRenderer::write_back_range(u32 phys, u32 len) {
    complete_write_backs_over(phys, static_cast<u64>(phys) + len);
    wb_why_ = Stats::kWbRange;
    const u64 b0 = phys, b1 = static_cast<u64>(phys) + len;
    const auto held_over = [&](const Surface& s) { return s.depth_holds && s.depth_holds < b1 && b0 < static_cast<u64>(s.depth_holds) + s.held_bytes(); };
    for (auto& [k, s] : surfaces_) {
        const u64 a0 = s->phys, a1 = static_cast<u64>(s->phys) + s->bytes();
        if (s->dirty && a0 < b1 && b0 < a1) queue_write_back(*s);
        if (s->zeta_dirty && s->depth_holds == s->zeta_phys && held_over(*s)) queue_write_back_zeta(*s);
    }
    finish_write_backs();
    for (auto& [k, s] : surfaces_)
        if (s->zeta_dirty && s->depth_holds != s->zeta_phys && held_over(*s)) write_back_held(*s);
    flush_parked(b0, b1);
    wb_why_ = Stats::kWbAll;
}

bool VulkanRenderer::copy_rect(u32 src_phys, u32 src_pitch, u32 dst_phys, u32 dst_pitch, u32 w, u32 h, u32 bpp) {
    if (debug_roundtrip_) return false;
    if (drawid_on()) std::fprintf(stderr, "drawid 2d: copy %08x pitch %u -> %08x pitch %u, %ux%u, %u B/px\n", src_phys, src_pitch, dst_phys, dst_pitch, w, h, bpp);
    // The pitch-linear target whose image holds the rectangle at `phys` (its pixel there, as x, y), current with RAM or newer.
    auto holder = [&](u32 phys, u32 pitch, u32& x, u32& y) -> Surface* {
        for (auto& [k, sp] : surfaces_) {
            Surface& s = *sp;
            if (s.swizzled || s.pitch != pitch || s.color_bytes() != bpp || s.color_ram_newer || phys < s.phys || phys >= static_cast<u64>(s.phys) + s.bytes()) continue;
            const u32 off = phys - s.phys;
            if ((off % pitch) % bpp) continue;
            x = (off % pitch) / bpp;
            y = off / pitch;
            if (x + w <= s.width && y + h <= s.height) return &s;
        }
        return nullptr;
    };
    u32 sx = 0, sy = 0, dx = 0, dy = 0;
    Surface* src = holder(src_phys, src_pitch, sx, sy);
    Surface* dst = src ? holder(dst_phys, dst_pitch, dx, dy) : nullptr;
    if (!dst || src->color.format != dst->color.format) {
        if (drawid_on()) std::fprintf(stderr, "drawid 2d: through RAM (source target %08x, destination target %08x)\n", src ? src->phys : 0u, dst ? dst->phys : 0u);
        return false;
    }
    if (src == dst && sx < dx + w && dx < sx + w && sy < dy + h && dy < sy + h) return false;  // overlapping within one image
    // Another target over the destination's memory would not see the copy (RAM would have shown it to it): through RAM.
    const u64 d0 = dst_phys, d1 = static_cast<u64>(dst_phys) + static_cast<u64>(h - 1) * dst_pitch + static_cast<u64>(w) * bpp;
    for (auto& [k, sp] : surfaces_)
        if (sp.get() != dst && sp->phys < d1 && d0 < static_cast<u64>(sp->phys) + sp->bytes()) return false;
    end_pass();
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    if (src->sc == dst->sc) {
        VkImageCopy region{};
        region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffset = {static_cast<i32>(sx * src->sc), static_cast<i32>(sy * src->sc), 0};
        region.dstOffset = {static_cast<i32>(dx * dst->sc), static_cast<i32>(dy * dst->sc), 0};
        region.extent = {w * src->sc, h * src->sc, 1};
        fn_->vkCmdCopyImage(cb_, src->color.image, VK_IMAGE_LAYOUT_GENERAL, dst->color.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    } else {
        VkImageBlit b{};
        b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[0] = {static_cast<i32>(sx * src->sc), static_cast<i32>(sy * src->sc), 0};
        b.srcOffsets[1] = {static_cast<i32>((sx + w) * src->sc), static_cast<i32>((sy + h) * src->sc), 1};
        b.dstOffsets[0] = {static_cast<i32>(dx * dst->sc), static_cast<i32>(dy * dst->sc), 0};
        b.dstOffsets[1] = {static_cast<i32>((dx + w) * dst->sc), static_cast<i32>((dy + h) * dst->sc), 1};
        fn_->vkCmdBlitImage(cb_, src->color.image, VK_IMAGE_LAYOUT_GENERAL, dst->color.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
    }
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    // The destination's image now holds what RAM would after the blit: newer than RAM, as after a draw.
    dst->dirty = true;
    ++dst->draw_serial;
    dst->last_rendered_frame = frame_index_;
    dst->last_used_vblank = src->last_used_vblank = vblank_index_;
    dst->rendered_this_frame = true;
    ++stats_.gpu_copies;
    return true;
}

// A GPU-side write to guest memory (a blit): decoded textures over it are stale, and so is any target's image
// the guest has not drawn into since.
void VulkanRenderer::invalidate(u32 phys, u32 len) {
    if (drawid_on()) std::fprintf(stderr, "drawid 2d: RAM written %08x..%08x (targets there reload)\n", phys, phys + len);
    invalidate_textures(phys, len);
    const u64 b0 = phys, b1 = static_cast<u64>(phys) + len;
    for (auto& [k, s] : surfaces_) {
        const u64 a0 = s->phys, a1 = a0 + s->bytes(), z0 = s->zeta_phys, z1 = z0 + s->zeta_bytes();
        if ((a0 < b1 && b0 < a1) || (z0 < b1 && b0 < z1)) s->ram_newer = true;
        if (s->depth_holds && s->depth_holds < b1 && b0 < static_cast<u64>(s->depth_holds) + s->held_bytes()) release_depth(*s);  // reloads from RAM
        drop_parked(*s, b0, b1, true);
        if (a0 < b1 && b0 < a1) {
            s->color_ram_newer = true;
            if (pass_open_ && pass_target_ == s.get()) end_pass();  // the next draw into it reopens the pass, reloaded
        }
    }
}

// A render target lives on the GPU; the CPU may still write its memory (one title copies its last frame into both
// buffers with the CPU when it pauses, then only draws its menu over them: drawing on over the GPU's own images
// showed two different stale frames in turn). The reference emulator marks a surface for upload when the CPU writes
// it. Here: the pages the CPU wrote keep what it wrote, the target's other pages get what the GPU rendered (written
// back around them when the image is newer than RAM), and the image reloads from RAM before its next draw.
void VulkanRenderer::cpu_wrote(const std::vector<u32>& pages) {
    if (pages.empty()) return;
    // Copies in flight over these pages land first, under the CPU's bytes (a write anywhere else needs no wait: most are
    // texture pages, and waiting there for every batch cost the GPU thread, with its lock held, several seconds a run).
    if (write_backs_in_flight())
        for (const PendingWriteBack& w : async_wb_) {
            const Surface& ws = *w.surface;
            const u64 a0 = w.zeta ? ws.zeta_phys : ws.phys, a1 = a0 + (w.zeta ? ws.zeta_bytes() : ws.bytes());
            const auto p = std::lower_bound(pages.begin(), pages.end(), static_cast<u32>(a0 / 4096));
            if (p != pages.end() && static_cast<u64>(*p) * 4096 < a1) {
                complete_write_backs(true, &pages);
                break;
            }
        }
    constexpr u32 kPage = 4096;
    for (auto& [k, sp] : surfaces_) {
        Surface& s = *sp;
        const u32 first = s.phys / kPage, last = static_cast<u32>((static_cast<u64>(s.phys) + s.bytes() - 1) / kPage);
        auto lo = std::lower_bound(pages.begin(), pages.end(), first);
        if (lo == pages.end() || *lo > last) continue;
        const auto hi = std::upper_bound(lo, pages.end(), last);
        // Only pages whose bytes changed since the image and RAM last agreed (remember_pages).
        std::vector<u32> written;
        for (auto p = lo; p != hi; ++p)
            if (s.page_hashes.size() != last - first + 1 || texture_content(*p * kPage, kPage, 0) != s.page_hashes[*p - first]) written.push_back(*p);
        if (debug_tex_log_)
            std::fprintf(stderr, "texlog: cpu wrote %zu pages (first %05x, %zu reported) of target %08x %ux%u dirty %d\n", written.size(), *lo, static_cast<std::size_t>(hi - lo), s.phys, s.width, s.height, s.dirty ? 1 : 0);
        if (written.empty()) {
            handles_.watch_writes(s.phys, s.bytes());
            continue;
        }
        if (s.dirty) {
            // Keep the CPU's pages across the write-back of the rest.
            std::vector<u8> kept(written.size() * kPage);
            std::size_t i = 0;
            for (u32 p : written)
                if (static_cast<u64>(p + 1) * kPage <= ram_size_) std::memcpy(kept.data() + i++ * kPage, ram_ + static_cast<std::size_t>(p) * kPage, kPage);
            wb_why_ = Stats::kWbRange;
            write_back(s);
            wb_why_ = Stats::kWbAll;
            i = 0;
            for (u32 p : written)
                if (static_cast<u64>(p + 1) * kPage <= ram_size_) std::memcpy(ram_ + static_cast<std::size_t>(p) * kPage, kept.data() + i++ * kPage, kPage);
        }
        s.color_ram_newer = true;
        s.ram_newer = true;
        if (pass_open_ && pass_target_ == &s) end_pass();
        ++stats_.cpu_written_targets;
        handles_.watch_writes(s.phys, s.bytes());  // the next write matters again
    }
    // A depth buffer's memory written by the CPU: the contents go back to RAM around the CPU's pages, and the next
    // target binding the address loads it from there.
    // The parked images (D35) the same way, each let go.
    const auto around = [&](u32 addr, u32 bytes, const auto& release) {
        const u32 first = addr / kPage, last = static_cast<u32>((static_cast<u64>(addr) + bytes - 1) / kPage);
        auto lo = std::lower_bound(pages.begin(), pages.end(), first);
        if (lo == pages.end() || *lo > last) return;
        const auto hi = std::upper_bound(lo, pages.end(), last);
        std::vector<u8> kept(static_cast<std::size_t>(hi - lo) * kPage);
        std::size_t i = 0;
        for (auto p = lo; p != hi; ++p, ++i)
            if (static_cast<u64>(*p + 1) * kPage <= ram_size_) std::memcpy(kept.data() + i * kPage, ram_ + static_cast<std::size_t>(*p) * kPage, kPage);
        release();
        i = 0;
        for (auto p = lo; p != hi; ++p, ++i)
            if (static_cast<u64>(*p + 1) * kPage <= ram_size_) std::memcpy(ram_ + static_cast<std::size_t>(*p) * kPage, kept.data() + i * kPage, kPage);
        ++stats_.cpu_written_targets;
    };
    for (auto& [k, sp] : surfaces_) {
        Surface& s = *sp;
        if (s.depth_holds)
            around(s.depth_holds, s.held_bytes(), [&] {
                if (pass_open_ && pass_target_ == &s) end_pass();
                release_depth(s);
            });
        for (Surface::ParkedDepth& p : s.parked)
            if (p.depth_holds) {
                const u32 addr = p.depth_holds;
                around(addr, s.parked_bytes(p), [&] { drop_parked(s, addr, static_cast<u64>(addr) + 1, true); });
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
    if (async_compile_ && shader_pending(true, key)) return VK_NULL_HANDLE;  // a worker is compiling it
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
    std::vector<u32> spirv;
    if (!spirv_ready(true, key, t.glsl, "kelvin-vp", spirv, log)) return VK_NULL_HANDLE;  // queued for a worker
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
    if (async_compile_ && shader_pending(false, k)) return VK_NULL_HANDLE;  // a worker is compiling it
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
    std::vector<u32> spirv;
    if (!spirv_ready(false, k, t.glsl, "kelvin-combiners", spirv, log)) return VK_NULL_HANDLE;  // queued for a worker
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

VkPipeline VulkanRenderer::pipeline(const KelvinState& s, VkShaderModule vs, VkShaderModule fs, VkPrimitiveTopology topology, Surface& t, u32 vertex_layout, u32 depth_mode, std::string& error) {
    struct RasterKey {
        VkShaderModule vs, fs;
        VkRenderPass pass;
        u32 topology, blend_enable, sfactor, dfactor, equation, depth_test, depth_func, depth_write, cull_enable, cull_face, front_face,
            color_mask, stencil_enable, stencil_func, stencil_mask, stencil_write_mask, op_fail, op_zfail, op_zpass;
        u32 vertex_layout;  // the attributes with per-vertex data (also fills the last eight bytes: the key is its bytes)
    } k{vs, fs, t.pass, static_cast<u32>(topology) | (debug_nocull_ ? 0x100u : 0u) | (debug_nodepth_ ? 0x200u : 0u) | (debug_noblend_ ? 0x400u : 0u) | (depth_mode << 11), s.blend(), s.reg(mthd::kBlendSfactor), s.reg(mthd::kBlendDfactor), s.reg(mthd::kBlendEquation),
        s.depth_test(), s.reg(mthd::kDepthFunc), s.depth_write(), s.cull_face(), s.reg(mthd::kCullFace), s.reg(mthd::kFrontFace), s.color_mask(),
        s.stencil_test(), s.reg(mthd::kStencilFunc), s.reg(mthd::kStencilFuncMask), s.reg(mthd::kStencilMask), s.reg(mthd::kStencilOpFail),
        s.reg(mthd::kStencilOpZfail), s.reg(mthd::kStencilOpZpass), vertex_layout};
    static_assert(std::has_unique_object_representations_v<RasterKey>);
    static_assert(sizeof(RasterKey) <= sizeof last_raster_key_);
    if (last_raster_pipeline_ != VK_NULL_HANDLE && std::memcmp(last_raster_key_.data(), &k, sizeof k) == 0) return last_raster_pipeline_;
    std::string key(reinterpret_cast<const char*>(&k), sizeof k);
    if (auto it = pipelines_.find(key); it != pipelines_.end()) {
        std::memcpy(last_raster_key_.data(), &k, sizeof k);
        last_raster_pipeline_ = it->second;
        return it->second;
    }
    if (auto it = pipelines_pending_.find(key); it != pipelines_pending_.end()) {
        PipelineBuild& b = *it->second;
        if (!b.done.load(std::memory_order_acquire)) return VK_NULL_HANDLE;  // a worker is creating it
        const VkPipeline p = b.pipeline;
        const VkResult r = b.result;
        pipelines_pending_.erase(it);
        if (r != VK_SUCCESS) {
            error = std::format("vkCreateGraphicsPipelines: {}", vk::result_name(r));
            return VK_NULL_HANDLE;
        }
        ++stats_.pipelines;
        pipelines_[key] = p;
        return p;
    }
    auto build = std::make_shared<PipelineBuild>();

    auto& stages = build->stages;
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
    auto& bindings = build->bindings;
    bindings[0] = {0, std::max(per_vertex, 1u) * 16, VK_VERTEX_INPUT_RATE_VERTEX};
    bindings[1] = {1, 0, VK_VERTEX_INPUT_RATE_VERTEX};
    auto& attrs = build->attrs;
    for (u32 i = 0, packed = 0; i < kAttributes; ++i)
        attrs[i] = (vertex_layout >> i) & 1u ? VkVertexInputAttributeDescription{i, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16 * packed++}
                                              : VkVertexInputAttributeDescription{i, 1, VK_FORMAT_R32G32B32A32_SFLOAT, i * 16};
    auto& vi = build->vi;
    vi.vertexBindingDescriptionCount = 2;
    vi.pVertexBindingDescriptions = bindings;
    vi.vertexAttributeDescriptionCount = kAttributes;
    vi.pVertexAttributeDescriptions = attrs;
    auto& ia = build->ia;
    ia.topology = topology;
    auto& vp = build->vp;
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    auto& rs = build->rs;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.0f;
    if (s.cull_face() && !debug_nocull_) {
        const u32 face = s.reg(mthd::kCullFace);
        rs.cullMode = face == 0x404 ? VK_CULL_MODE_FRONT_BIT : (face == 0x408 ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_BACK_BIT);
    }
    rs.frontFace = (s.reg(mthd::kFrontFace) == 0x900) != debug_flip_front_ ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    // depth_mode bit 0: the pixel shader writes the console's depth (exact steps, its clip range and polygon offset), so
    // the host neither clips on z (it clamps, where the device can) nor biases; bit 1: SET_ZMIN_MAX_CONTROL clamps.
    // Otherwise the host's depth with the offset as a bias (set per draw), clamped where the console clamps.
    rs.depthClampEnable = dev_.depth_clamp() && (depth_mode & 3u) != 0 ? VK_TRUE : VK_FALSE;
    rs.depthBiasEnable = VK_TRUE;
    auto& ms = build->ms;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    auto& ds = build->ds;
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
    auto& cb = build->cb;
    cb.blendEnable = (s.blend() && !debug_noblend_) ? VK_TRUE : VK_FALSE;
    cb.srcColorBlendFactor = cb.srcAlphaBlendFactor = blend_factor(s.reg(mthd::kBlendSfactor));
    cb.dstColorBlendFactor = cb.dstAlphaBlendFactor = blend_factor(s.reg(mthd::kBlendDfactor));
    cb.colorBlendOp = cb.alphaBlendOp = blend_op(s.reg(mthd::kBlendEquation));
    if (drawid_on()) {
        cb.blendEnable = VK_TRUE;
        cb.srcColorBlendFactor = cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
        cb.dstColorBlendFactor = cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cb.colorBlendOp = cb.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    // A B8 or G8B8 target keeps blue in its first channel: the blue mask bit writes it (and red, if ever, the third).
    const u32 raw_mask = s.color_mask();
    const u32 mask = swaps_rb(t) ? (raw_mask & 0xFF00FF00u) | ((raw_mask & 0xFFu) << 16) | ((raw_mask >> 16) & 0xFFu) : raw_mask;
    cb.colorWriteMask = (mask & (1u << 16) ? static_cast<u32>(VK_COLOR_COMPONENT_R_BIT) : 0u) | (mask & (1u << 8) ? static_cast<u32>(VK_COLOR_COMPONENT_G_BIT) : 0u) |
                        (mask & 1u ? static_cast<u32>(VK_COLOR_COMPONENT_B_BIT) : 0u) | (mask & (1u << 24) ? static_cast<u32>(VK_COLOR_COMPONENT_A_BIT) : 0u);
    auto& blend = build->blend;
    blend.attachmentCount = 1;
    blend.pAttachments = &cb;
    auto& dynamics = build->dynamics;
    auto& dyn = build->dyn;
    dyn.dynamicStateCount = 5;
    dyn.pDynamicStates = dynamics;
    auto& gpi = build->gpi;
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
    if (async_compile_ && dev_.cache_control()) {
        // The cache first: a pipeline it holds is made here at once and the draw goes ahead; only one the driver has
        // to compile goes to a worker (the boot animation's last effect stood undrawn for two seconds, every new
        // pipeline of it waiting its turn on a worker).
        gpi.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT;
        VkPipeline p = VK_NULL_HANDLE;
        const VkResult r = fn_->vkCreateGraphicsPipelines(dev_.device(), pipeline_cache_, 1, &gpi, nullptr, &p);
        gpi.flags &= ~static_cast<VkPipelineCreateFlags>(VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT);
        if (r == VK_SUCCESS && p) {
            ++stats_.pipelines;
            ++stats_.pipeline_cache_hits;
            pipelines_[key] = p;
            return p;
        }
        if (p) fn_->vkDestroyPipeline(dev_.device(), p, nullptr);
    }
    if (async_compile_) {
        pipelines_pending_.emplace(key, build);
        ++stats_.pipelines_queued;
        submit([this, build] {
            build->result = fn_->vkCreateGraphicsPipelines(dev_.device(), pipeline_cache_, 1, &build->gpi, nullptr, &build->pipeline);
            build->done.store(true, std::memory_order_release);
        });
        return VK_NULL_HANDLE;
    }
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
void VulkanRenderer::hash_pages(TextureEntry& e) const {
    constexpr u32 kPage = 4096;
    e.page_hashes.clear();
    if (e.length == 0) return;
    const u32 first = e.phys / kPage, last = static_cast<u32>((static_cast<u64>(e.phys) + e.length - 1) / kPage);
    e.page_hashes.reserve(last - first + 1);
    for (u32 page = first; page <= last; ++page) {
        const u32 a = std::max(e.phys, page * kPage), b = static_cast<u32>(std::min<u64>(static_cast<u64>(e.phys) + e.length, static_cast<u64>(page + 1) * kPage));
        e.page_hashes.push_back(texture_content(a, b - a, 0));
    }
    e.palette_hash = e.palette_phys ? texture_content(e.palette_phys, 256 * 4, 0) : 0;
}

bool VulkanRenderer::texture_current(TextureEntry& e) {
    const u32 newest = std::max(handles_.newest_write(e.phys, e.length), e.palette_phys ? handles_.newest_write(e.palette_phys, 256 * 4) : 0u);
    if (newest <= e.generation) return true;
    // Only the pages written since the decode are hashed again, each against its own hash.
    constexpr u32 kPage = 4096;
    const u32 first = e.phys / kPage;
    for (std::size_t i = 0; i < e.page_hashes.size(); ++i) {
        const u32 page = first + static_cast<u32>(i);
        if (handles_.page_write(page) <= e.generation) continue;
        const u32 a = std::max(e.phys, page * kPage), b = static_cast<u32>(std::min<u64>(static_cast<u64>(e.phys) + e.length, static_cast<u64>(page + 1) * kPage));
        if (texture_content(a, b - a, 0) != e.page_hashes[i]) return false;
    }
    if (e.palette_phys && handles_.newest_write(e.palette_phys, 256 * 4) > e.generation && texture_content(e.palette_phys, 256 * 4, 0) != e.palette_hash) return false;
    e.generation = handles_.write_generation();
    handles_.watch_writes(e.phys, e.length);  // watched again: its next write is the one that matters
    if (e.palette_phys) handles_.watch_writes(e.palette_phys, 256 * 4);
    ++stats_.textures_unchanged;
    return true;
}

// One short wait for a background compile, out of the vblank's budget; false once the budget is spent.
bool VulkanRenderer::wait_a_little() {
    if (pending_wait_budget_ns_ <= 0) return false;
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    pending_wait_budget_ns_ -= std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    ++stats_.pending_waits;
    return true;
}

bool VulkanRenderer::overlaps_target(u32 phys, u32 length) {
    if (!target_ranges_valid_) {
        target_ranges_.clear();
        const auto add = [&](u64 a, u64 n) {
            if (a && n) target_ranges_.emplace_back(a, a + n);
        };
        for (const auto& [k, sp] : surfaces_) {
            const Surface& s = *sp;
            target_ranges_.emplace_back(s.phys, static_cast<u64>(s.phys) + s.bytes());  // colour (a target at 0 too)
            add(s.zeta_phys, s.zeta_bytes());
            add(s.depth_holds, s.held_bytes());
            for (const Surface::ParkedDepth& p : s.parked) add(p.depth_holds, s.parked_bytes(p));
        }
        target_ranges_valid_ = true;
    }
    const u64 b0 = phys, b1 = static_cast<u64>(phys) + length;
    for (const auto& [a0, a1] : target_ranges_)
        if (a0 < b1 && b0 < a1) return true;
    return false;
}

const VulkanRenderer::TextureEntry* VulkanRenderer::texture(const KelvinState& s, u32 stage, std::string& error) {
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
    // A texture over no target: none of the scans below can find anything (the ranges are gathered once a draw).
    if (overlaps_target(phys, length)) {
        if (TextureEntry* e = ts.cubemap() ? cube_texture(ts, phys) : ts.mip_levels() > 1 ? mip_texture(ts, phys) : surface_texture(ts, phys)) {
            sc = StageCache{regs, phys, length, e};
            return e;
        }
        // Read through RAM from here: a copy in flight over it lands first (sampling in place above needs none).
        complete_write_backs_over(phys, static_cast<u64>(phys) + length);
        // A render target the guest now samples: its pixels must be in RAM first — the colour image, or the depth
        // buffer when the texture is a shadow map.
        wb_why_ = Stats::kWbTexture;
        for (auto& [k, surf] : surfaces_) {
            const u64 a0 = surf->phys, a1 = static_cast<u64>(surf->phys) + surf->bytes();
            if (debug_tex_log_ && ((surf->dirty && a0 < static_cast<u64>(phys) + length && static_cast<u64>(phys) < a1) || (surf->zeta_dirty && surf->zeta_phys < static_cast<u64>(phys) + length && static_cast<u64>(phys) < static_cast<u64>(surf->zeta_phys) + surf->zeta_bytes())))
                std::fprintf(stderr, "texlog: forced write-back %08x/%08x for a texture at %08x fmt %02x %ux%u: open %d ram_newer %d\n", surf->phys, surf->zeta_phys, phys, ts.color_format(), surf->width, surf->height, (pass_open_ && pass_target_ == surf.get()) ? 1 : 0, surf->ram_newer ? 1 : 0);
            const auto forced = [&](const char* what) {
                ++stats_.forced_by[std::format("{} of {:08x} {}x{} {} fmt {:x} <- texture {:08x} fmt {:02x} {}x{} {}", what, surf->phys, surf->width, surf->height, surf->swizzled ? "swizzled" : "linear",
                                               surf->color_format, phys, ts.color_format(), format_info(ts.color_format()).linear ? ts.rect_width() : 1u << ts.log_width(),
                                               format_info(ts.color_format()).linear ? ts.rect_height() : 1u << ts.log_height(), ts.cubemap() ? "cube" : ts.mip_levels() > 1 ? "mips" : "") +
                                " (" + sample_miss_ + ")"];
            };
            if (surf->dirty && a0 < static_cast<u64>(phys) + length && static_cast<u64>(phys) < a1) {
                forced("colour");
                queue_write_back(*surf);
            }
            const u64 z0 = surf->zeta_phys, z1 = static_cast<u64>(surf->zeta_phys) + surf->zeta_bytes();
            if (surf->zeta_dirty && z0 < static_cast<u64>(phys) + length && static_cast<u64>(phys) < z1) {
                forced("depth");
                queue_write_back_zeta(*surf);
            }
        }
        finish_write_backs();
        // A depth image holding an address its target no longer binds, and the parked ones (D35).
        for (auto& [k, surf] : surfaces_)
            if (surf->zeta_dirty && surf->depth_holds && surf->depth_holds != surf->zeta_phys && surf->depth_holds < static_cast<u64>(phys) + length &&
                static_cast<u64>(phys) < static_cast<u64>(surf->depth_holds) + surf->held_bytes())
                write_back_held(*surf);
        flush_parked(phys, static_cast<u64>(phys) + length);
        wb_why_ = Stats::kWbAll;
    }
    if (same && sc.entry->valid && !sc.entry->surface && texture_current(*sc.entry)) return sc.entry;
    const TextureKey key{phys, ts.format, ts.control1, ts.image_rect, ts.palette};
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
    // Its pages are watched before they are read: a write from here on marks the texture stale.
    handles_.watch_writes(phys, length);
    if (palette_phys) handles_.watch_writes(palette_phys, 256 * 4);
    if (!decode_texture(ts, phys, MemoryView{ram_, ram_size_}, palette, d, native_dxt_)) {
        error = d.error;
        return nullptr;
    }
    ++stats_.textures_decoded;
    if (debug_tex_log_) {
        bool target = false;
        for (auto& [k2, surf] : surfaces_)
            target = target || (surf->phys < phys + length && phys < surf->phys + surf->bytes()) || (surf->zeta_phys < phys + length && phys < surf->zeta_phys + surf->zeta_bytes());
        std::string head;
        if (const u8* src = MemoryView{ram_, ram_size_}.at(phys, 16))
            for (u32 i = 0; i < 16; ++i) head += std::format(" {:02x}", src[i]);
        std::fprintf(stderr, "texlog: decode %08x format %02x %ux%u levels %u faces %u filter %08x%s; bytes%s\n", phys, ts.color_format(), d.width, d.height, d.levels, d.faces, ts.filter,
                     target ? " (a render target)" : "", head.c_str());
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
    VkFormat format = VK_FORMAT_R32_SFLOAT;
    switch (d.pixel) {
        case HostPixel::Rgba8: format = VK_FORMAT_R8G8B8A8_UNORM; break;
        case HostPixel::R16: format = VK_FORMAT_R16_UNORM; break;
        case HostPixel::R32f: format = VK_FORMAT_R32_SFLOAT; break;
        case HostPixel::Bc1: format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK; break;
        case HostPixel::Bc2: format = VK_FORMAT_BC2_UNORM_BLOCK; break;
        case HostPixel::Bc3: format = VK_FORMAT_BC3_UNORM_BLOCK; break;
    }
    if (d.pixel == HostPixel::Bc1 || d.pixel == HostPixel::Bc2 || d.pixel == HostPixel::Bc3) ++stats_.textures_native_dxt;
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
    hash_pages(*entry);
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
    sample_miss_ = "texture format or shape not sampled in place";
    if (debug_roundtrip_ || ts.cubemap() || ts.dimensionality() != 2 || ts.mip_levels() > 1) return nullptr;
    const u32 fmt = ts.color_format();
    // The colour formats whose texels are a target image's bytes: A8R8G8B8 and X8R8G8B8 (B8G8R8A8), R5G6B5, A1R5G5B5 and
    // X1R5G5B5 (A1R5G5B5); the X formats read alpha as one. 16-bit targets went through RAM every time (Fable's 16-bit
    // picture, sampled each frame).
    VkFormat want = VK_FORMAT_UNDEFINED;
    switch (fmt) {
        case 0x06: case 0x12: case 0x07: case 0x1E: want = VK_FORMAT_B8G8R8A8_UNORM; break;
        case 0x3A: case 0x3F: want = VK_FORMAT_B8G8R8A8_UNORM; break;  // A8B8G8R8: the same bytes, red first (abgr below)
        case 0x05: case 0x11: want = VK_FORMAT_R5G6B5_UNORM_PACK16; break;
        case 0x02: case 0x10: case 0x03: case 0x1C: want = VK_FORMAT_A1R5G5B5_UNORM_PACK16; break;
        case 0x00: case 0x13: want = VK_FORMAT_R8_UNORM; break;  // Y8 on a B8 target: (Y, Y, Y, 1) through its view (PGR2)
        default: break;
    }
    const bool color = want != VK_FORMAT_UNDEFINED;
    const bool y8 = want == VK_FORMAT_R8_UNORM;
    const bool abgr = fmt == 0x3A || fmt == 0x3F;
    const bool x8 = fmt == 0x07 || fmt == 0x1E || fmt == 0x03 || fmt == 0x1C || y8;
    const bool depth16 = fmt == 0x2C || fmt == 0x30, depth24 = fmt == 0x2A || fmt == 0x2E;  // the fixed-point depths
    if (!color && !depth16 && !depth24) return nullptr;
    const bool linear = format_info(fmt).linear;
    const u32 w = linear ? ts.rect_width() : 1u << ts.log_width(), h = linear ? ts.rect_height() : 1u << ts.log_height();
    sample_miss_ = "no target image holds the address";
    for (auto& [k, sp] : surfaces_) {
        Surface& s = *sp;
        if (debug_tex_log_ && (s.phys == phys || s.zeta_phys == phys))
            std::fprintf(stderr, "texlog: stage %08x fmt %02x %ux%u %s pitch %u vs target %08x/%08x %ux%u %s pitch %u/%u colour %u zeta %u dirty %d/%d aa %ux%u open %d\n", phys, fmt, w, h,
                         linear ? "linear" : "swizzled", ts.pitch(), s.phys, s.zeta_phys, s.width, s.height, s.swizzled ? "swizzled" : "linear", s.pitch, s.zeta_pitch,
                         s.color_format, s.zeta_format, s.dirty ? 1 : 0, s.zeta_dirty ? 1 : 0, s.aa_x, s.aa_y, (pass_open_ && pass_target_ == &s) ? 1 : 0);
        // A rectangle of a pitch-linear region (a view drawn inside it, or a smaller shape at its base): a copy of that
        // rectangle taken on the GPU.
        if (want == VK_FORMAT_B8G8R8A8_UNORM && !abgr && linear && !s.swizzled && (s.width != w || s.height != h || s.phys != phys) && ts.pitch() == s.pitch && s.color_bytes() == 4 &&
            color_vk_format(s.color_format) == VK_FORMAT_B8G8R8A8_UNORM && phys >= s.phys && phys < static_cast<u64>(s.phys) + s.bytes() && !s.ram_newer && s.aa_x == 1 &&
            s.aa_y == 1 && ((phys - s.phys) % s.pitch) % 4 == 0) {
            const u32 x0 = ((phys - s.phys) % s.pitch) / 4, y0 = (phys - s.phys) / s.pitch;
            if (x0 + w <= s.width && y0 + h <= s.height) return subrect_texture(s, ts, phys, x0, y0, w, h, x8);
        }
        // The layout the target was last set up with does not matter: its image holds pixels, and the guest clears in
        // pitch layout (one colour, the same either way) and draws in the layout it samples.
        if (s.width != w || s.height != h) {
            if (color ? s.phys == phys : s.depth_holds == phys) sample_miss_ = "size differs";
            continue;
        }
        if (!color && !s.parked.empty())
            if (TextureEntry* e = parked_texture(s, ts, phys, w, h, linear, depth16)) return e;
        const bool recent = !s.ram_newer && s.last_rendered_frame + 1 >= frame_index_;
        // The depth buffer read as colour at its own address: converted on the GPU (Z24S8 on a device with D24S8).
        if (want == VK_FORMAT_B8G8R8A8_UNORM && s.depth_holds == phys && s.phys != phys && s.zeta_format == 2 && depth24_ == VK_FORMAT_D24_UNORM_S8_UINT &&
            (s.zeta_dirty || recent) && linear == !s.swizzled && (!linear || ts.pitch() == (s.holds_pitch ? s.holds_pitch : s.zeta_pitch)))
            return zeta_color_texture(s, ts, phys, w, h, linear, x8, abgr);
        if (color) {
            if (s.phys != phys || !(s.dirty || recent)) continue;
            if (abgr) {
                sample_miss_ = "a colour target read with red and blue swapped";
                continue;
            }
            if (color_vk_format(s.color_format) != want) {
                sample_miss_ = "colour format differs";
                continue;
            }
            if (linear && ts.pitch() != s.pitch) {
                sample_miss_ = "pitch differs";
                continue;
            }
        } else {
            if (s.depth_holds != phys || !(s.zeta_dirty || recent)) continue;
            if ((s.zeta_format == 1) != depth16) {
                sample_miss_ = "depth format differs";
                continue;
            }
            if (!(depth16 ? depth16_sampled_ : depth24_sampled_)) {
                sample_miss_ = "device cannot sample the depth format";
                continue;
            }
            if (linear && ts.pitch() != (s.holds_pitch ? s.holds_pitch : s.zeta_pitch)) {
                sample_miss_ = "pitch differs";
                continue;
            }
        }
        if (&s == draw_target_) {
            if (y8) {
                sample_miss_ = "B8 target sampled as Y8 by its own draw";
                return nullptr;
            }
            // A draw sampling the target it draws into reads what the draws before it left: a GPU copy of that image.
            return color ? snapshot_texture(s, ts, phys, w, h, linear, x8) : depth_snapshot_texture(s, ts, phys, w, h, linear);
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
                if (y8) vci.components.r = vci.components.g = vci.components.b = VK_COMPONENT_SWIZZLE_R;
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
        e.phys = phys;  // a depth entry moves with its image between addresses (parked, D35)
        e.length = texture_size_in_memory(ts);
        if (attachments_written_) surface_barrier();
        sampled_since_barrier_ = true;
        ++stats_.surface_samples;
        s.last_used_vblank = vblank_index_;
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

// A draw samples the depth buffer of the target it draws into (a shadow map tested in the pass that drew it): on the
// console the texture unit reads RAM, which holds the depth the draws before it left. A copy of the depth image taken
// now on the GPU serves it, retaken only after the target was drawn into or cleared since; before, each such draw wrote
// the depth back to RAM, waited for the GPU and decoded it again (a flight title did that 46 times a frame, 27 ms frames).
VulkanRenderer::TextureEntry* VulkanRenderer::depth_snapshot_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear) {
    const vk::Image& src = s.depth;
    if (!src.image) return nullptr;
    const bool z16 = s.zeta_format == 1;
    if (s.zsnap.image && (s.zsnap.format != src.format || s.zsnap.width != src.width || s.zsnap.height != src.height)) {
        retire(s.zsnap_view);
        retire(s.zsnap);
        s.zsnap_tex = TextureEntry{};
    }
    if (!s.zsnap.image) {
        std::string error;
        if (!dev_.create_image(src.format, src.width, src.height, 1, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_DEPTH_BIT | (z16 ? 0u : static_cast<u32>(VK_IMAGE_ASPECT_STENCIL_BIT)), false, s.zsnap, error)) {
            note("depth snapshot image: " + error);
            return nullptr;
        }
        end_pass();
        fresh_image(s.zsnap);
        s.zsnap_serial = ~0ull;
    }
    if (s.zsnap_serial != s.draw_serial) {
        end_pass();
        // The depth writes before the copy reads them; earlier draws' reads of the copy before it is overwritten.
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0,
                                  nullptr);
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.extent = {src.width, src.height, 1};
        fn_->vkCmdCopyImage(cb_, src.image, VK_IMAGE_LAYOUT_GENERAL, s.zsnap.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        s.zsnap_serial = s.draw_serial;
        ++stats_.depth_snapshots;
    }
    TextureEntry& e = s.zsnap_tex;
    if (!e.valid) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.image = s.zsnap.image;
        vci.format = s.zsnap.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        if (fn_->vkCreateImageView(dev_.device(), &vci, nullptr, &view) != VK_SUCCESS) return nullptr;
        s.zsnap_view = view;
        e.image = s.zsnap;
        e.image.view = view;
        e.image.levels = 1;
        e.width = w;
        e.height = h;
        e.linear = linear;
        e.surface = true;
        e.valid = true;
    }
    e.phys = phys;
    e.length = texture_size_in_memory(ts);
    sampled_since_barrier_ = true;
    ++stats_.surface_samples;
    s.last_used_vblank = vblank_index_;
    return &e;
}

// The conversion's pieces, made the first time a draw reads a depth buffer as colour.
bool VulkanRenderer::init_zeta_convert() {
    if (zc_pipeline_) return zc_in_.buffer || zc_buffers();  // the buffers go when the internal resolution changes
    if (zc_failed_) return false;
    zc_failed_ = true;
    const auto& f = *fn_;
    // Word i of the depth copy holds the depth in its low 24 bits; the stencil bytes follow the words, packed.
    const CompiledShader cs = compile_glsl(ShaderStage::Compute,
                                           "#version 450\n"
                                           "layout(local_size_x = 64) in;\n"
                                           "layout(std430, binding = 0) readonly buffer In { uint words[]; } src;\n"
                                           "layout(std430, binding = 1) writeonly buffer Out { uint words[]; } dst;\n"
                                           "layout(push_constant) uniform P { uint count; uint stencil_word; } pc;\n"
                                           "void main() {\n"
                                           "    uint i = gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 64u + gl_GlobalInvocationID.x;  // rows of groups: past 65535 of them\n"
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
    if (!zc_buffers()) return false;
    zc_failed_ = false;
    return true;
}

// The conversion's buffers, for at least `texels` texels (kZetaConvertMaxTexels at the least), and the descriptor set
// pointed at them. They grow when a larger target needs them (once the GPU is idle) and go when the internal
// resolution changes (apply_scale).
bool VulkanRenderer::zc_buffers(u32 texels) {
    std::string error;
    const VkDeviceSize n = std::max<VkDeviceSize>(kZetaConvertMaxTexels, texels);
    if (!dev_.create_buffer(n * 5, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk::BufferMemory::Device, zc_in_, error) ||
        !dev_.create_buffer(n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk::BufferMemory::Device, zc_out_, error)) {
        note("depth conversion buffers: " + error);
        dev_.destroy_buffer(zc_in_);
        return false;
    }
    zc_capacity_ = static_cast<u32>(n);
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
    fn_->vkUpdateDescriptorSets(dev_.device(), 2, writes, 0, nullptr);
    return true;
}

// A draw reads a Z24S8 depth buffer as an A8R8G8B8 texture at the same address (a title masking its glare by
// depth): the texels are the words RAM would hold, the depth in bits 31-8 above the stencil byte. They are made on
// the GPU — depth and stencil copied into a buffer, packed by a compute pass, copied into a colour image — where
// before each read wrote the whole depth buffer back to RAM, waited for the GPU and decoded it again.
VulkanRenderer::TextureEntry* VulkanRenderer::zeta_color_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear, bool x8, bool abgr) {
    const u32 n = s.iw() * s.ih();  // the image's texels (the internal resolution's)
    if (!init_zeta_convert()) return nullptr;
    if (n > zc_capacity_) {
        // A larger target than the buffers hold (a high internal resolution): made again once nothing in flight uses them.
        finish();
        dev_.destroy_buffer(zc_in_);
        dev_.destroy_buffer(zc_out_);
        zc_capacity_ = 0;
        if (!zc_buffers(n)) return nullptr;
    }
    if (!s.zcolor.image) {
        std::string error;
        if (!dev_.create_image(VK_FORMAT_B8G8R8A8_UNORM, s.iw(), s.ih(), 1, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
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
        regions[0].imageExtent = {s.iw(), s.ih(), 1};
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
        // Groups in rows of at most 65535 (the least every device takes in one dimension; 6x already needs more).
        const u32 groups = (n + 63) / 64, gx = std::min(groups, 65535u);
        fn_->vkCmdDispatch(cb_, gx, (groups + gx - 1) / gx, 1);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy out{};
        out.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        out.imageExtent = {s.iw(), s.ih(), 1};
        fn_->vkCmdCopyBufferToImage(cb_, zc_out_.buffer, s.zcolor.image, VK_IMAGE_LAYOUT_GENERAL, 1, &out);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        s.zcolor_serial = s.draw_serial;
        ++stats_.zeta_conversions;
    }
    TextureEntry& e = abgr ? s.zcolor_abgr_tex : x8 ? s.zcolor_x8_tex : s.zcolor_tex;
    if (!e.valid) {
        VkImageView view = s.zcolor.view;
        if (abgr) {
            // A8B8G8R8: the depth word's low byte (blue in the image's format) is red to the texture.
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.image = s.zcolor.image;
            vci.format = s.zcolor.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vci.components.r = VK_COMPONENT_SWIZZLE_B;
            vci.components.b = VK_COMPONENT_SWIZZLE_R;
            if (fn_->vkCreateImageView(dev_.device(), &vci, nullptr, &view) != VK_SUCCESS) return nullptr;
            s.zcolor_abgr_view = view;
        } else if (x8) {
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

// A texture over a rectangle of a pitch-linear region: the rectangle copied into an image of its own on the GPU (the
// texture reads what the draws left in that memory), retaken only after the region was drawn into or cleared since.
VulkanRenderer::TextureEntry* VulkanRenderer::subrect_texture(Surface& s, const TextureStage& ts, u32 phys, u32 x0, u32 y0, u32 w, u32 h, bool x8) {
    const u64 key = static_cast<u64>(x0) | (static_cast<u64>(y0) << 16) | (static_cast<u64>(w) << 32) | (static_cast<u64>(h) << 48);
    auto found = s.subs.find(key);
    if (found == s.subs.end()) {
        if (s.subs.size() >= 16) retire_subs(s);
        Surface::SubCopy c;
        std::string error;
        if (!dev_.create_image(s.color.format, w * s.sc, h * s.sc, 1, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false, c.image, error)) {
            note("sub-rectangle image: " + error);
            return nullptr;
        }
        end_pass();
        fresh_image(c.image);
        found = s.subs.emplace(key, std::move(c)).first;
    }
    Surface::SubCopy& c = found->second;
    if (c.serial != s.draw_serial) {
        end_pass();
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0,
                                  nullptr);
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.srcOffset = {static_cast<i32>(x0 * s.sc), static_cast<i32>(y0 * s.sc), 0};
        region.extent = {w * s.sc, h * s.sc, 1};
        fn_->vkCmdCopyImage(cb_, s.color.image, VK_IMAGE_LAYOUT_GENERAL, c.image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        c.serial = s.draw_serial;
        ++stats_.subrect_copies;
    }
    TextureEntry& e = x8 ? c.x8_tex : c.tex;
    if (!e.valid) {
        VkImageView view = c.image.view;
        if (x8) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.image = c.image.image;
            vci.format = c.image.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vci.components.a = VK_COMPONENT_SWIZZLE_ONE;
            if (fn_->vkCreateImageView(dev_.device(), &vci, nullptr, &view) != VK_SUCCESS) return nullptr;
            c.x8_view = view;
        }
        e.image = c.image;
        e.image.view = view;
        e.image.levels = 1;
        e.phys = phys;
        e.width = w;
        e.height = h;
        e.linear = true;
        e.surface = true;
        e.valid = true;
    }
    e.length = texture_size_in_memory(ts);
    s.last_used_vblank = vblank_index_;
    return &e;
}

// An environment map rendered each frame: the guest draws the six faces of a cube map as six render targets, then samples
// them as one cube texture. The texture unit reads RAM, so before, all six were written back, waited for and decoded again
// every frame (six of OutRun 2's ~7.5 write-backs a frame in a race). Here the six target images are copied into the
// layers of a cube image on the GPU, at the targets' internal resolution, and retaken only when a face was drawn since.
VulkanRenderer::TextureEntry* VulkanRenderer::cube_texture(const TextureStage& ts, u32 phys) {
    if (debug_roundtrip_ || !ts.cubemap() || ts.mip_levels() > 1 || ts.log_width() != ts.log_height()) return nullptr;
    const u32 fmt = ts.color_format();
    // The swizzled colour formats whose texels are a target image's bytes: A8R8G8B8 / X8R8G8B8 (B8G8R8A8), R5G6B5, A1R5G5B5 /
    // X1R5G5B5; the X formats read alpha as one. PGR2's reflections are six R5G6B5 faces, each written back every frame
    // while only 32-bit faces were put together here.
    VkFormat want = VK_FORMAT_UNDEFINED;
    switch (fmt) {
        case 0x06: case 0x07: want = VK_FORMAT_B8G8R8A8_UNORM; break;
        case 0x05: want = VK_FORMAT_R5G6B5_UNORM_PACK16; break;
        case 0x02: case 0x03: want = VK_FORMAT_A1R5G5B5_UNORM_PACK16; break;
        default: return nullptr;
    }
    const bool x8 = fmt == 0x07 || fmt == 0x03;
    const u32 size = 1u << ts.log_width(), face_bytes = texture_size_in_memory(ts) / 6;
    std::array<Surface*, 6> faces{};
    for (u32 f = 0; f < 6; ++f) {
        const auto it = surfaces_.find(phys + f * face_bytes);
        if (it == surfaces_.end()) return nullptr;
        Surface& s = *it->second;
        const bool recent = !s.ram_newer && s.last_rendered_frame + 1 >= frame_index_;
        if (s.width != size || s.height != size || color_vk_format(s.color_format) != want || !(s.dirty || recent) ||
            s.color_ram_newer || (f && s.sc != faces[0]->sc))
            return nullptr;
        faces[f] = &s;
    }
    const u32 isz = size * faces[0]->sc;
    CubeEntry& c = cubes_[phys];
    if (!c.image.image || c.image.width != isz || c.image.format != want) {
        retire(c.x8_view);
        retire(c.image);
        c = CubeEntry{};
        std::string error;
        if (!dev_.create_image(want, isz, isz, 1, 1, 6, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, true, c.image,
                               error)) {
            note("cube image: " + error);
            cubes_.erase(phys);
            return nullptr;
        }
        end_pass();
        fresh_image(c.image);
        stage_cache_.fill({});  // a stage may point at the old entry
    }
    bool stale = false;
    for (u32 f = 0; f < 6; ++f) stale = stale || c.faces[f] != faces[f] || c.serials[f] != faces[f]->draw_serial;
    if (stale) {
        end_pass();
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        std::array<VkImageCopy, 6> regions{};
        for (u32 f = 0; f < 6; ++f) {
            regions[f].srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            regions[f].dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, f, 1};
            regions[f].extent = {isz, isz, 1};
            fn_->vkCmdCopyImage(cb_, faces[f]->color.image, VK_IMAGE_LAYOUT_GENERAL, c.image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &regions[f]);
            c.faces[f] = faces[f];
            c.serials[f] = faces[f]->draw_serial;
        }
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        ++stats_.cube_assemblies;
    }
    TextureEntry& e = x8 ? c.x8_tex : c.tex;
    if (!e.valid) {
        VkImageView view = c.image.view;
        if (x8) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
            vci.image = c.image.image;
            vci.format = c.image.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
            vci.components.a = VK_COMPONENT_SWIZZLE_ONE;
            if (fn_->vkCreateImageView(dev_.device(), &vci, nullptr, &view) != VK_SUCCESS) return nullptr;
            c.x8_view = view;
        }
        e.image = c.image;
        e.image.view = view;
        e.image.levels = 1;
        e.phys = phys;
        e.width = e.height = size;
        e.linear = false;
        e.surface = true;
        e.valid = true;
    }
    e.length = texture_size_in_memory(ts);
    for (Surface* s : faces) s->last_used_vblank = vblank_index_;
    ++stats_.surface_samples;
    return &e;
}

// A chain of downsampled copies drawn as one render target per level (a bloom or blur pyramid), then sampled as one
// mip-mapped texture: before, every level went through RAM each frame (two of Fable's write-backs a frame in play).
// Here the level images are copied into the mips of one image on the GPU, at the targets' internal resolution.
VulkanRenderer::TextureEntry* VulkanRenderer::mip_texture(const TextureStage& ts, u32 phys) {
    if (debug_roundtrip_ || ts.cubemap() || ts.dimensionality() != 2 || ts.mip_levels() < 2) return nullptr;
    const u32 fmt = ts.color_format();
    if (fmt != 0x06 && fmt != 0x07) return nullptr;  // swizzled A8R8G8B8 and X8R8G8B8: a target's B8G8R8A8 bytes
    const bool x8 = fmt == 0x07;
    const u32 w = 1u << ts.log_width(), h = 1u << ts.log_height();
    const u32 count = std::min(ts.mip_levels(), std::max(ts.log_width(), ts.log_height()) + 1);
    std::vector<Surface*> levels(count);
    u32 offset = phys;
    for (u32 l = 0; l < count; ++l) {
        const u32 lw = std::max(w >> l, 1u), lh = std::max(h >> l, 1u);
        const auto it = surfaces_.find(offset);
        if (it == surfaces_.end()) return nullptr;
        Surface& s = *it->second;
        const bool recent = !s.ram_newer && s.last_rendered_frame + 1 >= frame_index_;
        if (s.width != lw || s.height != lh || color_vk_format(s.color_format) != VK_FORMAT_B8G8R8A8_UNORM || !(s.dirty || recent) || s.color_ram_newer ||
            (l && s.sc != levels[0]->sc))
            return nullptr;
        levels[l] = &s;
        offset += lw * lh * 4;
    }
    const u32 sc = levels[0]->sc;
    MipEntry& m = mips_[phys];
    if (!m.image.image || m.image.width != w * sc || m.image.height != h * sc || m.image.levels != count) {
        retire(m.x8_view);
        retire(m.image);
        m = MipEntry{};
        std::string error;
        if (!dev_.create_image(VK_FORMAT_B8G8R8A8_UNORM, w * sc, h * sc, 1, count, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false,
                               m.image, error)) {
            note("mip chain image: " + error);
            mips_.erase(phys);
            return nullptr;
        }
        end_pass();
        fresh_image(m.image);
        m.serials.assign(count, ~0ull);
        m.levels.assign(count, nullptr);
        stage_cache_.fill({});
    }
    bool stale = false;
    for (u32 l = 0; l < count; ++l) stale = stale || m.levels[l] != levels[l] || m.serials[l] != levels[l]->draw_serial;
    if (stale) {
        end_pass();
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        for (u32 l = 0; l < count; ++l) {
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
            region.extent = {levels[l]->iw(), levels[l]->ih(), 1};
            fn_->vkCmdCopyImage(cb_, levels[l]->color.image, VK_IMAGE_LAYOUT_GENERAL, m.image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            m.levels[l] = levels[l];
            m.serials[l] = levels[l]->draw_serial;
        }
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        ++stats_.mip_assemblies;
    }
    TextureEntry& e = x8 ? m.x8_tex : m.tex;
    if (!e.valid) {
        VkImageView view = m.image.view;
        if (x8) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.image = m.image.image;
            vci.format = m.image.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, count, 0, 1};
            vci.components.a = VK_COMPONENT_SWIZZLE_ONE;
            if (fn_->vkCreateImageView(dev_.device(), &vci, nullptr, &view) != VK_SUCCESS) return nullptr;
            m.x8_view = view;
        }
        e.image = m.image;
        e.image.view = view;
        e.image.levels = count;
        e.phys = phys;
        e.width = w;
        e.height = h;
        e.linear = false;
        e.surface = true;
        e.valid = true;
    }
    e.length = texture_size_in_memory(ts);
    for (Surface* s : levels) s->last_used_vblank = vblank_index_;
    ++stats_.surface_samples;
    return &e;
}

void VulkanRenderer::retire_cubes() {
    for (auto& [k, c] : cubes_) {
        retire(c.x8_view);
        retire(c.image);
    }
    cubes_.clear();
    for (auto& [k, m] : mips_) {
        retire(m.x8_view);
        retire(m.image);
    }
    mips_.clear();
    stage_cache_.fill({});
}

void VulkanRenderer::retire_subs(Surface& s) {
    for (auto& [k, c] : s.subs) {
        retire(c.x8_view);
        retire(c.image);
    }
    s.subs.clear();
    stage_cache_.fill({});  // a stage may point at a retired copy's entry
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
    if (s.zcolor_abgr_view) fn_->vkDestroyImageView(dev_.device(), s.zcolor_abgr_view, nullptr);
    s.zcolor_abgr_view = VK_NULL_HANDLE;
    s.zcolor_abgr_tex = TextureEntry{};
    if (s.zsnap_view) fn_->vkDestroyImageView(dev_.device(), s.zsnap_view, nullptr);
    s.zsnap_view = VK_NULL_HANDLE;
    s.zsnap_tex = TextureEntry{};
    s.x8_view = s.depth_view = s.snap_x8_view = s.zcolor_x8_view = VK_NULL_HANDLE;
    s.color_tex = s.color_x8_tex = s.depth_tex = s.snap_tex = s.snap_x8_tex = s.zcolor_tex = s.zcolor_x8_tex = TextureEntry{};
}

// The border a clamp-to-border stage shows is its BORDER_COLOR (0x1B24, A8R8G8B8), as the hardware captures show; a
// device without custom border colours takes the nearest built-in one (transparent black, opaque black or opaque white).
VkSampler VulkanRenderer::sampler(const TextureStage& ts, u32 levels) {
    // Anisotropy for a linearly filtered minification: the stage's own (TEXCTL0 bits 5:4), raised to the option's minimum,
    // within the device's limit; its log2 rides in key bits 27:25 (unused by ADDRESS).
    const u32 min_f = ts.min_filter();
    const bool linear_min = !(min_f == 1 || min_f == 3 || min_f == 5);
    u32 aniso = linear_min && dev_.anisotropy() ? std::max(1u << ts.log_max_anisotropy(), min_anisotropy_.load(std::memory_order_relaxed)) : 1u;
    aniso = std::min(aniso, std::max(1u, static_cast<u32>(dev_.limits().maxSamplerAnisotropy)));
    const u32 aniso_log = static_cast<u32>(std::bit_width(aniso) - 1);
    const u64 key = (static_cast<u64>(ts.filter) << 32) | ts.address | (static_cast<u64>(levels) << 28) | (static_cast<u64>(aniso_log & 7u) << 25);
    // Colour textures only: a depth or 16-bit luminance texture (one channel, read through the depth path) keeps the opaque
    // black border it always had; the register colour on it flooded a title's intro movie white (not yet checked against
    // a hardware capture of a depth texture's border).
    const FormatInfo fi = format_info(ts.color_format());
    const bool colour_texture = !fi.depth && ts.color_format() != 0x32 && ts.color_format() != 0x35;
    const bool border = colour_texture && (ts.wrap_u() == 4 || ts.wrap_v() == 4 || ts.wrap_p() == 4);
    if (!border) {
        if (auto it = samplers_.find(key); it != samplers_.end()) return it->second;
    } else if (auto it = border_samplers_.find({key, ts.border_color}); it != border_samplers_.end()) {
        return it->second;
    }
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkSamplerCustomBorderColorCreateInfoEXT custom{VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    if (border) {
        const u32 c = ts.border_color;
        const float a = static_cast<float>(c >> 24) / 255.0f, r = static_cast<float>((c >> 16) & 0xFFu) / 255.0f,
                    g = static_cast<float>((c >> 8) & 0xFFu) / 255.0f, b = static_cast<float>(c & 0xFFu) / 255.0f;
        if (dev_.custom_border()) {
            custom.customBorderColor.float32[0] = r;
            custom.customBorderColor.float32[1] = g;
            custom.customBorderColor.float32[2] = b;
            custom.customBorderColor.float32[3] = a;
            custom.format = dev_.custom_border_unformatted() ? VK_FORMAT_UNDEFINED : VK_FORMAT_R8G8B8A8_UNORM;
            sci.pNext = &custom;
            sci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
        } else {
            sci.borderColor = a < 0.5f ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
                              : (r + g + b) > 1.5f ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        }
    }
    const u32 min = ts.min_filter(), mag = ts.mag_filter();
    sci.magFilter = (mag == 1) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    sci.minFilter = (min == 1 || min == 3 || min == 5) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    sci.mipmapMode = (min == 5 || min == 6) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = wrap_mode(ts.wrap_u());
    sci.addressModeV = wrap_mode(ts.wrap_v());
    sci.addressModeW = wrap_mode(ts.wrap_p());
    sci.maxLod = (min >= 3 && min <= 6) ? static_cast<float>(levels) : 0.25f;
    if (aniso > 1) {
        sci.anisotropyEnable = VK_TRUE;
        sci.maxAnisotropy = static_cast<float>(aniso);
    }
    VkSampler sm = VK_NULL_HANDLE;
    fn_->vkCreateSampler(dev_.device(), &sci, nullptr, &sm);
    // Instrument: MV_DEBUG_SAMPLERS=1 names every sampler that reads a border colour or filters anisotropically as it is made.
    static const bool log_samplers = [] { const char* e = std::getenv("MV_DEBUG_SAMPLERS"); return e && *e == '1'; }();
    if ((border || aniso > 1) && log_samplers)
        std::fprintf(stderr, "sampler: border %s colour %08x anisotropy %u address %08x format %08x filter %08x\n", border ? "yes" : "no", ts.border_color, aniso, ts.address, ts.format, ts.filter);
    if (border) border_samplers_[{key, ts.border_color}] = sm;
    else samplers_[key] = sm;
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
    ++stats_.staging_flushes;
    // The frame's region is full: the copies so far go with its submission (still ahead, in queue order, of the
    // draws recorded after them), and the next frame's region is free once its last submission has run.
    submit(true);
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
    const u32 rect_h = s.reg(mthd::kClearRectH), rect_v = s.reg(mthd::kClearRectV);
    u32 x0 = rect_h & 0xFFFFu, x1 = rect_h >> 16, y0 = rect_v & 0xFFFFu, y1 = rect_v >> 16;
    if (x1 < x0 || y1 < y0 || (x0 == 0 && x1 == 0 && y0 == 0 && y1 == 0)) {
        x0 = y0 = 0;
        x1 = view_.w - 1;
        y1 = view_.h - 1;
    } else {
        // The clear rectangle is in logical pixels too.
        x0 *= t->aa_x;
        x1 = (x1 + 1) * t->aa_x - 1;
        y0 *= t->aa_y;
        y1 = (y1 + 1) * t->aa_y - 1;
    }
    x1 = std::min(x1, view_.w - 1);
    y1 = std::min(y1, view_.h - 1);
    // The depth buffer is bound only when the clear touches it: a target's passes that never use depth leave the
    // buffer at its address where it is (a title drew three shapes at one address a frame, each binding it through RAM).
    // A clear of all of it (and of the stencil a Z24S8 buffer has) needs nothing it held.
    const bool whole = view_.x == 0 && view_.y == 0 && view_.w == t->width && view_.h == t->height && x0 == 0 && y0 == 0 && x1 + 1 == t->width && y1 + 1 == t->height;
    if (debug_tex_log_ && (bits & 3u))
        std::fprintf(stderr, "texlog: clear %08x zeta %08x bits %02x rect %u,%u..%u,%u view %u,%u %ux%u of %ux%u\n", t->phys, t->zeta_phys, bits, x0, y0, x1, y1, view_.x, view_.y, view_.w,
                     view_.h, t->width, t->height);
    if (bits & 3u) bind_depth(*t, whole && (bits & 1u) && (t->zeta_format == 1 || (bits & 2u)));
    ensure_pass(*t, (bits & 0xF0u) != 0);
    if (bits & 3u) {
        t->depth_target().zeta_dirty = true;
        if (t->borrow) ++t->borrow->draw_serial;  // the lender's depth-as-colour conversion is stale
    }
    if (bits & 0xF0u) t->dirty = true;
    ++t->clears_this_frame;
    ++t->draw_serial;
    VkClearAttachment att[2]{};
    u32 n = 0;
    if (drawid_on())
        std::fprintf(stderr, "drawid clear: surface %08x bits %02x colour %08x zstencil %08x rect %u,%u..%u,%u of %ux%u (registers %08x %08x)\n", t->phys, bits, s.clear_color(),
                     s.clear_zstencil(), x0, y0, x1, y1, t->width, t->height, rect_h, rect_v);
    float color[4];
    clear_color_rgba(t->color_format, s.clear_color(), color);
    if (swaps_rb(*t)) std::swap(color[0], color[2]);  // B8 / G8B8: blue is the image's first channel
    // CLEAR_SURFACE bits 4-7 select R, G, B and A. All four is an attachment clear; fewer is a full-screen
    // triangle that writes only those channels (an attachment clear ignores the write mask).
    const u32 channels = swaps_rb(*t) ? (((bits >> 4) & 0xAu) | ((bits >> 6) & 1u) | (((bits >> 4) & 1u) << 2)) : (bits >> 4) & 0xFu;
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
    // Within the view, in the image's pixels (the internal resolution's).
    const u32 sc = t->sc;
    rect.rect = {{static_cast<i32>((view_.x + x0) * sc), static_cast<i32>((view_.y + y0) * sc)}, {(x1 - x0 + 1) * sc, (y1 - y0 + 1) * sc}};
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
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(t->iw()), static_cast<float>(t->ih()), 0.0f, 1.0f};
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
    last_ticket_ = 0;
    // A frame whose queries are used up goes to the GPU before this draw (its pass reopens below); and with counting
    // on, every few hundred draws: the GPU works on them while the CPU goes on, so a count asked for later is usually
    // in already instead of a submit and a wait for the whole batch (5.6 % of a title's CPU thread waited there).
    // Counted: every draw for the timed pusher, or only the z-pass-counted ones for the occlusion reports.
    const bool counting = count_draws_ || (count_zpass_ && (s.reg(kZpassEnable) & 1u) != 0);
    if (counting && frames_[slot_].queries >= kSubmitDraws) submit();
    LapClock clock(stats_.ns_draw);
    ++stats_.draws;
    ++frame_draws_;
    // The renderer's draw number since it started (unique across frames: the replay's frames and the flips differ).
    const u32 draw_id = static_cast<u32>(stats_.draws);
    if (drawid_on())
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
    if (s.blend()) {
        const TextureStage t0 = s.texture(0);
        const u64 key = (static_cast<u64>(s.reg(mthd::kBlendSfactor) & 0xFFFFu) << 48) | (static_cast<u64>(s.reg(mthd::kBlendDfactor) & 0xFFFFu) << 32) |
                        (static_cast<u64>(s.reg(mthd::kBlendEquation) & 0xFFFFu) << 16) | ((s.reg(mthd::kControl0) & (1u << 16)) ? 0x8000u : 0u) |
                        (static_cast<u64>(t0.color_format() & 0x7Fu) << 8) | ((t0.wrap_u() & 0xFu) << 4) | (t0.wrap_v() & 0xFu);
        ++stats_.blend_modes[key];
    }
    std::string error;
    Surface* t = target_for(s);
    if (!t) {
        ++stats_.skipped;
        return;
    }
    if (debug_tex_log_) std::fprintf(stderr, "texlog: draw into %08x %s\n", t->phys, s.surface().swizzled() ? "swizzled" : "linear");
    if (drawid_on()) {
        // Has anything but this renderer written the target's RAM since the last draw into it?
        static std::unordered_map<u32, u32> seen;
        const u32 newest = handles_.newest_write(t->phys, t->bytes());
        u32& last = seen[t->phys];
        if (last && newest > last) std::fprintf(stderr, "drawid %u: target %08x RAM written by the CPU or a device since its last draw (generation %u -> %u)\n", draw_id, t->phys, last, newest);
        last = std::max(newest, handles_.write_generation());
    }
    // The fetched vertices and the host's index list live in members: their buffers are reused draw after draw.
    FetchedVertices& v = fetched_;
    clock.skip();
    const bool fetched = fetch_vertices(s, handles_, MemoryView{ram_, ram_size_}, batch, v);
    clock.lap(stats_.ns_fetch);
    if (!fetched) {
        note("vertex fetch: " + v.error);
        ++stats_.skipped;
        return;
    }
    if (v.count == 0) return;

    // Shaders. With background compiles a shader not ready yet comes back null with no error: both are asked for (so
    // both compiles start together), then the draw is skipped until they are.
    error.clear();
    VkShaderModule vs = vertex_shader(s, error);
    clock.lap(stats_.ns_vs);
    if (!vs && !error.empty()) {
        note(error);
        ++stats_.skipped;
        return;
    }
    if (drawid_on())
        if (auto it = vs_debug_ids_.find(vs); it != vs_debug_ids_.end()) std::fprintf(stderr, "drawid %u: vertex shader vs-%u\n", draw_id, it->second);
    psh::Key pkey = psh::key_from_state(s);
    pkey.swap_rb = swaps_rb(*t);
    if (debug_shadow_pass_) pkey.shadow_compare_func = 7;  // every shadow lookup lit
    // The console's exact depth from the pixel shader wherever depth is stored or a test must land on the console's
    // step: depth writes (what later draws and shadow-map compares read; the host's own depth for them drew a grid over a
    // racing title's road), occlusion-counted draws (a test quad at a light's depth) and EQUAL tests. Draws that only
    // test (blended and transparent layers, most of the overdraw) keep the host's depth and its early rejection: the
    // shader's depth for every draw cost that title's race 15.9 -> 20.1 ms a vblank at 6x, the writes alone nothing
    // measurable (15.45).
    // The w-buffer mode (SET_CONTROL0 bit 16) stores w, which the host's depth cannot give: every draw that tests or
    // writes depth there takes the exact path (a racing title draws its whole scene so and reads its depth buffer back
    // as a texture).
    // MV_WBUFFER=0: the w-buffer mode ignored (depth z/w as before): the control for comparing a title both ways.
    static const bool w_buffer_on = [] { const char* e = std::getenv("MV_WBUFFER"); return !(e && *e == '0'); }();
    const bool w_buffer = w_buffer_on && (s.reg(mthd::kControl0) & (1u << 16)) != 0;
    const bool exact_depth = (s.depth_test() || s.depth_write()) &&
                             (w_buffer || s.depth_write() || counting || s.reg(mthd::kDepthFunc) == 0x202);
    pkey.exact_depth = exact_depth ? (w_buffer ? 3u : 1u) : 0u;
    if (drawid_on()) std::fprintf(stderr, "drawid %u: depth %s, zeta %08x\n", draw_id, exact_depth ? "exact" : "host", t->zeta_phys);
    const u32 depth_mode = (exact_depth ? 1u : 0u) | (((s.reg(kZminMaxControl) >> 4) & 0xFu) == 1 ? 2u : 0u);
    std::array<psh::SamplerKind, 4> kinds{};
    VkShaderModule fs = fragment_shader(pkey, kinds, error);
    clock.lap(stats_.ns_fs);
    if (!fs && !error.empty()) {
        note(error);
        ++stats_.skipped;
        return;
    }
    while ((!vs || !fs) && wait_a_little()) {
        if (!vs) vs = vertex_shader(s, error);
        if (!fs) fs = fragment_shader(pkey, kinds, error);
        if (!error.empty()) break;
    }
    if (!vs || !fs) {
        if (!error.empty()) note(error);
        ++stats_.pending_skips;
        return;
    }

    // Textures (may write surfaces back and flush: before the pass for this draw begins).
    draw_target_ = t;
    target_ranges_valid_ = false;  // gathered again for this draw's stages
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
    clock.lap(stats_.ns_texture);

    // Topology and the index list the host understands: the fetched list itself, unless the primitive needs it
    // rewritten (then a copy in draw_indices_).
    const std::vector<u32>* index_list = &v.indices;
    auto sequential = [&]() {
        if (index_list->empty()) {
            draw_indices_.resize(v.count);
            for (u32 i = 0; i < v.count; ++i) draw_indices_[i] = i;
        } else {
            draw_indices_ = *index_list;
        }
        index_list = &draw_indices_;
    };
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    switch (v.primitive) {
        case Primitive::Points: topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case Primitive::Lines: topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case Primitive::LineStrip: topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        case Primitive::LineLoop:
            topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            sequential();
            draw_indices_.push_back(draw_indices_.front());
            break;
        case Primitive::Triangles: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
        case Primitive::TriangleStrip: case Primitive::QuadStrip: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case Primitive::TriangleFan: case Primitive::Polygon: topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case Primitive::Quads: {
            if (index_list->empty()) sequential();
            const std::vector<u32>& quads = *index_list;
            quad_indices_.clear();
            quad_indices_.reserve(quads.size() / 4 * 6);
            for (std::size_t i = 0; i + 3 < quads.size(); i += 4)
                quad_indices_.insert(quad_indices_.end(), {quads[i], quads[i + 1], quads[i + 2], quads[i], quads[i + 2], quads[i + 3]});
            index_list = &quad_indices_;
            topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            break;
        }
        case Primitive::End: return;
    }

    clock.lap(stats_.ns_indices);
    error.clear();
    VkPipeline p = pipeline(s, vs, fs, topology, *t, v.enabled, depth_mode, error);
    while (!p && error.empty() && wait_a_little()) p = pipeline(s, vs, fs, topology, *t, v.enabled, depth_mode, error);
    clock.lap(stats_.ns_pipeline);
    if (!p && error.empty()) {  // a worker is creating it
        ++stats_.pending_skips;
        return;
    }
    if (!p) {
        note(error);
        ++stats_.skipped;
        return;
    }

    // Vertex data: the attributes with per-vertex data as vec4s packed in order, then the sixteen constants the others
    // read (binding 1, a stride of zero). Writing every attribute for every vertex cost a draw-heavy title 3 ms a frame.
    const u32 per_vertex = static_cast<u32>(std::popcount(static_cast<u32>(v.enabled)));
    const VkDeviceSize vertex_bytes = static_cast<VkDeviceSize>(v.count) * per_vertex * 16;
    VkDeviceSize vbytes = vertex_bytes + kAttributes * 16, voffset = 0;
    // The rings and the descriptor pool hold one batch of draws. A draw that no longer fits submits the batch so far
    // and moves to the next frame in flight's regions, so a frame can hold any number of draws (only a single draw
    // larger than a ring region is refused below).
    if (!fits_in_batch(vbytes, index_list->size() * 4)) {
        submit(true);
        ++stats_.batch_flushes;
    }
    if (!ring_alloc(vertex_ring_, vbytes, 16, voffset)) {
        note("vertex ring full for this frame");
        ++stats_.skipped;
        return;
    }
    float* vdst = reinterpret_cast<float*>(static_cast<u8*>(vertex_ring_.buffer.mapped) + voffset);
    {
        std::array<const float*, kAttributes> src{};
        u32 n_src = 0;
        for (u32 a = 0; a < kAttributes; ++a)
            if (v.enabled & (1u << a)) src[n_src++] = v.attribute[a].data();
        float* d = vdst;
        for (u32 n = 0; n < v.count; ++n)
            for (u32 i = 0; i < n_src; ++i, d += 4) std::memcpy(d, src[i] + static_cast<std::size_t>(n) * 4, 16);
        for (u32 a = 0; a < kAttributes; ++a, d += 4) std::memcpy(d, v.constant[a].data(), 16);
    }
    VkDeviceSize ioffset = 0;
    if (!index_list->empty()) {
        if (!ring_alloc(index_ring_, index_list->size() * 4, 4, ioffset)) {
            note("index ring full for this frame");
            ++stats_.skipped;
            return;
        }
        std::memcpy(static_cast<u8*>(index_ring_.buffer.mapped) + ioffset, index_list->data(), index_list->size() * 4);
    }

    // Uniforms: built on the CPU, and uploaded only when they differ from the block the frame last uploaded (a
    // run of draws under the same transforms and combiners binds the copy already in the ring).
    VkDeviceSize uoffset_vs = 0, uoffset_ps = 0;
    {
        float* u = vs_uniforms_.data();
        std::memcpy(u, s.constants.data(), kConstants * 16);
        u[kConstants * 4 + 0] = static_cast<float>(view_.w / t->aa_x);  // the logical size the program maps to (the view's)
        u[kConstants * 4 + 1] = static_cast<float>(view_.h / t->aa_y);
        u[kConstants * 4 + 2] = static_cast<float>(t->sc);  // image pixels per stored pixel: point sizes, the vertex snap
        u[kConstants * 4 + 3] = (t->sc > 1 && t->aa_x == 1 && t->aa_y == 1) ? 1.0f : 0.0f;  // 2D edges on a pixel centre cover whole blocks
        u[kConstants * 4 + 4] = 0.0f;
        u[kConstants * 4 + 5] = t->zeta_format == 1 ? 65535.0f : 16777215.0f;
        u[kConstants * 4 + 6] = 0.0f;
        u[kConstants * 4 + 7] = 0.0f;
        // The fixed function's material, lights and planes.
        vsh::fixed_function_uniforms(s, u + kConstants * 4 + 8);
        psh::Uniforms pu{};
        psh::uniforms_from_state(s, scales, pu);
        {
            // The depth the pixel shader writes as the console does: the format's range (the vertex side's z scale), the
            // clip range, the polygon offset for triangle-class primitives with the front mode's enable (its slope term
            // per image pixel: sc times larger at the internal resolution), and SET_ZMIN_MAX_CONTROL's cull or clamp.
            const u32 mode = s.reg(mthd::kFrontPolygonMode);
            const u32 enable = mode == 0x1B00 ? mthd::kPolyOffsetPointEnable : mode == 0x1B01 ? mthd::kPolyOffsetLineEnable : mthd::kPolyOffsetFillEnable;
            const bool offset_on = v.primitive >= Primitive::Triangles && s.reg(enable) != 0;
            const bool float_depth = (s.reg(mthd::kControl0) & (1u << 12)) != 0;  // SET_CONTROL0 Z_FORMAT
            pu.depth[0][0] = t->zeta_format == 1 ? 65535.0f : 16777215.0f;
            pu.depth[0][1] = s.regf(mthd::kClipMin);
            pu.depth[0][2] = s.regf(mthd::kClipMax);
            pu.depth[0][3] = offset_on ? std::bit_cast<float>(s.reg(mthd::kPolygonOffsetBias)) : 0.0f;
            pu.depth[1][0] = offset_on ? std::bit_cast<float>(s.reg(mthd::kPolygonOffsetFactor)) * static_cast<float>(t->sc) : 0.0f;
            pu.depth[1][1] = float_depth ? 2.0f : (t->zeta_format == 1 ? 0.0f : 1.0f);
            pu.depth[1][2] = ((s.reg(kZminMaxControl) >> 4) & 0xFu) == 0 ? 1.0f : 0.0f;  // ZCLAMP_EN: 0 cull, 1 clamp
            pu.depth[1][3] = 0.0f;
        }
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

    clock.lap(stats_.ns_upload);
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

    clock.lap(stats_.ns_descriptors);
    // Record.
    // A depth test that always passes and writes nothing leaves the buffer alone: its contents do not matter to the
    // draw (Half-Life 2 draws two small targets so over a depth buffer of another shape, each bind a write-back and a
    // load through RAM).
    const bool uses_depth = (s.depth_test() && (s.depth_write() || s.reg(mthd::kDepthFunc) != 0x207)) || s.stencil_test();
    if (debug_tex_log_ && uses_depth && t->depth_holds != t->zeta_phys)
        std::fprintf(stderr, "texlog: draw binds depth %08x into %08x: test %d func %03x write %d stencil %d func %03x ops %03x/%03x/%03x\n", t->zeta_phys, t->phys, s.depth_test() ? 1 : 0,
                     s.reg(mthd::kDepthFunc), s.depth_write() ? 1 : 0, s.stencil_test() ? 1 : 0, s.reg(mthd::kStencilFunc), s.reg(mthd::kStencilOpFail), s.reg(mthd::kStencilOpZfail),
                     s.reg(mthd::kStencilOpZpass));
    if (uses_depth) bind_depth(*t);  // only a draw that tests depth or stencil needs the buffer in place
    ensure_pass(*t, s.color_mask() != 0);
    if ((s.depth_test() && s.depth_write()) || s.stencil_test()) {
        t->depth_target().zeta_dirty = true;
        if (t->borrow) ++t->borrow->draw_serial;  // the lender's depth-as-colour conversion is stale
    }
    if (s.color_mask() != 0) t->dirty = true;
    ++t->draws_this_frame;
    ++t->draw_serial;
    fn_->vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    // A bound set stays bound across pipelines (they share one layout) and passes, for the whole command buffer.
    if (set != bound_set_) {
        fn_->vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &set, 0, nullptr);
        bound_set_ = set;
    }
    // The view's rectangle of the target's image (the whole image for a target of its own), in the image's pixels.
    const View vw = view_;
    const u32 sc = t->sc;
    VkViewport viewport{static_cast<float>(vw.x * sc), static_cast<float>(vw.y * sc), static_cast<float>(vw.w * sc), static_cast<float>(vw.h * sc), 0.0f, 1.0f};
    fn_->vkCmdSetViewport(cb_, 0, 1, &viewport);
    const SurfaceState sd = s.surface();
    const u32 cx = sd.clip_x() * t->aa_x, cy = sd.clip_y() * t->aa_y, cw = sd.clip_width() * t->aa_x, ch = sd.clip_height() * t->aa_y;
    VkRect2D scissor{{static_cast<i32>(std::min(cx, vw.w - 1)), static_cast<i32>(std::min(cy, vw.h - 1))},
                     {std::max(1u, std::min(cw, vw.w - std::min(cx, vw.w - 1))), std::max(1u, std::min(ch, vw.h - std::min(cy, vw.h - 1)))}};
    if (sd.clip_width() == 0 || sd.clip_height() == 0) scissor = {{0, 0}, {vw.w, vw.h}};
    scissor.offset.x = (scissor.offset.x + static_cast<i32>(vw.x)) * static_cast<i32>(sc);
    scissor.offset.y = (scissor.offset.y + static_cast<i32>(vw.y)) * static_cast<i32>(sc);
    scissor.extent.width *= sc;
    scissor.extent.height *= sc;
    fn_->vkCmdSetScissor(cb_, 0, 1, &scissor);
    const u32 bc = s.reg(mthd::kBlendColor);
    float constants[4] = {static_cast<float>((bc >> 16) & 0xFFu) / 255.0f, static_cast<float>((bc >> 8) & 0xFFu) / 255.0f,
                                static_cast<float>(bc & 0xFFu) / 255.0f, static_cast<float>(bc >> 24) / 255.0f};
    if (swaps_rb(*t)) std::swap(constants[0], constants[2]);  // the shader's output is swapped for such a target
    if (drawid_on()) {
        constants[0] = static_cast<float>(draw_id & 0xFFu) / 255.0f;
        constants[1] = static_cast<float>((draw_id >> 8) & 0xFFu) / 255.0f;
        constants[2] = static_cast<float>((draw_id >> 16) & 0xFFu) / 255.0f;
        constants[3] = 1.0f;
    }
    fn_->vkCmdSetBlendConstants(cb_, constants);
    fn_->vkCmdSetStencilReference(cb_, VK_STENCIL_FACE_FRONT_AND_BACK, s.reg(mthd::kStencilRef) & 0xFFu);
    {
        // The host's depth (draws without the exact path): the polygon offset as the depth bias (bias + factor x slope, in
        // the attachment's steps, the slope sc times smaller at the internal resolution); the host rounds to the step as
        // the exact draws do. The exact draws make all of it in the shader.
        const u32 mode = s.reg(mthd::kFrontPolygonMode);
        const u32 enable = mode == 0x1B00 ? mthd::kPolyOffsetPointEnable : mode == 0x1B01 ? mthd::kPolyOffsetLineEnable : mthd::kPolyOffsetFillEnable;
        const bool on = !exact_depth && v.primitive >= Primitive::Triangles && s.reg(enable) != 0;
        fn_->vkCmdSetDepthBias(cb_, on ? std::bit_cast<float>(s.reg(mthd::kPolygonOffsetBias)) : 0.0f, 0.0f,
                               on ? std::bit_cast<float>(s.reg(mthd::kPolygonOffsetFactor)) * static_cast<float>(sc) : 0.0f);
    }
    const VkDeviceSize vb_offsets[2] = {voffset, voffset + vertex_bytes};
    const VkBuffer vbs[2] = {vertex_ring_.buffer.buffer, vertex_ring_.buffer.buffer};
    fn_->vkCmdBindVertexBuffers(cb_, 0, 2, vbs, vb_offsets);
    Frame& qf = frames_[slot_];
    const u32 query = qf.queries;
    if (counting) {
        fn_->vkCmdBeginQuery(cb_, qf.occlusion, query, dev_.occlusion_precise() ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
    }
    if (!index_list->empty()) {
        fn_->vkCmdBindIndexBuffer(cb_, index_ring_.buffer.buffer, ioffset, VK_INDEX_TYPE_UINT32);
        fn_->vkCmdDrawIndexed(cb_, static_cast<u32>(index_list->size()), 1, 0, 0, 0);
    } else {
        fn_->vkCmdDraw(cb_, v.count, 1, 0, 0);
    }
    if (counting) {
        fn_->vkCmdEndQuery(cb_, qf.occlusion, query);
        qf.aa_scale[query] = t->aa_x * t->aa_y;
        qf.res_scale[query] = sc * sc;
        ++qf.queries;
        last_ticket_ = qf.first_ticket + query;
        if (drawid_on()) std::fprintf(stderr, "drawid %u: counted, ticket %llu, %u vertices, %zu indices, viewport %g,%g %gx%g scissor %d,%d %ux%u\n", draw_id, static_cast<unsigned long long>(last_ticket_), v.count, index_list->size(), static_cast<double>(viewport.x), static_cast<double>(viewport.y), static_cast<double>(viewport.width), static_cast<double>(viewport.height), scissor.offset.x, scissor.offset.y, scissor.extent.width, scissor.extent.height);
        // Instrument (drawid window, counted draws): each vertex as the program leaves it (screen x, y, depth units, w),
        // from the CPU interpreter, with the clip range: where an occlusion test quad lands.
        if (drawid_on() && s.vertex_program_mode()) {
            const vsh::Program prog = vsh::extract_program(s);
            std::string line;
            for (u32 n = 0; n < std::min<u32>(v.count, 8); ++n) {
                std::array<vsh::Vec4, kAttributes> in{};
                for (u32 a = 0; a < kAttributes; ++a)
                    for (u32 c = 0; c < 4; ++c) in[a][c] = (v.enabled & (1u << a)) ? v.attribute[a][static_cast<std::size_t>(n) * 4 + c] : v.constant[a][c];
                const vsh::Outputs o = vsh::run(prog, in, s.constants);
                line += std::format(" ({:.2f}, {:.2f}, {:.1f}, {:.3f})", o.pos[0], o.pos[1], o.pos[2], o.pos[3]);
            }
            std::fprintf(stderr, "drawid %u: positions%s; clip %g..%g, zmin/max control %08x\n", draw_id, line.c_str(), static_cast<double>(s.regf(mthd::kClipMin)),
                         static_cast<double>(s.regf(mthd::kClipMax)), s.reg(kZminMaxControl));
        }
        next_ticket_ = last_ticket_ + 1;
    }
    clock.lap(stats_.ns_record);
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
    if (requested_scale_.load(std::memory_order_relaxed) != scale_) apply_scale();
    // A frame boundary: the next submit moves to the next frame in flight, so what the frame retired is freed in time.
    rotate_pending_ = true;
    pending_wait_budget_ns_ = pending_wait_per_vblank_ns_;
    // Targets nothing has drawn into or sampled for ten seconds, and that hold nothing newer than RAM, retire.
    if (++vblank_index_ % 60 == 0)
        for (auto it = surfaces_.begin(); it != surfaces_.end();) {
            const Surface& s = *it->second;
            if ((phys < s.phys || phys >= static_cast<u64>(s.phys) + s.bytes()) && !s.dirty && !s.zeta_dirty && s.last_used_vblank + 600 < vblank_index_ && !(pass_open_ && pass_target_ == &s)) {
                ++stats_.retired_unused;
                it = erase_surface(it);
            } else {
                ++it;
            }
        }
    if (debug_wb_all_) write_back_all();
    if (scale_ > 1) {
        std::lock_guard g(display_mutex_);
        shown_ = nullptr;  // until a target below holds the picture
    }
    scan_phys_ = phys;
    wb_why_ = Stats::kWbScanout;
    for (auto& [k, s] : surfaces_) {
        if (phys < s->phys || phys >= static_cast<u64>(s->phys) + s->bytes()) continue;  // the region holding the scanout
        if (s->dirty) queue_write_back(*s);
        // Above 1x the picture is the GPU image (capture_display), so the copy in RAM is for the guest alone: it lands in
        // the background (complete_write_backs) instead of the vblank waiting for the GPU to finish the whole frame.
        if (s->sc > 1) finish_write_backs_async();
        else finish_write_backs();
        capture_display(*s);  // a scaled target: its picture at the internal resolution, not waited for here
        wb_why_ = Stats::kWbAll;
        return true;
    }
    if (debug_tex_log_ && scale_ > 1) std::fprintf(stderr, "texlog: no target holds the scanout %08x\n", phys);
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

// ── the internal resolution (D34) ───────────────────────────────────────────────────────────────────────────────

// The scale changes: every target reaches RAM and goes (each comes back from RAM at its next use, at the new scale),
// and the GPU finishes, so the depth conversion's buffers can be made again for the new size.
void VulkanRenderer::apply_scale() {
    const u32 next = requested_scale_.load(std::memory_order_relaxed);
    end_pass();
    complete_write_backs(true);
    write_back_all();
    for (auto it = surfaces_.begin(); it != surfaces_.end();) it = erase_surface(it);
    retire_cubes();
    depth_owner_.clear();
    finish();
    dev_.destroy_buffer(zc_in_);
    dev_.destroy_buffer(zc_out_);
    zc_capacity_ = 0;
    {
        // The GPU is idle (finish above). The picture the frontend holds goes once it is let go.
        std::lock_guard g(display_mutex_);
        for (auto it = captures_.begin(); it != captures_.end();) {
            if (it->get() == held_) {
                (*it)->retired = true;
                (*it)->valid = false;
                ++it;
            } else {
                destroy_capture(**it);
                it = captures_.erase(it);
            }
        }
        shown_ = nullptr;
        last_capture_ = nullptr;
    }
    scale_ = next;
}

u32 VulkanRenderer::scale_for(VkFormat color, u32 width, u32 height) const {
    if (scale_ <= 1 || !dev_.format_supports(color, VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) return 1;
    const VkPhysicalDeviceLimits& l = dev_.limits();
    const u32 limit = std::min({l.maxImageDimension2D, l.maxFramebufferWidth, l.maxFramebufferHeight});
    u32 sc = scale_;
    while (sc > 1 && (width * sc > limit || height * sc > limit)) --sc;
    return sc;
}

bool VulkanRenderer::ensure_lo(Surface& s) {
    if (s.lo.image) return true;
    std::string error;
    if (!dev_.create_image(s.color.format, s.width, s.height, 1, 1, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false, s.lo,
                           error)) {
        note("stored-size copy of a scaled target: " + error);
        return false;
    }
    s.lo_linear = dev_.format_supports(s.color.format, VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
    end_pass();
    fresh_image(s.lo);
    return true;
}

// Down: the image into its stored-size copy, filtered (2x and 4x average whole blocks; 3x takes each block's middle).
// Up: the copy widened back, NEAREST, so every stored pixel keeps its value over its sc x sc block.
void VulkanRenderer::blit_lo(Surface& s, bool down) {
    end_pass();
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    const vk::Image& src = down ? s.color : s.lo;
    const vk::Image& dst = down ? s.lo : s.color;
    VkImageBlit b{};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstSubresource = b.srcSubresource;
    b.srcOffsets[1] = {static_cast<i32>(src.width), static_cast<i32>(src.height), 1};
    b.dstOffsets[1] = {static_cast<i32>(dst.width), static_cast<i32>(dst.height), 1};
    fn_->vkCmdBlitImage(cb_, src.image, VK_IMAGE_LAYOUT_GENERAL, dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, down && s.lo_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
}

bool VulkanRenderer::ensure_zeta_lo(Surface& s) {
    if (debug_depth_rows_) return false;
    const vk::Image& d = s.depth;
    if (s.zeta_lo.image && s.zeta_lo.format == d.format && s.zeta_lo.width == s.width && s.zeta_lo.height == s.height) return true;
    if (!dev_.format_supports(d.format, VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) return false;
    retire(s.zeta_lo);
    std::string error;
    if (!dev_.create_image(d.format, s.width, s.height, 1, 1, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, d.aspect, false, s.zeta_lo, error)) {
        note("stored-size copy of a scaled depth buffer: " + error);
        return false;
    }
    end_pass();
    fresh_image(s.zeta_lo);
    return true;
}

// The depth image down to its stored size. NEAREST (the only filter a depth blit has) takes the texel under each stored
// pixel's centre, (x + 0.5) * sc: the middle of its sc x sc block, as the row read-back does (at an even sc the centre is
// a texel edge and the device may take either neighbour; a block drawn at one depth, or loaded from RAM, is the same).
void VulkanRenderer::blit_zeta_lo(Surface& s) {
    end_pass();
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    VkImageBlit b{};
    b.srcSubresource = {s.zeta_lo.aspect, 0, 0, 1};
    b.dstSubresource = b.srcSubresource;
    b.srcOffsets[1] = {static_cast<i32>(s.iw()), static_cast<i32>(s.ih()), 1};
    b.dstOffsets[1] = {static_cast<i32>(s.width), static_cast<i32>(s.height), 1};
    fn_->vkCmdBlitImage(cb_, s.depth.image, VK_IMAGE_LAYOUT_GENERAL, s.zeta_lo.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
}

std::size_t VulkanRenderer::zeta_rows_bytes(const Surface& s) const {
    const bool z16 = s.zeta_format == 1;
    const std::size_t drow = (static_cast<std::size_t>(s.iw()) * (z16 ? 2u : 4u) + 3) / 4 * 4, srow = z16 ? 0 : (static_cast<std::size_t>(s.iw()) + 3) / 4 * 4;
    return (drow + srow) * s.height;
}

// Rows y0 .. y0 + rows of the stored size: the depth words of each (the middle image row of its block), then the
// stencil bytes, at offset in the read-back buffer.
void VulkanRenderer::read_back_zeta_rows(Surface& s, VkDeviceSize offset, u32 y0, u32 rows) {
    end_pass();
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    const bool z16 = s.zeta_format == 1;
    const VkDeviceSize drow = (static_cast<VkDeviceSize>(s.iw()) * (z16 ? 2u : 4u) + 3) / 4 * 4, srow = z16 ? 0 : (static_cast<VkDeviceSize>(s.iw()) + 3) / 4 * 4;
    std::vector<VkBufferImageCopy> regions;
    regions.reserve(static_cast<std::size_t>(rows) * (z16 ? 1 : 2));
    for (u32 y = 0; y < rows; ++y) {
        VkBufferImageCopy r{};
        r.bufferOffset = offset + y * drow;
        r.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        r.imageOffset = {0, static_cast<i32>((y0 + y) * s.sc + s.sc / 2), 0};
        r.imageExtent = {s.iw(), 1, 1};
        regions.push_back(r);
        if (z16) continue;
        r.bufferOffset = offset + rows * drow + y * srow;
        r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        regions.push_back(r);
    }
    fn_->vkCmdCopyImageToBuffer(cb_, s.depth.image, VK_IMAGE_LAYOUT_GENERAL, readback_.buffer, static_cast<u32>(regions.size()), regions.data());
}

void VulkanRenderer::thin_zeta_rows_into(const Surface& s, const u8* in, u32 y0, u32 rows, u8* out) const {
    const bool z16 = s.zeta_format == 1;
    const u32 dsz = z16 ? 2u : 4u, w = s.width, sc = s.sc, mid = sc / 2;
    const std::size_t drow = (static_cast<std::size_t>(s.iw()) * dsz + 3) / 4 * 4, srow = z16 ? 0 : (static_cast<std::size_t>(s.iw()) + 3) / 4 * 4;
    const std::size_t n = static_cast<std::size_t>(w) * s.height;
    for (u32 y = 0; y < rows; ++y) {
        const u8* d = in + y * drow;
        const u8* st = in + rows * drow + y * srow;
        for (u32 x = 0; x < w; ++x) {
            const std::size_t i = static_cast<std::size_t>(y0 + y) * w + x, sx = static_cast<std::size_t>(x) * sc + mid;
            std::memcpy(&out[i * dsz], d + sx * dsz, dsz);
            if (!z16) out[n * 4 + i] = st[sx];
        }
    }
}

std::vector<u8> VulkanRenderer::thin_zeta_rows(const Surface& s, const u8* rows) const {
    const std::size_t n = static_cast<std::size_t>(s.width) * s.height;
    std::vector<u8> out(s.zeta_format == 1 ? n * 2 : n * 5);
    thin_zeta_rows_into(s, rows, 0, s.height, out.data());
    return out;
}

// A scaled depth buffer whose rows do not fit the read-back buffer at once: what was queued before it reaches RAM first,
// then its rows in bands, each copied, waited for and thinned in turn.
void VulkanRenderer::write_back_zeta_banded(Surface& s) {
    complete_write_backs_over(s.zeta_phys, static_cast<u64>(s.zeta_phys) + s.zeta_bytes());
    finish_write_backs();
    const bool z16 = s.zeta_format == 1;
    const VkDeviceSize drow = (static_cast<VkDeviceSize>(s.iw()) * (z16 ? 2u : 4u) + 3) / 4 * 4, srow = z16 ? 0 : (static_cast<VkDeviceSize>(s.iw()) + 3) / 4 * 4;
    const u32 band = static_cast<u32>(std::max<VkDeviceSize>(1, readback_.size / (drow + srow)));
    const std::size_t n = static_cast<std::size_t>(s.width) * s.height;
    std::vector<u8> out(z16 ? n * 2 : n * 5);
    for (u32 y0 = 0; y0 < s.height; y0 += band) {
        const u32 rows = std::min(band, s.height - y0);
        read_back_zeta_rows(s, 0, y0, rows);
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        finish();
        thin_zeta_rows_into(s, static_cast<const u8*>(readback_.mapped), y0, rows, out.data());
    }
    ++stats_.writebacks_by[wb_why_];
    store_zeta(s, out.data());
}

// The scanout target's picture at the internal resolution, read after its write-back into a host buffer: one of two
// per target scanned out (a double or triple buffer's few), retaken only when the target was drawn into (or reloaded)
// since. A target whose memory another engine wrote shows RAM's picture until it is drawn again. The buffer is read in
// place by the frontend: at 6x a picture is 44 MB, and every extra copy of it cost milliseconds a frame. Its copy
// carries a fence of its own and the vblank does not wait for it (D36): the frontend does, while the CPU runs on, and
// the next vblank writes the target's other buffer, never the one the frontend holds.
void VulkanRenderer::capture_display(Surface& s) {
    if (s.sc <= 1 || s.swizzled || s.color_ram_newer) {
        if (debug_tex_log_) std::fprintf(stderr, "texlog: no capture of %08x: sc %u swizzled %d colour from RAM pending %d\n", s.phys, s.sc, s.swizzled ? 1 : 0, s.color_ram_newer ? 1 : 0);
        return;
    }
    const bool convert = init_display_convert();
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(s.iw()) * s.ih() * (convert ? 4u : s.color_bytes());
    Capture* c = nullptr;
    {
        std::lock_guard g(display_mutex_);
        // Captures retired by a scale change go once the frontend lets them go.
        for (auto it = captures_.begin(); it != captures_.end();) {
            if ((*it)->retired && it->get() != held_) {
                destroy_capture(**it);
                it = captures_.erase(it);
            } else {
                ++it;
            }
        }
        Capture* newest = nullptr;
        for (auto& e : captures_)
            if (e->phys == s.phys && (!newest || e->taken > newest->taken)) newest = e.get();
        // A colour table that changed since (a fade) needs the picture again even when the target did not, and so does
        // a video overlay (its next frame).
        if (newest && newest->valid && newest->serial == s.draw_serial && newest->sc == s.sc && newest->width == s.width && newest->height == s.height &&
            newest->pitch == s.pitch && !(convert && display_lut_on_) && !(convert && display_ov_)) {
            shown_ = newest;
            return;
        }
        // The target's other buffer, when it has one the frontend does not hold; else a new one, the least recent of
        // the others going when there are enough.
        for (auto& e : captures_)
            if (e->phys == s.phys && e.get() != newest && e.get() != held_) c = e.get();
        if (!c) {
            if (captures_.size() >= kMaxCaptures) {
                auto oldest = captures_.end();
                for (auto it = captures_.begin(); it != captures_.end(); ++it)
                    if (it->get() != held_ && it->get() != newest && it->get() != shown_ && (oldest == captures_.end() || (*it)->taken < (*oldest)->taken)) oldest = it;
                if (oldest != captures_.end()) {
                    if (oldest->get() == last_capture_) last_capture_ = nullptr;
                    destroy_capture(**oldest);
                    captures_.erase(oldest);
                }
            }
            auto fresh = std::make_unique<Capture>();
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            if (fn_->vkCreateFence(dev_.device(), &fci, nullptr, &fresh->fence) != VK_SUCCESS) return;
            c = fresh.get();
            captures_.push_back(std::move(fresh));
        }
        c->valid = false;
        if (c->submitted) {
            // Taken two of its target's vblanks ago: long landed, normally.
            fn_->vkWaitForFences(dev_.device(), 1, &c->fence, VK_TRUE, ~0ull);
            c->submitted = false;
        }
        if (c->buffer.size < bytes) {
            dev_.destroy_buffer(c->buffer);
            std::string error;
            // Handed to a frontend on the GPU (D38) it stays in device memory; else the host reads it where it lies.
            if (!dev_.create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    handoff_on_ ? vk::BufferMemory::Device : vk::BufferMemory::HostRead, c->buffer, error)) {
                note("scaled picture buffer: " + error);
                if (c == last_capture_) last_capture_ = nullptr;
                destroy_capture(*c);
                captures_.erase(std::find_if(captures_.begin(), captures_.end(), [&](const auto& e) { return e.get() == c; }));
                return;
            }
        }
        c->phys = s.phys;
        c->xrgb = convert;
    }
    // The conversion's descriptor set and colour table were last used by the previous capture, which may still be on
    // its way.
    if (last_capture_ && last_capture_->submitted) fn_->vkWaitForFences(dev_.device(), 1, &last_capture_->fence, VK_TRUE, ~0ull);
    fn_->vkResetFences(dev_.device(), 1, &c->fence);
    end_pass();
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                              &before, 0, nullptr, 0, nullptr);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bool overlaid = false;
    if (convert) {
        // The colour table as 768 words (the identity when the RAMDAC has none), then the pass over the image.
        auto* lut = static_cast<u32*>(dc_lut_.mapped);
        for (u32 i = 0; i < 768; ++i) lut[i] = display_lut_on_ ? display_lut_[i] : i / 3;
        DisplayConvertPush push{};
        push.width = s.iw();
        push.height = s.ih();
        overlaid = overlay_push(s, push);
        VkDescriptorImageInfo img{dc_sampler_, s.color.view, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo bufs[3] = {{c->buffer.buffer, 0, VK_WHOLE_SIZE}, {dc_lut_.buffer, 0, VK_WHOLE_SIZE}, {dc_ov_.buffer, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[4]{};
        for (u32 i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = dc_set_;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            if (i == 0) writes[i].pImageInfo = &img;
            else writes[i].pBufferInfo = &bufs[i - 1];
        }
        fn_->vkUpdateDescriptorSets(dev_.device(), 4, writes, 0, nullptr);
        fn_->vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, dc_pipeline_);
        fn_->vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, dc_layout_, 0, 1, &dc_set_, 0, nullptr);
        fn_->vkCmdPushConstants(cb_, dc_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
        fn_->vkCmdDispatch(cb_, (s.iw() + 15) / 16, (s.ih() + 15) / 16, 1);
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    } else {
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {s.iw(), s.ih(), 1};
        fn_->vkCmdCopyImageToBuffer(cb_, s.color.image, VK_IMAGE_LAYOUT_GENERAL, c->buffer.buffer, 1, &r);
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fn_->vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    }
    // Submitted now, and an empty submission after it signals the capture's fence once everything before it is done.
    submit();
    dev_.submit(0, nullptr, c->fence);
    last_capture_ = c;
    std::lock_guard g(display_mutex_);
    c->submitted = true;
    c->width = s.width;
    c->height = s.height;
    c->pitch = s.pitch;
    c->bpp = s.color_bytes();
    c->sc = s.sc;
    c->serial = s.draw_serial;
    c->overlaid = overlaid;
    c->taken = ++capture_clock_;
    c->valid = true;
    shown_ = c;
}

bool VulkanRenderer::overlay_push(const Surface& s, DisplayConvertPush& push) {
    if (!display_ov_) return false;
    const Overlay& ov = *display_ov_;
    const u32 sc = s.sc, bpp = s.color_bytes();
    // The scanout's first pixel in the target (it may start inside it): the overlay's rectangle is relative to it.
    if (scan_phys_ < s.phys || s.pitch == 0) return false;
    const u32 off = scan_phys_ - s.phys;
    if ((off % s.pitch) % bpp) return false;
    push.ox = (off % s.pitch) / bpp * sc;
    push.oy = off / s.pitch * sc;
    // Nothing to draw, as draw_overlay decides it: the picture is finished as it is.
    const u64 bytes = static_cast<u64>(ov.pitch) * ov.in_h;
    if (ov.in_w == 0 || ov.in_h == 0 || ov.out_w == 0 || ov.out_h == 0 || ov.pitch == 0 || ov.address + bytes > ram_size_) return true;
    // The rectangle and the steps as grab_frame scales them for the CPU (whole steps per scaled pixel), and the products
    // the shader forms kept within 32 bits.
    const u64 x0 = static_cast<u64>(ov.out_x) * sc, y0 = static_cast<u64>(ov.out_y) * sc;
    const u64 w = static_cast<u64>(ov.out_w) * sc, h = static_cast<u64>(ov.out_h) * sc;
    const u32 dsdx = ov.dsdx / sc, dtdy = ov.dtdy / sc;
    if (x0 + w > (1ull << 30) || y0 + h > (1ull << 30) || w * dsdx >= (1ull << 32) || h * dtdy >= (1ull << 32)) return false;
    const VkDeviceSize need = (bytes + 3) / 4 * 4;
    if (dc_ov_.size < need) {
        dev_.destroy_buffer(dc_ov_);  // the last capture to use it has landed (capture_display waited for it)
        std::string error;
        if (!dev_.create_buffer(need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk::BufferMemory::HostWrite, dc_ov_, error)) {
            note("picture overlay source: " + error);
            dev_.create_buffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk::BufferMemory::HostWrite, dc_ov_, error);
            return false;
        }
    }
    std::memcpy(dc_ov_.mapped, ram_ + ov.address, static_cast<std::size_t>(bytes));
    push.ov_on = 1;
    push.x0 = static_cast<u32>(x0);
    push.y0 = static_cast<u32>(y0);
    push.x1 = static_cast<u32>(x0 + w);
    push.y1 = static_cast<u32>(y0 + h);
    push.s_hi = ov.s0 >> 4;
    push.s_lo = (ov.s0 & 0xFu) << 16;
    push.t_hi = ov.t0 >> 4;
    push.t_lo = (ov.t0 & 0xFu) << 16;
    push.dsdx = dsdx;
    push.dtdy = dtdy;
    push.in_w = ov.in_w;
    push.in_h = ov.in_h;
    push.pitch = ov.pitch;
    push.yuy2 = ov.yuy2 ? 1u : 0u;
    push.key_on = ov.color_key ? 1u : 0u;
    push.key = ov.key & 0xFFFFFFu;
    return true;
}


// ── The picture handed to the frontend on the GPU (D38) ──────────────────────────────────────────────────────────

void VulkanRenderer::destroy_handoff(Handoff& h) {
    const auto& f = *fn_;
    if (h.pending) f.vkWaitForFences(dev_.device(), 1, &h.fence, VK_TRUE, ~0ull);
    h.pending = false;
    if (h.view) f.vkDestroyImageView(dev_.device(), h.view, nullptr);
    h.view = VK_NULL_HANDLE;
    dev_.destroy_image(h.image);
    dev_.destroy_buffer(h.staging);
}

bool VulkanRenderer::hand_over(u32 sync_index, const ScaledPicture* pic, u32 width, u32 height, const u32* cpu, std::size_t cpu_pitch, HandedOver& out) {
    const auto& f = *fn_;
    if (width == 0 || height == 0 || (!pic && !cpu) || (pic && !pic->buffer)) return false;
    if (!handoff_pool_) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = dev_.queue_family();
        if (f.vkCreateCommandPool(dev_.device(), &pci, nullptr, &handoff_pool_) != VK_SUCCESS) return false;
    }
    if (handoffs_.size() <= sync_index) handoffs_.resize(sync_index + 1);
    Handoff& h = handoffs_[sync_index];
    if (!h.cb) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = handoff_pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (f.vkAllocateCommandBuffers(dev_.device(), &cai, &h.cb) != VK_SUCCESS || f.vkCreateFence(dev_.device(), &fci, nullptr, &h.fence) != VK_SUCCESS) return false;
    }
    if (h.pending) {
        f.vkWaitForFences(dev_.device(), 1, &h.fence, VK_TRUE, ~0ull);
        f.vkResetFences(dev_.device(), 1, &h.fence);
        h.pending = false;
    }
    std::string error;
    if (h.image.image && (h.image.width != width || h.image.height != height)) destroy_handoff(h);  // the frontend let go of it
    if (!h.image.image) {
        if (!dev_.create_image(VK_FORMAT_B8G8R8A8_UNORM, width, height, 1, 1, 1,
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false, h.image, error,
                               VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)) {
            note("picture for the frontend: " + error);
            return false;
        }
        // XRGB8888 in memory is B, G, R and an unused byte: the view reads alpha as one.
        h.info = VkImageViewCreateInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        h.info.image = h.image.image;
        h.info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        h.info.format = VK_FORMAT_B8G8R8A8_UNORM;
        h.info.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_ONE};
        h.info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (f.vkCreateImageView(dev_.device(), &h.info, nullptr, &h.view) != VK_SUCCESS) {
            destroy_handoff(h);
            return false;
        }
        h.fresh = true;
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * height * 4;
    if (!pic) {
        if (h.staging.size < bytes) {
            dev_.destroy_buffer(h.staging);
            if (!dev_.create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk::BufferMemory::HostWrite, h.staging, error)) {
                note("picture for the frontend: " + error);
                return false;
            }
        }
        for (u32 y = 0; y < height; ++y)
            std::memcpy(static_cast<u8*>(h.staging.mapped) + static_cast<std::size_t>(y) * width * 4, reinterpret_cast<const u8*>(cpu) + y * cpu_pitch, static_cast<std::size_t>(width) * 4);
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    f.vkBeginCommandBuffer(h.cb, &bi);
    // The capture (submitted at the vblank, earlier on the same queue) has landed before the copy reads it; the
    // frontend's earlier reads of this image are done (wait_sync_index).
    VkImageMemoryBarrier to{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    to.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to.oldLayout = h.fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    to.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    to.srcQueueFamilyIndex = to.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to.image = h.image.image;
    to.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkMemoryBarrier src{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    src.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    f.vkCmdPipelineBarrier(h.cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &src, 0, nullptr, 1, &to);
    h.fresh = false;
    VkBufferImageCopy r{};
    r.bufferOffset = pic ? pic->offset : 0;
    r.bufferRowLength = pic ? pic->row_pixels : width;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {width, height, 1};
    f.vkCmdCopyBufferToImage(h.cb, pic ? pic->buffer : h.staging.buffer, h.image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    // Ready for whatever the frontend does with it next on the queue (sampling it, copying it).
    VkImageMemoryBarrier done = to;
    done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    done.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    done.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    f.vkCmdPipelineBarrier(h.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &done);
    f.vkEndCommandBuffer(h.cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &h.cb;
    if (dev_.submit(1, &si, h.fence) != VK_SUCCESS) return false;
    h.pending = true;
    out.view = h.view;
    out.info = &h.info;
    out.width = width;
    out.height = height;
    return true;
}

void VulkanRenderer::destroy_capture(Capture& c) {
    if (c.submitted) fn_->vkWaitForFences(dev_.device(), 1, &c.fence, VK_TRUE, ~0ull);
    if (c.fence) fn_->vkDestroyFence(dev_.device(), c.fence, nullptr);
    c.fence = VK_NULL_HANDLE;
    c.submitted = false;
    dev_.destroy_buffer(c.buffer);
}

bool VulkanRenderer::init_display_convert() {
    if (dc_pipeline_) return true;
    if (dc_failed_) return false;
    dc_failed_ = true;
    const auto& f = *fn_;
    std::string error;
    if (!dev_.create_buffer(768 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk::BufferMemory::HostWrite, dc_lut_, error)) {
        note("picture conversion table: " + error);
        return false;
    }
    if (!dev_.create_buffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vk::BufferMemory::HostWrite, dc_ov_, error)) {  // grows with the overlay
        note("picture overlay source: " + error);
        return false;
    }
    // Pixel (x, y) of the image as the scanout shows it: 8-bit channels through the colour table, the unused byte clear;
    // then the video overlay over it where it covers (and the colour key, when on, matches the finished pixel), each
    // pixel worked out as draw_overlay does it on the CPU: the source texel in 12.20 fixed point from the 12.4 start
    // (the 32-bit sum split so nothing carries out), BT.601 with 8.8 coefficients (yuv601_to_xrgb).
    const CompiledShader cs = compile_glsl(ShaderStage::Compute,
                                           "#version 450\n"
                                           "layout(local_size_x = 16, local_size_y = 16) in;\n"
                                           "layout(binding = 0) uniform sampler2D src;\n"
                                           "layout(std430, binding = 1) writeonly buffer Out { uint px[]; } dst;\n"
                                           "layout(std430, binding = 2) readonly buffer Lut { uint v[768]; } lut;\n"
                                           "layout(std430, binding = 3) readonly buffer Ov { uint v[]; } ov;\n"
                                           "layout(push_constant) uniform P {\n"
                                           "    uint width, height, ov_on, ox, oy, x0, y0, x1, y1;\n"
                                           "    uint s_hi, s_lo, t_hi, t_lo, dsdx, dtdy, in_w, in_h, pitch, yuy2, key_on, key;\n"
                                           "} pc;\n"
                                           "uint ov_byte(uint o) { return (ov.v[o >> 2] >> ((o & 3u) * 8u)) & 0xFFu; }\n"
                                           "uint step_at(uint hi, uint lo, uint b) { return hi + (b >> 20) + ((lo + (b & 0xFFFFFu)) >> 20); }\n"
                                           "void main() {\n"
                                           "    uvec2 p = gl_GlobalInvocationID.xy;\n"
                                           "    if (p.x >= pc.width || p.y >= pc.height) return;\n"
                                           "    uvec3 c = uvec3(round(clamp(texelFetch(src, ivec2(p), 0).rgb, 0.0, 1.0) * 255.0));\n"
                                           "    uint px = (lut.v[c.r * 3u] << 16) | (lut.v[c.g * 3u + 1u] << 8) | lut.v[c.b * 3u + 2u];\n"
                                           "    if (pc.ov_on != 0u && p.x >= pc.ox + pc.x0 && p.y >= pc.oy + pc.y0 && p.x < pc.ox + pc.x1 && p.y < pc.oy + pc.y1 &&\n"
                                           "        (pc.key_on == 0u || px == pc.key)) {\n"
                                           "        uint sx = min(step_at(pc.s_hi, pc.s_lo, (p.x - pc.ox - pc.x0) * pc.dsdx), pc.in_w - 1u);\n"
                                           "        uint sy = min(step_at(pc.t_hi, pc.t_lo, (p.y - pc.oy - pc.y0) * pc.dtdy), pc.in_h - 1u);\n"
                                           "        uint o = sy * pc.pitch + (sx & ~1u) * 2u;\n"
                                           "        int b0 = int(ov_byte(o)), b1 = int(ov_byte(o + 1u)), b2 = int(ov_byte(o + 2u)), b3 = int(ov_byte(o + 3u));\n"
                                           "        int y0 = pc.yuy2 != 0u ? b0 : b1, u = pc.yuy2 != 0u ? b1 : b0, y1 = pc.yuy2 != 0u ? b2 : b3, v = pc.yuy2 != 0u ? b3 : b2;\n"
                                           "        int cc = (((sx & 1u) != 0u ? y1 : y0) - 16) * 298, d = u - 128, e = v - 128;\n"
                                           "        int r = clamp((cc + 409 * e + 128) >> 8, 0, 255);\n"
                                           "        int g = clamp((cc - 100 * d - 208 * e + 128) >> 8, 0, 255);\n"
                                           "        int b = clamp((cc + 516 * d + 128) >> 8, 0, 255);\n"
                                           "        px = (uint(r) << 16) | (uint(g) << 8) | uint(b);\n"
                                           "    }\n"
                                           "    dst.px[p.y * pc.width + p.x] = px;\n"
                                           "}\n",
                                           "picture-to-xrgb");
    if (!cs.ok()) {
        note("picture conversion shader: " + cs.log);
        return false;
    }
    VkShaderModule mod = module(cs.spirv);
    VkDescriptorSetLayoutBinding bindings[4]{};
    for (u32 i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 4;
    dli.pBindings = bindings;
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DisplayConvertPush)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dc_set_layout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (f.vkCreateDescriptorSetLayout(dev_.device(), &dli, nullptr, &dc_set_layout_) != VK_SUCCESS || f.vkCreatePipelineLayout(dev_.device(), &pli, nullptr, &dc_layout_) != VK_SUCCESS ||
        f.vkCreateSampler(dev_.device(), &sci, nullptr, &dc_sampler_) != VK_SUCCESS) {
        note("picture conversion layouts failed");
        f.vkDestroyShaderModule(dev_.device(), mod, nullptr);
        return false;
    }
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = dc_layout_;
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    const VkResult pr = f.vkCreateComputePipelines(dev_.device(), pipeline_cache_, 1, &cpi, nullptr, &dc_pipeline_);
    f.vkDestroyShaderModule(dev_.device(), mod, nullptr);
    if (pr != VK_SUCCESS || f.vkCreateDescriptorPool(dev_.device(), &dpi, nullptr, &dc_pool_) != VK_SUCCESS) {
        note("picture conversion pipeline failed");
        if (dc_pipeline_) f.vkDestroyPipeline(dev_.device(), dc_pipeline_, nullptr);
        dc_pipeline_ = VK_NULL_HANDLE;
        return false;
    }
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dc_pool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &dc_set_layout_;
    if (f.vkAllocateDescriptorSets(dev_.device(), &dai, &dc_set_) != VK_SUCCESS) {
        note("picture conversion descriptor set failed");
        f.vkDestroyPipeline(dev_.device(), dc_pipeline_, nullptr);
        dc_pipeline_ = VK_NULL_HANDLE;
        return false;
    }
    dc_failed_ = false;
    return true;
}

bool VulkanRenderer::scaled_picture(u32 phys, u32 pitch, u32 bpp, u32 w, u32 h, ScaledPicture& out) const {
    std::lock_guard g(display_mutex_);
    const Capture* c = shown_;
    if (!c || !c->valid || phys < c->phys || c->pitch != pitch || c->bpp != bpp) return false;
    const u32 off = phys - c->phys;
    if ((off % pitch) % bpp) return false;
    const u32 x0 = (off % pitch) / bpp, y0 = off / pitch;
    if (x0 + w > c->width || y0 + h > c->height) return false;
    const u32 px = c->xrgb ? 4u : bpp;  // bytes a pixel in the buffer
    out.sc = c->sc;
    out.xrgb = c->xrgb;
    out.overlaid = c->overlaid;
    out.pitch = c->width * c->sc * px;
    out.offset = static_cast<VkDeviceSize>(y0) * c->sc * out.pitch + static_cast<VkDeviceSize>(x0) * c->sc * px;
    out.pixels = c->buffer.mapped ? static_cast<const u8*>(c->buffer.mapped) + out.offset : nullptr;
    out.buffer = c->buffer.buffer;
    out.row_pixels = c->width * c->sc;
    out.ready = c->submitted ? c->fence : VK_NULL_HANDLE;
    held_ = c;
    return true;
}

void VulkanRenderer::picture_ready(const ScaledPicture& p) const {
    if (p.ready) fn_->vkWaitForFences(dev_.device(), 1, &p.ready, VK_TRUE, ~0ull);
}

void VulkanRenderer::picture_done() const {
    std::lock_guard g(display_mutex_);
    held_ = nullptr;
}

bool VulkanRenderer::scaled_scanout(u32 phys, u32 pitch, u32 bpp, bool rgb565, u32 w, u32 h, std::vector<u32>& xrgb, u32& scale) const {
    ScaledPicture p;
    if (!scaled_picture(phys, pitch, bpp, w, h, p)) return false;
    picture_ready(p);
    struct Done {
        const VulkanRenderer* r;
        ~Done() { r->picture_done(); }
    } done{this};
    const u32 fw = w * p.sc, fh = h * p.sc;
    xrgb.resize(static_cast<std::size_t>(fw) * fh);
    if (p.xrgb) {
        for (u32 y = 0; y < fh; ++y) std::memcpy(xrgb.data() + static_cast<std::size_t>(y) * fw, p.pixels + static_cast<std::size_t>(y) * p.pitch, static_cast<std::size_t>(fw) * 4);
    } else {
        scanout_to_xrgb(p.pixels, p.pitch, bpp, rgb565, fw, fh, reinterpret_cast<u8*>(xrgb.data()));
        for (u32& px : xrgb) px &= 0x00FFFFFFu;
    }
    scale = p.sc;
    return true;
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

std::vector<u32> VulkanRenderer::spirv_cached(bool vertex, const std::string& glsl) {
    if (cache_dir_.empty()) return {};
    const std::string keyed = std::string(vertex ? "vs\n" : "fs\n") + glsl;
    const u64 check = fnv1a(keyed, kCheckSeed);
    std::vector<char> data;
    if (!read_file(cache_dir_ / "spirv" / std::format("{:016x}.spv", fnv1a(keyed, kNameSeed)), data) || data.size() <= 20 || data.size() % 4 != 0) return {};
    u32 head[4];
    std::memcpy(head, data.data(), sizeof head);
    u32 first = 0;
    std::memcpy(&first, data.data() + sizeof head, 4);
    if (head[0] != kSpvFileMagic || head[1] != kSpvFileVersion || (static_cast<u64>(head[3]) << 32 | head[2]) != check || first != 0x07230203u) return {};
    std::vector<u32> spirv((data.size() - sizeof head) / 4);
    std::memcpy(spirv.data(), data.data() + sizeof head, spirv.size() * 4);
    ++stats_.shaders_from_disk;
    return spirv;
}

std::vector<u32> VulkanRenderer::spirv_compile(bool vertex, const std::string& glsl, const char* name, std::string& log) const {
    CompiledShader c = compile_glsl(vertex ? ShaderStage::Vertex : ShaderStage::Fragment, glsl, name);
    log = c.log;
    if (c.ok() && !cache_dir_.empty()) {
        const std::string keyed = std::string(vertex ? "vs\n" : "fs\n") + glsl;
        const u64 check = fnv1a(keyed, kCheckSeed);
        std::vector<u32> out = {kSpvFileMagic, kSpvFileVersion, static_cast<u32>(check), static_cast<u32>(check >> 32)};
        out.insert(out.end(), c.spirv.begin(), c.spirv.end());
        write_file(cache_dir_ / "spirv" / std::format("{:016x}.spv", fnv1a(keyed, kNameSeed)), out.data(), out.size() * 4);
    }
    return std::move(c.spirv);
}

std::vector<u32> VulkanRenderer::spirv_for(bool vertex, const std::string& glsl, const char* name, std::string& log) {
    if (std::vector<u32> spirv = spirv_cached(vertex, glsl); !spirv.empty()) return spirv;
    return spirv_compile(vertex, glsl, name, log);
}

// ── background compiles ──────────────────────────────────────────────────────────────────────────────────────────

void VulkanRenderer::set_async_compile(bool on) {
    async_compile_ = on;
    if (!on || !workers_.empty()) return;
    // A few workers: a scene change brings a burst of new shaders, and each compile is single-threaded.
    const u32 n = std::clamp(std::thread::hardware_concurrency() / 4u, 1u, 3u);
    for (u32 i = 0; i < n; ++i)
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock lock(work_mutex_);
                    work_cv_.wait(lock, [this] { return work_stop_ || !work_.empty(); });
                    if (work_stop_) return;
                    job = std::move(work_.front());
                    work_.pop_front();
                }
                job();
            }
        });
}

void VulkanRenderer::submit(std::function<void()> job) {
    {
        std::lock_guard lock(work_mutex_);
        work_.push_back(std::move(job));
    }
    work_cv_.notify_one();
}

void VulkanRenderer::stop_workers() {
    {
        std::lock_guard lock(work_mutex_);
        work_stop_ = true;
        work_.clear();
    }
    work_cv_.notify_all();
    for (auto& w : workers_) w.join();
    workers_.clear();
}

bool VulkanRenderer::shader_pending(bool vertex, const std::string& key) const {
    const auto& pending = vertex ? vs_pending_ : fs_pending_;
    const auto it = pending.find(key);
    return it != pending.end() && !it->second->done.load(std::memory_order_acquire);
}

bool VulkanRenderer::spirv_ready(bool vertex, const std::string& key, const std::string& glsl, const char* name, std::vector<u32>& spirv, std::string& log) {
    if (!async_compile_) {
        spirv = spirv_for(vertex, glsl, name, log);
        return true;
    }
    auto& pending = vertex ? vs_pending_ : fs_pending_;
    if (auto it = pending.find(key); it != pending.end()) {
        if (!it->second->done.load(std::memory_order_acquire)) return false;
        spirv = std::move(it->second->spirv);
        log = std::move(it->second->log);
        pending.erase(it);
        return true;
    }
    spirv = spirv_cached(vertex, glsl);
    if (!spirv.empty()) return true;
    auto job = std::make_shared<PendingShader>();
    pending.emplace(key, job);
    ++stats_.compiles_queued;
    submit([this, job, vertex, glsl, label = std::string(name)] {
        job->spirv = spirv_compile(vertex, glsl, label.c_str(), job->log);
        job->done.store(true, std::memory_order_release);
    });
    return false;
}

}  // namespace mv::pgraph
