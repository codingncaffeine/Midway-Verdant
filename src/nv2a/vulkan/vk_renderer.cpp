#include "nv2a/vulkan/vk_renderer.h"

#include <algorithm>
#include <chrono>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <format>

#include "nv2a/shaders/glsl_compiler.h"

namespace mv::pgraph {

namespace {

constexpr VkDeviceSize kVertexRingBytes = 32u << 20, kIndexRingBytes = 8u << 20, kUniformRingBytes = 8u << 20, kStagingBytes = 16u << 20;
constexpr u32 kVertexStride = kAttributes * 16;
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
    for (auto& [k, s] : surfaces_) {
        destroy_surface_views(*s);
        if (s->framebuffer) f.vkDestroyFramebuffer(dev_.device(), s->framebuffer, nullptr);
        dev_.destroy_image(s->color);
        dev_.destroy_image(s->depth);
    }
    for (auto& [k, p] : passes_) f.vkDestroyRenderPass(dev_.device(), p, nullptr);
    for (auto& [k, m] : vs_cache_) f.vkDestroyShaderModule(dev_.device(), m, nullptr);
    for (auto& [k, m] : fs_cache_) f.vkDestroyShaderModule(dev_.device(), m, nullptr);
    for (auto& [k, p] : pipelines_) f.vkDestroyPipeline(dev_.device(), p, nullptr);
    for (auto& [k, t] : textures_) dev_.destroy_image(t->image);
    for (auto& [k, s] : samplers_) f.vkDestroySampler(dev_.device(), s, nullptr);
    if (dummy_sampler_) f.vkDestroySampler(dev_.device(), dummy_sampler_, nullptr);
    dev_.destroy_image(dummy2d_);
    dev_.destroy_image(dummycube_);
    dev_.destroy_image(dummy3d_);
    if (descriptor_pool_) f.vkDestroyDescriptorPool(dev_.device(), descriptor_pool_, nullptr);
    if (pipeline_layout_) f.vkDestroyPipelineLayout(dev_.device(), pipeline_layout_, nullptr);
    if (set_layout_) f.vkDestroyDescriptorSetLayout(dev_.device(), set_layout_, nullptr);
    dev_.destroy_buffer(vertex_ring_);
    dev_.destroy_buffer(index_ring_);
    dev_.destroy_buffer(uniform_ring_);
    dev_.destroy_buffer(staging_);
    dev_.destroy_buffer(readback_);
    if (fence_) f.vkDestroyFence(dev_.device(), fence_, nullptr);
    if (cb_) f.vkFreeCommandBuffers(dev_.device(), dev_.pool(), 1, &cb_);
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

    if (!dev_.create_buffer(kVertexRingBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk::BufferMemory::HostWrite, vertex_ring_, error)) return false;
    if (!dev_.create_buffer(kIndexRingBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, vk::BufferMemory::HostWrite, index_ring_, error)) return false;
    if (!dev_.create_buffer(kUniformRingBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, vk::BufferMemory::HostWrite, uniform_ring_, error)) return false;
    if (!dev_.create_buffer(kStagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk::BufferMemory::HostWrite, staging_, error)) return false;
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
    if (f.vkCreateDescriptorPool(dev_.device(), &dpi, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        error = "vkCreateDescriptorPool failed";
        return false;
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
        std::memset(staging_.mapped, 0xFF, 64);
        cb = dev_.begin_once();
        for (vk::Image* img : {&dummy2d_, &dummycube_, &dummy3d_}) {
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, img->layers};
            region.imageExtent = {1, 1, img->depth};
            f.vkCmdCopyBufferToImage(cb, staging_.buffer, img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        }
        dev_.finish_once(cb);
    }
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = 1.0f;
    f.vkCreateSampler(dev_.device(), &sci, nullptr, &dummy_sampler_);

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = dev_.pool();
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    f.vkAllocateCommandBuffers(dev_.device(), &cai, &cb_);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    f.vkCreateFence(dev_.device(), &fci, nullptr, &fence_);
    begin_commands();
    return true;
}

void VulkanRenderer::begin_commands() {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    fn_->vkBeginCommandBuffer(cb_, &bi);
    recording_ = true;
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

void VulkanRenderer::flush() {
    const auto& f = *fn_;
    if (!recording_) return;
    Timed timed(stats_.ns_flush);
    end_pass();
    f.vkEndCommandBuffer(cb_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb_;
    f.vkResetFences(dev_.device(), 1, &fence_);
    f.vkQueueSubmit(dev_.queue(), 1, &si, fence_);
    f.vkWaitForFences(dev_.device(), 1, &fence_, VK_TRUE, ~0ull);
    ++stats_.flushes;
    f.vkResetCommandBuffer(cb_, 0);
    begin_commands();
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
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 2;
    rpi.pAttachments = att;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sub;
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
        if (s0.dirty) write_back(s0);
        destroy_surface_views(s0);
        if (s0.framebuffer) fn_->vkDestroyFramebuffer(dev_.device(), s0.framebuffer, nullptr);
        dev_.destroy_image(s0.color);
        dev_.destroy_image(s0.depth);
        surfaces_.erase(it);
        stage_cache_.fill({});  // a stage may point at the erased target's entries
    }
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
    // an existing picture).
    flush();
    {
        VkCommandBuffer cb = dev_.begin_once();
        dev_.transition(cb, t.color, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        dev_.transition(cb, t.depth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        dev_.finish_once(cb);
    }
    {
        // Upload the current RAM contents as the starting picture.
        const u32 bpp = t.color_bytes();
        std::vector<u8> pixels(static_cast<std::size_t>(width) * height * bpp, 0);
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const u32 off = t.swizzled ? swizzled_offset(x, y, 0, t.log_w, t.log_h, 0) * bpp : y * t.pitch + x * bpp;
                if (static_cast<u64>(phys) + off + bpp <= ram_size_) std::memcpy(&pixels[(static_cast<std::size_t>(y) * width + x) * bpp], ram_ + phys + off, bpp);
            }
        }
        if (pixels.size() <= kStagingBytes) {
            std::memcpy(staging_.mapped, pixels.data(), pixels.size());
            VkCommandBuffer cb = dev_.begin_once();
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {width, height, 1};
            fn_->vkCmdCopyBufferToImage(cb, staging_.buffer, t.color.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            dev_.finish_once(cb);
        }
    }
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

bool VulkanRenderer::ensure_pass(Surface& t) {
    if (pass_open_ && pass_target_ == &t) return true;
    end_pass();
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

void VulkanRenderer::read_back(const vk::Image& image, VkImageAspectFlags aspect, u32 width, u32 height) {
    Timed timed_gpu(stats_.ns_write_back_gpu);
    VkCommandBuffer cb = dev_.begin_once();
    VkBufferImageCopy region{};
    region.imageSubresource = {aspect, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    fn_->vkCmdCopyImageToBuffer(cb, image.image, VK_IMAGE_LAYOUT_GENERAL, readback_.buffer, 1, &region);
    dev_.finish_once(cb);
}

void VulkanRenderer::write_back(Surface& s) {
    if (!s.dirty) return;
    Timed timed(stats_.ns_write_back);
    if (pass_target_ == &s) end_pass();
    flush();
    const u32 bpp = s.color_bytes();
    const std::size_t bytes = static_cast<std::size_t>(s.width) * s.height * bpp;
    if (bytes > kStagingBytes) {
        note("surface too large to write back");
        return;
    }
    read_back(s.color, VK_IMAGE_ASPECT_COLOR_BIT, s.width, s.height);
    const u8* pixels = static_cast<const u8*>(readback_.mapped);
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
    if (!s.zeta_dirty) return;
    s.zeta_dirty = false;
    if (s.zeta_format == 0 || s.zeta_phys == 0) return;  // no depth surface bound (an offset of zero is never a real one)
    if (pass_target_ == &s) end_pass();
    flush();
    const bool z16 = s.zeta_format == 1;
    const u32 bpp = z16 ? 2u : 4u;  // D16; a 24-bit or float depth copies as a 32-bit word
    const std::size_t bytes = static_cast<std::size_t>(s.width) * s.height * bpp;
    if (bytes > kStagingBytes) {
        note("depth surface too large to write back");
        return;
    }
    if (static_cast<u64>(s.zeta_phys) + s.zeta_bytes() > ram_size_) return;
    read_back(s.depth, VK_IMAGE_ASPECT_DEPTH_BIT, s.width, s.height);
    const u8* px = static_cast<const u8*>(readback_.mapped);
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
        // Z24S8: the depth above a stencil byte left zero, converted a row at a time.
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
                row[x] = depth24 << 8;
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
        if (s->dirty) write_back(*s);
        if (s->zeta_dirty) write_back_zeta(*s);
    }
}

void VulkanRenderer::write_back_range(u32 phys, u32 len) {
    const u64 b0 = phys, b1 = static_cast<u64>(phys) + len;
    for (auto& [k, s] : surfaces_) {
        const u64 a0 = s->phys, a1 = static_cast<u64>(s->phys) + s->bytes();
        if (s->dirty && a0 < b1 && b0 < a1) write_back(*s);
        const u64 z0 = s->zeta_phys, z1 = static_cast<u64>(s->zeta_phys) + s->zeta_bytes();
        if (s->zeta_dirty && z0 < b1 && b0 < z1) write_back_zeta(*s);
    }
}

// A GPU-side write to guest memory (a blit): decoded textures over it are stale, and so is any target's image
// the guest has not drawn into since.
void VulkanRenderer::invalidate(u32 phys, u32 len) {
    invalidate_textures(phys, len);
    const u64 b0 = phys, b1 = static_cast<u64>(phys) + len;
    for (auto& [k, s] : surfaces_) {
        const u64 a0 = s->phys, a1 = a0 + s->bytes(), z0 = s->zeta_phys, z1 = z0 + s->zeta_bytes();
        if ((a0 < b1 && b0 < a1) || (z0 < b1 && b0 < z1)) s->ram_newer = true;
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
    const bool fog = (s.reg(0x02A4) & 1u) != 0;
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
    const CompiledShader c = compile_glsl(ShaderStage::Vertex, t.glsl, "kelvin-vp");
    if (!c.ok()) {
        error = "vertex program GLSL: " + c.log;
        return VK_NULL_HANDLE;
    }
    VkShaderModule m = module(c.spirv);
    vs_cache_[key] = m;
    remember(m);
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
    const CompiledShader c = compile_glsl(ShaderStage::Fragment, t.glsl, "kelvin-combiners");
    if (!c.ok()) {
        error = "combiner GLSL: " + c.log;
        return VK_NULL_HANDLE;
    }
    VkShaderModule m = module(c.spirv);
    fs_cache_[k] = m;
    fs_kinds_[k] = t.samplers;
    kinds = t.samplers;
    remember(m);
    return m;
}

VkPipeline VulkanRenderer::pipeline(const KelvinState& s, VkShaderModule vs, VkShaderModule fs, VkPrimitiveTopology topology, Surface& t, std::string& error) {
    Timed timed(stats_.ns_pipeline);
    struct RasterKey {
        VkShaderModule vs, fs;
        VkRenderPass pass;
        u32 topology, blend_enable, sfactor, dfactor, equation, depth_test, depth_func, depth_write, cull_enable, cull_face, front_face,
            color_mask, stencil_enable, stencil_func, stencil_mask, stencil_write_mask, op_fail, op_zfail, op_zpass;
    } k{vs, fs, t.pass, static_cast<u32>(topology) | (debug_nocull_ ? 0x100u : 0u) | (debug_nodepth_ ? 0x200u : 0u) | (debug_noblend_ ? 0x400u : 0u), s.blend(), s.reg(mthd::kBlendSfactor), s.reg(mthd::kBlendDfactor), s.reg(mthd::kBlendEquation),
        s.depth_test(), s.reg(mthd::kDepthFunc), s.depth_write(), s.cull_face(), s.reg(mthd::kCullFace), s.reg(mthd::kFrontFace), s.color_mask(),
        s.stencil_test(), s.reg(mthd::kStencilFunc), s.reg(mthd::kStencilFuncMask), s.reg(mthd::kStencilMask), s.reg(mthd::kStencilOpFail),
        s.reg(mthd::kStencilOpZfail), s.reg(mthd::kStencilOpZpass)};
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

    VkVertexInputBindingDescription binding{0, kVertexStride, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[kAttributes]{};
    for (u32 i = 0; i < kAttributes; ++i) attrs[i] = {i, 0, VK_FORMAT_R32G32B32A32_SFLOAT, i * 16};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
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
    const VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 4;
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
    if (VkResult r = fn_->vkCreateGraphicsPipelines(dev_.device(), VK_NULL_HANDLE, 1, &gpi, nullptr, &p); r != VK_SUCCESS) {
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
    if (const TextureEntry* e = surface_texture(ts, phys)) {
        sc = StageCache{regs, phys, length, e};
        return e;
    }
    // A render target the guest now samples: its pixels must be in RAM first — the colour image, or the depth
    // buffer when the texture is a shadow map.
    for (auto& [k, surf] : surfaces_) {
        const u64 a0 = surf->phys, a1 = static_cast<u64>(surf->phys) + surf->bytes();
        if (surf->dirty && a0 < static_cast<u64>(phys) + length && static_cast<u64>(phys) < a1) write_back(*surf);
        const u64 z0 = surf->zeta_phys, z1 = static_cast<u64>(surf->zeta_phys) + surf->zeta_bytes();
        if (surf->zeta_dirty && z0 < static_cast<u64>(phys) + length && static_cast<u64>(phys) < z1) write_back_zeta(*surf);
    }
    if (same && sc.entry->valid && !sc.entry->surface) return sc.entry;
    struct Ident {
        u32 phys, format, control1, rect, palette;
    } id{phys, ts.format, ts.control1, ts.image_rect, ts.palette};
    std::string key(reinterpret_cast<const char*>(&id), sizeof id);
    auto it = textures_.find(key);
    if (it != textures_.end() && it->second->valid) {
        sc = StageCache{regs, phys, length, it->second.get()};
        return it->second.get();
    }

    DecodedTexture d;
    const u8* palette = nullptr;
    if (ts.color_format() == 0x0B) {
        const DmaObject pdma = handles_.resolve(s.dma_handle((ts.palette & 1u) ? mthd::kDmaB : mthd::kDmaA));
        palette = MemoryView{ram_, ram_size_}.at(pdma.base + (ts.palette & 0xFFFFFFC0u), 256 * 4);
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
        std::fprintf(stderr, "texlog: decode %08x format %02x %ux%u levels %u faces %u%s\n", phys, ts.color_format(), d.width, d.height, d.levels, d.faces, target ? " (a render target)" : "");
    }
    const VkFormat format = d.pixel == HostPixel::Rgba8 ? VK_FORMAT_R8G8B8A8_UNORM : (d.pixel == HostPixel::R16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT);
    auto entry = std::make_unique<TextureEntry>();
    if (it != textures_.end()) {
        entry = std::move(it->second);
        textures_.erase(it);
        dev_.destroy_image(entry->image);
    }
    if (!dev_.create_image(format, d.width, d.height, d.depth, d.levels, d.faces, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT, d.faces == 6, entry->image, error))
        return nullptr;
    flush();
    VkCommandBuffer cb = dev_.begin_once();
    dev_.transition(cb, entry->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    dev_.finish_once(cb);
    // Upload each face and level through the staging buffer.
    for (u32 face = 0; face < d.faces; ++face) {
        u32 w = d.width, h = d.height, dp = d.depth;
        for (u32 level = 0; level < d.levels; ++level) {
            const auto& img = d.image(face, level);
            if (img.size() > kStagingBytes) {
                error = "texture level too large for the staging buffer";
                return nullptr;
            }
            std::memcpy(staging_.mapped, img.data(), img.size());
            cb = dev_.begin_once();
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, face, 1};
            region.imageExtent = {w, h, dp};
            fn_->vkCmdCopyBufferToImage(cb, staging_.buffer, entry->image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            dev_.finish_once(cb);
            w = std::max(w >> 1, 1u);
            h = std::max(h >> 1, 1u);
            if (d.depth > 1) dp = std::max(dp >> 1, 1u);
        }
    }
    entry->phys = phys;
    entry->length = length;
    entry->valid = true;
    entry->width = d.width;
    entry->height = d.height;
    entry->linear = format_info(ts.color_format()).linear;
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
const VulkanRenderer::TextureEntry* VulkanRenderer::surface_texture(const TextureStage& ts, u32 phys) {
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
        if (color) {
            if (s.phys != phys || !(s.dirty || recent) || s.color_bytes() != 4 || color_vk_format(s.color_format) != VK_FORMAT_B8G8R8A8_UNORM) continue;
            if (linear && ts.pitch() != s.pitch) continue;
        } else {
            if (s.zeta_phys != phys || !(s.zeta_dirty || recent) || (s.zeta_format == 1) != depth16) continue;
            if (!(depth16 ? depth16_sampled_ : depth24_sampled_) || (linear && ts.pitch() != s.zeta_pitch)) continue;
        }
        if (pass_open_ && pass_target_ == &s) return nullptr;  // drawn into by the open pass: through RAM as before
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
    s.x8_view = s.depth_view = VK_NULL_HANDLE;
    s.color_tex = s.color_x8_tex = s.depth_tex = TextureEntry{};
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
    auto fits = [](const vk::Buffer& ring, VkDeviceSize cursor, VkDeviceSize bytes, VkDeviceSize alignment) {
        return (cursor + alignment - 1) / alignment * alignment + bytes <= ring.size;
    };
    const VkDeviceSize ualign = std::max<VkDeviceSize>(dev_.limits().minUniformBufferOffsetAlignment, 16);
    const VkDeviceSize uniforms = (vsh::kUniformBytes + ualign - 1) / ualign * ualign + sizeof(psh::Uniforms);
    return fits(vertex_ring_, vertex_cursor_, vertex_bytes, 16) && (index_bytes == 0 || fits(index_ring_, index_cursor_, index_bytes, 4)) &&
           fits(uniform_ring_, uniform_cursor_, uniforms, ualign) && descriptor_sets_used_ < kMaxDescriptorSets;
}

bool VulkanRenderer::ring_alloc(vk::Buffer& ring, VkDeviceSize& cursor, VkDeviceSize bytes, VkDeviceSize alignment, VkDeviceSize& offset) {
    const VkDeviceSize start = (cursor + alignment - 1) / alignment * alignment;
    if (start + bytes > ring.size) return false;
    offset = start;
    cursor = start + bytes;
    return true;
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
    if (bits & 0xF0u) {
        const u32 c = s.clear_color();
        att[n].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        att[n].colorAttachment = 0;
        att[n].clearValue.color.float32[0] = static_cast<float>((c >> 16) & 0xFFu) / 255.0f;
        att[n].clearValue.color.float32[1] = static_cast<float>((c >> 8) & 0xFFu) / 255.0f;
        att[n].clearValue.color.float32[2] = static_cast<float>(c & 0xFFu) / 255.0f;
        att[n].clearValue.color.float32[3] = static_cast<float>(c >> 24) / 255.0f;
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
    if (n == 0) return;
    VkClearRect rect{};
    rect.rect = {{static_cast<i32>(x0), static_cast<i32>(y0)}, {x1 - x0 + 1, y1 - y0 + 1}};
    rect.layerCount = 1;
    fn_->vkCmdClearAttachments(cb_, n, att, 1, &rect);
}

void VulkanRenderer::draw(const KelvinState& s, const DrawBatch& batch) {
    Timed timed(stats_.ns_draw);
    ++stats_.draws;
    const u32 draw_id = frame_draws_++;
    if (debug_draw_id_)
        std::fprintf(stderr, "drawid %u: surface %08x, stages %05x, blend %u, %zu indices, %zu ranges, %zu inline words, %u immediate\n", draw_id, s.surface().color_offset,
                     s.reg(mthd::kShaderStageProgram), s.blend() ? 1u : 0u, batch.indices.size(), batch.ranges.size(), batch.inline_words.size(), batch.immediate_count);
    if (debug_skip_blend_ && s.blend()) return;  // MV_RENDER_DEBUG=skipblend: the opaque passes alone
    std::string error;
    Surface* t = target_for(s);
    if (!t) {
        ++stats_.skipped;
        return;
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
    VkPipeline p = pipeline(s, vs, fs, topology, *t, error);
    if (!p) {
        note(error);
        ++stats_.skipped;
        return;
    }

    // Vertex data: every attribute as a vec4; one without per-vertex data reads its constant value.
    t_part = std::chrono::steady_clock::now();
    VkDeviceSize vbytes = static_cast<VkDeviceSize>(v.count) * kVertexStride, voffset = 0;
    // The rings and the descriptor pool hold one batch of draws. A draw that no longer fits submits the batch so far
    // and starts the rings over, so a frame can hold any number of draws (only a single draw larger than a ring is
    // refused below).
    if (!fits_in_batch(vbytes, indices.size() * 4)) {
        flush_frame();
        ++stats_.batch_flushes;
    }
    if (!ring_alloc(vertex_ring_, vertex_cursor_, vbytes, 16, voffset)) {
        note("vertex ring full for this frame");
        ++stats_.skipped;
        return;
    }
    float* vdst = reinterpret_cast<float*>(static_cast<u8*>(vertex_ring_.mapped) + voffset);
    for (u32 n = 0; n < v.count; ++n) {
        for (u32 a = 0; a < kAttributes; ++a) {
            float* d = vdst + (static_cast<std::size_t>(n) * kAttributes + a) * 4;
            if (v.enabled & (1u << a)) std::memcpy(d, &v.attribute[a][static_cast<std::size_t>(n) * 4], 16);
            else std::memcpy(d, v.constant[a].data(), 16);
        }
    }
    VkDeviceSize ioffset = 0;
    if (!indices.empty()) {
        if (!ring_alloc(index_ring_, index_cursor_, indices.size() * 4, 4, ioffset)) {
            note("index ring full for this frame");
            ++stats_.skipped;
            return;
        }
        std::memcpy(static_cast<u8*>(index_ring_.mapped) + ioffset, indices.data(), indices.size() * 4);
    }

    // Uniforms.
    VkDeviceSize uoffset_vs = 0, uoffset_ps = 0;
    const VkDeviceSize ualign = std::max<VkDeviceSize>(dev_.limits().minUniformBufferOffsetAlignment, 16);
    if (!ring_alloc(uniform_ring_, uniform_cursor_, vsh::kUniformBytes, ualign, uoffset_vs) ||
        !ring_alloc(uniform_ring_, uniform_cursor_, sizeof(psh::Uniforms), ualign, uoffset_ps)) {
        note("uniform ring full for this frame");
        ++stats_.skipped;
        return;
    }
    {
        float* u = reinterpret_cast<float*>(static_cast<u8*>(uniform_ring_.mapped) + uoffset_vs);
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
        std::memcpy(static_cast<u8*>(uniform_ring_.mapped) + uoffset_ps, &pu, sizeof pu);
    }

    stats_.ns_upload += ns_since(t_part);
    t_part = std::chrono::steady_clock::now();
    // Descriptors.
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = descriptor_pool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &set_layout_;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (fn_->vkAllocateDescriptorSets(dev_.device(), &dai, &set) != VK_SUCCESS) {
        note("descriptor pool exhausted for this frame");
        ++stats_.skipped;
        return;
    }
    ++descriptor_sets_used_;
    VkDescriptorBufferInfo bufs[2] = {{uniform_ring_.buffer, uoffset_vs, vsh::kUniformBytes}, {uniform_ring_.buffer, uoffset_ps, sizeof(psh::Uniforms)}};
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

    stats_.ns_descriptors += ns_since(t_part);
    t_part = std::chrono::steady_clock::now();
    // Record.
    ensure_pass(*t);
    if (s.color_mask() != 0) t->dirty = true;
    ++t->draws_this_frame;
    fn_->vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    fn_->vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &set, 0, nullptr);
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
    const VkDeviceSize vb_offset = voffset;
    fn_->vkCmdBindVertexBuffers(cb_, 0, 1, &vertex_ring_.buffer, &vb_offset);
    if (!indices.empty()) {
        fn_->vkCmdBindIndexBuffer(cb_, index_ring_.buffer, ioffset, VK_INDEX_TYPE_UINT32);
        fn_->vkCmdDrawIndexed(cb_, static_cast<u32>(indices.size()), 1, 0, 0, 0);
    } else {
        fn_->vkCmdDraw(cb_, v.count, 1, 0, 0);
    }
    stats_.ns_record += ns_since(t_part);
}

void VulkanRenderer::flush_frame() {
    flush();
    vertex_cursor_ = index_cursor_ = uniform_cursor_ = 0;
    fn_->vkResetDescriptorPool(dev_.device(), descriptor_pool_, 0);
    descriptor_sets_used_ = 0;
}

bool VulkanRenderer::write_back_at(u32 phys) {
    for (auto& [k, s] : surfaces_) {
        if (s->phys != phys) continue;
        if (s->dirty) write_back(*s);
        return true;
    }
    return false;
}

void VulkanRenderer::end_frame() {
    frame_draws_ = 0;
    ++frame_index_;
    flush();
    // Colour reaches RAM every frame (the scanout and the CPU read it); depth only when a texture samples it.
    for (auto& [k, s] : surfaces_)
        if (s->dirty) write_back(*s);
    vertex_cursor_ = index_cursor_ = uniform_cursor_ = 0;
    fn_->vkResetDescriptorPool(dev_.device(), descriptor_pool_, 0);
    descriptor_sets_used_ = 0;
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
