#pragma once
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pgraph/texture.h"
#include "nv2a/shaders/psh_translate.h"
#include "nv2a/shaders/vsh_translate.h"
#include "nv2a/vulkan/vk_device.h"

namespace mv::pgraph {

// The Vulkan renderer: a RenderSink that turns the Kelvin state machine's clears and draws into Vulkan work.
// Render targets live in device images keyed by their guest address and are written back into guest RAM when
// a texture reads them or the frame ends (the console's memory is unified: a render target is RAM). Textures are
// decoded from RAM through the texture decoder and cached until the RAM they came from is written.
class VulkanRenderer : public RenderSink {
public:
    VulkanRenderer(vk::Device& device, const HandleResolver& handles, u8* ram, u32 ram_size);
    ~VulkanRenderer() override;
    bool init(std::string& error);

    void clear(const KelvinState& state, u32 clear_bits) override;
    void draw(const KelvinState& state, const DrawBatch& batch) override;
    // Submits the frame's work, waits, writes every rendered surface back into guest RAM.
    void end_frame();
    // The live machine's frame boundary: submit the work and recycle the rings, but leave the surfaces on the
    // device — they reach RAM when a texture samples them, and the scanout one at vblank (write_back_at).
    void flush_frame();
    bool write_back_at(u32 phys);
    // Guest RAM in [phys, phys + len) was written: textures decoded from it are stale.
    void invalidate(u32 phys, u32 len);

    struct SurfaceInfo {
        u32 phys{0}, width{0}, height{0}, pitch{0}, bytes_per_pixel{0}, color_format{0};
        bool swizzled{false};
        bool rendered_this_frame{false};
        u64 draws_this_frame{0}, clears_this_frame{0};
    };
    std::vector<SurfaceInfo> surfaces() const;
    // The pixels of a surface as RGBA8 rows, read from guest RAM (call after end_frame).
    bool surface_rgba(const SurfaceInfo& s, std::vector<u8>& rgba) const;

    struct Stats {
        u64 draws{0}, clears{0}, skipped{0}, pipelines{0}, shaders{0}, textures_decoded{0}, writebacks{0}, flushes{0};
        // Where the time goes, in nanoseconds: the whole draw and its parts (the texture part includes the
        // write-backs a sample forces; the shader parts are the compiles, a lookup costs nothing measurable); the
        // submits that wait; the copies back to RAM (and the device copy inside them).
        u64 ns_draw{0}, ns_fetch{0}, ns_texture{0}, ns_pipeline{0}, ns_flush{0}, ns_write_back{0}, ns_write_back_gpu{0};
        u64 ns_vs{0}, ns_fs{0}, ns_indices{0}, ns_upload{0}, ns_descriptors{0}, ns_record{0};
        u64 vs_memo_hits{0}, vs_cache_hits{0}, fs_memo_hits{0}, fs_cache_hits{0};
        std::vector<std::string> errors;  // distinct messages, in order of first appearance
    };
    const Stats& stats() const { return stats_; }

private:
    struct Surface {
        u32 phys{0};
        u32 zeta_phys{0};
        u32 format_word{0};
        u32 width{0}, height{0};
        u32 log_w{0}, log_h{0};
        u32 aa_x{1}, aa_y{1};  // the anti-aliasing scale from logical to stored pixels
        u32 pitch{0}, zeta_pitch{0};
        bool swizzled{false};
        u32 color_format{0}, zeta_format{0};
        vk::Image color, depth;
        VkRenderPass pass{VK_NULL_HANDLE};
        VkFramebuffer framebuffer{VK_NULL_HANDLE};
        bool dirty{false};           // colour rendered since the last write-back
        bool zeta_dirty{false};      // depth rendered since the last write-back of the zeta surface
        bool rendered_this_frame{false};
        u64 draws_this_frame{0}, clears_this_frame{0};
        u32 bytes() const { return color_bytes() * (swizzled ? width * height : pitch * height); }
        u32 color_bytes() const;
        u32 zeta_bytes() const { return (zeta_format == 1 ? 2u : 4u) * (swizzled ? width * height : zeta_pitch * height); }
    };
    struct TextureEntry {
        vk::Image image;
        u32 phys{0}, length{0};
        bool valid{false};
        u32 width{0}, height{0};
        bool linear{false};
    };

    void note(const std::string& message);
    void begin_commands();
    void flush();  // end the pass, submit, wait, begin again
    void end_pass();
    Surface* target_for(const KelvinState& s);
    bool ensure_pass(Surface& t);
    void write_back(Surface& s);
    // The depth buffer into guest RAM in the zeta format (Z16, or Z24S8 with the stencil left zero): a shadow map
    // the guest samples as a depth texture lives there.
    void write_back_zeta(Surface& s);
    // Copies one image into the read-back buffer and waits for it.
    void read_back(const vk::Image& image, VkImageAspectFlags aspect, u32 width, u32 height);
    VkRenderPass render_pass(VkFormat color, VkFormat depth);
    VkShaderModule module(const std::vector<u32>& spirv);
    VkShaderModule vertex_shader(const KelvinState& s, std::string& error);
    VkShaderModule fragment_shader(const psh::Key& key, std::array<psh::SamplerKind, 4>& kinds, std::string& error);
    VkPipeline pipeline(const KelvinState& s, VkShaderModule vs, VkShaderModule fs, VkPrimitiveTopology topology, Surface& t, std::string& error);
    const TextureEntry* texture(const KelvinState& s, u32 stage, std::string& error);
    VkSampler sampler(const TextureStage& ts, u32 levels);
    bool ring_alloc(vk::Buffer& ring, VkDeviceSize& cursor, VkDeviceSize bytes, VkDeviceSize alignment, VkDeviceSize& offset);
    u32 texture_phys(const KelvinState& s, const TextureStage& ts) const;

    vk::Device& dev_;
    const HandleResolver& handles_;
    u8* ram_;
    u32 ram_size_;
    const vk::Functions* fn_{nullptr};

    VkCommandBuffer cb_{VK_NULL_HANDLE};
    VkFence fence_{VK_NULL_HANDLE};
    bool recording_{false};
    bool pass_open_{false};
    Surface* pass_target_{nullptr};

    std::unordered_map<u32, std::unique_ptr<Surface>> surfaces_;
    std::unordered_map<u64, VkRenderPass> passes_;
    std::unordered_map<std::string, VkShaderModule> vs_cache_, fs_cache_;
    // The last shaders answered, checked before the string-keyed caches: a draw usually repeats the previous
    // draw's program (the guest re-uploads it every draw, so the words are compared, not a version), and its
    // combiner key, and decoding and hashing them again cost most of the renderer's time.
    struct VsMemo {
        u32 start{0}, flags{0};
        std::vector<u32> words;  // a vertex program's words from start through its final instruction
        VkShaderModule module{VK_NULL_HANDLE};
    };
    struct FsMemo {
        psh::Key key;
        std::array<psh::SamplerKind, 4> kinds{};
        VkShaderModule module{VK_NULL_HANDLE};
    };
    // A few entries each, replaced round robin: a frame alternates between a handful of programs.
    static constexpr u32 kMemoEntries = 8;
    std::array<VsMemo, kMemoEntries> vs_memo_{};
    std::array<FsMemo, kMemoEntries> fs_memo_{};
    u32 vs_memo_next_{0}, fs_memo_next_{0};
    std::unordered_map<std::string, std::array<psh::SamplerKind, 4>> fs_kinds_;
    std::unordered_map<std::string, VkPipeline> pipelines_;
    std::unordered_map<std::string, std::unique_ptr<TextureEntry>> textures_;
    // The last binding of each stage: the registers it was built from and what it resolved to, so a draw that
    // changes nothing about its textures skips the address maths, the handle lookups and the hashing.
    struct StageCache {
        std::array<u32, 11> regs{};
        u32 phys{0}, length{0};
        const TextureEntry* entry{nullptr};
    };
    std::array<StageCache, 4> stage_cache_{};
    std::unordered_map<u64, VkSampler> samplers_;
    VkDescriptorSetLayout set_layout_{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    vk::Buffer vertex_ring_, index_ring_, uniform_ring_, staging_, readback_;
    VkDeviceSize vertex_cursor_{0}, index_cursor_{0}, uniform_cursor_{0};
    vk::Image dummy2d_, dummycube_, dummy3d_;
    VkSampler dummy_sampler_{VK_NULL_HANDLE};
    VkFormat depth24_{VK_FORMAT_D24_UNORM_S8_UINT};
    Stats stats_;
    // Debug overrides from MV_RENDER_DEBUG (comma separated): flat (white fragments), nodepth, nocull, noblend.
    bool debug_flat_{false}, debug_nodepth_{false}, debug_nocull_{false}, debug_noblend_{false}, debug_shadow_pass_{false}, debug_show_v0_{false}, debug_skip_blend_{false}, debug_draw_id_{false}, debug_flip_front_{false};
    u32 frame_draws_{0};  // draws since the frame began (the index mv-pbcap draws lists)
};

}  // namespace mv::pgraph
