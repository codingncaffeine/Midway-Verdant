#pragma once
#include <array>
#include <filesystem>
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
    // Compiled shaders and the driver's pipeline cache kept between runs in `dir` (call after init): a shader seen in
    // an earlier run loads instead of compiling, and its pipelines come from the driver's cache, so a scene seen before
    // does not stall on first use. The pipeline cache is written back when the renderer closes.
    void open_disk_cache(const std::filesystem::path& dir);

    void clear(const KelvinState& state, u32 clear_bits) override;
    void draw(const KelvinState& state, const DrawBatch& batch) override;
    // Submits the frame's work and writes every rendered surface back into guest RAM (waiting for the GPU only when
    // one was drawn).
    void end_frame();
    // The live machine's frame boundary: submit the work without waiting for it, and leave the surfaces on the
    // device — they reach RAM when a texture samples them, and the scanout one at vblank (write_back_at).
    void flush_frame();
    bool write_back_at(u32 phys);
    // Every render target overlapping [phys, phys + len) reaches RAM (colour and depth): the 2D engine reads there.
    void write_back_range(u32 phys, u32 len);
    // Every dirty render target reaches RAM (colour and depth). The live machine calls it when the GPU has been idle
    // for a whole frame: the hardware wrote those pixels when it drew them, and RAM a program reuses later must not
    // receive them then (a title launched after the boot animation got the animation's targets over its own data).
    void write_back_all();
    // Guest RAM in [phys, phys + len) was written: textures decoded from it are stale.
    void invalidate(u32 phys, u32 len);
    // The guest CPU (or a device) wrote these RAM pages (4 KiB, sorted): a render target over them reloads from RAM
    // before it is drawn into again, its own unwritten pages written back first.
    void cpu_wrote(const std::vector<u32>& pages);

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
        u64 batch_flushes{0};  // batches submitted mid-frame because the rings or the descriptor pool were full
        u64 waits{0};          // times the CPU stopped for the GPU (a fence wait)
        u64 uniform_reuses{0};  // uniform blocks equal to the frame's last upload, bound where that lies
        u64 descriptor_reuses{0};  // draws that kept the previous draw's descriptor set
        u64 surface_samples{0};  // textures bound straight to a render target's image
        u64 snapshots{0};  // copies of a target taken for draws that sample the target they draw into
        u64 zeta_conversions{0};  // depth buffers converted on the GPU for a colour texture at their address
        u64 cpu_written_targets{0};  // render targets whose memory the CPU wrote: reloaded from RAM
        u64 depth_handovers{0}, depth_uploads{0};  // depth buffers moved between targets sharing an address; loaded from RAM
        u64 shaders_from_disk{0};  // shader compiles the disk cache answered
        u64 textures_unchanged{0};
        u64 surface_reloads{0};  // render targets reloaded from RAM after another engine wrote them  // textures whose pages were written but whose bytes hashed the same: kept
        // Write-backs by cause: frame end, a texture sampling the target, the scanout at vblank, a 2D or CPU range,
        // a target relaid out, the idle vblank (all).
        enum WriteBackWhy : unsigned { kWbFrameEnd, kWbTexture, kWbScanout, kWbRange, kWbRelayout, kWbAll, kWbWhyCount };
        u64 writebacks_by[kWbWhyCount]{};
        // Where the time goes, in nanoseconds: the whole draw and its parts (the texture part includes the
        // write-backs a sample forces; the shader parts are the compiles, a lookup costs nothing measurable); the
        // submits and the fence waits; the copies back to RAM, colour and depth (and the wait for the device copy
        // inside them).
        u64 ns_draw{0}, ns_fetch{0}, ns_texture{0}, ns_pipeline{0}, ns_flush{0}, ns_write_back{0}, ns_write_back_gpu{0};
        u64 ns_vs{0}, ns_fs{0}, ns_indices{0}, ns_upload{0}, ns_descriptors{0}, ns_record{0};
        u64 vs_memo_hits{0}, vs_cache_hits{0}, fs_memo_hits{0}, fs_cache_hits{0};
        std::vector<std::string> errors;  // distinct messages, in order of first appearance
    };
    const Stats& stats() const { return stats_; }

private:
    struct TextureEntry {
        vk::Image image;
        u32 phys{0}, length{0};
        bool valid{false};
        u32 width{0}, height{0};
        bool linear{false};
        bool surface{false};  // a render target's own image, sampled in place (owned by the surface)
        u32 generation{0};       // the RAM write generation its texels and palette were decoded (or last checked) at
        u32 palette_phys{0};
        u64 content{0};          // a hash of the texels and palette in RAM when decoded
    };
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
        u32 depth_holds{0};          // the zeta address whose contents the depth image holds (0: none, undefined)
        bool rendered_this_frame{false};
        u64 draws_this_frame{0}, clears_this_frame{0};
        // The target sampled in place: views for an X8 texture (alpha one) and for the depth aspect alone, and the
        // entries a draw binds.
        VkImageView x8_view{VK_NULL_HANDLE}, depth_view{VK_NULL_HANDLE};
        TextureEntry color_tex, color_x8_tex, depth_tex;
        // A copy of the colour image for draws that sample the target they draw into (a blur done in place): it holds
        // what the draws before them left, as RAM would; retaken whenever the target was drawn into since.
        vk::Image snap;
        VkImageView snap_x8_view{VK_NULL_HANDLE};
        TextureEntry snap_tex, snap_x8_tex;
        u64 draw_serial{0}, snap_serial{~0ull};
        // The depth buffer read as a colour texture at its address (the Z24S8 words as A8R8G8B8 texels), converted on
        // the GPU; retaken whenever the target was drawn into or cleared since.
        vk::Image zcolor;
        VkImageView zcolor_x8_view{VK_NULL_HANDLE};
        TextureEntry zcolor_tex, zcolor_x8_tex;
        u64 zcolor_serial{~0ull};
        // The frame the target was last drawn in, and whether a GPU-side copy (a blit) wrote its memory since: a clean
        // target drawn this frame or the last, with nothing written over it, still holds what RAM holds.
        u64 last_rendered_frame{0};
        bool ram_newer{false};
        bool color_ram_newer{false};  // another engine wrote the colour memory: reload it before drawing into it
        // The guest memory each image covers: a pitch is already in bytes.
        u32 bytes() const { return swizzled ? color_bytes() * width * height : pitch * height; }
        u32 color_bytes() const;
        u32 zeta_bytes() const { return swizzled ? (zeta_format == 1 ? 2u : 4u) * width * height : zeta_pitch * height; }
    };
    void note(const std::string& message);
    // Ends the pass and submits the recording frame without waiting for it, then starts the next frame in flight
    // (waiting only if that frame's previous submission has not finished).
    void submit();
    // Submits, then waits until the GPU has executed everything submitted so far.
    void finish();
    // Makes a frame the recording one: waits for its last submission, frees what it retired, resets its descriptor
    // pool and ring regions, and begins its command buffer.
    void begin_frame(u32 slot);
    // Destroys an object once the GPU is done with the commands recorded so far (the recording frame's fence).
    void retire(vk::Image& image);
    void retire(VkImageView& view);
    void retire(VkFramebuffer& framebuffer);
    void end_pass();
    Surface* target_for(const KelvinState& s);
    bool ensure_pass(Surface& t);
    void upload_color_from_ram(Surface& t);
    // Write-backs come in batches: each queued surface records its copy after the frame's draws, and
    // finish_write_backs() waits once for all of them, then stores each into guest RAM in the order queued.
    void queue_write_back(Surface& s);
    // The depth buffer into guest RAM in the zeta format (Z16, or Z24S8: depth in bits 31-8, stencil in 7-0): a
    // shadow map the guest samples as a depth texture lives there.
    void queue_write_back_zeta(Surface& s);
    void finish_write_backs();
    void write_back(Surface& s);       // one surface's colour, queued and finished
    void write_back_zeta(Surface& s);  // one surface's depth, queued and finished
    void store_color(Surface& s, const u8* pixels);
    void store_zeta(Surface& s, const u8* pixels);
    // Records the copy of one image into the read-back buffer at offset (depth and stencil together: the depth's
    // words, then the stencil bytes).
    void read_back(const vk::Image& image, VkImageAspectFlags aspect, u32 width, u32 height, VkDeviceSize offset);
    // Space in the read-back buffer for the batch, storing the batch so far first when the buffer is full.
    bool readback_alloc(VkDeviceSize bytes, VkDeviceSize& offset);
    VkRenderPass render_pass(VkFormat color, VkFormat depth);
    VkShaderModule module(const std::vector<u32>& spirv);
    VkShaderModule vertex_shader(const KelvinState& s, std::string& error);
    VkShaderModule fragment_shader(const psh::Key& key, std::array<psh::SamplerKind, 4>& kinds, std::string& error);
    VkPipeline pipeline(const KelvinState& s, VkShaderModule vs, VkShaderModule fs, VkPrimitiveTopology topology, Surface& t, u32 vertex_layout, std::string& error);
    // The pipeline of a clear that writes only some colour channels (the colour comes in as the blend constants).
    VkPipeline clear_pipeline(VkRenderPass pass, VkColorComponentFlags mask, std::string& error);
    const TextureEntry* texture(const KelvinState& s, u32 stage, std::string& error);
    // A render target the stage can sample in place (same address and size, a format of the same bytes), or null.
    TextureEntry* surface_texture(const TextureStage& ts, u32 phys);
    TextureEntry* snapshot_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear, bool x8);
    Surface* draw_target_{nullptr};  // the target of the draw whose textures are being looked up
    // The depth buffer at each zeta address lives in one target's depth image at a time: the target that last drew with
    // it. Another target binding the same address takes the contents first (a GPU copy, or through RAM).
    std::unordered_map<u32, Surface*> depth_owner_;
    void bind_depth(Surface& t);
    void upload_depth_from_ram(Surface& t);
    void release_depth(Surface& s);  // a target going away: its depth contents to RAM, its ownership dropped
    TextureEntry* zeta_color_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear, bool x8);
    bool init_zeta_convert();
    // The depth-to-colour conversion: a compute pass over two device buffers (the copied depth words and stencil
    // bytes in, the Z24S8 words out), made on first use.
    static constexpr u32 kZetaConvertMaxTexels = 1024 * 1024;
    VkDescriptorSetLayout zc_set_layout_{VK_NULL_HANDLE};
    VkPipelineLayout zc_layout_{VK_NULL_HANDLE};
    VkShaderModule zc_module_{VK_NULL_HANDLE};
    VkPipeline zc_pipeline_{VK_NULL_HANDLE};
    VkDescriptorPool zc_pool_{VK_NULL_HANDLE};
    VkDescriptorSet zc_set_{VK_NULL_HANDLE};
    vk::Buffer zc_in_, zc_out_;
    bool zc_failed_{false};
    // Orders the attachment writes recorded so far before later sampling, and earlier sampling before later writes.
    void surface_barrier();
    void destroy_surface_views(Surface& s);
    void invalidate_textures(u32 phys, u32 len);
    u64 texture_content(u32 phys, u32 length, u32 palette_phys) const;  // the bytes a decode reads, hashed
    bool texture_current(TextureEntry& e);  // RAM still holds what e was decoded from (rehashed when pages were written)  // the decoded textures over a range (RAM now holds what a target wrote)
    VkSampler sampler(const TextureStage& ts, u32 levels);
    // A host buffer with one region per frame in flight; the recording frame allocates from its own region only.
    struct Ring {
        vk::Buffer buffer;
        VkDeviceSize region{0}, begin{0}, cursor{0};
        VkDeviceSize end() const { return begin + region; }
    };
    bool create_ring(Ring& ring, VkDeviceSize region, VkBufferUsageFlags usage, std::string& error);
    bool ring_alloc(Ring& ring, VkDeviceSize bytes, VkDeviceSize alignment, VkDeviceSize& offset);
    // Staging space for an upload in the recording frame, submitting the frame first when its region is full.
    bool staging_alloc(VkDeviceSize bytes, VkDeviceSize& offset);
    // A new image enters GENERAL before its first use (nothing earlier uses it, so nothing is waited for).
    void fresh_image(const vk::Image& image);
    // The uploads recorded so far land before the draws that sample them and the copies that read them.
    void upload_barrier();
    // Whether a draw with these vertex and index sizes still fits the rings and the descriptor pool of the batch.
    bool fits_in_batch(VkDeviceSize vertex_bytes, VkDeviceSize index_bytes) const;
    u32 texture_phys(const KelvinState& s, const TextureStage& ts) const;
    // Fills a new descriptor set: the two uniform blocks at their ring offsets and the four texture stages.
    void write_descriptors(VkDescriptorSet set, VkDeviceSize uoffset_vs, VkDeviceSize uoffset_ps, const VkDescriptorImageInfo* imgs);

    vk::Device& dev_;
    const HandleResolver& handles_;
    u8* ram_;
    u32 ram_size_;
    const vk::Functions* fn_{nullptr};

    // The frames in flight. Each owns a command buffer, the fence its submission signals, a descriptor pool and a
    // region of every ring, so a submit does not wait: a frame's resources are reused only once its fence signals.
    static constexpr u32 kFramesInFlight = 3;
    struct Frame {
        VkCommandBuffer cb{VK_NULL_HANDLE};
        VkFence fence{VK_NULL_HANDLE};
        VkDescriptorPool pool{VK_NULL_HANDLE};
        u64 serial{0};  // the submission that last used it (0: never submitted)
        // Objects retired while this frame recorded: commands already recorded (here or earlier) may use them.
        std::vector<vk::Image> dead_images;
        std::vector<VkImageView> dead_views;
        std::vector<VkFramebuffer> dead_framebuffers;
    };
    std::array<Frame, kFramesInFlight> frames_{};
    u32 slot_{0};                       // the recording frame
    u64 submitted_{0}, completed_{0};  // submissions made, and the last one known to have finished
    VkCommandBuffer cb_{VK_NULL_HANDLE};  // the recording frame's command buffer
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
    std::unordered_map<u64, VkPipeline> clear_pipelines_;  // by render pass and channel mask
    VkShaderModule clear_vs_{VK_NULL_HANDLE}, clear_fs_{VK_NULL_HANDLE};
    std::unordered_map<std::string, std::unique_ptr<TextureEntry>> textures_;
    // The last binding of each stage: the registers it was built from and what it resolved to, so a draw that
    // changes nothing about its textures skips the address maths, the handle lookups and the hashing.
    struct StageCache {
        std::array<u32, 11> regs{};
        u32 phys{0}, length{0};
        TextureEntry* entry{nullptr};
    };
    std::array<StageCache, 4> stage_cache_{};
    std::unordered_map<u64, VkSampler> samplers_;
    VkDescriptorSetLayout set_layout_{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
    Ring vertex_ring_, index_ring_, uniform_ring_, staging_;
    vk::Buffer readback_;
    struct PendingWriteBack {
        Surface* surface{nullptr};
        bool zeta{false};
        VkDeviceSize offset{0};  // where its copy lands in the read-back buffer
    };
    std::vector<PendingWriteBack> write_backs_;
    unsigned wb_why_{Stats::kWbAll};  // the cause the write-backs being queued count under
    VkDeviceSize readback_cursor_{0};
    u32 descriptor_sets_used_{0};  // from the recording frame's pool
    // The vertex uniform block being built, and the last block of each kind uploaded in the recording frame.
    std::array<float, vsh::kUniformBytes / 4> vs_uniforms_{};
    struct UniformUpload {
        std::vector<u8> bytes;
        VkDeviceSize offset{0};
        bool valid{false};
    };
    UniformUpload last_vs_upload_, last_ps_upload_;
    // What the frame's last descriptor set binds, the set itself, and the set bound in the command buffer.
    struct DescriptorKey {
        VkDeviceSize vs_offset{0}, ps_offset{0};
        std::array<VkImageView, 4> views{};
        std::array<VkSampler, 4> samplers{};
        bool operator==(const DescriptorKey&) const = default;
    };
    DescriptorKey last_set_key_;
    VkDescriptorSet last_set_{VK_NULL_HANDLE}, bound_set_{VK_NULL_HANDLE};
    vk::Image dummy2d_, dummycube_, dummy3d_;
    VkSampler dummy_sampler_{VK_NULL_HANDLE};
    // The disk cache (open_disk_cache): where compiled shaders go, and the driver's pipeline cache.
    std::filesystem::path cache_dir_;
    VkPipelineCache pipeline_cache_{VK_NULL_HANDLE};
    // SPIR-V for a translated shader: from the disk cache when it holds this exact source, else compiled (and stored).
    std::vector<u32> spirv_for(bool vertex, const std::string& glsl, const char* name, std::string& log);
    void save_pipeline_cache();
    VkFormat depth24_{VK_FORMAT_D24_UNORM_S8_UINT};
    Stats stats_;
    // Debug overrides from MV_RENDER_DEBUG (comma separated): flat (white fragments), nodepth, nocull, noblend.
    bool debug_flat_{false}, debug_nodepth_{false}, debug_nocull_{false}, debug_noblend_{false}, debug_shadow_pass_{false}, debug_show_v0_{false}, debug_skip_blend_{false}, debug_draw_id_{false}, debug_flip_front_{false}, debug_ff_dump_{false};
    u64 frame_index_{1};  // frames ended so far, plus one
    bool attachments_written_{false}, sampled_since_barrier_{false};  // hazards the next surface_barrier() covers
    bool depth16_sampled_{false}, depth24_sampled_{false};  // the device samples (and filters) these depth formats
    bool debug_tex_log_{false};
    bool debug_no_fog_{false};
    bool debug_vsh_dump_{false};
    std::unordered_map<VkShaderModule, u32> vs_debug_ids_;  // drawid/vshdump: each vertex shader's dump number
    bool debug_roundtrip_{false};  // MV_RENDER_DEBUG=roundtrip: sampled targets go through RAM as before (an A/B control)
    u32 frame_draws_{0};  // draws since the frame began (the index mv-pbcap draws lists)
};

}  // namespace mv::pgraph
