#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nv2a/overlay.h"
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
    // Background compiles (the fast core's default; off by default): a shader or pipeline seen for the first time is
    // compiled on a worker thread and the draws that need it are skipped until it is ready (a frame or two), instead of
    // stalling the frame for tens of milliseconds as each new scene appears. Never for the accurate core: what a
    // frame shows would depend on host timing.
    void set_async_compile(bool on);
    // Anisotropic filtering of linearly filtered textures: at least `n` (1 = only what each stage asks for), up to the
    // device's limit (the libretro core's "Anisotropic filtering" option).
    void set_min_anisotropy(u32 n) { min_anisotropy_.store(std::max(n, 1u), std::memory_order_relaxed); }  // any thread: new samplers take it
    // Internal resolution (PLAN D34, the libretro core's option; any thread): render targets drawn at `s` times their
    // size (1-8), RAM still at the console's. A change takes effect at the next vblank, where every target is written
    // back and dropped (each comes back from RAM at the new scale), or at once while there are no targets yet.
    void set_resolution_scale(u32 s) { requested_scale_.store(std::clamp(s, 1u, 8u), std::memory_order_relaxed); }
    // The RAMDAC's colour table for the picture the next vblank captures (none: the identity), applied on the GPU as the
    // scaled picture is read back (the vblank's thread, before write_back_at).
    void set_display_lut(const std::optional<std::array<u8, 768>>& lut) {
        display_lut_on_ = lut.has_value();
        if (lut) display_lut_ = *lut;
    }
    // The video overlay (PVIDEO) the next vblank scans out over the picture (none: off), drawn in by the same pass on
    // the GPU as grab_frame would on the CPU (a movie over a rendered picture at 6x was 20 ms a frame on four threads).
    void set_display_overlay(const std::optional<Overlay>& ov) { display_ov_ = ov; }
    // The scanned-out picture at the internal resolution: the target holding the scanout as read back at the last
    // vblank, cropped to the scanout (w x h pixels of bpp bytes at phys, rows pitch bytes apart in RAM), as `scale`
    // x w by `scale` x h XRGB8888 pixels (alpha clear, rows tightly packed). False at 1x, or when RAM holds the
    // picture (no target there, or another engine wrote it since): the caller widens the RAM picture instead.
    bool scaled_scanout(u32 phys, u32 pitch, u32 bpp, bool rgb565, u32 w, u32 h, std::vector<u32>& xrgb, u32& scale) const;
    // The same picture where it lies (no copy): the rectangle's first pixel, its row pitch and its scale, in the frame
    // buffer's format. The copy that lands it was submitted at the vblank and not waited for there (D36): picture_ready
    // waits, with no lock held, so the CPU can run the next frame meanwhile. The frontend holds the picture from
    // scaled_picture until picture_done: no capture writes its buffer in between (the next vblank's goes to the
    // target's other one).
    struct ScaledPicture {
        const u8* pixels{nullptr};
        u32 pitch{0}, sc{1};
        bool xrgb{false};  // finished XRGB8888 (the colour table applied, alpha clear), not the frame buffer's format
        bool overlaid{false};  // the video overlay drawn in on the GPU (set_display_overlay)
        VkFence ready{VK_NULL_HANDLE};
        // Where it lies on the device (with set_handoff the buffer is device memory and `pixels` is null): the
        // rectangle's first byte and the buffer's row length in pixels.
        VkBuffer buffer{VK_NULL_HANDLE};
        VkDeviceSize offset{0};
        u32 row_pixels{0};
    };
    bool scaled_picture(u32 phys, u32 pitch, u32 bpp, u32 w, u32 h, ScaledPicture& out) const;
    void picture_ready(const ScaledPicture& p) const;
    void picture_done() const;

    // The picture handed to a frontend on the GPU (the libretro Vulkan interface, PLAN D38): with set_handoff the
    // captures stay in device memory (nothing read back), and hand_over copies the frame on the GPU into the image of
    // the frontend's sync index: the GPU's picture (`pic`, from scaled_picture), or the CPU's (`cpu`, XRGB8888 rows
    // `cpu_pitch` bytes apart). B8G8R8A8 with the view's alpha one, kept in GENERAL (the frontend does no
    // transitions); the caller waited for the frontend to be done with the index (wait_sync_index). Any thread.
    void set_handoff(bool on) { handoff_on_ = on; }
    struct HandedOver {
        VkImageView view{VK_NULL_HANDLE};
        const VkImageViewCreateInfo* info{nullptr};
        u32 width{0}, height{0};
    };
    bool hand_over(u32 sync_index, const ScaledPicture* pic, u32 width, u32 height, const u32* cpu, std::size_t cpu_pitch, HandedOver& out);

    void clear(const KelvinState& state, u32 clear_bits) override;
    void draw(const KelvinState& state, const DrawBatch& batch) override;
    // The next draw's vertices, fetched already (the render thread's draws: fetched when the stream was decoded, before
    // the guest could reuse the memory); the draw takes them instead of reading RAM. Null: it reads RAM.
    void set_prefetched(FetchedVertices* v) { prefetched_ = v; }
    // Submits the frame's work and writes every rendered surface back into guest RAM (waiting for the GPU only when
    // one was drawn).
    void end_frame();
    // The live machine's frame boundary: submit the work without waiting for it, and leave the surfaces on the
    // device — they reach RAM when a texture samples them, and the scanout one at vblank (write_back_at).
    void flush_frame();
    // Draw counting (the timed pusher's pixels, PLAN D28 stage B): every draw is wrapped in an occlusion query (the
    // samples that passed, exact by the Vulkan definition: the core stays deterministic); the last
    // draw's ticket (0: it drew nothing) is read after the draw, and draw_counts gives its counts in the target's
    // logical pixels (divided by its anti-aliasing scale), submitting and waiting for the GPU only if they are not in.
    void set_count_draws(bool on) { count_draws_ = on; }
    // Z-pass counting alone (the fast core's occlusion reports): only draws made with SET_ZPASS_PIXEL_COUNT_ENABLE on are
    // wrapped in a query and given a ticket, as the reference emulator counts them.
    void set_count_zpass(bool on) { count_zpass_ = on; }
    u64 last_draw_ticket() const { return last_ticket_; }
    struct DrawCounts {
        u64 fragments{0};  // pixels written: the passed samples in the target's logical pixels
        u64 samples{0};    // samples that passed the depth and stencil tests, at the stored (anti-aliased) size
    };
    DrawCounts draw_counts(u64 ticket);
    bool write_back_at(u32 phys);
    // Every render target overlapping [phys, phys + len) reaches RAM (colour and depth): the 2D engine reads there.
    void write_back_range(u32 phys, u32 len);
    // The 2D engine's copy of a rectangle (w x h pixels of bpp bytes, rows src_pitch / dst_pitch bytes apart) when both
    // sides lie inside render targets whose images are current: an image copy on the GPU, the destination left newer than
    // RAM. False (nothing done) when RAM has to carry it: the caller copies there.
    bool copy_rect(u32 src_phys, u32 src_pitch, u32 dst_phys, u32 dst_pitch, u32 w, u32 h, u32 bpp);
    // Every dirty render target reaches RAM (colour and depth). The live machine calls it when the GPU has been idle
    // for a whole frame: the hardware wrote those pixels when it drew them, and RAM a program reuses later must not
    // receive them then (a title launched after the boot animation got the animation's targets over its own data).
    // With `in_background` the colour and depth copies go to the GPU without a wait and reach RAM later
    // (complete_write_backs), as the scanout's does above 1x.
    void write_back_all(bool in_background = false);
    // Write-backs queued without a wait (an idle vblank's, the scanout's above 1x: the CPU and the GPU thread used to wait
    // there for the GPU to finish the whole frame, with both locks held). They reach RAM here: once the GPU is done with
    // them, or at once with `wait`. RAM pages in `cpu_pages` (sorted) were written by the CPU or a device since they were
    // queued and keep those bytes. False when they are still in flight and `wait` is not set.
    bool complete_write_backs(bool wait, const std::vector<u32>* cpu_pages = nullptr);
    bool write_backs_in_flight() const { return async_pending_.load(std::memory_order_acquire); }
    // In flight and done on the GPU: the CPU's thread may store them now (it can name the pages the guest wrote since).
    bool write_backs_ready();
    // Guest RAM in [phys, phys + len) was written: textures decoded from it are stale.
    void invalidate(u32 phys, u32 len);
    // The guest CPU (or a device) wrote these RAM pages (4 KiB, sorted): a render target over them reloads from RAM
    // before it is drawn into again, its own unwritten pages written back first. `generation`: the write generation the
    // pages carry, the stream position the deferred stores after this point are tagged with.
    void cpu_wrote(const std::vector<u32>& pages, u32 generation = 0);

    // PLAN 12d stage 2: with the render thread the renderer runs behind the stream position whose fences the CPU has
    // seen, so its stores must not reach RAM from its own thread (Halo 2: a write-back landed over kernel structures the
    // game had put in a target's old memory once the fence passed). Deferred, every store goes to a journal of whole
    // pages, each with the bytes RAM held when the renderer first stored there and the write generation of its place in
    // the stream; the renderer's own reads of RAM see the journal. The CPU's thread lands it (land_stores) between exits.
    void set_deferred_stores(bool on) { deferred_ = on; }
    bool stores_pending() const { return journal_pages_.load(std::memory_order_acquire) != 0; }
    // A journal page reaches RAM unless the CPU or a device wrote it later than the store's place in the stream: RAM no
    // longer holds the bytes it was stored over (a write since), or `written_after(page, tag)` (the write log names it
    // after the tag: a write between that place and the store, which the bytes cannot show). Then the CPU's bytes stay.
    // Returns the pages stored. Only where the guest cannot run (the CPU's thread between exits).
    std::size_t land_stores(const std::function<bool(u32 page, u32 tag)>& written_after);
    // The 2D engine's copy through RAM (h rows of `row` bytes) as the renderer sees RAM, stored as the renderer stores.
    void ram_copy_rows(u32 dst, u32 dst_pitch, u32 src, u32 src_pitch, u32 row, u32 h);
    struct JournalStats {
        u64 stores{0}, pages{0}, landed{0}, kept{0}, dropped{0}, overlaid_reads{0};
    };
    JournalStats journal_stats() const {
        std::lock_guard g(journal_mutex_);
        return journal_stats_;
    }

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
        u64 staging_flushes{0};  // batches submitted mid-frame because the upload staging ring was full
        u64 waits{0};          // times the CPU stopped for the GPU (a fence wait)
        u64 uniform_reuses{0};  // uniform blocks equal to the frame's last upload, bound where that lies
        u64 descriptor_reuses{0};  // draws that kept the previous draw's descriptor set
        u64 surface_samples{0};  // textures bound straight to a render target's image
        u64 snapshots{0};  // copies of a target taken for draws that sample the target they draw into
        u64 background_write_back_batches{0}, background_write_back_waits{0}, background_pages_kept{0};
        u64 pending_waits{0};
        u64 textures_native_dxt{0};  // DXT textures uploaded as blocks  // short waits for background compiles (wait_a_little)
        u64 depth_snapshots{0};  // the same for draws that sample the depth buffer of the target they draw into
        u64 zeta_conversions{0};  // depth buffers converted on the GPU for a colour texture at their address
        u64 cpu_written_targets{0};  // render targets whose memory the CPU wrote: reloaded from RAM
        u64 cpu_written_depth{0};    // of them, depth buffers (held or parked) let go because the CPU wrote their memory
        u64 depth_handovers{0}, depth_uploads{0};  // depth buffers moved between targets sharing an address; loaded from RAM
        u64 depth_unparks{0}, parked_samples{0};  // parked depth images bound again; sampled in place
        u64 depth_loads_skipped{0};  // depth buffers bound for a clear of the whole buffer: nothing loaded
        u64 depth_borrows{0};  // binds that drew with a bigger target's depth image of the same pitch (D37)
        u64 evicted{0}, retired_unused{0};  // targets another target overlapped; targets nothing used for 10 s
        u64 subrect_copies{0};  // rectangles of a pitch-linear region copied for a texture
        u64 cube_assemblies{0};  // cube maps put together on the GPU from six render targets (cube_texture)
        u64 mip_assemblies{0};  // mip chains put together on the GPU from their level targets (mip_texture)
        u64 gpu_copies{0};  // 2D-engine copies made between target images (copy_rect)
        u64 shaders_from_disk{0};  // shader compiles the disk cache answered
        u64 compiles_queued{0}, pipelines_queued{0}, pending_skips{0};  // background compiles started; draws skipped waiting
        u64 pipeline_cache_hits{0};  // pipelines the cache held, made at once with background compiles on
        u64 textures_unchanged{0};
        // Instrument: the write-backs a texture forced, by target and texture (what is not sampled in place, and why).
        std::map<std::string, u64> forced_by;
        // Blended draws by source factor, destination factor, equation, w-buffer and stage 0's format and edge modes
        // (key bits: sf 63:48, df 47:32, equation 31:16, w-buffer 15, format 14:8, wrap u 7:4, wrap v 3:0).
        std::unordered_map<u64, u64> blend_modes;
        // Instrument: the CPU side of storing write-backs in RAM, by target: count and nanoseconds.
        std::map<std::string, std::pair<u64, u64>> store_cost;
        std::map<std::string, u64> relayouts;  // a target given up for one at its memory, by old -> new layout (relaid out or evicted)
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
        u64 ns_wait_slot{0}, ns_wait_finish{0};  // waits for a frame slot the GPU still runs, and for everything submitted (finish)
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
        // The texels' hash per 4 KiB page of RAM they lie in (each clipped to the texels) and the palette's: a write to
        // some pages rehashes those alone (a title rewriting a texture's neighbours had whole textures rehashed ~24 times
        // a frame).
        std::vector<u64> page_hashes;
        u64 palette_hash{0};
    };
    struct Surface {
        u32 phys{0};
        u32 zeta_phys{0};
        u32 format_word{0};
        u32 width{0}, height{0};
        u32 log_w{0}, log_h{0};
        u32 aa_x{1}, aa_y{1};  // the anti-aliasing scale from logical to stored pixels
        // The internal resolution's scale from stored pixels (RAM's) to the images' (D34): width and height stay the
        // stored size, the colour and depth images are iw() x ih().
        u32 sc{1};
        u32 iw() const { return width * sc; }
        u32 ih() const { return height * sc; }
        // At sc > 1: a stored-size copy of the colour image, between the image and RAM (blitted down for a write-back,
        // up after a load).
        vk::Image lo;
        bool lo_linear{false};  // the device filters the format: the way down is a box filter, not one sample in sc x sc
        // At sc > 1: a stored-size copy of the depth image, blitted down (NEAREST) for a write-back. Recording one copy
        // region per stored row cost the driver 5 ms for a 1024 x 1024 buffer at 6x (PGR2 races, each frame).
        vk::Image zeta_lo;
        u32 pitch{0}, zeta_pitch{0};
        bool swizzled{false};
        u32 color_format{0}, zeta_format{0};
        vk::Image color, depth;
        VkRenderPass pass{VK_NULL_HANDLE};
        VkFramebuffer framebuffer{VK_NULL_HANDLE};
        bool dirty{false};           // colour rendered since the last write-back
        // Each page of its colour memory hashed when the image and RAM last agreed (loaded from RAM or written back):
        // a page the write log reports is the CPU's only if its bytes changed since (cpu_wrote).
        std::vector<u64> page_hashes;
        bool zeta_dirty{false};      // depth rendered since the last write-back of the zeta surface
        u32 depth_holds{0};          // the zeta address whose contents the depth image holds (0: none, undefined)
        u32 holds_pitch{0};          // that address's zeta pitch when it was bound
        u64 last_used_vblank{0};     // the last vblank it was drawn into or sampled
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
        // The same for a draw that samples the depth buffer of the target it draws into (a shadow map tested while the
        // target is drawn): a copy of the depth image, retaken after the target was drawn into or cleared since.
        vk::Image zsnap;
        VkImageView zsnap_view{VK_NULL_HANDLE};
        TextureEntry zsnap_tex;
        u64 zsnap_serial{~0ull};
        // Copies of rectangles of a pitch-linear region that textures read (key: x, y, width, height), each retaken after
        // the region was drawn into since.
        struct SubCopy {
            vk::Image image;
            VkImageView x8_view{VK_NULL_HANDLE};
            TextureEntry tex, x8_tex;
            u64 serial{~0ull};
        };
        std::unordered_map<u64, SubCopy> subs;
        // The depth buffer read as a colour texture at its address (the Z24S8 words as A8R8G8B8 texels), converted on
        // the GPU; retaken whenever the target was drawn into or cleared since.
        vk::Image zcolor;
        VkImageView zcolor_x8_view{VK_NULL_HANDLE};
        TextureEntry zcolor_tex, zcolor_x8_tex;
        VkImageView zcolor_abgr_view{VK_NULL_HANDLE};  // the same bytes read as A8B8G8R8 (red first)
        TextureEntry zcolor_abgr_tex;
        u64 zcolor_serial{~0ull};
        // Depth images of other zeta addresses the target bound before, kept on the GPU with their contents (D35): a
        // target that alternates between depth buffers (the boot animation's two shadow depths, each frame) swaps
        // images instead of writing one back and loading the other through RAM. A parked image is owned
        // (depth_owner_) like the bound one and is sampled in place; anything else touching its memory sends it to
        // RAM first (drop_parked, flush_parked). One that holds nothing (depth_holds 0) is a spare.
        struct ParkedDepth {
            vk::Image depth;
            VkImageView depth_view{VK_NULL_HANDLE};
            TextureEntry depth_tex;
            VkFramebuffer framebuffer{VK_NULL_HANDLE};
            bool zeta_dirty{false};
            u32 depth_holds{0}, holds_pitch{0};
            u64 used{0};
        };
        std::vector<ParkedDepth> parked;
        // Drawing with another target's depth image (D37): a pitch-linear depth buffer is memory, and a smaller target of
        // the same zeta pitch binding it uses the top-left rectangle of the image that holds it, as on the hardware
        // (Gauntlet Seven Sorrows' 720x480 frame buffers test the stencil of a 768x512 target's depth buffer each frame).
        // borrow_fb pairs this colour image with that depth image; it holds while the lender's bound image is the same.
        Surface* borrow{nullptr};
        VkFramebuffer borrow_fb{VK_NULL_HANDLE};
        VkImage borrow_image{VK_NULL_HANDLE};
        Surface& depth_target() { return borrow ? *borrow : *this; }  // the target whose depth image the passes draw into
        u32 parked_bytes(const ParkedDepth& p) const { return swizzled ? (zeta_format == 1 ? 2u : 4u) * width * height : p.holds_pitch * height; }
        u32 held_bytes() const { return swizzled ? (zeta_format == 1 ? 2u : 4u) * width * height : (holds_pitch ? holds_pitch : zeta_pitch) * height; }
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
    // Ends the pass and submits what the recording frame has recorded without waiting for it, then records on into a new
    // command buffer of the same frame. `rotate` (the draw rings or the staging ring are full; the vblank asked for it;
    // every submit while draws are counted) starts the next frame in flight instead
    // (waiting only if that frame's last submission has not finished).
    void submit(bool rotate = false);
    void begin_commands();  // the recording frame's next command buffer, begun
    bool rotate_pending_{false};  // set at the vblank: the next submit starts the next frame in flight
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
    // B8 (R8) and G8B8 (R8G8) targets keep blue in the image's first channel: outputs, masks, clears and blend constants
    // swap red and blue for them, so each byte in RAM is the one the hardware writes.
    static bool swaps_rb(const Surface& t) { return t.color_format == 9 || t.color_format == 10; }
    // The rectangle of the target the current draw or clear uses (stored pixels): a pitch-linear target is memory, and
    // a title may draw at an address inside it (a smaller shape at its base, a band further down) with the same pitch,
    // which is a rectangle of the same image, as on the hardware.
    struct View {
        u32 x{0}, y{0}, w{0}, h{0};
    } view_;
    // writes_color: the draw or clear writes the colour image; a pass that does not leaves a colour reload from RAM for
    // later (the boot animation's depth-only target sat over kernel memory the CPU writes every frame).
    bool ensure_pass(Surface& t, bool writes_color = true);
    void upload_color_from_ram(Surface& t);
    // Write-backs come in batches: each queued surface records its copy after the frame's draws, and
    // finish_write_backs() waits once for all of them, then stores each into guest RAM in the order queued.
    void queue_write_back(Surface& s);
    // The depth buffer into guest RAM in the zeta format (Z16, or Z24S8: depth in bits 31-8, stencil in 7-0): a
    // shadow map the guest samples as a depth texture lives there.
    void queue_write_back_zeta(Surface& s);
    void finish_write_backs();
    // The same without the wait: the copies go to the GPU and are stored by complete_write_backs (one batch in flight).
    void finish_write_backs_async();
    // A batch in flight that covers [b0, b1) is stored first (anything about to read that RAM).
    void complete_write_backs_over(u64 b0, u64 b1);
    void write_back(Surface& s);       // one surface's colour, queued and finished
    void write_back_zeta(Surface& s);  // one surface's depth, queued and finished
    void store_color(Surface& s, const u8* pixels);
    void store_zeta(Surface& s, const u8* pixels);
    void dump_target(Surface& s);  // MV_DEBUG_TARGET_DUMP: the scaled image to a PNG
    // Records the copy of one image into the read-back buffer at offset (depth and stencil together: the depth's
    // words, then the stencil bytes).
    void read_back(const vk::Image& image, VkImageAspectFlags aspect, u32 width, u32 height, VkDeviceSize offset);
    // Space in the read-back buffer for the batch, storing the batch so far first when the buffer is full.
    bool readback_alloc(VkDeviceSize bytes, VkDeviceSize& offset);
    VkRenderPass render_pass(VkFormat color, VkFormat depth);
    VkShaderModule module(const std::vector<u32>& spirv);
    VkShaderModule vertex_shader(const KelvinState& s, std::string& error);
    VkShaderModule fragment_shader(const psh::Key& key, std::array<psh::SamplerKind, 4>& kinds, std::string& error);
    VkPipeline pipeline(const KelvinState& s, VkShaderModule vs, VkShaderModule fs, VkPrimitiveTopology topology, Surface& t, u32 vertex_layout, u32 depth_mode, std::string& error);
    // The pipeline of a clear that writes only some colour channels (the colour comes in as the blend constants).
    VkPipeline clear_pipeline(VkRenderPass pass, VkColorComponentFlags mask, std::string& error);
    const TextureEntry* texture(const KelvinState& s, u32 stage, std::string& error);
    // A render target the stage can sample in place (same address and size, a format of the same bytes), or null.
    TextureEntry* surface_texture(const TextureStage& ts, u32 phys);
    TextureEntry* snapshot_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear, bool x8);
    TextureEntry* depth_snapshot_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear);
    // Why the last surface_texture call could not sample a target in place (for the unload report's forced lines).
    const char* sample_miss_{""};
    TextureEntry* subrect_texture(Surface& s, const TextureStage& ts, u32 phys, u32 x0, u32 y0, u32 w, u32 h, bool x8);
    // A cube map whose six faces are render targets drawn this frame or the last: the faces copied into a cube image on
    // the GPU (at their internal resolution), retaken when any face was drawn into since; or null.
    TextureEntry* cube_texture(const TextureStage& ts, u32 phys);
    struct CubeEntry {
        vk::Image image;
        VkImageView x8_view{VK_NULL_HANDLE};
        TextureEntry tex, x8_tex;
        std::array<u64, 6> serials{};
        std::array<const void*, 6> faces{};  // the targets the copies came from
    };
    std::unordered_map<u32, CubeEntry> cubes_;
    void retire_cubes();
    // A mip-mapped texture whose every level is a render target drawn this frame or the last (a chain of downsampled
    // copies): the levels copied into one image's mips on the GPU, retaken when a level was drawn into since; or null.
    TextureEntry* mip_texture(const TextureStage& ts, u32 phys);
    struct MipEntry {
        vk::Image image;
        VkImageView x8_view{VK_NULL_HANDLE};
        TextureEntry tex, x8_tex;
        std::vector<u64> serials;
        std::vector<const void*> levels;  // the targets the copies came from
    };
    std::unordered_map<u32, MipEntry> mips_;
    void retire_subs(Surface& s);
    Surface* draw_target_{nullptr};  // the target of the draw whose textures are being looked up
    // The depth buffer at each zeta address lives in one target's depth image at a time: the target that last drew with
    // it. Another target binding the same address takes the contents first (a GPU copy, or through RAM).
    std::unordered_map<u32, Surface*> depth_owner_;
    // cleared: the caller clears the whole depth buffer (and its stencil) before anything reads it, so what the address
    // held is not loaded (Half-Life 2's four targets of two shapes sharing one depth buffer, each through RAM).
    void bind_depth(Surface& t, bool cleared = false);
    bool borrow_depth(Surface& t, Surface& owner);  // D37: t draws with owner's depth image; false when it cannot
    void end_borrow(Surface& t);
    void end_borrows_from(Surface& owner);  // the lender goes away: its borrowers draw with their own images again
    void write_back_held(Surface& s);  // the bound image to the address it holds, when newer there
    void upload_depth_from_ram(Surface& t);
    void release_depth(Surface& s);  // a target going away: its depth contents to RAM, its ownership dropped
    // Parked depth images (Surface::parked). swap_depth exchanges the bound image with a parked one; park_for takes z's
    // parked image back, or parks the bound one and binds a spare (made, or the least recently used written back);
    // flush_parked writes back the dirty ones over a range and keeps them; drop_parked writes back and lets them go.
    static constexpr std::size_t kMaxParked = 3;
    u64 park_clock_{0};
    void swap_depth(Surface& s, Surface::ParkedDepth& p);
    bool park_for(Surface& t, u32 z);  // true: z's own image came back (no load needed)
    bool spare_depth(Surface& t, Surface::ParkedDepth& p);
    void write_back_parked(Surface& s, Surface::ParkedDepth& p);
    void flush_parked(u64 b0, u64 b1);
    bool drop_parked(Surface& s, u64 b0, u64 b1, bool write);
    void drop_parked(u64 b0, u64 b1);
    TextureEntry* parked_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear, bool depth16);
    // Finishes with a target: what it rendered reaches RAM, its depth goes back to its address, its images retire once
    // the GPU is done with them. Returns the next entry.
    std::unordered_map<u32, std::unique_ptr<Surface>>::iterator erase_surface(std::unordered_map<u32, std::unique_ptr<Surface>>::iterator it);
    u64 vblank_index_{0};  // vblanks seen (write_back_at), for retiring targets nothing uses
    TextureEntry* zeta_color_texture(Surface& s, const TextureStage& ts, u32 phys, u32 w, u32 h, bool linear, bool x8, bool abgr = false);
    bool init_zeta_convert();
    bool zc_buffers(u32 texels = 0);
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
    void remember_pages(Surface& s);  // s.page_hashes from RAM as it is now
    // The hash of the part of RAM page `page` the target occupies (a CPU write elsewhere in the page is not the target's).
    u64 target_page_hash(const Surface& s, u32 page) const;
    bool texture_current(TextureEntry& e);  // RAM still holds what e was decoded from (rehashed when pages were written)  // the decoded textures over a range (RAM now holds what a target wrote)
    void hash_pages(TextureEntry& e) const;  // e.page_hashes and e.palette_hash from RAM as it is now
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
    void fresh_image(const vk::Image& image, VkCommandBuffer cb = VK_NULL_HANDLE);  // into cb_ unless another is named
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

    // The deferred stores (set_deferred_stores). begin_store hands out where a store of [phys, phys + len) writes (RAM,
    // or a copy of its whole pages as the renderer sees them) and end_store journals that copy. ram_view is a read of
    // RAM with the journal over it (`keep` holds the copy when one is needed).
    u8* begin_store(u32 phys, u64 len);
    void end_store();
    const u8* ram_view(u32 phys, u64 len, std::vector<u8>& keep) const;
    MemoryView mem_view(u32 phys, u64 len, std::vector<u8>& keep) const;
    // The CPU's pages keep their bytes across a store of the rest of a target: read before, put back after (deferred:
    // in the journal, or out of it where RAM holds them).
    std::vector<u8> keep_pages(const std::vector<u32>& pages) const;
    void put_back_pages(const std::vector<u32>& pages, const std::vector<u8>& kept);
    // The CPU wrote these pages: copies over the bytes RAM held are dropped (a page reported but unchanged keeps its copy).
    void drop_overwritten_stores(const std::vector<u32>& pages);
    bool deferred_{false};
    struct JournalPage {
        std::unique_ptr<u8[]> base, bytes;  // what RAM held when first stored over, and what the renderer stored
        u32 tag{0};                         // the stream generation of the last store
    };
    mutable std::mutex journal_mutex_;
    std::unordered_map<u32, JournalPage> journal_;
    std::atomic<std::size_t> journal_pages_{0};
    mutable JournalStats journal_stats_;
    std::vector<u8> store_copy_, store_base_;
    u32 store_first_page_{0}, store_pages_{0};
    u32 stream_generation_{0};  // where the renderer is in the stream (cpu_wrote)
    u32 async_tag_{0};          // the stream generation of the write-backs in flight (finish_write_backs_async)

    // The frames in flight. Each owns command buffers and the fences their submissions signal, a descriptor pool and a
    // region of every ring, so a submit does not wait: a frame's resources are reused only once its fence signals.
    static constexpr u32 kFramesInFlight = 3;
    struct Frame {
        // Its submissions: each has a command buffer and a fence of its own, so the frame goes on recording into the next
        // while the GPU runs the last (a submit no longer moves to another frame in flight and waits for it); `used` of
        // them so far, and the fence of the last, which signals once everything submitted before it has run too.
        std::vector<VkCommandBuffer> cbs;
        std::vector<VkCommandBuffer> upload_cbs;  // each submission's new-texture uploads, submitted just ahead of it
        std::vector<VkFence> fences;
        u32 used{0};
        VkFence last{VK_NULL_HANDLE};
        VkDescriptorPool pool{VK_NULL_HANDLE};
        u64 serial{0};  // the submission that last used it (0: never submitted)
        // Objects retired while this frame recorded: commands already recorded (here or earlier) may use them.
        std::vector<vk::Image> dead_images;
        std::vector<VkImageView> dead_views;
        std::vector<VkFramebuffer> dead_framebuffers;
        // Draw counting: the queries this frame used, the ticket of its first, each draw's anti-aliasing scale, and
        // whether its results have been read (they are, before the pools are reset for the frame's next use).
        VkQueryPool occlusion{VK_NULL_HANDLE};
        u32 queries{0};
        u64 first_ticket{1};
        std::vector<u32> aa_scale;
        std::vector<u32> res_scale;  // and its internal resolution's samples per stored pixel (sc squared)
        bool harvested{true};
    };
    static constexpr u32 kQueriesPerFrame = 4096;
    static constexpr u32 kSubmitDraws = 32;  // counted draws a submission carries at most (<= kQueriesPerFrame)
    bool count_draws_{false}, count_zpass_{false};
    u64 next_ticket_{1}, last_ticket_{0};
    std::unordered_map<u64, DrawCounts> counts_;
    void harvest(Frame& fr);
    std::array<Frame, kFramesInFlight> frames_{};
    u32 slot_{0};                       // the recording frame
    u64 submitted_{0}, completed_{0};  // submissions made, and the last one known to have finished
    VkCommandBuffer cb_{VK_NULL_HANDLE};  // the recording frame's command buffer
    bool recording_{false};
    // A new texture's upload goes to the submission's upload command buffer, which the queue runs before its draws: no
    // draw recorded so far can sample an image just made, and the render pass stays open (an upload in cb_ ended it, and
    // at 6x every pass begun again reloaded and stored the whole target: Crimson Skies met 176 new textures in a frame).
    VkCommandBuffer upload_cb();  // begun on first use in a submission
    VkCommandBuffer upload_cb_{VK_NULL_HANDLE};
    bool uploads_recording_{false};
    bool hoist_uploads_{true};  // MV_HOIST_UPLOADS=0: uploads in cb_ as before (a control)
    bool pass_open_{false};
    Surface* pass_target_{nullptr};
    VkFramebuffer pass_fb_{VK_NULL_HANDLE};  // the open pass's framebuffer (the target's own, or its borrow_fb)

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
    // Decoded textures by what identifies them (address, format, CONTROL1, rectangle, palette), a fixed-size key: a
    // lookup allocates nothing.
    using TextureKey = std::array<u32, 5>;
    struct TextureKeyHash {
        std::size_t operator()(const TextureKey& k) const {
            u64 h = 0xcbf29ce484222325ull;
            for (const u32 w : k) h = (h ^ w) * 0x100000001b3ull;
            return static_cast<std::size_t>(h ^ (h >> 29));
        }
    };
    std::unordered_map<TextureKey, std::unique_ptr<TextureEntry>, TextureKeyHash> textures_;
    // The last pipeline found for draws and its rasterizer key's bytes: a run of draws under one state finds it without
    // building or hashing a key. Cleared with the pipelines.
    std::array<u8, 128> last_raster_key_{};
    VkPipeline last_raster_pipeline_{VK_NULL_HANDLE};
    // The last binding of each stage: the registers it was built from and what it resolved to, so a draw that
    // changes nothing about its textures skips the address maths, the handle lookups and the hashing.
    struct StageCache {
        std::array<u32, 11> regs{};
        u32 phys{0}, length{0};
        TextureEntry* entry{nullptr};
    };
    std::array<StageCache, 4> stage_cache_{};
    // The guest memory every target's images may stand for (colour, its depth address, the depth its image holds, the
    // parked ones), gathered once per draw: a texture that overlaps none of it cannot be a target, so its lookup skips
    // the scans of every target (three per stage per draw before, most of the texture time in heavy frames).
    std::vector<std::pair<u64, u64>> target_ranges_;
    bool target_ranges_valid_{false};
    bool overlaps_target(u32 phys, u32 length);
    // Background compiles: a draw whose shader or pipeline a worker is still making waits for it, up to this much
    // waiting per vblank (MV_PENDING_WAIT_MS, default 10), before it is skipped. Skipping at once left whole effects
    // undrawn for frames whenever the cache was cold (the boot animation's X drawn stretched, now and then).
    i64 pending_wait_budget_ns_{0};
    i64 pending_wait_per_vblank_ns_{10'000'000};
    bool wait_a_little();
    // DXT textures uploaded as their blocks (BC1-3) where the device samples them: decoding them on the CPU took 3-4 ms
    // a texture inside the frame (a racing title's dips) and uploaded four to eight times the bytes.
    bool native_dxt_requested_{true}, native_dxt_{false};
    std::unordered_map<u64, VkSampler> samplers_;
    std::map<std::pair<u64, u32>, VkSampler> border_samplers_;  // samplers that read the stage's border colour, by it
    VkDescriptorSetLayout set_layout_{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
    Ring vertex_ring_, index_ring_, uniform_ring_, staging_;
    vk::Buffer readback_;
    struct PendingWriteBack {
        Surface* surface{nullptr};
        bool zeta{false};
        VkDeviceSize offset{0};  // where its copy lands in the read-back buffer
        bool rows{false};  // a scaled depth buffer read back a row per region: thinned when stored
        u64 serial{0};     // the target's draw serial when queued: drawn into since, it stays dirty once stored
    };
    std::vector<PendingWriteBack> write_backs_;
    void store_write_backs(const std::vector<PendingWriteBack>& list);
    unsigned wb_why_{Stats::kWbAll};  // the cause the write-backs being queued count under
    VkDeviceSize readback_cursor_{0};
    // The batch in flight (finish_write_backs_async): its entries, the fence of the submission that carries it, that
    // submission's serial; MV_ASYNC_WRITEBACK=0 stores every write-back at once (the control).
    std::vector<PendingWriteBack> async_wb_;
    VkFence async_fence_{VK_NULL_HANDLE};
    u64 async_serial_{0};
    std::atomic<bool> async_pending_{false};
    bool async_wb_on_{true};
    u32 descriptor_sets_used_{0};  // from the recording frame's pool
    // The vertex uniform block being built, and the last block of each kind uploaded in the recording frame.
    std::array<float, vsh::kUniformBytes / 4> vs_uniforms_{};
    // A draw's fetched vertices and rewritten index lists, kept so their buffers serve draw after draw.
    FetchedVertices fetched_;
    FetchedVertices* prefetched_{nullptr};  // set_prefetched: taken (swapped into fetched_) by the next draw
    std::vector<u32> draw_indices_, quad_indices_;
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
    std::vector<u32> spirv_cached(bool vertex, const std::string& glsl);  // the disk cache's copy, or empty
    // Compiles and stores in the disk cache; touches no renderer state (worker threads call it).
    std::vector<u32> spirv_compile(bool vertex, const std::string& glsl, const char* name, std::string& log) const;
    // The background compiles: SPIR-V or a pipeline being made on a worker, by the cache key that will hold it.
    struct PendingShader {
        std::atomic<bool> done{false};
        std::vector<u32> spirv;
        std::string log;
    };
    struct PipelineBuild;  // a pipeline's create-info with everything it points at (the .cpp)
    // SPIR-V ready for `key`: true with `spirv` (empty + `log` on a compile error), or false while a worker compiles it.
    bool spirv_ready(bool vertex, const std::string& key, const std::string& glsl, const char* name, std::vector<u32>& spirv, std::string& log);
    bool shader_pending(bool vertex, const std::string& key) const;
    void submit(std::function<void()> job);
    void stop_workers();
    bool async_compile_{false};
    std::atomic<u32> min_anisotropy_{1};
    // The internal resolution (D34): the scale new targets get, and the one asked for (applied by apply_scale).
    u32 scale_{1};
    std::atomic<u32> requested_scale_{1};
    // At the vblank, or before the first target: every target written back and dropped, the new scale taken.
    void apply_scale();
    // The scale a new target of this colour format and stored size gets: the renderer's, or less where the device cannot
    // blit the format or make an image that large.
    u32 scale_for(VkFormat color, u32 width, u32 height) const;
    // Colour between the scaled image and its stored-size copy (lo): down before a write-back, up after a load.
    bool ensure_lo(Surface& s);
    // The stored-size depth copy: false when the device cannot blit the format (the rows are read back instead).
    bool ensure_zeta_lo(Surface& s);
    void blit_zeta_lo(Surface& s);
    void blit_lo(Surface& s, bool down);
    // The scanout target's scaled picture, read back after its write-back at the vblank into one of the target's two
    // capture buffers, with a fence of its own that the frontend waits for (D36).
    void capture_display(Surface& s);
    // A scaled target's depth write-back: the middle row of every sc-row block, whole (depth words, then stencil bytes,
    // rows padded to four bytes), then thinned to the stored size in the layout store_zeta takes.
    std::size_t zeta_rows_bytes(const Surface& s) const;
    void read_back_zeta_rows(Surface& s, VkDeviceSize offset, u32 y0, u32 rows);
    void thin_zeta_rows_into(const Surface& s, const u8* in, u32 y0, u32 rows, u8* out) const;
    void write_back_zeta_banded(Surface& s);  // rows too many for the read-back buffer at once: in bands, synchronously
    std::vector<u8> thin_zeta_rows(const Surface& s, const u8* rows) const;
    struct Capture {
        u32 phys{0}, width{0}, height{0}, pitch{0}, bpp{0}, sc{1};
        bool valid{false};         // the buffer holds (once its fence signals) the picture these describe
        bool xrgb{false};          // converted on the GPU to XRGB8888 with the colour table applied (else the target's bytes)
        bool overlaid{false};      // the video overlay drawn in by the same pass
        bool submitted{false};     // its fence was submitted since it was last reset
        bool retired{false};       // the scale changed while the frontend held it: it goes once let go
        u64 serial{0};             // the target's draw serial when read
        u64 taken{0};              // when (capture_clock_): the newer of a target's two is its latest picture
        vk::Buffer buffer;         // sc x width by sc x height, tightly packed (host-visible: read in place)
        VkFence fence{VK_NULL_HANDLE};
    };
    static constexpr std::size_t kMaxCaptures = 6;  // two each for a triple buffer's targets
    std::vector<std::unique_ptr<Capture>> captures_;
    u64 capture_clock_{0};
    const Capture* shown_{nullptr};          // the picture the last vblank scanned out (null: RAM holds it)
    mutable const Capture* held_{nullptr};   // the picture the frontend reads (scaled_picture to picture_done)
    Capture* last_capture_{nullptr};         // the capture that last used the conversion's descriptor set and table
    void destroy_capture(Capture& c);        // waits for its copy, then frees it
    mutable std::mutex display_mutex_;
    // The capture's conversion on the GPU: a compute pass from the target image to XRGB8888 words in the capture's
    // buffer, the colour table applied (made on first use; failing that, captures copy the target's bytes).
    bool init_display_convert();
    VkDescriptorSetLayout dc_set_layout_{VK_NULL_HANDLE};
    VkPipelineLayout dc_layout_{VK_NULL_HANDLE};
    VkPipeline dc_pipeline_{VK_NULL_HANDLE};
    VkDescriptorPool dc_pool_{VK_NULL_HANDLE};
    VkDescriptorSet dc_set_{VK_NULL_HANDLE};
    VkSampler dc_sampler_{VK_NULL_HANDLE};
    vk::Buffer dc_lut_;
    bool dc_failed_{false};
    std::array<u8, 768> display_lut_{};
    // The picture conversion's push constants (init_display_convert's shader): the image size, then the video overlay:
    // on, the scanout's origin in the image, its rectangle relative to that (scaled pixels, end exclusive), the source
    // start split into whole texels and the 20-bit fraction, the steps per scaled pixel, the source's size and pitch, its
    // byte order, the colour key.
    struct DisplayConvertPush {
        u32 width, height, ov_on, ox, oy, x0, y0, x1, y1;
        u32 s_hi, s_lo, t_hi, t_lo, dsdx, dtdy, in_w, in_h, pitch, yuy2, key_on, key;
    };
    // The overlay's part of the push constants and its source in dc_ov_; true when the pass finishes the picture with it
    // (false: grab_frame draws it on the CPU, as when the steps would not fit the shader's 32-bit arithmetic).
    bool overlay_push(const Surface& s, DisplayConvertPush& push);
    std::optional<Overlay> display_ov_;
    // The frontend's images (hand_over), one per sync index, each with its own staging buffer, command buffer and fence.
    bool handoff_on_{false};
    struct Handoff {
        vk::Image image;
        VkImageView view{VK_NULL_HANDLE};
        VkImageViewCreateInfo info{};
        vk::Buffer staging;
        VkCommandBuffer cb{VK_NULL_HANDLE};
        VkFence fence{VK_NULL_HANDLE};
        bool pending{false};  // its fence was submitted
        bool fresh{false};    // still UNDEFINED: the next copy moves it to GENERAL first
    };
    std::vector<Handoff> handoffs_;
    VkCommandPool handoff_pool_{VK_NULL_HANDLE};
    void destroy_handoff(Handoff& h);
    vk::Buffer dc_ov_;        // the overlay's source bytes, copied from RAM for the pass
    u32 scan_phys_{0};        // the scanout's start (write_back_at): the overlay's rectangle is relative to it
    bool display_lut_on_{false};
    u32 zc_capacity_{0};  // texels the depth-to-colour buffers hold
    std::unordered_map<std::string, std::shared_ptr<PendingShader>> vs_pending_, fs_pending_;
    std::unordered_map<std::string, std::shared_ptr<PipelineBuild>> pipelines_pending_;
    std::vector<std::thread> workers_;
    std::mutex work_mutex_;
    std::condition_variable work_cv_;
    std::deque<std::function<void()>> work_;
    bool work_stop_{false};
    void save_pipeline_cache();
    VkFormat depth24_{VK_FORMAT_D24_UNORM_S8_UINT};
    Stats stats_;
    // Debug overrides from MV_RENDER_DEBUG (comma separated): flat (white fragments), nodepth, nocull, noblend.
    bool debug_flat_{false}, debug_nodepth_{false}, debug_nocull_{false}, debug_noblend_{false}, debug_shadow_pass_{false}, debug_show_v0_{false}, debug_skip_blend_{false}, debug_draw_id_{false}, debug_flip_front_{false}, debug_ff_dump_{false}, debug_wb_all_{false};
    u64 drawid_from_{0}, drawid_to_{~0ull};  // MV_DRAWID_VBLANKS: the drawid lines of these vblanks only
    bool drawid_on() const { return debug_draw_id_ && vblank_index_ >= drawid_from_ && vblank_index_ <= drawid_to_; }
    u64 frame_index_{1};  // frames ended so far, plus one
    bool attachments_written_{false}, sampled_since_barrier_{false};  // hazards the next surface_barrier() covers
    bool depth16_sampled_{false}, depth24_sampled_{false};  // the device samples (and filters) these depth formats
    u32 signed_half_{0};  // the half of a signed blend whose pipeline is asked for (psh::Key::signed_half)
    bool debug_tex_log_{false};
    bool debug_no_fog_{false};
    bool debug_vsh_dump_{false};
    std::unordered_map<VkShaderModule, u32> vs_debug_ids_;  // drawid/vshdump: each vertex shader's dump number
    bool debug_roundtrip_{false};  // MV_RENDER_DEBUG=roundtrip: sampled targets go through RAM as before (an A/B control)
    bool debug_depth_rows_{false};  // MV_RENDER_DEBUG=depthrows: scaled depth read back a row per region (an A/B control)
    u32 frame_draws_{0};  // draws since the frame began (the index mv-pbcap draws lists)
};

}  // namespace mv::pgraph
