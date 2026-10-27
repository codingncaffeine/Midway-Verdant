// mv-pbcap: reads a pushbuffer capture written by the emulator (--capture) and reports what is in it: the
// pusher runs, the methods by class, the frames (one per semaphore release of the 3D class), the draws and
// clears per frame, the RAM pages carried. The renderer's offline instrument.
//   mv-pbcap summary FILE        the report
//   mv-pbcap methods FILE        every method as "class method value", one per line (diffs against traces)
//   mv-pbcap replay FILE [N]     run the Kelvin state machine over the capture (N frames, default all): draws, vertices,
//                                primitives, texture and surface formats per frame, fetch errors
//   mv-pbcap shaders FILE FRAME DRAW  print the state, the vertex program listing and both GLSL shaders of one draw
//   mv-pbcap render FILE DIR [N] render N frames (default all) through the Vulkan backend; every surface a frame touched
//                                goes to DIR as frameNN_ADDRESS.png, and the scanout surface as frameNN_scanout.png
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <string>
#include <vector>

#include "nv2a/capture.h"
#include "nv2a/pgraph/kelvin.h"
#include "nv2a/pgraph/replay.h"
#include "nv2a/render_thread.h"
#include "nv2a/vulkan/vk_device.h"
#include "nv2a/vulkan/vk_renderer.h"
#include "nv2a/shaders/psh_translate.h"
#include "nv2a/shaders/vsh_translate.h"
#include "nv2a/pgraph/texture.h"
#include "debug/png.h"
#include <filesystem>
#include <set>

namespace {

struct ClassStats {
    std::map<mv::u32, mv::u64> by_method;
    mv::u64 total{0};
};

struct Frame {
    mv::u64 methods{0}, draws{0}, clears{0}, pages{0};
    mv::u64 first_cycles{0};
    mv::u32 semaphore_value{0};
};

int summary(const char* path) {
    mv::capture::Reader r;
    std::string error;
    if (!r.open(path, error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    std::map<mv::u32, ClassStats> classes;
    std::vector<Frame> frames(1);
    mv::u64 runs = 0, pages = 0, page_bytes = 0, objects = 0, registers = 0, methods = 0;
    mv::u64 cycles = 0;
    mv::u32 last_scanout = 0;
    mv::capture::Record rec;
    while (r.next(rec)) {
        switch (rec.type) {
            case mv::capture::Rec::Pages:
                ++pages;
                page_bytes += static_cast<mv::u64>(rec.page_count) * mv::capture::kPageSize;
                frames.back().pages += rec.page_count;
                break;
            case mv::capture::Rec::Run:
                ++runs;
                cycles = rec.cycles;
                break;
            case mv::capture::Rec::Object:
                ++objects;
                break;
            case mv::capture::Rec::Register:
                ++registers;
                if (rec.offset == 0x600800) last_scanout = rec.value;
                break;
            case mv::capture::Rec::Method: {
                ++methods;
                auto& c = classes[rec.cls];
                ++c.total;
                ++c.by_method[rec.method];
                Frame& f = frames.back();
                if (f.methods == 0) f.first_cycles = cycles;
                ++f.methods;
                if (rec.cls == 0x97) {
                    if (rec.method == 0x17FC && rec.value != 0) ++f.draws;
                    if (rec.method == 0x1D94) ++f.clears;
                    if (rec.method == 0x1D70) {
                        f.semaphore_value = rec.value;
                        frames.emplace_back();
                    }
                }
                break;
            }
        }
    }
    if (!r.error().empty()) std::fprintf(stderr, "mv-pbcap: %s (reporting what was read)\n", r.error().c_str());
    std::printf("ram %u MiB; %llu pusher runs, %llu methods, %llu objects, %llu register writes, %llu page records (%llu KiB)\n",
                r.ram_size() >> 20, static_cast<unsigned long long>(runs), static_cast<unsigned long long>(methods),
                static_cast<unsigned long long>(objects), static_cast<unsigned long long>(registers), static_cast<unsigned long long>(pages),
                static_cast<unsigned long long>(page_bytes >> 10));
    std::printf("scanout start (last PCRTC START write): %08x\n", last_scanout);
    for (const auto& [cls, stats] : classes) {
        std::printf("class %04x: %llu methods, %zu distinct\n", cls, static_cast<unsigned long long>(stats.total), stats.by_method.size());
        std::vector<std::pair<mv::u64, mv::u32>> top;
        for (const auto& [m, n] : stats.by_method) top.emplace_back(n, m);
        std::sort(top.rbegin(), top.rend());
        std::size_t shown = 0;
        for (const auto& [n, m] : top) {
            if (shown++ >= 24) break;
            std::printf("  %04x x%llu", m, static_cast<unsigned long long>(n));
        }
        std::printf("\n");
    }
    std::printf("frames (ended by a 3D semaphore release): %zu\n", frames.size() - (frames.back().methods == 0 ? 1 : 0));
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const Frame& f = frames[i];
        if (f.methods == 0) continue;
        std::printf("  frame %zu: %llu methods, %llu draws, %llu clears, %llu pages, at %.3f ms, semaphore %08x\n", i,
                    static_cast<unsigned long long>(f.methods), static_cast<unsigned long long>(f.draws),
                    static_cast<unsigned long long>(f.clears), static_cast<unsigned long long>(f.pages),
                    static_cast<double>(f.first_cycles) / 733333.333, f.semaphore_value);
    }
    return 0;
}

int methods(const char* path) {
    mv::capture::Reader r;
    std::string error;
    if (!r.open(path, error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    mv::capture::Record rec;
    while (r.next(rec)) {
        switch (rec.type) {
            case mv::capture::Rec::Method: std::printf("[%04x] %04x <- %08x\n", rec.cls, rec.method, rec.value); break;
            case mv::capture::Rec::Run: std::printf("run at %llu cycles get %08x put %08x\n", static_cast<unsigned long long>(rec.cycles), rec.get, rec.put); break;
            case mv::capture::Rec::Pages: std::printf("pages %u + %u\n", rec.first_page, rec.page_count); break;
            case mv::capture::Rec::Object:
                std::printf("object %08x: %08x %08x %08x %08x\n", rec.handle, rec.dw[0], rec.dw[1], rec.dw[2], rec.dw[3]);
                break;
            case mv::capture::Rec::Register: std::printf("register %06x <- %08x\n", rec.offset, rec.value); break;
        }
    }
    if (!r.error().empty()) {
        std::fprintf(stderr, "mv-pbcap: %s\n", r.error().c_str());
        return 1;
    }
    return 0;
}

// The replay instrument: counts what each frame asks of a renderer.
struct CountingSink : mv::pgraph::RenderSink {
    const mv::pgraph::Replay* replay{nullptr};
    struct FrameCount {
        mv::u64 draws{0}, vertices{0}, indices{0}, clears{0}, errors{0};
        std::map<mv::pgraph::Primitive, mv::u64> primitives;
        std::set<mv::u32> surface_formats, texture_formats, stage_programs;
        std::set<mv::u32> program_modes;
        std::string first_error;
    } frame;
    void clear(const mv::pgraph::KelvinState& s, mv::u32) override {
        ++frame.clears;
        frame.surface_formats.insert(s.surface().format);
    }
    void draw(const mv::pgraph::KelvinState& s, const mv::pgraph::DrawBatch& batch) override {
        ++frame.draws;
        frame.surface_formats.insert(s.surface().format);
        frame.program_modes.insert(s.transform_mode());
        frame.stage_programs.insert(s.reg(mv::pgraph::mthd::kShaderStageProgram));
        for (mv::u32 t = 0; t < 4; ++t)
            if (s.texture(t).enabled() && s.shader_stage_program(t) != 0) frame.texture_formats.insert(s.texture(t).format);
        mv::pgraph::FetchedVertices v;
        if (!mv::pgraph::fetch_vertices(s, *replay, replay->memory(), batch, v)) {
            ++frame.errors;
            if (frame.first_error.empty()) frame.first_error = v.error;
            return;
        }
        frame.vertices += v.count;
        frame.indices += v.indices.empty() ? v.count : v.indices.size();
        ++frame.primitives[v.primitive];
    }
};

struct ReplayObserver : mv::pgraph::Replay::Observer {
    CountingSink* sink{nullptr};
    void frame_end(mv::u32 index, mv::u32 semaphore) override {
        auto& f = sink->frame;
        std::printf("frame %u: %llu draws, %llu vertices, %llu indices, %llu clears%s; semaphore %08x\n", index,
                    static_cast<unsigned long long>(f.draws), static_cast<unsigned long long>(f.vertices),
                    static_cast<unsigned long long>(f.indices), static_cast<unsigned long long>(f.clears),
                    f.errors ? (" (" + std::to_string(f.errors) + " FETCH ERRORS: " + f.first_error + ")").c_str() : "", semaphore);
        std::printf("   primitives:");
        for (const auto& [p, n] : f.primitives) std::printf(" %s x%llu", mv::pgraph::primitive_name(p), static_cast<unsigned long long>(n));
        std::printf("; transform modes:");
        for (auto m : f.program_modes) std::printf(" %u", m);
        std::printf("; surfaces:");
        for (auto s : f.surface_formats) std::printf(" %08x", s);
        std::printf("; stage programs:");
        for (auto s : f.stage_programs) std::printf(" %05x", s);
        std::printf("; textures:");
        for (auto t : f.texture_formats) std::printf(" %08x", t);
        std::printf("\n");
        f = CountingSink::FrameCount{};
    }
};

int replay(const char* path, mv::u32 frames) {
    mv::pgraph::Replay r;
    std::string error;
    if (!r.open(path, error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    mv::pgraph::Kelvin kelvin;
    CountingSink sink;
    sink.replay = &r;
    kelvin.set_sink(&sink);
    ReplayObserver observer;
    observer.sink = &sink;
    if (!r.run(kelvin, &observer, error, frames)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    std::printf("%u frames, %llu methods; scanout start %08x; final surface format %08x colour offset %08x pitch %08x clip %ux%u\n", r.frames(),
                static_cast<unsigned long long>(r.methods()), r.scanout_start(), kelvin.state().surface().format,
                kelvin.state().surface().color_offset, kelvin.state().surface().pitch, kelvin.state().surface().clip_width(),
                kelvin.state().surface().clip_height());
    return 0;
}

struct ShaderDumpSink : mv::pgraph::RenderSink {
    const mv::pgraph::Replay* replay{nullptr};
    mv::u32 want_frame{0}, want_draw{0}, frame{0}, n{0};
    bool done{false};
    void clear(const mv::pgraph::KelvinState&, mv::u32) override {}
    void semaphore_release(const mv::pgraph::KelvinState&, mv::u32) override { ++frame; n = 0; }
    void draw(const mv::pgraph::KelvinState& s, const mv::pgraph::DrawBatch& batch) override {
        if (frame != want_frame || n++ != want_draw || done) return;
        done = true;
        using namespace mv::pgraph;
        const SurfaceState d = s.surface();
        std::printf("surface %08x format %08x %ux%u; alpha test %d func %x ref %x; blend %d %x/%x eq %x; depth test %d func %x write %d; cull %d; color mask %08x; shade %x\n",
                    d.color_offset, d.format, d.clip_width(), d.clip_height(), s.alpha_test(), s.reg(mthd::kAlphaFunc), s.reg(mthd::kAlphaRef), s.blend(),
                    s.reg(mthd::kBlendSfactor), s.reg(mthd::kBlendDfactor), s.reg(mthd::kBlendEquation), s.depth_test(), s.reg(mthd::kDepthFunc), s.depth_write(),
                    s.cull_face(), s.color_mask(), s.reg(mthd::kShadeModel));
        for (mv::u32 t = 0; t < 4; ++t) {
            const TextureStage ts = s.texture(t);
            const DmaObject dma = replay->resolve(s.dma_handle(ts.dma_select() == 2 ? mthd::kDmaB : mthd::kDmaA));
            std::printf("texture %u: mode %u enabled %d format %08x (colour %02x dim %u cube %d levels %u %ux%u rect %ux%u pitch %u) at %08x filter %08x address %08x\n", t,
                        s.shader_stage_program(t), ts.enabled(), ts.format, ts.color_format(), ts.dimensionality(), ts.cubemap(), ts.mip_levels(), 1u << ts.log_width(),
                        1u << ts.log_height(), ts.rect_width(), ts.rect_height(), ts.pitch(), dma.base + ts.offset, ts.filter, ts.address);
            // Its first 48 bytes as the replay's RAM holds them (a small texture's whole mip chain).
            if (const mv::u8* bytes = replay->memory().at(dma.base + ts.offset, 48)) {
                std::printf("   bytes:");
                for (mv::u32 b = 0; b < 48; ++b) std::printf("%s%02x", b % 16 == 0 ? "  " : " ", bytes[b]);
                std::printf("\n");
            }
        }
        std::printf("vertex arrays (format, offset):");
        for (mv::u32 i = 0; i < 16; ++i)
            if ((s.reg(mthd::kVertexDataArrayFormat + 4 * i) >> 4) & 0xFu)
                std::printf(" v%u %08x@%08x", i, s.reg(mthd::kVertexDataArrayFormat + 4 * i), s.reg(mthd::kVertexDataArrayOffset + 4 * i));
        std::printf("\n");
        // The first vertices' raw attribute bytes as the draw's arrays hold them (what a colour or alpha really is).
        std::printf("zeta %08x pitch %08x; cull face %x front face %x\n", d.zeta_offset, d.pitch, s.reg(mthd::kCullFace), s.reg(mthd::kFrontFace));
        std::printf("ranges:");
        for (const auto& [first, count] : batch.ranges) std::printf(" %u+%u", first, count);
        std::printf("; %zu indices, %zu inline words\n", batch.indices.size(), batch.inline_words.size());
        for (std::size_t w = 0; w < batch.inline_words.size(); ++w) std::printf("%s%08x", w % 8 == 0 ? (w ? "\n  " : "  ") : " ", batch.inline_words[w]);
        if (!batch.inline_words.empty()) std::printf("\n");
        const mv::u32 first = batch.ranges.empty() ? (batch.indices.empty() ? 0u : batch.indices[0]) : batch.ranges[0].first;
        for (mv::u32 v = first; v < first + 6; ++v) {
            std::printf("  vertex %u:", v);
            for (mv::u32 i = 0; i < 16; ++i) {
                const mv::u32 fmt = s.reg(mthd::kVertexDataArrayFormat + 4 * i), off = s.reg(mthd::kVertexDataArrayOffset + 4 * i);
                const mv::u32 type = fmt & 0xFu, count = (fmt >> 4) & 0xFu;
                if (count == 0) continue;
                const mv::u32 size = type == 2 ? 4 * count : (type == 0 || type == 4 ? count : (type == 6 ? 4 : 2 * count));
                const DmaObject vd = replay->resolve(s.dma_handle(off & 0x80000000u ? mthd::kDmaVertexB : mthd::kDmaVertexA));
                const mv::u8* p = replay->memory().at(vd.base + (off & 0x7FFFFFFFu) + v * (fmt >> 8), size);
                std::printf(" v%u[", i);
                for (mv::u32 k = 0; p && k < size; ++k) std::printf("%02x", p[k]);
                for (mv::u32 k = 0; p && type == 2 && k < count; ++k) {
                    float f;
                    std::memcpy(&f, p + 4 * k, 4);
                    std::printf(" %g", static_cast<double>(f));
                }
                std::printf("]");
            }
            std::printf("\n");
        }
        std::printf("combiner control %08x; stage programs %08x; other stage input %08x\n", s.reg(mthd::kCombinerControl), s.reg(mthd::kShaderStageProgram), s.reg(mthd::kShaderOtherStageInput));
        for (mv::u32 i = 0; i < 8; ++i)
            std::printf("stage %u: colour icw %08x ocw %08x alpha icw %08x ocw %08x factor0 %08x factor1 %08x\n", i, s.reg(mthd::kCombinerColorIcw + 4 * i),
                        s.reg(mthd::kCombinerColorOcw + 4 * i), s.reg(mthd::kCombinerAlphaIcw + 4 * i), s.reg(mthd::kCombinerAlphaOcw + 4 * i),
                        s.reg(mthd::kCombinerFactor0 + 4 * i), s.reg(mthd::kCombinerFactor1 + 4 * i));
        std::printf("final: cw0 %08x cw1 %08x; specular fog factors %08x %08x\n", s.reg(mthd::kCombinerSpecFogCw0), s.reg(mthd::kCombinerSpecFogCw1), s.reg(0x1E20), s.reg(0x1E24));
        const vsh::Program p = vsh::extract_program(s);
        const vsh::Translation vt = vsh::translate(p, vsh::Options{});
        std::printf("---- vertex program (%zu instructions)\n", p.instructions.size());
        for (const auto& line : vt.disassembly) std::printf("  %s\n", line.c_str());
        std::printf("---- vertex shader GLSL\n%s", vt.glsl.c_str());
        const psh::Translation pt = psh::translate(psh::key_from_state(s));
        std::printf("---- fragment shader GLSL%s\n%s", pt.error.empty() ? "" : (" (ERROR: " + pt.error + ")").c_str(), pt.glsl.c_str());
    }
};

// One line per draw of a frame: where it lands, how it is transformed and shaded, what it samples. The instrument
// for "which draw painted (or failed to paint) that surface".
struct DrawListSink : mv::pgraph::RenderSink {
    const mv::pgraph::Replay* replay{nullptr};
    mv::u32 want_frame{0}, frame{0}, n{0};
    void clear(const mv::pgraph::KelvinState& s, mv::u32 what) override {
        if (frame != want_frame) return;
        const mv::pgraph::SurfaceState d = s.surface();
        std::printf("clear  -> %08x %ux%u (%s) what %x colour %08x\n", d.color_offset, d.clip_width(), d.clip_height(), d.swizzled() ? "swizzled" : "pitch", what, s.clear_color());
    }
    void semaphore_release(const mv::pgraph::KelvinState&, mv::u32) override { ++frame; n = 0; }
    void draw(const mv::pgraph::KelvinState& s, const mv::pgraph::DrawBatch& b) override {
        if (frame != want_frame) return;
        using namespace mv::pgraph;
        const SurfaceState d = s.surface();
        std::string tex;
        for (mv::u32 t = 0; t < 4; ++t) {
            const TextureStage ts = s.texture(t);
            if (!ts.enabled() || s.shader_stage_program(t) == 0) continue;
            const DmaObject dma = replay->resolve(s.dma_handle(ts.dma_select() == 2 ? mthd::kDmaB : mthd::kDmaA));
            tex += std::format(" t{}:{:02x}@{:08x}/{}x{}{}", t, ts.color_format(), dma.base + ts.offset, 1u << ts.log_width(), 1u << ts.log_height(), ts.cubemap() ? "cube" : "");
        }
        std::printf("draw %3u -> %08x %ux%u %s; %s x%zu/%zu; mode %u lit %u; stages %05x combiners %u; blend %d %x/%x eq %x; depth %d func %x; cull %x; alpha %d;%s\n", n++,
                    d.color_offset, d.clip_width(), d.clip_height(), d.swizzled() ? "swizzled" : "pitch", primitive_name(b.primitive), b.indices.size(),
                    b.immediate_vertices.size() / 16, s.transform_mode(), s.reg(mthd::kLightingEnable) & 1u, s.reg(mthd::kShaderStageProgram),
                    s.combiner_stages(), s.blend(), s.reg(mthd::kBlendSfactor), s.reg(mthd::kBlendDfactor), s.reg(mthd::kBlendEquation), s.depth_test(), s.reg(mthd::kDepthFunc), s.cull_face() ? s.reg(mthd::kCullFace) : 0u, s.alpha_test(), tex.c_str());
    }
};

int draws(const char* path, mv::u32 frame) {
    mv::pgraph::Replay r;
    std::string error;
    if (!r.open(path, error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    mv::pgraph::Kelvin kelvin;
    DrawListSink sink;
    sink.replay = &r;
    sink.want_frame = frame;
    kelvin.set_sink(&sink);
    r.run(kelvin, nullptr, error, frame + 1);
    return 0;
}

// The textures one draw samples, decoded from the capture's memory: their means per channel (the alpha is what
// an additive blend or an alpha test lives on) and a PNG of level 0.
struct TextureDumpSink : mv::pgraph::RenderSink {
    const mv::pgraph::Replay* replay{nullptr};
    mv::u32 want_frame{0}, want_draw{0}, frame{0}, n{0};
    std::filesystem::path dir;
    bool done{false};
    void clear(const mv::pgraph::KelvinState&, mv::u32) override {}
    void semaphore_release(const mv::pgraph::KelvinState&, mv::u32) override { ++frame; n = 0; }
    void draw(const mv::pgraph::KelvinState& s, const mv::pgraph::DrawBatch&) override {
        if (frame != want_frame || n++ != want_draw || done) return;
        done = true;
        using namespace mv::pgraph;
        for (mv::u32 t = 0; t < 4; ++t) {
            const TextureStage ts = s.texture(t);
            if (!ts.enabled() || s.shader_stage_program(t) == 0) continue;
            const DmaObject dma = replay->resolve(s.dma_handle(ts.dma_select() == 2 ? mthd::kDmaB : mthd::kDmaA));
            const mv::u32 phys = dma.base + ts.offset;
            DecodedTexture d;
            if (!decode_texture(ts, phys, replay->memory(), nullptr, d)) {
                std::printf("texture %u: format %02x at %08x: %s\n", t, ts.color_format(), phys, d.error.c_str());
                continue;
            }
            const auto& img = d.image(0, 0);
            if (d.pixel != HostPixel::Rgba8) {
                std::printf("texture %u: format %02x at %08x: %ux%u, %s texels\n", t, ts.color_format(), phys, d.width, d.height, d.pixel == HostPixel::R16 ? "16-bit" : "float");
                continue;
            }
            double sum[4]{};
            mv::u32 opaque = 0;
            const mv::u32 npx = d.width * d.height;
            for (mv::u32 i = 0; i < npx; ++i) {
                for (int c = 0; c < 4; ++c) sum[c] += img[i * 4 + static_cast<mv::u32>(c)];
                if (img[i * 4 + 3] != 0) ++opaque;
            }
            std::printf("texture %u: format %02x at %08x: %ux%u levels %u faces %u; mean r %.3f g %.3f b %.3f a %.3f; %.1f%% texels with alpha\n", t,
                        ts.color_format(), phys, d.width, d.height, d.levels, d.faces, sum[0] / npx / 255.0, sum[1] / npx / 255.0, sum[2] / npx / 255.0,
                        sum[3] / npx / 255.0, 100.0 * opaque / npx);
            // A cube map: every face's mean (the line above is face 0's), each face's picture beside the first.
            for (mv::u32 face = 1; face < d.faces; ++face) {
                const auto& fimg = d.image(face, 0);
                double fs[3]{};
                for (mv::u32 i = 0; i < npx; ++i)
                    for (int c = 0; c < 3; ++c) fs[c] += fimg[i * 4 + static_cast<mv::u32>(c)];
                std::printf("   face %u: mean r %.3f g %.3f b %.3f\n", face, fs[0] / npx / 255.0, fs[1] / npx / 255.0, fs[2] / npx / 255.0);
                std::string face_error;
                const auto face_path = dir / std::format("f{:02}_d{:03}_t{}_{:08x}_face{}.png", want_frame, want_draw, t, phys, face);
                if (!mv::png::write(face_path, d.width, d.height, fimg.data(), d.width * 4, face_error)) std::fprintf(stderr, "mv-pbcap: %s\n", face_error.c_str());
            }
            std::string error;
            const auto path = dir / std::format("f{:02}_d{:03}_t{}_{:08x}.png", want_frame, want_draw, t, phys);
            if (!mv::png::write(path, d.width, d.height, img.data(), d.width * 4, error)) std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        }
    }
};

int textures(const char* path, mv::u32 frame, mv::u32 draw_index, const char* dir) {
    mv::pgraph::Replay r;
    std::string error;
    if (!r.open(path, error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    std::filesystem::create_directories(dir);
    mv::pgraph::Kelvin kelvin;
    TextureDumpSink sink;
    sink.replay = &r;
    sink.want_frame = frame;
    sink.want_draw = draw_index;
    sink.dir = dir;
    kelvin.set_sink(&sink);
    r.run(kelvin, nullptr, error, frame + 1);
    if (!sink.done) std::printf("no draw %u in frame %u\n", draw_index, frame);
    return 0;
}

int shaders(const char* path, mv::u32 frame, mv::u32 draw_index) {
    mv::pgraph::Replay r;
    std::string error;
    if (!r.open(path, error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    mv::pgraph::Kelvin kelvin;
    ShaderDumpSink sink;
    sink.replay = &r;
    sink.want_frame = frame;
    sink.want_draw = draw_index;
    kelvin.set_sink(&sink);
    r.run(kelvin, nullptr, error, frame + 1);
    if (!sink.done) std::printf("no draw %u in frame %u\n", draw_index, frame);
    return 0;
}

struct RenderObserver : mv::pgraph::Replay::Observer {
    mv::pgraph::VulkanRenderer* renderer{nullptr};
    std::filesystem::path dir;
    const mv::pgraph::Replay* replay{nullptr};
    mv::u32 scanout{0};
    mv::u32 first_png{0};  // frames before this render but write no picture (a long capture's late frames)
    mv::u32 stride{1};     // from first_png on, a picture every stride-th frame
    // MV_PBCAP_RENDER_THREAD=1: the renderer is the render thread's until frame_end has it (own), and its deferred stores
    // reach RAM before the pictures are read (land).
    std::function<std::unique_lock<std::mutex>()> own;
    std::function<void()> land;
    void register_write(mv::u32 offset, mv::u32 value) override {
        if (offset == 0x600800) scanout = value;
    }
    void frame_end(mv::u32 index, mv::u32) override {
        const auto owned = own ? own() : std::unique_lock<std::mutex>{};
        renderer->end_frame();
        if (land) land();
        std::string error;
        for (const auto& s : renderer->surfaces()) {
            std::printf("   surface %08x %ux%u %s format %x: %llu draws, %llu clears%s\n", s.phys, s.width, s.height, s.swizzled ? "swizzled" : "pitch",
                        s.color_format, static_cast<unsigned long long>(s.draws_this_frame), static_cast<unsigned long long>(s.clears_this_frame),
                        s.rendered_this_frame ? "" : " (untouched)");
            if (!s.rendered_this_frame || index < first_png || (index - first_png) % stride != 0) continue;
            std::vector<mv::u8> rgba;
            if (!renderer->surface_rgba(s, rgba)) continue;
            // MV_PBCAP_ALPHA=1: a 32-bit pitch surface's alpha byte as grey (what a later blend weighs by).
            if (std::getenv("MV_PBCAP_ALPHA") && s.bytes_per_pixel == 4 && !s.swizzled)
                for (mv::u32 y = 0; y < s.height; ++y)
                    for (mv::u32 x = 0; x < s.width; ++x)
                        if (const mv::u8* p = replay->memory().at(s.phys + y * s.pitch + x * 4, 4)) {
                            mv::u8* d = &rgba[(static_cast<std::size_t>(y) * s.width + x) * 4];
                            d[0] = d[1] = d[2] = p[3];
                        }
            const auto path = dir / std::format("frame{:02}_{:08x}.png", index, s.phys);
            if (!mv::png::write(path, s.width, s.height, rgba.data(), s.width * 4, error)) std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
            else std::printf("frame %u: %s (%ux%u, %s, format %x)\n", index, path.string().c_str(), s.width, s.height, s.swizzled ? "swizzled" : "pitch", s.color_format);
        }
        const auto& st = renderer->stats();
        std::printf("frame %u: %llu draws, %llu clears, %llu skipped, %llu pipelines, %llu shaders, %llu textures, %llu write-backs; ms draw %.1f (fetch %.1f, textures %.1f, pipelines %.1f), flush %.1f, write-back %.1f\n", index,
                    static_cast<unsigned long long>(st.draws), static_cast<unsigned long long>(st.clears), static_cast<unsigned long long>(st.skipped),
                    static_cast<unsigned long long>(st.pipelines), static_cast<unsigned long long>(st.shaders),
                    static_cast<unsigned long long>(st.textures_decoded), static_cast<unsigned long long>(st.writebacks),
                    static_cast<double>(st.ns_draw) / 1e6, static_cast<double>(st.ns_fetch) / 1e6, static_cast<double>(st.ns_texture) / 1e6,
                    static_cast<double>(st.ns_pipeline) / 1e6, static_cast<double>(st.ns_flush) / 1e6, static_cast<double>(st.ns_write_back) / 1e6);
        std::printf("   %llu textures sampled straight from a render target\n", static_cast<unsigned long long>(st.surface_samples));
        std::printf("   %llu submissions, %llu waits for the GPU; %llu uniform blocks and %llu descriptor sets reused\n", static_cast<unsigned long long>(st.flushes),
                    static_cast<unsigned long long>(st.waits), static_cast<unsigned long long>(st.uniform_reuses), static_cast<unsigned long long>(st.descriptor_reuses));
        std::printf("   draw parts: shader compiles vs %.1f fs %.1f, indices %.1f, upload %.1f, descriptors %.1f, record %.1f ms\n", static_cast<double>(st.ns_vs) / 1e6, static_cast<double>(st.ns_fs) / 1e6,
                    static_cast<double>(st.ns_indices) / 1e6, static_cast<double>(st.ns_upload) / 1e6, static_cast<double>(st.ns_descriptors) / 1e6,
                    static_cast<double>(st.ns_record) / 1e6);
        std::printf("   shader lookups: vs memo %llu, vs cache %llu; fs memo %llu, fs cache %llu\n", static_cast<unsigned long long>(st.vs_memo_hits),
                    static_cast<unsigned long long>(st.vs_cache_hits), static_cast<unsigned long long>(st.fs_memo_hits), static_cast<unsigned long long>(st.fs_cache_hits));
        std::printf("   write-back gpu copy+wait %.1f ms (textures above include the write-backs a sample forced)\n", static_cast<double>(st.ns_write_back_gpu) / 1e6);
    }
};

// MV_PBCAP_SKIP=FRAME:FIRST-LAST: those draws of that frame never reach the renderer (a pass's state before the
// draws that consume it, or which draw paints a fault).
struct SkipSink : mv::pgraph::RenderSink {
    mv::pgraph::RenderSink* next{nullptr};
    mv::u32 skip_frame{~0u}, skip_first{0}, skip_last{0}, frame{0}, n{0};
    void clear(const mv::pgraph::KelvinState& s, mv::u32 bits) override { next->clear(s, bits); }
    void semaphore_release(const mv::pgraph::KelvinState& s, mv::u32 v) override {
        ++frame;
        n = 0;
        next->semaphore_release(s, v);
    }
    void draw(const mv::pgraph::KelvinState& s, const mv::pgraph::DrawBatch& b) override {
        const mv::u32 i = n++;
        if (frame == skip_frame && i >= skip_first && i <= skip_last) return;
        next->draw(s, b);
    }
};

int render(const char* path, const char* dir, mv::u32 frames, mv::u32 first_png, mv::u32 stride) {
    mv::vk::Device device;
    std::string error;
    if (!device.init(error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    std::printf("vulkan device: %s\n", device.name().c_str());
    mv::pgraph::Replay r;
    if (!r.open(path, error)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    std::filesystem::create_directories(dir);
    mv::pgraph::VulkanRenderer renderer(device, r, r.mutable_ram(), r.memory().size);
    if (!renderer.init(error)) {
        std::fprintf(stderr, "mv-pbcap: renderer: %s\n", error.c_str());
        return 1;
    }
    mv::pgraph::Kelvin kelvin;
    SkipSink skip;
    skip.next = &renderer;
    if (const char* k = std::getenv("MV_PBCAP_SKIP"); k && std::sscanf(k, "%u:%u-%u", &skip.skip_frame, &skip.skip_first, &skip.skip_last) == 3)
        kelvin.set_sink(&skip);
    else
        kelvin.set_sink(&renderer);
    kelvin.set_vertex_memory(&r, r.memory());
    RenderObserver observer;
    observer.renderer = &renderer;
    observer.first_png = first_png;
    observer.stride = stride;
    observer.dir = dir;
    observer.replay = &r;
    // MV_PBCAP_CPU_WROTE=1: the capture's pages reach the renderer as the live machine reports them (cpu_wrote, with a
    // write generation each), not as a plain invalidation. MV_PBCAP_RENDER_THREAD=1 (implies it): the live machine's
    // render thread, wired as Nv2aStub wires it, behind a decoder that fetches each draw's vertices as it is decoded; the
    // replay writes the capture's pages into RAM as soon as it reaches them, ahead of the recording, as the guest does.
    const bool threaded = [] { const char* e = std::getenv("MV_PBCAP_RENDER_THREAD"); return e && *e == '1'; }();
    const bool wrote = threaded || [] { const char* e = std::getenv("MV_PBCAP_CPU_WROTE"); return e && *e == '1'; }();
    std::unordered_map<mv::u32, mv::u32> page_generation;
    mv::u32 generation = 1;
    std::unique_ptr<mv::RenderThread> rt;
    std::unique_ptr<mv::PrefetchSink> prefetch;
    if (threaded) {
        renderer.set_deferred_stores(true);
        rt = std::make_unique<mv::RenderThread>(renderer);
        prefetch = std::make_unique<mv::PrefetchSink>(*rt, r, r.memory());
        kelvin.set_sink(prefetch.get());
        // MV_PBCAP_PUBLISH=N: the replay hands the thread its work every N methods (default 256); 0 = only at a frame's end,
        // the thread then records a whole frame behind the pages the replay has written (the widest lag).
        const mv::u32 publish_every = [] { const char* e = std::getenv("MV_PBCAP_PUBLISH"); return e ? static_cast<mv::u32>(std::strtoul(e, nullptr, 10)) : 256u; }();
        mv::u32 since_publish = 0;
        r.set_method_hook([&, since_publish, publish_every](mv::u32 m, mv::u32 v) mutable {
            rt->method(m, v);
            if (prefetch->take_drew()) rt->commit_packet(kelvin.state().immediate);
            if (m == mv::pgraph::mthd::kSemaphoreRelease) rt->call([](mv::pgraph::VulkanRenderer& rr) { rr.flush_frame(); });
            if (publish_every && ++since_publish == publish_every) {
                since_publish = 0;
                rt->publish();
            }
        });
        observer.own = [&] {
            rt->publish();
            return rt->catch_up();
        };
        observer.land = [&] {
            renderer.land_stores([&](mv::u32 page, mv::u32 tag) {
                const auto it = page_generation.find(page);
                return it != page_generation.end() && it->second > tag;
            });
        };
    }
    r.set_page_observer([&](mv::u32 first_page, mv::u32 count) {
        if (!wrote) {
            renderer.invalidate(first_page * 4096, count * 4096);
            return;
        }
        ++generation;
        std::vector<mv::u32> pages;
        for (mv::u32 p = first_page; p < first_page + count; ++p) {
            pages.push_back(p);
            page_generation[p] = generation;
        }
        if (rt) rt->call([pages, g = generation](mv::pgraph::VulkanRenderer& rr) { rr.cpu_wrote(pages, g); });
        else renderer.cpu_wrote(pages, generation);
    });
    if (rt) {
        const auto owned_blit = [&] {
            rt->publish();
            return rt->catch_up();
        };
        r.set_blit_hooks([&, owned_blit](mv::u32 phys, mv::u32 len) { const auto own = owned_blit(); renderer.write_back_range(phys, len); },
                         [&, owned_blit](mv::u32 phys, mv::u32 len) { const auto own = owned_blit(); renderer.invalidate(phys, len); });
        r.set_blit_ram_copy([&, owned_blit](mv::u32 dp, mv::u32 dpitch, mv::u32 sp, mv::u32 spitch, mv::u32 row, mv::u32 h) {
            const auto own = owned_blit();
            renderer.ram_copy_rows(dp, dpitch, sp, spitch, row, h);
        });
    } else {
        r.set_blit_hooks([&](mv::u32 phys, mv::u32 len) { renderer.write_back_range(phys, len); }, [&](mv::u32 phys, mv::u32 len) { renderer.invalidate(phys, len); });
    }
    if (!r.run(kelvin, &observer, error, frames)) {
        std::fprintf(stderr, "mv-pbcap: %s\n", error.c_str());
        return 1;
    }
    for (const auto& e : renderer.stats().errors) std::printf("renderer note: %s\n", e.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: mv-pbcap summary|methods|replay FILE [FRAMES] | draws FILE FRAME | shaders FILE FRAME DRAW | render FILE DIR [FRAMES [FIRST_PNG [STRIDE]]]\n");
        return 2;
    }
    if (std::strcmp(argv[1], "draws") == 0 && argc >= 4) return draws(argv[2], static_cast<mv::u32>(std::atoi(argv[3])));
    if (std::strcmp(argv[1], "textures") == 0 && argc >= 6)
        return textures(argv[2], static_cast<mv::u32>(std::atoi(argv[3])), static_cast<mv::u32>(std::atoi(argv[4])), argv[5]);
    if (std::strcmp(argv[1], "summary") == 0) return summary(argv[2]);
    if (std::strcmp(argv[1], "methods") == 0) return methods(argv[2]);
    if (std::strcmp(argv[1], "replay") == 0) return replay(argv[2], argc > 3 ? static_cast<mv::u32>(std::atoi(argv[3])) : 0u);
    if (std::strcmp(argv[1], "shaders") == 0 && argc >= 5) return shaders(argv[2], static_cast<mv::u32>(std::atoi(argv[3])), static_cast<mv::u32>(std::atoi(argv[4])));
    if (std::strcmp(argv[1], "render") == 0 && argc >= 4)
        return render(argv[2], argv[3], argc > 4 ? static_cast<mv::u32>(std::atoi(argv[4])) : 0u, argc > 5 ? static_cast<mv::u32>(std::atoi(argv[5])) : 0u,
                      argc > 6 ? std::max(1u, static_cast<mv::u32>(std::atoi(argv[6]))) : 1u);
    std::fprintf(stderr, "mv-pbcap: unknown command %s\n", argv[1]);
    return 2;
}
