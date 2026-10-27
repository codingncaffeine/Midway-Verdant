#pragma once
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "core/types.h"
#include "nv2a/pgraph/kelvin.h"

namespace mv::pgraph {
class VulkanRenderer;
}

namespace mv {

// PLAN 12d: the renderer on a thread of its own. The FIFO thread walks the command stream, decodes it, fetches each
// draw's vertices (before the guest may reuse that memory) and answers the fences and polls as it reaches them; this
// thread decodes the same Kelvin methods again with a state machine of its own and records the Vulkan work. The game's
// fences and register reads stop waiting for the recording (Crimson Skies: 14.7 s of 288 on fences and 10 s on the
// GPU lock, while the FIFO thread recorded). MV_RENDER_THREAD=1 turns it on.
class RenderThread {
public:
    using Latched = std::array<std::array<float, 4>, pgraph::kAttributes>;

    explicit RenderThread(pgraph::VulkanRenderer& renderer);
    ~RenderThread();
    RenderThread(const RenderThread&) = delete;
    RenderThread& operator=(const RenderThread&) = delete;

    // The producer (the FIFO thread), in stream order. A Kelvin method:
    void method(u32 method, u32 value);
    // The slot the next draw's vertices are fetched into, and its commit once the producer's decoder has latched the
    // current values the draw leaves behind: the method that ended the draw comes first.
    pgraph::FetchedVertices& packet_slot();
    void commit_packet(const Latched& latched);
    // Anything else that keeps its place in the stream (a frame's flush, the pages the CPU wrote since the last run).
    void call(std::function<void(pgraph::VulkanRenderer&)> f);
    // The thread's state machine starts again (the GPU's reset).
    void reset_decoder();
    // What was queued so far goes to the thread (the producer publishes at the end of every run of the walk, so a
    // thread that holds the GPU lock sees nothing left unpublished).
    void publish();

    // Any thread: everything published so far is executed, then the renderer is the caller's until the lock is
    // released (the vblank's write-backs, a 2D blit, a read of a target).
    std::unique_lock<std::mutex> catch_up();
    // The renderer if the thread is not using it, without catching up (work that can wait for a later try).
    std::unique_lock<std::mutex> try_renderer();

    u64 catch_ups() const { return catch_ups_.load(std::memory_order_relaxed); }
    u64 catch_up_ns() const { return catch_up_ns_.load(std::memory_order_relaxed); }
    u64 busy_ns() const { return busy_ns_.load(std::memory_order_relaxed); }

private:
    enum class Kind : u32 { Method, Call, Reset };
    struct Command {
        Kind kind;
        u32 a, b;  // Method: method, value; Call: the call's index
    };
    struct Packet {
        pgraph::FetchedVertices fetched;
        Latched latched{};
    };
    // A run of the stream as the producer published it. Its vectors keep their buffers when it comes back for reuse.
    struct Batch {
        std::vector<Command> commands;
        std::vector<Packet> packets;
        std::size_t packet_count{0};
        std::vector<std::function<void(pgraph::VulkanRenderer&)>> calls;
        void clear() {
            commands.clear();
            packet_count = 0;
            calls.clear();
        }
        bool empty() const { return commands.empty(); }
    };
    // The thread's decoder hands its draws to the renderer with the producer's fetched vertices and latched values.
    struct Sink : pgraph::RenderSink {
        RenderThread* owner{nullptr};
        void clear(const pgraph::KelvinState& state, u32 clear_bits) override;
        void draw(const pgraph::KelvinState& state, const pgraph::DrawBatch& batch) override;
        void semaphore_release(const pgraph::KelvinState& state, u32 value) override;
        bool latch(pgraph::KelvinState& state) override;
    };

    void main();
    void execute(Batch& b);

    pgraph::VulkanRenderer& renderer_;
    std::unique_ptr<Batch> building_;  // the producer's, unpublished
    std::mutex queue_mutex_;
    std::condition_variable work_cv_, done_cv_;
    std::deque<std::unique_ptr<Batch>> queue_;
    std::vector<std::unique_ptr<Batch>> spare_;
    u64 published_{0}, executed_{0};
    bool stop_{false};
    std::mutex renderer_mutex_;  // the renderer's owner: the thread while it executes a batch, or catch_up's caller
    // The thread's side.
    pgraph::Kelvin kelvin_;
    Sink sink_;
    Batch* current_{nullptr};
    std::size_t next_packet_{0};
    Packet* drawn_{nullptr};  // the packet of the draw just recorded, for its latch
    std::atomic<u64> catch_ups_{0}, catch_up_ns_{0}, busy_ns_{0};
    std::thread thread_;
};

// The producer's decoder sink: a draw's vertices are fetched into the render thread's next packet as the stream is
// decoded, and nothing is recorded here.
class PrefetchSink : public pgraph::RenderSink {
public:
    PrefetchSink(RenderThread& rt, const pgraph::HandleResolver& handles, pgraph::MemoryView ram) : rt_(rt), handles_(handles), ram_(ram) {}
    void clear(const pgraph::KelvinState& state, u32 clear_bits) override;
    void draw(const pgraph::KelvinState& state, const pgraph::DrawBatch& batch) override;
    // Whether the last method ended a draw (its packet is filled and waits for commit_packet).
    bool take_drew() { return std::exchange(drew_, false); }

private:
    RenderThread& rt_;
    const pgraph::HandleResolver& handles_;
    pgraph::MemoryView ram_;
    bool drew_{false};
};

}  // namespace mv
