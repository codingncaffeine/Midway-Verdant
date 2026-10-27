#include "nv2a/render_thread.h"

#include <chrono>

#include "nv2a/vulkan/vk_renderer.h"
#if defined(__linux__)
#include <pthread.h>
#endif

namespace mv {
namespace {

u64 ns_since(std::chrono::steady_clock::time_point t0) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

}  // namespace

RenderThread::RenderThread(pgraph::VulkanRenderer& renderer) : renderer_(renderer), building_(std::make_unique<Batch>()) {
    sink_.owner = this;
    kelvin_.set_sink(&sink_);
    thread_ = std::thread([this] { main(); });
}

RenderThread::~RenderThread() {
    publish();
    {
        std::lock_guard g(queue_mutex_);
        stop_ = true;
    }
    work_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void RenderThread::method(u32 method, u32 value) { building_->commands.push_back({Kind::Method, method, value}); }

pgraph::FetchedVertices& RenderThread::packet_slot() {
    Batch& b = *building_;
    if (b.packet_count == b.packets.size()) b.packets.emplace_back();
    return b.packets[b.packet_count].fetched;
}

void RenderThread::commit_packet(const Latched& latched) {
    Batch& b = *building_;
    b.packets[b.packet_count].latched = latched;
    ++b.packet_count;
}

void RenderThread::call(std::function<void(pgraph::VulkanRenderer&)> f) {
    building_->commands.push_back({Kind::Call, static_cast<u32>(building_->calls.size()), 0});
    building_->calls.push_back(std::move(f));
}

void RenderThread::reset_decoder() { building_->commands.push_back({Kind::Reset, 0, 0}); }

void RenderThread::publish() {
    if (building_->empty()) return;
    std::unique_ptr<Batch> next;
    {
        std::lock_guard g(queue_mutex_);
        queue_.push_back(std::move(building_));
        ++published_;
        if (!spare_.empty()) {
            next = std::move(spare_.back());
            spare_.pop_back();
        }
    }
    work_cv_.notify_one();
    building_ = next ? std::move(next) : std::make_unique<Batch>();
}

std::unique_lock<std::mutex> RenderThread::catch_up() {
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock g(queue_mutex_);
        const u64 target = published_;
        done_cv_.wait(g, [&] { return executed_ >= target; });
    }
    std::unique_lock r(renderer_mutex_);
    catch_ups_.fetch_add(1, std::memory_order_relaxed);
    catch_up_ns_.fetch_add(ns_since(t0), std::memory_order_relaxed);
    return r;
}

std::unique_lock<std::mutex> RenderThread::try_renderer() { return std::unique_lock(renderer_mutex_, std::try_to_lock); }

void RenderThread::main() {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "mv-nv2a-render");
#endif
    for (;;) {
        std::unique_ptr<Batch> b;
        {
            std::unique_lock g(queue_mutex_);
            work_cv_.wait(g, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) break;  // stopping, with everything executed
            b = std::move(queue_.front());
            queue_.pop_front();
        }
        {
            std::lock_guard r(renderer_mutex_);
            const auto t0 = std::chrono::steady_clock::now();
            execute(*b);
            busy_ns_.fetch_add(ns_since(t0), std::memory_order_relaxed);
        }
        b->clear();
        {
            std::lock_guard g(queue_mutex_);
            ++executed_;
            spare_.push_back(std::move(b));
        }
        done_cv_.notify_all();
    }
}

void RenderThread::execute(Batch& b) {
    current_ = &b;
    next_packet_ = 0;
    for (const Command& c : b.commands) {
        switch (c.kind) {
            case Kind::Method: kelvin_.method(c.a, c.b); break;
            case Kind::Call: b.calls[c.a](renderer_); break;
            case Kind::Reset: kelvin_ = pgraph::Kelvin(); kelvin_.set_sink(&sink_); break;
        }
    }
    current_ = nullptr;
}

void RenderThread::Sink::clear(const pgraph::KelvinState& state, u32 clear_bits) { owner->renderer_.clear(state, clear_bits); }

void RenderThread::Sink::draw(const pgraph::KelvinState& state, const pgraph::DrawBatch& batch) {
    // Both decoders end a draw at the same method, so the packets come in the order the draws do.
    Batch& b = *owner->current_;
    if (owner->next_packet_ >= b.packet_count) {
        owner->drawn_ = nullptr;  // a stream the two decoders read differently: the renderer fetches from RAM
        owner->renderer_.draw(state, batch);
        return;
    }
    Packet& p = b.packets[owner->next_packet_++];
    owner->drawn_ = &p;
    owner->renderer_.set_prefetched(&p.fetched);
    owner->renderer_.draw(state, batch);
}

void RenderThread::Sink::semaphore_release(const pgraph::KelvinState& state, u32 value) { owner->renderer_.semaphore_release(state, value); }

bool RenderThread::Sink::latch(pgraph::KelvinState& state) {
    Packet* p = std::exchange(owner->drawn_, nullptr);
    if (!p) return false;
    state.immediate = p->latched;
    return true;
}

void PrefetchSink::clear(const pgraph::KelvinState&, u32) {}

void PrefetchSink::draw(const pgraph::KelvinState& state, const pgraph::DrawBatch& batch) {
    pgraph::FetchedVertices& v = rt_.packet_slot();
    if (!pgraph::fetch_vertices(state, handles_, ram_, batch, v) && v.error.empty()) v.error = "vertex fetch failed";
    drew_ = true;
}

}  // namespace mv
