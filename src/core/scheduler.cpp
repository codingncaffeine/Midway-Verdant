#include "core/scheduler.h"

#include <chrono>
#include <utility>

namespace mv {

void Scheduler::schedule(TimerEvent& ev, Cycles when) {
    if (ev.scheduled()) {
        ev.when = when;
        sift_up(ev.heap_index);
        sift_down(ev.heap_index);
        return;
    }
    ev.when = when;
    ev.heap_index = heap_.size();
    heap_.push_back(&ev);
    sift_up(ev.heap_index);
}

void Scheduler::cancel(TimerEvent& ev) {
    if (!ev.scheduled()) return;
    const std::size_t i = ev.heap_index;
    const std::size_t last = heap_.size() - 1;
    if (i != last) {
        swap_nodes(i, last);
    }
    heap_.pop_back();
    ev.heap_index = TimerEvent::kNotScheduled;
    if (i < heap_.size()) {
        sift_up(i);
        sift_down(i);
    }
}

std::size_t Scheduler::advance_to(Cycles until) {
    std::size_t fired = 0;
    while (!heap_.empty() && heap_.front()->when <= until) {
        TimerEvent* ev = heap_.front();
        now_ = ev->when > now_ ? ev->when : now_;
        cancel(*ev);
        ++fired;
        const std::string_view name = ev->name;  // the handler may destroy or rename its event
        const auto t0 = std::chrono::steady_clock::now();
        ev->fire(now_);
        EventCost& c = cost_of(name);
        ++c.fired;
        c.host_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }
    if (until > now_) now_ = until;
    return fired;
}

Scheduler::EventCost& Scheduler::cost_of(std::string_view name) {
    for (EventCost& c : costs_)
        if (c.name.data() == name.data()) return c;
    return costs_.emplace_back(EventCost{name});
}

void Scheduler::swap_nodes(std::size_t a, std::size_t b) {
    std::swap(heap_[a], heap_[b]);
    heap_[a]->heap_index = a;
    heap_[b]->heap_index = b;
}

void Scheduler::sift_up(std::size_t i) {
    while (i > 0) {
        const std::size_t parent = (i - 1) / 2;
        if (heap_[parent]->when <= heap_[i]->when) break;
        swap_nodes(i, parent);
        i = parent;
    }
}

void Scheduler::sift_down(std::size_t i) {
    const std::size_t n = heap_.size();
    for (;;) {
        std::size_t smallest = i;
        const std::size_t l = 2 * i + 1, r = 2 * i + 2;
        if (l < n && heap_[l]->when < heap_[smallest]->when) smallest = l;
        if (r < n && heap_[r]->when < heap_[smallest]->when) smallest = r;
        if (smallest == i) break;
        swap_nodes(i, smallest);
        i = smallest;
    }
}

}  // namespace mv
