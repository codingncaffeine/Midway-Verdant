#pragma once
#include <cstddef>
#include <functional>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace mv {

// One event on the master timeline. Owned by its device, which keeps it alive while scheduled.
struct TimerEvent {
    std::string_view name{"?"};
    std::function<void(Cycles now)> fire;
    Cycles when{0};
    std::size_t heap_index{kNotScheduled};  // maintained by the scheduler
    static constexpr std::size_t kNotScheduled = static_cast<std::size_t>(-1);
    bool scheduled() const { return heap_index != kNotScheduled; }
};

// The master timeline: current time in CPU cycles and the events due on it. Single-threaded by contract:
// the machine lock is held by whoever calls into it (the vCPU thread on an exit, the io thread on a tick).
class Scheduler {
public:
    Cycles now() const { return now_; }

    // Schedule or reschedule; firing at a time in the past fires at the next run.
    void schedule(TimerEvent& ev, Cycles when);
    void cancel(TimerEvent& ev);

    // Advance the timeline to `until`, firing every event whose time has come, in time order. An event that
    // reschedules itself for a time <= until fires again in the same call. Returns the number fired.
    std::size_t advance_to(Cycles until);

    // Fire whatever is due at the current time without advancing.
    std::size_t run_due() { return advance_to(now_); }

    bool has_pending() const { return !heap_.empty(); }
    Cycles next_deadline() const { return heap_.empty() ? ~Cycles{0} : heap_.front()->when; }
    std::size_t pending_count() const { return heap_.size(); }

private:
    void sift_up(std::size_t i);
    void sift_down(std::size_t i);
    void swap_nodes(std::size_t a, std::size_t b);

    Cycles now_{0};
    std::vector<TimerEvent*> heap_;
};

}  // namespace mv
