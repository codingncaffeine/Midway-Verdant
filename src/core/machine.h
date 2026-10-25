#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "core/bus.h"
#include "core/kernel_image.h"
#include "core/scheduler.h"
#include "cpu/backend.h"
#include "flash/flash.h"
#include "ide/ata_disk.h"
#include "ide/ide.h"
#include "mcpx/isa_misc.h"
#include "mcpx/lpc.h"
#include "mcpx/pic.h"
#include "mcpx/pit.h"
#include "mcpx/rtc.h"
#include "nv2a/capture.h"
#include "nv2a/nv2a_stub.h"
#include "aci/aci.h"
#include "mcpx/nvnet.h"
#include "apu/apu.h"
#include "pci/pci.h"
#include "smbus/smbus.h"
#include "smbus/smc.h"
#include "usb/ohci.h"
#include "usb/xbox_pad.h"

namespace mv {

namespace vk {
class Device;
}
namespace pgraph {
class VulkanRenderer;
}

struct MachineConfig {
    std::filesystem::path mcpx_rom;
    std::filesystem::path flash_image;
    // Loaded when the file exists; otherwise generated from the kernel's own key at first use and written there.
    std::optional<std::filesystem::path> eeprom_image;
    // A raw hard-disk image (8 GiB for a retail layout); absent = no drive on the cable.
    std::optional<std::filesystem::path> hdd_image;
    bool hdd_locked{true};
    // Compiled shaders and the driver's pipeline cache, kept between runs (absent = compiled afresh every run).
    std::optional<std::filesystem::path> shader_cache_dir;
    // A game disc image (XISO or a full dump) in the DVD tray at power-on; absent = an empty, closed tray.
    std::optional<std::filesystem::path> dvd_image;
    // Pads plugged into the front ports at power-on, from port 1 (0-4).
    unsigned pads{1};
    std::size_t ram_bytes{64u * 1024 * 1024};
    Smc::Revision smc_revision{Smc::Revision::P01};
    Smc::AvPack av_pack{Smc::AvPack::Hdtv};
    // The SMC scratch register the kernel reads at boot (bit 2 asks for the short animation, bit 3 for the dashboard).
    u8 smc_scratch{0};
    std::optional<i64> rtc_base_time;  // unix seconds; unset = host clock
    // Write a pushbuffer capture (methods, RAM pages, objects, scanout registers) there.
    std::optional<std::filesystem::path> capture_path;
    // Instrument: physical RAM words the io thread polls every tick, logging each change with its guest time.
    std::vector<PhysAddr> watch;
    // Instrument: when software writes this SMBus (7-bit address, command) pair, log the CPU's registers and
    // stack under the boot trace — the call chain behind a device write.
    std::optional<std::pair<u8, u8>> break_smbus_write;
    // Instrument: log the CPU's registers and stack the first time execution reaches each of these addresses.
    std::vector<u32> break_eips;
    // Instrument: log the CPU's registers and stack at a guest write to this physical MMIO address (with `value`,
    // only when the written value matches), up to `count` times. Fast mode only.
    struct MmioBreak {
        u32 address{0};
        std::optional<u32> value;
        u32 count{1};
    };
    std::optional<MmioBreak> break_mmio_write;
    // Instrument: press the front-panel reset this many seconds of host time into the run.
    std::optional<double> reset_at_seconds;
    // The live GPU: a Vulkan renderer fed by the pushbuffer, writing frames back into guest RAM.
    bool gpu{false};
    // The CPU: "kvm" (the fast mode) or "p6" (our interpreter, the accurate mode); with `oracle` the
    // interpreter is checked instruction by instruction against a KVM vCPU.
    std::string cpu{"kvm"};
    bool oracle{false};
    // The fast mode's timeline (D16): "cputime" = the guest clock (the work the virtual CPU did), "wall" = the host
    // clock (the old behaviour, a control). MV_GUEST_CLOCK overrides it.
    std::string guest_clock{"cputime"};
    // How much faster than the host time it took guest time runs inside the guest (a host core is many times a
    // 733 MHz Pentium III); the guest's TSC runs as much faster. MV_FAST_CPU_SCALE overrides it.
    double fast_cpu_scale{2.0};
    // The DVD drive's read rate as a multiple of the console's ~8 MB/s (the libretro core's "Disc loading" option: 8, or 1).
    u32 disc_speed{1};
    // Shaders and pipelines compiled on worker threads, a draw that needs one not ready yet skipped (the fast core only:
    // the accurate core's pictures must not depend on host timing). MV_ASYNC_SHADERS=0/1 overrides it.
    bool async_shaders{false};
    // The least anisotropic filtering a linearly filtered texture gets (1 = what each texture asks for).
    u32 anisotropy{1};
    // The internal resolution (PLAN D34): render targets at this many times their size, 1-8 (the fast core only; the
    // accurate core always renders at 1x).
    u32 resolution_scale{1};
    // Pacing of a CPU that runs free (the fast mode with its guest clock): frame steps stop the CPU at every vblank
    // until the front end asks for the next frame (the libretro core); otherwise guest time keeps to the host clock
    // times `speed` (1 = real time), or runs as fast as it can with speed 0.
    bool frame_step{false};
    double speed{0.0};
};

// The console: every device on one bus and one timeline, a CPU backend running the firmware against them, and
// the lock that serialises the two threads that touch devices (the CPU's exits and the io thread's timers).
class Machine {
public:
    explicit Machine(const MachineConfig& config);
    const MachineConfig& config() const { return config_; }
    ~Machine();
    Machine(const Machine&) = delete;
    Machine& operator=(const Machine&) = delete;

    bool init(std::string& error);

    // Run the machine from reset for at most `host_duration` of wall time, or until `until` returns true
    // (checked on every timer tick) or the CPU stops. The caller's thread becomes the io thread.
    enum class StopReason { Duration, Condition, CpuStopped, PowerOff };
    StopReason run(std::chrono::nanoseconds host_duration, std::function<bool(Machine&)> until = {});
    // The same in steps, for a host that drives the machine frame by frame (the libretro core): start the CPU,
    // pump the io thread's work until `until` holds or the wall clock reaches `deadline` (the CPU keeps running
    // between pumps), and finish when done.
    void start();
    StopReason pump(std::chrono::steady_clock::time_point deadline, const std::function<bool(Machine&)>& until);
    void finish();

    Bus& bus() { return bus_; }
    Scheduler& scheduler() { return sched_; }
    Flash& flash() { return flash_; }
    Pic& pic() { return pic_; }
    Pit& pit() { return pit_; }
    Rtc& rtc() { return rtc_; }
    PostPort& post() { return post_; }
    Smc& smc() { return smc_; }
    Eeprom& eeprom() { return eeprom_; }
    SmbusController& smbus() { return smbus_; }
    Nv2aStub& nv2a() { return nv2a_; }
    IdeController& ide() { return ide_; }
    AtaDisk* hdd() { return hdd_; }
    Apu& apu() { return apu_; }
    Aci& aci() { return aci_; }
    Ohci& usb0() { return usb0_; }
    Ohci& usb1() { return usb1_; }
    // The gamepad on a front port (0-3); present whether or not it is plugged in.
    usb::XidGamepad& pad(unsigned front_port) { return pads_[front_port].gamepad; }
    PciBus& pci() { return pci_; }
    Lpc& lpc() { return lpc_; }
    CpuBackend* cpu() { return cpu_.get(); }
    const pgraph::VulkanRenderer* renderer() const { return renderer_.get(); }
    pgraph::VulkanRenderer* renderer() { return renderer_.get(); }
    std::string gpu_name() const;  // the Vulkan device's name when --gpu is on, else empty

    // The machine lock and the wake-ups: `wake_cpu` for the CPU thread's waits (HLT, pacing), `wake` for those and
    // the io thread's (a paused frame, a reset or power-off request, the CPU stopping).
    std::mutex& lock() { return lock_; }
    void wake_cpu() { wake_cv_.notify_all(); }
    // The CPU is back in the guest after a stretch out of it (its clock stood still): the io thread re-plans its wake.
    void wake_io() { io_cv_.notify_all(); }
    void wake() {
        wake_cv_.notify_all();
        io_cv_.notify_all();
    }
    template <class Pred>
    void wait_for_wake(Pred pred) {
        std::unique_lock<std::mutex> lk(lock_, std::adopt_lock);
        wake_cv_.wait(lk, pred);
        lk.release();
    }
    // Bring the timeline up to the CPU's clock (or, in the fast mode's wall-clock control, the host's) and fire what
    // is due. Called with the lock held.
    void advance_time();
    Cycles wall_cycles() const;

    // Pacing a free-running CPU (D16). The CPU thread calls this with the lock held before it enters the guest, at
    // guest time `now`. When the CPU must not run on yet (the frame step's vblank came, or it is ahead of the host
    // clock) it waits, releasing the lock, until it may, `stopping` turns true or the machine needs the io thread,
    // and returns false so the caller loops. The io thread kicks the CPU out of the guest when it must come here.
    bool pace_cpu(std::unique_lock<std::mutex>& lk, Cycles now, const std::atomic<bool>& stopping);
    // Frame steps: let the CPU run until the vblank count reaches `vblanks`. With the lock held; wakes the CPU.
    void set_frame_limit(u64 vblanks);
    // Frame steps: the vblank the front end asked for has passed (the accurate core holds at once, at the boundary
    // after the vblank's event, so a frame ends at the same instruction whichever way the core executes).
    bool frame_hold_due() const { return config_.frame_step && nv2a_.vblanks() >= frame_limit_; }
    // The accurate core stopped at that boundary (with the lock held): until it runs again, input lands there.
    void note_frame_end() { cpu_at_frame_end_ = true; }
    // A front end's pad state, with the lock held. The accurate core applies it where the CPU holds for a frame step
    // (now, if it holds; else when it next comes to a hold), so the guest sees input change at a vblank, never at a
    // host-timed instant; the fast mode applies it at once.
    void set_pad_input(unsigned front_port, const usb::PadState& s);
    bool cpu_paused() const { return cpu_paused_; }
    // Instrument: when the CPU last came to a hold at a frame step's vblank (host clock).
    std::chrono::steady_clock::time_point frame_hold_at() const { return frame_hold_at_; }
    // Guest time at which the PIC last raised INTR (an instrument: interrupt latency).
    Cycles intr_rose_at() const { return intr_rose_at_; }
    // Host time the io thread spent firing device events (an instrument: device work off the CPU thread).
    u64 io_event_ns() const { return io_event_ns_; }

    // Boot milestones recognised from behaviour (times on the master timeline, or nullopt).
    struct Milestones {
        std::optional<Cycles> rom_hidden, smc_challenge_answered, bootloader_magic, kernel_header, power_off;
    };
    const Milestones& milestones() const { return milestones_; }
    void request_power_off(std::string why);
    std::string power_off_reason() const { return power_off_reason_; }
    // A warm reset, as the SMC's reset line gives one: the CPU restarts at its vector, every device on the board
    // returns to power-on and the boot ROM shows again; RAM, the RTC, the SMC and the EEPROM keep their state.
    // Performed on the io thread at the next tick.
    void request_reset(std::string why);
    unsigned boots() const { return boots_; }
    // The kernel image in RAM, once unpacked.
    KernelImage kernel() const { return KernelImage(bus_); }
    bool eeprom_generated() const { return eeprom_generated_; }
    capture::Writer* capture() { return capture_.get(); }

private:
    void check_memory_milestones();
    // Pacing as the io thread sees it: whether the CPU must come back to pace_cpu now, and the guest time at which
    // real-time pacing will want it back (or the end of time).
    bool cpu_must_hold(Cycles now) const;
    Cycles pace_horizon() const;
    void perform_reset(std::unique_lock<std::mutex>& lk);

    MachineConfig config_;
    Bus bus_;
    Scheduler sched_;
    Cycles trace_now_{0};
    PciBus pci_;
    Flash flash_;
    Lpc lpc_;
    PmBlock pm_;
    Pic pic_;
    Pit pit_;
    Rtc rtc_;
    PostPort post_;
    IsaMisc isa_;
    SmbusController smbus_;
    Smc smc_;
    Eeprom eeprom_;
    Adm1032 adm1032_;
    VideoEncoder encoder_;
    Nv2aStub nv2a_;
    Apu apu_;
    Aci aci_;
    Ohci usb0_, usb1_;
    // The pads on the front ports, plugged into USB0 at power-on (front ports 1-4 = root ports 3, 4, 1, 2).
    usb::FrontPanel front_panel_;
    std::array<usb::XboxPad, 4> pads_;
    std::array<std::optional<usb::PadState>, 4> pending_pads_;
    bool cpu_at_frame_end_{false};
    void apply_pending_pads();
    IdeController ide_;
    NvNet nvnet_;
    DiskImage hdd_image_;
    AtaDisk* hdd_{nullptr};
    std::vector<std::unique_ptr<PciDevice>> placeholders_;
    std::unique_ptr<CpuBackend> cpu_;
    std::unique_ptr<vk::Device> gpu_device_;
    std::unique_ptr<pgraph::VulkanRenderer> renderer_;

    std::mutex lock_;
    std::condition_variable wake_cv_, io_cv_;
    std::chrono::steady_clock::time_point start_;
    // Pacing (pace_cpu): the frame step's vblank limit, and real time's reference point (the guest time that
    // matched a host time; moved forward when the host falls behind, so the guest never races to catch up).
    u64 frame_limit_{0};
    bool cpu_paused_{false};
    std::atomic<bool> nv2a_service_{false};  // the GPU's FIFO thread has an interrupt change for the machine
    std::chrono::steady_clock::time_point frame_hold_at_{};
    Cycles intr_rose_at_{0};
    u64 io_event_ns_{0};
    std::chrono::steady_clock::time_point pace_wall_;
    Cycles pace_guest_{0};
    bool power_off_{false};
    std::string power_off_reason_;
    bool reset_pending_{false};
    bool reset_at_done_{false};
    std::string reset_reason_;
    unsigned boots_{0};
    Milestones milestones_;
    bool eeprom_generated_{false};
    std::vector<u32> watch_last_;
    std::unique_ptr<capture::Writer> capture_;
    // In accurate mode the milestones are polled on the timeline (every 0.1 ms of guest time) so their
    // timestamps are deterministic; in fast mode the io thread polls them in host time.
    TimerEvent milestone_poll_;
public:
    u64 apu_frames() const { return apu_.frames(); }
};

}  // namespace mv
