#pragma once
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
#include "apu/apu.h"
#include "pci/pci.h"
#include "smbus/smbus.h"
#include "smbus/smc.h"

namespace mv {

struct MachineConfig {
    std::filesystem::path mcpx_rom;
    std::filesystem::path flash_image;
    // Loaded when the file exists; otherwise generated from the kernel's own key at first use and written there.
    std::optional<std::filesystem::path> eeprom_image;
    // A raw hard-disk image (8 GiB for a retail layout); absent = no drive on the cable.
    std::optional<std::filesystem::path> hdd_image;
    bool hdd_locked{true};
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
};

// The console: every device on one bus and one timeline, a CPU backend running the firmware against them, and
// the lock that serialises the two threads that touch devices (the CPU's exits and the io thread's timers).
class Machine {
public:
    explicit Machine(const MachineConfig& config);
    ~Machine();
    Machine(const Machine&) = delete;
    Machine& operator=(const Machine&) = delete;

    bool init(std::string& error);

    // Run the machine from reset for at most `host_duration` of wall time, or until `until` returns true
    // (checked on every timer tick) or the CPU stops. The caller's thread becomes the io thread.
    enum class StopReason { Duration, Condition, CpuStopped, PowerOff };
    StopReason run(std::chrono::nanoseconds host_duration, std::function<bool(Machine&)> until = {});

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
    PciBus& pci() { return pci_; }
    Lpc& lpc() { return lpc_; }
    CpuBackend* cpu() { return cpu_.get(); }

    // The machine lock and the wake-up used by the CPU thread's HLT wait.
    std::mutex& lock() { return lock_; }
    void wake() { wake_cv_.notify_all(); }
    template <class Pred>
    void wait_for_wake(Pred pred) {
        std::unique_lock<std::mutex> lk(lock_, std::adopt_lock);
        wake_cv_.wait(lk, pred);
        lk.release();
    }
    // Fast mode: bring the timeline up to the host clock and fire what is due. Called with the lock held.
    void advance_time();
    Cycles wall_cycles() const;

    // Boot milestones recognised from behaviour (times on the master timeline, or nullopt).
    struct Milestones {
        std::optional<Cycles> rom_hidden, smc_challenge_answered, bootloader_magic, kernel_header, power_off;
    };
    const Milestones& milestones() const { return milestones_; }
    void request_power_off(std::string why);
    std::string power_off_reason() const { return power_off_reason_; }
    // The kernel image in RAM, once unpacked.
    KernelImage kernel() const { return KernelImage(bus_); }
    bool eeprom_generated() const { return eeprom_generated_; }
    capture::Writer* capture() { return capture_.get(); }

private:
    void check_memory_milestones();

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
    IdeController ide_;
    DiskImage hdd_image_;
    AtaDisk* hdd_{nullptr};
    std::vector<std::unique_ptr<PciDevice>> placeholders_;
    std::unique_ptr<CpuBackend> cpu_;

    std::mutex lock_;
    std::condition_variable wake_cv_;
    std::chrono::steady_clock::time_point start_;
    bool power_off_{false};
    std::string power_off_reason_;
    Milestones milestones_;
    bool eeprom_generated_{false};
    std::vector<u32> watch_last_;
    std::unique_ptr<capture::Writer> capture_;
public:
    u64 apu_frames() const { return apu_.frames(); }
};

}  // namespace mv
