#include "core/machine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
#include <fstream>

#include "core/clock.h"
#include "debug/guest_stack.h"
#include "debug/trace.h"
#include "nv2a/vulkan/vk_device.h"
#include "nv2a/vulkan/vk_renderer.h"
#include "smbus/eeprom_image.h"

namespace mv {

Machine::Machine(const MachineConfig& config)
    : config_(config),
      bus_(config.ram_bytes),
      pci_(bus_),
      flash_(),
      lpc_(bus_, flash_),
      pm_(sched_, [this](u16 type) { request_power_off(std::format("ACPI sleep type {}", type)); }),
      pic_(bus_),
      pit_(bus_, sched_, [this](bool level) { pic_.set_irq(0, level); }),
      rtc_(bus_, sched_, [this](bool level) { pic_.set_irq(8, level); }),
      post_(bus_, sched_),
      isa_(bus_, sched_, pit_),
      smbus_(bus_, sched_, [this](bool level) { pic_.set_irq(11, level); }),
      smc_(config.smc_revision),
      eeprom_(),
      adm1032_(),
      encoder_(VideoEncoder::Kind::Conexant),
      nv2a_(bus_, sched_, [this](bool level) { pic_.set_irq(3, level); }),
      apu_(bus_, sched_, [this](bool level) { pic_.set_irq(5, level); }),
      aci_(bus_, sched_, [this](bool level) { pic_.set_irq(6, level); }),
      usb0_("ohci0", bus_, sched_, 1, [this](bool level) { pic_.set_irq(1, level); }),
      usb1_("ohci1", bus_, sched_, 9, [this](bool level) { pic_.set_irq(9, level); }),
      ide_(bus_, sched_, [this](bool level) { pic_.set_irq(14, level); }) {
    trace::set_clock_source(&trace_now_);
    smc_.set_av_pack(config.av_pack);
    smc_.set_scratch(config.smc_scratch);
    if (config.rtc_base_time) rtc_.set_base_time(*config.rtc_base_time);

    // The PCI functions of a retail console (bus 0 is the MCPX; the AGP bridge leads to bus 1 and the GPU).
    pci_.add({0, 1, 0}, lpc_);
    pci_.add({0, 1, 1}, smbus_);
    pci_.add({1, 0, 0}, nv2a_);
    pci_.add({0, 5, 0}, apu_);
    pci_.add({0, 6, 0}, aci_);
    pci_.add({0, 9, 0}, ide_);
    auto placeholder = [this](PciAddress at, std::string_view name, u16 device, u32 class_code, u8 rev, u8 header = 0) {
        auto d = std::make_unique<PciPlaceholder>(name, device, class_code, rev, header);
        pci_.add(at, *d);
        placeholders_.push_back(std::move(d));
        return placeholders_.back().get();
    };
    PciDevice* host = placeholder({0, 0, 0}, "host", 0x02A5, 0x060000, 0xA1);
    host->set_cfg32(0x10, 0x40000008);
    placeholder({0, 0, 3}, "dram", 0x02A6, 0x050000, 0xA1);
    pci_.add({0, 2, 0}, usb0_);
    pci_.add({0, 3, 0}, usb1_);
    {
        static constexpr unsigned kRootPortOfFrontPort[4] = {3, 4, 1, 2};
        for (unsigned i = 0; i < std::min(config.pads, 4u); ++i) usb0_.attach(kRootPortOfFrontPort[i], &pads_[i].hub);
    }
    PciDevice* nic = placeholder({0, 4, 0}, "nvnet", 0x01C3, 0x020000, 0xB1);
    nic->define_bar(0, PciDevice::BarKind::Memory, 1024);
    nic->define_bar(1, PciDevice::BarKind::Io, 16);
    nic->set_cfg8(0x3C, 4);
    PciDevice* bridge = placeholder({0, 8, 0}, "pci-bridge", 0x01B8, 0x060400, 0xB1, 1);
    bridge->set_cfg8(0x19, 2);  // secondary bus
    bridge->set_cfg8(0x1A, 2);
    PciDevice* agp = placeholder({0, 0x1E, 0}, "agp-bridge", 0x01B7, 0x060400, 0xA1, 1);
    agp->set_cfg8(0x19, 1);
    agp->set_cfg8(0x1A, 1);

    // SMBus slaves of a 1.0-1.3 board.
    smbus_.attach(0x10, smc_);
    smbus_.attach(0x54, eeprom_);
    smbus_.attach(0x4C, adm1032_);
    smbus_.attach(0x45, encoder_);
    lpc_.attach_pm_block(&pm_, PmBlock::kSize);
    bus_.map_mmio(Flash::kBase, Flash::kWindow, &flash_);

    smc_.set_on_reset_request([this](u8 bits) {
        if (bits & Smc::kResetBitPowerOff)
            request_power_off(std::format("SMC power request {:02x}", bits));
        else if (bits & (Smc::kResetBitReset | Smc::kResetBitPowerCycle))
            request_reset(std::format("SMC {} request", (bits & Smc::kResetBitPowerCycle) ? "power-cycle" : "reset"));
    });
    flash_.add_on_rom_hidden([this] { milestones_.rom_hidden = sched_.now(); });
    if (config_.break_smbus_write) {
        smbus_.set_cycle_hook([this](const SmbusController::Transaction& t) {
            if (t.read || t.address7 != config_.break_smbus_write->first || t.command != config_.break_smbus_write->second) return;
            CpuBackend::Regs r{};
            if (cpu_ && cpu_->read_regs(r))
                trace::log(trace::Cat::Boot, "break: smbus {:02x} cmd {:02x} write {:04x}\n{}", t.address7, t.command, t.data,
                           debug::describe_stack(bus_, r));
        });
    }
}

Machine::~Machine() {
    if (cpu_) {
        cpu_->stop();
        cpu_->join();
    }
    nv2a_.set_capture(nullptr, nullptr);
    nv2a_.set_renderer(nullptr);
    renderer_.reset();
    gpu_device_.reset();
    if (capture_) capture_->close();
    trace::set_clock_source(nullptr);
}

bool Machine::init(std::string& error) {
    if (!flash_.load_image(config_.flash_image, error)) return false;
    if (!flash_.load_mcpx_rom(config_.mcpx_rom, error)) return false;
    if (config_.eeprom_image && std::filesystem::exists(*config_.eeprom_image)) {
        std::ifstream f(*config_.eeprom_image, std::ios::binary);
        if (!f) {
            error = std::format("cannot open EEPROM image {}", config_.eeprom_image->string());
            return false;
        }
        f.read(reinterpret_cast<char*>(eeprom_.image().data()), 256);
        eeprom_.mark_provisioned();
    } else {
        eeprom_.set_provisioner([this](std::array<u8, 256>& image) {
            const KernelImage k(bus_);
            std::string why;
            if (!k.present(&why)) {
                trace::log(trace::Cat::Boot, "EEPROM needed before the kernel is in RAM ({}); left blank", why);
                return false;
            }
            const auto key = k.eeprom_key();
            if (!key) {
                trace::log(trace::Cat::Boot, "kernel export table has no EEPROM key; EEPROM left blank");
                return false;
            }
            eeprom::Settings s;
            for (std::size_t i = 0; i < 16; ++i) s.hdd_key[i] = static_cast<u8>(0x4D + 7 * i);  // any key: the disk image is unlocked with it
            image = eeprom::generate(*key, s);
            eeprom_generated_ = true;
            const auto v = k.version();
            trace::log(trace::Cat::Boot, "EEPROM generated with the kernel's own key (kernel {}.{}.{}.{})", v ? v->major : 0,
                       v ? v->minor : 0, v ? v->build : 0, v ? v->qfe : 0);
            if (config_.eeprom_image) {
                std::ofstream f(*config_.eeprom_image, std::ios::binary);
                if (f) {
                    f.write(reinterpret_cast<const char*>(image.data()), 256);
                    trace::log(trace::Cat::Boot, "EEPROM saved to {}", config_.eeprom_image->string());
                }
            }
            return true;
        });
    }
    // The cable: the hard disk as master when an image is given, the DVD drive as slave always.
    if (config_.hdd_image) {
        if (!hdd_image_.open(*config_.hdd_image, false, error)) return false;
        auto disk = std::make_unique<AtaDisk>(hdd_image_);
        hdd_ = disk.get();
        disk->set_locked(config_.hdd_locked);
        disk->set_password_oracle([this]() -> std::optional<std::array<u8, 32>> {
            // The password the kernel will present: HMAC of the drive identity with the HDD key the EEPROM holds,
            // decrypted with the kernel's own EEPROM key.
            const KernelImage k(bus_);
            const auto key = k.eeprom_key();
            if (!key) return std::nullopt;
            std::string why;
            const auto hdd_key = eeprom::verify(*key, eeprom_.image(), &why);
            if (!hdd_key) {
                trace::log(trace::Cat::Boot, "HDD password oracle: EEPROM does not verify ({})", why);
                return std::nullopt;
            }
            return AtaDisk::password_for(*hdd_key, hdd_->model(), hdd_->serial());
        });
        ide_.attach(0, std::move(disk));
        trace::log(trace::Cat::Boot, "HDD image {} ({} sectors){}", config_.hdd_image->string(), hdd_image_.sectors(),
                   config_.hdd_locked ? ", locked" : "");
    }
    auto dvd = std::make_unique<AtapiCdrom>();
    if (config_.dvd_image) {
        auto disc = std::make_unique<DvdImage>();
        if (!disc->open(*config_.dvd_image, error)) return false;
        trace::log(trace::Cat::Boot, "DVD image {} ({}, {} sectors)", config_.dvd_image->string(), disc->kind_name(), disc->sectors());
        dvd->insert(std::move(disc));
        smc_.set_tray(Smc::Tray::ClosedMedia);
    }
    ide_.attach(1, std::move(dvd));
    if (config_.cpu == "p6") {
        cpu_ = make_p6_backend(config_.oracle);
    } else if (config_.cpu == "kvm") {
        std::string why_not;
        cpu_ = make_fast_backend(why_not);
        if (!cpu_) {
            error = why_not;
            return false;
        }
    } else {
        error = "unknown CPU backend " + config_.cpu + " (kvm or p6)";
        return false;
    }
    if (config_.capture_path) {
        capture_ = std::make_unique<capture::Writer>();
        if (!capture_->open(*config_.capture_path, static_cast<u32>(bus_.ram_size()), error)) return false;
        cpu_->set_dirty_tracking(true);
        bus_.set_dirty_tracking(true);
    }
    if (!cpu_->init(*this, error)) return false;
    if (capture_) nv2a_.set_capture(capture_.get(), cpu_.get());
    if (config_.gpu) {
        gpu_device_ = std::make_unique<vk::Device>();
        if (!gpu_device_->init(error)) return false;
        renderer_ = std::make_unique<pgraph::VulkanRenderer>(*gpu_device_, nv2a_, bus_.ram(), static_cast<u32>(bus_.ram_size()));
        if (!renderer_->init(error)) return false;
        nv2a_.set_renderer(renderer_.get());
        trace::log(trace::Cat::Boot, "live GPU on {}", gpu_device_->name());
    }
    pic_.set_on_intr([this](bool asserted) {
        if (asserted && cpu_) cpu_->kick();
    });
    if (cpu_->owns_timeline()) {
        milestone_poll_.name = "milestones";
        milestone_poll_.fire = [this](Cycles now) {
            check_memory_milestones();
            sched_.schedule(milestone_poll_, now + cycles_from_ns(100'000));
        };
        sched_.schedule(milestone_poll_, cycles_from_ns(100'000));
    }
    return true;
}

Cycles Machine::wall_cycles() const {
    const auto elapsed = std::chrono::steady_clock::now() - start_;
    return cycles_from_ns(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
}

void Machine::advance_time() {
    if (cpu_ && cpu_->owns_timeline()) {
        // The interpreter's clock is the timeline and it advances the scheduler itself at every deadline;
        // here only what is already due fires.
        sched_.advance_to(cpu_->timeline_cycles());
    } else {
        sched_.advance_to(wall_cycles());
    }
    trace_now_ = sched_.now();
}

void Machine::request_power_off(std::string why) {
    if (power_off_) return;
    power_off_ = true;
    power_off_reason_ = std::move(why);
    milestones_.power_off = sched_.now();
    trace::log(trace::Cat::Boot, "power off: {}", power_off_reason_);
    wake();
}

void Machine::request_reset(std::string why) {
    if (reset_pending_ || power_off_) return;
    reset_pending_ = true;
    reset_reason_ = std::move(why);
    trace::log(trace::Cat::Boot, "reset requested: {}", reset_reason_);
    wake();
}

void Machine::perform_reset(std::unique_lock<std::mutex>& lk) {
    reset_pending_ = false;
    // The CPU thread must be out of the hypervisor before its state and the memory slots change.
    cpu_->stop();
    lk.unlock();
    cpu_->join();
    lk.lock();
    ++boots_;
    trace::log(trace::Cat::Boot, "reset: {} (boot {})", reset_reason_, boots_ + 1);
    // Everything the reset line reaches: the PCI functions (BARs unmapped, the bridge's ROM-hide bit cleared), the
    // flash (the boot ROM shows again), the legacy devices, the engines, the drives. RAM keeps its contents (the
    // kernel's launch data page lives there), and the RTC, the SMC and the EEPROM are powered through a reset.
    pci_.reset_all();
    flash_.reset();
    pic_.reset();
    pit_.reset();
    smbus_.reset();
    nv2a_.reset();
    apu_.reset();
    aci_.reset();
    usb0_.reset();
    usb1_.reset();
    ide_.reset();
    milestones_ = {};
    cpu_->reset();
    cpu_->start();
}

void Machine::check_memory_milestones() {
    if (!config_.watch.empty()) {
        if (watch_last_.size() != config_.watch.size()) watch_last_.assign(config_.watch.size(), 0);
        for (std::size_t i = 0; i < config_.watch.size(); ++i) {
            u32 v = 0;
            if (bus_.read_ram(config_.watch[i], &v, 4) && v != watch_last_[i]) {
                trace::log(trace::Cat::Boot, "watch {:08x}: {:08x} -> {:08x}", config_.watch[i], watch_last_[i], v);
                watch_last_[i] = v;
            }
        }
    }
    if (!milestones_.bootloader_magic) {
        u32 magic = 0;
        if (bus_.read_ram(0x95FE4, &magic, 4) && magic == 0x7854794A) {
            milestones_.bootloader_magic = sched_.now();
            trace::log(trace::Cat::Boot, "second bootloader decrypted at 0x90000 (magic at 0x95fe4)");
        }
    }
    if (!milestones_.kernel_header) {
        u8 mz[2]{};
        if (bus_.read_ram(0x10000, mz, 2) && mz[0] == 'M' && mz[1] == 'Z') {
            u32 pe_off = 0;
            bus_.read_ram(0x1003C, &pe_off, 4);
            u8 pe[2]{};
            if (pe_off < 0x1000 && bus_.read_ram(0x10000 + pe_off, pe, 2) && pe[0] == 'P' && pe[1] == 'E') {
                milestones_.kernel_header = sched_.now();
                trace::log(trace::Cat::Boot, "kernel image present at 0x80010000 (MZ/PE)");
            }
        }
    }
    if (!milestones_.smc_challenge_answered && smc_.challenge_answered()) milestones_.smc_challenge_answered = sched_.now();
}

Machine::StopReason Machine::run(std::chrono::nanoseconds host_duration, std::function<bool(Machine&)> until) {
    start_ = std::chrono::steady_clock::now();
    const auto deadline = start_ + host_duration;
    cpu_->start();
    StopReason reason = StopReason::Duration;
    for (;;) {
        std::unique_lock lk(lock_);
        advance_time();
        check_memory_milestones();
        if (config_.reset_at_seconds && !reset_at_done_ &&
            std::chrono::steady_clock::now() - start_ >= std::chrono::duration<double>(*config_.reset_at_seconds)) {
            reset_at_done_ = true;
            request_reset("front-panel reset");
        }
        if (reset_pending_) {
            perform_reset(lk);
            continue;
        }
        if (power_off_) {
            reason = StopReason::PowerOff;
            break;
        }
        if (cpu_->stop_reason() != CpuBackend::Stop::None) {
            reason = StopReason::CpuStopped;
            break;
        }
        if (until && until(*this)) {
            reason = StopReason::Condition;
            break;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        // Sleep until the next timer or a short tick (milestones are polled), whichever is sooner.
        auto wake_at = now + std::chrono::milliseconds(2);
        if (sched_.has_pending() && !cpu_->owns_timeline()) {
            const u64 next_ns = ns_from_cycles(sched_.next_deadline());
            const auto next_tp = start_ + std::chrono::nanoseconds(static_cast<i64>(next_ns));
            if (next_tp < wake_at) wake_at = next_tp;
        }
        if (deadline < wake_at) wake_at = deadline;
        lk.unlock();
        std::this_thread::sleep_until(wake_at);
        // A timer may have become due: the CPU thread must see interrupts promptly.
        {
            std::lock_guard g(lock_);
            advance_time();
            if (pic_.intr_asserted()) cpu_->kick();
        }
    }
    cpu_->stop();
    cpu_->join();
    return reason;
}

}  // namespace mv

namespace mv {
std::string Machine::gpu_name() const { return gpu_device_ ? gpu_device_->name() : std::string(); }
}  // namespace mv
