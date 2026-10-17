#include "core/machine.h"

#include <chrono>
#include <cstring>
#include <format>
#include <fstream>

#include "core/clock.h"
#include "debug/trace.h"
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
    PciDevice* ohci0 = placeholder({0, 2, 0}, "ohci0", 0x01C2, 0x0C0310, 0xB1);
    ohci0->define_bar(0, PciDevice::BarKind::Memory, 4096);
    ohci0->set_cfg8(0x3C, 1);
    PciDevice* ohci1 = placeholder({0, 3, 0}, "ohci1", 0x01C2, 0x0C0310, 0xB1);
    ohci1->define_bar(0, PciDevice::BarKind::Memory, 4096);
    ohci1->set_cfg8(0x3C, 9);
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
        if (bits & (Smc::kResetBitPowerOff | Smc::kResetBitPowerCycle)) request_power_off(std::format("SMC power request {:02x}", bits));
        else if (bits & Smc::kResetBitReset) request_power_off("SMC reset request (reboot not modelled yet)");
    });
    flash_.add_on_rom_hidden([this] { milestones_.rom_hidden = sched_.now(); });
}

Machine::~Machine() {
    if (cpu_) {
        cpu_->stop();
        cpu_->join();
    }
    nv2a_.set_capture(nullptr, nullptr);
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
    ide_.attach(1, std::make_unique<AtapiCdrom>());
    std::string why_not;
    cpu_ = make_fast_backend(why_not);
    if (!cpu_) {
        error = why_not;
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
    pic_.set_on_intr([this](bool asserted) {
        if (asserted && cpu_) cpu_->kick();
    });
    return true;
}

Cycles Machine::wall_cycles() const {
    const auto elapsed = std::chrono::steady_clock::now() - start_;
    return cycles_from_ns(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
}

void Machine::advance_time() {
    sched_.advance_to(wall_cycles());
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

void Machine::check_memory_milestones() {
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
        if (sched_.has_pending()) {
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
