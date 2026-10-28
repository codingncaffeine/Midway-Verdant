#include "usb/ohci.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "core/clock.h"
#include "debug/trace.h"

namespace mv {

namespace {
// Register offsets (OpenHCI 1.0a chapter 7).
enum : u32 {
    kRevision = 0x00, kControl = 0x04, kCommandStatus = 0x08, kInterruptStatus = 0x0C, kInterruptEnable = 0x10, kInterruptDisable = 0x14,
    kHcca = 0x18, kPeriodCurrentEd = 0x1C, kControlHeadEd = 0x20, kControlCurrentEd = 0x24, kBulkHeadEd = 0x28, kBulkCurrentEd = 0x2C,
    kDoneHead = 0x30, kFmInterval = 0x34, kFmRemaining = 0x38, kFmNumber = 0x3C, kPeriodicStart = 0x40, kLsThreshold = 0x44,
    kRhDescriptorA = 0x48, kRhDescriptorB = 0x4C, kRhStatus = 0x50, kRhPortStatus = 0x54,
};
// HcControl.
[[maybe_unused]] constexpr u32 kCtlPle = 1u << 2, kCtlIe = 1u << 3, kCtlCle = 1u << 4, kCtlBle = 1u << 5, kCtlHcfsShift = 6, kCtlHcfs = 3u << 6;
constexpr u32 kCtlIr = 1u << 8, kCtlRwc = 1u << 9;
enum : u32 { kUsbReset = 0, kUsbResume = 1, kUsbOperational = 2, kUsbSuspend = 3 };
// HcCommandStatus.
constexpr u32 kCmdHcr = 1u << 0, kCmdClf = 1u << 1, kCmdBlf = 1u << 2, kCmdOcr = 1u << 3;
// HcInterruptStatus / Enable.
[[maybe_unused]] constexpr u32 kIntSo = 1u << 0, kIntWdh = 1u << 1, kIntSf = 1u << 2, kIntFno = 1u << 5, kIntRhsc = 1u << 6, kIntOc = 1u << 30,
              kIntMie = 1u << 31, kIntAll = 0x4000007Fu;
// HcRhPortStatus (read) and the write meanings of its low bits.
constexpr u32 kPortCcs = 1u << 0, kPortPes = 1u << 1, kPortPss = 1u << 2, kPortPrs = 1u << 4, kPortPps = 1u << 8;
[[maybe_unused]] constexpr u32 kPortCsc = 1u << 16, kPortPesc = 1u << 17, kPortPssc = 1u << 18, kPortPrsc = 1u << 20, kPortChanges = 0x001F0000u;
// The root hub: four ports, no power switching (always powered), no over-current protection, 2 ms power-good.
constexpr u32 kRhA = 0x02000000u | (1u << 12) | (1u << 9) | Ohci::kPorts;
// Condition codes.
constexpr u32 kCcNoError = 0, kCcStall = 4, kCcDeviceNotResponding = 5, kCcDataUnderrun = 9;

// One millisecond of guest time, and the 12 MHz bit clock that HcFmRemaining counts in.
Cycles frame_cycles() { return cycles_from_ns(1'000'000); }
}  // namespace

Ohci::Ohci(std::string_view name, Bus& bus, Scheduler& sched, u8 irq_line, std::function<void(bool)> irq)
    : PciDevice(name, 0x10DE, 0x01C2, 0x0C0310, 0xB1), bus_(bus), sched_(sched), irq_(std::move(irq)) {
    define_bar(0, BarKind::Memory, kMmioSize);
    set_cfg8(0x3C, irq_line);
    set_cfg8(0x3D, 1);
    frame_.name = "ohci.frame";
    frame_.fire = [this](Cycles) { start_frame(); };
    rh_a_ = kRhA;
}

void Ohci::on_bar_changed(int index, u32 old_base, u32 new_base) {
    if (index != 0) return;
    if (old_base) bus_.unmap_mmio(old_base, kMmioSize);
    mmio_at_ = new_base;
    if (new_base) bus_.map_mmio(new_base, kMmioSize, this);
    trace::log(trace::Cat::Usb, "{} registers at {:08x}", name(), new_base);
}

u32 Ohci::rd32(u32 addr) const {
    u32 v = 0;
    bus_.read_ram(addr, &v, 4);
    return v;
}

void Ohci::wr32(u32 addr, u32 v) { bus_.write_ram(addr, &v, 4); }

void Ohci::attach(unsigned port, usb::Device* device) {
    ports_[port - 1] = device;
    port_connect(port);
}

void Ohci::port_connect(unsigned port) {
    u32& st = port_status_[port - 1];
    st = kPortPps;
    if (ports_[port - 1]) st |= kPortCcs | kPortCsc;
}

void Ohci::reset() {
    sched_.cancel(frame_);
    control_ = 0;  // USBRESET
    command_status_ = 0;
    int_status_ = 0;
    int_enable_ = 0;
    hcca_ = period_current_ = control_head_ = control_current_ = bulk_head_ = bulk_current_ = done_head_ = 0;
    fm_interval_ = 0x2EDF;
    fm_number_ = 0;
    periodic_start_ = 0;
    ls_threshold_ = 0x628;
    rh_a_ = kRhA;
    rh_b_ = 0;
    rh_status_ = 0;
    done_counter_ = 7;
    for (unsigned p = 1; p <= kPorts; ++p) {
        if (ports_[p - 1]) ports_[p - 1]->bus_reset();
        port_connect(p);
    }
    update_irq();
}

// HCR: the operational registers to their reset values, USBSUSPEND; InterruptRouting, RemoteWakeupConnected and the
// root hub are kept.
void Ohci::software_reset() {
    sched_.cancel(frame_);
    control_ = (control_ & (kCtlIr | kCtlRwc)) | (kUsbSuspend << kCtlHcfsShift);
    command_status_ = 0;
    int_status_ = 0;
    int_enable_ = 0;
    hcca_ = period_current_ = control_head_ = control_current_ = bulk_head_ = bulk_current_ = done_head_ = 0;
    fm_interval_ = 0x2EDF;
    fm_number_ = 0;
    periodic_start_ = 0;
    ls_threshold_ = 0x628;
    done_counter_ = 7;
    update_irq();
}

void Ohci::set_functional_state(u32 hcfs) {
    const u32 old = (control_ & kCtlHcfs) >> kCtlHcfsShift;
    if (hcfs == old) return;
    trace::log(trace::Cat::Usb, "{} functional state {} -> {}", name(), old, hcfs);
    if (hcfs == kUsbOperational) {
        frame_started_ = sched_.now();
        sched_.schedule(frame_, sched_.now() + frame_cycles());  // the first SOF 1 ms after the transition
    } else {
        sched_.cancel(frame_);
    }
}

u32 Ohci::read_reg(u32 offset) {
    switch (offset) {
        case kRevision: return 0x10;
        case kControl: return control_;
        case kCommandStatus: return command_status_;
        case kInterruptStatus: return int_status_;
        case kInterruptEnable:
        case kInterruptDisable: return int_enable_;
        case kHcca: return hcca_;
        case kPeriodCurrentEd: return period_current_;
        case kControlHeadEd: return control_head_;
        case kControlCurrentEd: return control_current_;
        case kBulkHeadEd: return bulk_head_;
        case kBulkCurrentEd: return bulk_current_;
        case kDoneHead: return done_head_;
        case kFmInterval: return fm_interval_;
        case kFmRemaining: {
            // Bit times left in this frame at 12 MHz, and the toggle copied from FmInterval at the frame start.
            const u32 fi = fm_interval_ & 0x3FFF;
            u32 remaining = 0;
            if (((control_ & kCtlHcfs) >> kCtlHcfsShift) == kUsbOperational) {
                const u64 elapsed_ns = static_cast<u64>(static_cast<double>(sched_.now() - frame_started_) * 1e9 / clk::kCpuHz);
                const u64 bits = elapsed_ns * 12 / 1000;
                remaining = bits >= fi ? 0 : static_cast<u32>(fi - bits);
            }
            return remaining | (frame_toggle_ ? (1u << 31) : 0u);
        }
        case kFmNumber: return fm_number_ & 0xFFFF;
        case kPeriodicStart: return periodic_start_;
        case kLsThreshold: return ls_threshold_;
        case kRhDescriptorA: return rh_a_;
        case kRhDescriptorB: return rh_b_;
        case kRhStatus: return rh_status_;
        default:
            if (offset >= kRhPortStatus && offset < kRhPortStatus + 4 * kPorts) return port_status_[(offset - kRhPortStatus) / 4];
            return 0;
    }
}

void Ohci::write_reg(u32 offset, u32 value) {
    switch (offset) {
        case kControl: {
            const u32 hcfs = (value & kCtlHcfs) >> kCtlHcfsShift;
            set_functional_state(hcfs);
            control_ = value & 0x7FF;
            break;
        }
        case kCommandStatus:
            if (value & kCmdHcr) {
                software_reset();
                trace::log(trace::Cat::Usb, "{} software reset", name());
                return;  // complete at once: HCR reads back clear
            }
            command_status_ |= value & (kCmdClf | kCmdBlf | kCmdOcr);
            break;
        case kInterruptStatus: int_status_ &= ~(value & kIntAll); break;
        case kInterruptEnable: int_enable_ |= value & (kIntAll | kIntMie); break;
        case kInterruptDisable: int_enable_ &= ~(value & (kIntAll | kIntMie)); break;
        case kHcca: hcca_ = value & 0xFFFFFF00u; break;
        case kControlHeadEd: control_head_ = value & ~0xFu; break;
        case kControlCurrentEd: control_current_ = value & ~0xFu; break;
        case kBulkHeadEd: bulk_head_ = value & ~0xFu; break;
        case kBulkCurrentEd: bulk_current_ = value & ~0xFu; break;
        case kFmInterval: fm_interval_ = value & 0xFFFF3FFFu; break;
        case kFmNumber: fm_number_ = value & 0xFFFF; break;
        case kPeriodicStart: periodic_start_ = value & 0x3FFF; break;
        case kLsThreshold: ls_threshold_ = value & 0xFFF; break;
        case kRhDescriptorA: rh_a_ = (rh_a_ & 0x00FFFFFFu) | (value & 0xFF000000u); break;  // only POTPGT is writable
        case kRhDescriptorB: rh_b_ = value; break;
        case kRhStatus:
            // Global power on/off (no power switching here), over-current change cleared, remote wakeup enable.
            if (value & (1u << 17)) rh_status_ &= ~(1u << 17);
            if (value & (1u << 15)) rh_status_ |= 1u << 15;
            if (value & (1u << 31)) rh_status_ &= ~(1u << 15);
            break;
        default:
            if (offset >= kRhPortStatus && offset < kRhPortStatus + 4 * kPorts) port_write((offset - kRhPortStatus) / 4 + 1, value);
            break;
    }
    update_irq();
}

// HcRhPortStatus writes: bit 0 ClearPortEnable, 1 SetPortEnable, 2 SetPortSuspend, 3 ClearSuspendStatus, 4 SetPortReset,
// 8 SetPortPower, 9 ClearPortPower; the change bits are write-1-to-clear. A port reset completes at once.
void Ohci::port_write(unsigned port, u32 value) {
    u32& st = port_status_[port - 1];
    const u32 before = st;
    st &= ~(value & kPortChanges);
    const bool connected = (st & kPortCcs) != 0;
    if ((value & 1u) && (st & kPortPes)) st &= ~kPortPes;
    if ((value & 2u) && connected) st |= kPortPes;
    if ((value & 4u) && connected) st |= kPortPss;
    if ((value & 8u) && (st & kPortPss)) st = (st & ~kPortPss) | kPortPssc;
    if ((value & 16u) && connected) {
        if (ports_[port - 1]) ports_[port - 1]->bus_reset();
        st = (st & ~kPortPrs) | kPortPes | kPortPrsc;
        trace::log(trace::Cat::Usb, "{} port {} reset", name(), port);
    }
    if (st != before && (st & kPortChanges) & ~(before & kPortChanges)) int_status_ |= kIntRhsc;
}

u32 Ohci::mmio_read(PhysAddr addr, unsigned size) {
    const u32 offset = static_cast<u32>(addr - mmio_at_);
    const u32 v = read_reg(offset & ~3u);
    // The root hub's registers and the interrupt status, as the driver sees them (not the frame-number polls).
    if ((offset & ~3u) != 0x38 && (offset & ~3u) != 0x3C && trace::on(trace::Cat::Usb))
        trace::log(trace::Cat::Usb, "{} read +{:02x} -> {:08x}", name(), offset & ~3u, v);
    const u32 shift = (offset & 3u) * 8;
    return size >= 4 ? v : (v >> shift) & ((1u << (size * 8)) - 1);
}

void Ohci::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    const u32 offset = static_cast<u32>(addr - mmio_at_);
    trace::log(trace::Cat::Usb, "{} write +{:02x} <- {:08x}", name(), offset, value);
    if (size < 4) {
        // Narrow writes merge into the register (drivers use 32-bit accesses; this keeps a stray one harmless).
        const u32 shift = (offset & 3u) * 8, mask = ((1u << (size * 8)) - 1) << shift;
        value = (read_reg(offset & ~3u) & ~mask) | ((value << shift) & mask);
    }
    write_reg(offset & ~3u, value);
}

void Ohci::update_irq() {
    const bool level = (int_enable_ & kIntMie) && (int_status_ & int_enable_ & kIntAll);
    if (level != irq_level_) {
        irq_level_ = level;
        irq_(level);
    }
}

usb::Device* Ohci::device_at(u8 address) {
    for (unsigned p = 0; p < kPorts; ++p)
        if (ports_[p] && (port_status_[p] & kPortPes))
            if (usb::Device* d = ports_[p]->find(address)) return d;
    return nullptr;
}

// The guest address of byte `offset` of a TD buffer that may cross one 4 KiB page boundary.
u32 Ohci::buffer_address(u32 cbp, u32 be, u32 offset) const {
    const u32 in_page = (cbp & 0xFFFu) + offset;
    if (in_page < 0x1000u || ((cbp ^ be) & ~0xFFFu) == 0) return cbp + offset;
    return (be & ~0xFFFu) + (in_page - 0x1000u);
}

// One endpoint descriptor's turn: its transfer descriptors run until one is NAKed, the ED halts or (in the periodic
// list) one transaction has been made.
Ohci::Visit Ohci::visit_ed(u32 ed_address, bool periodic) {
    u32 ed[4];
    bus_.read_ram(ed_address, ed, sizeof ed);
    const bool skip = (ed[0] >> 14) & 1u, iso = (ed[0] >> 15) & 1u;
    u32 head = ed[2] & ~0xFu;
    const u32 tail = ed[1] & ~0xFu;
    bool halted = ed[2] & 1u;
    u32 carry = (ed[2] >> 1) & 1u;
    if (skip || halted || head == tail) return Visit::Empty;
    if (iso) return Visit::Busy;  // isochronous transfers are not implemented: nothing on the console uses them

    const u8 function = static_cast<u8>(ed[0] & 0x7F), endpoint = static_cast<u8>((ed[0] >> 7) & 0xF);
    const u32 ed_dir = (ed[0] >> 11) & 3u, mps = std::max(1u, (ed[0] >> 16) & 0x7FFu);
    usb::Device* dev = device_at(function);
    unsigned budget = periodic ? 1 : 16;
    while (head != tail && !halted && budget--) {
        u32 td[4];
        bus_.read_ram(head, td, sizeof td);
        const u32 dir = (ed_dir == 1 || ed_dir == 2) ? ed_dir : ((td[0] >> 19) & 3u);  // 0 SETUP, 1 OUT, 2 IN
        const u32 cbp = td[1], be = td[3];
        const u32 total = cbp == 0 ? 0 : (((cbp ^ be) & ~0xFFFu) ? (0x1000u - (cbp & 0xFFFu)) + (be & 0xFFFu) + 1 : be - cbp + 1);
        u32 toggle = (td[0] & (1u << 25)) ? ((td[0] >> 24) & 1u) : carry;
        u32 cc = kCcNoError, moved = 0, packets = 0;
        bool nak = false;
        if (!dev) {
            cc = kCcDeviceNotResponding;
            ++stats_.no_device;
        } else if (dir == 0 || dir == 1) {
            // SETUP or OUT: the buffer goes out in max-packet pieces (a zero-length packet when empty).
            std::vector<u8> data(total);
            for (u32 i = 0; i < total; ++i) bus_.read_ram(buffer_address(cbp, be, i), &data[i], 1);
            do {
                const u32 n = std::min(mps, total - moved);
                u32 actual = 0;
                const auto st = dev->transact(dir == 0 ? usb::Pid::Setup : usb::Pid::Out, endpoint, data.data() + moved, n, actual);
                if (st == usb::Status::Nak) {
                    nak = true;
                    break;
                }
                if (st == usb::Status::Stall) {
                    cc = kCcStall;
                    break;
                }
                moved += n;
                ++packets;
            } while (moved < total);
        } else {
            // IN: max-packet pieces until the buffer is full or the device sends a short packet.
            std::vector<u8> data(std::max(total, 1u));
            for (;;) {
                const u32 want = std::min(mps, total - moved);
                u32 actual = 0;
                const auto st = dev->transact(usb::Pid::In, endpoint, data.data() + moved, want, actual);
                if (st == usb::Status::Nak) {
                    nak = true;
                    break;
                }
                if (st == usb::Status::Stall) {
                    cc = kCcStall;
                    break;
                }
                moved += actual;
                ++packets;
                if (actual < want || moved >= total) break;
            }
            for (u32 i = 0; i < moved; ++i) bus_.write_ram(buffer_address(cbp, be, i), &data[i], 1);
        }
        toggle ^= packets & 1u;
        if (nak && cc == kCcNoError) {
            ++stats_.naks;
            // Keep the TD on the ED; what moved so far stays moved.
            if (moved) {
                td[1] = buffer_address(cbp, be, moved);
                td[0] = (td[0] & ~(3u << 24)) | (2u | toggle) << 24;
                bus_.write_ram(head, td, sizeof td);
            }
            break;
        }
        // Retire the TD: its condition code, the buffer pointer (0 when everything moved), onto the done queue.
        const bool short_in = dir == 2 && cc == kCcNoError && moved < total;
        if (short_in && !((td[0] >> 18) & 1u)) cc = kCcDataUnderrun;  // a short packet only passes with bufferRounding
        if (cc == kCcStall) ++stats_.stalls;
        const u32 next = td[2] & ~0xFu;
        if (trace::on(trace::Cat::Usb)) {
            std::string bytes;
            for (u32 i = 0; i < std::min<u32>(moved, 18); ++i) {
                u8 b = 0;
                bus_.read_ram(buffer_address(cbp, be, i), &b, 1);
                bytes += std::format(" {:02x}", b);
            }
            trace::log(trace::Cat::Usb, "{} td {:08x}: addr {} ep {} {} {} of {} bytes, cc {}{}{}", name(), head, function, endpoint,
                       dir == 0 ? "SETUP" : dir == 1 ? "OUT" : "IN", moved, total, cc, bytes.empty() ? "" : " :", bytes);
        }
        td[0] = (td[0] & 0x00FFFFFFu & ~(3u << 24)) | ((2u | toggle) << 24) | (cc << 28);
        td[1] = (moved >= total || cbp == 0) ? 0 : buffer_address(cbp, be, moved);
        td[2] = done_head_;
        bus_.write_ram(head, td, sizeof td);
        done_head_ = head;
        const u32 di = (td[0] >> 21) & 7u;
        if (di != 7) done_counter_ = std::min(done_counter_, di);
        // A TD retired with an error interrupts at once, whatever its delay (OHCI 1.0a 4.3.1.3.6): the driver's
        // remaining TDs sit behind a halted ED, and a later TD's interrupt would never come.
        if (cc != kCcNoError) done_counter_ = 0;
        ++stats_.transfers;
        head = next;
        carry = toggle;
        if (cc != kCcNoError) halted = true;
    }
    ed[2] = head | (carry << 1) | (halted ? 1u : 0u);
    bus_.write_ram(ed_address + 8, &ed[2], 4);
    return (head != tail && !halted) ? Visit::Busy : Visit::Empty;
}

// Walks one ED list; true when an ED still holds transfer descriptors.
bool Ohci::walk_list(u32 head, bool periodic) {
    bool busy = false;
    for (unsigned guard = 0; head && guard < 4096; ++guard) {
        if (visit_ed(head, periodic) == Visit::Busy) busy = true;
        head = rd32(head + 12) & ~0xFu;
    }
    return busy;
}

void Ohci::start_frame() {
    ++stats_.frames;
    // The end of the last frame: the done queue goes back to the HCCA when its delay interrupt is due.
    if (done_head_ && done_counter_ == 0 && !(int_status_ & kIntWdh)) {
        const u32 others = (int_status_ & int_enable_ & kIntAll & ~kIntWdh) ? 1u : 0u;
        wr32(hcca_ + 0x84, done_head_ | others);
        trace::log(trace::Cat::Usb, "{} done queue {:08x}{} to the HCCA", name(), done_head_, others ? " (others pending)" : "");
        done_head_ = 0;
        done_counter_ = 7;
        int_status_ |= kIntWdh;
        ++stats_.writebacks;
    } else if (done_counter_ != 7 && done_counter_ > 0) {
        --done_counter_;
    }
    // The new frame: its number, SOF, then the lists.
    const u32 old = fm_number_;
    fm_number_ = (fm_number_ + 1) & 0xFFFF;
    frame_toggle_ = (fm_interval_ >> 31) & 1u;
    frame_started_ = sched_.now();
    if (hcca_) {
        const u32 word = fm_number_;  // HccaFrameNumber, then a zero pad
        wr32(hcca_ + 0x80, word);
    }
    int_status_ |= kIntSf;
    if ((old ^ fm_number_) & 0x8000u) int_status_ |= kIntFno;
    if (hcca_ && (control_ & kCtlPle)) walk_list(rd32(hcca_ + 4 * (fm_number_ & 31)) & ~0xFu, true);
    if ((control_ & kCtlCle) && (command_status_ & kCmdClf)) {
        command_status_ &= ~kCmdClf;
        if (walk_list(control_head_, false)) command_status_ |= kCmdClf;
        control_current_ = 0;
    }
    if ((control_ & kCtlBle) && (command_status_ & kCmdBlf)) {
        command_status_ &= ~kCmdBlf;
        if (walk_list(bulk_head_, false)) command_status_ |= kCmdBlf;
        bulk_current_ = 0;
    }
    update_irq();
    sched_.schedule(frame_, sched_.now() + frame_cycles());
}

}  // namespace mv
