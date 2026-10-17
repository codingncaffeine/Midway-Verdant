#include "mcpx/rtc.h"

#include <chrono>

#include "core/clock.h"
#include "debug/trace.h"

namespace mv {

namespace {
constexpr u8 kRegSeconds = 0, kRegSecondsAlarm = 1, kRegMinutes = 2, kRegMinutesAlarm = 3, kRegHours = 4, kRegHoursAlarm = 5,
             kRegWeekday = 6, kRegDay = 7, kRegMonth = 8, kRegYear = 9, kRegA = 10, kRegB = 11, kRegC = 12, kRegD = 13;
constexpr u8 kAUip = 0x80;
constexpr u8 kBSet = 0x80, kBPie = 0x40, kBAie = 0x20, kBUie = 0x10, kBBinary = 0x04, kB24h = 0x02;
constexpr u8 kCIrqf = 0x80, kCPf = 0x40, kCAf = 0x20, kCUf = 0x10;
}  // namespace

Rtc::Rtc(Bus& bus, Scheduler& sched, std::function<void(bool)> irq8) : sched_(sched), irq8_(std::move(irq8)) {
    bus.map_pio(0x70, 4, this);
    cmos_[kRegA] = 0x26;  // 32.768 kHz, 1024 Hz periodic rate
    cmos_[kRegB] = kB24h | kBBinary;
    cmos_[kRegD] = 0x80;  // battery good
    const auto now = std::chrono::system_clock::now();
    base_seconds_ = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    periodic_event_.name = "rtc.periodic";
    periodic_event_.fire = [this](Cycles) {
        cmos_[kRegC] |= kCPf;
        update_irq();
        schedule_periodic();
    };
    update_event_.name = "rtc.update";
    update_event_.fire = [this](Cycles) {
        cmos_[kRegC] |= kCUf;
        if (cmos_[kRegB] & kBAie) {
            refresh_time_registers();
            auto match = [this](u8 reg, u8 alarm) { return (cmos_[alarm] & 0xC0) == 0xC0 || cmos_[alarm] == cmos_[reg]; };
            if (match(kRegSeconds, kRegSecondsAlarm) && match(kRegMinutes, kRegMinutesAlarm) && match(kRegHours, kRegHoursAlarm))
                cmos_[kRegC] |= kCAf;
        }
        update_irq();
        schedule_update();
    };
    schedule_update();
}

void Rtc::set_base_time(i64 unix_seconds) { base_seconds_ = unix_seconds; }

i64 Rtc::current_time() const { return base_seconds_ + static_cast<i64>(ns_from_cycles(sched_.now()) / 1'000'000'000ull); }

// Howard Hinnant's civil-from-days, so no host time zone is involved.
Rtc::Civil Rtc::civil_from_unix(i64 t) {
    i64 days = t / 86400;
    i64 rem = t % 86400;
    if (rem < 0) {
        rem += 86400;
        --days;
    }
    const i64 z = days + 719468;
    const i64 era = (z >= 0 ? z : z - 146096) / 146097;
    const i64 doe = z - era * 146097;
    const i64 yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const i64 y = yoe + era * 400;
    const i64 doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const i64 mp = (5 * doy + 2) / 153;
    const i64 d = doy - (153 * mp + 2) / 5 + 1;
    const i64 m = mp < 10 ? mp + 3 : mp - 9;
    Civil c;
    c.year = static_cast<int>(m <= 2 ? y + 1 : y);
    c.month = static_cast<int>(m);
    c.day = static_cast<int>(d);
    c.weekday = static_cast<int>(((days % 7) + 11) % 7) + 1;  // 1 = Sunday
    c.hour = static_cast<int>(rem / 3600);
    c.minute = static_cast<int>((rem % 3600) / 60);
    c.second = static_cast<int>(rem % 60);
    return c;
}

i64 Rtc::unix_from_civil(const Civil& c) {
    const i64 y = c.month <= 2 ? c.year - 1 : c.year;
    const i64 era = (y >= 0 ? y : y - 399) / 400;
    const i64 yoe = y - era * 400;
    const i64 m = c.month;
    const i64 doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + c.day - 1;
    const i64 doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const i64 days = era * 146097 + doe - 719468;
    return days * 86400 + c.hour * 3600 + c.minute * 60 + c.second;
}

u8 Rtc::to_reg(int v, bool hour) const {
    const bool binary = (cmos_[kRegB] & kBBinary) != 0;
    bool pm = false;
    if (hour && !(cmos_[kRegB] & kB24h)) {
        pm = v >= 12;
        v = v % 12;
        if (v == 0) v = 12;
    }
    u8 r = binary ? static_cast<u8>(v) : static_cast<u8>(((v / 10) << 4) | (v % 10));
    if (pm) r |= 0x80;
    return r;
}

int Rtc::from_reg(u8 v, bool hour) const {
    const bool binary = (cmos_[kRegB] & kBBinary) != 0;
    bool pm = false;
    if (hour && !(cmos_[kRegB] & kB24h)) {
        pm = (v & 0x80) != 0;
        v &= 0x7F;
    }
    int n = binary ? v : ((v >> 4) * 10 + (v & 0x0F));
    if (hour && !(cmos_[kRegB] & kB24h)) {
        if (n == 12) n = 0;
        if (pm) n += 12;
    }
    return n;
}

void Rtc::refresh_time_registers() {
    const Civil c = civil_from_unix(current_time());
    cmos_[kRegSeconds] = to_reg(c.second);
    cmos_[kRegMinutes] = to_reg(c.minute);
    cmos_[kRegHours] = to_reg(c.hour, true);
    cmos_[kRegWeekday] = to_reg(c.weekday);
    cmos_[kRegDay] = to_reg(c.day);
    cmos_[kRegMonth] = to_reg(c.month);
    cmos_[kRegYear] = to_reg(c.year % 100);
    cmos_[0x32] = to_reg(c.year / 100);  // century byte, where PC firmware keeps it
}

void Rtc::apply_time_registers() {
    Civil c;
    c.second = from_reg(cmos_[kRegSeconds]);
    c.minute = from_reg(cmos_[kRegMinutes]);
    c.hour = from_reg(cmos_[kRegHours], true);
    c.day = from_reg(cmos_[kRegDay]);
    c.month = from_reg(cmos_[kRegMonth]);
    const int century = from_reg(cmos_[0x32]);
    c.year = (century ? century : 20) * 100 + from_reg(cmos_[kRegYear]);
    c.weekday = 1;
    const i64 wanted = unix_from_civil(c);
    base_seconds_ = wanted - static_cast<i64>(ns_from_cycles(sched_.now()) / 1'000'000'000ull);
}

void Rtc::update_irq() {
    const u8 b = cmos_[kRegB];
    const bool level = ((cmos_[kRegC] & kCPf) && (b & kBPie)) || ((cmos_[kRegC] & kCAf) && (b & kBAie)) ||
                       ((cmos_[kRegC] & kCUf) && (b & kBUie));
    if (level)
        cmos_[kRegC] |= kCIrqf;
    else
        cmos_[kRegC] &= static_cast<u8>(~kCIrqf);
    if (level != irq_level_) {
        irq_level_ = level;
        if (irq8_) irq8_(level);
    }
}

void Rtc::schedule_periodic() {
    const unsigned rate = cmos_[kRegA] & 0x0F;
    if (rate == 0 || !(cmos_[kRegB] & kBPie)) {
        sched_.cancel(periodic_event_);
        return;
    }
    const u64 hz = 32768u >> (rate - 1);
    // Tick n fires at the first cycle at or after n/hz seconds; a deadline never computes to the current cycle.
    const u64 ticks_so_far = ticks_from_cycles(sched_.now(), hz);
    Cycles when = cycles_from_ticks_ceil(ticks_so_far + 1, hz);
    if (when <= sched_.now()) when = cycles_from_ticks_ceil(ticks_so_far + 2, hz);
    sched_.schedule(periodic_event_, when);
}

void Rtc::schedule_update() {
    // The next second boundary of the clock.
    const u64 ns = ns_from_cycles(sched_.now());
    const u64 next_second_ns = (ns / 1'000'000'000ull + 1) * 1'000'000'000ull;
    sched_.schedule(update_event_, cycles_from_ns_ceil(next_second_ns));
}

u32 Rtc::pio_read(u16 port, unsigned size) {
    (void)size;
    if ((port & 1) == 0) return index_ | 0x80;
    const u8 reg = index_ & 0x7F;
    if (reg <= kRegYear && !(cmos_[kRegB] & kBSet)) refresh_time_registers();
    if (reg == kRegA) {
        // UIP is set for the last 244 us of each second.
        const u64 ns_in_second = ns_from_cycles(sched_.now()) % 1'000'000'000ull;
        const bool uip = ns_in_second >= 1'000'000'000ull - 244'000ull;
        return static_cast<u32>((cmos_[kRegA] & 0x7F) | (uip ? kAUip : 0));
    }
    if (reg == kRegC) {
        const u8 v = cmos_[kRegC];
        cmos_[kRegC] = 0;
        update_irq();
        return v;
    }
    return cmos_[reg];
}

void Rtc::pio_write(u16 port, unsigned size, u32 value) {
    (void)size;
    const u8 v = static_cast<u8>(value);
    if ((port & 1) == 0) {
        index_ = static_cast<u8>(v & 0x7F);
        return;
    }
    const u8 reg = index_ & 0x7F;
    switch (reg) {
        case kRegA:
            cmos_[kRegA] = static_cast<u8>(v & 0x7F);
            schedule_periodic();
            trace::log(trace::Cat::Timer, "rtc A <- {:02x} (periodic rate {})", v, v & 0x0F);
            break;
        case kRegB: {
            const u8 old = cmos_[kRegB];
            cmos_[kRegB] = v;
            if ((old & kBSet) && !(v & kBSet)) apply_time_registers();
            schedule_periodic();
            update_irq();
            trace::log(trace::Cat::Timer, "rtc B <- {:02x}", v);
            break;
        }
        case kRegC:
        case kRegD:
            break;  // read-only
        default:
            cmos_[reg] = v;
            if (reg <= kRegYear && !(cmos_[kRegB] & kBSet)) apply_time_registers();
            break;
    }
}

}  // namespace mv
