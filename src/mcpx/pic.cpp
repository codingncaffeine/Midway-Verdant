#include "mcpx/pic.h"

#include "debug/trace.h"

namespace mv {

Pic::Pic(Bus& bus) {
    bus.map_pio(0x20, 2, this);
    bus.map_pio(0xA0, 2, this);
    bus.map_pio(0x4D0, 2, this);  // edge/level control registers
}

// Priority order with rotation: irq (i + priority_add) & 7 has priority i.
int Pic::highest_pending(const Unit& u) const {
    const u8 ready = static_cast<u8>(u.irr & ~u.imr);
    if (!ready) return -1;
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned irq = (i + u.priority_add) & 7u;
        if (ready & (1u << irq)) return static_cast<int>(irq);
    }
    return -1;
}

int Pic::highest_in_service(const Unit& u) const {
    u8 isr = u.isr;
    if (u.special_mask) isr = static_cast<u8>(isr & ~u.imr);
    if (!isr) return -1;
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned irq = (i + u.priority_add) & 7u;
        if (isr & (1u << irq)) return static_cast<int>(irq);
    }
    return -1;
}

namespace {
// Does a pending request at `irq` beat what is in service? (fully nested mode)
bool beats_in_service(const Pic::Unit& u, int irq, int in_service, bool special_fully_nested_cascade) {
    if (in_service < 0) return true;
    const unsigned p_irq = (static_cast<unsigned>(irq) - u.priority_add) & 7u;
    const unsigned p_isr = (static_cast<unsigned>(in_service) - u.priority_add) & 7u;
    if (special_fully_nested_cascade) return p_irq <= p_isr;
    return p_irq < p_isr;
}
}  // namespace

void Pic::unit_set_line(Unit& u, unsigned line, bool asserted) {
    const u8 bit = static_cast<u8>(1u << line);
    const bool was = (u.lines & bit) != 0;
    if (asserted) {
        u.lines = static_cast<u8>(u.lines | bit);
        if (u.level_triggered || (u.elcr & bit)) {
            u.irr = static_cast<u8>(u.irr | bit);
        } else if (!was) {
            u.irr = static_cast<u8>(u.irr | bit);  // rising edge
        }
    } else {
        u.lines = static_cast<u8>(u.lines & ~bit);
        if (u.level_triggered || (u.elcr & bit)) u.irr = static_cast<u8>(u.irr & ~bit);
    }
}

void Pic::set_irq(unsigned irq, bool asserted) {
    if (irq > 15 || irq == 2) return;
    if (irq < 8)
        unit_set_line(units_[0], irq, asserted);
    else
        unit_set_line(units_[1], irq - 8, asserted);
    update_output();  // the cascade line follows the slave's state there
}

int Pic::pending_vector() {
    Unit& m = units_[0];
    const int irq = highest_pending(m);
    if (irq < 0) return -1;
    if (!beats_in_service(m, irq, highest_in_service(m), false)) return -1;
    if (irq == 2) {
        Unit& s = units_[1];
        const int sirq = highest_pending(s);
        if (sirq < 0 || !beats_in_service(s, sirq, highest_in_service(s), m.special_fully_nested)) return -1;
        return static_cast<int>(s.vector_base + sirq);
    }
    return static_cast<int>(m.vector_base + irq);
}

int Pic::acknowledge() {
    Unit& m = units_[0];
    const int irq = highest_pending(m);
    if (irq < 0 || !beats_in_service(m, irq, highest_in_service(m), false)) {
        // Spurious: the 8259 returns IRQ7's vector.
        return static_cast<int>(m.vector_base + 7);
    }
    int vector;
    if (irq == 2) {
        Unit& s = units_[1];
        const int sirq = highest_pending(s);
        if (sirq < 0) {
            vector = static_cast<int>(s.vector_base + 7);
        } else {
            vector = static_cast<int>(s.vector_base + sirq);
            const u8 sbit = static_cast<u8>(1u << sirq);
            if (!(s.level_triggered || (s.elcr & sbit))) s.irr = static_cast<u8>(s.irr & ~sbit);
            if (s.auto_eoi) {
                if (s.priority_add) s.priority_add = static_cast<u8>((sirq + 1) & 7);
            } else {
                s.isr = static_cast<u8>(s.isr | sbit);
            }
        }
    } else {
        vector = static_cast<int>(m.vector_base + irq);
    }
    const u8 mbit = static_cast<u8>(1u << irq);
    if (irq != 2 && !(m.level_triggered || (m.elcr & mbit))) m.irr = static_cast<u8>(m.irr & ~mbit);
    if (m.auto_eoi) {
        // rotate on auto EOI only when enabled through OCW2; plain AEOI leaves priorities
    } else {
        m.isr = static_cast<u8>(m.isr | mbit);
    }
    update_output();
    trace::log(trace::Cat::Irq, "acknowledge irq {} -> vector {:02x}", irq == 2 ? 8 + highest_pending(units_[1]) : irq, vector);
    if (irq != 2 && on_ack_) on_ack_(static_cast<unsigned>(irq));
    return vector;
}

void Pic::reset() {
    units_ = {};
    update_output();
}

void Pic::update_output() {
    // Master line 2 is the slave's INT output, a wire that follows the slave's state after every change of either
    // unit: an acknowledge lowers it, and an EOI or a mask write on the slave that uncovers a request raises it
    // again. Re-evaluating it only on slave line changes lost the console's IDE completion behind an SMBus EOI.
    Unit& m = units_[0];
    const Unit& s = units_[1];
    const int sirq = highest_pending(s);
    const bool slave_out = sirq >= 0 && beats_in_service(s, sirq, highest_in_service(s), false);
    if (slave_out) {
        m.lines = static_cast<u8>(m.lines | 0x04u);
        m.irr = static_cast<u8>(m.irr | 0x04u);
    } else {
        m.lines = static_cast<u8>(m.lines & ~0x04u);
        m.irr = static_cast<u8>(m.irr & ~0x04u);
    }
    const bool now = pending_vector() >= 0;
    if (now != intr_) {
        intr_ = now;
        if (on_intr_) on_intr_(now);
    }
}

void Pic::write_command(Unit& u, u8 v, bool is_master) {
    if (v & 0x10) {  // ICW1
        u.init_state = 1;
        u.icw4_needed = (v & 0x01) != 0;
        u.level_triggered = (v & 0x08) != 0;
        u.imr = 0;
        u.isr = 0;
        u.irr = 0;
        u.priority_add = 0;
        u.auto_eoi = false;
        u.special_mask = false;
        u.read_isr = false;
        trace::log(trace::Cat::Irq, "{} ICW1 {:02x} ({}-triggered{})", is_master ? "master" : "slave", v,
                   u.level_triggered ? "level" : "edge", u.icw4_needed ? ", ICW4" : "");
        update_output();
        return;
    }
    if (v & 0x08) {  // OCW3
        if (v & 0x40) u.special_mask = (v & 0x20) != 0;
        if (v & 0x02) u.read_isr = (v & 0x01) != 0;
        if (v & 0x04) u.poll = true;
        return;
    }
    // OCW2
    const u8 cmd = static_cast<u8>(v >> 5);
    switch (cmd) {
        case 1: {  // non-specific EOI
            const int irq = highest_in_service(u);
            if (irq >= 0) u.isr = static_cast<u8>(u.isr & ~(1u << irq));
            break;
        }
        case 3: {  // specific EOI
            u.isr = static_cast<u8>(u.isr & ~(1u << (v & 7)));
            break;
        }
        case 5: {  // rotate on non-specific EOI
            const int irq = highest_in_service(u);
            if (irq >= 0) {
                u.isr = static_cast<u8>(u.isr & ~(1u << irq));
                u.priority_add = static_cast<u8>((irq + 1) & 7);
            }
            break;
        }
        case 7: {  // rotate on specific EOI
            u.isr = static_cast<u8>(u.isr & ~(1u << (v & 7)));
            u.priority_add = static_cast<u8>(((v & 7) + 1) & 7);
            break;
        }
        case 6:  // set priority
            u.priority_add = static_cast<u8>(((v & 7) + 1) & 7);
            break;
        case 0:  // rotate in automatic EOI mode (clear)
        case 4:  // rotate in automatic EOI mode (set) — tracked loosely
        default:
            break;
    }
    update_output();
}

void Pic::write_data(Unit& u, u8 v) {
    switch (u.init_state) {
        case 1:  // ICW2
            u.vector_base = static_cast<u8>(v & 0xF8);
            u.init_state = 2;
            break;
        case 2:  // ICW3
            u.init_state = u.icw4_needed ? 3 : 0;
            break;
        case 3:  // ICW4
            u.auto_eoi = (v & 0x02) != 0;
            u.special_fully_nested = (v & 0x10) != 0;
            u.init_state = 0;
            break;
        default:  // OCW1
            u.imr = v;
            break;
    }
    update_output();
}

u8 Pic::read_data(const Unit& u) const { return u.imr; }

u8 Pic::read_command(Unit& u) {
    if (u.poll) {
        u.poll = false;
        const int irq = highest_pending(u);
        if (irq < 0) return 0;
        return static_cast<u8>(0x80 | irq);
    }
    return u.read_isr ? u.isr : u.irr;
}

u32 Pic::pio_read(u16 port, unsigned size) {
    (void)size;
    if (port == 0x4D0) return units_[0].elcr;
    if (port == 0x4D1) return units_[1].elcr;
    Unit& u = units_[port >= 0xA0 ? 1 : 0];
    return (port & 1) ? read_data(u) : read_command(u);
}

void Pic::pio_write(u16 port, unsigned size, u32 value) {
    (void)size;
    if (port == 0x4D0 || port == 0x4D1) {
        Unit& u = units_[port - 0x4D0];
        u.elcr = static_cast<u8>(value & (port == 0x4D0 ? 0xF8 : 0xDE));  // IRQ 0,1,2 and 8,13 are always edge
        trace::log(trace::Cat::Irq, "ELCR {} <- {:02x} (level-triggered lines)", port == 0x4D0 ? "master" : "slave", u.elcr);
        update_output();
        return;
    }
    const bool is_master = port < 0xA0;
    Unit& u = units_[is_master ? 0 : 1];
    if (port & 1)
        write_data(u, static_cast<u8>(value));
    else
        write_command(u, static_cast<u8>(value), is_master);
}

}  // namespace mv
