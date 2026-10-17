#pragma once
#include <array>
#include <functional>

#include "core/bus.h"
#include "core/types.h"

namespace mv {

// Two cascaded 8259A interrupt controllers (slave on master IRQ2), as the console's kernel programs them:
// edge-triggered, fully nested, ICW4 with or without automatic EOI. The output is a level to the CPU core:
// `pending_vector` tells the backend which vector to deliver when the core takes interrupts.
class Pic : public PioHandler {
public:
    explicit Pic(Bus& bus);
    std::string_view name() const override { return "pic"; }

    // Interrupt request lines 0..15 (2 is the cascade and is ignored).
    void set_irq(unsigned irq, bool asserted);

    // The vector of the highest-priority pending unmasked request, or -1. Does not acknowledge.
    int pending_vector();
    // Acknowledge the pending request (the INTA cycle): sets ISR, clears IRR in edge mode, returns the vector.
    int acknowledge();
    // The reset line: both units back to power-on (everything masked, nothing pending, edge mode).
    void reset();

    // Called whenever the INTR output changes level.
    void set_on_intr(std::function<void(bool asserted)> cb) { on_intr_ = std::move(cb); }
    bool intr_asserted() const { return intr_; }

    u32 pio_read(u16 port, unsigned size) override;
    void pio_write(u16 port, unsigned size, u32 value) override;

    struct Unit {
        u8 irr{0}, isr{0}, imr{0xFF};
        u8 vector_base{0};
        u8 lines{0};  // current line levels
        u8 last_lines{0};
        u8 priority_add{0};  // rotating priority
        u8 init_state{0};    // 0 = ready, 1..3 = expecting ICW2..ICW4
        bool icw4_needed{false}, auto_eoi{false}, level_triggered{false};
        bool read_isr{false};  // OCW3 RIS
        bool poll{false};
        bool special_mask{false};
        bool special_fully_nested{false};
        u8 elcr{0};
    };
    const Unit& master() const { return units_[0]; }
    const Unit& slave() const { return units_[1]; }

private:
    int highest_pending(const Unit& u) const;  // irq index 0..7 or -1
    int highest_in_service(const Unit& u) const;
    void update_output();
    void write_command(Unit& u, u8 v, bool is_master);
    void write_data(Unit& u, u8 v);
    u8 read_data(const Unit& u) const;
    u8 read_command(Unit& u);
    void unit_set_line(Unit& u, unsigned line, bool asserted);

    std::array<Unit, 2> units_{};
    bool intr_{false};
    std::function<void(bool)> on_intr_;
};

}  // namespace mv
