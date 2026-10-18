// The system half of the core: descriptors and segment loads, far transfers, interrupt and exception delivery,
// control registers, MSRs, CPUID and the privileged instructions. Written from SDM Vol. 3.
#include <format>

#include "cpu/p6/cpu.h"
#include "debug/trace.h"

namespace mv::p6 {

// ---------------------------------------------------------------------------------------------------------------
// Descriptors

Cpu::Descriptor Cpu::read_descriptor(u16 selector, u8 fault_vector) {
    const u32 index = selector & ~7u;
    const bool ldt = (selector & 4) != 0;
    u32 base, limit;
    if (ldt) {
        if (st_.ldtr.unusable() || st_.ldtr.selector == 0) raise(fault_vector, selector & 0xFFFC);
        base = st_.ldtr.base;
        limit = st_.ldtr.limit;
    } else {
        base = st_.gdtr.base;
        limit = st_.gdtr.limit;
    }
    if (index + 7 > limit) raise(fault_vector, selector & 0xFFFC);
    const u32 addr = base + index;
    Descriptor d;
    d.lo = mem_.read(addr, 4);
    d.hi = mem_.read(addr + 4, 4);
    return d;
}

void Cpu::set_accessed(u16 selector) {
    const u32 index = selector & ~7u;
    const u32 base = (selector & 4) ? st_.ldtr.base : st_.gdtr.base;
    const u32 addr = base + index + 4;
    const u32 hi = mem_.read(addr, 4);
    if (!(hi & 0x100)) mem_.write(addr, 4, hi | 0x100);
}

void Cpu::load_cached(u8 seg, u16 selector, const Descriptor& d) {
    Segment& s = st_.seg[seg];
    s.selector = selector;
    s.base = d.base();
    s.limit = d.limit();
    s.ar = d.ar() | 0x1;  // accessed, as the hardware leaves it
}

void Cpu::set_real_mode_segment(u8 seg, u16 selector) {
    Segment& s = st_.seg[seg];
    s.selector = selector;
    s.base = static_cast<u32>(selector) << 4;
    // Limit and rights stay as they were: a protected-mode load survives the switch back (big real mode).
    s.ar &= ~Segment::kUnusable;
}

// MOV sreg, POP sreg, LxS (not CS).
void Cpu::load_segment(u8 seg, u16 selector) {
    if (!st_.protected_mode() || st_.v8086()) {
        set_real_mode_segment(seg, selector);
        if (st_.v8086()) {
            st_.seg[seg].limit = 0xFFFF;
            st_.seg[seg].ar = 0xF3;
        }
        return;
    }
    if ((selector & 0xFFFC) == 0) {
        if (seg == kSs) raise_gp(0);
        Segment& s = st_.seg[seg];
        s.selector = selector;
        s.base = 0;
        s.limit = 0;
        s.ar = Segment::kUnusable;
        return;
    }
    const Descriptor d = read_descriptor(selector);
    const unsigned rpl = selector & 3;
    const unsigned cpl = st_.cpl();
    if (seg == kSs) {
        if (rpl != cpl || d.system() || !d.writable() || d.dpl() != cpl) raise_gp(selector & 0xFFFC);
        if (!d.present()) raise(kSS, selector & 0xFFFC);
    } else {
        if (d.system() || !d.readable()) raise_gp(selector & 0xFFFC);
        // Data and non-conforming code: DPL >= max(CPL, RPL).
        if (!d.conforming() && (d.dpl() < cpl || d.dpl() < rpl)) raise_gp(selector & 0xFFFC);
        if (!d.present()) raise(kNP, selector & 0xFFFC);
    }
    set_accessed(selector);
    load_cached(seg, selector, d);
}

void Cpu::load_cs(u16 selector, const Descriptor& d) {
    set_accessed(selector);
    load_cached(kCs, selector, d);
}

// ---------------------------------------------------------------------------------------------------------------
// Far transfers

void Cpu::jump_far(u16 selector, u32 offset, const Insn& in) {
    const unsigned size = in.size;
    if (!st_.protected_mode() || st_.v8086()) {
        if (offset > st_.seg[kCs].limit && size == 2) raise_gp(0);
        set_real_mode_segment(kCs, selector);
        st_.eip = size == 2 ? (offset & 0xFFFFu) : offset;
        return;
    }
    if ((selector & 0xFFFC) == 0) raise_gp(0);
    const Descriptor d = read_descriptor(selector);
    if (d.system()) {
        // Call gates, task gates and TSS targets.
        throw Unimplemented{"far jump through a system descriptor (call gate / task switch)"};
    }
    if (!d.code()) raise_gp(selector & 0xFFFC);
    const unsigned rpl = selector & 3, cpl = st_.cpl();
    if (d.conforming()) {
        if (d.dpl() > cpl) raise_gp(selector & 0xFFFC);
    } else {
        if (rpl > cpl || d.dpl() != cpl) raise_gp(selector & 0xFFFC);
    }
    if (!d.present()) raise(kNP, selector & 0xFFFC);
    const u32 target = size == 2 ? (offset & 0xFFFFu) : offset;
    if (target > d.limit()) raise_gp(0);
    load_cs(static_cast<u16>((selector & 0xFFFC) | cpl), d);
    st_.eip = target;
}

void Cpu::call_far(u16 selector, u32 offset, const Insn& in) {
    const unsigned size = in.size;
    if (!st_.protected_mode() || st_.v8086()) {
        const u32 sp = stack_pointer();
        push(size, st_.seg[kCs].selector);
        try {
            push(size, st_.eip);
        } catch (const Fault&) {
            set_stack_pointer(sp);
            throw;
        }
        set_real_mode_segment(kCs, selector);
        st_.eip = size == 2 ? (offset & 0xFFFFu) : offset;
        return;
    }
    if ((selector & 0xFFFC) == 0) raise_gp(0);
    const Descriptor d = read_descriptor(selector);
    if (d.system()) throw Unimplemented{"far call through a system descriptor (call gate / task switch)"};
    if (!d.code()) raise_gp(selector & 0xFFFC);
    const unsigned rpl = selector & 3, cpl = st_.cpl();
    if (d.conforming()) {
        if (d.dpl() > cpl) raise_gp(selector & 0xFFFC);
    } else {
        if (rpl > cpl || d.dpl() != cpl) raise_gp(selector & 0xFFFC);
    }
    if (!d.present()) raise(kNP, selector & 0xFFFC);
    const u32 target = size == 2 ? (offset & 0xFFFFu) : offset;
    if (target > d.limit()) raise_gp(0);
    const u32 sp = stack_pointer();
    push(size, st_.seg[kCs].selector);
    try {
        push(size, st_.eip);
    } catch (const Fault&) {
        set_stack_pointer(sp);
        throw;
    }
    load_cs(static_cast<u16>((selector & 0xFFFC) | cpl), d);
    st_.eip = target;
}

void Cpu::ret_far(unsigned size, u32 pop_bytes) {
    const u32 sp = stack_pointer();
    const u32 mask = size == 2 ? 0xFFFFu : 0xFFFFFFFFu;
    const u32 off = mem_.read(linear(kSs, sp, size, Access::Read), size) & mask;
    const u16 sel = static_cast<u16>(mem_.read(linear(kSs, (sp + size) & (st_.stack32() ? 0xFFFFFFFFu : 0xFFFFu), size, Access::Read), size));
    if (!st_.protected_mode() || st_.v8086()) {
        set_real_mode_segment(kCs, sel);
        st_.eip = off;
        set_stack_pointer(sp + 2 * size + pop_bytes);
        return;
    }
    if ((sel & 0xFFFC) == 0) raise_gp(0);
    const Descriptor d = read_descriptor(sel);
    const unsigned rpl = sel & 3, cpl = st_.cpl();
    if (rpl < cpl) raise_gp(sel & 0xFFFC);
    if (d.system() || !d.code()) raise_gp(sel & 0xFFFC);
    if (d.conforming() ? d.dpl() > rpl : d.dpl() != rpl) raise_gp(sel & 0xFFFC);
    if (!d.present()) raise(kNP, sel & 0xFFFC);
    if (rpl > cpl) {
        // Return to an outer privilege level: the new SS:ESP follow on the stack.
        const u32 smask = st_.stack32() ? 0xFFFFFFFFu : 0xFFFFu;
        const u32 new_sp = mem_.read(linear(kSs, (sp + 2 * size + pop_bytes) & smask, size, Access::Read), size);
        const u16 new_ss = static_cast<u16>(mem_.read(linear(kSs, (sp + 3 * size + pop_bytes) & smask, size, Access::Read), size));
        if ((new_ss & 0xFFFC) == 0) raise_gp(0);
        const Descriptor sd = read_descriptor(new_ss);
        if ((new_ss & 3u) != rpl || sd.system() || !sd.writable() || sd.dpl() != rpl) raise_gp(new_ss & 0xFFFC);
        if (!sd.present()) raise(kSS, new_ss & 0xFFFC);
        if (off > d.limit()) raise_gp(0);
        load_cs(sel, d);
        st_.eip = off;
        set_accessed(new_ss);
        load_cached(kSs, new_ss, sd);
        set_stack_pointer(new_sp + pop_bytes);
        // Data segments the outer level may not keep are cleared.
        for (u8 s : {kEs, kDs, kFs, kGs}) {
            Segment& seg = st_.seg[s];
            if (seg.unusable()) continue;
            if (!seg.conforming() && seg.dpl() < rpl && (seg.data() || !seg.conforming())) {
                if (seg.data() || seg.code()) {
                    if (seg.dpl() < rpl && !(seg.code() && seg.conforming())) {
                        seg.selector = 0;
                        seg.ar = Segment::kUnusable;
                    }
                }
            }
        }
        return;
    }
    if (off > d.limit()) raise_gp(0);
    load_cs(sel, d);
    st_.eip = off;
    set_stack_pointer(sp + 2 * size + pop_bytes);
}

void Cpu::iret(unsigned size) {
    if (st_.v8086()) throw Unimplemented{"IRET in virtual-8086 mode"};
    if (st_.flags & fl::NT) throw Unimplemented{"IRET task return (NT set)"};
    const u32 sp = stack_pointer();
    const u32 smask = st_.stack32() ? 0xFFFFFFFFu : 0xFFFFu;
    const u32 mask = size == 2 ? 0xFFFFu : 0xFFFFFFFFu;
    const u32 off = mem_.read(linear(kSs, sp, size, Access::Read), size) & mask;
    const u16 sel = static_cast<u16>(mem_.read(linear(kSs, (sp + size) & smask, size, Access::Read), size));
    const u32 flags_img = mem_.read(linear(kSs, (sp + 2 * size) & smask, size, Access::Read), size);
    const u32 old_flags = st_.eflags();
    auto apply_flags = [&](unsigned cpl_now) {
        u32 writable = fl::kWritable;
        if (size == 2) writable &= 0xFFFFu;
        if (st_.protected_mode()) {
            if (cpl_now > 0) writable &= ~fl::IOPL;
            if (cpl_now > st_.iopl()) writable &= ~fl::IF;
        }
        u32 nf = (old_flags & ~writable) | (flags_img & writable);
        if (size == 4 && cpl_now == 0) nf = (nf & ~fl::VM) | (flags_img & fl::VM);
        if (nf & fl::VM) throw Unimplemented{"IRET into virtual-8086 mode"};
        st_.set_eflags(nf & ~fl::RF);
    };
    if (!st_.protected_mode()) {
        set_real_mode_segment(kCs, sel);
        st_.eip = off;
        apply_flags(0);
        set_stack_pointer(sp + 3 * size);
        return;
    }
    if ((sel & 0xFFFC) == 0) raise_gp(0);
    const Descriptor d = read_descriptor(sel);
    const unsigned rpl = sel & 3, cpl = st_.cpl();
    if (rpl < cpl) raise_gp(sel & 0xFFFC);
    if (d.system() || !d.code()) raise_gp(sel & 0xFFFC);
    if (d.conforming() ? d.dpl() > rpl : d.dpl() != rpl) raise_gp(sel & 0xFFFC);
    if (!d.present()) raise(kNP, sel & 0xFFFC);
    if (off > d.limit()) raise_gp(0);
    if (rpl > cpl) {
        const u32 new_sp = mem_.read(linear(kSs, (sp + 3 * size) & smask, size, Access::Read), size);
        const u16 new_ss = static_cast<u16>(mem_.read(linear(kSs, (sp + 4 * size) & smask, size, Access::Read), size));
        if ((new_ss & 0xFFFC) == 0) raise_gp(0);
        const Descriptor sd = read_descriptor(new_ss);
        if ((new_ss & 3u) != rpl || sd.system() || !sd.writable() || sd.dpl() != rpl) raise_gp(new_ss & 0xFFFC);
        if (!sd.present()) raise(kSS, new_ss & 0xFFFC);
        load_cs(sel, d);
        st_.eip = off;
        apply_flags(cpl);
        set_accessed(new_ss);
        load_cached(kSs, new_ss, sd);
        set_stack_pointer(new_sp);
        for (u8 s : {kEs, kDs, kFs, kGs}) {
            Segment& seg = st_.seg[s];
            if (seg.unusable()) continue;
            if (seg.dpl() < rpl && !(seg.code() && seg.conforming())) {
                seg.selector = 0;
                seg.ar = Segment::kUnusable;
            }
        }
        return;
    }
    load_cs(sel, d);
    st_.eip = off;
    apply_flags(cpl);
    set_stack_pointer(sp + 3 * size);
}

// ---------------------------------------------------------------------------------------------------------------
// Interrupt and exception delivery

void Cpu::deliver(u8 vector, bool external, bool software, bool has_error, u32 error_code) {
    trace::log(trace::Cat::Cpu, "deliver vector {:02x}{}{} at {:04x}:{:08x}", vector, external ? " (external)" : "", software ? " (software)" : "",
               st_.seg[kCs].selector, st_.eip);
    const u32 ext = external ? 1u : 0u;
    if (!st_.protected_mode()) {
        const u32 entry = static_cast<u32>(vector) * 4;
        if (entry + 3 > st_.idtr.limit) raise_gp(0);
        const u32 off = mem_.read(st_.idtr.base + entry, 2);
        const u16 sel = static_cast<u16>(mem_.read(st_.idtr.base + entry + 2, 2));
        const u32 sp = stack_pointer();
        const u32 flags = st_.eflags();
        try {
            push(2, flags & 0xFFFFu);
            push(2, st_.seg[kCs].selector);
            push(2, st_.eip);
        } catch (const Fault&) {
            set_stack_pointer(sp);
            throw;
        }
        st_.set_eflags(flags & ~(fl::IF | fl::TF | fl::AC | fl::RF));
        set_real_mode_segment(kCs, sel);
        st_.eip = off;
        return;
    }
    if (st_.v8086()) throw Unimplemented{"interrupt delivery in virtual-8086 mode"};
    const u32 entry = static_cast<u32>(vector) * 8;
    if (entry + 7 > st_.idtr.limit) raise_gp(entry | 2 | ext);
    Descriptor g;
    g.lo = mem_.read(st_.idtr.base + entry, 4);
    g.hi = mem_.read(st_.idtr.base + entry + 4, 4);
    const u8 type = g.type();
    const bool gate32 = type == 0xE || type == 0xF;
    const bool gate16 = type == 6 || type == 7;
    if (type == 5) throw Unimplemented{"task gate in the IDT"};
    if (!g.system() || !(gate32 || gate16)) raise_gp(entry | 2 | ext);
    if (software && g.dpl() < st_.cpl()) raise_gp(entry | 2);
    if (!g.present()) raise(kNP, entry | 2 | ext);
    const u16 sel = static_cast<u16>(g.lo >> 16);
    const u32 off = (g.lo & 0xFFFFu) | (g.hi & 0xFFFF0000u);
    if ((sel & 0xFFFC) == 0) raise_gp(ext);
    const Descriptor d = read_descriptor(sel);
    if (d.system() || !d.code()) raise_gp((sel & 0xFFFC) | ext);
    const unsigned cpl = st_.cpl();
    if (d.dpl() > cpl) raise_gp((sel & 0xFFFC) | ext);
    if (!d.present()) raise(kNP, (sel & 0xFFFC) | ext);
    const unsigned push_size = gate32 ? 4 : 2;
    const u32 flags = st_.eflags();
    const u32 target = gate32 ? off : (off & 0xFFFFu);
    if (target > d.limit()) raise_gp(ext);
    if (!d.conforming() && d.dpl() < cpl) {
        // Inner privilege level: the stack comes from the TSS.
        const Segment& tr = st_.tr;
        if (tr.unusable()) raise(kTS, ext);
        const bool tss32 = (tr.type() & 8) != 0;
        const u32 sp_off = tss32 ? 4 + d.dpl() * 8 : 2 + d.dpl() * 4;
        if (sp_off + (tss32 ? 7 : 3) > tr.limit) raise(kTS, (tr.selector & 0xFFFC) | ext);
        const u32 new_sp = tss32 ? mem_.read(tr.base + sp_off, 4) : mem_.read(tr.base + sp_off, 2);
        const u16 new_ss = static_cast<u16>(mem_.read(tr.base + sp_off + (tss32 ? 4 : 2), 2));
        if ((new_ss & 0xFFFC) == 0) raise(kTS, ext);
        const Descriptor sd = read_descriptor(new_ss, kTS);
        if ((new_ss & 3u) != d.dpl() || sd.system() || !sd.writable() || sd.dpl() != d.dpl()) raise(kTS, (new_ss & 0xFFFC) | ext);
        if (!sd.present()) raise(kSS, (new_ss & 0xFFFC) | ext);
        const Segment old_ss = st_.seg[kSs];
        const u32 old_sp = st_.gpr[kEsp];
        set_accessed(new_ss);
        load_cached(kSs, new_ss, sd);
        set_stack_pointer(new_sp);
        try {
            push(push_size, old_ss.selector);
            push(push_size, old_sp);
            push(push_size, flags);
            push(push_size, st_.seg[kCs].selector);
            push(push_size, st_.eip);
            if (has_error) push(push_size, error_code);
        } catch (const Fault&) {
            st_.seg[kSs] = old_ss;
            st_.gpr[kEsp] = old_sp;
            throw;
        }
        load_cs(static_cast<u16>((sel & 0xFFFC) | d.dpl()), d);
    } else {
        const u32 sp = stack_pointer();
        try {
            push(push_size, flags);
            push(push_size, st_.seg[kCs].selector);
            push(push_size, st_.eip);
            if (has_error) push(push_size, error_code);
        } catch (const Fault&) {
            set_stack_pointer(sp);
            throw;
        }
        load_cs(static_cast<u16>((sel & 0xFFFC) | cpl), d);
    }
    u32 nf = flags & ~(fl::TF | fl::RF | fl::NT | fl::VM);
    if (type == 0xE || type == 6) nf &= ~fl::IF;
    st_.set_eflags(nf);
    st_.eip = target;
}

// ---------------------------------------------------------------------------------------------------------------
// Control registers, MSRs, CPUID

u32 Cpu::read_cr(unsigned n) {
    if (st_.cpl() != 0) raise_gp(0);
    switch (n) {
        case 0: return st_.cr0;
        case 2: return st_.cr2;
        case 3: return st_.cr3;
        case 4: return st_.cr4;
        default: raise_ud();
    }
}

void Cpu::write_cr(unsigned n, u32 v) {
    if (st_.cpl() != 0) raise_gp(0);
    switch (n) {
        case 0: {
            const u32 old = st_.cr0;
            v = (v & 0xE005003Fu) | cr0::ET;  // PG CD NW AM WP NE ET TS EM MP PE
            if ((v & cr0::PG) && !(v & cr0::PE)) raise_gp(0);
            if ((v & cr0::NW) && !(v & cr0::CD)) raise_gp(0);
            st_.cr0 = v;
            if ((old ^ v) & (cr0::PG | cr0::WP | cr0::PE)) mem_.flush_tlb(false);
            if ((old ^ v) & (cr0::PE | cr0::PG))
                trace::log(trace::Cat::Cpu, "cr0 {:08x} -> {:08x} ({} mode, paging {})", old, v, (v & cr0::PE) ? "protected" : "real",
                           (v & cr0::PG) ? "on" : "off");
            return;
        }
        case 2: st_.cr2 = v; return;
        case 3:
            st_.cr3 = v & 0xFFFFF018u;
            mem_.flush_tlb((st_.cr4 & cr4::PGE) != 0);
            return;
        case 4: {
            if (v & ~cr4::kKnown) raise_gp(0);
            if (v & cr4::PAE) throw Unimplemented{"CR4.PAE (36-bit paging)"};
            const u32 old = st_.cr4;
            st_.cr4 = v;
            if ((old ^ v) & (cr4::PSE | cr4::PGE | cr4::PAE)) mem_.flush_tlb(false);
            return;
        }
        default: raise_ud();
    }
}

u64 Cpu::read_msr(u32 index) {
    switch (index) {
        case 0x10: return cycles_ + st_.tsc;
        case 0x17: return 0;  // IA32_PLATFORM_ID
        case 0x1B: return st_.apic_base;
        case 0x2A: return 0;  // EBL_CR_POWERON
        case 0x79: return 0;  // BIOS_UPDT_TRIG
        case 0x8B: return st_.bios_sign_id;
        case 0xC1: case 0xC2: return 0;  // PERFCTR0/1
        case 0xFE: return 0x508;  // MTRRcap: 8 variable ranges, fixed ranges, write-combining
        case 0x174: return st_.sysenter_cs;
        case 0x175: return st_.sysenter_esp;
        case 0x176: return st_.sysenter_eip;
        case 0x179: return 0;  // MCG_CAP: no banks
        case 0x17A: return 0;  // MCG_STATUS
        case 0x186: case 0x187: return 0;  // PERFEVTSEL0/1
        case 0x1A0: return st_.misc_enable;
        case 0x1D9: return 0;  // DEBUGCTL
        case 0x1DB: case 0x1DC: case 0x1DD: case 0x1DE: return 0;  // LastBranch/Int From/To
        case 0x277: return st_.pat;
        case 0x2FF: return st_.mtrr_def_type;
        default: break;
    }
    if (index >= 0x200 && index <= 0x20F) return st_.mtrr_var[index - 0x200];
    if (index == 0x250) return st_.mtrr_fixed[0];
    if (index == 0x258) return st_.mtrr_fixed[1];
    if (index == 0x259) return st_.mtrr_fixed[2];
    if (index >= 0x268 && index <= 0x26F) return st_.mtrr_fixed[3 + index - 0x268];
    if (index >= 0x400 && index <= 0x413) return 0;  // MC banks
    trace::log(trace::Cat::Cpu, "rdmsr of unknown MSR {:08x} at {:04x}:{:08x}", index, insn_cs_, insn_eip_);
    raise_gp(0);
}

void Cpu::write_msr(u32 index, u64 v) {
    switch (index) {
        case 0x10: st_.tsc = v - cycles_; return;
        case 0x1B: st_.apic_base = v; return;
        case 0x79: return;  // BIOS update trigger: no microcode here
        case 0x8B: st_.bios_sign_id = v; return;
        case 0xC1: case 0xC2: case 0x186: case 0x187: return;
        case 0x174: st_.sysenter_cs = static_cast<u32>(v); return;
        case 0x175: st_.sysenter_esp = static_cast<u32>(v); return;
        case 0x176: st_.sysenter_eip = static_cast<u32>(v); return;
        case 0x17A: return;
        case 0x1A0: st_.misc_enable = v; return;
        case 0x1D9: return;
        case 0x277: st_.pat = v; return;
        case 0x2FF: st_.mtrr_def_type = v; return;
        default: break;
    }
    if (index >= 0x200 && index <= 0x20F) {
        st_.mtrr_var[index - 0x200] = v;
        return;
    }
    if (index == 0x250) { st_.mtrr_fixed[0] = v; return; }
    if (index == 0x258) { st_.mtrr_fixed[1] = v; return; }
    if (index == 0x259) { st_.mtrr_fixed[2] = v; return; }
    if (index >= 0x268 && index <= 0x26F) {
        st_.mtrr_fixed[3 + index - 0x268] = v;
        return;
    }
    if (index >= 0x400 && index <= 0x413) return;
    trace::log(trace::Cat::Cpu, "wrmsr to unknown MSR {:08x} = {:016x} at {:04x}:{:08x}", index, v, insn_cs_, insn_eip_);
    raise_gp(0);
}

void Cpu::cpuid() {
    // A Coppermine as the fast mode presents it; functions past the highest basic leaf answer like that leaf.
    u32 fn = st_.gpr[kEax];
    if (fn > 2) fn = 2;
    u32 a = 0, b = 0, c = 0, d = 0;
    switch (fn) {
        case 0: a = 2; b = 0x756E6547; d = 0x49656E69; c = 0x6C65746E; break;
        case 1: a = cpuid_signature_; b = 0; c = 0; d = 0x0383F9FF; break;
        default: a = 0x03020101; b = 0; c = 0; d = 0x0C040841; break;
    }
    st_.gpr[kEax] = a;
    st_.gpr[kEbx] = b;
    st_.gpr[kEcx] = c;
    st_.gpr[kEdx] = d;
    last_cpuid_ = true;
}

// ---------------------------------------------------------------------------------------------------------------
// The privileged and system instructions

void Cpu::exec_system(const Insn& in) {
    const unsigned size = in.size;
    switch (in.op) {
        case Op::Hlt:
            if (st_.cpl() != 0) raise_gp(0);
            st_.halted = true;
            return;
        case Op::Wait:
            if ((st_.cr0 & cr0::MP) && (st_.cr0 & cr0::TS)) raise(kNM);
            if ((st_.fpu.fsw & 0x80) && (st_.cr0 & cr0::NE)) raise(kMF);
            return;
        case Op::Cpuid: cpuid(); return;
        case Op::Rdtsc: {
            if ((st_.cr4 & cr4::TSD) && st_.cpl() != 0) raise_gp(0);
            const u64 t = cycles_ + st_.tsc;
            st_.gpr[kEax] = static_cast<u32>(t);
            st_.gpr[kEdx] = static_cast<u32>(t >> 32);
            last_rdtsc_ = true;
            return;
        }
        case Op::Rdmsr: {
            if (st_.cpl() != 0) raise_gp(0);
            const u64 v = read_msr(st_.gpr[kEcx]);
            st_.gpr[kEax] = static_cast<u32>(v);
            st_.gpr[kEdx] = static_cast<u32>(v >> 32);
            if (st_.gpr[kEcx] == 0x10) last_rdtsc_ = true;
            return;
        }
        case Op::Wrmsr:
            if (st_.cpl() != 0) raise_gp(0);
            write_msr(st_.gpr[kEcx], (static_cast<u64>(st_.gpr[kEdx]) << 32) | st_.gpr[kEax]);
            return;
        case Op::Rdpmc:
            if ((st_.gpr[kEcx] & 0x7FFFFFFFu) > 1) raise_gp(0);
            if (st_.cpl() != 0 && !(st_.cr4 & cr4::PCE)) raise_gp(0);
            st_.gpr[kEax] = 0;
            st_.gpr[kEdx] = 0;
            return;
        case Op::Sysenter: {
            if (!st_.protected_mode() || (st_.sysenter_cs & 0xFFFC) == 0) raise_gp(0);
            st_.materialise();
            st_.flags &= ~(fl::VM | fl::IF | fl::RF);
            Segment& cs = st_.seg[kCs];
            cs.selector = static_cast<u16>(st_.sysenter_cs & 0xFFFC);
            cs.base = 0;
            cs.limit = 0xFFFFFFFFu;
            cs.ar = 0xC09B;
            Segment& ss = st_.seg[kSs];
            ss.selector = static_cast<u16>((st_.sysenter_cs + 8) & 0xFFFC);
            ss.base = 0;
            ss.limit = 0xFFFFFFFFu;
            ss.ar = 0xC093;
            st_.gpr[kEsp] = st_.sysenter_esp;
            st_.eip = st_.sysenter_eip;
            return;
        }
        case Op::Sysexit: {
            if (!st_.protected_mode() || (st_.sysenter_cs & 0xFFFC) == 0 || st_.cpl() != 0) raise_gp(0);
            Segment& cs = st_.seg[kCs];
            cs.selector = static_cast<u16>(((st_.sysenter_cs + 16) & 0xFFFC) | 3);
            cs.base = 0;
            cs.limit = 0xFFFFFFFFu;
            cs.ar = 0xC0FB;
            Segment& ss = st_.seg[kSs];
            ss.selector = static_cast<u16>(((st_.sysenter_cs + 24) & 0xFFFC) | 3);
            ss.base = 0;
            ss.limit = 0xFFFFFFFFu;
            ss.ar = 0xC0F3;
            st_.gpr[kEsp] = st_.gpr[kEcx];
            st_.eip = st_.gpr[kEdx];
            return;
        }
        case Op::Wbinvd: case Op::Invd:
            if (st_.cpl() != 0) raise_gp(0);
            return;
        case Op::Ud2: case Op::Rsm: raise_ud();
        case Op::Bound: {
            const u32 lin = linear(in.mem_seg, effective_address(in), in.src_size, Access::Read);
            const i32 lo = static_cast<i32>(sign_extend(mem_.read(lin, size), size));
            const i32 hi = static_cast<i32>(sign_extend(mem_.read(lin + size, size), size));
            const i32 v = static_cast<i32>(sign_extend(st_.reg(in.dst.reg, size), size));
            if (v < lo || v > hi) raise(kBR);
            return;
        }
        case Op::Arpl: {
            if (!st_.protected_mode() || st_.v8086()) raise_ud();
            const u32 dst = read_op(in, in.dst, 2);
            const u32 src = read_op(in, in.src, 2);
            st_.materialise();
            if ((dst & 3) < (src & 3)) {
                write_op(in, in.dst, 2, (dst & 0xFFFCu) | (src & 3));
                st_.flags |= fl::ZF;
            } else {
                st_.flags &= ~fl::ZF;
            }
            return;
        }
        case Op::Lar: case Op::Lsl: {
            if (!st_.protected_mode() || st_.v8086()) raise_ud();
            const u16 sel = static_cast<u16>(read_op(in, in.src, 2));
            st_.materialise();
            bool ok = (sel & 0xFFFC) != 0;
            Descriptor d{};
            if (ok) {
                const u32 index = sel & ~7u;
                const bool ldt = (sel & 4) != 0;
                const u32 base = ldt ? st_.ldtr.base : st_.gdtr.base;
                const u32 limit = ldt ? st_.ldtr.limit : st_.gdtr.limit;
                if ((ldt && st_.ldtr.unusable()) || index + 7 > limit) {
                    ok = false;
                } else {
                    d.lo = mem_.read(base + index, 4);
                    d.hi = mem_.read(base + index + 4, 4);
                    if (d.system()) {
                        static constexpr bool kLarOk[16] = {false, true, true, true, true, true, false, false, false, true, false, true, true, false, true, true};
                        static constexpr bool kLslOk[16] = {false, true, true, true, false, false, false, false, false, true, false, true, false, false, false, false};
                        ok = in.op == Op::Lar ? kLarOk[d.type()] : kLslOk[d.type()];
                    } else if (!(d.code() && d.conforming())) {
                        const unsigned rpl = sel & 3;
                        if (d.dpl() < st_.cpl() || d.dpl() < rpl) ok = false;
                    }
                }
            }
            if (!ok) {
                st_.flags &= ~fl::ZF;
                return;
            }
            st_.flags |= fl::ZF;
            if (in.op == Op::Lar) st_.set_reg(in.dst.reg, size, d.hi & (size == 2 ? 0xFF00u : 0x00FFFF00u));
            else st_.set_reg(in.dst.reg, size, d.limit());
            return;
        }
        case Op::MovToCr: write_cr(in.dst.reg, st_.gpr[in.src.reg]); return;
        case Op::MovFromCr: st_.gpr[in.dst.reg] = read_cr(in.src.reg); return;
        case Op::MovToDr: write_op(in, in.dst, 4, st_.gpr[in.src.reg]); return;
        case Op::MovFromDr: st_.gpr[in.dst.reg] = read_op(in, in.src, 4); return;
        case Op::Sldt:
            if (!st_.protected_mode()) raise_ud();
            if (in.dst.kind == OpndKind::Mem) write_op(in, in.dst, 2, st_.ldtr.selector);
            else write_op(in, in.dst, size, st_.ldtr.selector);
            return;
        case Op::Str:
            if (!st_.protected_mode()) raise_ud();
            if (in.dst.kind == OpndKind::Mem) write_op(in, in.dst, 2, st_.tr.selector);
            else write_op(in, in.dst, size, st_.tr.selector);
            return;
        case Op::Lldt: {
            if (!st_.protected_mode()) raise_ud();
            if (st_.cpl() != 0) raise_gp(0);
            const u16 sel = static_cast<u16>(read_op(in, in.dst, 2));
            if ((sel & 0xFFFC) == 0) {
                st_.ldtr = Segment{sel, 0, 0, Segment::kUnusable};
                return;
            }
            if (sel & 4) raise_gp(sel & 0xFFFC);
            const Descriptor d = read_descriptor(sel);
            if (!d.system() || d.type() != 2) raise_gp(sel & 0xFFFC);
            if (!d.present()) raise(kNP, sel & 0xFFFC);
            st_.ldtr = Segment{sel, d.base(), d.limit(), d.ar()};
            return;
        }
        case Op::Ltr: {
            if (!st_.protected_mode()) raise_ud();
            if (st_.cpl() != 0) raise_gp(0);
            const u16 sel = static_cast<u16>(read_op(in, in.dst, 2));
            if ((sel & 0xFFFC) == 0 || (sel & 4)) raise_gp(sel & 0xFFFC);
            const Descriptor d = read_descriptor(sel);
            if (!d.system() || (d.type() != 1 && d.type() != 9)) raise_gp(sel & 0xFFFC);
            if (!d.present()) raise(kNP, sel & 0xFFFC);
            // Mark the TSS busy.
            const u32 addr = st_.gdtr.base + (sel & ~7u) + 4;
            mem_.write(addr, 4, d.hi | 0x200);
            st_.tr = Segment{sel, d.base(), d.limit(), d.ar() | 0x2};
            return;
        }
        case Op::Verr: case Op::Verw: {
            if (!st_.protected_mode()) raise_ud();
            const u16 sel = static_cast<u16>(read_op(in, in.dst, 2));
            st_.materialise();
            bool ok = (sel & 0xFFFC) != 0;
            if (ok) {
                const u32 index = sel & ~7u;
                const bool ldt = (sel & 4) != 0;
                const u32 base = ldt ? st_.ldtr.base : st_.gdtr.base;
                const u32 limit = ldt ? st_.ldtr.limit : st_.gdtr.limit;
                if ((ldt && st_.ldtr.unusable()) || index + 7 > limit) {
                    ok = false;
                } else {
                    Descriptor d;
                    d.lo = mem_.read(base + index, 4);
                    d.hi = mem_.read(base + index + 4, 4);
                    if (d.system()) ok = false;
                    else if (in.op == Op::Verr) ok = d.readable();
                    else ok = d.writable();
                    if (ok && !(d.code() && d.conforming())) {
                        if (d.dpl() < st_.cpl() || d.dpl() < (sel & 3u)) ok = false;
                    }
                }
            }
            st_.flags = (st_.flags & ~fl::ZF) | (ok ? fl::ZF : 0);
            return;
        }
        case Op::Sgdt: case Op::Sidt: {
            const DescTable& t = in.op == Op::Sgdt ? st_.gdtr : st_.idtr;
            const u32 lin = linear(in.mem_seg, effective_address(in), 6, Access::Write);
            mem_.probe_write(lin, 6);
            mem_.write(lin, 2, t.limit);
            mem_.write(lin + 2, 4, size == 2 ? (t.base & 0x00FFFFFFu) : t.base);
            return;
        }
        case Op::Lgdt: case Op::Lidt: {
            if (st_.cpl() != 0) raise_gp(0);
            const u32 lin = linear(in.mem_seg, effective_address(in), 6, Access::Read);
            DescTable t;
            t.limit = mem_.read(lin, 2);
            t.base = mem_.read(lin + 2, 4);
            if (size == 2) t.base &= 0x00FFFFFFu;
            if (in.op == Op::Lgdt) st_.gdtr = t;
            else st_.idtr = t;
            trace::log(trace::Cat::Cpu, "{} base {:08x} limit {:04x}", in.op == Op::Lgdt ? "lgdt" : "lidt", t.base, t.limit);
            return;
        }
        case Op::Smsw:
            if (in.dst.kind == OpndKind::Mem) write_op(in, in.dst, 2, st_.cr0 & 0xFFFFu);
            else write_op(in, in.dst, size, st_.cr0);
            return;
        case Op::Lmsw: {
            if (st_.cpl() != 0) raise_gp(0);
            const u32 v = read_op(in, in.dst, 2) & 0xF;
            // LMSW cannot clear PE.
            write_cr(0, (st_.cr0 & ~0xEu) | v | (st_.cr0 & cr0::PE));
            return;
        }
        case Op::Invlpg:
            if (st_.cpl() != 0) raise_gp(0);
            mem_.flush_page(st_.seg[in.mem_seg].base + effective_address(in));
            return;
        case Op::Clts:
            if (st_.cpl() != 0) raise_gp(0);
            st_.cr0 &= ~cr0::TS;
            return;
        default: break;
    }
    throw Unimplemented{op_name(in.op)};
}

}  // namespace mv::p6
