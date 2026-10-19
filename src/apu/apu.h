#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "apu/processor.h"
#include "apu/setup_engine.h"
#include "core/bus.h"
#include "core/scheduler.h"
#include "pci/pci.h"

// The MCPX audio processing unit as the kernel programs it: PCI function 0:5.0 with one 512 KiB register window.
// The window holds the global registers (interrupt status and enable, the front end, the setup engine with the
// scatter-gather table bases and the voice lists, the processors' FIFO rings), the voice processor's method window,
// and the two programmable processors — the global processor (GP) and the encode processor (EP) — each a DSP56300
// core whose X, Y and P memories appear in the window with a reset word at the top of its block. A processor's
// scratch and FIFO address spaces are concatenations of 4 KiB RAM pages named by scatter-gather tables the kernel
// builds in RAM; a FIFO is a ring inside the FIFO space described by base/end/current registers.
// Written from the xboxdevwiki APU and DSP pages, the DSP56300 family manual and traces of the kernel's register
// programming; register offsets per the public register listing of the reference emulator (behaviour, no code).
namespace mv {

class Apu : public PciDevice, public MmioHandler, public apu::SetupEngineHost {
public:
    static constexpr u32 kBarSize = 512u * 1024;
    // Global registers.
    static constexpr u32 kIsts = 0x1000, kIen = 0x1004;  // interrupt status (write one to clear) and enable
    // Status bit 0 is the summary: set while another enabled bit is pending and bit 0 is enabled; the line follows it.
    static constexpr u32 kIstsGlobal = 1u << 0, kIstsFrontEndTrap = 1u << 4;
    static constexpr u32 kFectl = 0x1100, kSectl = 0x2000;
    // Front-end control: the method mode (bits 7:5; all set = trapped, waiting for the host) and the trap reason
    // (bits 11:8; all set = requested); the trapped method and its parameter.
    static constexpr u32 kFectlMethodMode = 0xE0, kFectlTrapped = 0xE0, kFectlTrapReason = 0xF00, kFectlTrapRequested = 0xF00;
    static constexpr u32 kFeDecMethod = 0x1300, kFeDecParam = 0x1304;
    static constexpr u32 kGpScratchTable = 0x2040, kGpFifoTable = 0x2044, kEpScratchTable = 0x2048, kEpFifoTable = 0x204C;
    // Each holds the highest valid entry index of its table.
    static constexpr u32 kGpScratchMaxSge = 0x20D4, kGpFifoMaxSge = 0x20D8, kEpScratchMaxSge = 0x20DC, kEpFifoMaxSge = 0x20E0;
    // FIFO rings: output 0-3 then input 0-1, 16 bytes each: base, end (exclusive), current.
    static constexpr u32 kGpFifoRegs = 0x3024, kEpFifoRegs = 0x4024;
    static constexpr u32 kFifoBase = 0, kFifoEnd = 4, kFifoCurrent = 8, kFifoStride = 0x10, kInputFifoRegs = 0x40;
    // Blocks of the window.
    static constexpr u32 kVpBlock = 0x20000, kGpBlock = 0x30000, kEpBlock = 0x50000, kBlockSize = 0x10000;
    static constexpr u32 kVpFree = 0x10;  // free entries in the voice processor's method queue
    // Inside a processor's block: the memories as 32-bit words holding 24 bits, and the reset word.
    static constexpr u32 kXMem = 0x0000, kMixBuf = 0x5000, kYMem = 0x6000, kPMem = 0xA000, kReset = 0xFFFC;
    static constexpr u32 kResetBlock = 1u << 0, kResetDsp = 1u << 1;  // both set = the processor runs
    static constexpr u32 kMixBufWord = 0x1400, kMixBufWords = 0x400;
    static constexpr u64 kDspHz = 160'000'000;  // the processors' clock per the wiki [verify]
    // The frame clock. Frames run only while the setup engine's count mode (SECTL bits 4:3) is on; the host stops
    // it, zeroes the global sample count XGSCNT, releases the GP and EP, then turns it on, so both processors see
    // the same first frame (the kernel and the titles do exactly this). XGSCNT counts 32 samples a frame.
    static constexpr u32 kSectlCountMode = 0x18, kXgsCnt = 0x200C;
    // The EP works in 256-sample frames: its start-of-frame flag rises when the count is a multiple of eight frames,
    // and it then writes 256 stereo samples to FIFO 0 (the AC'97 rate); the GP's rings in scratch hold sixteen
    // frames, two of the EP's (xboxdevwiki APU). Like the GP it is resumed from halt every frame: a title's EP halts
    // inside its initialisation and finishes it on the next frame, then polls the flag; resumed only every eighth
    // frame its first block would land after the AC'97, started with an 8 ms lead, had already passed it [inferred].
    static constexpr u32 kEpFrameDivider = 8;

    Apu(Bus& bus, Scheduler& sched, std::function<void(bool)> irq);
    ~Apu() override;
    Apu(const Apu&) = delete;
    Apu& operator=(const Apu&) = delete;
    // The processors' frame runs on its own thread: wait until the one in flight (if any) has finished. Everything
    // that reads or writes their memories, their DMA or the FIFO registers from the machine's side calls it first.
    void dsp_sync() const;
    std::string_view name() const override { return PciDevice::name(); }
    u32 mmio_read(PhysAddr addr, unsigned size) override;
    void mmio_write(PhysAddr addr, unsigned size, u32 value) override;
    // The reset line: both processors held, every register cleared, no trap pending.
    void reset();

    // A register word as the kernel would read it (offset into the window, 4-byte aligned).
    u32 reg(u32 offset) {
        if (!in_vp_window(offset)) dsp_sync();
        return read_word(offset & ~3u);
    }
    // Raise interrupt status bits from a device-side event.
    void raise(u32 bits);
    bool irq_level() const { return irq_level_; }
    u64 reads() const { return reads_; }
    u64 writes() const { return writes_; }
    u64 frames() const { return frames_; }
    apu::Processor& gp() {
        dsp_sync();
        return gp_;
    }
    apu::Processor& ep() {
        dsp_sync();
        return ep_;
    }
    u64 fifo_out_bytes(bool ep, u32 fifo) const {
        dsp_sync();
        return (ep ? ep_memory_ : gp_memory_).fifo_out_bytes(fifo);
    }
    u64 fifo_in_bytes(bool ep, u32 fifo) const {
        dsp_sync();
        return (ep ? ep_memory_ : gp_memory_).fifo_in_bytes(fifo);
    }
    apu::SetupEngine& setup_engine() { return se_; }
    u64 traps() const { return traps_; }
    bool front_end_trapped() const { return fe_trapped_; }
    // SetupEngineHost.
    u32 global_reg(u32 offset) override { return regs_[offset / 4]; }
    void set_global_reg(u32 offset, u32 value) override { regs_[offset / 4] = value; }
    void front_end_trap(u32 method, u32 parameter) override;
    void raise_interrupt(u32 ists_bits) override { raise(ists_bits); }
    // The last frame's cycle use and the longest so far, per processor.
    struct FrameStats {
        u64 last_cycles{0}, max_cycles{0}, frames_run{0};
        u64 host_ns{0};       // host time inside the processor's frames
        u64 total_cycles{0};  // DSP cycles they used (idle-wait skips included)
    };
    const FrameStats& gp_stats() const {
        dsp_sync();
        return gp_stats_;
    }
    const FrameStats& ep_stats() const {
        dsp_sync();
        return ep_stats_;
    }
    u64 setup_engine_host_ns() const { return se_host_ns_; }  // host time inside the setup engine's frames
    // MV_DEBUG_DSP_PCHIST=gp,ep: the program addresses each processor spent its executed cycles at, busiest first.
    std::string pc_histogram_report(std::size_t top) const;

protected:
    void on_bar_changed(int index, u32 old_base, u32 new_base) override;

private:
    // One processor's view of RAM: its scratch and FIFO spaces through the kernel's tables; the rings in registers.
    class Memory : public dsp::DmaMemory {
    public:
        Memory(Apu& apu, u32 scratch_table, u32 scratch_max, u32 fifo_table, u32 fifo_max, u32 fifo_regs)
            : apu_(apu), scratch_table_(scratch_table), scratch_max_(scratch_max), fifo_table_(fifo_table), fifo_max_(fifo_max),
              fifo_regs_(fifo_regs) {}
        bool scratch_read(u32 offset, u8* out, u32 len) override;
        bool scratch_write(u32 offset, const u8* in, u32 len) override;
        bool fifo_read(u32 fifo, u8* out, u32 len) override;
        bool fifo_write(u32 fifo, const u8* in, u32 len) override;
        const std::string& last_problem() const { return problem_; }
        // Instruments: bytes the program wrote to each output FIFO and read from each input FIFO.
        u64 fifo_out_bytes(u32 fifo) const { return fifo < 4 ? out_bytes_[fifo] : 0; }
        u64 fifo_in_bytes(u32 fifo) const { return fifo < 2 ? in_bytes_[fifo] : 0; }

    private:
        bool translate(u32 table_reg, u32 max_reg, u32 offset, PhysAddr& phys);
        bool copy(u32 table_reg, u32 max_reg, u32 offset, u8* out, const u8* in, u32 len);
        bool ring_transfer(u32 regs, u8* out, const u8* in, u32 len);
        Apu& apu_;
        u32 scratch_table_, scratch_max_, fifo_table_, fifo_max_, fifo_regs_;
        std::string problem_;
        u64 out_bytes_[4]{}, in_bytes_[2]{};
    public:
        double out0_energy_{0};  // instrument: sum of squares of the 16-bit samples written to output FIFO 0
        u64 out0_samples_{0};
        u64 out0_frames_{0}, out0_differ_{0};  // instrument: stereo frames written to output FIFO 0, and those with L != R
    private:
    };

    u32 read_word(u32 offset);
    void write_word(u32 offset, u32 value);
    u32 processor_word(apu::Processor& p, u32 offset, u32 reset_word);
    bool processor_write(apu::Processor& p, u32 offset, u32 value);
    void apply_reset(apu::Processor& p, u32& reset_word, u32 value, std::string_view which);
    void run_frame(apu::Processor& p, FrameStats& stats, std::string_view which, bool start_of_frame);
    void run_processors(u64 frame, bool ep_start);
    void dsp_worker();
    static bool in_vp_window(u32 offset) { return offset >= kVpBlock && offset < kGpBlock; }
    void update_frame_clock();
    void on_frame(Cycles now);
    void update_irq();
    void deliver_trap(u32 method, u32 parameter);

    Bus& bus_;
    Scheduler& sched_;
    std::function<void(bool)> irq_;
    u32 mapped_at_{0};
    std::vector<u32> regs_;                 // the global registers and the voice processor's window
    std::unordered_map<u32, u32> other_;    // words of the processor blocks that are not memories
    u32 gp_reset_{0}, ep_reset_{0};
    Memory gp_memory_, ep_memory_;
    apu::Processor gp_, ep_;
    apu::SetupEngine se_;
    FrameStats gp_stats_, ep_stats_;
    u64 se_host_ns_{0};
    // The processors' thread: a frame posted by on_frame (dsp_busy_ while it runs), its number and EP start flag.
    bool dsp_threaded_{true};
    mutable std::mutex dsp_mutex_;
    mutable std::condition_variable dsp_cv_;
    std::atomic<bool> dsp_busy_{false};
    bool dsp_stop_{false};
    u64 dsp_job_frame_{0};
    bool dsp_job_ep_start_{false};
    std::thread dsp_thread_;  // started in the constructor's body, joined in the destructor's
    std::vector<u64> gp_hist_, ep_hist_;
    std::deque<std::pair<u32, u32>> trap_queue_;
    bool fe_trapped_{false};
    u64 traps_{0};
    TimerEvent frame_;
    u64 frames_{0};
    u64 mix_samples_{0}, mix_differ_{0};  // instrument: MIXBUF bin 0/1 samples with sound, and those where the two differ
    bool counting_{false};     // the frame clock runs (SECTL count mode on)
    Cycles frame_epoch_{0};    // when it was turned on
    u64 epoch_frames_{0};      // frames since then
    u32 xgscnt_{0};            // the global sample count
    bool irq_level_{false};
    u64 reads_{0}, writes_{0};
    bool gp_problem_logged_{false}, ep_problem_logged_{false};
    u64 ep_releases_{0}, ep_runs_since_release_{0};  // instruments
};

}  // namespace mv
