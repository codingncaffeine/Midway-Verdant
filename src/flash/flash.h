#pragma once
#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "core/aligned_buffer.h"
#include "core/bus.h"
#include "core/types.h"

namespace mv {

// The flash ROM (256 KiB, 512 KiB or 1 MiB, aliased through the top 16 MiB of the address space) and the
// 512-byte MCPX boot ROM that overlays its last 512 bytes until the second bootloader hides it through the
// LPC bridge's configuration register 0x80.
class Flash : public MmioHandler {
public:
    static constexpr PhysAddr kBase = 0xFF000000u;
    static constexpr u64 kWindow = 16u * 1024 * 1024;
    static constexpr PhysAddr kRomBase = 0xFFFFFE00u;
    static constexpr u32 kRomSize = 512;

    Flash();
    std::string_view name() const override { return "flash"; }

    // Image sizes other than 256 KiB, 512 KiB or 1 MiB are rejected with a message.
    bool load_image(const std::filesystem::path& path, std::string& error);
    bool load_image(const std::vector<u8>& bytes, std::string& error);
    bool load_mcpx_rom(const std::filesystem::path& path, std::string& error);
    bool load_mcpx_rom(const std::array<u8, kRomSize>& bytes);

    std::size_t image_size() const { return image_size_; }
    bool has_image() const { return image_size_ != 0; }
    bool has_rom() const { return has_rom_; }
    bool rom_visible() const { return has_rom_ && rom_visible_; }

    // The 16 MiB window as the CPU sees the flash alone (no overlay). Stable address for the lifetime.
    const u8* window() const { return window_.data(); }
    // The last 4 KiB page with the MCPX ROM laid over its top 512 bytes.
    const u8* overlay_page() const { return overlay_page_.data(); }

    void hide_rom();
    // Several parties care (the machine for its milestone, the fast-mode backend for its memory slot).
    void add_on_rom_hidden(std::function<void()> cb) { on_rom_hidden_.push_back(std::move(cb)); }

    u32 mmio_read(PhysAddr addr, unsigned size) override;
    void mmio_write(PhysAddr addr, unsigned size, u32 value) override;

    // Reads honouring the overlay, for tools and the interpreter.
    u8 read8(PhysAddr addr) const;

private:
    void rebuild_overlay_page();

    AlignedBuffer window_;
    AlignedBuffer overlay_page_;
    std::array<u8, kRomSize> rom_{};
    std::size_t image_size_{0};
    bool has_rom_{false};
    bool rom_visible_{true};
    std::vector<std::function<void()>> on_rom_hidden_;
};

}  // namespace mv
