#include "flash/flash.h"

#include <cstring>
#include <format>
#include <fstream>

#include "debug/trace.h"

namespace mv {

Flash::Flash() : window_(kWindow, 0xFF), overlay_page_(4096, 0xFF) {}

namespace {
bool read_file(const std::filesystem::path& path, std::vector<u8>& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = std::format("cannot open {}", path.string());
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}
}  // namespace

bool Flash::load_image(const std::filesystem::path& path, std::string& error) {
    std::vector<u8> bytes;
    if (!read_file(path, bytes, error)) return false;
    return load_image(bytes, error);
}

bool Flash::load_image(const std::vector<u8>& bytes, std::string& error) {
    const std::size_t n = bytes.size();
    if (n != 256u * 1024 && n != 512u * 1024 && n != 1024u * 1024) {
        error = std::format("flash image is {} bytes; expected 256 KiB, 512 KiB or 1 MiB", n);
        return false;
    }
    for (std::size_t off = 0; off < kWindow; off += n) std::memcpy(window_.data() + off, bytes.data(), n);
    image_size_ = n;
    rebuild_overlay_page();
    return true;
}

bool Flash::load_mcpx_rom(const std::filesystem::path& path, std::string& error) {
    std::vector<u8> bytes;
    if (!read_file(path, bytes, error)) return false;
    if (bytes.size() != kRomSize) {
        error = std::format("MCPX ROM is {} bytes; expected 512", bytes.size());
        return false;
    }
    std::array<u8, kRomSize> a{};
    std::memcpy(a.data(), bytes.data(), kRomSize);
    return load_mcpx_rom(a);
}

bool Flash::load_mcpx_rom(const std::array<u8, kRomSize>& bytes) {
    rom_ = bytes;
    has_rom_ = true;
    rom_visible_ = true;
    rebuild_overlay_page();
    return true;
}

void Flash::rebuild_overlay_page() {
    std::memcpy(overlay_page_.data(), window_.data() + kWindow - overlay_page_.size(), overlay_page_.size());
    if (has_rom_) std::memcpy(overlay_page_.data() + overlay_page_.size() - kRomSize, rom_.data(), kRomSize);
}

void Flash::hide_rom() {
    if (!rom_visible_) return;
    rom_visible_ = false;
    trace::log(trace::Cat::Boot, "MCPX ROM hidden (LPC config 0x80 bit 1)");
    for (auto& cb : on_rom_hidden_) cb();
}

u8 Flash::read8(PhysAddr addr) const {
    if (rom_visible() && addr >= kRomBase) return rom_[addr - kRomBase];
    return window_[addr - kBase];
}

u32 Flash::mmio_read(PhysAddr addr, unsigned size) {
    u32 v = 0;
    for (unsigned i = 0; i < size; ++i) v |= static_cast<u32>(read8(addr + i)) << (8 * i);
    return v;
}

void Flash::mmio_write(PhysAddr addr, unsigned size, u32 value) {
    // Flash programming is not emulated; the kernel never writes its flash in normal operation.
    trace::log(trace::Cat::Flash, "write to flash ignored: {:08x} size {} value {:08x}", addr, size, value);
}

}  // namespace mv
