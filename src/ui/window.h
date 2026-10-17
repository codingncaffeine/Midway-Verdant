#pragma once
#include <atomic>
#include <string>

namespace mv {
class Machine;
}

namespace mv::ui {

// An SDL3 window that shows the console's scanout: the frame buffer the CRTC points at, copied out of guest RAM
// at the host display's refresh rate. The shell's business; the core never sees a window.
class ScanoutWindow {
public:
    ScanoutWindow();
    ~ScanoutWindow();
    ScanoutWindow(const ScanoutWindow&) = delete;
    ScanoutWindow& operator=(const ScanoutWindow&) = delete;

    bool open(std::string& error);
    // Pumps events and presents frames until `stop` is set (by the machine ending) or the window is closed or
    // Escape pressed, in which case it sets `stop` itself. Returns the number of frames presented.
    unsigned run(Machine& machine, std::atomic<bool>& stop);

private:
    struct Impl;
    Impl* impl_{nullptr};
};

}  // namespace mv::ui
