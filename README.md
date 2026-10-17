# Midway Verdant

A clean-room, low-level emulator of the original Xbox for Linux (Windows to follow). The console's own boot ROM, BIOS and kernel run unmodified on an emulated machine: MCPX southbridge, NV2A graphics, the APU with its DSPs, IDE storage, USB controllers and the SMBus devices. Nothing in the firmware is patched or replaced with a high-level substitute. The goal is cycle accuracy, with a hardware-virtualised fast mode for play speed.

**Status:** early development. Under the fast mode the firmware boots, initialises every device, runs its boot animation's audio engine on our DSP56300 cores, passes its audio handshake, and the animation's GPU stream renders offline through the Vulkan backend. Nothing is shown in a window yet. The [wiki](https://github.com/codingncaffeine/Midway-Verdant/wiki) describes every hardware block and keeps the [status page](https://github.com/codingncaffeine/Midway-Verdant/wiki/Status) current.

## Clean room

Midway Verdant is written without leaked or proprietary source material of any kind. Every hardware block is derived from public documentation and datasheets, community reverse-engineering (xboxdevwiki and published papers), public register listings, and observation of the console's own firmware running on the emulated machine. Open-source emulators are read for behaviour only; no code is copied. The wiki's [Clean-Room Methodology](https://github.com/codingncaffeine/Midway-Verdant/wiki/Clean-Room-Methodology) page explains the discipline, and the [Clean-Room Receipts](https://github.com/codingncaffeine/Midway-Verdant/wiki/Clean-Room-Receipts) ledger lists each load-bearing hardware fact with its source and how it was confirmed. A pre-commit scan refuses console dumps, game data and known proprietary hashes, so nothing of the kind can reach the repository.

## What you need

Midway Verdant contains no Microsoft code or data. You supply dumps from your own console:

- the 512-byte MCPX boot ROM
- a flash BIOS image (256 KiB, 512 KiB or 1 MiB)
- a hard-disk image (raw; a qcow2 image can be converted)
- an EEPROM image (one can be generated from the running kernel's own key)

Game discs are read from XISO or Redump-format images.

## Building

Linux first: a C++23 compiler (GCC 14+ or Clang 18+), CMake 3.28+, Ninja, SDL3, a Vulkan 1.1 driver, glslang (fetched when the system has none). KVM (`/dev/kvm`) runs the fast mode; the cycle-counted CPU core is in development.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## License

GPL-3.0. See `LICENSE`.
