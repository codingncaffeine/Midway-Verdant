<p align="center"><img src="assets/branding/banner.png" alt="Midway Verdant" width="768"></p>

# Midway Verdant

A clean-room, low-level emulator of the original Xbox for Linux (Windows to follow). The console's own boot ROM, BIOS and kernel run unmodified on an emulated machine: MCPX southbridge, NV2A graphics, the APU with its DSPs, IDE storage, USB controllers and the SMBus devices. Nothing in the firmware is patched or replaced with a high-level substitute. The goal is cycle accuracy, with a hardware-virtualised fast mode for play speed.

**Status:** early development, past first boot. Under the fast mode the firmware boots, initialises every device and plays its boot animation with sound: the graphics render live through the Vulkan backend and the animation's audio runs on our DSP56300 cores. Retail game discs (XISO or full dumps) load through the emulated DVD drive; USB gamepads work with rumble; titles reach their menus and gameplay, several at their full frame rate on a desktop CPU, with known inaccuracies and some titles that still stall. A standalone SDL3 window and a libretro core run the machine. The wiki describes every hardware block and keeps the [status page](https://github.com/codingncaffeine/Midway-Verdant/wiki/Status) current.

| Area | State |
|---|---|
| Boot chain | Boot ROM, second-stage loader and kernel run unmodified; the kernel mounts its hard disk and starts a disc or the dashboard. Warm reboot works. |
| CPU | KVM fast mode (Linux, default). Our own Pentium III interpreter with a P6 pipeline cycle model (`--cpu p6`), checked instruction by instruction against a KVM oracle; not yet fast enough for play. |
| Graphics (NV2A) | Pushbuffer walker, Kelvin state machine, vertex-program and register-combiner translators (including the fixed-function pipeline and the texture-shader modes), texture decoder, 2D engine, video overlay, colour table. A Vulkan renderer with render targets kept on the GPU, on its own FIFO thread by default, with a persistent shader and pipeline cache. |
| Audio | APU voice processor (PCM, ADPCM, streams, envelopes, filters, multipass voices), two DSP56300 cores (a compiled execution path) on their own thread, AC'97 output. |
| Input | Two OHCI USB controllers, the front-panel hub and the gamepad (four ports, rumble) from any SDL3 gamepad, the keyboard or a libretro RetroPad. |
| Storage | ATA hard disk with the security lock, a blank retail-layout drive created on demand, ATAPI DVD drive reading XISO and full-dump images on a reader thread. |
| Network | Controller with no cable plugged in; no frames are sent or received. |
| Shells | `midway-verdant` (SDL3 window and headless runner), `midway_verdant_libretro` (core for libretro frontends), and the test tools `mv-pbcap`, `mv-retro-host`, `mv-dsp-bench` and `mv-fatx`. |

Not yet: the cycle-counted core at play speed, a Windows hypervisor backend, save states, memory units and the headset, and network traffic. Compatibility is partial; the wiki's status page says what is open.

## Clean room

Midway Verdant is written without leaked or proprietary source material of any kind. Every hardware block is derived from public documentation and datasheets, community reverse-engineering (xboxdevwiki and published papers), public register listings, and observation of the console's own firmware running on the emulated machine. Open-source emulators are read for behaviour only; no code is copied. The wiki's [Clean-Room Methodology](https://github.com/codingncaffeine/Midway-Verdant/wiki/Clean-Room-Methodology) page explains the discipline, and the [Clean-Room Receipts](https://github.com/codingncaffeine/Midway-Verdant/wiki/Clean-Room-Receipts) ledger lists each load-bearing hardware fact with its source and how it was confirmed. A pre-commit scan refuses console dumps, game data and known proprietary hashes, so nothing of the kind can reach the repository.

## What you need

Midway Verdant contains no Microsoft code or data. You supply dumps from your own console:

- the 512-byte MCPX boot ROM
- a flash BIOS image (256 KiB, 512 KiB or 1 MiB)
- a hard-disk image (raw; a qcow2 image can be converted)
- an EEPROM image (one can be generated from the running kernel's own key)

Game discs are read from XISO or full-dump (Redump-layout) images. For the libretro core, the files go in a `midway-verdant` folder inside the frontend's system directory; a blank hard disk is created there when none exists.

## Building

Linux first: a C++23 compiler (GCC 14+ or Clang 18+), CMake 3.28+, Ninja, SDL3 (from the system or fetched), a Vulkan 1.1 driver, glslang (fetched when the system has none). KVM (`/dev/kvm`) runs the fast mode. Windows builds with MSVC as a secondary platform; it has no hypervisor backend yet, so only the interpreter runs there.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Running, with your own dumps:

```
build/src/midway-verdant --mcpx mcpx.bin --bios flash.bin --hdd disk.img --dvd game.iso --gpu --window
```

`--help` lists every option. `--create-hdd FILE` writes a blank retail-layout drive. The build also produces `midway_verdant_libretro` for libretro frontends.

## License

GPL-3.0. See `LICENSE`.
