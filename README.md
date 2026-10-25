<p align="center"><img src="assets/branding/banner.png" alt="Midway Verdant" width="768"></p>

# Midway Verdant

A clean-room, low-level emulator of the original Xbox for Linux, built to play games at full speed. The console's own boot ROM, BIOS and kernel run unmodified on an emulated machine: MCPX southbridge, NV2A graphics, the APU with its DSPs, IDE storage, USB controllers and the SMBus devices. Nothing in the firmware is patched or replaced with a high-level substitute. The CPU runs under hardware virtualisation (KVM), so the console's Pentium III code executes natively while every device around it is emulated in software.

It ships as a libretro core, `midway_verdant_libretro`, for libretro frontends; [Emutastic for Linux](https://github.com/codingncaffeine/Emutastic-For-Linux) offers it as an optional Xbox core. A standalone SDL3 window is included for development.

**Status:** many retail games boot and play, a good number at their full frame rate on a desktop CPU; others still stall, glitch or sound wrong. The wiki describes every hardware block and keeps the [status page](https://github.com/codingncaffeine/Midway-Verdant/wiki/Status) current.

| Area | State |
|---|---|
| Boot chain | Boot ROM, second-stage loader and kernel run unmodified; the kernel mounts its hard disk and starts a disc or the dashboard. Warm reboot works. |
| CPU | KVM on Linux: the guest's code runs on the host CPU at full speed, with the console's own clock kept from the work it does. A Pentium III interpreter and recompiler serve as the reference for testing. |
| Graphics (NV2A) | Pushbuffer walker, Kelvin state machine, vertex-program and register-combiner translators (including the fixed-function pipeline and the texture-shader modes), texture decoder, 2D engine, video overlay, colour table. A Vulkan renderer with render targets kept on the GPU at up to 6x the internal resolution, on its own thread, shaders compiled in the background, and a persistent shader and pipeline cache. |
| Audio | APU voice processor (PCM, ADPCM, streams, envelopes, filters, multipass voices), two DSP56300 cores running the console's own audio programs, AC'97 output; an optional direct mix that plays every voice straight to stereo. |
| Video | Full-motion video through the video overlay and through game-decoded movies. |
| Input | Two OHCI USB controllers, the front-panel hub and the gamepad (four ports, rumble) from any SDL3 gamepad, the keyboard or a libretro RetroPad. |
| Storage | ATA hard disk with the security lock, a blank retail-layout drive created on demand, ATAPI DVD drive reading XISO and full-dump images on a reader thread. |
| Network | Controller with no cable plugged in; no frames are sent or received. |
| Shells | `midway_verdant_libretro` (the core), `midway-verdant` (SDL3 window and headless runner), and the test tools `mv-pbcap`, `mv-retro-host`, `mv-dsp-bench` and `mv-fatx`. |

Not yet: save states, memory units and the headset, network traffic, and Windows and macOS builds of the core. Compatibility is partial; the wiki's status page says what is open.

## Core options

| Option | Values |
|---|---|
| CPU | fast (hardware virtualization) · accurate (the reference interpreter; not play speed) |
| Audio | console (the console's DSP effects, such as reverb) · direct mix (every voice straight to stereo, no DSP effects) |
| Internal resolution | 1x (native) · 2x · 4x · 6x (the fast CPU; render targets drawn at that many times their size, changed while a game runs) |
| Shader compiling | in the background (no stutter; a new effect can appear a frame late) · in the frame |
| Anisotropic filtering | off (as the game asks) · 2x · 4x · 8x · 16x |
| Disc loading | fast (8x the console's drive) · console speed |
| Run the next frame while the frontend shows this one | enabled (faster, up to a frame more input lag) · disabled |
| Controllers plugged in | 1 to 4 |
| Hard drive | Drive 1 · Drive 2 · Drive 3 |

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
