# Midway Verdant

A low-level emulator of the original Xbox for Linux (Windows to follow). The console's own boot ROM, BIOS and kernel run unmodified on an emulated machine: MCPX southbridge, NV2A graphics, APU with its DSPs, IDE storage, USB controllers and the SMBus devices.

**Status:** planning and early development. Nothing runs yet.

## What you need

Midway Verdant contains no Microsoft code or data. You supply dumps from your own console:

- the 512-byte MCPX boot ROM
- a flash BIOS image (256 KiB, 512 KiB or 1 MiB)
- a hard-disk image (raw; a qcow2 image can be converted)
- an EEPROM image (one can be generated)

Game discs are read from XISO or Redump-format images.

## Building

Linux first: a C++23 compiler (GCC 14+ or Clang 18+), CMake 3.28+, Ninja, SDL3, a Vulkan 1.1 driver. KVM (`/dev/kvm`) runs the guest CPU.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## License

GPL-3.0. See `LICENSE`.
