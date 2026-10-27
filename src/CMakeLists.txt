# mvcore: the machine, its devices and the CPU backends; everything the front ends and tools link.
# It contains no window, input or audio code: those belong to the clients (the SDL3 shell, the libretro core).
add_library(mvcore STATIC
  core/version.cpp
  core/clock.cpp
  core/guest_clock.cpp
  core/scheduler.cpp
  core/bus.cpp
  core/machine.cpp
  core/kernel_image.cpp
  crypto/sha1.cpp
  cpu/kvm/kvm_backend.cpp
  cpu/p6/decode.cpp
  cpu/p6/memory.cpp
  cpu/p6/cpu.cpp
  cpu/p6/system.cpp
  cpu/p6/fpu.cpp
  cpu/p6/simd.cpp
  cpu/p6/timing.cpp
  cpu/p6/block.cpp
  cpu/p6/jit/jit_x64.cpp
  cpu/p6/p6_backend.cpp
  cpu/p6/oracle.cpp
  debug/trace.cpp
  debug/guest_stack.cpp
  flash/flash.cpp
  ide/ide.cpp
  ide/ata_disk.cpp
  ide/disk_image.cpp
  ide/dvd_image.cpp
  ide/hdd_format.cpp
  usb/ohci.cpp
  usb/usb_device.cpp
  usb/xbox_pad.cpp
  libretro/retropad_map.cpp
  mcpx/isa_misc.cpp
  mcpx/lpc.cpp
  mcpx/pic.cpp
  mcpx/pit.cpp
  mcpx/rtc.cpp
  nv2a/capture.cpp
  nv2a/nv2a_stub.cpp
  nv2a/render_thread.cpp
  nv2a/gpu_timing.cpp
  nv2a/overlay.cpp
  nv2a/scanout.cpp
  nv2a/pgraph/kelvin.cpp
  nv2a/pgraph/replay.cpp
  nv2a/pgraph/texture.cpp
  nv2a/pgraph/twod.cpp
  nv2a/shaders/vsh_translate.cpp
  nv2a/shaders/psh_translate.cpp
  nv2a/shaders/vsh_interpreter.cpp
  apu/dsp/decode.cpp
  apu/dsp/core.cpp
  apu/dsp/dma.cpp
  apu/processor.cpp
  apu/apu.cpp
  apu/setup_engine.cpp
  aci/aci.cpp
  mcpx/nvnet.cpp
  nv2a/shaders/glsl_compiler.cpp
  nv2a/vulkan/vk_loader.cpp
  nv2a/vulkan/vk_device.cpp
  nv2a/vulkan/vk_renderer.cpp
  debug/png.cpp
  nv2a/pushbuffer.cpp
  pci/pci.cpp
  smbus/smbus.cpp
  smbus/smc.cpp
  smbus/eeprom_image.cpp
)
target_include_directories(mvcore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/generated")
target_link_libraries(mvcore PUBLIC mv_warnings)
# The Linux KVM header is not ISO C++ (flexible array members); that one file is built without -Wpedantic.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_source_files_properties(cpu/kvm/kvm_backend.cpp cpu/p6/oracle.cpp PROPERTIES COMPILE_OPTIONS "-Wno-pedantic")
endif()
find_package(Threads REQUIRED)
target_link_libraries(mvcore PUBLIC Threads::Threads)
# The shader compiler is an implementation detail of the renderer; its headers stay out of our public includes.
target_link_libraries(mvcore PRIVATE glslang::glslang glslang::SPIRV glslang::glslang-default-resource-limits)
target_link_libraries(mvcore PUBLIC Vulkan::Headers)
target_compile_definitions(mvcore PUBLIC VK_NO_PROTOTYPES)
if(NOT WIN32)
  target_link_libraries(mvcore PUBLIC ${CMAKE_DL_LIBS})
  # timer_create and friends were in librt until glibc 2.34 (an empty stub from then on).
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(mvcore PUBLIC rt)
  endif()
  # Vulkan structures are filled sType-first with the rest zeroed: the idiom the API documents.
  set_source_files_properties(nv2a/vulkan/vk_device.cpp nv2a/vulkan/vk_renderer.cpp tools/mv-retro-host.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-old-style-cast")
endif()
# The x87 and SSE units run host floating point under the guest's rounding mode and read the exception flags back:
# MSVC may move or fold float operations across those control-word changes unless the environment is honoured.
if(MSVC)
  set_source_files_properties(cpu/p6/fpu.cpp cpu/p6/simd.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
endif()
# glslang's headers are not warning-clean under our flags; that one file is built with them as system headers.
if(NOT MSVC)
  set_source_files_properties(nv2a/shaders/glsl_compiler.cpp PROPERTIES COMPILE_OPTIONS "-Wno-shadow;-Wno-conversion")
endif()

# midway-verdant: the emulator shell — the SDL3 window around the core (the core itself never sees SDL).
if(MV_BUILD_SHELL)
include(${CMAKE_SOURCE_DIR}/cmake/SDL3.cmake)
add_executable(midway-verdant ui/main.cpp ui/window.cpp ui/pad_input.cpp)
target_link_libraries(midway-verdant PRIVATE mvcore SDL3::SDL3)
endif()

# mv-pbcap: the pushbuffer capture reader (summaries and dumps of what --capture wrote).
add_executable(mv-pbcap tools/mv-pbcap.cpp)
target_link_libraries(mv-pbcap PRIVATE mvcore)
# mv-dsp-bench: times (and cross-checks) the DSP core's two execution paths on a snapshot of a game's GP.
add_executable(mv-dsp-bench tools/mv-dsp-bench.cpp)
target_link_libraries(mv-dsp-bench PRIVATE mvcore)

# mv-fatx: reads files from a hard-disk image's FATX partitions (test results, saves).
add_executable(mv-fatx tools/mv-fatx.cpp)
target_link_libraries(mv-fatx PRIVATE mvcore)

# midway_verdant_libretro: the core for libretro frontends (Emutastic, RetroArch). No SDL: the frontend owns the
# window, the audio device and the controllers.
add_library(midway_verdant_libretro SHARED libretro/libretro_core.cpp)
target_link_libraries(midway_verdant_libretro PRIVATE mvcore)
set_target_properties(midway_verdant_libretro PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden)
if(MV_PORTABLE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # The C++ runtime inside the core, and only retro_* exported: the static libraries' symbols stay the core's own.
  target_link_options(midway_verdant_libretro PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL -Wl,--no-undefined)
endif()

# mv-retro-host: a headless libretro frontend that drives the core frame by frame for tests (POSIX dlopen).
if(NOT WIN32)
  add_executable(mv-retro-host tools/mv-retro-host.cpp)
  target_link_libraries(mv-retro-host PRIVATE mvcore ${CMAKE_DL_LIBS})
  if(MV_PORTABLE)
    target_link_options(mv-retro-host PRIVATE -static-libstdc++ -static-libgcc)  # runs the release core on other distributions
  endif()
endif()
