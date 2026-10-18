# mvcore: the machine, its devices and the CPU backends; everything the front ends and tools link.
# It contains no window, input or audio code: those belong to the clients (the SDL3 shell, the libretro core).
add_library(mvcore STATIC
  core/version.cpp
  core/clock.cpp
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
  cpu/p6/p6_backend.cpp
  cpu/p6/oracle.cpp
  debug/trace.cpp
  debug/guest_stack.cpp
  flash/flash.cpp
  ide/ide.cpp
  ide/ata_disk.cpp
  ide/disk_image.cpp
  ide/dvd_image.cpp
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
  nv2a/overlay.cpp
  nv2a/pgraph/kelvin.cpp
  nv2a/pgraph/replay.cpp
  nv2a/pgraph/texture.cpp
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
  # Vulkan structures are filled sType-first with the rest zeroed: the idiom the API documents.
  set_source_files_properties(nv2a/vulkan/vk_device.cpp nv2a/vulkan/vk_renderer.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-old-style-cast")
endif()
# glslang's headers are not warning-clean under our flags; that one file is built with them as system headers.
if(NOT MSVC)
  set_source_files_properties(nv2a/shaders/glsl_compiler.cpp PROPERTIES COMPILE_OPTIONS "-Wno-shadow;-Wno-conversion")
endif()

# midway-verdant: the emulator shell — the SDL3 window around the core (the core itself never sees SDL).
include(${CMAKE_SOURCE_DIR}/cmake/SDL3.cmake)
add_executable(midway-verdant ui/main.cpp ui/window.cpp ui/pad_input.cpp)
target_link_libraries(midway-verdant PRIVATE mvcore SDL3::SDL3)

# mv-pbcap: the pushbuffer capture reader (summaries and dumps of what --capture wrote).
add_executable(mv-pbcap tools/mv-pbcap.cpp)
target_link_libraries(mv-pbcap PRIVATE mvcore)

# midway_verdant_libretro: the core for libretro frontends (Emutastic, RetroArch). No SDL: the frontend owns the
# window, the audio device and the controllers.
add_library(midway_verdant_libretro SHARED libretro/libretro_core.cpp)
target_link_libraries(midway_verdant_libretro PRIVATE mvcore)
set_target_properties(midway_verdant_libretro PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(midway_verdant_libretro PRIVATE RETRO_IMPORT_SYMBOLS=0)

# mv-retro-host: a headless libretro frontend that drives the core frame by frame for tests (POSIX dlopen).
if(NOT WIN32)
  add_executable(mv-retro-host tools/mv-retro-host.cpp)
  target_link_libraries(mv-retro-host PRIVATE mvcore ${CMAKE_DL_LIBS})
endif()
