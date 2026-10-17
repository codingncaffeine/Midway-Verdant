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
  debug/trace.cpp
  flash/flash.cpp
  ide/ide.cpp
  ide/ata_disk.cpp
  ide/disk_image.cpp
  mcpx/isa_misc.cpp
  mcpx/lpc.cpp
  mcpx/pic.cpp
  mcpx/pit.cpp
  mcpx/rtc.cpp
  nv2a/capture.cpp
  nv2a/nv2a_stub.cpp
  nv2a/pgraph/kelvin.cpp
  nv2a/pgraph/replay.cpp
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
  set_source_files_properties(cpu/kvm/kvm_backend.cpp PROPERTIES COMPILE_OPTIONS "-Wno-pedantic")
endif()
find_package(Threads REQUIRED)
target_link_libraries(mvcore PUBLIC Threads::Threads)

# midway-verdant: the emulator. Headless until the window shell lands.
add_executable(midway-verdant ui/main.cpp)
target_link_libraries(midway-verdant PRIVATE mvcore)

# mv-pbcap: the pushbuffer capture reader (summaries and dumps of what --capture wrote).
add_executable(mv-pbcap tools/mv-pbcap.cpp)
target_link_libraries(mv-pbcap PRIVATE mvcore)
