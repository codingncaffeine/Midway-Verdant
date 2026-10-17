# mvcore: the machine, its devices and the CPU backends; everything the front ends and tools link.
# It contains no window, input or audio code: those belong to the clients (the SDL3 shell, the libretro core).
add_library(mvcore STATIC
  core/version.cpp
  core/clock.cpp
  core/scheduler.cpp
  core/bus.cpp
  core/machine.cpp
  cpu/kvm/kvm_backend.cpp
  debug/trace.cpp
  flash/flash.cpp
  mcpx/isa_misc.cpp
  mcpx/lpc.cpp
  mcpx/pic.cpp
  mcpx/pit.cpp
  mcpx/rtc.cpp
  nv2a/nv2a_stub.cpp
  pci/pci.cpp
  smbus/smbus.cpp
  smbus/smc.cpp
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
