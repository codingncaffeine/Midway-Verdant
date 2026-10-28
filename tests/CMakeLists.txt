include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  GIT_SHALLOW TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(mv_tests
  unit/version_test.cpp
  unit/thread_stack_test.cpp
  unit/clock_test.cpp
  unit/guest_clock_test.cpp
  unit/bus_test.cpp
  unit/legacy_test.cpp
  unit/nvnet_test.cpp
  unit/nv2a_core_test.cpp
  unit/gpu_timing_test.cpp
  unit/twod_test.cpp
  unit/smbus_test.cpp
  unit/crypto_test.cpp
  unit/eeprom_test.cpp
  unit/pushbuffer_test.cpp
  unit/capture_test.cpp
  unit/kelvin_test.cpp
  unit/vsh_test.cpp
  unit/psh_test.cpp
  unit/texture_test.cpp
  unit/overlay_test.cpp
  unit/retropad_test.cpp
  unit/usb_test.cpp
  unit/vulkan_test.cpp
  unit/vsh_interpreter_test.cpp
  unit/dsp_decode_test.cpp
  unit/dsp_core_test.cpp
  unit/dsp_dma_test.cpp
  unit/apu_test.cpp
  unit/aci_test.cpp
  unit/ide_test.cpp
  unit/p6_test.cpp
  unit/p6_block_test.cpp
)
target_link_libraries(mv_tests PRIVATE mvcore gtest_main)
target_compile_definitions(mv_tests PRIVATE MV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
include(GoogleTest)
gtest_discover_tests(mv_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 120)

# The core neither needs nor offers glslang by name (its own is built in and hidden), and compiles against that copy's
# own headers even with a glslang package installed: a frontend exporting its own glslang took over a dynamically linked
# core's shader compiles and crashed it (Alpine's RetroArch). Both hold for the built-in copy only (MV_SYSTEM_GLSLANG=ON
# gives them up).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND MV_GLSLANG_BUILT_IN)
  add_test(NAME core_symbols COMMAND ${CMAKE_SOURCE_DIR}/tests/tools/core-symbols.sh $<TARGET_FILE:midway_verdant_libretro>)
  add_test(NAME glslang_headers COMMAND ${CMAKE_SOURCE_DIR}/tests/tools/glslang-headers.sh ${CMAKE_BINARY_DIR}/compile_commands.json ${glslang_SOURCE_DIR})
endif()

# The emulator answers --version; a build that cannot is not a build.
if(MV_BUILD_SHELL)
  add_test(NAME cli_version COMMAND midway-verdant --version)
  set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "^Midway Verdant [0-9]+\\.[0-9]+\\.[0-9]+")
endif()

# The boot smoke test runs only where the user's console dumps are (local-notes/user-files.env); elsewhere it skips.
# It is a bash script around the KVM fast mode, so it exists on Linux only until the WHPX fast mode lands (Phase 8), and
# only where bash is (Alpine has none by default).
find_program(MV_BASH bash)
if(NOT WIN32 AND MV_BASH)
  add_test(NAME boot_smoke COMMAND "${MV_BASH}" "${CMAKE_SOURCE_DIR}/tests/boot/boot-smoke.sh" "${CMAKE_BINARY_DIR}")
  set_tests_properties(boot_smoke PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
