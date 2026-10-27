cmake_minimum_required(VERSION 3.28)
project(MidwayVerdant LANGUAGES CXX VERSION 0.0.1)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MV_WERROR "Treat warnings as errors" ON)
option(MV_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(MV_BUILD_TESTS "Build the test suite" ON)
# Release cores: glslang built in (no distribution's copy), the C++ runtime linked in and every symbol but the libretro
# API hidden, so one core loads on any distribution from its build baseline on and never meets a frontend's own copies.
option(MV_PORTABLE "Build a self-contained core for release" OFF)
option(MV_BUILD_SHELL "Build the SDL3 shell (midway-verdant); a release core needs none of it" ON)

include(cmake/Warnings.cmake)
include(cmake/Shaders.cmake)
include(cmake/Vulkan.cmake)

if(MV_SANITIZE)
  if(MSVC)
    add_compile_options(/fsanitize=address)
  else()
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

configure_file(src/core/version.h.in "${CMAKE_BINARY_DIR}/generated/mv/version.h" @ONLY)

add_subdirectory(src)
if(MV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
