# glslang compiles the GLSL our translators emit into SPIR-V at run time. The system package is used where it
# provides CMake targets (Arch, recent Debian/Ubuntu); otherwise, and always for a portable core (MV_PORTABLE), a pinned
# copy is fetched and built in statically.
if(NOT MV_PORTABLE)
  find_package(glslang CONFIG QUIET)
endif()
if(glslang_FOUND AND TARGET glslang::glslang AND TARGET glslang::SPIRV)
  message(STATUS "glslang: system ${glslang_VERSION}")
else()
  message(STATUS "glslang: fetching a pinned copy")
  include(FetchContent)
  set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
  set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
  set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
  set(ENABLE_SPIRV_TOOLS OFF CACHE BOOL "" FORCE)
  set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
  set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG 15.4.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(glslang)
  if(NOT TARGET glslang::glslang)
    add_library(glslang::glslang ALIAS glslang)
  endif()
  if(NOT TARGET glslang::SPIRV)
    add_library(glslang::SPIRV ALIAS SPIRV)
  endif()
  if(NOT TARGET glslang::glslang-default-resource-limits)
    add_library(glslang::glslang-default-resource-limits ALIAS glslang-default-resource-limits)
  endif()
endif()
