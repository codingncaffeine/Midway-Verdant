# glslang compiles the GLSL our translators emit into SPIR-V at run time. A pinned copy is fetched and built in statically,
# and the libretro core hides its symbols (src/CMakeLists.txt), so its calls can never land in another glslang: a
# frontend may carry its own and export it (Alpine's RetroArch exports 79 glslang symbols), and a core linked to the
# shared system glslang then had its calls resolved to the frontend's copy, a different version, and crashed in its
# first shader compile. MV_SYSTEM_GLSLANG=ON uses the system package instead, for packagers who must; the core is then
# exposed to exactly that.
if(MV_SYSTEM_GLSLANG AND NOT MV_PORTABLE)
  find_package(glslang CONFIG QUIET)
  if(glslang_FOUND)
    message(WARNING "MV_SYSTEM_GLSLANG: the core links the shared system glslang; in a frontend that carries its own (Alpine's RetroArch) its shader compiles run in the frontend's copy and crash")
  endif()
endif()
if(glslang_FOUND AND TARGET glslang::glslang AND TARGET glslang::SPIRV)
  message(STATUS "glslang: system ${glslang_VERSION}")
  set(MV_GLSLANG_BUILT_IN OFF)
else()
  message(STATUS "glslang: fetching a pinned copy")
  set(MV_GLSLANG_BUILT_IN ON)
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
