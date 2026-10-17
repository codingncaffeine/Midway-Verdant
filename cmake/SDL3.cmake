# SDL3 for the shell: the system package when it has a CMake config, else a pinned release built here.
find_package(SDL3 CONFIG QUIET)
if(SDL3_FOUND)
  message(STATUS "SDL3 ${SDL3_VERSION} from the system")
else()
  message(STATUS "SDL3 not installed; fetching release-3.4.18")
  include(FetchContent)
  set(SDL_SHARED OFF CACHE BOOL "" FORCE)
  set(SDL_STATIC ON CACHE BOOL "" FORCE)
  set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
  set(SDL_TESTS OFF CACHE BOOL "" FORCE)
  set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.4.18
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(SDL3)
endif()
