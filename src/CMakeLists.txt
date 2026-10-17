# mvcore: the machine, its devices and the CPU backends; everything the front ends and tools link.
add_library(mvcore STATIC
  core/version.cpp
)
target_include_directories(mvcore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/generated")
target_link_libraries(mvcore PUBLIC mv_warnings)

# midway-verdant: the emulator. Headless until the window shell lands.
add_executable(midway-verdant ui/main.cpp)
target_link_libraries(midway-verdant PRIVATE mvcore)
