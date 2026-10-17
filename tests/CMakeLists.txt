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
)
target_link_libraries(mv_tests PRIVATE mvcore gtest_main)
include(GoogleTest)
gtest_discover_tests(mv_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 120)

# The emulator answers --version; a build that cannot is not a build.
add_test(NAME cli_version COMMAND midway-verdant --version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "^Midway Verdant [0-9]+\\.[0-9]+\\.[0-9]+")
