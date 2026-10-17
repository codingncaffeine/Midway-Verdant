# One warning set for every target. A warning left to fester is a defect that has not been triggered yet,
# so warnings are errors from the first commit (MV_WERROR, on by default).
add_library(mv_warnings INTERFACE)
if(MSVC)
  target_compile_options(mv_warnings INTERFACE /W4 /permissive- /utf-8 /EHsc /Zc:preprocessor)
  if(MV_WERROR)
    target_compile_options(mv_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(mv_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wdouble-promotion
    -Wconversion -Wsign-conversion -Wimplicit-fallthrough -Wcast-align -Wformat=2 -ffp-contract=off)
  if(MV_WERROR)
    target_compile_options(mv_warnings INTERFACE -Werror)
  endif()
endif()
