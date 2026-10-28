#!/bin/sh
# The libretro core neither needs nor offers glslang or SPIRV symbols by name: its glslang is its own, built in and
# hidden. A core that needed them (linked to the shared system glslang) had its calls taken over by a frontend that
# exports its own glslang (Alpine's RetroArch) and crashed in its first shader compile. Usage: core-symbols.sh CORE.so
set -u
so=${1:?core}  # POSIX sh: Alpine has no bash by default
command -v nm >/dev/null || { echo "SKIP: no nm"; exit 0; }
needed=$(nm -D --undefined-only "$so" | grep -cE 'glslang|spv')
offered=$(nm -D --defined-only "$so" | grep -cE 'glslang|spv')
libs=$(objdump -p "$so" | awk '/NEEDED/ && /glslang|SPIRV/ {printf "%s ", $2}')
echo "glslang/SPIRV symbols needed: $needed, offered: $offered; shared glslang libraries: ${libs:-none}"
[ "$needed" -eq 0 ] && [ "$offered" -eq 0 ] && [ -z "$libs" ]
