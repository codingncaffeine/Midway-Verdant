#!/usr/bin/env bash
# The Linux release core: built in the manylinux_2_28 container (AlmaLinux 8: glibc 2.28, GCC 14), self-contained
# (MV_PORTABLE: glslang and the C++ runtime inside, only retro_* exported), so it loads on distributions from 2019 on.
# Run on the host (podman or docker) or inside the container itself (CI):
#   tools/build-linux-core.sh [OUT_DIR]     (default scratch/release/linux)
# OUT_DIR receives midway_verdant_libretro.so and a line of what it needs from the system.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-scratch/release/linux}
image=quay.io/pypa/manylinux_2_28_x86_64

if [ ! -f /etc/almalinux-release ] && [ -z "${MV_IN_CONTAINER:-}" ]; then
    engine=$(command -v podman || command -v docker) || { echo "needs podman or docker"; exit 1; }
    mkdir -p "$out"
    exec "$engine" run --rm -e MV_IN_CONTAINER=1 -v "$PWD:/src:Z" -w /src "$image" tools/build-linux-core.sh "$out"
fi

build=scratch/release/build-linux
cmake -S . -B "$build" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DMV_PORTABLE=ON -DMV_BUILD_SHELL=OFF -DMV_BUILD_TESTS=OFF -DMV_WERROR=ON
cmake --build "$build" --target midway_verdant_libretro mv-retro-host -j "$(nproc)"
mkdir -p "$out"
cp "$build/src/midway_verdant_libretro.so" "$out/"
mkdir -p "$out/../test-host" && cp "$build/src/mv-retro-host" "$out/../test-host/"  # for tools/test-linux-distros.sh, not shipped
strip --strip-unneeded "$out/midway_verdant_libretro.so"

# What it asks of the system (shared libraries, the newest glibc symbol version) and what it offers besides the API: the
# C++ library's template code it instantiates (weak W/V and unique u symbols, which the library's headers give default
# visibility; the same in every module that has them), never glslang (core-symbols.sh fails the build if it does).
so="$out/midway_verdant_libretro.so"
needed=$(objdump -p "$so" | awk '/NEEDED/ { printf "%s ", $2 }')
glibc=$(objdump -T "$so" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
others=$(nm -D --defined-only "$so" | awk '$3 !~ /^retro_/ { n[$2]++ } END { s = ""; for (k in n) { printf "%s%d %s", s, n[k], k; s = ", " } }')
echo "core: $(du -h "$so" | cut -f1); needs: $needed; newest symbol: $glibc; exported besides retro_*: ${others:-nothing}"
tests/tools/core-symbols.sh "$so"
[ "$exported" = 0 ] || { echo "the core exports symbols besides the libretro API"; exit 1; }
