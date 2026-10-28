#!/usr/bin/env bash
# Builds Midway Verdant FROM SOURCE on other distributions, each in a container with its own compiler, C library and
# packages, with the default options (warnings as errors, the tests on), then runs the unit tests there. The other
# half of tools/test-linux-distros.sh (which runs the release core): this one finds what breaks a user's own build
# (musl's declarations, a newer GCC's warnings, Clang, a distribution's SDL3).
#   [FW=FIRMWARE_DIR GAME=DISC.iso] tools/build-on-distros.sh [DISTRO...]   (default: every one below; PARALLEL=N, default 3)
# Results: scratch/release/source-builds/DISTRO/{log,build/}, a line per distribution.
set -uo pipefail
cd "$(dirname "$0")/.."
engine=$(command -v podman || command -v docker) || { echo "needs podman or docker"; exit 1; }

# Each setup line is the README's install line for that distribution (keep them the same).
declare -A image setup env cmakeargs
image[alpine]=alpine:latest              # musl
setup[alpine]="apk add -q build-base cmake samurai git linux-headers sdl3-dev"
image[debian-13]=debian:13
setup[debian-13]="apt-get update -qq && apt-get install -y -qq build-essential cmake ninja-build git pkg-config libsdl3-dev"
image[ubuntu-24.04]=ubuntu:24.04          # GCC 13 by default: the build needs 14; SDL3 is not packaged: the core alone
setup[ubuntu-24.04]="apt-get update -qq && apt-get install -y -qq g++-14 cmake ninja-build git pkg-config"
cmakeargs[ubuntu-24.04]="-DMV_BUILD_SHELL=OFF"
env[ubuntu-24.04]="CC=gcc-14 CXX=g++-14"
image[fedora]=fedora:latest               # the newest GCC
setup[fedora]="dnf install -y -q gcc-c++ cmake ninja-build git SDL3-devel"
image[opensuse]=opensuse/tumbleweed
setup[opensuse]="zypper -q install -y gcc-c++ cmake ninja git SDL3-devel"
image[arch]=archlinux:latest
setup[arch]="pacman -Sy --noconfirm --needed base-devel cmake ninja git sdl3 >/dev/null"
image[arch-clang]=archlinux:latest        # Clang with GCC's standard library
setup[arch-clang]="pacman -Sy --noconfirm --needed base-devel clang cmake ninja git sdl3 >/dev/null"
env[arch-clang]="CC=clang CXX=clang++"
order=(alpine debian-13 ubuntu-24.04 fedora opensuse arch arch-clang)
[ $# -gt 0 ] && order=("$@")

# Then, with FW (a folder with the MCPX boot ROM and the BIOS, found by size) and GAME (a disc image) set, the core it
# built boots that game under KVM from a first-start folder, through the distribution's own Vulkan loader and software
# driver: it passes when pictures are shown. These are the packages a player needs too (not part of the build lines).
declare -A runtime
runtime[alpine]="apk add -q vulkan-loader mesa-vulkan-swrast"
runtime[debian-13]="apt-get install -y -qq libvulkan1 mesa-vulkan-drivers"
runtime[ubuntu-24.04]="apt-get install -y -qq libvulkan1 mesa-vulkan-drivers"
runtime[fedora]="dnf install -y -q vulkan-loader mesa-vulkan-drivers"
runtime[opensuse]="zypper -q install -y libvulkan1 libvulkan_lvp"
runtime[arch]="pacman -S --noconfirm --needed vulkan-icd-loader vulkan-swrast >/dev/null"
runtime[arch-clang]="pacman -S --noconfirm --needed vulkan-icd-loader vulkan-swrast >/dev/null"
fw=${FW:-} game=${GAME:-}
boot=0; [ -n "$fw" ] && [ -n "$game" ] && [ -e /dev/kvm ] && boot=1

jobs_each=$(( $(nproc) / ${PARALLEL:-3} )); [ "$jobs_each" -lt 2 ] && jobs_each=2
mkdir -p scratch/release/source-builds
build_one() {
    local d=$1 out=scratch/release/source-builds/$1
    rm -rf "$out"; mkdir -p "$out/system/midway-verdant" "$out/png"
    local mounts=(-v "$PWD:/src:ro" -v "$PWD/$out:/out") bootcmd="echo BOOT SKIPPED"
    if [ "$boot" = 1 ]; then
        find -L "$fw" -maxdepth 1 -type f \( -size 512c -o -size 262144c -o -size 524288c -o -size 1048576c \) -exec cp {} "$out/system/midway-verdant/" \;
        mounts+=(--device /dev/kvm -v "$(realpath "$game"):/game.iso:ro")
        bootcmd="${runtime[$d]} && echo BOOT START && timeout 300 /out/build/src/mv-retro-host /out/build/src/midway_verdant_libretro.so /out/system /game.iso --frames 300 --png-every 300 /out/png; echo \"BOOT EXIT \$?\""
    fi
    "$engine" run --rm -e DEBIAN_FRONTEND=noninteractive "${mounts[@]}" -w /src "${image[$d]}" sh -c "
        ${setup[$d]} || { echo 'SETUP FAILED'; exit 3; }
        grep -h PRETTY_NAME /etc/os-release; export ${env[$d]:-NOTHING=0}; \${CXX:-c++} --version | head -1
        cmake -S /src -B /out/build -G Ninja -DCMAKE_BUILD_TYPE=Release ${cmakeargs[$d]:-} || exit 4
        cmake --build /out/build -j $jobs_each || exit 5
        timeout 600 /out/build/tests/mv_tests --gtest_brief=1 2>&1 | tail -3
        $bootcmd" > "$out/log" 2>&1
    local rc=$? what
    case $rc in 0) what=ok ;; 3) what="packages did not install" ;; 4) what="configure FAILED" ;; 5) what="build FAILED" ;; *) what="exit $rc" ;; esac
    grep -qE 'FAILED TESTS?' "$out/log" && what="TESTS FAILED"
    local os cxx tests first shown gpu bootres
    os=$(grep -m1 PRETTY_NAME "$out/log" | cut -d'"' -f2)
    cxx=$(grep -m1 -E '^(g\+\+|c\+\+|clang|.*(GCC|clang version))' "$out/log" | cut -c1-40)
    tests=$(grep -m1 -oE '\[  PASSED  \] [0-9]+ tests|[0-9]+ FAILED TESTS?' "$out/log")
    first=$(grep -m1 -E '\.(c|cc|cpp|h|hpp):[0-9]+:[0-9]+: (fatal )?error:|CMake Error' "$out/log" | sed 's|^/src/||' | cut -c1-150)  # the compiler's or CMake's, not a package manager's
    shown=$(grep -m1 -oE '[0-9]+ displayed' "$out/log" | cut -d' ' -f1)
    gpu=$(grep -m1 -oE '\] gpu: .*' "$out/log" | cut -c8- | cut -c1-24)
    if [ "$boot" != 1 ]; then bootres="boot skipped (set FW and GAME)"
    elif [ "${shown:-0}" -gt 0 ]; then bootres="boots: $shown pictures ($gpu)"
    else bootres="BOOT FAILED ($(grep -m1 -oE 'BOOT EXIT [0-9]+' "$out/log" || echo no run))"; [ "$what" = ok ] && what="BOOT FAILED"; fi
    printf '%-13s %-32s %-34s %s | %s | %s%s\n' "$d" "${os:-?}" "${cxx:-?}" "$what" "${tests:-no tests run}" "$bootres" "${first:+ | $first}"
    rm -rf "$out/build/_deps" "$out/system/midway-verdant/xbox_hdd.raw"   # the fetched sources; the blank disk the boot made
}
for d in "${order[@]}"; do
    build_one "$d" &
    while [ "$(jobs -r | wc -l)" -ge "${PARALLEL:-3}" ]; do wait -n; done
done
wait
