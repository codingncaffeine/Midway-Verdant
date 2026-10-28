#!/usr/bin/env bash
# Builds Midway Verdant FROM SOURCE on other distributions, each in a container with its own compiler, C library and
# packages, with the default options (warnings as errors, the tests on), then runs the unit tests there. The other
# half of tools/test-linux-distros.sh (which runs the release core): this one finds what breaks a user's own build
# (musl's declarations, a newer GCC's warnings, Clang, a distribution's glslang or SDL3).
#   tools/build-on-distros.sh [DISTRO...]     (default: every one below; PARALLEL=N at a time, default 3)
# Results: scratch/release/source-builds/DISTRO/{log,build/}, a line per distribution.
set -uo pipefail
cd "$(dirname "$0")/.."
engine=$(command -v podman || command -v docker) || { echo "needs podman or docker"; exit 1; }

# Each setup line is the README's install line for that distribution (keep them the same).
declare -A image setup env cmakeargs
image[alpine]=alpine:latest              # musl
setup[alpine]="apk add -q build-base cmake samurai git linux-headers glslang-dev sdl3-dev"
image[debian-13]=debian:13
setup[debian-13]="apt-get update -qq && apt-get install -y -qq build-essential cmake ninja-build git pkg-config libsdl3-dev glslang-dev glslang-tools"
image[ubuntu-24.04]=ubuntu:24.04          # GCC 13 by default: the build needs 14; SDL3 is not packaged: the core alone
setup[ubuntu-24.04]="apt-get update -qq && apt-get install -y -qq g++-14 cmake ninja-build git pkg-config"
cmakeargs[ubuntu-24.04]="-DMV_BUILD_SHELL=OFF"
env[ubuntu-24.04]="CC=gcc-14 CXX=g++-14"
image[fedora]=fedora:latest               # the newest GCC
setup[fedora]="dnf install -y -q gcc-c++ cmake ninja-build git SDL3-devel glslang-devel"
image[opensuse]=opensuse/tumbleweed
setup[opensuse]="zypper -q install -y gcc-c++ cmake ninja git SDL3-devel glslang-devel"
image[arch]=archlinux:latest
setup[arch]="pacman -Sy --noconfirm --needed base-devel cmake ninja git sdl3 glslang >/dev/null"
image[arch-clang]=archlinux:latest        # Clang with GCC's standard library
setup[arch-clang]="pacman -Sy --noconfirm --needed base-devel clang cmake ninja git sdl3 glslang >/dev/null"
env[arch-clang]="CC=clang CXX=clang++"
order=(alpine debian-13 ubuntu-24.04 fedora opensuse arch arch-clang)
[ $# -gt 0 ] && order=("$@")

jobs_each=$(( $(nproc) / ${PARALLEL:-3} )); [ "$jobs_each" -lt 2 ] && jobs_each=2
mkdir -p scratch/release/source-builds
build_one() {
    local d=$1 out=scratch/release/source-builds/$1
    rm -rf "$out"; mkdir -p "$out"
    "$engine" run --rm -e DEBIAN_FRONTEND=noninteractive -v "$PWD:/src:ro" -v "$PWD/$out:/out" -w /src "${image[$d]}" sh -c "
        ${setup[$d]} || { echo 'SETUP FAILED'; exit 3; }
        grep -h PRETTY_NAME /etc/os-release; export ${env[$d]:-NOTHING=0}; \${CXX:-c++} --version | head -1
        cmake -S /src -B /out/build -G Ninja -DCMAKE_BUILD_TYPE=Release ${cmakeargs[$d]:-} || exit 4
        cmake --build /out/build -j $jobs_each || exit 5
        timeout 600 /out/build/tests/mv_tests --gtest_brief=1 2>&1 | tail -3" > "$out/log" 2>&1
    local rc=$? what
    case $rc in 0) what=ok ;; 3) what="packages did not install" ;; 4) what="configure FAILED" ;; 5) what="build FAILED" ;; *) what="exit $rc" ;; esac
    local os cxx tests first
    os=$(grep -m1 PRETTY_NAME "$out/log" | cut -d'"' -f2)
    cxx=$(grep -m1 -E '^(g\+\+|c\+\+|clang|.*(GCC|clang version))' "$out/log" | cut -c1-40)
    tests=$(grep -m1 -oE '\[  PASSED  \] [0-9]+ tests|[0-9]+ FAILED TESTS?' "$out/log")
    first=$(grep -m1 -E 'error:' "$out/log" | sed 's|^/src/||' | cut -c1-150)
    printf '%-13s %-32s %-40s %s | %s%s\n' "$d" "${os:-?}" "${cxx:-?}" "$what" "${tests:-no tests run}" "${first:+ | $first}"
    rm -rf "$out/build/_deps"   # the fetched sources: hundreds of MB a distribution
}
for d in "${order[@]}"; do
    build_one "$d" &
    while [ "$(jobs -r | wc -l)" -ge "${PARALLEL:-3}" ]; do wait -n; done
done
wait
