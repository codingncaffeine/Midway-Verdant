#!/usr/bin/env bash
# The release core on other distributions: each in a container with /dev/kvm and its own Vulkan loader, a first start
# (only the MCPX ROM and the BIOS in the system folder: the EEPROM is generated and a blank hard disk made), and a game
# booted headless. The GPU: each distribution's software Vulkan (lavapipe, slow), or with GPU=nvidia the host's through
# the container toolkit's CDI device (see below).
# Built first by tools/build-linux-core.sh.
#   tools/test-linux-distros.sh FIRMWARE_DIR GAME.iso [DISTRO...]     (default: every distribution below)
# FIRMWARE_DIR holds the 512-byte MCPX ROM and the flash BIOS (found by size). Results in scratch/release/distros/.
set -uo pipefail
cd "$(dirname "$0")/.."
fw=$1 game=$2; shift 2
engine=$(command -v podman || command -v docker) || { echo "needs podman or docker"; exit 1; }
core=scratch/release/linux/midway_verdant_libretro.so host=scratch/release/test-host/mv-retro-host
[ -f "$core" ] && [ -f "$host" ] || { echo "build first: tools/build-linux-core.sh"; exit 1; }
# GPU=nvidia: the host GPU through the container toolkit's CDI device. OPEN (2026-10-27, driver 615.78.08, toolkit 1.20.1):
# the NVIDIA ICD loads in the containers but answers VK_ERROR_INCOMPATIBLE_DRIVER, even privileged; so software by default.
gpu=${GPU:-soft}
# The GPU reaches the main menu of most games in 2400 frames; software Vulkan boots and draws the start of the animation.
frames=${FRAMES:-$([ "$gpu" = nvidia ] && echo 2400 || echo 300)}
cap=${CAP:-420}  # seconds a container may run the core
gpu_args=(); [ "$gpu" = nvidia ] && gpu_args=(--device nvidia.com/gpu=all)

declare -A image loader soft
apt="apt-get update -qq && { apt-get install -y -qq PKGS || { apt-get update -qq && apt-get install -y -qq --fix-missing PKGS; }; }"
for d in debian-11 debian-12 debian-13 ubuntu-20.04 ubuntu-22.04 ubuntu-24.04; do
    # The NVIDIA driver's Vulkan library needs libX11 and libXext (every desktop has them); no recommended packages, or a
    # software driver comes along and serves the core in the GPU's place.
    image[$d]=${d/-/:}; loader[$d]=${apt//PKGS/--no-install-recommends libvulkan1 libx11-6 libxext6}; soft[$d]=${apt//PKGS/libvulkan1 mesa-vulkan-drivers}
done
image[fedora]=fedora:latest;         loader[fedora]="dnf install -y -q --setopt=install_weak_deps=False vulkan-loader libX11 libXext";   soft[fedora]="dnf install -y -q vulkan-loader mesa-vulkan-drivers"
image[arch]=archlinux:latest;        loader[arch]="pacman -Sy --noconfirm --needed vulkan-icd-loader libx11 libxext >/dev/null"
soft[arch]="pacman -Sy --noconfirm --needed vulkan-icd-loader vulkan-swrast >/dev/null"
image[opensuse]=opensuse/tumbleweed; loader[opensuse]="zypper -q install -y --no-recommends libvulkan1 libX11-6 libXext6"; soft[opensuse]="zypper -q install -y libvulkan1 libvulkan_lvp"
order=(debian-11 debian-12 debian-13 ubuntu-20.04 ubuntu-22.04 ubuntu-24.04 fedora arch opensuse)
[ $# -gt 0 ] && order=("$@")

mkdir -p scratch/release/distros
run_one() {
    local d=$1 out=scratch/release/distros/$1
    rm -rf "$out"; mkdir -p "$out/system/midway-verdant" "$out/png"
    # The firmware only, by size (whatever the files are called), as a new user's folder holds it.
    find -L "$fw" -maxdepth 1 -type f \( -size 512c -o -size 262144c -o -size 524288c -o -size 1048576c \) -exec cp {} "$out/system/midway-verdant/" \;
    local setup=${loader[$d]}; [ "$gpu" = soft ] && setup=${soft[$d]}
    "$engine" run --rm --device /dev/kvm "${gpu_args[@]}" -e DEBIAN_FRONTEND=noninteractive \
        -v "$PWD/$core:/mv/midway_verdant_libretro.so:ro" -v "$PWD/$host:/mv/mv-retro-host:ro" -v "$(realpath "$game"):/game.iso:ro" \
        -v "$PWD/$out:/out" "${image[$d]}" bash -c "$setup && grep -h PRETTY_NAME /etc/os-release && ldd --version | head -1 && \
        timeout $cap /mv/mv-retro-host /mv/midway_verdant_libretro.so /out/system /game.iso --frames $frames --png-every $((frames / 2)) /out/png" > "$out/log" 2>&1
    echo "exit $?" >> "$out/log"
    rm -f "$out/system/midway-verdant/xbox_hdd.raw"
    local os ran shown eeprom
    os=$(grep -m1 PRETTY_NAME "$out/log" | cut -d'"' -f2)
    ran=$(grep -m1 -oE '^ran [0-9]+ frames in [0-9.]+ s' "$out/log")
    shown=$(grep -m1 -oE '[0-9]+ frames rendered, [0-9]+ displayed' "$out/log")
    eeprom=$([ -f "$out/system/midway-verdant/eeprom.bin" ] && echo "eeprom made" || echo "NO eeprom")
    local dev; dev=$(grep -m1 -oE '\] gpu: .*' "$out/log" | cut -c8-)
    printf '%-13s %-38s %s | %s | %s | %s | %s\n' "$d" "${os:-?}" "${ran:-DID NOT RUN}" "${shown:-no frames}" "$eeprom" "${dev:-no gpu line}" "$(tail -1 "$out/log")"
}
# PARALLEL at a time (default 4): lavapipe and the package installs keep cores busy.
for d in "${order[@]}"; do
    run_one "$d" &
    while [ "$(jobs -r | wc -l)" -ge "${PARALLEL:-4}" ]; do wait -n; done
done
wait
