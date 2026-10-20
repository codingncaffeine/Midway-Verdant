#!/usr/bin/env bash
# The recompiler's oracle on a title: the accurate core runs the disc twice through the libretro core, once with the
# block interpreter (MV_P6_JIT=0, the reference) and once recompiled, from the same disk image and a fixed clock, both
# stopped at the same cycle 30 frames before the last. Passes when the delivery logs (cycle, instruction count, EIP
# and a hash of the whole CPU state at every interrupt and fault), the final CPU state and RAM hashes and the pictures
# (every 600 frames) are identical. Extra arguments go to mv-retro-host (presses that drive the title into play).
# Needs the user's console files (local-notes/user-files.env, as av-smoke.sh); work in scratch/p6-oracle/LABEL.
#   tests/boot/p6-oracle.sh BUILD_DIR DISC LABEL FRAMES [mv-retro-host args...]
#   env: DISK=raw hard-disk image to start from (default the system folder's), TMO=seconds per run (default 7200),
#        GOLDEN=DIR: with no golden in DIR yet, the full comparison runs and, when it passes, its reference (the block
#        interpreter's deliveries, final hashes and pictures) is kept there; with one, only the recompiler runs and is
#        compared with it (a speed change must leave the guest's run unchanged: minutes instead of the interpreter's
#        hour). Delete the golden when the guest's behaviour is meant to change.
set -uo pipefail
cd "$(dirname "$0")/../.."
[[ $# -ge 4 ]] || { sed -n '2,9p' "$0"; exit 2; }
build=$1 disc=$2 label=$3 frames=$4
shift 4
env_file=local-notes/user-files.env
[[ -f "$env_file" ]] || { echo "SKIP: $env_file missing"; exit 77; }
# shellcheck disable=SC1090
source "$env_file"
sys="${MV_RETRO_SYS:-scratch/retro-sys}"
disk="${DISK:-$sys/midway-verdant/xbox_hdd.raw}"
[[ -f "$disc" && -f "$disk" && -f "$sys/midway-verdant/eeprom.bin" ]] || { echo "SKIP: disc, disk image or $sys/midway-verdant missing"; exit 77; }
out="scratch/p6-oracle/$label"
rm -rf "$out"
stop=$(( (frames - 30) * 733333333 / 60 ))
run() {  # run JIT [mv-retro-host args]
    local j=$1
    shift
    local d="$out/j$j"
    mkdir -p "$d/sys/midway-verdant"
    cp "$sys"/midway-verdant/*.bin "$d/sys/midway-verdant/"
    cp --reflink=auto "$disk" "$d/sys/midway-verdant/xbox_hdd.raw"
    MV_FRAME_WAIT_MS=0 MV_RTC_TIME=1000000000 MV_P6_JIT=$j MV_P6_STOP_CYCLES=$stop MV_P6_DELIVERY_LOG="$d/deliveries.txt" timeout "${TMO:-7200}" \
        "$build/src/mv-retro-host" "$build/src/midway_verdant_libretro.so" "$d/sys" "$disc" --frames "$frames" \
        --option midway_verdant_cpu=accurate --option midway_verdant_pipeline=disabled --png-every 600 "$d/png" "$@" > "$d/log" 2>&1
    echo $? > "$d/exit"
    rm -f "$d/sys/midway-verdant/xbox_hdd.raw"
}
golden="${GOLDEN:-}"
if [[ -n "$golden" && -f "$golden/deliveries.txt" ]]; then
    # Against the kept reference: the recompiler alone; the reference stands in for the interpreter's run.
    run 1 "$@"
    mkdir -p "$out/j0"
    cp "$golden/deliveries.txt" "$out/j0/deliveries.txt"
    cp "$golden/log" "$out/j0/log"
    cp -r "$golden/png" "$out/j0/png"
    echo 0 > "$out/j0/exit"
    echo "against the golden in $golden"
else
    run 0 "$@" &
    run 1 "$@" &
    wait
fi
fail=0
for j in 0 1; do
    [[ "$(cat "$out/j$j/exit")" == 0 ]] || { echo "FAIL: the run with MV_P6_JIT=$j exited $(cat "$out/j$j/exit") (see $out/j$j/log)"; fail=1; }
    at=$(grep -ohE "stopped at cycle [0-9]+" "$out/j$j/log" | grep -oE "[0-9]+$")
    [[ -n "$at" && "$at" -ge "$stop" ]] || { echo "FAIL: the run with MV_P6_JIT=$j did not reach the stop cycle ($stop; it stopped at ${at:-?})"; fail=1; }
done
if cmp -s "$out/j0/deliveries.txt" "$out/j1/deliveries.txt"; then
    echo "deliveries: identical ($(wc -l < "$out/j0/deliveries.txt"))"
else
    echo "FAIL: deliveries differ: $(cmp "$out/j0/deliveries.txt" "$out/j1/deliveries.txt" 2>&1)"; fail=1
fi
h0=$(grep -h "state hash" "$out/j0/log") h1=$(grep -h "state hash" "$out/j1/log")
if [[ -n "$h0" && "$h0" == "$h1" ]]; then echo "final state and RAM: identical"; else echo "FAIL: final state or RAM differs: [$h0] [$h1]"; fail=1; fi
n=0 bad=0
for p in "$out"/j0/png/*.png; do
    [[ -f "$p" ]] || continue
    n=$((n + 1))
    cmp -s "$p" "$out/j1/png/$(basename "$p")" || bad=$((bad + 1))
done
if [[ $bad -eq 0 ]]; then echo "pictures: $n identical"; else echo "FAIL: $bad of $n pictures differ"; fail=1; fi
if [[ $fail -eq 0 && -n "$golden" && ! -f "$golden/deliveries.txt" ]]; then
    mkdir -p "$golden"
    cp "$out/j0/deliveries.txt" "$out/j0/log" "$golden/"
    cp -r "$out/j0/png" "$golden/png"
    echo "golden kept in $golden"
fi
[[ $fail -eq 0 ]] && echo "P6 ORACLE PASSED: $label" || echo "P6 ORACLE FAILED: $label"
exit $fail
