#!/usr/bin/env bash
# The boot animation's pace and sound through the libretro core, the way a frontend runs it: 720 frames (12 s of
# console time) headless, then
#   - the closing jingle's voice (0x46) started,
#   - the animation rendered as many frames as a run of the same 12 s without the GPU, within 2 per cent,
#   - with a baseline (written by --write-baseline), the sound's level in every second within 1 dB of it.
# --load runs the core pinned to two cores with four busy loops on them: guest time is the virtual CPU's work, so
# the same checks must pass under host load (AV-PLAN F4). Needs the user's console files: MV_MCPX and MV_BIOS in
# local-notes/user-files.env and a system folder MV_RETRO_SYS (default scratch/retro-sys) holding the EEPROM and an
# isolated hard-disk image; skipped without them.
#   tests/boot/av-smoke.sh BUILD_DIR [--load] [--write-baseline]
set -uo pipefail
cd "$(dirname "$0")/../.."
build="${1:-scratch/build}"
shift || true
load=0 write_baseline=0
for a in "$@"; do
    case "$a" in
        --load) load=1 ;;
        --write-baseline) write_baseline=1 ;;
        *) echo "unknown option $a"; exit 2 ;;
    esac
done
env_file=local-notes/user-files.env
[[ -f "$env_file" ]] || { echo "SKIP: $env_file missing"; exit 77; }
# shellcheck disable=SC1090
source "$env_file"
sys="${MV_RETRO_SYS:-scratch/retro-sys}"
[[ -f "${MV_MCPX:-}" && -f "${MV_BIOS:-}" && -f "$sys/midway-verdant/eeprom.bin" && -f "$sys/midway-verdant/xbox_hdd.raw" ]] ||
    { echo "SKIP: console files or $sys/midway-verdant missing"; exit 77; }
[[ -w /dev/kvm ]] || { echo "SKIP: no /dev/kvm"; exit 77; }
out=scratch/av-smoke
mkdir -p "$out"
baseline=scratch/av-baseline.txt
fail=0

# The reference: the same 12 s of console time without the GPU (the renderer costs the guest nothing, so the
# animation's frames must not depend on it).
"$build/src/midway-verdant" --mcpx "$MV_MCPX" --bios "$MV_BIOS" --eeprom "$sys/midway-verdant/eeprom.bin" \
    --hdd "$sys/midway-verdant/xbox_hdd.raw" --rtc-time 1000000000 --guest-seconds 12 --speed 0 > "$out/nogpu.log" 2>&1
ref=$(grep -oE '[0-9]+ semaphore releases' "$out/nogpu.log" | grep -oE '^[0-9]+')

hogs=()
if [[ $load -eq 1 ]]; then
    for _ in 1 2 3 4; do taskset -c 2,3 timeout 120 sh -c 'while :; do :; done' & hogs+=($!); done
    sleep 1
    run=(taskset -c 2,3)
else
    run=()
fi
"${run[@]}" "$build/src/mv-retro-host" "$build/src/midway_verdant_libretro.so" "$sys" --frames 720 --wav "$out/core.wav" > "$out/core.log" 2>&1
rc=$?
for p in "${hogs[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null
[[ $rc -eq 0 ]] || { echo "FAIL: mv-retro-host exit $rc"; exit 1; }
grep -E 'console time|retro_run:|ran 720' "$out/core.log"

voices=$(grep -oE 'voices started:.*' "$out/core.log")
if grep -qE ' 46@' <<< "$voices"; then echo "ok: the jingle's voice 0x46 started ($(grep -oE '46@[0-9.]+' <<< "$voices"))"; else echo "FAIL: no voice 0x46: $voices"; fail=1; fi

frames=$(grep -oE '[0-9]+ frames rendered' "$out/core.log" | grep -oE '^[0-9]+')
if [[ -n "$frames" && -n "$ref" ]] && awk -v f="$frames" -v r="$ref" 'BEGIN { d = (f - r) / r; if (d < 0) d = -d; exit !(d <= 0.02) }'; then
    echo "ok: $frames frames rendered against $ref without the GPU (within 2%)"
else
    echo "FAIL: ${frames:-no} frames rendered against ${ref:-no} without the GPU"; fail=1
fi

# The sound's RMS level per second of the 12 (dBFS).
levels=$(od -An -v -td2 -w4 -j44 "$out/core.wav" | awk '{ s += $1 * $1 + $2 * $2; n += 2; if (n == 96000) { printf "%.1f\n", (s > 0 ? 10 * log(s / n / 1073741824) / log(10) : -120); s = 0; n = 0 } }' | head -12)
if [[ $write_baseline -eq 1 ]]; then
    printf '%s\n' "$levels" > "$baseline"
    echo "baseline written to $baseline: $(tr '\n' ' ' < "$baseline")"
elif [[ -f "$baseline" ]]; then
    if paste -d' ' "$baseline" <(printf '%s\n' "$levels") | awk '{ d = $1 - $2; if (d < 0) d = -d; if (d > 1.0) { bad = 1; printf "  second %d: %s dB against %s\n", NR, $2, $1 } } END { exit bad }'; then
        echo "ok: the level of every second within 1 dB of the baseline"
    else
        echo "FAIL: the level differs from the baseline by more than 1 dB"; fail=1
    fi
else
    echo "note: no baseline ($baseline); levels: $(tr '\n' ' ' <<< "$levels")"
fi
[[ $fail -eq 0 ]] && echo "AV SMOKE OK$([[ $load -eq 1 ]] && echo ' (under load)')"
exit $fail
