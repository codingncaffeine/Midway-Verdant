#!/usr/bin/env bash
# Boots the user's own MCPX ROM and BIOS in fast mode and checks the boot chain's milestones against bands.
# The files are named in local-notes/user-files.env (gitignored): MV_MCPX=... MV_BIOS=... [MV_EEPROM=...].
# Without them the test is skipped (exit 77): CI has no console dumps and never will.
#   tests/boot/boot-smoke.sh BUILD_DIR
set -uo pipefail
cd "$(dirname "$0")/../.."
build="${1:-scratch/build}"
env_file=local-notes/user-files.env
[[ -f "$env_file" ]] || { echo "SKIP: $env_file missing"; exit 77; }
# shellcheck disable=SC1090
source "$env_file"
[[ -f "${MV_MCPX:-}" && -f "${MV_BIOS:-}" ]] || { echo "SKIP: MV_MCPX/MV_BIOS not present"; exit 77; }
[[ -w /dev/kvm ]] || { echo "SKIP: no /dev/kvm"; exit 77; }
mkdir -p scratch/boot
out=scratch/boot/smoke.out
"$build/src/midway-verdant" --mcpx "$MV_MCPX" --bios "$MV_BIOS" ${MV_EEPROM:+--eeprom "$MV_EEPROM"} --seconds 2 --rtc-time 1000000000 > "$out" 2>&1
rc=$?
cat "$out"
[[ $rc -eq 0 ]] || { echo "FAIL: runner exit $rc"; exit 1; }

fail=0
need() { if ! grep -qE "$1" "$out"; then echo "FAIL: expected /$1/"; fail=1; fi; }
# The chain: ROM hidden, bootloader magic, SMC challenge, kernel image — each inside its band of guest time.
ms_of() { grep -oE "$1 [0-9.]+ ms" "$out" | grep -oE '[0-9.]+' | head -1; }
band() {  # name value lo hi
    local v="$2"
    [[ -n "$v" ]] || { echo "FAIL: $1 never happened"; fail=1; return; }
    if awk -v v="$v" -v lo="$3" -v hi="$4" 'BEGIN { exit !(v >= lo && v <= hi) }'; then
        echo "ok: $1 at $v ms (band $3..$4)"
    else
        echo "FAIL: $1 at $v ms, outside $3..$4"; fail=1
    fi
}
band "rom hidden" "$(ms_of 'rom hidden')" 1 50
band "bootloader" "$(ms_of 'bootloader')" 1 50
band "smc challenge" "$(ms_of 'smc challenge')" 1 60
band "kernel" "$(ms_of 'kernel')" 2 100
# The timer: 1 kHz of injected interrupts over the run, within 5 per cent, so the PIT, PIC and injection path all work.
injected=$(grep -oE 'timer interrupts \(irq0, vector [0-9a-f]+\): [0-9]+' "$out" | grep -oE '[0-9]+$')
guest_ms=$(grep -oE '[0-9.]+ ms guest' "$out" | grep -oE '^[0-9.]+')
if [[ -n "$injected" && -n "$guest_ms" ]]; then
    if awk -v n="$injected" -v ms="$guest_ms" 'BEGIN { r = n / (ms / 1000); exit !(r >= 950 && r <= 1050) }'; then
        echo "ok: timer interrupts $injected over $guest_ms ms (950..1050 Hz)"
    else
        echo "FAIL: timer interrupts $injected over $guest_ms ms"; fail=1
    fi
else
    echo "FAIL: no interrupt count"; fail=1
fi
need 'cpu: cs 0008 eip 8[0-9a-f]{7}'  # protected mode, in the kernel
[[ $fail -eq 0 ]] && echo "BOOT SMOKE OK"
exit $fail
