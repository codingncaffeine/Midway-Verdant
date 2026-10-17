#!/usr/bin/env bash
# The scan's positive controls: plant each kind of forbidden file in a scratch folder and prove the scan
# refuses every one of them, then prove it passes a clean source file. A scan that cannot see the fault
# is worse than none. Exit 0 = every control behaved.
set -uo pipefail
cd "$(dirname "$0")/../.."
work=scratch/scan-controls
rm -rf "$work"; mkdir -p "$work"
# The scan needs the private pattern file; CI has none, so a stand-in pattern is planted for the run.
if [[ ! -s local-notes/private-patterns.txt ]]; then
    mkdir -p local-notes; printf 'planted-private-token\n' > local-notes/private-patterns.txt
    trap 'rm -f local-notes/private-patterns.txt' EXIT
fi

head -c 512 /dev/urandom > "$work/rom512.dat"
head -c 1048576 /dev/urandom > "$work/bios1m.dat"
{ printf 'XBEH'; head -c 300 /dev/urandom; } > "$work/program.dat"
{ head -c 100 /dev/urandom; printf 'MICROSOFT*XBOX*MEDIA'; head -c 100 /dev/urandom; } > "$work/disc.dat"
printf 'int x = 1; // the ''user said this\n' > "$work/chatty.cpp"
printf 'int clean = 1;\n' > "$work/clean.cpp"
head -c 256 /dev/urandom > "$work/eeprom.bin"

rc=0
expect_refused() {
    if tools/precommit-scan.sh --files "$1" >/dev/null 2>&1; then echo "CONTROL FAILED: scan passed $1"; rc=1; else echo "refused as expected: $1"; fi
}
expect_refused "$work/rom512.dat"
expect_refused "$work/bios1m.dat"
expect_refused "$work/program.dat"
expect_refused "$work/disc.dat"
expect_refused "$work/chatty.cpp"
expect_refused "$work/eeprom.bin"
if tools/precommit-scan.sh --files "$work/clean.cpp" >/dev/null 2>&1; then echo "passed as expected: clean.cpp"; else echo "CONTROL FAILED: scan refused a clean file"; rc=1; fi
rm -rf "$work"
[[ $rc -eq 0 ]] && echo "scan controls OK"
exit $rc
