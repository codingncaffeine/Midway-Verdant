#!/usr/bin/env bash
# The gate before a push: a fresh Release configure and build with warnings as errors, then the whole suite
# the way CI runs it, N times (default 2) on two CPUs — the runners have two to four, and this machine's
# thirty-two hide the interleavings they see. Run it in the background and keep working; push once it
# says GATE PASSED. Logs are in scratch/gate/.
#   tools/gate.sh [N]
set -uo pipefail
cd "$(dirname "$0")/.."
N="${1:-2}"
export SDL_AUDIODRIVER=dummy SDL_AUDIO_DRIVER=dummy SDL_VIDEODRIVER=dummy
mkdir -p scratch/gate
build=scratch/gate/build
rm -rf "$build"

echo "── configure + build, warnings as errors"
if ! cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMV_WERROR=ON > scratch/gate/configure.log 2>&1; then
    tail -30 scratch/gate/configure.log; echo "GATE FAILED: configure"; exit 1
fi
if ! cmake --build "$build" > scratch/gate/build.log 2>&1; then
    grep -E "error|warning" scratch/gate/build.log | sort -u | head -30; echo "GATE FAILED: the build"; exit 1
fi

echo "── scans"
tools/precommit-scan.sh --all > scratch/gate/scan.log 2>&1 || { tail -20 scratch/gate/scan.log; echo "GATE FAILED: the content scan"; exit 1; }
tests/tools/scan-controls.sh > scratch/gate/scan-controls.log 2>&1 || { tail -20 scratch/gate/scan-controls.log; echo "GATE FAILED: the scan controls"; exit 1; }

rc=0
for i in $(seq 1 "$N"); do
    log="scratch/gate/run-$i.log"
    if command -v taskset > /dev/null; then
        taskset -c 0,1 ctest --test-dir "$build" --output-on-failure --timeout 120 --no-tests=error > "$log" 2>&1
    else
        ctest --test-dir "$build" --output-on-failure --timeout 120 --no-tests=error > "$log" 2>&1
    fi
    r=$?
    line=$(grep -E "tests passed|tests failed" "$log" | tail -1)
    echo "run $i of $N: ${line:-no result, see $log}"
    if [[ $r -ne 0 ]]; then rc=1; grep -E "\*\*\*Failed|Timeout" "$log" | sed 's/^/    /'; fi
done

if (( rc == 0 )); then echo "GATE PASSED: $N runs on 2 CPUs, CI-style"; else echo "GATE FAILED"; fi
exit $rc
