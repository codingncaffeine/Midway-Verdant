#!/usr/bin/env bash
# Lands the local commits on main only after CI has run them on Linux (Windows runs too, informational): the commits go to the `gate`
# branch first, this waits for that run, and main moves when it is green. A red run prints the failed jobs and
# leaves main as it was. The content and identity scans run first. Start it in the background after
# tools/gate.sh has passed and keep working.
#   tools/push.sh [COMMIT]     (default HEAD)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p scratch/gate
sha=$(git rev-parse "${1:-HEAD}")
short=${sha:0:7}

echo "── scans"
tools/precommit-scan.sh --all > scratch/gate/scan.log 2>&1 || { tail -20 scratch/gate/scan.log; echo "PUSH REFUSED: the content scan"; exit 1; }
tools/identity-audit.sh > scratch/gate/identity.log 2>&1 || { tail -20 scratch/gate/identity.log; echo "PUSH REFUSED: the identity audit"; exit 1; }

echo "── gate branch: $short"
git push -q --force origin "$sha:refs/heads/gate" || { echo "PUSH FAILED: the gate branch"; exit 1; }

echo "── CI on gate: Linux (Windows informational)"
run=""
for _ in $(seq 1 40); do
    run=$(gh run list --branch gate --commit "$sha" --json databaseId --jq '.[0].databaseId // empty' 2>/dev/null)
    [ -n "$run" ] && break
    sleep 15
done
[ -n "$run" ] || { echo "PUSH FAILED: no CI run appeared for $short on gate"; exit 1; }

# The run's own conclusion decides, never the watcher's exit code (a GitHub 502 mid-watch ends the watch early).
: > "scratch/gate/ci-$short.log"
conclusion=""
for _ in $(seq 1 240); do
    gh run watch "$run" --interval 30 >> "scratch/gate/ci-$short.log" 2>&1
    state=$(gh run view "$run" --json status,conclusion --jq '"\(.status) \(.conclusion)"' 2>/dev/null)
    case "$state" in
        "completed "*) conclusion=${state#completed }; break ;;
    esac
    sleep 30
done

if [ "$conclusion" = success ]; then
    git push -q origin "$sha:main" || { echo "PUSH FAILED: main"; exit 1; }
    echo "PUSHED $short to main: CI run $run green on Linux"
    gh run view "$run" --json jobs --jq '.jobs[] | "  \(.name): \(.conclusion)"'  # Windows' result, for the record
else
    echo "CI FAILED on gate for $short (run $run); main unchanged."
    gh run view "$run" --json jobs --jq '.jobs[] | "  \(.name): \(.conclusion)"'
    gh run view "$run" --log-failed 2>/dev/null | grep -E "error|Failed|FAILED|Assert|Expected|Which is" | cut -c1-200 | sort -u | head -30
    exit 1
fi
