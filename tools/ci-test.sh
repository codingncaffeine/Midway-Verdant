#!/usr/bin/env bash
# Runs the test suite the way CI does: every test named, each bounded by its timeout, and a test that fails
# is run once more on its own. Passing then, it is reported as FLAKY and the job stays green; failing twice fails.
#   tools/ci-test.sh BUILD_DIR [CONFIG]
set -uo pipefail
build="${1:?build dir}"; config="${2:-Release}"
ctest --test-dir "$build" -C "$config" --output-on-failure --timeout 120 --no-tests=error
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "── rerunning the failed tests once"
    if ctest --test-dir "$build" -C "$config" --rerun-failed --output-on-failure --timeout 120; then
        echo "FLAKY: the failed tests passed on their own; fix them before they fail twice"
        exit 0
    fi
    echo "TESTS FAILED twice"
    exit 1
fi
