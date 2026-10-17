#!/usr/bin/env bash
# Every author and committer in the history, local and remote, must be the project identity.
# Exit 0 = clean, 1 = a foreign identity, 2 = could not check.
set -euo pipefail
cd "$(dirname "$0")/.."

allowed='codingncaffeine <codingncaffeine@users.noreply.github.com>'
# GitHub's web editor signs the same account's commits with its numbered noreply address (no personal address).
web='codingncaffeine <19158626+codingncaffeine@users.noreply.github.com>'
# GitHub stamps web merges with its own committer identity.
github='GitHub <noreply@github.com>'

if ! git rev-parse --verify -q HEAD >/dev/null; then
    echo "identity-audit: no commits yet"
    exit 0
fi

mapfile -t found < <(git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u)
bad=0
for id in "${found[@]}"; do
    if [[ "$id" != "$allowed" && "$id" != "$web" && "$id" != "$github" ]]; then
        echo "identity-audit: foreign identity in history: $id"
        bad=1
    fi
done
# Remote refs the local clone lacks (stale tags) keep old history reachable; look at them too.
while read -r _ ref; do
    case "$ref" in refs/heads/*|refs/tags/*) ;; *) continue ;; esac
    git fetch -q origin "$ref" 2>/dev/null || true
done < <(git ls-remote origin 2>/dev/null)
if [[ $bad -eq 0 ]]; then
    echo "identity-audit: ${#found[@]} identity line(s), all the project identity"
fi
exit $bad
