#!/usr/bin/env bash
# Scans what is about to be committed for things that must never reach the public repository:
# console dumps and game data (ROMs, BIOS images, kernels, EEPROMs, discs, executables), local-only paths,
# personal data and conversation in comments.
#
# Personal patterns live in the gitignored local-notes/private-patterns.txt (one extended regex per line) so
# the patterns themselves never reach the repository. The scan refuses to run without that file, and proves
# every pattern matches its own planted sample before trusting it.
#
# Usage: tools/precommit-scan.sh                 (scans the staged changes)
#        tools/precommit-scan.sh --all           (scans every tracked file)
#        tools/precommit-scan.sh --files F...    (scans the named files as they are on disk; the controls use it)
# Exit 0 = clean, 1 = something found, 2 = could not scan.
set -euo pipefail
cd "$(dirname "$0")/.."

private=local-notes/private-patterns.txt
if [[ ! -s "$private" ]]; then
    echo "precommit-scan: $private is missing or empty; refusing to report clean" >&2
    exit 2
fi

conversation='the user (said|asked|wants)|you asked|we decided|as discussed|per our|per the user'
# Known dumps by MD5: the MCPX 1.0 boot ROM and the two BIOS images beside it.
known_md5='d49c52a4102f6df7bcf8d0617ac475ed|ec00e31e746de2473acfe7903c5a4cb7|21445c6f28fca7285b0f167ea770d1e5'

mode="${1:-staged}"
case "$mode" in
    --all)
        mapfile -t files < <(git ls-files)
        content() { cat -- "$1"; } ;;
    --files)
        shift; files=("$@")
        content() { cat -- "$1"; } ;;
    *)
        mapfile -t files < <(git diff --cached --name-only --diff-filter=ACMR)
        content() { git show ":$1"; } ;;
esac

found=0

# Positive controls: each pattern must find its own sample, or the scan cannot see anything.
while IFS= read -r pattern; do
    [[ -z "$pattern" || "$pattern" == \#* ]] && continue
    sample="${pattern//\\/}"
    sample="${sample//[\^\$]/}"
    if ! printf '%s\n' "$sample" | command grep -qiE -- "$pattern"; then
        echo "precommit-scan: pattern does not match its own sample, fix it: $pattern" >&2
        exit 2
    fi
done < "$private"
if ! printf 'per our call\n' | command grep -qiE -- "$conversation"; then
    echo "precommit-scan: the conversation pattern failed its control" >&2
    exit 2
fi
if ! printf 'XBEH' | command grep -q 'XBEH'; then
    echo "precommit-scan: grep is not working" >&2
    exit 2
fi

for f in "${files[@]}"; do
    if [[ "$mode" != --files ]]; then
        case "$f" in
            local-notes/*|scratch/*|snapshots/*|reference/*)
                echo "precommit-scan: local-only path staged: $f"; found=1; continue ;;
        esac
    fi
    base="${f##*/}"
    case "${base,,}" in
        *.bin|*.rom|*.iso|*.xiso|*.xbe|*.qcow2|*.img|*.raw|*.cso)
            echo "precommit-scan: console dump or game data staged: $f"; found=1; continue ;;
    esac
    [[ -f "$f" ]] || continue
    size=$(content "$f" | wc -c)
    if content "$f" | file -b --mime - | command grep -q 'charset=binary'; then
        # A binary the size of a boot ROM or a BIOS image is one until proven otherwise.
        case "$size" in
            512|262144|524288|1048576)
                echo "precommit-scan: binary of a console-dump size ($size bytes): $f"; found=1; continue ;;
        esac
        if content "$f" | head -c 4 | command grep -q 'XBEH'; then
            echo "precommit-scan: Xbox executable (XBEH) staged: $f"; found=1; continue
        fi
        if content "$f" | command grep -aq 'MICROSOFT\*XBOX\*MEDIA'; then
            echo "precommit-scan: Xbox disc image staged: $f"; found=1; continue
        fi
        if content "$f" | md5sum | command grep -qE "^($known_md5)"; then
            echo "precommit-scan: known console dump staged: $f"; found=1; continue
        fi
        continue
    fi
    if content "$f" | command grep -n -iE -f <(command grep -vE '^(#|$)' "$private") >/dev/null; then
        echo "precommit-scan: personal data in $f:"
        content "$f" | command grep -n -iE -f <(command grep -vE '^(#|$)' "$private") | sed 's/^/    /'
        found=1
    fi
    if [[ "$f" != tools/precommit-scan.sh ]] && content "$f" | command grep -n -iE -- "$conversation" >/dev/null; then
        echo "precommit-scan: conversational wording in $f:"
        content "$f" | command grep -n -iE -- "$conversation" | sed 's/^/    /'
        found=1
    fi
done

if [[ $found -eq 0 ]]; then
    echo "precommit-scan: ${#files[@]} file(s) clean"
fi
exit $found
