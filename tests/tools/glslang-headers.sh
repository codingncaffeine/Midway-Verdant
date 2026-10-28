#!/bin/sh
# The glslang headers the renderer compiles against come from the glslang the core links: with the built-in copy, its
# source tree, never an installed glslang package's headers beside it (one version's headers over another's code is
# the mix the core's own glslang exists to prevent). Runs glsl_compiler.cpp's own compile command, preprocessing only,
# and reads the headers it opened.
set -u
db=${1:?compile_commands.json} tree=${2:?glslang source dir}/  # POSIX sh: Alpine has no bash by default
entry() { awk -v key="\"$1\":" '$1 == key {v = $0} /"file":.*glsl_compiler\.cpp"/ {print v; exit}' "$db"; }
unquote() { sed -e 's/^ *"[a-z]*": "//' -e 's/",\{0,1\} *$//' -e 's/\\"/"/g' -e 's/\\\\/\\/g'; }
dir=$(entry directory | unquote)
cmd=$(entry command | unquote | sed -e 's/ -MD//' -e 's/ -MT [^ ]*//' -e 's/ -MF [^ ]*//' -e 's/ -o [^ ]*//' -e 's/ -c / /')
[ -n "$cmd" ] || { echo "FAIL: no compile command for glsl_compiler.cpp in $db"; exit 1; }
opened=$(cd "$dir" && eval "$cmd -E -H -o /dev/null" 2>&1 >/dev/null | sed -n 's/^\.\.* //p' | grep -E '/(glslang|SPIRV)/')
inside=$(printf '%s\n' "$opened" | awk -v t="$tree" 'index($0, t) == 1')
outside=$(printf '%s\n' "$opened" | awk -v t="$tree" 'NF && index($0, t) != 1')
echo "glslang headers from $tree: $(printf '%s' "$inside" | grep -c .), from elsewhere: $(printf '%s' "$outside" | grep -c .)"
[ -z "$outside" ] || { printf '%s\n' "$outside" | sed 's/^/FAIL: from elsewhere: /'; exit 1; }
# The control: the built-in tree's own GlslangToSpv.h was among them, so the listing is real.
printf '%s\n' "$inside" | grep -q '/SPIRV/GlslangToSpv\.h$' || { echo "FAIL: ${tree}SPIRV/GlslangToSpv.h not among the headers read"; exit 1; }
