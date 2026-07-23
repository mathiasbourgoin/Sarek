#!/usr/bin/env bash
# Guard against silently-unwired test executables (the "vacuous e2e" failure
# mode fixed in 2026-07: an (executable) with no run rule builds green but
# never executes). Fails if any executable declared in sarek/tests/e2e/dune
# is not referenced by at least one alias/rule in that file — i.e. every
# test binary must be classified: runtest (executing or build-gated),
# compile-only, e2e-gpu, or e2e-manual.
set -euo pipefail

DUNE_FILE="sarek/tests/e2e/dune"
[ -f "$DUNE_FILE" ] || { echo "ERROR: $DUNE_FILE not found (run from repo root)"; exit 2; }

python3 - "$DUNE_FILE" <<'PYEOF'
import re, sys

path = sys.argv[1]
src = open(path).read()
# Strip line comments so names mentioned in prose don't count as wiring.
src = re.sub(r";[^\n]*", "", src)

# Parse top-level s-expressions.
def toplevel_forms(s):
    forms, depth, start = [], 0, None
    for i, c in enumerate(s):
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and start is not None:
                forms.append(s[start : i + 1])
                start = None
    return forms

declared = set()
for form in toplevel_forms(src):
    head = form[1:].split(None, 1)[0] if form[1:].split(None, 1) else ""
    if head == "executables":
        m = re.search(r"\(names((?:[^()])*)\)", form)
        if m:
            declared.update(m.group(1).split())
    elif head == "executable":
        m = re.search(r"\(name\s+([A-Za-z0-9_]+)\s*\)", form)
        if m:
            declared.add(m.group(1))

# Referenced names: every FOO.exe token (rule run actions use %{dep:FOO.exe},
# alias deps list FOO.exe directly - both match).
referenced = set(re.findall(r"([A-Za-z0-9_]+)\.exe", src))

if not declared:
    print(f"ERROR: no executables parsed from {path} - parser broken or file restructured")
    sys.exit(2)
if src.count("(") != src.count(")"):
    print(f"ERROR: unbalanced parentheses after comment-stripping in {path} - parser assumptions violated")
    sys.exit(2)

missing = sorted(declared - referenced)
if missing:
    for exe in missing:
        print(f"UNWIRED: {exe} is declared in {path} but referenced by no alias or run rule")
    print()
    print("Every test executable must be attached to an alias: runtest (executing")
    print("via (rule (alias runtest) (action (run ...))) or build-gated),")
    print("compile-only, e2e-gpu, or e2e-manual. See the comment block in")
    print(f"{path} and briefs/make-tests-actually-run-impl-notes.md.")
    sys.exit(1)

print(f"OK: all {len(declared)} e2e test executables are wired to an alias")
PYEOF
