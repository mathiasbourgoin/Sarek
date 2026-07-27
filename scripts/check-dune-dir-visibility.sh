#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Guard against build directories that are invisible to dune (#147).
#
# A `(dirs ...)` / `(data_only_dirs ...)` / `(vendored_dirs ...)` stanza in a
# `dune` file restricts which subdirectories dune traverses beneath it. A
# directory excluded that way is not merely unbuilt — it does not exist as far
# as dune is concerned. Its `dune` file is never read, its tests never run,
# never format-check, and never appear in any alias. `dune build` stays green
# and `dune build @fmt` stays green, because there is nothing there to be red.
#
# That is how sarek/sarek/test/'s seven test executables sat unbuilt: a
# `(dirs ir_extract)` line in sarek/sarek/dune, inherited from a 2025-12
# refactor whose purpose (excluding a legacy camlp4 `extension/` directory) had
# been obsolete since that directory was deleted.
#
# This check fails if any tracked `dune` file declaring a build stanza
# (library / executable(s) / test(s)) sits in a directory that an ancestor
# `dune` file excludes from traversal.
#
# It is deliberately static: no dune, no OCaml switch, no built tree. Asking
# dune whether it can see a directory it has been told does not exist is not a
# question dune can answer.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python3 - <<'PYEOF'
import re
import subprocess
import sys

BUILD_STANZAS = ("library", "libraries", "executable", "executables", "test", "tests")
# Stanzas that remove a subdirectory from dune's build traversal.
SCOPE_STANZAS = ("dirs", "data_only_dirs", "vendored_dirs")


def strip_comments(src):
    """Remove dune line comments so names in prose do not parse as stanzas."""
    return re.sub(r";[^\n]*", "", src)


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


def head_of(form):
    body = form[1:].split(None, 1)
    return body[0].strip("()") if body else ""


def admits(predicate, name):
    """Evaluate a dune dir-predicate against one immediate subdirectory name.

    Supports the forms this repository actually uses: bare names, :standard,
    the * glob, and \\ set difference. Anything else exits 2 rather than
    guessing -- a guard that silently mis-parses its input is worse than no
    guard, because it reads green.
    """
    if "\\" in predicate:
        included, excluded = predicate.split("\\", 1)
    else:
        included, excluded = predicate, ""

    def matches(tokens, name):
        for tok in tokens.split():
            if tok == ":standard":
                # dune's :standard for dirs = everything not dot- or underscore-prefixed
                if not name.startswith(".") and not name.startswith("_"):
                    return True
            elif tok == "*":
                return True
            elif re.fullmatch(r"[A-Za-z0-9_.\-]+", tok):
                if tok == name:
                    return True
            else:
                print(
                    f"ERROR: unsupported dir-predicate token {tok!r} -- "
                    "this guard's parser does not cover it, so it cannot "
                    "honestly report on it. Extend admits() in "
                    "scripts/check-dune-dir-visibility.sh."
                )
                sys.exit(2)
        return False

    return matches(included, name) and not matches(excluded, name)


tracked = subprocess.run(
    ["git", "ls-files", "*dune", "dune"],
    capture_output=True, text=True, check=True,
).stdout.split()
dune_files = sorted({p for p in tracked if p == "dune" or p.endswith("/dune")})

if not dune_files:
    print("ERROR: no tracked dune files found -- run from the repo root")
    sys.exit(2)

# dir -> {stanza_name: predicate} for every dune file carrying a scope stanza
scopes = {}
# dirs whose dune file declares something buildable
build_dirs = []

for path in dune_files:
    d = path[: -len("dune")].rstrip("/")
    with open(path) as fh:
        src = strip_comments(fh.read())
    if src.count("(") != src.count(")"):
        print(f"ERROR: unbalanced parentheses in {path} -- parser assumptions violated")
        sys.exit(2)
    forms = toplevel_forms(src)
    heads = {head_of(f) for f in forms}
    if heads & set(BUILD_STANZAS):
        build_dirs.append((d, sorted(heads & set(BUILD_STANZAS))))
    for form in forms:
        h = head_of(form)
        if h in SCOPE_STANZAS:
            inner = form[1 + len(h) : -1].strip()
            scopes.setdefault(d, {})[h] = inner

if not build_dirs:
    print("ERROR: no build stanzas parsed from any dune file -- parser broken")
    sys.exit(2)

violations = []
for d, stanzas in build_dirs:
    parts = d.split("/") if d else []
    # Walk ancestors from the root down; the child component must survive each
    # ancestor's scope stanza.
    for i in range(len(parts)):
        ancestor = "/".join(parts[:i])
        child = parts[i]
        for stanza, predicate in scopes.get(ancestor, {}).items():
            if stanza == "dirs":
                visible = admits(predicate, child)
            else:
                # data_only_dirs / vendored_dirs: matching means NOT built
                visible = not admits(predicate, child)
            if not visible:
                violations.append((d, stanzas, ancestor or ".", stanza, predicate, child))

if violations:
    for d, stanzas, ancestor, stanza, predicate, child in violations:
        print(f"INVISIBLE: {d}/dune declares {', '.join(stanzas)} but dune never reads it")
        print(f"           {ancestor}/dune has ({stanza} {predicate}), which excludes {child!r}")
        print()
    print("A directory excluded from dune's traversal does not exist as far as")
    print("dune is concerned: its tests never build, never run, and never")
    print("format-check, while `dune build` and `dune build @fmt` both stay green.")
    print("Either widen the scope stanza to admit the directory, or delete the")
    print("dune file that is pretending to declare a build there.")
    sys.exit(1)

print(f"OK: all {len(build_dirs)} directories with build stanzas are visible to dune")
PYEOF
