#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Guard against build directories that are invisible to dune (#147).
#
# A `(dirs ...)` or `(data_only_dirs ...)` stanza in a `dune` file restricts
# which subdirectories dune traverses beneath it. A directory excluded that way
# is not merely unbuilt -- it does not exist as far as dune is concerned. Its
# `dune` file is never read, its tests never run, never format-check, and never
# appear in any alias. `dune build` stays green and `dune build @fmt` stays
# green, because there is nothing there to be red.
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
# NOT CHECKED: `vendored_dirs`. It is tempting to lump it in with the two
# above, and doing so is wrong. Vendoring does not remove a directory from
# dune's traversal -- dune still parses vendored `dune` files and still builds
# vendored libraries and executables. What changes is defaults: vendored code
# is dropped from the default alias, is not format-checked, is not installed,
# and builds without the dev profile's warnings-as-errors. So a vendored
# directory containing a `(library)` is a legitimate layout, not a defect, and
# flagging it would be a false positive. That matters more than the coverage it
# costs: a guard that cries wolf on a valid layout is a guard the next person
# disables, and then it catches nothing at all.
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
# Stanzas that genuinely remove a subdirectory from dune's build traversal.
# `vendored_dirs` is deliberately absent -- see the header comment.
SCOPE_STANZAS = ("dirs", "data_only_dirs")


def lex(src):
    """Blank out comments and mark string-literal spans, preserving offsets.

    In dune syntax `;` opens a comment only OUTSIDE a string, and a string may
    itself contain `;`, `(` and `)` -- so a naive regex strip mangles
    `(dirs "a;b")` and then miscounts parentheses on `(dirs "(a;(x)")`.

    Returns (blanked, in_str): `blanked` is `src` with every comment character
    replaced by a space (same length, so all indices stay valid), and
    `in_str[i]` is True when `src[i]` lies inside a double-quoted string
    literal, its delimiters included.
    """
    out = list(src)
    in_str = [False] * len(src)
    i, n, state = 0, len(src), "code"
    while i < n:
        c = src[i]
        if state == "code":
            if c == '"':
                state, in_str[i] = "str", True
            elif c == ";":
                state, out[i] = "comment", " "
        elif state == "str":
            in_str[i] = True
            if c == "\\" and i + 1 < n:
                in_str[i + 1] = True
                i += 2
                continue
            if c == '"':
                state = "code"
        else:  # comment
            if c == "\n":
                state = "code"
            else:
                out[i] = " "
        i += 1
    return "".join(out), in_str


def balanced(src, in_str):
    depth = 0
    for i, c in enumerate(src):
        if in_str[i]:
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def toplevel_spans(src, in_str):
    """Index spans of top-level s-expressions, ignoring parens inside strings."""
    spans, depth, start = [], 0, None
    for i, c in enumerate(src):
        if in_str[i]:
            continue
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, i + 1))
                start = None
    return spans


def tokenize(text, mask):
    """Split a dir-predicate into (value, was_quoted) tokens.

    Quoted tokens keep their contents verbatim: a directory may legitimately be
    named `a;b` or contain parentheses, and such a name is a literal, never a
    glob or `:standard`.
    """
    toks, cur, quoted, seen = [], "", False, False
    i = 0
    while i < len(text):
        c = text[i]
        if mask[i]:
            seen = True
            if c == "\\" and i + 1 < len(text) and mask[i + 1]:
                cur += text[i + 1]
                i += 2
                continue
            if c == '"':
                quoted = True
            else:
                cur += c
            i += 1
            continue
        if c.isspace():
            if cur or seen:
                toks.append((cur, quoted))
                cur, quoted, seen = "", False, False
        else:
            cur += c
        i += 1
    if cur or seen:
        toks.append((cur, quoted))
    return toks


def admits(text, mask, name, where):
    """Evaluate a dune dir-predicate against one immediate subdirectory name.

    Supports the forms this repository actually uses: bare names, quoted names,
    :standard, the * glob, and \\ set difference. Anything else exits 2 rather
    than guessing -- a guard that silently mis-parses its input is worse than
    no guard, because it reads green.
    """
    # The set-difference separator is a backslash OUTSIDE a string literal.
    cut = next((i for i, c in enumerate(text) if c == "\\" and not mask[i]), None)
    if cut is None:
        inc, exc = tokenize(text, mask), []
    else:
        inc = tokenize(text[:cut], mask[:cut])
        exc = tokenize(text[cut + 1 :], mask[cut + 1 :])

    def matches(toks):
        for value, was_quoted in toks:
            if was_quoted:
                # A quoted token is always a literal directory name.
                if value == name:
                    return True
            elif value == ":standard":
                # dune's :standard for dirs = everything not dot- or
                # underscore-prefixed.
                if not name.startswith(".") and not name.startswith("_"):
                    return True
            elif value == "*":
                return True
            elif re.fullmatch(r"[A-Za-z0-9_.\-]+", value):
                if value == name:
                    return True
            else:
                print(
                    f"ERROR: unsupported dir-predicate token {value!r} in {where} -- "
                    "this guard's parser does not cover it, so it cannot honestly "
                    "report on it. Extend admits() in "
                    "scripts/check-dune-dir-visibility.sh."
                )
                sys.exit(2)
        return False

    return matches(inc) and not matches(exc)


tracked = subprocess.run(
    ["git", "ls-files", "-z", "*dune", "dune"],
    capture_output=True, check=True,
).stdout.decode().split("\0")
dune_files = sorted({p for p in tracked if p == "dune" or p.endswith("/dune")})

if not dune_files:
    print("ERROR: no tracked dune files found -- run from the repo root")
    sys.exit(2)

scopes = {}      # dir -> {stanza: (text, mask)}
build_dirs = []  # (dir, [stanza names])

for path in dune_files:
    d = path[: -len("dune")].rstrip("/")
    with open(path) as fh:
        blanked, in_str = lex(fh.read())
    if not balanced(blanked, in_str):
        print(f"ERROR: unbalanced parentheses in {path} -- parser assumptions violated")
        sys.exit(2)
    heads = set()
    for a, b in toplevel_spans(blanked, in_str):
        form = blanked[a:b]
        body = form[1:].split(None, 1)
        h = body[0].strip("()") if body else ""
        heads.add(h)
        if h in SCOPE_STANZAS:
            s, e = a + 1 + len(h), b - 1
            scopes.setdefault(d, {})[h] = (blanked[s:e], in_str[s:e])
    if heads & set(BUILD_STANZAS):
        build_dirs.append((d, sorted(heads & set(BUILD_STANZAS))))

if not build_dirs:
    print("ERROR: no build stanzas parsed from any dune file -- parser broken")
    sys.exit(2)

violations = []
for d, stanzas in build_dirs:
    parts = d.split("/") if d else []
    # Walk ancestors from the root down; the child component must survive each
    # ancestor's scope stanza.
    for i in range(len(parts)):
        ancestor, child = "/".join(parts[:i]), parts[i]
        for stanza, (text, mask) in scopes.get(ancestor, {}).items():
            where = f"{ancestor or '.'}/dune ({stanza} ...)"
            hit = admits(text, mask, child, where)
            # data_only_dirs: matching means the directory is data, not built.
            visible = hit if stanza == "dirs" else not hit
            if not visible:
                violations.append(
                    (d, stanzas, ancestor or ".", stanza, text.strip(), child)
                )

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
