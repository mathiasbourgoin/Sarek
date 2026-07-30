#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Every repo-relative path cited in a source comment must resolve to a TRACKED
# file.
#
# WHY THIS EXISTS. Four source files cited design notes under `roster/` — a
# working directory that was never published, so no clone has ever had it:
#
#   sarek/ppx/Sarek_tag_erasure.ml:13
#     "See roster/ptx-limits-campaign/L14-static-tag-erasure.md."
#
# A reader following that pointer finds nothing, and cannot tell from the
# comment whether the file was renamed, deleted, or never shipped. The citations
# survived a doc sweep precisely because nothing reads comments. Running this
# gate for the first time found six MORE of the same shape, under `briefs/`,
# in files nobody had connected to the four.
#
# TRACKED, NOT PRESENT. Resolution is against `git ls-files`, never the
# filesystem. That distinction is the whole point: `roster/` and `briefs/` exist
# on the machine that wrote these comments and in no clone, so a filesystem
# check would pass here AND pass in CI while every reader hit a dead end.
#
# THREE THINGS IT DELIBERATELY DOES NOT FLAG, each measured against this tree
# (26 raw candidates -> 10 real findings, 0 false positives):
#
#  1. Documentation placeholders — `path/to/Sarek_gemm.ml`, `.../Sarek_df64.ml`.
#     Five usage examples write a path shape, not a path.
#  2. Ancestor-relative references — `theories/PtxLayout.v` cited from
#     formal/codegen-ptx/test/ means formal/codegen-ptx/theories/PtxLayout.v.
#     A citation resolves from the repo root or ANY ancestor of the citing file,
#     which is how a reader resolves it.
#  3. PTX mnemonics that lex like paths — `ld/st.shared` is not a shell script.
#     The extension must end the token; without that guard `st.shared` matched
#     `st.sh` and three prose lines read as dangling citations.
#
# COMMENTS ONLY. The scan sees only OCaml comment text; string literals and
# code are blanked out first (newlines preserved, so reported line numbers stay
# real). Without that, a legitimate string such as
#
#     let fixture = "fixtures/missing.md" in ...
#
# reads as a dangling citation and fails CI over a path that was never a
# citation at all — the gate blocking a correct change. Raised by CodeRabbit on
# PR #387; the string-literal boundary is pinned by a prove-red case below.
#
# WRAPPED CITATIONS ARE UNWRAPPED FIRST. ocamlformat breaks a long comment
# mid-path, and one of the four roster/ citations that motivated this gate was
# written that way:
#
#     * flows through the queue. See roster/ptx-limits-campaign/L16-dynamic-
#     * parallelism.md for the CDP-vs-worklist rationale
#
# A line-oriented scan sees no `.md` on either line and reports nothing — the
# gate would have missed the very citation it exists to catch. Comment
# continuations are therefore joined before matching.
#
# An earlier version of this gate had a pre-filter requiring the first path
# component to name a tracked directory. It is gone: it made the gate blind to
# a citation whose whole directory is missing — the roster/ case, i.e. exactly
# what the gate is for. The check now stands on the three exclusions above.
#
# Exit codes: 0 = every citation resolves, 1 = a citation dangles,
# 2 = cannot run (not a git tree, no sources, no python3) — fail closed rather
# than scan nothing and report success.

set -uo pipefail

git rev-parse --show-toplevel >/dev/null 2>&1 || {
  echo "check-cited-paths-exist: not inside a git work tree" >&2
  exit 2
}
cd "$(git rev-parse --show-toplevel)" || exit 2

command -v python3 >/dev/null 2>&1 || {
  echo "check-cited-paths-exist: python3 not found" >&2
  exit 2
}

python3 - <<'PY'
import os, re, subprocess, sys


def ls(*args):
    out = subprocess.run(["git", "ls-files", *args], capture_output=True, text=True)
    if out.returncode != 0:
        print("check-cited-paths-exist: git ls-files failed", file=sys.stderr)
        sys.exit(2)
    return [l for l in out.stdout.splitlines() if l]


sources = ls("*.ml", "*.mli")
if not sources:
    print("check-cited-paths-exist: no tracked .ml/.mli files found", file=sys.stderr)
    sys.exit(2)
tracked = set(ls())

# The trailing (?![A-Za-z0-9]) is load-bearing: without it "ld/st.shared" in a
# PTX comment matches "st.sh" and reads as a missing shell script.
CITATION = re.compile(
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+\.(?:ml|mli|v|md|sh|json|ya?ml)(?![A-Za-z0-9])"
)
PLACEHOLDER = re.compile(r"(?:^|/)(?:path/to|\.\.\.)/")


def comments_only(src):
    """Blank everything that is not inside an OCaml (* ... *) comment.

    Newlines are preserved so line numbers in findings match the real file.
    OCaml comments nest, and a string literal may contain "(*" or "*)", so this
    is a small state machine rather than a regex: tracking string state is what
    keeps a comment-looking substring inside a literal from opening a comment,
    and tracking depth is what stops a nested close from ending the outer one.
    """
    out = []
    i, n = 0, len(src)
    depth = 0          # comment nesting depth; 0 = not in a comment
    in_string = False  # inside "..." (only tracked outside comments)
    while i < n:
        c = src[i]
        two = src[i : i + 2]
        if depth == 0 and in_string:
            # Blank the literal's contents; an escaped quote does not close it.
            if c == "\\" and i + 1 < n:
                out.append("\n" if src[i + 1] == "\n" else " ")
                out.append(" ")
                i += 2
                continue
            if c == '"':
                in_string = False
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if depth == 0 and c == '"':
            in_string = True
            out.append(" ")
            i += 1
            continue
        if two == "(*":
            depth += 1
            out.append("  ")
            i += 2
            continue
        if two == "*)" and depth > 0:
            depth -= 1
            out.append("  ")
            i += 2
            continue
        if depth > 0:
            # Inside a comment: keep the text. A string literal inside a comment
            # is still comment text, and citations legitimately appear in [".."].
            out.append(c)
        else:
            out.append("\n" if c == "\n" else " ")
        i += 1
    return "".join(out)


def resolves(citation, citing_file):
    """A citation resolves from the repo root or any ancestor of its file."""
    if citation in tracked:
        return True
    d = os.path.dirname(citing_file)
    while True:
        if os.path.normpath(os.path.join(d, citation)) in tracked:
            return True
        if not d:
            return False
        d = os.path.dirname(d)


dangling = []
for f in sources:
    try:
        raw = open(f, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    # Comments only, THEN join a comment line broken mid-token: a trailing "-"
    # followed by the next line's comment prefix is one path, not two. Line
    # numbers are reported from the ORIGINAL text, so a finding still points at
    # a real line.
    text = re.sub(r"-\n\s*\*?\s*", "-", comments_only(raw))
    for c in sorted(set(CITATION.findall(text))):
        if "://" in c or "github.com/" in c:
            continue
        if PLACEHOLDER.search(c) or c.startswith("..."):
            continue
        if resolves(c, f):
            continue
        # Report against the ORIGINAL text so the line number is real. A
        # wrapped citation matches no single line, so fall back to the line
        # carrying its first segment.
        lines = raw.splitlines()
        lineno = next(
            (i for i, line in enumerate(lines, 1) if c in line), None
        ) or next(
            (i for i, line in enumerate(lines, 1) if c.split("/")[0] + "/" in line),
            None,
        )
        dangling.append((f, lineno, c))

for f, lineno, c in dangling:
    print(f"{f}:{lineno if lineno else '?'}: cites '{c}', which is not a tracked file")

if dangling:
    print()
    print(f"{len(dangling)} source comment(s) point at a path no reader can open.")
    print("Either fix the path, or say in the comment that the document is not")
    print("part of this repository so the pointer is not a dead end.")
    sys.exit(1)

print(
    "check-cited-paths-exist: OK — every repo-relative path cited in "
    f"{len(sources)} tracked .ml/.mli files resolves to a tracked file"
)
PY
