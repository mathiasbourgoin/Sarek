#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Red-path test for scripts/check-cited-paths-exist.sh.
#
# WHY A .test.sh AND NOT A prove-red-spec BLOCK. prove-red.sh builds its scratch
# tree with `tempfile.mkdtemp` — a bare directory with no `.git`. The subject
# resolves citations against `git ls-files` on purpose (tracked, not merely
# present, is the whole point: `roster/` and `briefs/` exist on a workstation and
# in no clone), so in that scratch it would exit 2 and the mandatory green
# baseline could never be established. This harness therefore builds its own
# throwaway git repository per case, which is the only way to exercise the
# tracked-vs-present distinction at all.
#
# Each case asserts an exit code AND a message, because "went red" and "went red
# for the reason claimed" are different observations.

set -uo pipefail

SUBJECT="$(cd "$(dirname "$0")" && pwd)/check-cited-paths-exist.sh"
[ -x "$SUBJECT" ] || {
  echo "FAIL: subject not executable: $SUBJECT" >&2
  exit 2
}

pass=0
fail=0

# Build a minimal tracked repo: one .ml file whose comment/body we vary.
# $1 = file body. Echoes the repo path.
mkfixture() {
  local body="$1" d
  d="$(mktemp -d)"
  (
    cd "$d" || exit 2
    git init --quiet .
    git config user.email t@example.invalid
    git config user.name t
    mkdir -p sub
    printf '%s' "$body" >sub/Thing.ml
    printf 'let real = 1\n' >sub/Existing.ml
    git add -A
    git commit --quiet -m fixture
  ) >/dev/null 2>&1
  printf '%s' "$d"
}

# $1 = case name, $2 = expected exit, $3 = expected message substring
# ("" = no message requirement), $4 = file body
check() {
  local name="$1" want="$2" msg="$3" body="$4" d out code
  d="$(mkfixture "$body")"
  out="$(cd "$d" && bash "$SUBJECT" 2>&1)"
  code=$?
  rm -rf "$d"
  if [ "$code" != "$want" ]; then
    echo "FAIL $name: exit $code, wanted $want"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  if [ -n "$msg" ] && ! printf '%s' "$out" | grep -qF "$msg"; then
    echo "FAIL $name: exit $want as wanted, but message did not contain '$msg'"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  echo "PASS $name (exit $code)"
  pass=$((pass + 1))
}

# --- positive control -------------------------------------------------------
# A green baseline first. Without it, every red below is unfalsifiable: a subject
# that always exits 1 would satisfy the whole rest of this file.
check "green: a citation that resolves" 0 "OK" \
  '(* See sub/Existing.ml for the real thing. *)
let x = 1
'

# --- the defect this gate exists for ---------------------------------------
check "red: citation into an unpublished directory" 1 "is not a tracked file" \
  '(* See roster/gone/L99-note.md for the design. *)
let x = 1
'

# ocamlformat breaks long comments mid-path, and one of the four roster/
# citations that motivated the gate was written exactly this way. A
# line-oriented scan sees no `.md` on either line and reports nothing.
check "red: the same citation WRAPPED across comment lines" 1 "is not a tracked file" \
  '(* See roster/gone/L99-
 * note.md for the design. *)
let x = 1
'

# --- the boundary CodeRabbit raised on PR #387 ------------------------------
# The identical path, in a string literal instead of a comment. Rows 2 and this
# one are the discriminating pair: same bytes, different syntactic position,
# opposite verdicts. If the comment scanner had simply blanked everything, row 2
# would be green and this gate would be inert.
check "green: the same path inside a STRING LITERAL is not a citation" 0 "OK" \
  'let fixture = "roster/gone/L99-note.md"
let x = 1
'

# A path inside a comment is comment text even when quoted — odoc code refs
# legitimately read ["some/path.ml"], and those must still be checked.
check "red: a path quoted INSIDE a comment is still a citation" 1 "is not a tracked file" \
  '(* See ["roster/gone/L99-note.md"] for the design. *)
let x = 1
'

# --- documentation placeholders are not citations --------------------------
check "green: path/to/ placeholder" 0 "OK" \
  '(* Usage: put it at path/to/Thing.ml and go. *)
let x = 1
'

# --- fails closed ----------------------------------------------------------
# Not a git tree at all. Exit 2, never 0: a check whose inputs are unavailable
# must refuse rather than report success — the vacuous-green failure mode this
# repo has been bitten by repeatedly.
outside="$(mktemp -d)"
out="$(cd "$outside" && bash "$SUBJECT" 2>&1)"
code=$?
rm -rf "$outside"
if [ "$code" = 2 ]; then
  echo "PASS red: outside a git tree (exit 2)"
  pass=$((pass + 1))
else
  echo "FAIL red: outside a git tree: exit $code, wanted 2"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "check-cited-paths-exist.test.sh: $pass passed, $fail FAILED"
  exit 1
fi
echo "check-cited-paths-exist.test.sh: all $pass cases passed"
exit 0
