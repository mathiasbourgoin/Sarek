#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Covering test for scripts/check-ppxlib-floor.sh.
#
# The guard exists because a declared dependency bound was an ASSERTION that
# nothing compared against the code. A guard for that which cannot itself fail
# would be the same defect one level up, so every rule below is exercised on its
# red path and the green baseline is exercised too.
#
# Each case builds a synthetic tree with the four files the guard reads, runs
# the REAL guard against it, and asserts the exit code:
#   1  green baseline                                          -> 0
#   2  dune-project and sarek.opam disagree                    -> 1
#   3  the KB records a different floor                        -> 1
#   4  the KB records no floor at all                          -> 1
#   5  the code matches Ptyp_open under too low a floor        -> 1
#   6  the code matches the parameterised Pexp_function, ditto -> 1
#   7  no tabled constructor appears anywhere (vacuous rule 3) -> 2
#   8  two ppxlib bounds in dune-project (parser refuses)      -> 2
set -uo pipefail

GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check-ppxlib-floor.sh"
[ -x "$GUARD" ] || { echo "FAIL: $GUARD not found or not executable"; exit 2; }

TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/ppxlib-floor-test.XXXXXX")"
trap 'rm -rf "$TMPROOT"' EXIT

pass=0
fail=0

# make_tree <name> <dune-bound> <opam-bound> <kb-floor-line> <ppx-source>
# The guard resolves its paths from its own location, so each synthetic tree
# needs a scripts/ dir holding a copy of the guard.
make_tree() {
  local name="$1" dpb="$2" opb="$3" kbline="$4" src="$5"
  local d="$TMPROOT/$name"
  mkdir -p "$d/scripts" "$d/kb/sarek/ppx" "$d/sarek/ppx"
  cp "$GUARD" "$d/scripts/"
  printf '(lang dune 3.15)\n (depends\n  (ocaml (>= 5.4.0))\n%s\n  (ctypes (>= 0.24.0)))\n' "$dpb" > "$d/dune-project"
  printf 'opam-version: "2.0"\ndepends: [\n  "ocaml" {>= "5.4.0"}\n%s\n]\n' "$opb" > "$d/sarek.opam"
  printf '# parser\n\n%s\n' "$kbline" > "$d/kb/sarek/ppx/parser.md"
  printf '%s\n' "$src" > "$d/sarek/ppx/Sarek_unsupported.ml"
  printf '%s\n' "$d"
}

expect() {
  local desc="$1" dir="$2" want="$3"
  local out got
  out="$("$dir/scripts/check-ppxlib-floor.sh" 2>&1)"
  got=$?
  if [ "$got" = "$want" ]; then
    printf '  PASS  %-58s exit=%s\n' "$desc" "$got"
    pass=$((pass + 1))
  else
    printf '  FAIL  %-58s exit=%s (wanted %s)\n' "$desc" "$got" "$want"
    printf '%s\n' "$out" | sed 's/^/        /'
    fail=$((fail + 1))
  fi
}

DP37='  (ppxlib (>= 0.37.0))'
DP22='  (ppxlib (>= 0.22.0))'
OP37='  "ppxlib" {>= "0.37.0"}'
OP22='  "ppxlib" {>= "0.22.0"}'
KB37='Declared ppxlib floor: `0.37.0`.'
KB22='Declared ppxlib floor: `0.22.0`.'
SRC_BOTH='let f d = match d with Ptyp_open _ -> "x" | _ -> "y"
let g e = match e with Pexp_function (_, _, _) -> 1 | _ -> 0'
SRC_OPEN_ONLY='let f d = match d with Ptyp_open _ -> "x" | _ -> "y"'
SRC_FUNC_ONLY='let g e = match e with Pexp_function (_, _, _) -> 1 | _ -> 0'
SRC_NEITHER='let f d = match d with Ptyp_var _ -> "x" | _ -> "y"'

echo "check-ppxlib-floor.test.sh"

expect "1 green baseline" \
  "$(make_tree t1 "$DP37" "$OP37" "$KB37" "$SRC_BOTH")" 0

expect "2 dune-project and sarek.opam disagree" \
  "$(make_tree t2 "$DP37" "$OP22" "$KB37" "$SRC_BOTH")" 1

expect "3 KB records a different floor" \
  "$(make_tree t3 "$DP37" "$OP37" "$KB22" "$SRC_BOTH")" 1

expect "4 KB records no floor at all" \
  "$(make_tree t4 "$DP37" "$OP37" "nothing recorded here" "$SRC_BOTH")" 1

expect "5 Ptyp_open matched under too low a floor" \
  "$(make_tree t5 "$DP22" "$OP22" "$KB22" "$SRC_OPEN_ONLY")" 1

expect "6 parameterised Pexp_function under too low a floor" \
  "$(make_tree t6 "$DP22" "$OP22" "$KB22" "$SRC_FUNC_ONLY")" 1

expect "7 no tabled constructor anywhere (rule 3 vacuous)" \
  "$(make_tree t7 "$DP37" "$OP37" "$KB37" "$SRC_NEITHER")" 2

t8="$(make_tree t8 "$DP37" "$OP37" "$KB37" "$SRC_BOTH")"
printf '  (ppxlib (>= 0.38.0))\n' >> "$t8/dune-project"
expect "8 two ppxlib bounds in dune-project (refuses to guess)" "$t8" 2

echo
if [ "$fail" -ne 0 ]; then
  echo "check-ppxlib-floor.test.sh: $fail failing case(s), $pass passing"
  exit 1
fi
echo "check-ppxlib-floor.test.sh: OK — $pass case(s), every rule proven red and the baseline green"
