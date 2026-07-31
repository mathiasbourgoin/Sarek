#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Refuse a declared ppxlib lower bound that the PPX's own code has outgrown,
# and refuse the three places that record it drifting apart.
#
# WHY THIS EXISTS. backlog-192 walked the ppxlib Parsetree and made the parser's
# refusal tables match `Ptyp_open` and the parameterised `Pexp_function`, both of
# which need the OCaml-5.2-shaped AST. `dune-project` went on declaring
# `(ppxlib (>= 0.22.0))` and `sarek.opam` went on publishing
# `"ppxlib" {>= "0.22.0"}`, so the package advertised a floor its code could not
# build at. Nothing noticed, because nothing compared the number against the
# code -- it was an assertion, not a check. Measured on the octez-setup switch
# (ppxlib 0.35.0, OCaml 5.3.0, dune 3.23.0): the commit before that walk built
# `sarek_frontend` at exit 0 and the commit after it failed at exit 1 with
#     Error: ... There is no constructor "Ptyp_open" within type "core_type_desc"
#     Error (warning 8): ... not matched: Pexp_fun (_, _, _, _)
#
# THREE RULES, and only the third of them is non-circular.
#
#  1. `dune-project` and `sarek.opam` must declare the SAME bound. `sarek.opam`
#     is generated from `dune-project`, but it is also TRACKED, and it is the
#     file opam clients actually read -- so a dune-project bump that is never
#     regenerated ships the old number to users while the repo looks fixed.
#
#  2. The floor recorded in kb/sarek/ppx/parser.md must equal the declared one,
#     so the prose cannot go stale behind the metadata.
#
#  3. THE ONE THAT TIES THE NUMBER TO THE CODE. Below is a table of Parsetree
#     constructors that do not exist in every ppxlib, each with the first
#     release that provides it and the command that established that. If the
#     PPX matches one of them, the declared floor must be at least that
#     release. This is what fails loudly the next time somebody reaches for a
#     constructor the declared floor does not have.
#
# WHAT THIS GATE DOES NOT DO. It cannot discover the introduction version of a
# constructor that is not in the table -- it only enforces the ones written
# down. A CI lane building at the minimum ppxlib is the only thing that would
# catch a brand-new constructor automatically, and that was judged too costly
# for the return here: opam's own solver already makes the floor unreachable
# from below, because this package requires `ocaml >= 5.4.0` and every ppxlib
# before 0.37.0 caps itself at `ocaml < 5.4.0`. The rationale is recorded in the
# KB beside the floor.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

DUNE_PROJECT=dune-project
OPAM_FILE=sarek.opam
KB_FILE=kb/sarek/ppx/parser.md
PPX_DIR=sarek/ppx

fail=0
note() { printf '%s\n' "$*" >&2; }
bad() { note "check-ppxlib-floor: $*"; fail=1; }

for f in "$DUNE_PROJECT" "$OPAM_FILE" "$KB_FILE"; do
  [ -f "$f" ] || { note "check-ppxlib-floor: missing $f"; exit 2; }
done

# --- rule 1: the two declarations agree -------------------------------------
dp_bound=$(sed -nE 's/^[[:space:]]*\(ppxlib \(>= ([0-9][0-9.]*)\)\).*/\1/p' "$DUNE_PROJECT")
op_bound=$(sed -nE 's/^[[:space:]]*"ppxlib" \{>= "([0-9][0-9.]*)"\}.*/\1/p' "$OPAM_FILE")

# Exactly one of each, or the regexes are reading something they do not
# understand and a silent empty match would read as agreement.
[ "$(printf '%s\n' "$dp_bound" | grep -c .)" = 1 ] ||
  { note "check-ppxlib-floor: expected exactly one ppxlib bound in $DUNE_PROJECT, found $(printf '%s\n' "$dp_bound" | grep -c .)"; exit 2; }
[ "$(printf '%s\n' "$op_bound" | grep -c .)" = 1 ] ||
  { note "check-ppxlib-floor: expected exactly one ppxlib bound in $OPAM_FILE, found $(printf '%s\n' "$op_bound" | grep -c .)"; exit 2; }

if [ "$dp_bound" != "$op_bound" ]; then
  bad "$DUNE_PROJECT declares ppxlib >= $dp_bound but $OPAM_FILE publishes >= $op_bound."
  bad "  $OPAM_FILE is generated from $DUNE_PROJECT AND tracked, and it is the one"
  bad "  opam clients read. Regenerate it, or apply the same line by hand."
fi

# --- rule 2: the KB records the same floor ----------------------------------
# ALL matches, not the first. This used `| head -1`, which meant a KB carrying a
# correct first floor and a stale second one satisfied the guard: the second
# declaration was never looked at, so the tree could hold two contradictory
# floors and this gate would report OK. That is the third gate-that-cannot-fail
# found in this PR, and the second in this gate. Caught by CodeRabbit on #398.
#
# The `sort -V | head -1` inside version_ge below is NOT the same shape and must
# not be "fixed": it is a min-of-exactly-two, where taking the first element is
# the whole point.
kb_bounds=$(sed -nE 's/.*[Dd]eclared ppxlib floor: `([0-9][0-9.]*)`.*/\1/p' "$KB_FILE")
kb_count=$(printf '%s\n' "$kb_bounds" | grep -c . || true)
if [ "$kb_count" -eq 0 ]; then
  bad "$KB_FILE records no floor. It must carry a line matching"
  bad "  'Declared ppxlib floor: \`<version>\`' so the measurement has a home."
elif [ "$kb_count" -gt 1 ]; then
  bad "$KB_FILE records $kb_count floors: $(printf '%s' "$kb_bounds" | tr '\n' ' ')."
  bad "  Exactly one is allowed. Two declarations cannot both be the floor, and"
  bad "  agreeing with whichever comes first is how a stale one survives."
else
  kb_bound="$kb_bounds"
  if [ "$kb_bound" != "$dp_bound" ]; then
    bad "$KB_FILE records a floor of $kb_bound but $DUNE_PROJECT declares $dp_bound."
  fi
fi

# --- rule 3: the code must not have outgrown the floor ----------------------
# constructor <TAB> first ppxlib providing it <TAB> how that was established
#
# Ptyp_open: absent from ppxlib 0.35.0 and present in 0.36.0. Established with
#   opam source ppxlib.0.35.0 --dir=/tmp/p35 && grep -c Ptyp_open /tmp/p35/ast/ast.ml   -> 0
#   opam source ppxlib.0.36.0 --dir=/tmp/p36 && grep -c Ptyp_open /tmp/p36/ast/ast.ml   -> 14
# Pexp_function with three arguments (params, constraint, body) REPLACED the
#   four-argument Pexp_fun in the same release:
#   grep -n '| Pexp_fun of' /tmp/p35/ast/ast.ml  -> 373: Pexp_fun of arg_label * ...
#   grep -n '| Pexp_fun of' /tmp/p36/ast/ast.ml  -> (no match)
CONSTRUCTOR_FLOORS="Ptyp_open	0.36.0
Pexp_function	0.36.0"

# Compare two dotted versions: 0 if $1 >= $2.
version_ge() {
  [ "$1" = "$2" ] && return 0
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]
}

used_any=0
while IFS=$'\t' read -r ctor floor _; do
  [ -n "$ctor" ] || continue
  # `grep -rqw` on the bare name, over *.ml and *.mli. This DOES match inside
  # an OCaml comment, and an earlier revision of this comment claimed it did
  # not -- an inaccurate claim beside a check, which is the defect class this
  # repository tracks under KB-GATE-PPX-CONSTRUCT-NAMES, committed in the gate
  # written to prevent the metadata version of it. Caught by CodeRabbit on
  # PR #398.
  #
  # The breadth is kept deliberately, and the comment case is the reason to keep
  # it rather than an accident to tolerate: a constructor named only in a
  # comment, or commented out, holds the declared floor UP. That is the safe
  # direction -- a false positive here can only ever demand a HIGHER floor,
  # while a false negative would let the drift this gate exists to catch
  # through. Narrowing it to code would trade a harmless over-demand for a
  # silent miss.
  if grep -rqw "$ctor" "$PPX_DIR" --include='*.ml' --include='*.mli'; then
    used_any=1
    if ! version_ge "$dp_bound" "$floor"; then
      bad "$PPX_DIR matches \`$ctor\`, which first exists in ppxlib $floor,"
      bad "  but the declared floor is $dp_bound. Raise the bound in $DUNE_PROJECT"
      bad "  (and in $OPAM_FILE), or stop matching the constructor."
    fi
  fi
done <<< "$CONSTRUCTOR_FLOORS"

# A table that matches nothing checks nothing, and would read as a pass forever
# after a rename. This is the gate refusing to be vacuous about itself.
if [ "$used_any" -eq 0 ]; then
  note "check-ppxlib-floor: none of the tabled constructors appears in $PPX_DIR."
  note "  Rule 3 checked nothing. Either the PPX stopped matching them (delete the"
  note "  rows) or the table has gone stale (fix the names)."
  exit 2
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi

echo "check-ppxlib-floor: OK — ppxlib floor $dp_bound is declared identically in $DUNE_PROJECT and $OPAM_FILE, recorded in $KB_FILE, and is at or above every tabled constructor's introducing release."
