#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Red-path harness for check-ppx-construct-names.sh (backlog-193).
#
# The gate has TWO independent red shapes, because the defect it exists for had
# two: a construct name the PPX never had (`ktype`), and a construct name that is
# correct in the source but collapses to a different one on the user's terminal
# (`[@@sarek.type]` in a Format string prints `[@sarek.type]`). A harness that
# exercised one would leave the other free to come back.
#
# It also pins the three ways this gate could read green while checking nothing:
#   - a comment mentioning a bogus construct must NOT fire (case 3): if it did,
#     the maintainer's first reflex would be to weaken the gate, and the
#     comment-stripping is what makes it precise enough to keep.
#   - `[%s]` in a message must NOT fire (case 4): a gate that flags every printf
#     conversion in the tree is a gate that gets deleted.
#   - with no sarek/ppx sources the name table is empty, and an empty table would
#     make EVERY name unknown or (worse, in an earlier draft) every name known.
#     It must exit 2 (case 5), not decide.
#
# Each case runs in a THROWAWAY git repo: the gate reads `git ls-files`, so the
# fixtures must actually be tracked, and doing that in the real repo would mean
# mutating its index. Nothing here touches the working repository.
#
# Exit: 0 all cases behaved - 1 a case did not - 2 setup failure (fails closed).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
GATE_SRC="$REPO_ROOT/scripts/check-ppx-construct-names.sh"
[ -x "$GATE_SRC" ] || {
  echo "::error::$GATE_SRC not found or not executable" >&2; exit 2; }
command -v git >/dev/null 2>&1 || { echo "::error::git required" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "::error::python3 required" >&2; exit 2; }

pass=0
fail=0

# A minimal PPX name table: the gate learns the legal construct names from the
# string literals under sarek/ppx/, so this one file makes "sarek.type" and
# "kernel" legal and nothing else.
PPX_TABLE='let a = Attribute.declare "sarek.type" ctx pat ()
let b = Extension.declare "kernel" ctx pat f'

#   $1 name  $2 expected exit  $3 expected substring in output
# Remaining args: alternating <relative path> <content> pairs.
check() {
  local name="$1" want_exit="$2" want_text="$3"; shift 3
  local tmp out got
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/ppx-construct-test.XXXXXX")" \
    || { echo "::error::mktemp failed" >&2; exit 2; }

  mkdir -p "$tmp/scripts"
  cp "$GATE_SRC" "$tmp/scripts/check-ppx-construct-names.sh"

  while [ "$#" -gt 0 ]; do
    local path="$1" content="$2"; shift 2
    mkdir -p "$tmp/$(dirname "$path")"
    printf '%s\n' "$content" > "$tmp/$path"
  done

  (
    cd "$tmp" || exit 2
    git init -q . && git add -A && \
      git -c user.email=t@t -c user.name=t commit -qm f
  ) >/dev/null 2>&1 || { echo "::error::could not build fixture repo" >&2
                         rm -rf "$tmp"; exit 2; }

  out="$("$tmp/scripts/check-ppx-construct-names.sh" 2>&1)"
  got=$?
  rm -rf "$tmp"

  if [ "$got" != "$want_exit" ]; then
    echo "  FAIL: $name -- expected exit $want_exit, got $got"
    printf '%s\n' "$out" | sed 's/^/        /'
    fail=$((fail + 1)); return
  fi
  if [ -n "$want_text" ] && ! printf '%s' "$out" | grep -qF -- "$want_text"; then
    echo "  FAIL: $name -- exit $got was right but the output never said"
    echo "        '$want_text'; a gate that fires for an unstated reason is not"
    echo "        evidence. Got:"
    printf '%s\n' "$out" | sed 's/^/        /'
    fail=$((fail + 1)); return
  fi
  echo "  PASS: $name (exit $got)"
  pass=$((pass + 1))
}

echo "=== check-ppx-construct-names.sh red-path harness ==="

# --- case 0: positive control -------------------------------------------------
# Both legal spellings, one Format-based and one not. If this is not green the
# red cases below prove nothing: "went red" and "is always red" would be the
# same observation.
check "case 0 (positive control): correct spellings pass" 0 "OK --" \
  "sarek/ppx/Sarek_ppx.ml" "$PPX_TABLE" \
  "sarek/core/A.ml" 'let f () =
  Location.raise_errorf ~loc "declare it with [@@@@sarek.type] please"
let g () = Printf.sprintf "declare it with [@@sarek.type] please"'

# --- case 1: the filed defect -------------------------------------------------
check "case 1: a message names a construct the PPX never had" 1 \
  "no 'ktype' is declared" \
  "sarek/ppx/Sarek_ppx.ml" "$PPX_TABLE" \
  "sarek/core/A.ml" 'let f () =
  Location.raise_errorf ~loc "register it with [%%ktype] first"'

# --- case 2: the rendering lie ------------------------------------------------
check "case 2: a Format message whose construct collapses on screen" 1 \
  "reaches the user as [@sarek.type]" \
  "sarek/ppx/Sarek_ppx.ml" "$PPX_TABLE" \
  "sarek/core/A.ml" 'let f () =
  Location.raise_errorf ~loc "declare it with [@@sarek.type] first"'

# --- case 3: comments are not claims -----------------------------------------
check "case 3: a bogus construct in a COMMENT does not fire" 0 "OK --" \
  "sarek/ppx/Sarek_ppx.ml" "$PPX_TABLE" \
  "sarek/core/A.ml" '(* historically this was [%%ktype], renamed long ago *)
let f () = Location.raise_errorf ~loc "no advice here"'

# --- case 4: printf conversions are not extension points ----------------------
check "case 4: [%s] in a message does not fire" 0 "OK --" \
  "sarek/ppx/Sarek_ppx.ml" "$PPX_TABLE" \
  "sarek/core/A.ml" 'let f () =
  Location.raise_errorf ~loc "bad thing [%s] at [%d] and [%Ld]" a b c'

# --- case 5: fails closed with no name table ---------------------------------
check "case 5: no sarek/ppx sources exits 2 rather than deciding" 2 \
  "cannot build the PPX name table" \
  "sarek/core/A.ml" 'let f () = Location.raise_errorf ~loc "[%%ktype]"'

# --- case 6: the test tree is out of scope by design --------------------------
# A negative test must be free to name the spelling it proves is REFUSED.
check "case 6: a bogus construct under sarek/tests/ does not fire" 0 "OK --" \
  "sarek/ppx/Sarek_ppx.ml" "$PPX_TABLE" \
  "sarek/tests/negative/t.ml" 'let f () =
  Location.raise_errorf ~loc "register it with [%%ktype]"'

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
echo "OK: check-ppx-construct-names.sh fires on both defect shapes and on neither"
echo "    of the two shapes that would have made it unkeepable."
exit 0
