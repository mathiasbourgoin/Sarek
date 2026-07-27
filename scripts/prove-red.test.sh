#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Covering test for scripts/prove-red.sh (backlog-151).
#
# prove-red.sh exists to catch checkers that cannot fail, which makes "prove-red
# itself cannot fail" the single most expensive thing that could go wrong here.
# So this asserts the four ways it could lie -- crediting an immune checker,
# crediting a checker that is red on arrival, reporting on a set it did not
# find, and treating `expect-message` as decoration -- plus every refusal in its
# spec parser.
#
# EXACT EXIT CODES, NOT `! cmd`. prove-red.sh's contract distinguishes 1 (a
# checker did not fail as declared -- a finding about the subject) from 2 (the
# mechanism could not produce evidence -- a finding about the declaration).
# An assertion that accepted "non-zero" would not notice the two being swapped,
# and that is not hypothetical: accepting non-zero where a code was specified is
# itself on the 2026-07-27 gate-vacuous list.
#
# Case 0 is the positive control. Without it every red below proves nothing --
# a prove-red.sh that exited 2 unconditionally would pass cases 1-17.
#
# Each case builds a synthetic root under a temp dir and points the REAL
# scripts/prove-red.sh at it with --root. Nothing here touches the working tree,
# and no case depends on the repository's own four subjects.
# ---------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$SCRIPT_DIR/prove-red.sh"
[ -x "$TOOL" ] || { echo "FAIL: $TOOL not found or not executable"; exit 2; }

TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/prove-red-test.XXXXXX")"
trap 'rm -rf "$TMPROOT"' EXIT

pass=0
fail=0

# newroot <name> -- an empty synthetic root with a scripts/ and a data/ input.
newroot() {
  local d="$TMPROOT/$1"
  mkdir -p "$d/scripts" "$d/data"
  printf 'MARKER-OK\n' > "$d/data/input.txt"
  echo "$d"
}

# gate_body <path> -- an honest little gate: exit 2 with "not found" when its
# input is absent, exit 1 with "does not carry MARKER-OK" when it is wrong,
# exit 0 with "OK: input is well-formed" otherwise.
gate_body() {
  cat > "$1" <<'GATE'
#!/usr/bin/env bash
set -euo pipefail
if [ ! -f data/input.txt ]; then
  echo "::error::data/input.txt not found"
  exit 2
fi
if ! grep -q MARKER-OK data/input.txt; then
  echo "::error::data/input.txt does not carry MARKER-OK"
  exit 1
fi
echo "OK: input is well-formed"
GATE
  chmod +x "$1"
}

# good_spec <path> -- the spec a correct subject carries. Appended as comments.
good_spec() {
  cat >> "$1" <<'SPEC'
# BEGIN prove-red-spec
# copy: scripts/gate.sh
# copy: data/input.txt
# invoke: scripts/gate.sh
# baseline-exit: 0
# baseline-message: OK: input is well-formed
#
# mutation: marker-removed
#   desc: input present but wrong
#   apply: printf 'nothing\n' > data/input.txt
#   expect-exit: 1
#   expect-message: does not carry MARKER-OK
#
# mutation: input-deleted
#   desc: input absent -- an environment mutation, not a source edit
#   apply: rm -f data/input.txt
#   expect-exit: 2
#   expect-message: not found
# END prove-red-spec
SPEC
}

# expect <desc> <root> <expect-subjects> <wanted-exit> [needle]
expect() {
  local desc="$1" root="$2" n="$3" want="$4" needle="${5:-}"
  local out got
  out="$("$TOOL" --root "$root" --expect-subjects "$n" 2>&1)"
  got=$?
  if [ "$got" != "$want" ]; then
    echo "  FAIL: $desc -- expected exit $want, got $got"
    printf '%s\n' "$out" | sed 's/^/        /'
    fail=$((fail + 1))
    return
  fi
  if [ -n "$needle" ] && ! printf '%s' "$out" | grep -qF -- "$needle"; then
    echo "  FAIL: $desc -- exit $got as expected, but output never mentioned '$needle'"
    printf '%s\n' "$out" | sed 's/^/        /'
    fail=$((fail + 1))
    return
  fi
  echo "  PASS: $desc (exit $got)"
  pass=$((pass + 1))
}

echo "prove-red.sh covering test"

# --- Case 0: positive control ------------------------------------------------
# A correct subject with two honest mutations. Without this, every red below is
# indistinguishable from a tool that always fails.
d="$(newroot case0)"
gate_body "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
expect "case0: an honest subject with working mutations is GREEN" \
  "$d" 1 0 "2 mutation(s)"

# --- Case 1: the immune checker ----------------------------------------------
# The whole point. A gate that prints its success message and exits 0 whatever
# happens to its world must be a FINDING (1), not an error (2) -- the tool
# worked perfectly, the subject is the problem.
d="$(newroot case1)"
cat > "$d/scripts/gate.sh" <<'GATE'
#!/usr/bin/env bash
echo "OK: input is well-formed"
exit 0
GATE
chmod +x "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
expect "case1: a checker immune to its own declared mutation is exit 1" \
  "$d" 1 1 "DID NOT FAIL"

# --- Case 2: red on arrival --------------------------------------------------
# Every mutation below such a subject "goes red" and none of those reds means
# anything. The positive control must stop the run before any is credited, and
# report it as a broken mechanism (2) rather than a finding (1).
d="$(newroot case2)"
cat > "$d/scripts/gate.sh" <<'GATE'
#!/usr/bin/env bash
echo "::error::data/input.txt does not carry MARKER-OK"
exit 1
GATE
chmod +x "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
expect "case2: a subject that is red on arrival is exit 2, and no mutation is credited" \
  "$d" 1 2 "BASELINE is not green"

# --- Case 3: a silent baseline -----------------------------------------------
# Exit 0 and nothing else. A positive control that checks only the exit code
# passes for a subject that did nothing and said nothing -- which is the shape
# of every gate on the vacuous list.
d="$(newroot case3)"
cat > "$d/scripts/gate.sh" <<'GATE'
#!/usr/bin/env bash
if [ ! -f data/input.txt ]; then exit 2; fi
grep -q MARKER-OK data/input.txt || exit 1
exit 0
GATE
chmod +x "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
expect "case3: baseline exits 0 but prints nothing -- exit 2" \
  "$d" 1 2 "never printed"

# --- Case 4: right failure, wrong code ---------------------------------------
# The gate does fail, with the promised words, under the wrong exit code. This
# is the sub-case P3 names: "non-zero" is not the contract.
d="$(newroot case4)"
cat > "$d/scripts/gate.sh" <<'GATE'
#!/usr/bin/env bash
set -euo pipefail
if [ ! -f data/input.txt ]; then
  echo "::error::data/input.txt not found"
  exit 1
fi
if ! grep -q MARKER-OK data/input.txt; then
  echo "::error::data/input.txt does not carry MARKER-OK"
  exit 1
fi
echo "OK: input is well-formed"
GATE
chmod +x "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
expect "case4: fails with the right message under the wrong exit code -- exit 1" \
  "$d" 1 1 "declared 2"

# --- Case 5: right code, wrong reason ----------------------------------------
# The gate exits 1 as declared, for something else entirely. Without the message
# half, `expect-message` is decoration and a subject can satisfy its declaration
# by failing for an unrelated reason.
d="$(newroot case5)"
gate_body "$d/scripts/gate.sh"
cat >> "$d/scripts/gate.sh" <<'SPEC'
# BEGIN prove-red-spec
# copy: scripts/gate.sh
# copy: data/input.txt
# invoke: scripts/gate.sh
# baseline-exit: 0
# baseline-message: OK: input is well-formed
#
# mutation: marker-removed
#   desc: input present but wrong
#   apply: printf 'nothing\n' > data/input.txt
#   expect-exit: 1
#   expect-message: a sentence this gate never prints
# END prove-red-spec
SPEC
expect "case5: exits as declared but with a different failure -- exit 1" \
  "$d" 1 1 "never mentioned"

# --- Case 6: a subject with no declared mutation -----------------------------
d="$(newroot case6)"
gate_body "$d/scripts/gate.sh"
cat >> "$d/scripts/gate.sh" <<'SPEC'
# BEGIN prove-red-spec
# copy: scripts/gate.sh
# copy: data/input.txt
# invoke: scripts/gate.sh
# baseline-exit: 0
# baseline-message: OK: input is well-formed
# END prove-red-spec
SPEC
expect "case6: a block declaring no mutation is exit 2" \
  "$d" 1 2 "declares no mutation"

# --- Case 7: a mistyped key --------------------------------------------------
# Silently ignoring it would leave a declaration that reads as executed and is
# not -- this file's entire subject, one level up.
d="$(newroot case7)"
gate_body "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
sed -i 's/^#   expect-message: not found$/#   expect-mesage: not found/' \
  "$d/scripts/gate.sh"
expect "case7: an unknown key is exit 2, not ignored" \
  "$d" 1 2 "unknown key"

# --- Case 8: duplicate mutation ids ------------------------------------------
d="$(newroot case8)"
gate_body "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
sed -i 's/^# mutation: input-deleted$/# mutation: marker-removed/' \
  "$d/scripts/gate.sh"
expect "case8: duplicate mutation ids are exit 2" \
  "$d" 1 2 "duplicate mutation id"

# --- Case 9: two blocks in one file ------------------------------------------
# Whichever one is authoritative is undefined, and a declaration can hide in the
# one nothing reads.
d="$(newroot case9)"
gate_body "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
expect "case9: two spec blocks in one subject is exit 2" \
  "$d" 1 2 "exactly one of each is allowed"

# --- Case 10: a mutation that mutates nothing --------------------------------
# The most dangerous shape available to this mechanism: the subject collects a
# red it did not earn, from a mutation that never happened.
d="$(newroot case10)"
gate_body "$d/scripts/gate.sh"
cat >> "$d/scripts/gate.sh" <<'SPEC'
# BEGIN prove-red-spec
# copy: scripts/gate.sh
# copy: data/input.txt
# invoke: scripts/gate.sh
# baseline-exit: 0
# baseline-message: OK: input is well-formed
#
# mutation: does-nothing
#   desc: an apply that succeeds and changes no file
#   apply: true
#   expect-exit: 1
#   expect-message: does not carry MARKER-OK
# END prove-red-spec
SPEC
expect "case10: an apply that changes no file is exit 2" \
  "$d" 1 2 "changed no file"

# --- Case 11: an apply that fails --------------------------------------------
d="$(newroot case11)"
gate_body "$d/scripts/gate.sh"
cat >> "$d/scripts/gate.sh" <<'SPEC'
# BEGIN prove-red-spec
# copy: scripts/gate.sh
# copy: data/input.txt
# invoke: scripts/gate.sh
# baseline-exit: 0
# baseline-message: OK: input is well-formed
#
# mutation: broken-apply
#   desc: the mutation itself does not run
#   apply: sed -i 's/x/y/' data/no-such-file.txt
#   expect-exit: 1
#   expect-message: does not carry MARKER-OK
# END prove-red-spec
SPEC
expect "case11: an apply that exits non-zero is exit 2" \
  "$d" 1 2 "script exited"

# --- Case 12: a copy path that does not exist --------------------------------
d="$(newroot case12)"
gate_body "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
sed -i 's|^# copy: data/input.txt$|# copy: data/absent.txt|' "$d/scripts/gate.sh"
expect "case12: a declared copy path that does not exist is exit 2" \
  "$d" 1 2 "does not exist under"

# --- Case 13: the subject count pin ------------------------------------------
# A scan that reports what it happened to find is complete about a set it chose.
d="$(newroot case13)"
gate_body "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
expect "case13: one subject found where two are pinned is exit 2" \
  "$d" 2 2 "found 1"

# --- Case 14: a mutation that asserts the subject did not notice -------------
d="$(newroot case14)"
gate_body "$d/scripts/gate.sh"
cat >> "$d/scripts/gate.sh" <<'SPEC'
# BEGIN prove-red-spec
# copy: scripts/gate.sh
# copy: data/input.txt
# invoke: scripts/gate.sh
# baseline-exit: 0
# baseline-message: OK: input is well-formed
#
# mutation: expects-green
#   desc: declares the baseline's exit code and the baseline's message
#   apply: printf 'nothing\n' > data/input.txt
#   expect-exit: 0
#   expect-message: OK: input is well-formed
# END prove-red-spec
SPEC
expect "case14: a mutation expecting the baseline verdict is exit 2" \
  "$d" 1 2 "did not notice"

# --- Case 15: expect-exit must be a number -----------------------------------
d="$(newroot case15)"
gate_body "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
sed -i 's/^#   expect-exit: 2$/#   expect-exit: nonzero/' "$d/scripts/gate.sh"
expect "case15: a non-integer expect-exit is exit 2" \
  "$d" 1 2 "not a contract"

# --- Case 16: covering tests are not subjects --------------------------------
# The exclusion is live, not merely intended: the fixture blocks inside THIS
# file would otherwise be discovered as real declarations.
d="$(newroot case16)"
gate_body "$d/scripts/gate.test.sh"
good_spec "$d/scripts/gate.test.sh"
expect "case16: a *.test.sh carrying a block is not discovered as a subject" \
  "$d" 1 2 "found 0"

# --- Case 18: an untracked copy target ---------------------------------------
# It exists here and is absent on a fresh clone, so the subject would be exit 2
# in CI while passing on the workstation that has the file. This is not a
# hypothetical: the first fixture log for scripts/test-suite-counts.sh was named
# `*.log`, which .gitignore swallows. The refusal only applies inside a git work
# tree, so this case makes one.
d="$(newroot case18)"
gate_body "$d/scripts/gate.sh"
good_spec "$d/scripts/gate.sh"
( cd "$d" && git init -q . && git config user.email t@t.t \
    && git config user.name t && printf 'data/\n' > .gitignore \
    && git add scripts/gate.sh .gitignore \
    && git commit -qm init ) >/dev/null 2>&1
expect "case18: a copy target git does not track is exit 2" \
  "$d" 1 2 "not tracked by git"

# --- Case 17: nothing to scan ------------------------------------------------
# An empty scan set turns this into a check that always passes.
d="$TMPROOT/case17"
mkdir -p "$d"
expect "case17: a root with no scripts/ and no ci/ is exit 2" \
  "$d" 0 2 "scan read nothing"

echo ""
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
