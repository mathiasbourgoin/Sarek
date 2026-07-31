#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Red-path test for scripts/check-opam-clean.sh.
#
# WHY THIS EXISTS, AND WHY IT ASSERTS TEXT RATHER THAN EXIT CODES. The subject
# reads two command statuses as data -- `cmp` between the two `make opam` runs,
# and `git diff --exit-code` against the tracked copies. Both were first written
# as a bare command followed by `case $?`. The subject runs under `set -e`, so
# the shell terminated AT the command: the `case` never ran, every diagnostic on
# both paths was DEAD CODE, and the gate exited with a bare status. That is the
# exact failure backlog-213 set out to remove from the `make opam` path --
# "exited through set -e with no message, indistinguishable from a dirty tree"
# -- reintroduced inside the code written to remove it. CodeRabbit found it on
# PR #399; three review passes and the author's own positive controls did not.
#
# The reason they did not is the whole design of this file. Those controls
# asserted EXIT CODES, and an unreached diagnostic still produces a plausible
# one. Measured against the pre-fix form with these same stubs:
#
#   forced condition      pre-fix exit   pre-fix message
#   cmp differs           1              MISSING     <- code looked correct
#   cmp exits 2           2              MISSING     <- code looked correct
#   git diff exits 128    128            MISSING     <- leaked git's own status
#
# TWO OF THREE HAD THE RIGHT EXIT CODE WITH A DEAD MESSAGE. No exit-code
# assertion could ever have caught this, so every red case below asserts the
# message the subject promises, and a case that goes red without printing its
# message is a FAILURE here, not a pass.
#
# WHY A .test.sh AND NOT A prove-red-spec BLOCK. prove-red.sh copies a declared
# file list into a bare `mktemp` scratch with no git and no toolchain, and
# requires a green baseline there. The subject shells out to `make opam` (a full
# `dune build @install`), resolves its own PROJECT_ROOT, reads `git ls-files`,
# and diffs against committed blobs -- none of which exist in that scratch. This
# harness instead runs the REAL subject against the REAL repository and replaces
# only the external commands whose status it must observe, via PATH stubs. That
# keeps the subject's own control flow -- the part that was broken -- under test.
#
# WHAT IT DOES NOT COVER. It never lets a real `make opam` run, so it says
# nothing about whether the .opam files actually converge; that is the subject's
# job in `make test-all`, not this file's. It stubs `dune` too, so it runs in
# CI's fast job with no switch -- meaning the version string in the verdicts it
# checks is a stub's, not a real dune's.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SUBJECT="$HERE/check-opam-clean.sh"
ROOT="$(cd "$HERE/.." && pwd)"

[ -x "$SUBJECT" ] || { echo "FAIL: subject not executable: $SUBJECT" >&2; exit 2; }
[ -n "$ROOT" ] && [ -d "$ROOT" ] ||
  { echo "FAIL: project root did not resolve: '$ROOT'" >&2; exit 2; }
[ -f "$ROOT/sarek.opam" ] ||
  { echo "FAIL: fixture missing: $ROOT/sarek.opam" >&2; exit 2; }

# The diverging-make case writes to the real sarek.opam. Restore from a byte
# copy rather than `git checkout`, so a concurrent worktree's index is never
# touched and the file returns to exactly what it was, tracked or not.
BACKUP="$(mktemp -d)"
cp -- "$ROOT/sarek.opam" "$BACKUP/sarek.opam" ||
  { echo "FAIL: could not back up sarek.opam" >&2; exit 2; }
restore() { cp -- "$BACKUP/sarek.opam" "$ROOT/sarek.opam"; }
trap 'restore; rm -rf "$BACKUP" "$STUBS"' EXIT

STUBS="$(mktemp -d)"

# A dune that answers --version and nothing else. The subject only ever runs
# `dune --version`; `make` is stubbed, so no build is ever attempted.
mk_dune() { printf '#!/bin/sh\n[ "$1" = --version ] && { echo 9.9.9-stub; exit 0; }\nexit 0\n' > "$1/dune"; chmod +x "$1/dune"; }
mk_make_noop() { printf '#!/bin/sh\n[ "$1" = opam ] || exec /usr/bin/make "$@"\nexit 0\n' > "$1/make"; chmod +x "$1/make"; }

pass=0; fail=0

# $1 name, $2 stub dir, $3 expected exit, $4 expected message substring
check() {
  local name="$1" dir="$2" want="$3" msg="$4" out code
  out="$(cd "$ROOT" && PATH="$dir:$PATH" bash "$SUBJECT" 2>&1)"
  code=$?
  restore
  local ok=1 why=""
  [ "$code" = "$want" ] || { ok=0; why="exit $code, wanted $want"; }
  case "$out" in
    *"$msg"*) ;;
    *) ok=0; why="${why:+$why; }message missing: '$msg'" ;;
  esac
  if [ "$ok" = 1 ]; then
    pass=$((pass + 1)); printf 'ok    %-28s exit %s, says %s\n' "$name" "$code" "'$msg'"
  else
    fail=$((fail + 1)); printf 'FAIL  %-28s %s\n' "$name" "$why"
    printf '%s\n' "$out" | sed 's/^/        | /'
  fi
}

# --- green baseline ---------------------------------------------------------
# Mandatory: without it, every red case below could be red for an unrelated
# reason and this file would report a row of passes about a broken subject.
d="$STUBS/green"; mkdir -p "$d"; mk_dune "$d"; mk_make_noop "$d"
check green-baseline "$d" 0 "converged over two runs"

# --- the two set -e sites, which are the reason this file exists ------------
# cmp branch 1: the two runs disagree. A stub make that appends a different
# line each call is the smallest thing that makes `make opam` non-convergent.
d="$STUBS/diverge"; mkdir -p "$d"; mk_dune "$d"
cat > "$d/make" <<'EOF'
#!/bin/sh
[ "$1" = opam ] || exec /usr/bin/make "$@"
n=$(cat "$TMPDIR_B213/n" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$TMPDIR_B213/n"
printf 'x-diverge: "%s"\n' "$n" >> sarek.opam
EOF
chmod +x "$d/make"
export TMPDIR_B213="$STUBS"; rm -f "$STUBS/n"
check cmp-runs-differ "$d" 1 "does not converge"
rm -f "$STUBS/n"

# cmp branch >1: cannot compare. Distinct from "differs" on purpose -- a
# contract that names 1 for a real difference and 2 for an unusable comparison
# is not satisfied by "something failed".
d="$STUBS/cmp2"; mkdir -p "$d"; mk_dune "$d"; mk_make_noop "$d"
printf '#!/bin/sh\nexit 2\n' > "$d/cmp"; chmod +x "$d/cmp"
check cmp-cannot-compare "$d" 2 "could not compare"

# git diff branch >1: git failed to answer. Pre-fix this leaked 128 as the
# gate's own status, which reads as neither "clean" nor "dirty".
d="$STUBS/git128"; mkdir -p "$d"; mk_dune "$d"; mk_make_noop "$d"
cat > "$d/git" <<'EOF'
#!/bin/sh
if [ "$1" = diff ] && [ "$2" = --exit-code ]; then exit 128; fi
exec /usr/bin/git "$@"
EOF
chmod +x "$d/git"
check gitdiff-cannot-answer "$d" 2 "could not compare the generated files"

# --- the toolchain-naming refusals -----------------------------------------
# `make opam` fails. Not a finding about the opam files, and before backlog-213
# this left through `set -e` with no message at all.
d="$STUBS/makefail"; mkdir -p "$d"; mk_dune "$d"
printf '#!/bin/sh\n[ "$1" = opam ] || exec /usr/bin/make "$@"\nexit 3\n' > "$d/make"; chmod +x "$d/make"
check make-opam-fails "$d" 2 "INCONCLUSIVE"

# A dune that exists but will not run: a dangling symlink into a deleted switch.
# Under `set -e` an unguarded `--version` would kill the script with dune's own
# status and print nothing.
d="$STUBS/baddune"; mkdir -p "$d"; mk_make_noop "$d"
printf '#!/bin/sh\nexit 7\n' > "$d/dune"; chmod +x "$d/dune"
check dune-not-runnable "$d" 2 "not runnable"

# No dune at all. PATH is replaced rather than prefixed here, so this case needs
# its own invocation.
out="$(cd "$ROOT" && PATH="/usr/bin:/bin" bash "$SUBJECT" 2>&1)"; code=$?
restore
if [ "$code" = 2 ] && case "$out" in *"no 'dune' on PATH"*) true ;; *) false ;; esac; then
  pass=$((pass + 1)); printf 'ok    %-28s exit 2, says %s\n' "no-dune-on-path" "\"no 'dune' on PATH\""
else
  fail=$((fail + 1)); printf 'FAIL  %-28s exit %s\n' "no-dune-on-path" "$code"
  printf '%s\n' "$out" | sed 's/^/        | /'
fi

# --- verdict ----------------------------------------------------------------
# A count that could silently become zero is the failure mode this repository
# keeps finding, so the expected number of cases is pinned rather than reported.
EXPECTED_CASES=7
total=$((pass + fail))
if [ "$total" -ne "$EXPECTED_CASES" ]; then
  echo "REFUSED: ran $total case(s), expected $EXPECTED_CASES. A harness that" >&2
  echo "  reports what it happened to run is complete about a set it chose." >&2
  exit 2
fi
if [ "$fail" -ne 0 ]; then
  echo "FAILED: $fail of $total case(s)." >&2
  exit 1
fi
echo "OK: $pass/$total case(s) — check-opam-clean.sh goes red with the message it promises on both set -e status reads, both cannot-compare paths, and all three toolchain refusals."
