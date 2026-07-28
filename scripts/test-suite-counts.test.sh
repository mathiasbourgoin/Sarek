#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Covering test for scripts/test-suite-counts.sh.
#
# The bug this script exists to prevent is a counting pattern that silently
# drops suites. A counter cannot be trusted because it printed a number -- it
# has to be shown mis-counting when fed the shape it used to miss.
set -uo pipefail

CNT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-suite-counts.sh"
[ -x "$CNT" ] || { echo "FAIL: $CNT not found or not executable"; exit 2; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/suite-counts-test.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0

check() {
  local desc="$1" got="$2" want="$3"
  if [ "$got" = "$want" ]; then
    echo "  PASS: $desc"
    pass=$((pass + 1))
  else
    echo "  FAIL: $desc -- expected '$want', got '$got'"
    fail=$((fail + 1))
  fi
}

echo "test-suite-counts.sh covering test"

# A log exercising every epilogue shape Alcotest and qcheck actually emit:
# plural, singular, zero, and a qcheck runner line.
cat > "$TMP/mixed.log" <<'LOG'
Testing `Sarek_float32'.
Test Successful in 0.002s. 61 tests run.
Testing `Solo'.
Test Successful in 0.001s. 1 test run.
Testing `Empty_probe'.
Test Successful in 0.000s. 0 test run.
Testing `Props'.
success (ran 12 tests)
LOG

# --min-suites 0 throughout this block: these logs are deliberately tiny, and
# the plausibility floor (asserted separately below) would otherwise reject
# them. Disabling it explicitly is the point -- it is the documented escape,
# and a covering test that silently depended on the floor being absent would
# stop noticing when the floor changed.
out="$("$CNT" --min-suites 0 "$TMP/mixed.log")"

# 61 + 1 + 0 = 62 across 3 alcotest suites. The singular and zero forms are
# exactly what a `[0-9]+ tests run` pattern drops.
check "counts plural + singular + zero alcotest suites" \
  "$(echo "$out" | /usr/bin/grep '^alcotest' | /usr/bin/tr -s ' ')" \
  "alcotest : 62 cases across 3 suites"
check "counts qcheck separately" \
  "$(echo "$out" | /usr/bin/grep '^qcheck' | /usr/bin/tr -s ' ')" \
  "qcheck : 12 cases across 1 suites"
check "reports zero-case suites" \
  "$(echo "$out" | /usr/bin/grep '^zero-case' | /usr/bin/tr -s ' ')" \
  "zero-case suites: 1"

# The regression itself: a plural-only pattern must give a DIFFERENT, lower
# answer on this log. If these agree, the log no longer exercises the bug and
# the test above has stopped proving anything.
plural_only="$(/usr/bin/grep -oE '[0-9]+ tests run' "$TMP/mixed.log" \
  | /usr/bin/awk '{s+=$1} END {print s+0" "NR}')"
check "plural-only pattern demonstrably under-counts (61 cases, 1 suite)" \
  "$plural_only" "61 1"

# Failure and skip tallies.
cat > "$TMP/redskip.log" <<'LOG'
Testing `A'.
> [FAIL]        thing            0   broken.
> [SKIP]        gpu              0   needs hardware.
1 failure! in 0.001s. 4 tests run.
LOG
out2="$("$CNT" --min-suites 0 "$TMP/redskip.log")"
check "counts FAIL lines" \
  "$(echo "$out2" | /usr/bin/grep '^FAIL' | /usr/bin/tr -s ' ')" "FAIL : 1"
check "counts SKIP lines" \
  "$(echo "$out2" | /usr/bin/grep '^SKIP' | /usr/bin/tr -s ' ')" "SKIP : 1"

# Drift detector: an epilogue whose case-count the pattern cannot read must
# exit 2 rather than print a confidently wrong total.
cat > "$TMP/drift.log" <<'LOG'
Testing `A'.
Test Successful in 0.002s. 61 tests run.
Testing `B'.
Test Successful in 0.001s. some-new-format.
LOG
"$CNT" --min-suites 0 "$TMP/drift.log" >/dev/null 2>&1
check "unparseable epilogue exits 2, not a wrong total" "$?" "2"

# Missing file is an error, not a silent zero.
"$CNT" "$TMP/nope.log" >/dev/null 2>&1
check "missing log exits 2" "$?" "2"

# ---------------------------------------------------------------------------
# backlog-150: an empty or unrecognisable log is a USAGE ERROR, never a result.
#
# The shipped script accepted all four inputs below and printed
# "0 cases / 0 FAIL / 0 SKIP", exit 0 -- a green run, reported by the very
# instrument this repo uses to audit whether its other gates can fail. The
# cases assert both halves: non-zero exit, AND no count line on stdout, because
# a caller that greps for `TOTAL` must not find one.
# ---------------------------------------------------------------------------

: > "$TMP/empty.log"

# Truncated before any suite finished: real `dune test` preamble, cut early.
cat > "$TMP/trunc-early.log" <<'LOG'
File "sarek/tests/unit/dune", line 1, characters 0-0:
Entering directory '/repo'
Testing `Sarek_float32'.
This run has ID `IGM5LJHD'.

  [OK]          add                 0   scalar add.
LOG

# Only skips, no epilogue at all.
cat > "$TMP/skips-only.log" <<'LOG'
  [SKIP]        gpu              0   needs hardware.
  [SKIP]        gpu              1   needs hardware.
LOG

# Not a test log in the first place. This fixture used to end
# "Command exited with code 2." and that line has been dropped deliberately.
# The backlog-157 reason was that it would route this input to gate 0 (exit 4)
# instead of gate 1 (exit 2), and a gate-1 case that passes because gate 0
# fired first has stopped testing gate 1. The review measurement gives a
# second, better reason to keep it out: dune 3.24.1 does not write that line in
# any display mode, so as a fixture it was attesting a wording nothing emits.
# The shapes dune ACTUALLY writes are asserted below, against gate 0, from real
# logs.
cat > "$TMP/notalog.log" <<'LOG'
ocamlfind: Package `sarek' not found
LOG

# Exit code asserted EXACTLY, not merely as non-zero. Gate 1 is specified to
# exit 2, and "non-zero" would also be satisfied by a python traceback (1) or
# by the plausibility floor firing instead (3) -- so a crash, or a regression
# that moved this input from gate 1 to gate 3, would pass a case whose entire
# purpose is to pin gate 1. A weakened assertion inside the covering test for
# the check that could not fail is the same class of defect one level up.
for case in empty trunc-early skips-only notalog; do
  got_out="$("$CNT" --min-suites 0 "$TMP/$case.log" 2>/dev/null)"
  got_rc=$?
  check "$case log exits 2 — usage error, not a result" "$got_rc" "2"
  check "$case log prints no counts on stdout" \
    "$(echo "$got_out" | /usr/bin/grep -c 'cases across')" "0"
done

# Positive control for the four cases above. Without it, "went red on an empty
# log" and "is red on everything" are the same observation. A log that IS a
# real run must still come back green and with the right total.
big=""
for i in $(seq 1 12); do
  big="${big}Testing \`Suite_$i'.
Test Successful in 0.002s. 5 tests run.
"
done
printf '%s' "$big" > "$TMP/plausible.log"
out3="$("$CNT" "$TMP/plausible.log")"
check "positive control: a plausible log still exits 0" "$?" "0"
check "positive control: and counts correctly" \
  "$(echo "$out3" | /usr/bin/grep '^TOTAL' | /usr/bin/tr -s ' ')" \
  "TOTAL : 60 cases across 12 suites"

# ---------------------------------------------------------------------------
# backlog-150: pipe mode must read the caller's pipe.
#
# `python3 - <<'PYEOF'` hands python its program on stdin, leaving the program
# nothing to read. The header's FIRST recommended invocation
# (`dune test --force 2>&1 | scripts/test-suite-counts.sh`) consequently
# reported 0 cases for any input whatsoever, exit 0. This is the regression
# test for that: same log, both routes, same answer.
# ---------------------------------------------------------------------------
piped="$(/usr/bin/cat "$TMP/plausible.log" | "$CNT" | /usr/bin/grep '^TOTAL' \
  | /usr/bin/tr -s ' ')"
check "pipe mode reads stdin, and agrees with file mode" \
  "$piped" "TOTAL : 60 cases across 12 suites"

# An empty pipe must be the same usage error an empty file is -- this is the
# fully-cached `dune test` case, the one that actually happens.
#
# --min-suites 0 here is deliberate. Without it the floor also rejects an empty
# pipe, so the case would pass even with the empty-log gate removed and would
# be attesting the floor rather than the thing it names.
#
# Exact 2 for the same reason as the loop above: non-zero would be satisfied by
# a crash or by the floor firing, neither of which is this gate.
: | "$CNT" --min-suites 0 >/dev/null 2>&1
check "empty pipe exits 2 — the fully-cached \`dune test\` case" "$?" "2"

# ---------------------------------------------------------------------------
# backlog-150: the plausibility floor.
#
# Partial caching under-reports without tripping either gate above: the suites
# that do appear parse perfectly, there are just fewer of them. The number
# looks like a number, which is what makes it worse than nothing.
# ---------------------------------------------------------------------------
"$CNT" "$TMP/mixed.log" >/dev/null 2>&1
check "below the default floor exits 3" "$?" "3"

floor_out="$("$CNT" "$TMP/mixed.log" 2>/dev/null || true)"
check "below-floor still shows what it parsed" \
  "$(echo "$floor_out" | /usr/bin/grep -c 'cases across')" "3"

"$CNT" --min-suites 0 "$TMP/mixed.log" >/dev/null 2>&1
check "--min-suites 0 disables the floor" "$?" "0"

"$CNT" --min-suites 999 "$TMP/plausible.log" >/dev/null 2>&1
check "floor is honoured at a raised threshold" "$?" "3"

# Argument handling: a malformed invocation is an error, not a default.
"$CNT" --min-suites nope "$TMP/plausible.log" >/dev/null 2>&1
check "non-numeric --min-suites exits 2" "$?" "2"
"$CNT" --bogus >/dev/null 2>&1
check "unknown option exits 2" "$?" "2"
"$CNT" "$TMP/plausible.log" "$TMP/mixed.log" >/dev/null 2>&1
check "two log files exit 2" "$?" "2"

# ---------------------------------------------------------------------------
# backlog-157: a log from a run that DID NOT COMPLETE is not a result either.
#
# This is the one failure the three gates above are structurally blind to. A
# `dune test` that dies on a compile error emits, verbatim and parseable, the
# epilogue of every suite that ran before the failure. Measured on this tree:
# a clean run is 1697 cases across 117 suites; append one unbound identifier to
# sarek/tests/unit/test_soa.ml and dune exits 1 while the pre-fix script
# printed 1692 across 116, 0 FAIL, exit 0. 99.7% of the truth, from a build
# that never happened.
#
# That measurement is also why the floor cannot be the fix, and the case
# "fires ahead of the plausibility floor" below is the assertion that keeps
# the two apart: no tolerance that admits `dune test spoc/ir` rejects 116/117.
#
# Exit 4 asserted EXACTLY throughout, for the reason the backlog-150 block
# gives one level up: 4 exists precisely to be a different answer from 0, 2
# and 3, and "non-zero" is satisfied by all of them.
# ---------------------------------------------------------------------------

FIXTURES="$(cd "$(dirname "$CNT")" && pwd)/prove-red-fixtures"
FAILED_BUILD_LOG="$FIXTURES/dune-test-failed-build-log.txt"
[ -f "$FAILED_BUILD_LOG" ] || {
  echo "  FAIL: fixture missing: $FAILED_BUILD_LOG"; fail=$((fail + 1)); }

# The fixture is a REAL log, not a written one: the complete unedited output of
#   dune test spoc/ir spoc/registry spoc/framework --force -j 1
# with one unbound identifier appended to
# spoc/registry/test/test_sarek_registry.ml. Dune exited 1; six suites ran
# first and their epilogues are intact.
got_out="$("$CNT" --min-suites 0 "$FAILED_BUILD_LOG" 2>/dev/null)"
got_rc=$?
check "genuine failed-build log exits 4 — the run did not complete" "$got_rc" "4"
check "genuine failed-build log prints no counts on stdout" \
  "$(echo "$got_out" | /usr/bin/grep -c 'cases across')" "0"
check "genuine failed-build log names the runner's own failure" \
  "$("$CNT" --min-suites 0 "$FAILED_BUILD_LOG" 2>&1 >/dev/null \
     | /usr/bin/grep -c 'did not complete')" "1"

# Gate 0 must fire AHEAD of the plausibility floor, not instead of it. The
# fixture has six suites, which is below the default floor of 10, so without
# this ordering the case above would pass with exit 3 and would be attesting
# the floor -- the very instrument backlog-157 showed cannot catch this.
"$CNT" "$FAILED_BUILD_LOG" >/dev/null 2>&1
check "gate 0 fires ahead of the plausibility floor (4, not 3)" "$?" "4"

# The pre-fix answer, reconstructed. If this stops being a plausible-looking
# total the fixture has stopped exercising the defect and the cases above have
# stopped proving anything -- the same positive-control argument the
# plural-only grep gets at the top of this file.
prefix_answer="$(/usr/bin/grep -oE '[0-9]+ tests? run' "$FAILED_BUILD_LOG" \
  | /usr/bin/awk '{s+=$1} END {print s+0" "NR}')"
check "fixture still yields a plausible-looking total (46 cases, 6 suites)" \
  "$prefix_answer" "46 6"
check "and no [FAIL] in it — the pre-fix report was '0 FAIL'" \
  "$(/usr/bin/grep -c '\[FAIL\]' "$FAILED_BUILD_LOG")" "0"

# --- the other two real dune failure shapes (backlog-157 review) ------------
#
# Both fixtures are REAL logs, generated the same way as the one above and for
# the same reason: the shape they carry was previously GUESSED, and the guess
# was wrong. Each was measured going green -- exit 0, "46 cases across 6
# suites, 0 FAIL" -- against the first version of gate 0.
WARN_LOG="$FIXTURES/dune-test-warning-as-error-log.txt"
VERBOSE_LOG="$FIXTURES/dune-verbose-command-exit-log.txt"
for f in "$WARN_LOG" "$VERBOSE_LOG"; do
  [ -f "$f" ] || { echo "  FAIL: fixture missing: $f"; fail=$((fail + 1)); }
done

# `dune test spoc/ir spoc/registry spoc/framework --force -j 1` with
# `let unused_backlog157_probe x = x + 1` appended to
# spoc/registry/test/test_sarek_registry.ml. Dune exit 1, and its failure
# report is `Error (warning 32 [unused-value-declaration]): ...` -- NOT
# `Error: `, which is all the shipped pattern matched. Warnings-as-errors is
# how this tree's dev profile fails most builds, so this was the common case.
got_out="$("$CNT" --min-suites 0 "$WARN_LOG" 2>/dev/null)"
got_rc=$?
check "warning-as-error build log exits 4" "$got_rc" "4"
check "warning-as-error build log prints no counts on stdout" \
  "$(echo "$got_out" | /usr/bin/grep -c 'cases across')" "0"

# Positive control for the fixture, not for the gate: if this stops being a
# plausible-looking total the fixture has stopped exercising the defect.
warn_prefix="$(/usr/bin/grep -oE '[0-9]+ tests? run' "$WARN_LOG" \
  | /usr/bin/awk '{s+=$1} END {print s+0" "NR}')"
check "warning-as-error fixture still yields a plausible total (46, 6)" \
  "$warn_prefix" "46 6"

# The same command with `let () = exit 3` appended instead and --display
# verbose added: a runtest rule that failed at RUNTIME. Dune writes
# `Command [17] exited with code 3:` -- a job number and a colon. The pattern
# this gate shipped with, `Command exited with code N.`, matches nothing dune
# 3.24.1 emits in any display mode, so it had been dead since it was written.
got_out="$("$CNT" --min-suites 0 "$VERBOSE_LOG" 2>/dev/null)"
got_rc=$?
check "verbose runtime-failure log exits 4" "$got_rc" "4"
check "verbose runtime-failure log prints no counts on stdout" \
  "$(echo "$got_out" | /usr/bin/grep -c 'cases across')" "0"

# The wording is asserted directly, because the two cases above would also
# pass if the fixture were matched for some unrelated reason.
check "and the fixture really carries dune's bracketed wording" \
  "$(/usr/bin/grep -cE '^Command \[[0-9]+\] exited with code [0-9]+:' \
     "$VERBOSE_LOG")" "1"
check "the guessed wording appears in no real dune log we have" \
  "$(/usr/bin/grep -c 'Command exited with code' "$WARN_LOG" \
     "$VERBOSE_LOG" "$FAILED_BUILD_LOG" | /usr/bin/awk -F: '{s+=$2} END {print s}')" \
  "0"

# --- the marker must be dune's voice, not a test's -------------------------
#
# Without this, "refuses a failed build" and "refuses any log containing the
# word Error" are the same observation, and the second one would make the
# counter useless on this repository: both lines below are verbatim from the
# clean 1697-case run.
cat > "$TMP/error-shaped-output.log" <<'LOG'
Testing `Ir_interp'.
  RMem (EVar unbound) -> RMemError [ok]
  [OK]          error_handling                0   Error handling.
  [OK]          backend_error                 1   Error: propagated.
Test Successful in 0.002s. 61 tests run.
LOG
out4="$("$CNT" --min-suites 0 "$TMP/error-shaped-output.log")"
check "indented test output mentioning Error is NOT a runner failure" "$?" "0"
check "and is still counted" \
  "$(echo "$out4" | /usr/bin/grep '^TOTAL' | /usr/bin/tr -s ' ')" \
  "TOTAL : 61 cases across 1 suites"

# ...and the harder half of the same claim: column 0 is NOT reliably dune's own
# voice on this tree. Dune passes an e2e rule's stdout straight through,
# unindented, and two things in a real full-suite run start a column-0 line
# with the word Error while the run is going perfectly well:
#
#   sarek/tests/e2e/Benchmarks.ml:247  `Error on %s: %s` for a GPU error the
#                                      very next line then TOLERATES
#   pocl/LLVM                          `Error executing LLVM compilation action.`
#
# Measured together in one real 6000-line log from a machine with a GPU: 65
# lines match `^Error\b` and zero match the pattern this script uses. That is
# the whole reason it is `^Error(?::| \()` and not `^Error\b`. This case is the
# guard on that difference -- widen the pattern and it goes red here first.
cat > "$TMP/tolerated-gpu-error.log" <<'LOG'
Testing `E2e'.
Test Successful in 0.002s. 61 tests run.
Error executing LLVM compilation action.
Error on AMD Radeon RX 7900 XTX (radeonsi, navi31): [OpenCL Runtime] failed
Errors were tolerated; the CPU baseline passed.
LOG
out4b="$("$CNT" --min-suites 0 "$TMP/tolerated-gpu-error.log")"
check "column-0 tolerated-GPU-error output is NOT a runner failure" "$?" "0"
check "and is still counted" \
  "$(echo "$out4b" | /usr/bin/grep '^TOTAL' | /usr/bin/tr -s ' ')" \
  "TOTAL : 61 cases across 1 suites"

# Positive control for the case above: the same log with dune's own wording
# added must go red. Without this, "the pattern is narrow" and "the pattern
# matches nothing" are the same observation.
/usr/bin/cat "$TMP/tolerated-gpu-error.log" > "$TMP/tolerated-plus-real.log"
echo 'Error (warning 32 [unused-value-declaration]): unused value probe.' \
  >> "$TMP/tolerated-plus-real.log"
"$CNT" --min-suites 0 "$TMP/tolerated-plus-real.log" >/dev/null 2>&1
check "positive control: the same log plus a real dune Error exits 4" "$?" "4"

# --- the reported line number ----------------------------------------------
#
# qcheck's progress bar redraws with a bare \r, and Python text mode rewrites
# a bare \r to \n. The first draft of gate 0 read the sample log in text mode
# and located its compile error at "line 301"; grep says 240, the difference
# being exactly the 61 carriage returns before it. The number is the operator's
# only pointer into a 4000-line log, and a pattern anchored to manufactured
# line starts would match text dune never wrote at column 0.
printf 'Testing `A'"'"'.\n[ ] 0 / 9\r[x] 9 / 9\r[done]\nTest Successful in 0.0s. 9 tests run.\nError: Unbound value zz\n' \
  > "$TMP/cr-progress.log"
cr_line="$("$CNT" --min-suites 0 "$TMP/cr-progress.log" 2>&1 >/dev/null \
  | /usr/bin/grep -oE 'at line [0-9]+' | /usr/bin/head -1)"
check "line number counts \\n only, agreeing with grep -n" \
  "$cr_line" "at line $(/usr/bin/grep -n '^Error: ' "$TMP/cr-progress.log" \
                        | /usr/bin/cut -d: -f1)"

# --- --runner-exit ---------------------------------------------------------
#
# The half that covers what the marker scan cannot see: a runner killed by a
# signal, a timeout or the OOM killer writes no failure report at all. Asserted
# against the PLAUSIBLE log, which passes every other gate, so nothing but the
# flag can be producing the red.
"$CNT" --runner-exit 1 "$TMP/plausible.log" >/dev/null 2>&1
check "--runner-exit 1 on an otherwise-clean log exits 4" "$?" "4"
check "--runner-exit 1 prints no counts on stdout" \
  "$("$CNT" --runner-exit 1 "$TMP/plausible.log" 2>/dev/null \
     | /usr/bin/grep -c 'cases across')" "0"
"$CNT" --runner-exit=137 "$TMP/plausible.log" >/dev/null 2>&1
check "--runner-exit=N form works too (137, the OOM kill)" "$?" "4"

# Positive control for the flag. Without it, "--runner-exit made it red" and
# "--runner-exit is red on everything" are indistinguishable.
out5="$("$CNT" --runner-exit 0 "$TMP/plausible.log")"
check "--runner-exit 0 is still a result" "$?" "0"
check "--runner-exit 0 counts identically to no flag at all" \
  "$(echo "$out5" | /usr/bin/grep '^TOTAL' | /usr/bin/tr -s ' ')" \
  "TOTAL : 60 cases across 12 suites"

# --- --runner-exit 0 outranks the marker scan (backlog-157 review) ----------
#
# The two sources are not equal evidence. The caller's exit status IS the fact;
# the scan guesses at the fact from text, and the case above shows the guess
# firing on text dune never wrote. When both are present the fact wins, so a
# caller who knows the run completed is not held hostage by a heuristic whose
# only remedy would be to stop passing the flag.
"$CNT" --min-suites 0 --runner-exit 0 "$WARN_LOG" >/dev/null 2>&1
check "--runner-exit 0 suppresses the marker scan" "$?" "0"

# ...and the control WITHOUT which the case above proves nothing: the very same
# log, same argv minus the flag, must still be exit 4. Otherwise "the flag
# suppressed the scan" and "the scan stopped working" read identically.
"$CNT" --min-suites 0 "$WARN_LOG" >/dev/null 2>&1
check "control: the same log without the flag is still exit 4" "$?" "4"

# A missing or malformed value is a usage error (2), never a silent default of
# zero -- defaulting to "the run was fine" is the shape this whole file exists
# to refuse.
"$CNT" --runner-exit nope "$TMP/plausible.log" >/dev/null 2>&1
check "non-numeric --runner-exit exits 2" "$?" "2"
"$CNT" --runner-exit >/dev/null 2>&1
check "--runner-exit with no value exits 2" "$?" "2"

# --- given-but-empty is a usage error, not "not given" (backlog-157 review) --
#
# Measured on this branch before the review, both forms below printed
# "TOTAL : 60 cases across 12 suites" and exited 0: the empty string is the
# parser's sentinel for "the caller did not say", so a caller who said nothing
# landed on it and the gate silently vanished. That is not a contrived typo --
# it is exactly what the two-line invocation this tool documents degrades to
# when `rc` is unset or misspelled, and the caller has no way to tell.
#
# Exit 2 asserted exactly. "Non-zero" would also be satisfied by 4, which is
# the wrong answer here (the log is fine; the INVOCATION is not), and by a
# python traceback.
"$CNT" --min-suites 0 --runner-exit "" "$TMP/plausible.log" >/dev/null 2>&1
check "--runner-exit \"\" exits 2, not a silent fallback to \"not given\"" "$?" "2"
"$CNT" --min-suites 0 --runner-exit= "$TMP/plausible.log" >/dev/null 2>&1
check "--runner-exit= (equals form) exits 2 too" "$?" "2"
check "--runner-exit \"\" prints no counts on stdout" \
  "$("$CNT" --min-suites 0 --runner-exit "" "$TMP/plausible.log" 2>/dev/null \
     | /usr/bin/grep -c 'cases across')" "0"
check "--runner-exit \"\" says what is wrong with the invocation" \
  "$("$CNT" --min-suites 0 --runner-exit "" "$TMP/plausible.log" 2>&1 >/dev/null \
     | /usr/bin/grep -c 'given an empty value')" "1"

# Pipe form: --runner-exit is unavailable there (PIPESTATUS does not exist yet
# inside the pipeline), so the marker scan is the ONLY defence and has to work
# on stdin as well as on a file.
: | true
/usr/bin/cat "$FAILED_BUILD_LOG" | "$CNT" --min-suites 0 >/dev/null 2>&1
check "pipe form catches the failed build too (stdin, exit 4)" \
  "${PIPESTATUS[1]}" "4"

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
echo "OK: test-suite-counts.sh counts singular/zero/plural Alcotest epilogues,"
echo "    separates qcheck, fails closed when the log format drifts, refuses to"
echo "    report a count for an empty, truncated, unrecognisable or implausibly"
echo "    small log (backlog-150), and refuses one for a run that did not"
echo "    complete -- by the caller's exit code or the runner's own failure"
echo "    report in the log (backlog-157), in the wordings dune 3.24.1 was"
echo "    measured emitting and not the three it never emitted, without"
echo "    mistaking a tolerated GPU error for one, and refusing an empty"
echo "    --runner-exit rather than reading it as \"not given\""
