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

# Real captured dune logs, used where the exact spelling the compiler emits is
# the subject. Asserted present rather than left to fail as a mystery exit code:
# a missing fixture makes `cat` produce nothing, which the counter reports as
# "no output recognised" (2) — a number that looks like an unrelated bug.
FIXTURES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prove-red-fixtures"
for f in dune-test-sample-log.txt dune-test-warning-as-error-log.txt \
         dune-verbose-command-exit-log.txt dune-verbose-signal-log.txt; do
  [ -f "$FIXTURES/$f" ] || { echo "FAIL: fixture $FIXTURES/$f is missing"; exit 2; }
done

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

# Not a test log in the first place -- a build failure, say.
cat > "$TMP/notalog.log" <<'LOG'
ocamlfind: Package `sarek' not found
Command exited with code 2.
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

# ---------------------------------------------------------------------------
# backlog-157: did the RUN succeed? Gates 1-3 only ever asked whether the total
# was complete for the suites present.
#
# The build-failure log below is the shape that made this necessary: `dune test
# --force` builds and runs suite by suite, so a compile error part-way through
# leaves every earlier epilogue intact. Measured on the real tree, that is 45
# parsed suites -- past the floor, consistent with their epilogues -- reported as
# a plausible total with exit 0 off a `dune test` that exited 1.
# ---------------------------------------------------------------------------
cat > "$TMP/build-failed.log" <<'LOG'
Testing `Sarek_float32'.
Test Successful in 0.002s. 61 tests run.
Testing `Solo'.
Test Successful in 0.001s. 1 test run.
File "sarek/tests/unit/test_thing.ml", line 12, characters 0-1:
12 | x
     ^
Error: Syntax error
LOG

"$CNT" --min-suites 0 "$TMP/build-failed.log" >/dev/null 2>&1
check "a build failure in the log exits 5" "$?" "5"

# The count still surfaces, but LABELLED partial — the reordering (exit 5 now
# runs before the count block) is what changed the wording, and "partial" is the
# more honest framing for a total taken over a run that died.
bf_out="$("$CNT" --min-suites 0 "$TMP/build-failed.log" 2>&1 || true)"
# Anchored on the `partial:` LINE, not on a bare substring. It used to grep for
# "ran before the failure" anywhere in the output, which silently also matched
# the ERROR paragraph once #367 reworded it from "suites built before" to "suites
# that ran before" — count 1 became 2 and the case failed for a reason that had
# nothing to do with the partial count. The assertion was looser than its own
# name ("shows the partial count"), so it was coupled to prose it does not own.
check "build failure still shows the partial count" \
  "$(echo "$bf_out" | /usr/bin/grep -cE '^partial: .* ran before the failure')" "1"

# --dune-exit is authoritative: the log parses clean, dune says otherwise.
"$CNT" --min-suites 0 --dune-exit 1 "$TMP/mixed.log" >/dev/null 2>&1
check "--dune-exit 1 on a clean log exits 5" "$?" "5"
"$CNT" --min-suites 0 --dune-exit 0 "$TMP/mixed.log" >/dev/null 2>&1
check "--dune-exit 0 on a clean log stays 0" "$?" "0"

# Failing CASES are a different fact from a run that died, and the exit code has
# to be able to say which -- collapsing them would lose that.
cat > "$TMP/failing.log" <<'LOG'
Testing `Sarek_float32'.
  [FAIL]        some suite    0   a case....
  [FAIL]        some suite    1   another....
2 failures! in 0.010s. 61 tests run.
Testing `Solo'.
Test Successful in 0.001s. 1 test run.
Testing `Third'.
Test Successful in 0.001s. 3 tests run.
LOG

"$CNT" --min-suites 0 "$TMP/failing.log" >/dev/null 2>&1
check "failing cases exit 4, not 0" "$?" "4"

fail_out="$("$CNT" --min-suites 0 "$TMP/failing.log" 2>&1 || true)"
check "failing-case counts are still printed" \
  "$(echo "$fail_out" | /usr/bin/grep -c 'FAIL     : 2')" "1"

# PRECEDENCE, at the DEFAULT floor. The first version of this change ran the
# completion check last, so these two cases reported 2 and 3 — "not a test log"
# and "caching problem" — for a run whose build had failed. Both non-zero, so
# neither was a false green, but both misclassified the failure and told the
# caller to re-run instead of to fix the build. CodeRabbit caught it on #357.
#
# Deliberately WITHOUT --min-suites 0: the floor is exactly what used to
# pre-empt the completion check, so disabling it would test the wrong thing.
cat > "$TMP/build-failed-early.log" <<'LOG'
File "sarek/tests/unit/test_thing.ml", line 12, characters 0-1:
Error: Syntax error
LOG
"$CNT" "$TMP/build-failed-early.log" >/dev/null 2>&1
check "build failure with NO epilogue exits 5, not 2" "$?" "5"

"$CNT" "$TMP/mixed.log" >/dev/null 2>&1
check "  (control) same log, no build error, is below-floor 3" "$?" "3"

"$CNT" --dune-exit 1 "$TMP/mixed.log" >/dev/null 2>&1
check "--dune-exit 1 under the default floor exits 5, not 3" "$?" "5"

printf '' > "$TMP/empty.log"
"$CNT" --dune-exit 1 "$TMP/empty.log" >/dev/null 2>&1
check "--dune-exit 1 on an EMPTY log exits 5, not 2" "$?" "5"

"$CNT" "$TMP/empty.log" >/dev/null 2>&1
check "  (control) empty log with no dune-exit is still 2" "$?" "2"

# A build failure OUTRANKS failing cases: if the run died, the FAIL tally is a
# count over a partial run and "2 tests failed" would understate it.
cat > "$TMP/both.log" <<'LOG'
Testing `Sarek_float32'.
  [FAIL]        some suite    0   a case....
1 failure! in 0.010s. 61 tests run.
File "sarek/tests/unit/test_thing.ml", line 12, characters 0-1:
Error: Syntax error
LOG
"$CNT" --min-suites 0 "$TMP/both.log" >/dev/null 2>&1
check "a died run outranks its failing cases (5, not 4)" "$?" "5"

# WARNING-AS-ERROR. This repo builds with warnings as errors, and the compiler
# prints that failure as `Error (warning 32 [...]): ...` — no colon after
# `Error`, so the `^Error:` pattern this file's own gate narrowed to never
# matched it. Measured on the real captured log below: exit 0 and a clean
# "46 cases across 6 suites" for a run whose build never completed.
#
# The fixture is a REAL 271-line dune log rather than a two-line heredoc,
# because the whole defect is in the exact spelling the compiler emits — a
# hand-written approximation is where a wrong guess about that spelling hides.
"$CNT" --min-suites 0 "$FIXTURES/dune-test-warning-as-error-log.txt" >/dev/null 2>&1
check "a warning-as-error build failure exits 5, not 0" "$?" "5"

# The composite is the dangerous shape, and the one the plausibility floor does
# NOT save: enough suites completed to clear the floor, then the build died. The
# single-fixture case above is only 6 suites, so on its own it would also have
# been caught by the floor for an unrelated reason.
{ cat "$FIXTURES/dune-test-sample-log.txt" "$FIXTURES/dune-test-sample-log.txt" \
       "$FIXTURES/dune-test-sample-log.txt" "$FIXTURES/dune-test-sample-log.txt" \
       "$FIXTURES/dune-test-sample-log.txt"
  printf 'File "sarek/ppx/Sarek_lower_ir.ml", line 850, characters 16-30:\n'
  printf 'Error (warning 26 [unused-var]): unused variable helper_binders.\n'
} > "$TMP/warn-after-suites.log"
"$CNT" "$TMP/warn-after-suites.log" >/dev/null 2>&1
check "warning-as-error AFTER enough suites to clear the floor is 5, not 0" "$?" "5"

# The control that keeps the widened pattern from becoming the `^File "` mistake
# in the other direction. A bare `^Error ` is NOT an error label: a test that
# prints a line starting with "Error handling..." must still read as a clean run.
{ cat "$FIXTURES/dune-test-sample-log.txt"
  printf 'Error handling works as expected\n'
} > "$TMP/prints-error-word.log"
"$CNT" --min-suites 0 "$TMP/prints-error-word.log" >/dev/null 2>&1
check "  (control) a test PRINTING \"Error handling...\" is still 0" "$?" "0"

# DUNE --verbose COMMAND FAILURES. Under `dune test --verbose` a test binary that
# exits non-zero or dies on a signal is reported not as `Error:` but as
#     Command [17] exited with code 3:
#     Command [17] got signal SEGV:
# so the `Error`-label pattern never matched either. Measured on two real captured
# logs: exit 0 and a clean "46 cases across 6 suites" for runs that FAILED.
# Third instance of this class after `^File "` (too broad) and `^Error:` (too
# narrow) — the heuristic has now been wrong in both directions and in a third
# spelling, which is the argument for --dune-exit being the authority and this
# being only the fallback.
"$CNT" --min-suites 0 "$FIXTURES/dune-verbose-command-exit-log.txt" >/dev/null 2>&1
check "a --verbose command exiting non-zero is 5, not 0" "$?" "5"

"$CNT" --min-suites 0 "$FIXTURES/dune-verbose-signal-log.txt" >/dev/null 2>&1
check "a --verbose command killed by a signal is 5, not 0" "$?" "5"

# Both controls that keep this widening from becoming the `^File "` mistake again.
# `exited with code 0` is dune reporting SUCCESS and must not read as a failure;
# an INDENTED mention is test output, not dune's own report.
{ cat "$FIXTURES/dune-test-sample-log.txt"
  printf 'Command [17] exited with code 0:\n'
  printf '  a test printed: Command [3] got signal handling right\n'
} > "$TMP/verbose-benign.log"
"$CNT" --min-suites 0 "$TMP/verbose-benign.log" >/dev/null 2>&1
check "  (control) 'exited with code 0' + an indented mention stay 0" "$?" "0"

# Argument handling: a malformed invocation is an error, not a default.
"$CNT" --min-suites nope "$TMP/plausible.log" >/dev/null 2>&1
check "non-numeric --min-suites exits 2" "$?" "2"
"$CNT" --dune-exit nope "$TMP/plausible.log" >/dev/null 2>&1
check "non-numeric --dune-exit exits 2" "$?" "2"

# GIVEN-EMPTY IS NOT NOT-GIVEN. `--dune-exit "$UNSET"` expands to an empty
# string; accepting it would silently demote the authoritative check back to the
# log heuristic and let a failed run exit 0 -- the very shape --dune-exit was
# added to close, reached through the flag that closes it. Found by testing the
# flag rather than by reading it.
"$CNT" --dune-exit "" "$TMP/plausible.log" >/dev/null 2>&1
check "empty --dune-exit exits 2 (not silently ignored)" "$?" "2"
"$CNT" --min-suites 0 "$TMP/mixed.log" >/dev/null 2>&1
check "omitting --dune-exit entirely is still fine" "$?" "0"
"$CNT" --bogus >/dev/null 2>&1
check "unknown option exits 2" "$?" "2"
"$CNT" "$TMP/plausible.log" "$TMP/mixed.log" >/dev/null 2>&1
check "two log files exit 2" "$?" "2"

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
echo "OK: test-suite-counts.sh counts singular/zero/plural Alcotest epilogues,"
echo "    separates qcheck, fails closed when the log format drifts, and"
echo "    refuses to report a count for an empty, truncated, unrecognisable or"
echo "    implausibly small log (backlog-150); and refuses to pass for green"
echo "    off a run that failed -- build failure or non-zero --dune-exit is 5,"
echo "    failing cases are 4, and a died run outranks its own FAIL tally"
echo "    (backlog-157)"
