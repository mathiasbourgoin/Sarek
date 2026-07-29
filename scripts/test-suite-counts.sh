#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Canonical test-suite counts from a `dune test` log (#143/#147 follow-up).
#
# WHY THIS EXISTS
#
# Two agents measured the same commit and reported different totals -- 1436
# cases across 84 suites versus 1447 across 99. Neither was lying and neither
# double-counted. The lower pair came from grepping `[0-9]+ tests run`, and
# **Alcotest singularises**: a suite that runs exactly one case prints
# "1 test run.", and one that runs none prints "0 test run.". A plural-only
# pattern silently drops every such suite -- here 15 suites and 11 cases.
#
# That matters more than bookkeeping. "0 FAIL out of N" is exactly as
# trustworthy as N, so in a repo whose discipline is "prove a gate can fail",
# two disagreeing answers to "how many tests are there" undermine every green
# result reported alongside them. This script is the single answer, so the next
# person reports a comparable number instead of inventing a sixteenth grep.
#
# It also surfaces zero-case suites. A suite printing "Test Successful" having
# run nothing is the "gate that cannot fail" shape: usually legitimate
# hardware gating, but never something to discover by accident.
#
# THE FOUR ZERO-CASE SUITES, RESOLVED (backlog-150)
#
# `zero-case suites: 4` has appeared in this output on every branch since the
# script was written, and had never been run down. They are ptx_stride_spike,
# ptx_atomics_probe, ptx_mma_probe (one case each) and cuda_f16_sass (two).
# All four are genuine hardware gates, not inert suites: each registers real
# Alcotest cases and calls `Alcotest.skip ()` when `Cuda_api.is_driver_available
# ()` is false or no ptxas is present. Alcotest's "N tests run" counts only
# cases that actually EXECUTED, so an all-skipped suite reports 0 -- compare
# ptx_external in the same run, 1 OK + 1 SKIP, which reports "1 test run.".
#
# Verified by removing the gate from test_ptx_stride_spike and re-running: the
# suite reported "1 failure! in 0.000s. 1 test run." (cuInit, no driver), which
# is the positive control -- a suite with no registered cases could not have
# gone red. Between them the four register FIVE cases, so they account for 5 of
# the 23 SKIPs on the machine that produced the log; the other 18 come from
# suites that also ran real cases and so are not zero-case. All four are
# expected to report cases on a runner with a CUDA device.
#
# This is NOT the defect this repo has seen before, where a `Printf` "[SKIP]"
# was swallowed by Alcotest's stdout capture and the case rendered [OK]. These
# four print a [SKIP] line AND return skip status to the runner, which is why
# they show as [SKIP] rather than [OK]. test_ptx_mma_probe.ml:166-169 documents
# that distinction at the call site.
#
# AND IT REFUSES TO ANSWER FROM NOTHING (backlog-150)
#
# The version of this script that shipped with the paragraphs above accepted an
# empty log and printed "0 cases / 0 FAIL / 0 SKIP", exit 0. A fully-cached
# `dune test` (no --force) produces exactly that: no output at all. So did a
# log truncated before the first suite finished, and so did a file that was not
# a test log in the first place. Every one of them rendered as a green run.
#
# That is the "gate given nothing to check" shape -- in the instrument this
# repo uses to audit its other gates, which makes it worse than an ordinary
# one. The whole argument above is that "0 FAIL out of N" is only as
# trustworthy as N; an N of zero is not a small N, it is the absence of a
# measurement, and it must not be reported in the same shape as a result.
#
# So: a log with no recognisable suite epilogue is a USAGE ERROR (exit 2) and
# prints no counts at all. Given-but-empty is never a fallback.
#
# There is also a plausibility floor. A run of this repository that reports a
# handful of suites is a caching or invocation problem, not a result -- the
# whole tree is ~112 suites, and the failure mode is partial caching, which
# yields a plausible-looking-but-wrong number rather than nothing. Below
# --min-suites (default 10) the counts are printed *and* the exit is 3, so the
# caller sees what was parsed and is still told not to quote it.
#
# AND A GREEN COUNT OFF A RUN THAT FAILED (backlog-157)
#
# Gates 1-3 all ask "is this total complete", and none of them asked the prior
# question: did the run this log came from SUCCEED. Two ways it did not, and
# neither was caught.
#
# A BUILD FAILURE PART-WAY THROUGH. `dune test --force` builds and runs suite by
# suite, so a compile error in one test leaves a log with every earlier suite's
# epilogue intact. Measured on this repo: breaking one test file yields 45 parsed
# suites, gate 2 satisfied (45 epilogues, 45 counts), gate 3 satisfied (45 > 10)
# -- a plausible total, exit 0, off a `dune test` that exited 1. The number is
# not wrong for what it counted; it is wrong as an answer to "how many tests are
# there", and that is the question it gets asked.
#
# FAILING CASES. `FAIL : 3` printed with exit 0 means
# `test-suite-counts.sh log && echo all good` prints "all good". The counts are
# honest; the exit code is not, because the shell reads only the code.
#
# So: failing cases are exit 4, and a run that did not complete is exit 5. The
# counts still print in both cases -- they are useful, they just must not pass
# for green in a chain.
#
# PRECEDENCE, and it is load-bearing. Exit 5 is checked FIRST and outranks 2, 3
# and 4. The first version of this change put the completion check last, which
# made the claim "--dune-exit is authoritative" false: a build error before the
# first suite epilogue tripped the empty-log gate (exit 2, "not a test log"),
# and one after only a few suites tripped the floor gate (exit 3, "caching or
# invocation problem"). Neither was a false green, but both told the caller to
# re-run when the correct instruction was to fix the build -- and a fact three
# earlier gates can pre-empt is a fallback, not an authority. Caught by review
# on PR #357, not by me.
#
# The consequence to state plainly: a run that died with failing cases ALREADY
# reported is exit 5, not 4. That is deliberate. Its FAIL tally is taken over a
# partial run, so 4 ("complete, N failures") would understate it.
#
# The build-failure check is a LOG HEURISTIC (an `Error:` or `Error (<...>):`
# label at column 0, which is dune's own formatting; the two spellings are
# argued below), and heuristics in a gate are what this file
# spends 60 lines arguing against. So it is the fallback, not the mechanism:
# --dune-exit lets the caller pass the one authoritative fact the script cannot
# derive from a log. Measured before choosing the pattern: 0 matches in a green
# 4885-line log and 0 in a log of genuinely FAILING alcotest cases (alcotest
# indents, dune does not), so the heuristic does not confuse a red test with a
# broken build -- which is exactly why they are separate exit codes.
#
# The pattern was `^Error:|^File "` for one revision and that was too broad:
# dune prints the `File "..."` line above a WARNING too, and above a truncated
# preamble, so a tree that merely warns read as a failed run. This file's own
# trunc-early fixture caught it.
#
# Then it was `^Error:` alone, and THAT was too narrow — the correction
# overshot. A warning promoted to an error (this repo builds with warnings as
# errors) is printed by the compiler as
#
#     Error (warning 32 [unused-value-declaration]): unused value foo.
#
# which has no colon after `Error`, so `^Error:` never matched it. Measured on a
# real captured log of exactly that failure: exit 0 and a clean
# "46 cases across 6 suites" for a run whose build never completed — the precise
# false green this gate exists to prevent, reintroduced by its own fix.
#
# The pattern therefore accepts `Error:` and `Error (<...>):`, and NOT a bare
# `^Error ` — a test that prints "Error handling works" starts a line with
# `Error ` and must not be read as a broken build. Requiring the closing `):`
# keeps the parenthesised form tight.
#
# Usage:
#   scripts/test-suite-counts.sh [--min-suites N] [--dune-exit N] <logfile>
#   dune test --force 2>&1 | scripts/test-suite-counts.sh [--min-suites N]
#
#   RECOMMENDED, because it carries dune's verdict instead of inferring it:
#     dune test --force > t.log 2>&1; scripts/test-suite-counts.sh --dune-exit $? t.log
#
#   The pipe form CANNOT see dune's exit status -- in `a | b` the shell reports
#   b's code, and `set -o pipefail` is the caller's to set, not this script's.
#   It stays supported and falls back to the log heuristic.
#
#   --min-suites N   floor below which a total is treated as an invocation
#                    problem (exit 3). 0 disables the floor; use it when
#                    counting a deliberately small log, as the covering test
#                    does.
#   --dune-exit N    the exit status of the `dune test` that produced this log.
#                    Non-zero means the run failed, whatever the log parses to.
#
# Exit codes:
#   0  counts printed and trustworthy
#   2  usage error: no such file, no recognisable test output, or a log format
#      this script can no longer parse. NOT a result.
#   3  counts printed but below the plausibility floor. NOT a result either.
#   4  counts printed and complete, but the run had FAILING CASES. A count, not
#      a pass.
#   5  the run DID NOT COMPLETE -- a build failure in the log, or a non-zero
#      --dune-exit. Any count printed is only the suites that ran before it
#      died. CHECKED FIRST: 5 outranks 2, 3 and 4, so a failed run reports 5
#      even when the log is empty, below the floor, or already shows failing
#      cases.
#
# NOTE: `dune test` without --force prints nothing for suites whose results are
# cached, so an incremental run will under-report. Always --force for a total.
set -euo pipefail

MIN_SUITES=10
DUNE_EXIT=""
DUNE_EXIT_GIVEN=""
SRC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --min-suites)
      [ $# -ge 2 ] || { echo "ERROR: --min-suites needs a value" >&2; exit 2; }
      MIN_SUITES="$2"
      shift 2
      ;;
    --min-suites=*)
      MIN_SUITES="${1#--min-suites=}"
      shift
      ;;
    --dune-exit)
      [ $# -ge 2 ] || { echo "ERROR: --dune-exit needs a value" >&2; exit 2; }
      DUNE_EXIT="$2"
      DUNE_EXIT_GIVEN=1
      shift 2
      ;;
    --dune-exit=*)
      DUNE_EXIT="${1#--dune-exit=}"
      DUNE_EXIT_GIVEN=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
usage: scripts/test-suite-counts.sh [--min-suites N] [--dune-exit N] <logfile>
   or: dune test --force 2>&1 | scripts/test-suite-counts.sh [--min-suites N]

recommended, so the counter is told dune's verdict rather than inferring it:
  dune test --force > t.log 2>&1; scripts/test-suite-counts.sh --dune-exit $? t.log

  --min-suites N   plausibility floor (default 10); 0 disables it.
  --dune-exit N    exit status of the `dune test` that produced the log.

exit 0  counts printed and trustworthy
exit 2  usage error, or no recognisable test output -- NOT a result
exit 3  counts printed but below the plausibility floor -- NOT a result
exit 4  counts complete but the run had FAILING CASES -- a count, not a pass
exit 5  the run DID NOT COMPLETE (build failure / non-zero --dune-exit) -- the
        total is a partial count of the suites that ran before it died
USAGE
      exit 0
      ;;
    -)
      SRC="-"
      shift
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      exit 2
      ;;
    *)
      [ -z "$SRC" ] || { echo "ERROR: more than one log file given" >&2; exit 2; }
      SRC="$1"
      shift
      ;;
  esac
done

case "$MIN_SUITES" in
  ''|*[!0-9]*) echo "ERROR: --min-suites must be a non-negative integer, got: $MIN_SUITES" >&2; exit 2 ;;
esac

# Validated rather than passed through, and "given empty" is NOT "not given".
# `--dune-exit "$SOME_UNSET_VAR"` yields an empty string; accepting it would
# silently demote an authoritative check to the log heuristic and let a failed
# run exit 0 -- the exact shape this file exists to refuse, reached through the
# flag added to prevent it. So a present-but-unparseable value is a usage error;
# only an ABSENT flag means "nobody asked dune".
if [ -n "${DUNE_EXIT_GIVEN:-}" ]; then
  case "$DUNE_EXIT" in
    ''|*[!0-9]*)
      echo "ERROR: --dune-exit must be a non-negative integer, got: '$DUNE_EXIT'" >&2
      exit 2
      ;;
  esac
fi

SRC="${SRC:--}"
if [ "$SRC" != "-" ] && [ ! -f "$SRC" ]; then
  echo "ERROR: no such log file: $SRC" >&2
  exit 2
fi

# Reading from a terminal is never what the caller meant, and hanging on an
# invisible read is how "the counter produced no output" gets misread as zero.
if [ "$SRC" = "-" ] && [ -t 0 ]; then
  echo "ERROR: no log file given and stdin is a terminal." >&2
  echo "       usage: scripts/test-suite-counts.sh [--min-suites N] <logfile>" >&2
  echo "          or: dune test --force 2>&1 | scripts/test-suite-counts.sh" >&2
  exit 2
fi

# The parser is captured into a variable rather than fed to `python3 -` on a
# heredoc, and that is load-bearing (backlog-150).
#
# `python3 - <<'PYEOF'` hands python its PROGRAM on stdin, which leaves nothing
# on stdin for the program to read. The advertised pipe form,
#
#     dune test --force 2>&1 | scripts/test-suite-counts.sh
#
# therefore never worked: `sys.stdin.read()` returned "" and the script
# reported "0 cases across 0 suites / 0 FAIL", exit 0, for a genuine 1644-case
# log. It was the same "green from nothing" this file now refuses -- arrived at
# by a different route, and reached by the usage the header recommends first.
# Keep the program in a variable so stdin belongs to the caller's pipe.
PYPROG=$(cat <<'PYEOF'
import re
import sys

src = sys.argv[1]
min_suites = int(sys.argv[2])
# "" when the caller did not pass --dune-exit. Absent is not the same as 0: one
# is "dune said it succeeded", the other is "nobody asked dune".
dune_exit = sys.argv[3] if len(sys.argv) > 3 else ""
text = sys.stdin.read() if src == "-" else open(src, errors="replace").read()

# Dune reports a compile error with an `Error` label at column 0 -- `Error: ...`
# or `Error (warning N [...]): ...` -- while alcotest indents everything it
# prints. Measured before the pattern was chosen: 0
# matches in a green 4885-line log AND 0 in a log of genuinely FAILING alcotest
# cases, so this separates "the build died" from "a test failed" — which is why
# they get different exit codes rather than one catch-all.
#
# An `Error` LABEL at column 0, deliberately not `^File "`. The first version
# also matched `^File "`, on the reasoning that dune prints `File "x.ml", line 1:`
# above the error. It does — but it prints the same line above a WARNING, and
# above a truncated preamble. This file's own trunc-early fixture opens with
# `File "sarek/tests/unit/dune", line 1, characters 0-0:` and started reporting
# exit 5 for a log whose build never failed. A tree that merely warns would have
# been called a failed run. The error line is the one that means an error; the
# File line means dune has something to say about a file.
#
# Both spellings of the label, because narrowing to `^Error:` alone missed
# warning-as-error ("Error (warning 32 [...]): ...") and reintroduced a false
# green — see the header. A bare `^Error ` is NOT accepted: a test printing
# "Error handling works" would match it.
build_errors = re.findall(
    r"(?m)^(?:Error(?::| \([^)\n]*\):)"
    r"|Command \[\d+\] got signal \w+"
    r"|Command \[\d+\] exited with code (?!0\b)\d+)",
    text,
)

# Alcotest: "Test Successful in 0.004s. 61 tests run." and the SINGULAR
# "1 test run." / "0 test run.". Matching `tests?` is the whole point.
alco = [int(m) for m in re.findall(r"(\d+) tests? run", text)]
# qcheck-core runner: "success (ran 12 tests)"
qcheck = [int(m) for m in re.findall(r"ran (\d+) tests?\)", text)]

fails = len(re.findall(r"\[FAIL\]", text))
skips = len(re.findall(r"\[SKIP\]", text))
zero = alco.count(0)

# Cross-check: every Alcotest suite epilogue should have been parsed. If the
# "Test Successful" count and the "N test(s) run" count disagree, the pattern
# above has drifted again -- say so instead of printing a confident wrong total.
epilogues = len(re.findall(r"Test Successful in", text)) + len(
    re.findall(r"\d+ failures?!", text)
)

suites = len(alco) + len(qcheck)

# GATE 0 -- did the run COMPLETE? (backlog-157, moved ahead of everything by
# CodeRabbit on PR #357)
#
# This was GATE 4, after the empty-log and floor gates, and that ordering made
# the PR's own claim -- "--dune-exit is authoritative" -- FALSE. Two ways:
#
#   a build error before the first suite epilogue leaves nothing to parse, so
#   the empty-log gate fired first and reported exit 2, "not a test log";
#   a build error after only a few suites fired the floor gate and reported
#   exit 3, "caching or invocation problem".
#
# Both are non-zero, so neither was a false green -- but both MISCLASSIFY a
# failed run as a malformed input, and in both the caller was told to re-run
# rather than to fix the build. An authoritative fact that three earlier gates
# can pre-empt is not authoritative; it is a fallback. So it goes first.
#
# Exit 5 therefore OUTRANKS 2, 3 and 4: a run that did not complete is not
# described by any of them, whatever its log happens to parse to.
if dune_exit not in ("", "0") or build_errors:
    if suites:
        print(f"partial: {sum(alco) + sum(qcheck)} case(s) across {suites} suite(s) "
              "ran before the failure")
    else:
        print("partial: no suite completed before the failure")
    if dune_exit not in ("", "0"):
        print()
        print(f"ERROR: the `dune test` that produced this log exited {dune_exit}, so "
              "the run did not succeed and any count above is only the suites that "
              "ran before it stopped. Not a result.")
    else:
        print()
        print(f"ERROR: this log contains {len(build_errors)} dune/compiler error "
              "marker(s) at column 0, so the run did not complete and any count "
              "above is only the suites built before the failure. Fix the build and "
              "re-run; pass --dune-exit to make this authoritative rather than "
              "inferred.")
    sys.exit(5)

# GATE 1 -- did we read a test log at all? (backlog-150)
#
# This runs BEFORE any count is printed, because the failure being prevented is
# a caller reading "0 FAIL" off a run that never happened. A result and the
# absence of a result must not share an output shape.
#
# "Recognisable" is deliberately the union of every suite-level marker, not the
# case-counts alone: a log truncated mid-epilogue has a "Test Successful in"
# with no number after it, and that is a drift/truncation error (gate 2), not
# an empty log. Only when NOTHING matched is the input simply not a test log.
if epilogues == 0 and suites == 0:
    what = "stdin" if src == "-" else src
    print(f"ERROR: no test-suite output recognised in {what}.", file=sys.stderr)
    print(
        "       This is not a result of zero cases -- it is the absence of a\n"
        "       measurement, and it is never reported as a count.\n"
        "       Most likely: `dune test` without --force (a fully-cached run\n"
        "       prints nothing), a truncated or empty log, or a file that is\n"
        "       not a dune test log. Re-run with:\n"
        "           dune test --force 2>&1 | scripts/test-suite-counts.sh",
        file=sys.stderr,
    )
    sys.exit(2)

print(f"alcotest : {sum(alco):5d} cases across {len(alco):3d} suites")
print(f"qcheck   : {sum(qcheck):5d} cases across {len(qcheck):3d} suites")
print(f"TOTAL    : {sum(alco) + sum(qcheck):5d} cases across "
      f"{len(alco) + len(qcheck):3d} suites")
print(f"FAIL     : {fails}")
print(f"SKIP     : {skips}")
print(f"zero-case suites: {zero}")

# GATE 2 -- drift/truncation. Every Alcotest suite epilogue should have been
# parsed. If the "Test Successful" count and the "N test(s) run" count
# disagree, either the format has drifted or the log was cut mid-epilogue --
# say so instead of printing a confident wrong total.
if epilogues != len(alco):
    print()
    print(f"ERROR: {epilogues} Alcotest suite epilogues but {len(alco)} parsed "
          "case-counts -- the log is truncated or the format has drifted, and "
          "this total is not trustworthy. Fix the pattern in "
          "scripts/test-suite-counts.sh.")
    sys.exit(2)

# GATE 3 -- plausibility floor. A partially-cached `dune test` under-reports
# without any of the markers gates 1 and 2 look for: the suites it does print
# parse perfectly, there is just an arbitrary subset of them. That is the
# dangerous shape, because the number looks like a number. The floor cannot
# tell 111 suites from 112, but it does catch the order-of-magnitude case,
# which is what partial caching and a wrong invocation actually produce.
if suites < min_suites:
    print()
    print(f"ERROR: {suites} suites is below the plausibility floor of "
          f"{min_suites}. A run of this repository reports ~112 suites, so "
          "this is a caching or invocation problem, not a result -- do not "
          "quote these numbers. Re-run with `dune test --force`, or pass "
          "--min-suites 0 if you meant to count a small log.")
    sys.exit(3)

# GATE 5 -- were there failing CASES? (backlog-157)
#
# Separate from gate 0 on purpose: a complete run with red tests is a different
# fact from a run that died, and collapsing them would make the exit code unable
# to say which. Printing `FAIL : 3` and exiting 0 meant
# `test-suite-counts.sh log && echo all good` printed "all good" -- the counts
# were honest and the exit code was not, and a shell reads only the code.
if fails > 0:
    print()
    print(f"ERROR: {fails} failing case(s) in this log. The counts above are "
          "complete and trustworthy; this is NOT a passing run, and the exit "
          "code says so because a caller chaining on `&&` reads only that.")
    sys.exit(4)
PYEOF
)
python3 -c "$PYPROG" "$SRC" "$MIN_SUITES" "$DUNE_EXIT"

# ---------------------------------------------------------------------------
# Red-path evidence, executed by scripts/prove-red.sh (backlog-151).
#
# This tool already has a covering test. It is a subject here anyway, and only
# for one reason: two of its three failure modes are mutations of the
# ENVIRONMENT rather than of any file -- an empty stdin and a missing
# --min-suites -- and if the mechanism that is supposed to police the
# gate-vacuous class could only edit source files it would miss the shape it
# was built for. `empty-stdin` is the literal backlog-150 defect: the advertised
# pipe form returned "0 cases / 0 FAIL", exit 0, for a genuine 1644-case log.
#
# `below-floor` pins exit 3 specifically. This script's contract distinguishes
# 2 (not a result) from 3 (a result below the plausibility floor), and an
# assertion that accepted "non-zero" would not notice the two being confused.
#
# BEGIN prove-red-spec
# copy: scripts/test-suite-counts.sh
# copy: scripts/prove-red-fixtures/dune-test-sample-log.txt
# invoke: scripts/test-suite-counts.sh
# baseline-argv: scripts/prove-red-fixtures/dune-test-sample-log.txt --min-suites 0
# baseline-exit: 0
# baseline-message: TOTAL    :    24 cases across   4 suites
#
# mutation: empty-stdin
#   desc: the pipe form with nothing on stdin -- a fully-cached `dune test` prints exactly this. The absence of a measurement must not share an output shape with a result.
#   stdin: empty
#   argv: --min-suites 0
#   expect-exit: 2
#   expect-message: no test-suite output recognised
#
# mutation: truncated-epilogue
#   desc: every suite epilogue loses its case count, as a log cut mid-write does. Three "Test Successful" markers and zero parsed counts must be a refusal, not a confident total of 0.
#   apply: sed -i 's/ [0-9]* tests\? run\.//' scripts/prove-red-fixtures/dune-test-sample-log.txt
#   expect-exit: 2
#   expect-message: the log is truncated or the format has drifted
#
# mutation: below-floor
#   desc: the same log counted without --min-suites 0. Four suites is a plausible-looking number and a partially-cached run produces exactly that, so it is exit 3 -- neither a pass nor the same failure as an unreadable log.
#   argv: scripts/prove-red-fixtures/dune-test-sample-log.txt
#   expect-exit: 3
#   expect-message: below the plausibility floor
#
# mutation: build-failed
#   desc: a dune compile error appended to an otherwise clean log -- the shape `dune test --force` leaves when a suite fails to build part-way through, with every earlier epilogue intact. A plausible total off a run that exited 1 must be exit 5, not a pass.
#   apply: printf 'File "sarek/tests/unit/test_thing.ml", line 12, characters 0-1:\nError: Syntax error\n' >> scripts/prove-red-fixtures/dune-test-sample-log.txt
#   argv: scripts/prove-red-fixtures/dune-test-sample-log.txt --min-suites 0
#   expect-exit: 5
#   expect-message: dune/compiler error marker
#
# mutation: dune-exit-nonzero
#   desc: the log parses clean and dune says the run failed. An ENVIRONMENT mutation, like empty-stdin: no file changes, and the authoritative fact is one a log cannot carry.
#   argv: scripts/prove-red-fixtures/dune-test-sample-log.txt --min-suites 0 --dune-exit 1
#   expect-exit: 5
#   expect-message: did not succeed
#
# mutation: failing-cases
#   desc: a [FAIL] marker in the log. Printing `FAIL : 1` and exiting 0 made `test-suite-counts.sh log && echo ok` print ok -- honest counts, dishonest exit code, and a shell reads only the code. Distinct from build-failed because a complete-but-red run is a different fact from a run that died.
#   apply: sed -i '1i\  [FAIL]        some suite    0   a case....' scripts/prove-red-fixtures/dune-test-sample-log.txt
#   argv: scripts/prove-red-fixtures/dune-test-sample-log.txt --min-suites 0
#   expect-exit: 4
#   expect-message: failing case
#
# mutation: warning-as-error
#   desc: the same shape as build-failed but with the spelling the compiler actually uses when a warning is promoted -- `Error (warning 32 [...]):`, no colon after Error. The `^Error:` pattern missed it entirely and reported a clean total for a run whose build never completed. Kept as its own mutation because build-failed passes on a pattern that has this hole.
#   apply: printf 'File "sarek/ppx/Sarek_lower_ir.ml", line 850, characters 16-30:\nError (warning 26 [unused-var]): unused variable helper_binders.\n' >> scripts/prove-red-fixtures/dune-test-sample-log.txt
#   argv: scripts/prove-red-fixtures/dune-test-sample-log.txt --min-suites 0
#   expect-exit: 5
#   expect-message: dune/compiler error marker
#
# mutation: verbose-command-exit
#   desc: under `dune test --verbose` a test binary that exits non-zero is reported as "Command [N] exited with code M:" -- no Error label at all, so the Error-only pattern read the whole run as clean. Distinct from build-failed because nothing failed to BUILD; a compiled test ran and returned failure.
#   apply: printf 'Command [17] exited with code 3:\n' >> scripts/prove-red-fixtures/dune-test-sample-log.txt
#   argv: scripts/prove-red-fixtures/dune-test-sample-log.txt --min-suites 0
#   expect-exit: 5
#   expect-message: dune/compiler error marker
#
# mutation: verbose-command-signal
#   desc: the same shape for a test killed by a signal -- "Command [N] got signal SEGV:". Its own mutation because a segfaulting test is the case this repo has actually hit (backlog-53/80, RADV driver segfaults) and it must not be reported as a clean count.
#   apply: printf 'Command [17] got signal SEGV:\n' >> scripts/prove-red-fixtures/dune-test-sample-log.txt
#   argv: scripts/prove-red-fixtures/dune-test-sample-log.txt --min-suites 0
#   expect-exit: 5
#   expect-message: dune/compiler error marker
# END prove-red-spec
#
# The other polarity -- a test that PRINTS "Error handling works" must NOT read
# as a broken build -- is deliberately NOT a mutation above. A prove-red mutation
# asserts the subject CATCHES a defect; a case whose correct answer is the
# baseline verdict asserts the opposite, and prove-red.sh's immune-checker
# rejects it by design ("asserts the subject did not notice"). That control lives
# in scripts/test-suite-counts.test.sh, which is where a negative case belongs.
# ---------------------------------------------------------------------------
