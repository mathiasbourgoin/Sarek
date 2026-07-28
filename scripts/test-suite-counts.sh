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
# AND IT REFUSES TO ANSWER FOR A RUN THAT DID NOT FINISH (backlog-157)
#
# The floor is not enough, and the measurement that says so was taken on this
# tree. A clean `dune test --force` here is 1697 cases across 117 suites, dune
# exit 0. Append one unbound identifier to sarek/tests/unit/test_soa.ml and
# re-run: dune exits 1 on the compile error, and this script -- the version
# with all three gates above -- printed
#
#     TOTAL    :  1692 cases across 116 suites
#     FAIL     : 0
#
# exit 0. That is 99.7% of the true case count and 99.1% of the true suite
# count, from a run that did not build. Every gate above is satisfied: the log
# is not empty, every epilogue in it parses, and 116 clears any floor a real
# run of this repository could set. The number looks like a number because it
# very nearly is one.
#
# That kills the obvious fix. A floor derived from the true total, or an
# expected-suite count supplied by the caller, would need a tolerance tighter
# than one suite in 117 to catch this -- and a tolerance that tight rejects
# every legitimately scoped run (`dune test spoc/ir`), which is the invocation
# the floor's own escape hatch, --min-suites 0, exists to serve. An instrument
# cannot be both.
#
# What separates the two cases is not the size of the number. It is whether the
# process that produced the log finished, and that fact is not in the counts at
# all. So this script now asks for it twice, from the two places it can be had:
#
#   --runner-exit N   the caller states the runner's exit status. The caller
#                     always knows it and, before backlog-157, always threw it
#                     away. Non-zero is exit 4 and no counts.
#
#   the log itself    dune reports its own failures in the log, unindented and
#                     at column 0. This scan runs whenever the caller has NOT
#                     said `--runner-exit 0`, because --runner-exit cannot be
#                     required: in the pipe form this header recommends first,
#                     `dune test --force 2>&1 | scripts/test-suite-counts.sh`,
#                     the caller has no exit status to pass yet -- PIPESTATUS
#                     only exists after the pipeline it would have to be inside
#                     of. A flag that the advertised invocation structurally
#                     cannot supply is not a flag that can be mandatory.
#
# Neither alone is sufficient and that is why both are here. The marker scan
# misses a runner killed by a signal, a timeout, or an OOM kill, none of which
# print anything; --runner-exit misses every caller who has not been told about
# it, which on the day it lands is all of them.
#
# `--runner-exit 0` SUPPRESSES the scan, and that is deliberate. The two
# sources are not equal evidence: the caller's exit status is the fact itself,
# the scan is a guess at that fact from text. When both are available the fact
# wins. This is not a theoretical concession -- column 0 in this repository's
# logs is NOT reliably dune's own voice, which the paragraph below measures.
# Without the suppression the weaker source could veto the stronger one, and
# the only remedy left to the caller would be to stop passing the flag.
#
# THE MARKER SET, MEASURED (backlog-157 review)
#
# Every pattern below was measured against dune 3.24.1 on this tree, in both
# directions, and the measurements are why the set is what it is. Re-measure
# after a dune upgrade; a frozen fixture cannot notice dune rewording itself.
#
#   `^Error: `        emitted for a compile error. Measured: `dune build` on a
#                     file with an unbound identifier prints
#                     `Error: Unbound value "aaa"` at column 0.
#
#   `^Error (`        emitted for a warning promoted to an error, which is how
#                     this tree's dev profile fails MOST builds. Measured:
#                     appending `let unused_backlog157_probe x = x + 1` to
#                     spoc/registry/test/test_sarek_registry.ml and running
#                     `dune test spoc/ir spoc/registry spoc/framework --force
#                     -j 1` prints
#                     `Error (warning 32 [unused-value-declaration]): ...`
#                     at column 0, dune exit 1. Before this review the script
#                     matched `^Error: ` only, so it read that log as
#                     "46 cases across 6 suites, 0 FAIL", exit 0 -- the exact
#                     backlog-157 defect, surviving the backlog-157 fix. The
#                     fixture is dune-test-warning-as-error-log.txt.
#
#   `^Command [N] exited with code N:` / `^Command [N] got signal SIG:`
#                     emitted under --display verbose for a runtest rule that
#                     failed at RUNTIME. Measured: appending `let () = exit 3`
#                     to the same file and re-running with --display verbose
#                     prints `Command [17] exited with code 3:`. The fixture is
#                     dune-verbose-command-exit-log.txt.
#
# The bracketed job number is not decoration, it is the whole reason this line
# is spelled out here. Until this review the pattern was `Command exited with
# code N.` -- no job number, and a period instead of a colon. That form matches
# NOTHING dune 3.24.1 emits in any display mode (default, short, progress,
# verbose), so the alternative had been dead since it was written. `Command got
# signal .*` was dead for the same reason (real: `Command [4] got signal SEGV:`),
# and `Had N errors` is a dune 2.x form this version never prints at all --
# measured with `dune build`, `dune build -k` and `dune build --display short
# -k` over two independently-broken modules. All three are gone.
#
# Note what that leaves uncovered, because it is a real hole and not an
# oversight: in dune 3.24.1's DEFAULT display a runtest rule that fails at
# runtime prints no column-0 marker whatsoever -- only a
# `File "…/dune", lines …:` header and the action's own output. Measured on a
# five-line probe project and again on this tree. The scan cannot see that
# case; --runner-exit is the only defence for it. `^File "` is not a marker
# candidate: the same shape introduces ordinary non-fatal warnings.
#
# The scan's false-positive risk was measured too, not asserted: on the
# 4793-line log of the clean run above, and on a 6000-line full-suite log from
# a machine with a GPU, the pattern set matches zero lines. The tempting wider
# form `^Error\b` matches 65 lines in that GPU log -- `Error executing LLVM
# compilation action.` from pocl, and `Error on <device>: ...` from
# sarek/tests/e2e/Benchmarks.ml:247, which prints exactly that for a GPU error
# it then TOLERATES. Column 0 is not reliably dune's voice on this tree: dune
# passes an e2e rule's own stdout straight through, unindented. Hence
# `^Error(?::| \()` and not `^Error\b` -- do not re-widen it.
#
# Usage:
#   scripts/test-suite-counts.sh [--min-suites N] [--runner-exit N] <logfile>
#   dune test --force 2>&1 | scripts/test-suite-counts.sh [--min-suites N]
#
#   --min-suites N   floor below which a total is treated as an invocation
#                    problem (exit 3). 0 disables the floor; use it when
#                    counting a deliberately small log, as the covering test
#                    does.
#
#   --runner-exit N  exit status of the command that produced the log. Non-zero
#                    means the run did not complete, so no counts are printed
#                    and the exit is 4. Zero is the caller ASSERTING that the
#                    run completed, and suppresses the marker scan. Pass it
#                    whenever you have it:
#                        dune test --force >log 2>&1; rc=$?
#                        scripts/test-suite-counts.sh --runner-exit "$rc" log
#                    In the pipe form use `${PIPESTATUS[0]}` afterwards, or
#                    redirect to a file and use the two-line form above.
#                    Given-but-empty (`--runner-exit ""`, which is what the
#                    two-line form above degrades to if `rc` is unset) is a
#                    USAGE ERROR (exit 2), not a fallback to "not given".
#
# Exit codes:
#   0  counts printed and trustworthy
#   2  usage error: no such file, no recognisable test output, or a log format
#      this script can no longer parse. NOT a result.
#   3  counts printed but below the plausibility floor. NOT a result either.
#   4  the run that produced this log did not complete -- the caller said so
#      via --runner-exit, or the log carries the runner's own failure report.
#      No counts are printed. NOT a result, and distinct from 2 and 3 because
#      the remedy is different: fix the build, do not re-invoke the counter.
#
# NOTE: `dune test` without --force prints nothing for suites whose results are
# cached, so an incremental run will under-report. Always --force for a total.
set -euo pipefail

MIN_SUITES=10
# Empty means "the caller did not say". That is not the same as 0, and the two
# must not collapse: 0 is an assertion that the run completed, empty is the
# absence of one. Only the marker scan defends the second case.
RUNNER_EXIT=""
# ...and "the caller did not pass the flag" is a THIRD state, tracked
# separately, because the flag's value is a string and the empty string is a
# legal thing to type. See the validation below.
RUNNER_EXIT_GIVEN=0
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
    --runner-exit)
      [ $# -ge 2 ] || { echo "ERROR: --runner-exit needs a value" >&2; exit 2; }
      RUNNER_EXIT="$2"
      RUNNER_EXIT_GIVEN=1
      shift 2
      ;;
    --runner-exit=*)
      RUNNER_EXIT="${1#--runner-exit=}"
      RUNNER_EXIT_GIVEN=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
usage: scripts/test-suite-counts.sh [--min-suites N] [--runner-exit N] <logfile>
   or: dune test --force 2>&1 | scripts/test-suite-counts.sh [--min-suites N]

  --min-suites N   plausibility floor (default 10); 0 disables it.
  --runner-exit N  exit status of the command that produced the log. Non-zero
                   means the run did not complete: no counts, exit 4. Zero
                   asserts that it did, and suppresses the marker scan. An
                   empty value is a usage error, not "not given".

exit 0  counts printed and trustworthy
exit 2  usage error, or no recognisable test output -- NOT a result
exit 3  counts printed but below the plausibility floor -- NOT a result
exit 4  the run that produced the log did not complete -- NOT a result
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

# "" is the sentinel the parser reads as "the caller did not say", so a caller
# who DID say, but said nothing, must not land on it. Measured on this branch
# before the fix:
#
#     scripts/test-suite-counts.sh --runner-exit "" plausible.log
#     TOTAL    :    60 cases across  12 suites   ... exit 0
#
# -- the gate silently absent, and the same for `--runner-exit=`. That is not a
# hypothetical typo. It is what the two-line form this header recommends
# degrades to when `rc` is unset or misspelled:
#
#     dune test --force >log 2>&1; rc=$?
#     scripts/test-suite-counts.sh --runner-exit "$RC" log   # <-- wrong name
#
# A gate whose caller believes it is armed and which is not is worse than no
# gate. Given-but-empty is never a fallback -- the same rule the empty-log
# refusal above is built on, and this was a live instance of breaking it.
if [ "$RUNNER_EXIT_GIVEN" = 1 ]; then
  case "$RUNNER_EXIT" in
    '')
      echo "ERROR: --runner-exit was given an empty value." >&2
      echo "       That is not the same as omitting the flag: omitting it" >&2
      echo "       leaves the marker scan as the only defence, whereas an" >&2
      echo "       empty value usually means the variable you passed is unset" >&2
      echo "       or misspelled, and the gate you think you armed is not." >&2
      echo "       Pass the runner's actual exit status, or omit the flag." >&2
      exit 2
      ;;
    *[!0-9]*) echo "ERROR: --runner-exit must be a non-negative integer, got: $RUNNER_EXIT" >&2; exit 2 ;;
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
runner_exit = sys.argv[3]  # "" when the caller did not say

# newline="" / .buffer: NO universal-newline translation, and that is not
# incidental. Python's text mode rewrites a bare \r to \n, and this repository's
# logs are full of bare \r -- qcheck's progress bar redraws one line per
# generation. The first draft of gate 0 read the log in text mode and reported
# the compile error in the 4793-line sample at "line 301"; grep says 240, and
# the difference is exactly the 61 carriage returns before it. Worse than a
# wrong number in a message: translation also manufactures line starts, so a
# `^`-anchored pattern would match mid-progress-bar text that dune never
# emitted at column 0, which is the entire premise of the scan below.
if src == "-":
    text = sys.stdin.buffer.read().decode(errors="replace")
else:
    text = open(src, errors="replace", newline="").read()

what = "stdin" if src == "-" else src

# GATE 0 -- did the run that produced this log actually finish? (backlog-157)
#
# First, because it is the only gate that can be true of a log every other gate
# accepts. A `dune test` that dies on a compile error still emits, verbatim and
# parseable, the epilogue of every suite that ran before the failure: on this
# tree that measured 1692 cases across 116 suites out of a true 1697/117, with
# 0 FAIL and exit 0. Gates 1-3 have nothing to object to. The missing fact is
# not in the counts.
#
# Two independent sources for it, because neither covers the other's blind
# spot. See the header: --runner-exit cannot be mandatory (the pipe form has no
# exit status to give), and the marker scan cannot see a signal or an OOM kill.
if runner_exit not in ("", "0"):
    print(
        f"ERROR: the runner exited {runner_exit}, so {what} is a log of a run "
        "that did not complete.",
        file=sys.stderr,
    )
    print(
        "       The suites that did run would parse perfectly and the total\n"
        "       would look like a total -- but an unknown number of suites\n"
        "       never ran, and '0 FAIL' over an unknown denominator is not a\n"
        "       measurement. No counts are printed. Fix the run, then count.",
        file=sys.stderr,
    )
    sys.exit(4)

# The caller stating `--runner-exit 0` is an assertion that the run completed,
# from the one place that actually knows. It outranks the scan, which only
# infers the same fact from text -- see THE MARKER SET in the header for why
# column 0 on this tree is not reliably dune's own voice (Benchmarks.ml prints
# unindented `Error on <device>: ...` for a GPU failure it then tolerates).
# Skipping the scan here is what keeps the weaker source from vetoing the
# stronger one.
#
# Every pattern below is a form dune 3.24.1 was MEASURED emitting; the header
# records each measurement and the three dead alternatives that were removed.
# `^Error(?::| \()` and not `^Error\b`: the wider form matches 65 lines of
# ordinary tolerated-GPU-error test output in a real full-suite log.
runner_failure = None
if runner_exit != "0":
    runner_failure = re.search(
        r"^(?:"
        r"Error(?::| \().*"
        r"|Command \[\d+\] (?:exited with code \d+|got signal \S+):.*"
        r")$",
        text,
        re.MULTILINE,
    )
if runner_failure is not None:
    line_no = text.count("\n", 0, runner_failure.start()) + 1
    quoted = runner_failure.group(0)
    if len(quoted) > 100:
        quoted = quoted[:97] + "..."
    print(
        f"ERROR: {what} is a log of a run that did not complete -- the runner "
        f"reported its own failure at line {line_no}:",
        file=sys.stderr,
    )
    print(f"           {quoted}", file=sys.stderr)
    print(
        "       Every suite that ran before that point parses perfectly, so a\n"
        "       count here looks like a result and is not one: an unknown\n"
        "       number of suites never ran. No counts are printed.\n"
        "       If this really is test output and not the runner's, the\n"
        "       pattern in scripts/test-suite-counts.sh needs narrowing --\n"
        "       do not widen the caller's tolerance instead.",
        file=sys.stderr,
    )
    sys.exit(4)

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
suites = len(alco) + len(qcheck)
if epilogues == 0 and suites == 0:
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
PYEOF
)
python3 -c "$PYPROG" "$SRC" "$MIN_SUITES" "$RUNNER_EXIT"

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
# `failed-build-log` and `runner-exit-nonzero` pin 4 for the same reason, and
# with more at stake: 4 is the code whose whole purpose is to be a DIFFERENT
# answer from 0 on a log that gates 1-3 are content with.
#
# The fixture `dune-test-failed-build-log.txt` is a real log, not a written
# one. It is the complete, unedited output of
#
#     dune test spoc/ir spoc/registry spoc/framework --force -j 1
#
# on this tree with one unbound identifier appended to
# spoc/registry/test/test_sarek_registry.ml. Dune exited 1. Six suites ran and
# their epilogues are intact: this script, before backlog-157, read it as
# "46 cases across 6 suites, 0 FAIL", exit 0. Regenerating it is that one
# command; nothing about it is hand-authored, which is the point -- a
# hand-written failure log is a guess at what dune prints, and the gate below
# is a claim about what dune actually prints.
#
# The other two fixtures are real logs of the same shape, produced the same
# way and added by the backlog-157 review, which measured both of them going
# GREEN (exit 0, "46 cases across 6 suites, 0 FAIL") against the first version
# of this gate:
#
#   dune-test-warning-as-error-log.txt
#       `dune test spoc/ir spoc/registry spoc/framework --force -j 1` with
#       `let unused_backlog157_probe x = x + 1` appended to
#       spoc/registry/test/test_sarek_registry.ml. Dune exit 1. Its failure
#       report is `Error (warning 32 [unused-value-declaration]): ...`, which
#       a `^Error: ` pattern does not match -- and warnings-as-errors is how
#       this tree's dev profile fails most builds, so this was the common case,
#       not the exotic one.
#
#   dune-verbose-command-exit-log.txt
#       the same command with `let () = exit 3` appended instead and
#       `--display verbose` added. Dune exit 1, and its report is
#       `Command [17] exited with code 3:` -- with a job number and a colon,
#       which is why the pattern this gate shipped with (`Command exited with
#       code N.`) matched nothing dune actually writes.
#
# Neither is hand-authored, and that is the point twice over: a hand-written
# failure log is a guess at what dune prints, and both of these exist because
# the previous guess was wrong.
#
# BEGIN prove-red-spec
# copy: scripts/test-suite-counts.sh
# copy: scripts/prove-red-fixtures/dune-test-sample-log.txt
# copy: scripts/prove-red-fixtures/dune-test-failed-build-log.txt
# copy: scripts/prove-red-fixtures/dune-test-warning-as-error-log.txt
# copy: scripts/prove-red-fixtures/dune-verbose-command-exit-log.txt
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
# mutation: failed-build-log
#   desc: a genuine `dune test` log from a build that failed -- dune exited 1, six suites had already run and every epilogue parses. This is backlog-157: gates 1-3 are all satisfied and the pre-fix answer was "46 cases across 6 suites, 0 FAIL", exit 0. Exit 4, no counts.
#   argv: scripts/prove-red-fixtures/dune-test-failed-build-log.txt --min-suites 0
#   expect-exit: 4
#   expect-message: a log of a run that did not complete
#
# mutation: runner-exit-nonzero
#   desc: the caller states the runner's exit status and it is not zero. Same clean log as the baseline, which passes every other gate -- so this pins the flag alone, with no help from the marker scan. It is the half that covers a runner killed by a signal or a timeout, which prints no marker at all.
#   argv: scripts/prove-red-fixtures/dune-test-sample-log.txt --min-suites 0 --runner-exit 1
#   expect-exit: 4
#   expect-message: the runner exited 1
#
# mutation: warning-as-error-log
#   desc: a real `dune test` log whose build died on a warning promoted to an error -- `Error (warning 32 ...)`, which is how this tree's dev profile fails most builds. The first version of this gate matched `^Error: ` only and read this log as "46 cases across 6 suites, 0 FAIL", exit 0. It is here so a narrowing of the pattern cannot silently un-cover the common case.
#   argv: scripts/prove-red-fixtures/dune-test-warning-as-error-log.txt --min-suites 0
#   expect-exit: 4
#   expect-message: a log of a run that did not complete
#
# mutation: verbose-command-exit-log
#   desc: a real --display verbose log of a runtest rule that failed at RUNTIME rather than at compile time. Dune's wording is `Command [17] exited with code 3:`; the pattern this gate shipped with was `Command exited with code N.`, which matches nothing dune 3.24.1 emits in any display mode. This pins the measured wording against the guessed one.
#   argv: scripts/prove-red-fixtures/dune-verbose-command-exit-log.txt --min-suites 0
#   expect-exit: 4
#   expect-message: a log of a run that did not complete
#
# mutation: runner-exit-empty
#   desc: the flag given with an empty value -- what `--runner-exit "$rc"` becomes when `rc` is unset or misspelled. Measured on this branch before the review: the gate silently vanished and the counter exited 0. Given-but-empty is a usage error (2), never a fallback to "not given", and 2 is asserted exactly so a regression to "not given" (0) or to the failure path (4) both stay red.
#   argv: scripts/prove-red-fixtures/dune-test-sample-log.txt --min-suites 0 --runner-exit=
#   expect-exit: 2
#   expect-message: given an empty value
# END prove-red-spec
# ---------------------------------------------------------------------------
