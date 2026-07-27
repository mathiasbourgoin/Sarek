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
# gone red. The 4 accounts for exactly the 23 SKIPs' worth of CUDA-less
# hardware on the machine that produced the log, and it is expected to be 0 on
# a runner with a CUDA device.
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
# Usage:
#   scripts/test-suite-counts.sh [--min-suites N] <logfile>
#   dune test --force 2>&1 | scripts/test-suite-counts.sh [--min-suites N]
#
#   --min-suites N   floor below which a total is treated as an invocation
#                    problem (exit 3). 0 disables the floor; use it when
#                    counting a deliberately small log, as the covering test
#                    does.
#
# Exit codes:
#   0  counts printed and trustworthy
#   2  usage error: no such file, no recognisable test output, or a log format
#      this script can no longer parse. NOT a result.
#   3  counts printed but below the plausibility floor. NOT a result either.
#
# NOTE: `dune test` without --force prints nothing for suites whose results are
# cached, so an incremental run will under-report. Always --force for a total.
set -euo pipefail

MIN_SUITES=10
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
    -h|--help)
      cat <<'USAGE'
usage: scripts/test-suite-counts.sh [--min-suites N] <logfile>
   or: dune test --force 2>&1 | scripts/test-suite-counts.sh [--min-suites N]

  --min-suites N   plausibility floor (default 10); 0 disables it.

exit 0  counts printed and trustworthy
exit 2  usage error, or no recognisable test output -- NOT a result
exit 3  counts printed but below the plausibility floor -- NOT a result
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
text = sys.stdin.read() if src == "-" else open(src, errors="replace").read()

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
PYEOF
)
python3 -c "$PYPROG" "$SRC" "$MIN_SUITES"
