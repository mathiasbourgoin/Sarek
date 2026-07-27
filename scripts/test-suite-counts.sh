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
# Usage:
#   scripts/test-suite-counts.sh <logfile>   # parse an existing log
#   dune test --force 2>&1 | scripts/test-suite-counts.sh   # or a pipe
#
# NOTE: `dune test` without --force prints nothing for suites whose results are
# cached, so an incremental run will under-report. Always --force for a total.
set -euo pipefail

SRC="${1:--}"
if [ "$SRC" != "-" ] && [ ! -f "$SRC" ]; then
  echo "ERROR: no such log file: $SRC" >&2
  exit 2
fi

python3 - "$SRC" <<'PYEOF'
import re
import sys

src = sys.argv[1]
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

print(f"alcotest : {sum(alco):5d} cases across {len(alco):3d} suites")
print(f"qcheck   : {sum(qcheck):5d} cases across {len(qcheck):3d} suites")
print(f"TOTAL    : {sum(alco) + sum(qcheck):5d} cases across "
      f"{len(alco) + len(qcheck):3d} suites")
print(f"FAIL     : {fails}")
print(f"SKIP     : {skips}")
print(f"zero-case suites: {zero}")

if epilogues != len(alco):
    print()
    print(f"ERROR: {epilogues} Alcotest suite epilogues but {len(alco)} parsed "
          "case-counts -- the log format has drifted and this total is not "
          "trustworthy. Fix the pattern in scripts/test-suite-counts.sh.")
    sys.exit(2)
PYEOF
