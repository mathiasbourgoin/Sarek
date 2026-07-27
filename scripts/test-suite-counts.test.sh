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

out="$("$CNT" "$TMP/mixed.log")"

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
out2="$("$CNT" "$TMP/redskip.log")"
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
"$CNT" "$TMP/drift.log" >/dev/null 2>&1
check "unparseable epilogue exits 2, not a wrong total" "$?" "2"

# Missing file is an error, not a silent zero.
"$CNT" "$TMP/nope.log" >/dev/null 2>&1
check "missing log exits 2" "$?" "2"

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
echo "OK: test-suite-counts.sh counts singular/zero/plural Alcotest epilogues,"
echo "    separates qcheck, and fails closed when the log format drifts"
