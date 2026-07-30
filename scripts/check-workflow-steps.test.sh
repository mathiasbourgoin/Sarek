#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Red-path test for scripts/check-workflow-steps.sh.
#
# WHY A .test.sh AND NOT A prove-red-spec BLOCK. The subject locates the repo
# with `git rev-parse --show-toplevel`; prove-red.sh's scratch is a bare
# `tempfile.mkdtemp` directory with no `.git`, where it exits 2 and no green
# baseline can be established. This harness builds a throwaway repo per case,
# which also lets each case carry a whole synthetic workflow rather than a patch
# against the real one.
#
# The first case is the ACTUAL defect: the HIP step inserted before the
# negative-case step's `run:` line. A gate verified only against synthetic
# mutations is weaker than one verified against the bug that motivated it.

set -uo pipefail

SUBJECT="$(cd "$(dirname "$0")" && pwd)/check-workflow-steps.sh"
[ -x "$SUBJECT" ] || {
  echo "FAIL: subject not executable: $SUBJECT" >&2
  exit 2
}

pass=0
fail=0

# $1 = case name, $2 = expected exit, $3 = expected message substring
# ("" = none required), $4 = workflow body
check() {
  local name="$1" want="$2" msg="$3" body="$4" d out code
  d="$(mktemp -d)"
  (
    cd "$d" || exit 2
    git init --quiet .
    mkdir -p .github/workflows
  ) >/dev/null 2>&1
  printf '%s' "$body" >"$d/.github/workflows/ci.yml"
  out="$(cd "$d" && bash "$SUBJECT" 2>&1)"
  code=$?
  rm -rf "$d"
  if [ "$code" != "$want" ]; then
    echo "FAIL $name: exit $code, wanted $want"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  if [ -n "$msg" ] && ! printf '%s' "$out" | grep -qF "$msg"; then
    echo "FAIL $name: exit $want as wanted, but message lacked '$msg'"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  echo "PASS $name (exit $code)"
  pass=$((pass + 1))
}

# --- positive control -------------------------------------------------------
# Green first. Without it, a subject that always exits 1 would satisfy every red
# case below and this file would prove nothing.
check "green: well-formed steps" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - name: One
        run: echo one
      - name: Two
        uses: actions/checkout@v4
        with:
          name: not-a-step-name
      - name: Three
        run: |
          echo "a block scalar containing owner: repo: body:"
          echo "which is text, not YAML keys"
'

# --- THE ACTUAL BUG (backlog-186 / PR #383) ---------------------------------
# Inserted before the first step's `run:` line. The negative-case step is left
# with no action; the HIP step gets two, and YAML keeps the LAST — so the step
# named for the HIP gate ran the other script and the HIP gate never executed.
check "red: the exact misordered insertion" 1 "does not do what its name says" 'name: CI
jobs:
  build:
    steps:
      - name: Check every negative case is asserted
      - name: Compile-gate the HIP backend and its tests
        run: dune build @sarek-hip/all

        run: ./scripts/check-negative-case-coverage.sh
'

# Both halves must be reported, not just the surviving one: a fix that gave the
# HIP step its action back but left the first step dead would still be broken.
check "red: the same case names the dead step too" 1 "it does nothing" 'name: CI
jobs:
  build:
    steps:
      - name: Check every negative case is asserted
      - name: Compile-gate the HIP backend and its tests
        run: dune build @sarek-hip/all

        run: ./scripts/check-negative-case-coverage.sh
'

# --- the two component defects on their own ---------------------------------
check "red: a step with no action at all" 1 "it does nothing" 'name: CI
jobs:
  build:
    steps:
      - name: Does nothing
      - name: Fine
        run: echo ok
'

check "red: a duplicated non-action key" 1 "repeats key" 'name: CI
jobs:
  build:
    steps:
      - name: First
        name: Second
        run: echo ok
'

# --- the two exclusions must NOT fire (measured false positives) ------------
# `with: {name: ...}` is an artifact name. The first version of this gate
# reported it as a duplicate step name.
check "green: nested with: keys are not step keys" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: _coverage
'

# A block scalar holding JavaScript whose object literals carry `owner:`/`repo:`
# twice. The first version reported four duplicate keys here.
check "green: repeated keys inside a block scalar are text" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - name: Comment
        uses: actions/github-script@v7
        with:
          script: |
            await gh.rest.issues.create({ owner: a, repo: b, body: c });
            await gh.rest.issues.update({ owner: a, repo: b, body: d });
'

# --- fails closed ----------------------------------------------------------
# No workflow files. Exit 2, never 0: a check with nothing to read must refuse
# rather than report success on a tree it never examined.
d="$(mktemp -d)"
(cd "$d" && git init --quiet .) >/dev/null 2>&1
out="$(cd "$d" && bash "$SUBJECT" 2>&1)"
code=$?
rm -rf "$d"
if [ "$code" = 2 ]; then
  echo "PASS red: no workflow files (exit 2)"
  pass=$((pass + 1))
else
  echo "FAIL red: no workflow files: exit $code, wanted 2"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "check-workflow-steps.test.sh: $pass passed, $fail FAILED"
  exit 1
fi
echo "check-workflow-steps.test.sh: all $pass cases passed"
exit 0
