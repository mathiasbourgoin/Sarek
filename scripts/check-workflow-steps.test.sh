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

# --- the block-scalar skip BOUNDARY (CodeRabbit, PR #383) -------------------
# When a step's FIRST key is itself a block scalar (`- run: |`, no preceding
# `name:`), skip_until_indent was set to child_indent - 1 while the sibling-key
# branch used child_indent. The skip test is `cur_ind > skip_until_indent`, so
# the off-by-one swallowed lines AT child_indent -- exactly where sibling keys
# sit. Every key after the block scalar was eaten as block-scalar text,
# including a second action, and this shape reported "OK — 1 steps ... exactly
# one action" at exit 0. The gate's core check, dead, on a shape GitHub accepts.
check "red: first-key block scalar must not swallow a sibling uses:" 1 "2 actions" 'name: CI
jobs:
  build:
    steps:
      - run: |
          echo "the first key IS the block scalar"
        uses: actions/checkout@v4
'

# The duplicate-KEY half of the same hole.
check "red: first-key block scalar must not swallow a duplicate key" 1 "repeats key" 'name: CI
jobs:
  build:
    steps:
      - run: |
          echo hi
        name: A
        name: B
'

# And the boundary must still do its job: a sibling key is at child_indent, the
# scalar TEXT is deeper. A fix that swallowed nothing would re-break the
# JavaScript case above, so pin the discrimination directly.
check "green: first-key block scalar still hides its own deeper text" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - run: |
          echo "owner: a"
          echo "run: not-a-key"
        name: Only one action here
'

# --- dash-items that are NOT steps (CodeRabbit, PR #383) --------------------
# STEP_START matched `^- word:` anywhere in the file, so a matrix `include:`
# entry read as a step and was reported "has neither run: nor uses:" — exit 1 on
# a perfectly valid workflow. This tree's own matrix lists are plain scalars
# (`- ubuntu-latest`, no colon), which is why it never fired here; a gate that
# false-positives gets switched off, and its true findings go with it.
check "green: a matrix include entry is not a step" 0 "OK" 'name: CI
jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            ocaml: 5.4.0
    steps:
      - name: Real step
        run: echo hi
'

# The same hole one level in: a multi-object `with:` value inside a real step.
# Treating `- name: a` as a step both closed the real step early (losing the keys
# after it) and added a phantom actionless step.
check "green: a nested list inside with: is not a step" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - uses: some/action@v1
        with:
          entries:
            - name: a
              value: b
            - name: c
              value: d
      - name: Second
        run: echo hi
'

# Scoping must not go so far that it stops seeing real steps. A duplicate action
# in a step that FOLLOWS a matrix include must still be caught — otherwise
# "scoped to steps:" would be indistinguishable from "stopped scanning".
check "red: a duplicate action after a matrix include is still caught" 1 "2 actions" 'name: CI
jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
    steps:
      - name: Broken
        run: one
        uses: actions/checkout@v4
'

# Two jobs, each with its own steps: list. The second list must be scanned too.
check "red: the steps list of a SECOND job is still scanned" 1 "does nothing" 'name: CI
jobs:
  first:
    steps:
      - name: Fine
        run: echo hi
  second:
    steps:
      - name: Dead
'

# --- FLOW MAPPINGS (found by running this gate's own attack list) -----------
# `- {name: x, run: a, run: b}` was INVISIBLE to the block-mapping scanner:
# STEP_START needs `word:` right after the dash, so the line matched nothing,
# the step was never counted, and the gate returned exit 0 on a step that runs
# the wrong command. That is the exact defect this gate exists for, in a syntax
# it could not see.
check "red: flow mapping with a duplicate run" 1 "does not do what its name says" 'name: CI
jobs:
  build:
    steps:
      - {name: x, run: correct, run: WRONG}
'

check "red: flow mapping with no action at all" 1 "it does nothing" 'name: CI
jobs:
  build:
    steps:
      - {name: x}
'

# The two false positives that would make the fix worse than the hole.
check "green: a legitimate single-action flow mapping" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - {name: x, run: a}
'

check "green: a nested with{} is not a duplicate step key" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - {name: x, with: {a: 1, a: 2}, run: a}
'

# A flow mapping the scanner cannot bound on one line must be REFUSED, not
# skipped: a step it cannot analyse must never be reported well-formed.
check "red: multi-line flow mapping is refused, not skipped" 1 "cannot analyse" 'name: CI
jobs:
  build:
    steps:
      - {name: x,
         run: a}
'

# --- zero recognised steps is exit 2, not a pass ----------------------------
# The previous version printed "OK — 0 steps across 1 workflow file(s)" and
# exited 0, so a workflow written entirely in a form the scanner cannot see
# reported full coverage of a file it had not read. The fail-closed case below
# covers zero FILES; it never covered zero STEPS. Found by adversarial review.
check "red: a workflow with no steps at all is exit 2" 2 "recognised ZERO steps" 'name: CI
jobs:
  build:
    runs-on: ubuntu-latest
'

# --- child indent is DERIVED, not assumed to be dash+2 ----------------------
# `-   name: A` with the keys at that deeper column is valid, common YAML. The
# hardcoded dash+2 rejected it as "has neither run: nor uses:" -- a false
# positive whose message was also wrong, since the step does have an action.
check "green: dash + 3 spaces is not a false positive" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      -   name: A
          run: one
'

# And the real defects must still be caught at that indentation.
check "red: duplicate run at a 3-space indent" 1 "does not do what its name says" 'name: CI
jobs:
  build:
    steps:
      -   name: A
          run: one
          run: WRONG
'

check "red: no action at a 3-space indent" 1 "it does nothing" 'name: CI
jobs:
  build:
    steps:
      -   name: A
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
