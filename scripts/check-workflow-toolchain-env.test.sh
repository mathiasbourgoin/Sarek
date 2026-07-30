#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Red-path test for scripts/check-workflow-toolchain-env.sh.
#
# WHY A .test.sh AND NOT A prove-red-spec BLOCK. Same reason as
# check-workflow-steps.test.sh: the subject locates the repo with
# `git rev-parse --show-toplevel`, and prove-red.sh's scratch has no `.git`, so
# it exits 2 there and no green baseline is possible. Each case here gets a
# throwaway repo and a whole synthetic workflow.
#
# The first red case is the ACTUAL defect that motivated the gate -- backlog-186's
# HIP compile-gate restored as a host `run: dune build @sarek-hip/all` in a job
# whose every other OCaml action goes through `docker run ... spoc-ci:latest`.
# It failed in CI at `dune: command not found`, exit 127.

set -uo pipefail

SUBJECT="$(cd "$(dirname "$0")" && pwd)/check-workflow-toolchain-env.sh"
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
# Green first. Without it a subject that always exits 1 would satisfy every red
# case below and this file would prove nothing.
check "green: toolchain work goes through docker run" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - name: A host script whose NAME contains dune
        run: ./scripts/check-dune-dir-visibility.sh
      - name: Build in the image
        run: |
          docker run --rm \
            -v ${{ github.workspace }}:/work \
            -w /work \
            spoc-ci:latest \
            bash -lc '"'"'eval $(opam env) && \
              dune build @sarek-hip/all'"'"'
'

# --- the ACTUAL defect ------------------------------------------------------
check "red: host dune in a job with no host toolchain" 1 "runs \`dune\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: Build in the image
        run: |
          docker run --rm spoc-ci:latest bash -lc '"'"'dune build'"'"'
      - name: Compile-gate the HIP backend and its tests
        run: dune build @sarek-hip/all
'

# The message must name the JOB, since a repo-wide "somewhere a dune ran" tells
# a reader nothing about which job lacks the switch.
check "red: the finding names the job" 1 "job 'build'" 'name: CI
jobs:
  build:
    steps:
      - name: In image
        run: |
          docker run --rm spoc-ci:latest bash -lc '"'"'dune build'"'"'
      - name: Host dune
        run: dune build @sarek-hip/all
'

# --- a host toolchain makes it legitimate -----------------------------------
# docs.yml and deploy-pr-preview.yml really do run `opam exec -- dune build` on
# the host, and they are correct: they provision a switch first. A gate that
# flagged those would be reverted within a day, and its real findings with it.
check "green: setup-ocaml job may run dune on the host" 0 "OK" 'name: CI
jobs:
  docs:
    steps:
      - uses: ocaml/setup-ocaml@v3
        with:
          ocaml-compiler: 5.4.0
      - name: Docs
        run: opam exec -- dune build @doc
'

# The exemption must be REPORTED, not silent. A workflow every job of which
# provisions its own switch leaves this gate nothing to examine, and that is a
# correct pass -- but "examined 0 host steps" has to appear in the output, or a
# tree that drifted entirely into exempt jobs would read like full coverage.
check "green: an all-exempt tree names the coverage it did not provide" 0 "examined 0 host \`run:\` step(s) of 1" 'name: CI
jobs:
  docs:
    steps:
      - uses: ocaml/setup-ocaml@v3
      - name: Docs
        run: opam exec -- dune build @doc
'

check "green: the exempt job count is named" 0 "1 job(s) skipped as ocaml/setup-ocaml" 'name: CI
jobs:
  docs:
    steps:
      - uses: ocaml/setup-ocaml@v3
      - name: Docs
        run: opam exec -- dune build @doc
'

# --- the SILENT variant, which is why exit 127 being loud is not enough ------
# `make` ships on ubuntu-latest. A host-side `make` in a container-only job runs
# the real Makefile against a tree with no switch, and a recipe that
# short-circuits exits 0 -- the same defect, with a green check on it.
check "red: host make is caught even though make exists on the runner" 1 "runs \`make\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: In image
        run: |
          docker run --rm spoc-ci:latest bash -lc '"'"'dune build'"'"'
      - name: Fast e2e
        run: make e2e-fast
'

check "red: host opam is caught" 1 "runs \`opam\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: In image
        run: |
          docker run --rm spoc-ci:latest bash -lc '"'"'dune build'"'"'
      - name: Install
        run: opam install -y ctypes
'

# --- command POSITION, not substring ----------------------------------------
# `./scripts/check-dune-dir-visibility.sh` and
# `node scripts/check-alcotest-registration.js` both contain toolchain words.
# A substring test false-positives on them, and 5 spurious findings is how the
# sibling gate's first run went.
check "green: toolchain words inside script paths are not invocations" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - name: Dune visibility
        run: ./scripts/check-dune-dir-visibility.sh
      - name: Alias coverage
        run: ./scripts/check-test-alias-coverage.sh
      - name: Makefile lint
        run: ./scripts/lint-makefile-targets.sh
      - name: Harness
        run: |
          set -e
          node scripts/ocaml-thing.js
          ./scripts/opam-audit.sh
'

# A separator-led command position must still be seen: `a && dune build` hides
# the invocation from a start-of-line-only matcher.
check "red: dune after && is still a command position" 1 "runs \`dune\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: In image
        run: |
          docker run --rm spoc-ci:latest bash -lc '"'"'dune build'"'"'
      - name: Sneaky
        run: |
          echo building && dune build @sarek-hip/all
'

# An env-var prefix is still a command position: `SAREK_X=1 dune build`.
check "red: dune behind an env-var prefix is still caught" 1 "runs \`dune\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: In image
        run: |
          docker run --rm spoc-ci:latest bash -lc '"'"'dune build'"'"'
      - name: Env-prefixed
        run: SAREK_F16_DUMP=/tmp/x dune exec sarek-hip/test/t.exe
'

# --- comments are not invocations -------------------------------------------
# ci.yml's real harness step carries the comment "a `make` target nobody invokes
# is not that". That is prose about make, not a call to it.
check "green: a shell comment mentioning make is not an invocation" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - name: Harness
        run: |
          set -e
          # a `make` target nobody invokes is not that, and dune build likewise
          node scripts/thing.js
      - name: Second host step
        run: ./scripts/noop.sh
'

# --- container-extent tracking ----------------------------------------------
# The quoted `bash -lc '...'` script spans several continuation lines. Every
# toolchain command in it is IN the container; treating the chain as host-side
# would flag every correct step in this repo.
check "green: multi-line docker continuation chain is container-side" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - name: Unit tests
        run: |
          docker run --rm \
            -v ${{ github.workspace }}:/work \
            -w /work \
            spoc-ci:latest \
            bash -lc '"'"'git config --global --add safe.directory /work && \
              export OPAMROOT=/work/.opam-ci && \
              eval $(opam env --switch=5.4.0) && \
              dune runtest'"'"'
      - name: Second host step
        run: ./scripts/noop.sh
'

# A host command AFTER a docker chain closes must still be seen -- otherwise
# "handled the container extent" would be indistinguishable from "stopped
# scanning at the first docker run".
check "red: host dune AFTER a docker chain is still found" 1 "runs \`dune\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: Both
        run: |
          docker run --rm \
            -v /w:/work \
            spoc-ci:latest \
            bash -lc '"'"'dune build'"'"'
          dune build @sarek-hip/all
'

# --- anti-vacuity: fail closed ---------------------------------------------
# The sibling gate printed "OK — 0 steps" on a file it could not parse. Zero
# host steps in view means this gate verified nothing, and that is exit 2.
check "red: a workflow with no host run step is exit 2, not a pass" 2 "guarding nothing" 'name: CI
jobs:
  build:
    steps:
      - uses: actions/checkout@v4
'

check "red: a file with no parseable job is exit 2, not a pass" 2 "declares no jobs" 'name: CI
on: push
'

echo
echo "check-workflow-toolchain-env.test: $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
