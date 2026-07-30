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

# --- apt provisioning -------------------------------------------------------
# ci.yml's build job installs `ocaml-nox` with apt so
# scripts/check-no-machine-identifiers.test.sh can drive benchmarks/machine_label.ml
# through the toplevel. That job DOES provision the toolchain, just not with
# ocaml/setup-ocaml. Every case below exists to keep that recognition from
# degenerating into a blanket exemption.

check "green: apt-get + version assertion provisions ocaml for later steps" 0 "OK" 'name: CI
jobs:
  build:
    steps:
      - name: OCaml toplevel for the label-shape harness
        run: |
          sudo apt-get update -qq
          sudo apt-get install -y --no-install-recommends ocaml-nox
          ocaml -version
      - name: Label-shape harness
        run: |
          set -e
          ocaml benchmarks/machine_label.ml
'

# The green line must NAME the apt provisioning. A recognition this gate performs
# silently is one no reader can audit, which is how the setup-ocaml exemption is
# already handled.
check "green: the apt-provisioned step count is named" 0 "1 step(s) provision a host tool via apt-get" 'name: CI
jobs:
  build:
    steps:
      - name: Toplevel
        run: |
          sudo apt-get install -y ocaml-nox
          ocaml -version
      - name: Harness
        run: ocaml thing.ml
'

# --- the original defect must STILL be red ----------------------------------
# THE load-bearing case. Installing ocaml-nox does not put `dune` on PATH, so
# backlog-186's actual failure -- a host `dune build @sarek-hip/all` in this very
# job -- must still be found AFTER the apt step. If this goes green the gate has
# been loosened into uselessness.
check "red: host dune is still caught in a job that apt-installed only ocaml" 1 "runs \`dune\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: Toplevel
        run: |
          sudo apt-get install -y --no-install-recommends ocaml-nox
          ocaml -version
      - name: Compile-gate the HIP backend and its tests
        run: dune build @sarek-hip/all
'

# `make` ships on the runner, so its variant is the silent one. ocaml-nox does
# not provide it either.
check "red: host make is still caught after an apt ocaml install" 1 "runs \`make\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: Toplevel
        run: |
          sudo apt-get install -y ocaml-nox
          ocaml -version
      - name: Fast e2e
        run: make e2e-fast
'

# The pre-#389 shape: a host `ocaml` with no apt-get anywhere in the job. This is
# what the gate said about ci.yml before the toplevel step existed, and it must
# keep saying it.
check "red: host ocaml with no provisioning at all is still caught" 1 "runs \`ocaml\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: In image
        run: |
          docker run --rm spoc-ci:latest bash -lc '"'"'dune build'"'"'
      - name: Label-shape harness
        run: ocaml benchmarks/machine_label.ml
'

# --- the assertion is load-bearing, not decoration --------------------------
# `apt-get install` exiting 0 is not the same claim as the binary resolving on
# PATH. Without the version assertion the job has made a claim, not a
# demonstration, and this gate does not accept claims.
check "red: apt-get install with no version assertion provisions nothing" 1 "no earlier \`apt-get install\`" 'name: CI
jobs:
  build:
    steps:
      - name: Toplevel
        run: sudo apt-get install -y --no-install-recommends ocaml-nox
      - name: Harness
        run: ocaml benchmarks/machine_label.ml
'

# The package table is an allow-list, so an unknown package provisions nothing --
# otherwise any apt-get in the job would launder every later toolchain call.
check "red: an unrelated apt package does not provision the toolchain" 1 "runs \`ocaml\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: Tools
        run: |
          sudo apt-get install -y jq
          jq --version
      - name: Harness
        run: ocaml benchmarks/machine_label.ml
'

# --- provisioning is ORDERED -------------------------------------------------
# "the job installs it somewhere" must not excuse a step that runs BEFORE the
# install. That is the exit-127 shape with an alibi.
check "red: a use before the install step is still caught" 1 "runs \`ocaml\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: Harness, too early
        run: ocaml benchmarks/machine_label.ml
      - name: Toplevel
        run: |
          sudo apt-get install -y ocaml-nox
          ocaml -version
'

# Same rule inside one step: the install must precede the use textually.
check "red: a use before the install within one step is still caught" 1 "comes LATER in this same step" 'name: CI
jobs:
  build:
    steps:
      - name: Both, wrong order
        run: |
          ocaml benchmarks/machine_label.ml
          sudo apt-get install -y ocaml-nox
          ocaml -version
'

# --- provisioning does not cross a job boundary -----------------------------
# Each job gets its own runner, so job B inherits nothing from job A.
check "red: apt provisioning in one job does not carry into another" 1 "job 'other' runs" 'name: CI
jobs:
  build:
    steps:
      - name: Toplevel
        run: |
          sudo apt-get install -y ocaml-nox
          ocaml -version
      - name: Harness
        run: ocaml thing.ml
  other:
    steps:
      - name: Harness with no switch
        run: ocaml thing.ml
'

# --- an apt-get inside the container provisions the CONTAINER ---------------
# The host PATH is untouched by an install that happens in `docker run`, so this
# must not read as host provisioning.
check "red: apt-get inside docker run does not provision the host" 1 "runs \`ocaml\` on the HOST" 'name: CI
jobs:
  build:
    steps:
      - name: In image
        run: |
          docker run --rm spoc-ci:latest \
            bash -lc '"'"'apt-get install -y ocaml-nox && ocaml -version'"'"'
      - name: Harness
        run: ocaml benchmarks/machine_label.ml
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
