#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Covering test for scripts/check-proof-ledger.py (task #95).
#
# WHY THIS EXISTS AS A TEST AND NOT AS A README LINE
#
# check-proof-ledger.py is a gate. The only property that matters about a gate is
# that it goes RED on the thing it claims to catch, and a gate is green on the
# committed tree by construction — which tells you nothing. Every scenario below
# is a mutation that MUST fail, asserted on the failure message, not just the
# exit code: a gate that fails for the wrong reason is a gate that will pass for
# the wrong reason later.
#
# It runs against a COPY of scripts/ and formal/ in a temporary directory, so the
# mutations never touch the working tree. check-proof-ledger.py locates the
# repository as dirname(dirname(__file__)), which is what makes the copy work.
#
# Deliberately does NOT invoke Rocq: the generated-ledger directory is synthesised
# from the committed ledgers. This tests the ENFORCEMENT, in a second or two, on
# every push. That the generation itself is faithful is established by
# check-formal-proofs.sh, which regenerates from a from-scratch rebuild and diffs
# — the two halves are separate claims and are checked separately.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0

# Runs the checker on the sandbox and asserts BOTH the exit status and that the
# output contains an expected fragment.
#
# `set +e` around the invocation, and output captured to a file rather than piped:
# the exit code is the thing under test, and a pipe to grep would report the
# grep's status instead. (`| tail` returning 0 for a failed command is how a
# broken gate reads green.)
expect() {
  local name="$1" want_rc="$2" want_msg="$3"
  local out="$WORK/out.txt" rc=0
  set +e
  python3 "$WORK/repo/scripts/check-proof-ledger.py" --generated "$WORK/gen" \
    > "$out" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne "$want_rc" ]; then
    echo "FAIL [$name]: exit $rc, expected $want_rc"
    sed 's/^/       /' "$out"
    fail=$((fail + 1))
    return
  fi
  if [ -n "$want_msg" ] && ! grep -qF -- "$want_msg" "$out"; then
    echo "FAIL [$name]: exit $rc as expected, but the message is wrong."
    echo "       expected to contain: $want_msg"
    sed 's/^/       /' "$out"
    fail=$((fail + 1))
    return
  fi
  echo "ok   [$name]"
  pass=$((pass + 1))
}

# Rebuilds the sandbox from the committed tree: repo/ is the checker plus the
# real formal/ directory, gen/ is a generated-ledger directory that starts out
# agreeing with it exactly.
reset_sandbox() {
  rm -rf "$WORK/repo" "$WORK/gen"
  mkdir -p "$WORK/repo/scripts" "$WORK/gen"
  cp "$REPO/scripts/check-proof-ledger.py" "$WORK/repo/scripts/"
  cp -r "$REPO/formal" "$WORK/repo/formal"
  for led in "$REPO"/formal/*/proof-ledger.json; do
    proj=$(basename "$(dirname "$led")")
    cp "$led" "$WORK/gen/$proj.json"
  done
}

edit_json() {  # edit_json <file> <python statement over `d`>
  python3 - "$1" "$2" <<'PY'
import json, sys
path, stmt = sys.argv[1], sys.argv[2]
d = json.load(open(path))
exec(stmt)
with open(path, "w") as fh:
    json.dump(d, fh, indent=2); fh.write("\n")
PY
}

echo "== baseline: the committed tree must pass, or nothing below means anything"
reset_sandbox
expect "unmutated tree is green" 0 "OK: ledgers match this build"

echo
echo "== drift: a committed ledger that no longer matches the build"
reset_sandbox
edit_json "$WORK/repo/formal/codegen-ptx/proof-ledger.json" \
  'd["counts"]["theorems"] = 999'
expect "edited count is caught" 1 "counts.theorems: committed 999, actual"

reset_sandbox
# The realistic shape: a theorem was added to the sources and the ledger was not
# regenerated. Simulated in the generated direction, which is where a new proof
# shows up first.
edit_json "$WORK/gen/convergence-safety.json" \
  'm = d["modules"]["ConvergenceSpec.ConvergenceSpec"]; m["theorems"].append("brand_new_lemma"); m["counts"]["theorems"] += 1; d["counts"]["theorems"] += 1'
expect "a new proof with a stale ledger is caught" 1 \
  "counts.theorems: committed 111, actual 112"

reset_sandbox
# The nastiest case: the totals still agree, only a NAME changed. This is what
# a comparison on counts alone would wave through.
edit_json "$WORK/gen/convergence-safety.json" \
  'm = d["modules"]["ConvergenceSpec.ConvergenceSpec"]; m["theorems"][0] = "renamed_lemma"'
expect "a renamed theorem at an unchanged count is caught" 1 \
  "differs in the per-module theorem lists"

reset_sandbox
rm "$WORK/repo/formal/codegen-ptx/proof-ledger.json"
expect "a deleted ledger is caught" 1 "proof-ledger.json is missing"

echo
echo "== allowlist: the check #95 existed to unblock"
reset_sandbox
edit_json "$WORK/gen/convergence-safety.json" \
  'd["axioms_project_local"].append("ConvergenceSpec.ConvergenceSpec.assume_it_works")'
edit_json "$WORK/repo/formal/convergence-safety/proof-ledger.json" \
  'd["axioms_project_local"].append("ConvergenceSpec.ConvergenceSpec.assume_it_works")'
expect "an unsanctioned axiom is caught" 1 \
  "ConvergenceSpec.ConvergenceSpec.assume_it_works"

reset_sandbox
printf 'CodegenPtx.AGpuSemantics.no_longer_used\n' \
  >> "$WORK/repo/formal/axiom-allowlist.txt"
expect "an allowlist entry no proof reaches is caught" 1 \
  "no proof depends on any more"

reset_sandbox
rm "$WORK/repo/formal/axiom-allowlist.txt"
expect "a missing allowlist fails rather than skips" 1 \
  "axiom-allowlist.txt is missing"

echo
echo "== anchors: a note naming a theorem that does not exist"
reset_sandbox
edit_json "$WORK/repo/formal/convergence-safety/proof-notes.json" \
  'd["theorems"]["check_env_nonvarying_uniform"] = {"tier": "T3", "note": "the historical phantom"}'
expect "a phantom anchor is caught" 1 \
  "not theorems in this build"

reset_sandbox
edit_json "$WORK/repo/formal/convergence-safety/proof-notes.json" \
  'd["counts"] = {"theorems": 111}'
expect "a counts key creeping back into the notes is caught" 1 \
  "Counts belong to the generated proof-ledger.json"

echo
echo "== the gate must not pass vacuously"
reset_sandbox
rm -f "$WORK/gen"/*.json
expect "an empty generated directory fails" 1 \
  "the per-project generation step did not run"

reset_sandbox
rm -f "$WORK/gen/type-safety.json"
expect "a project whose ledger was never generated fails" 1 \
  "no ledger was generated for: type-safety"

echo
echo "-- $pass passed, $fail failed"
[ "$fail" -eq 0 ]
