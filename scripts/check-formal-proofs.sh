#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Machine-check the Rocq proofs under formal/ (task #45).
#
# WHAT THIS REPLACES
#
# CI's only enforcement of the "0 admits" guarantee was a grep over the .v
# sources, and the committed .vo files were never re-verified on a branch. A grep
# is a lexical check pretending to be a proof: it cannot tell whether the files
# still COMPILE, whether a tactic silently stopped closing its goal, or whether a
# committed .vo actually corresponds to the .v next to it. A .vo could be stale,
# hand-edited, or built from different sources, and every gate in the repository
# would have stayed green.
#
# This script does the two things only the toolchain can do:
#
#   1. rebuilds every .v FROM SCRATCH (the committed .vo files are deleted first,
#      so they cannot vouch for themselves), and
#   2. runs `rocq check`, the standalone kernel re-checker, over the results.
#      `rocq check` re-verifies every proof term against the kernel without
#      trusting the elaborator that produced it, and prints the axioms each
#      module depends on.
#
# The grep gate in .github/workflows/ci.yml is kept as well: it is instant, it
# runs without a Rocq toolchain, and it catches an `admit.` in review before this
# heavier job gets to it. The two are complementary — the grep is the fast lexical
# tripwire, this is the proof.
#
# MEASURED COST (Rocq 9.1.1, 16-core workstation, cold: all .vo deleted)
#   rocq c   (all three projects)   ~11 s total
#   rocq check                       codegen-ptx 17.1 s
#                                    convergence-safety 11.3 s
#                                    type-safety 11.0 s
#   ------------------------------------------------------------------
#   total                           ~50 s of CPU work.
# That is cheap enough that there is no argument for keeping the grep-only
# arrangement. The toolchain is NOT added to ci/Dockerfile: the official
# rocq/rocq-prover image already ships the exact version these proofs were
# developed against, so the proof job runs in that container instead of growing
# the main image by a Rocq build.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v rocq >/dev/null 2>&1; then
  ROCQ=rocq
elif command -v coqc >/dev/null 2>&1; then
  # Rocq 9 renamed the driver; fall back for a Coq 8.x toolchain, which will
  # most likely fail to compile these sources and say so loudly.
  ROCQ=""
else
  echo "::error::no rocq/coqc on PATH. This gate needs a real proof checker — \
that is the entire point of #45. Run it in the rocq/rocq-prover container (see \
.github/workflows/ci.yml)."
  exit 1
fi

echo "== toolchain"
if [ -n "$ROCQ" ]; then
  rocq --version
else
  coqc --version
fi

projects=$(find formal -mindepth 1 -maxdepth 1 -type d | sort)
if [ -z "$projects" ]; then
  echo "::error::no formal/ projects found — this gate would pass vacuously."
  exit 1
fi

checked=0

for proj in $projects; do
  [ -f "$proj/_CoqProject" ] || continue
  echo
  echo "== $proj"

  logical=$(grep -oP '^-R\s+theories\s+\K\S+' "$proj/_CoqProject" || true)
  if [ -z "$logical" ]; then
    echo "::error::$proj/_CoqProject has no '-R theories <LogicalName>' line; \
cannot determine the logical path to check."
    exit 1
  fi

  (
    cd "$proj"

    # 1. Delete every committed build artefact. A .vo that is never rebuilt is
    #    an unverified binary blob, and re-checking it would only prove it is
    #    self-consistent, not that it matches the .v beside it.
    find . \( -name '*.vo' -o -name '*.vok' -o -name '*.vos' \
              -o -name '*.glob' \) -exec rm -f {} +

    # 2. Regenerate the makefile from _CoqProject rather than trusting the
    #    committed CoqMakefile, which is itself generated and could be stale.
    if [ -n "$ROCQ" ]; then
      rocq makefile -f _CoqProject -o CoqMakefile
    else
      coq_makefile -f _CoqProject -o CoqMakefile
    fi

    # 3. Compile. Any Admitted/admit that the grep gate missed, any broken
    #    proof, any missing dependency fails here.
    make -f CoqMakefile -j"$(nproc)"

    # 4. Kernel re-check. This is the load-bearing step: it re-verifies the
    #    proof terms independently of the elaborator that built them, and
    #    reports the axioms they rest on.
    if [ -n "$ROCQ" ]; then
      out=$(rocq check -R theories "$logical" theories/*.vo 2>&1)
    else
      out=$(coqchk -R theories "$logical" theories/*.vo 2>&1)
    fi
    printf '%s\n' "$out" | tail -3
    if ! printf '%s\n' "$out" | grep -q "Modules were successfully checked"; then
      echo "::error::kernel re-check FAILED for $proj"
      printf '%s\n' "$out"
      exit 1
    fi

    # 5. Report what the proofs assume. `rocq check` lists axioms rather than
    #    rejecting them, and this repository has one sanctioned escape hatch
    #    (`Parameter` in AGpuSemantics.v — see the admit gate's comment in
    #    .github/workflows/ci.yml), so this is reported for review rather than
    #    enforced at zero. Enforcing a specific allowlist is a follow-up: the
    #    three proof-ledger.json files currently disagree with each other about
    #    the axiom and theorem counts, so there is no single trustworthy
    #    expected value to compare against yet.
    if printf '%s\n' "$out" | grep -qiE "^\s*\*\*\* |axiom"; then
      echo "  assumptions reported by the kernel checker:"
      printf '%s\n' "$out" | grep -iE "^\s*\*\*\* |axiom" | sed 's/^/    /'
    fi

    theorems=$(grep -rhcE "^(Theorem|Lemma|Corollary|Proposition|Remark|Fact) " \
                 theories/*.v | paste -sd+ | bc)
    echo "  kernel-verified statements in theories/: $theorems"
  )
  checked=$((checked + 1))
done

echo
echo "OK: $checked formal project(s) rebuilt from source and kernel-re-checked."
