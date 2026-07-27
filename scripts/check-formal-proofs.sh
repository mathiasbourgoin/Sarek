#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Machine-check the Rocq proofs under formal/ (task #45), and the repository's
# claims about them (task #95).
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
# and then (#95) it derives each project's proof-ledger.json from that build and
# fails if the committed one differs, if the kernel found a project-local axiom
# formal/axiom-allowlist.txt does not sanction, or if a proof-notes.json
# annotates a theorem that does not exist. Proving the proofs and proving that
# what the repository SAYS about them is still true are two different claims;
# before #95 only the first was checked.
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

# How many formal/ projects must be found and checked. Pinned so that a project
# silently disappearing (a moved _CoqProject, a bad merge) fails the gate
# instead of quietly shrinking its scope.
EXPECTED_PROJECTS=3

cd "$(dirname "$0")/.."
# Absolute, because the per-project steps run in a subshell that has cd'd into
# formal/<project> and still need to invoke scripts/ from the repository root.
ROOT=$(pwd)

# Writability pre-flight.
#
# This script rebuilds in place: it deletes committed .vo files and regenerates
# CoqMakefile next to the sources. In CI that happens inside a container whose
# user need not own the mounted checkout — rocq/rocq-prover runs as `rocq`
# (uid 1000) while a GitHub-hosted runner owns the workspace as `runner`
# (uid 1001), and a uid-1000 process cannot write a uid-1001 0755 directory.
# Without this probe the first symptom is an EACCES from `rm` several steps
# later, which reads like a broken proof rather than a broken mount.
#
# This cannot be caught by running the script locally: a developer's checkout is
# already owned by the user running it. That is precisely why it is asserted.
probe=".formal-write-probe.$$"
if ! : > "$probe" 2>/dev/null; then
  echo "::error::$(pwd) is not writable by uid $(id -u). This script rebuilds \
the proofs in place. In CI, mount a checkout owned by the container's user (see \
the formal-proofs job in .github/workflows/ci.yml)."
  exit 1
fi
rm -f "$probe"

# Refuse to run on a formal/ tree that already has uncommitted changes.
#
# Step 6 restores the tree with `git checkout -- .` because the build rewrites
# ~68 extracted .ml/.mli files. That restore cannot tell build output from work
# in progress, so on a dirty tree it would silently destroy a developer's
# uncommitted edits. Refusing here is what makes the restore safe: after this
# check, any drift observed later must have been produced by the build.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  dirty=$(git status --porcelain -- formal | grep -v '^?? ' || true)
  if [ -n "$dirty" ] && [ "${FORMAL_ALLOW_DIRTY:-0}" != "1" ]; then
    echo "::error::formal/ has uncommitted changes to tracked files. This \
script rebuilds in place and restores the tree afterwards, which would discard \
them. Commit or stash first, or set FORMAL_ALLOW_DIRTY=1 to accept the loss."
    printf '%s\n' "$dirty" | sed 's/^/    /'
    exit 1
  fi
fi

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

if ! command -v python3 >/dev/null 2>&1; then
  echo "::error::no python3 on PATH. scripts/gen-proof-ledger.py derives each \
project's proof-ledger.json from the toolchain, and without it the ledger \
comparison and the axiom allowlist (#95) cannot run at all — which is the \
state this gate was written to end. rocq/rocq-prover:9.1.1 ships python3."
  exit 1
fi

projects=$(find formal -mindepth 1 -maxdepth 1 -type d | sort)
if [ -z "$projects" ]; then
  echo "::error::no formal/ projects found — this gate would pass vacuously."
  exit 1
fi

# Generated ledgers land OUTSIDE the project directories, deliberately. The
# per-project `restore_tree` trap runs `git checkout -- .` inside the project,
# so a ledger written next to the sources would be reverted to the committed
# copy before it could be compared against it — the comparison would pass by
# comparing the committed file with itself. Writing elsewhere is what makes the
# drift check able to fail.
LEDGER_OUT=$(mktemp -d)
trap 'rm -rf "$LEDGER_OUT"' EXIT

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

    # Restore the working tree on EVERY exit path, installed BEFORE the first
    # destructive step.
    #
    # `make -f CoqMakefile` re-runs extraction, and the extracted .ml/.mli files
    # are committed in ocamlformat-formatted form while extraction emits them
    # raw. A full run therefore rewrites ~68 tracked files with semantically
    # identical but differently formatted output, and also leaves a regenerated
    # CoqMakefile behind. Harmless on a throwaway CI checkout, destructive for
    # anyone running this documented gate on their own tree.
    #
    # As a trap rather than a final step because the interesting exits are the
    # early ones: a failed compile or a failed kernel re-check used to leave the
    # committed .vo files deleted and the tree rewritten, and — now that this
    # script refuses to start on a dirty formal/ — that residue would block the
    # very next run with a message about uncommitted changes the developer never
    # made. Restoring unconditionally keeps a failed run repeatable.
    #
    # Safe precisely because of the dirty-tree check above: the tree was clean
    # when this started, so anything to restore was produced by this script.
    # The lia/nia caches are untracked build residue the tactics drop next to
    # the sources; they would otherwise be left behind.
    restore_tree() {
      rm -f .lia.cache .nia.cache
      if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        drift=$(git status --porcelain -- . 2>/dev/null | grep -v '^?? ' || true)
        if [ -n "$drift" ]; then
          n=$(printf '%s\n' "$drift" | wc -l)
          echo "  NOTE: the build rewrote $n tracked file(s) (extraction output" \
               "vs the ocamlformat-formatted committed copies); restoring them."
          git checkout -- . 2>/dev/null || \
            echo "  WARNING: could not restore; tree left dirty."
        fi
      fi
    }
    trap restore_tree EXIT

    # 1. Delete every committed build artefact. A .vo that is never rebuilt is
    #    an unverified binary blob, and re-checking it would only prove it is
    #    self-consistent, not that it matches the .v beside it.
    #
    #    The generated makefile and its dependency files go too, and that is not
    #    tidiness — it is the stale-generated-artefact hazard this script exists
    #    to catch, biting the script itself. The tracked .CoqMakefile.d files
    #    hard-code /usr/lib/ocaml/rocq-runtime/rocqworker, the path of a
    #    system-packaged Rocq. That path exists on a distro install and does not
    #    exist in the opam-based rocq/rocq-prover image, so reusing them fails
    #    the build with "No rule to make target" before a single proof is
    #    checked.
    find . \( -name '*.vo' -o -name '*.vok' -o -name '*.vos' \
              -o -name '*.glob' -o -name '*.CoqMakefile.d' \
              -o -name 'CoqMakefile' -o -name 'CoqMakefile.conf' \) \
         -exec rm -f {} +

    # 2. Regenerate the makefile from _CoqProject. Never reuse the committed
    #    CoqMakefile: see above.
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
    #    `set +e` around the capture is load-bearing, not defensive habit.
    #    `out=$(rocq check ...)` is a plain assignment in a command body, not an
    #    `if` condition, so `set -e` applies to it in full. `rocqchk` exits
    #    non-zero when it finds an error, so on a REAL kernel-check failure the
    #    subshell died on this very line — before the diagnostics below could
    #    run — and because `2>&1` had redirected the checker's entire output
    #    into `$out`, that output was discarded with it. The job still failed
    #    (no false pass), but reported none of the context this step exists to
    #    provide. Untested until now because the committed proofs all pass.
    set +e
    if [ -n "$ROCQ" ]; then
      out=$(rocq check -R theories "$logical" theories/*.vo 2>&1)
    else
      out=$(coqchk -R theories "$logical" theories/*.vo 2>&1)
    fi
    rc=$?
    set -e
    printf '%s\n' "$out" | tail -3
    # Two independent conditions, both fatal: a non-zero exit from the checker,
    # and the absence of its success marker. Neither implies the other — a
    # checker that dies on a missing .vo exits non-zero without ever printing a
    # verdict, and a truncated or partial run can exit 0 having checked nothing.
    #
    # Substring match on the variable, NOT `printf ... | grep -q`: grep -q exits
    # on its first match and closes the pipe, printf takes SIGPIPE (141), and
    # `set -o pipefail` propagates that as a failure. With `rocq check` output
    # running to thousands of lines this fires reliably, and the check would
    # report a failure on a run where every proof passed.
    case "$out" in
      *"Modules were successfully checked"*) marker=1 ;;
      *) marker=0 ;;
    esac
    if [ "$rc" -ne 0 ] || [ "$marker" -ne 1 ]; then
      echo "::error::kernel re-check FAILED for $proj (exit $rc, success marker \
present: $marker)"
      printf '%s\n' "$out"
      exit 1
    fi

    # 5. Derive this project's proof ledger from the build that just happened.
    #
    #    This REPLACES a grep over `$out` for lines matching /axiom/i, which
    #    reported ~2200 lines per project because `rocq check`'s trace names
    #    every Stdlib module it loads and several of them are called
    #    `...Uint63Axioms` / `...FloatAxioms`. It printed module names, never an
    #    axiom, and was labelled "assumptions reported by the kernel checker".
    #
    #    gen-proof-ledger.py parses the CONTEXT SUMMARY that `rocq check -o`
    #    prints — the actual transitive assumption closure — and cross-checks it
    #    against theories/*.glob. It fails loudly if the summary is missing or
    #    its format changed, so a parse that finds nothing cannot be mistaken
    #    for a project with no axioms.
    #
    #    Written to $LEDGER_OUT, not next to the sources: see the comment where
    #    LEDGER_OUT is created.
    python3 "$ROOT/scripts/gen-proof-ledger.py" . \
      -o "$LEDGER_OUT/$(basename "$PWD").json"

    # 6. The working tree is put back by the restore_tree EXIT trap installed
    #    at the top of this subshell, so that it also runs on the failure paths
    #    above. See the comment there.
  )
  checked=$((checked + 1))
done

echo
# A gate that passes when it found nothing to check is the exact failure mode
# this script was written to remove. Moving the three _CoqProject files aside
# used to yield "OK: 0 formal project(s)" and exit 0.
if [ "$checked" -eq 0 ]; then
  echo "::error::no formal/ project had a _CoqProject — nothing was verified. \
This gate must never pass vacuously."
  exit 1
fi
if [ "$checked" -ne "$EXPECTED_PROJECTS" ]; then
  echo "::error::expected $EXPECTED_PROJECTS formal projects, checked $checked. \
If a project was added or removed on purpose, update EXPECTED_PROJECTS at the \
top of this script."
  exit 1
fi


# ---------------------------------------------------------------------------
# Ledger drift, axiom allowlist, note anchors (task #95).
#
# Everything above proves the proofs. This proves that what the repository SAYS
# about the proofs is still true — the half that #45 deliberately left open,
# because the three hand-written proof-ledger.json files disagreed with each
# other and there was no trustworthy expected value to compare against.
#
# Three checks, each failing for a different reason:
#
#   drift     — the committed ledger differs from what this build produced.
#               Someone edited it, or changed the proofs without regenerating.
#   allowlist — the kernel found a project-local axiom nobody sanctioned, or an
#               entry in formal/axiom-allowlist.txt no proof depends on any more.
#               This is the check the "reported for review rather than enforced"
#               note used to stand in for.
#   anchors   — proof-notes.json names a theorem that does not exist. The old
#               ledgers carried one such entry (an anchor pointing at a PAIR of
#               lemmas under a name Rocq never saw); nothing could detect it.
#
# Runs after the loop, so the per-project restore_tree traps have already put
# the working tree back and the committed ledgers are the committed ledgers.
# ---------------------------------------------------------------------------
echo
echo "== ledger, axioms, anchors"
python3 "$ROOT/scripts/check-proof-ledger.py" --generated "$LEDGER_OUT"

echo
echo "OK: $checked formal project(s) rebuilt from source and kernel-re-checked."
