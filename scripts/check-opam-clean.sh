#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Guard against a regression of the `make opam` target mutating the generated
# .opam files (it used to append a bogus `available: [ os = "linux" ]` line on
# every run, and also tried to append to a non-existent sarek_ppx.opam). This
# script asserts:
#   - `make opam` can be run twice in a row without error
#   - sarek.opam gains no `available:` line
#   - sarek.opam gains no duplicated trailing line
#   - the second run produces byte-identical output to the first, for EVERY
#     package dune-project declares (that is what "idempotent" means, and
#     until backlog-213 only sarek.opam was compared while the verdict spoke
#     for all of them)
#   - every one of those .opam files is left byte-identical to the tracked
#     version, and each must exist and be tracked or the run is INCONCLUSIVE
#   - the Makefile no longer references sarek_ppx.opam (that package does
#     not exist in this repo)
#
# EVERY VERDICT THIS SCRIPT PRINTS NAMES THE DUNE IT RAN UNDER. Not every line
# -- `make`'s own output and these messages' continuation lines do not -- but
# every sentence that states a result. The reason is backlog-213: this gate was
# reported red on a clean tree under the project switch's dune 3.24.1 and green
# under an ambient dune 3.23.0, and a bare exit code with no named toolchain is
# a claim wider than its measurement. It reads as "`make opam` is idempotent"
# when what was observed is "idempotent under whichever dune was first on PATH".
#
# What the reproduction attempts actually found, recorded here because the
# convenient version of this story is not the true one: the reported green
# under 3.23.0 could NOT be reproduced on that machine, and should not be
# quoted as an observation of this script. No dune 3.23.0 there can complete
# `make opam` at all -- in the project switch because ctypes 0.24.0 installs a
# `(lang dune 3.24)` dune-package 3.23 refuses to read, and in the octez-setup
# switch because that switch's OCaml 5.3.0 and ppxlib cannot build the PPX.
# Before backlog-213 both of those left through `set -e` carrying no message of
# this script's own and an exit code indistinguishable from "the tree is dirty",
# which is the sharper reason the divergence took three attempts to pin, and is
# what the INCONCLUSIVE path below now exists for. The divergence itself was
# established at the generation layer, where no dependency resolution is
# involved: `dune build sarek.opam` emits one bound under 3.23.0 and two under
# 3.24.1 -- 3.23.0 being the single release that deduplicates the pair for a
# project at `(lang dune 3.15)`, since 3.23.1 gated that behaviour away again.
#
# That divergence is now refused statically, with no build and no second
# toolchain, by scripts/check-dune-opam-portability.sh. This script's job is
# unchanged; naming the toolchain is so the NEXT divergence -- of whatever
# shape -- is one measurement to pin instead of three.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# The dune this run is actually about. Every branch below that cannot measure
# exits 2 (INCONCLUSIVE), never 1 — 1 is this gate's code for "the opam files
# are wrong", and spending it on a toolchain that could not be interrogated is
# how a diagnosis goes wrong for a third time.
inconclusive() {
    # Names the dune once TOOLCHAIN is set. The two callers that run before it
    # is set are exactly the two that could not determine a toolchain, so the
    # header's "every verdict names the dune it ran under" holds wherever there
    # is one to name.
    echo -e "${RED}INCONCLUSIVE${TOOLCHAIN:+ [under $TOOLCHAIN]}: $1${NC}"
    exit 2
}
TOOLCHAIN=""

# `command -v` resolves the same binary `make opam` will invoke — the Makefile
# sets no PATH or SHELL override — so the name printed is the one measured.
DUNE_BIN="$(command -v dune || true)"
[ -n "$DUNE_BIN" ] ||
    inconclusive "no 'dune' on PATH — this gate has no toolchain to measure."

# Guarded, because `set -e` would otherwise kill the script here with dune's own
# exit status (most likely 1, i.e. this gate's "the tree is dirty") and print
# nothing of its own. A dangling symlink into a deleted switch does exactly that.
if ! DUNE_VERSION="$("$DUNE_BIN" --version 2>&1)"; then
    inconclusive "'$DUNE_BIN --version' failed — there is a dune on PATH but it is not runnable, so nothing was measured."
fi
TOOLCHAIN="dune $DUNE_VERSION ($DUNE_BIN)"

fail() {
    echo -e "${RED}FAIL [under $TOOLCHAIN]: $1${NC}"
    exit 1
}

# `make opam` failing is NOT a finding about idempotence, and until backlog-213
# it left through `set -e` with no message of this gate's own — an exit code
# indistinguishable from "the tree is dirty". Both dune 3.23.0 attempts against
# this tree on 2026-07-31 left exactly here (see the header for the two build
# reasons), which is why that reproduction told nobody anything. Exit 2, and say
# which invocation.
run_make_opam() {
    if ! make opam; then
        echo -e "${RED}INCONCLUSIVE [under $TOOLCHAIN]: '$1' invocation of 'make opam' did not${NC}"
        echo -e "${RED}  succeed, so idempotence was never measured. This is a build failure,${NC}"
        echo -e "${RED}  not a finding about the opam files. Fix the build, or run this gate${NC}"
        echo -e "${RED}  under a toolchain that can build the tree, and try again.${NC}"
        exit 2
    fi
}

# The set the verdict speaks for. Until backlog-213 the script diffed sarek.opam
# alone while printing "'make opam' is clean and idempotent" — six of the seven
# generated files could drift and this gate would print a green about all of
# them.
#
# The set is PINNED to the packages dune-project declares, not discovered by a
# glob and not taken from `git ls-files`. Both of those shrink silently: delete
# or `git rm --cached` a generated .opam and the file simply stops being looked
# at, the denominator in the PASS line gets smaller, and the gate stays green
# about a set it let shrink. Deriving the set from dune-project makes a missing
# or untracked generated file a REFUSAL instead. The package list comes from
# check-dune-opam-portability.sh --list-packages so there is one parser and one
# answer rather than a second copy of the list here.
PACKAGES_RAW="$("$SCRIPT_DIR/check-dune-opam-portability.sh" --list-packages)" ||
    inconclusive "could not read the declared package list out of dune-project (check-dune-opam-portability.sh --list-packages failed), so the set of .opam files this gate speaks for is unknown."
mapfile -t PACKAGES <<< "$PACKAGES_RAW"

OPAM_FILES=()
untracked=()
absent=()
for pkg in "${PACKAGES[@]}"; do
    [ -n "$pkg" ] || continue
    f="$pkg.opam"
    if [ ! -f "$f" ]; then
        absent+=("$f")
        continue
    fi
    # Tracked, because the whole point of the final comparison is a diff against
    # the committed copy. An untracked generated file has nothing to be compared
    # with, and `git diff` would pass it in silence.
    if ! git ls-files --error-unmatch -- "$f" > /dev/null 2>&1; then
        untracked+=("$f")
        continue
    fi
    OPAM_FILES+=("$f")
done

[ "${#absent[@]}" -eq 0 ] ||
    inconclusive "dune-project declares package(s) whose .opam file is missing: ${absent[*]}. 'make opam' generates one per package; a set this gate cannot see in full is not one it can pass."
[ "${#untracked[@]}" -eq 0 ] ||
    inconclusive "generated .opam file(s) are not tracked by git: ${untracked[*]}. The final check diffs against the committed copy, and an untracked file has none — it would pass by being invisible."
[ "${#OPAM_FILES[@]}" -gt 0 ] ||
    inconclusive "dune-project declares no packages — there is nothing for 'make opam' to be idempotent about, so this gate has no subject."

echo "Running 'make opam' twice under $TOOLCHAIN to check for corruption/non-idempotence..."
run_make_opam first

# Snapshot between the runs. `git diff` alone compares only against the tracked
# copy, so a target that oscillated — run one differing, run two landing back on
# the tracked bytes — would read as convergent. That is the literal meaning of
# "twice in a row", and nothing was checking it.
# Guarded, not left to `set -e`: a failing mktemp or cp is this gate unable to
# measure (exit 2), never "the opam files are wrong" (exit 1), and `set -e`
# would hand back the tool's own status with no message.
SNAPSHOT_DIR="$(mktemp -d)" ||
    inconclusive "could not create a scratch directory for the between-runs snapshot."
trap 'rm -rf "$SNAPSHOT_DIR"' EXIT
for f in "${OPAM_FILES[@]}"; do
    mkdir -p "$SNAPSHOT_DIR/$(dirname "$f")" ||
        inconclusive "could not create the snapshot directory for $f."
    cp -- "$f" "$SNAPSHOT_DIR/$f" ||
        inconclusive "could not snapshot $f between the two 'make opam' runs."
done

run_make_opam second

for f in "${OPAM_FILES[@]}"; do
    # cmp: 0 same, 1 differs, >1 could not compare. The third is not a finding.
    #
    # `rc=0; cmd || rc=$?` and NOT a bare `cmd` followed by `case $?`. `set -e`
    # is in force (line 49), so a bare `cmp` returning nonzero terminates the
    # shell AT the cmp: the `case` never runs, every message below it is dead
    # code, and the gate exits 1 silently. This script was written to remove
    # exactly that failure mode from the `make opam` path and reintroduced it
    # here; CodeRabbit caught it on PR #399. `||` makes the command part of a
    # compound, which `set -e` does not kill.
    rc=0
    cmp -s -- "$SNAPSHOT_DIR/$f" "$f" || rc=$?
    case $rc in
        0) ;;
        1) fail "$f differs between the first and second 'make opam' — the target does not converge" ;;
        *) inconclusive "could not compare $f against its snapshot (cmp exited $rc)." ;;
    esac
done

if grep -q 'available:' sarek.opam; then
    fail "sarek.opam contains an 'available:' line after 'make opam'"
fi

# Duplicate trailing line check: last line must not equal the second-to-last
# non-empty line (guards against the old double-append corruption).
last_line="$(tail -n 1 sarek.opam)"
second_last_line="$(tail -n 2 sarek.opam | head -n 1)"
if [ -n "$last_line" ] && [ "$last_line" = "$second_last_line" ]; then
    fail "sarek.opam has a duplicated trailing line"
fi

# `git diff --exit-code`: 0 identical, 1 differs, anything else is git failing
# to answer. Treating 128 as "differs" would report a broken repository as an
# opam finding.
#
# Same `|| git_status=$?` shape, for the same reason as the cmp above: under
# `set -e` a bare `git diff --exit-code` that finds a difference terminates the
# shell on the spot, so the `case`, the file names it prints and the whole
# distinction between 1 and 128 would never execute.
git_status=0
git diff --exit-code -- "${OPAM_FILES[@]}" > /dev/null || git_status=$?
case $git_status in
    0) ;;
    1)
        changed="$(git diff --name-only -- "${OPAM_FILES[@]}" | tr '\n' ' ')"
        fail "generated .opam file(s) differ from the tracked version after 'make opam': $changed"
        ;;
    *) inconclusive "'git diff' exited $git_status — it could not compare the generated files against the tracked copies, so nothing was measured." ;;
esac

if grep -q 'sarek_ppx.opam' Makefile; then
    fail "Makefile still references the non-existent sarek_ppx.opam"
fi

if [ -f sarek_ppx.opam ]; then
    fail "sarek_ppx.opam was created by 'make opam' but should not exist"
fi

echo -e "${GREEN}PASS [under $TOOLCHAIN]: 'make opam' converged over two runs and left all ${#OPAM_FILES[@]} generated .opam file(s) matching the tracked copy.${NC}"
echo "      That is a statement about ONE dune. Nothing here compares dune versions;"
echo "      scripts/check-dune-opam-portability.sh is the cross-version rule, and it"
echo "      needs no second toolchain to check."
