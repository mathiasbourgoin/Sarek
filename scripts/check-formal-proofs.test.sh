#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Covering test for scripts/check-formal-proofs.sh (backlog-229).
#
# FORMAL_ALLOW_DIRTY=1 was documented as bypassing the pre-flight dirty-tree
# refusal. It did that, but the per-project `restore_tree` EXIT trap still ran
# unconditionally afterwards, and `git checkout -- .` inside it cannot tell a
# developer's own uncommitted edits from build output — so a tree the flag had
# just been told to tolerate got destroyed at exit anyway. It already happened
# once to a real agent's edits during this campaign.
#
# This test never runs the real Rocq toolchain (that would need
# rocq/rocq-prover and take ~50s, and is out of scope for a fast covering
# test). Instead it substitutes a stub `rocq` on PATH that satisfies
# `--version`, then fails as soon as `rocq makefile` is invoked -- late enough
# to have installed the `restore_tree` trap being tested, early enough to need
# nothing else from the toolchain. The stub also appends a line to a tracked
# file to stand in for the real script's own documented drift (extraction
# rewriting ~68 .ml/.mli files), so the "did the trap wrongly restore/fail to
# restore" question can be asked without a real build.
#
# Three cases, both polarities of the flag:
#
#   * flag=1,   pre-existing dirty tree  -> nothing is discarded (backlog-229)
#   * flag unset, clean tree             -> build-induced drift IS restored,
#                                            i.e. the default behaviour this
#                                            trap exists for is unchanged
#   * flag unset, pre-existing dirty tree -> pre-flight still refuses outright
#                                            and never reaches the trap at all
#
# Every fixture is a throwaway `git init` under mktemp, never a real formal/
# project and never a real worktree. Per the project's own recorded hazard
# (`git -C ""` silently operates on the caller's cwd instead of failing), every
# fixture directory is asserted non-empty before it is ever passed to a git
# command, and all git invocations below use `cd "$dir" && git ...` in a
# subshell rather than `git -C`, so an empty variable cannot silently retarget
# a command at this script's own working tree.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE_SRC="$SCRIPT_DIR/check-formal-proofs.sh"
[ -x "$GATE_SRC" ] || { echo "FAIL: $GATE_SRC not found or not executable"; exit 2; }

TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/check-formal-proofs-test.XXXXXX")"
trap 'rm -rf "$TMPROOT"' EXIT

pass=0
fail=0

# require_nonempty <name> <value> -- refuse to let an empty fixture path near
# a git command. The incident this guards against (`git -C ""` operating on
# the caller's own cwd instead of failing) is specifically a TMPROOT going
# empty: `mktemp -d` can fail with its output still captured by a bare `$(...)`
# under this script's `set -uo pipefail` (no `-e`), so TMPROOT="" is reachable
# in practice, not just in principle. Every other call site here checks a
# value built by concatenating TMPROOT with a fixed literal (a case name, or a
# path returned by make_fixture) and so cannot be empty once TMPROOT itself has
# been checked -- those calls are defense-in-depth against a future edit
# breaking that construction, not each guarding against their own reachable
# incident.
require_nonempty() {
  if [ -z "$2" ]; then
    echo "FATAL: $1 is empty; refusing to run git against an unknown directory." >&2
    exit 2
  fi
}

require_nonempty TMPROOT "$TMPROOT"

# stub rocq: enough to get check-formal-proofs.sh past its toolchain probe and
# into one project's subshell, where it appends a drift line to a BUILD-OWNED
# tracked file -- extraction/model.ml, matching the real script's own
# ':(glob)**/extraction/*.ml' restore pattern, standing in for extraction
# rewriting real .ml/.mli files -- and then fails, which is exactly where the
# trap under test fires, without compiling anything. driftfile.txt (below) is
# a SEPARATE, non-build-owned tracked file standing in for a developer's own
# source edit; the two must be distinguishable for this test to say anything
# about the discriminating restore, so they are never the same file.
FAKEBIN="$TMPROOT/fakebin"
mkdir -p "$FAKEBIN"
cat > "$FAKEBIN/rocq" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  --version)
    echo "stub rocq 0.0.0 (test double, not the real toolchain)"
    ;;
  makefile)
    # Stand-in for the real script's documented drift: a rebuild rewrites a
    # BUILD-OWNED tracked file (here, the extraction output). Then fail,
    # before anything is actually compiled -- all this test needs is for the
    # subshell's EXIT trap to fire.
    echo "stub-build-drift" >> extraction/model.ml
    exit 1
    ;;
  *)
    exit 1
    ;;
esac
STUB
chmod +x "$FAKEBIN/rocq"

# make_fixture <name> -- a minimal repo with one formal/ project, laid out
# exactly as scripts/check-formal-proofs.sh expects (script one directory
# below repo root, project directory holding a _CoqProject with a `-R theories
# <Name>` line). Returns the repo root on stdout.
make_fixture() {
  local dir="$TMPROOT/$1"
  require_nonempty "fixture dir ($1)" "$dir"
  mkdir -p "$dir/scripts" "$dir/formal/testproj/theories" "$dir/formal/testproj/extraction"
  cp "$GATE_SRC" "$dir/scripts/"
  chmod +x "$dir/scripts/check-formal-proofs.sh"
  printf -- '-R theories Test\n' > "$dir/formal/testproj/_CoqProject"
  printf 'committed\n' > "$dir/formal/testproj/driftfile.txt"
  printf 'committed\n' > "$dir/formal/testproj/extraction/model.ml"
  ( cd "$dir" && git init -q . && git config user.email t@t.t && git config user.name t \
    && git add -A && git commit -q -m init ) >/dev/null 2>&1
  echo "$dir"
}

# run_gate <dir> <extra-env...> -- invoke the real gate with the stub rocq
# ahead of any real one on PATH.
run_gate() {
  local dir="$1"; shift
  require_nonempty "run_gate dir" "$dir"
  ( cd "$dir" && env "$@" PATH="$FAKEBIN:$PATH" ./scripts/check-formal-proofs.sh ) >/tmp/check-formal-proofs-test-out.$$ 2>&1
  local rc=$?
  cat /tmp/check-formal-proofs-test-out.$$
  rm -f /tmp/check-formal-proofs-test-out.$$
  return $rc
}

echo "check-formal-proofs.sh covering test (backlog-229)"

# --- Case A: FORMAL_ALLOW_DIRTY=1, tree already had an uncommitted edit -----
# The exact regression: a developer's own uncommitted work (outside the
# build-owned set) must survive. Separately -- this is the #229 fix itself,
# not just the flag -- the build-owned artefact the stub rewrote must NOT be
# left dirty, or a following unflagged run is refused for drift this run
# caused (the absorbing state the discriminating restore exists to remove).
d="$(make_fixture case-a)"
require_nonempty "case-a dir" "$d"
printf 'committed\nunstaged-user-edit\n' > "$d/formal/testproj/driftfile.txt"
out="$(run_gate "$d" FORMAL_ALLOW_DIRTY=1)"
content="$(cat "$d/formal/testproj/driftfile.txt")"
owned_after="$(cat "$d/formal/testproj/extraction/model.ml")"
if printf '%s' "$content" | grep -qF 'unstaged-user-edit'; then
  echo "  PASS: case-a: FORMAL_ALLOW_DIRTY=1 preserves the developer's uncommitted edit" \
       "(outside the build-owned set)"
  pass=$((pass + 1))
else
  echo "  FAIL: case-a: uncommitted edit was discarded despite FORMAL_ALLOW_DIRTY=1"
  echo "    driftfile.txt now contains:"
  printf '%s\n' "$content" | sed 's/^/      /'
  echo "    gate output:"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi
if [ "$owned_after" = "committed" ]; then
  echo "  PASS: case-a: the build-owned artefact the stub rewrote was restored" \
       "even under FORMAL_ALLOW_DIRTY=1 -- the next unflagged run is not bricked"
  pass=$((pass + 1))
else
  echo "  FAIL: case-a: build-owned extraction/model.ml was left dirty" \
       "(got: $(printf '%s' "$owned_after" | tr '\n' '|')) -- this is exactly" \
       "the absorbing state #229's fix was supposed to remove"
  fail=$((fail + 1))
fi
if printf '%s' "$out" | grep -qF 'build-owned tracked file' \
   && printf '%s' "$out" | grep -qF 'only the build-owned paths above were touched'; then
  echo "  PASS: case-a: gate names the mechanism -- restored the build-owned set," \
       "left your other edits alone -- not merely the outcome"
  pass=$((pass + 1))
else
  echo "  FAIL: case-a: expected output to name both the build-owned restore and" \
       "the flag's narrower scope; got:"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

# --- Case B: FORMAL_ALLOW_DIRTY unset, tree starts CLEAN --------------------
# Pins the other polarity: the trap's default job -- discarding build-induced
# drift on a tree that started clean -- must be untouched by this fix.
d="$(make_fixture case-b)"
require_nonempty "case-b dir" "$d"
before="$(cat "$d/formal/testproj/extraction/model.ml")"
out="$(run_gate "$d")"
after="$(cat "$d/formal/testproj/extraction/model.ml")"
if [ "$after" = "$before" ]; then
  echo "  PASS: case-b: default behaviour (flag unset) still restores build-induced drift"
  pass=$((pass + 1))
else
  echo "  FAIL: case-b: drift was left in place with FORMAL_ALLOW_DIRTY unset --" \
       "the default (documented, load-bearing) restore regressed"
  echo "    expected: $(printf '%s' "$before" | tr '\n' '|')"
  echo "    got:      $(printf '%s' "$after" | tr '\n' '|')"
  fail=$((fail + 1))
fi
# M4a: content-equality alone cannot distinguish "the trap restored the
# drift" from "the trap never ran at all" -- both leave before == after, and a
# mutation that suppresses the stub's drift line satisfies this assertion
# vacuously (case-0, no drift was ever produced). Pin the trap's own restore
# NOTE, printed only when it actually found and restored build-owned drift
# (checked-formal-proofs.sh's restore_tree), so a mutation that removes the
# drift can no longer pass by removing the thing being restored along with it.
if printf '%s' "$out" | grep -qF 'restoring them'; then
  echo "  PASS: case-b: gate's output names the restore -- not merely a" \
       "before/after match that a no-op trap would also produce"
  pass=$((pass + 1))
else
  echo "  FAIL: case-b: expected the gate to report restoring build-owned" \
       "drift; got:"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

# --- Case C: FORMAL_ALLOW_DIRTY unset, tree already had an uncommitted edit -
# The pre-flight refusal itself must still fire, and fire BEFORE the trap-
# bearing subshell is ever entered -- confirming the flag's off-state is
# exactly as before this fix, not just "less destructive than it was".
d="$(make_fixture case-c)"
require_nonempty "case-c dir" "$d"
printf 'committed\nunstaged-user-edit\n' > "$d/formal/testproj/driftfile.txt"
out="$(run_gate "$d")"
rc=$?
content="$(cat "$d/formal/testproj/driftfile.txt")"
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qF '::error::formal/ has uncommitted changes' \
   && printf '%s' "$content" | grep -qF 'unstaged-user-edit'; then
  echo "  PASS: case-c: flag unset + dirty tree still refuses outright (exit $rc), edit untouched"
  pass=$((pass + 1))
else
  echo "  FAIL: case-c: expected an outright refusal naming the dirty tree, got exit $rc"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

echo ""
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
