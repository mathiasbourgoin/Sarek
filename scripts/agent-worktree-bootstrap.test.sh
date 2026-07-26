#!/usr/bin/env bash
# Tests for scripts/agent-worktree-bootstrap.sh.
#
# A guard nobody has watched trigger is not a guard. Every check in the
# bootstrap gets an explicitly constructed bad input here, and the refusal is
# asserted on exit code AND on the message that names the failure.
set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/agent-worktree-bootstrap.sh"
[ -x "$SCRIPT" ] || { echo "not executable: $SCRIPT" >&2; exit 2; }

PASS=0
FAIL=0
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

ok()   { PASS=$((PASS+1)); echo "  ok — $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL — $1"; }

# expect <label> <expected-exit> <expected-substring> -- <cmd...>
expect() {
  local label="$1" want_rc="$2" want_msg="$3"; shift 4
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -ne "$want_rc" ]; then
    bad "$label: exit $rc, wanted $want_rc"; echo "$out" | sed 's/^/      /'; return
  fi
  if ! printf '%s' "$out" | grep -qF -- "$want_msg"; then
    bad "$label: message missing '$want_msg'"; echo "$out" | sed 's/^/      /'; return
  fi
  ok "$label (exit $rc, said '$want_msg')"
}

git_quiet() { git -c init.defaultBranch=main -c user.email=t@t -c user.name=t "$@"; }

echo "== 1. project is an UNTRACKED directory inside another repo (the #101 defect)"
OUTER="$TMP/outer"
mkdir -p "$OUTER"
git_quiet init -q "$OUTER"
echo outer > "$OUTER/README"
git_quiet -C "$OUTER" add README
git_quiet -C "$OUTER" commit -qm init
mkdir -p "$OUTER/proj/src"
echo 'let () = ()' > "$OUTER/proj/src/main.ml"   # present on disk, untracked in OUTER
expect "refuses wrong-repo dispatch" 3 "WRONG REPOSITORY" -- \
  bash "$SCRIPT" --agent tester --project "$OUTER/proj" --check-only
expect "names the enclosing repo in the refusal" 3 "enclosing repo:   $OUTER" -- \
  bash "$SCRIPT" --agent tester --project "$OUTER/proj" --check-only

echo "== 2. same layout, but the subdirectory IS tracked by the outer repo -> allowed"
git_quiet -C "$OUTER" add proj
git_quiet -C "$OUTER" commit -qm "track proj"
out="$(bash "$SCRIPT" --agent tester --project "$OUTER/proj" --base HEAD --check-only 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -qF "subdirectory of the repo"; then
  ok "tracked subdirectory warns but proceeds"
else
  bad "tracked subdirectory should proceed with a warning (exit $rc)"; echo "$out" | sed 's/^/      /'
fi

echo "== 3. project is not in a git repository at all"
mkdir -p "$TMP/loose"
expect "refuses a non-repository project" 3 "is not inside any git repository" -- \
  bash "$SCRIPT" --agent tester --project "$TMP/loose" --check-only

echo "== 4. base ref does not resolve"
SOLO="$TMP/solo"
git_quiet init -q "$SOLO"
echo a > "$SOLO/a"; git_quiet -C "$SOLO" add a; git_quiet -C "$SOLO" commit -qm init
expect "refuses an unresolvable base" 3 "base ref does not resolve" -- \
  bash "$SCRIPT" --agent tester --project "$SOLO" --base origin/nope --check-only

echo "== 5. base ref is STALE relative to the remote"
REMOTE="$TMP/remote.git"
git_quiet init -q --bare "$REMOTE"
CLONE="$TMP/clone"
git_quiet clone -q "$REMOTE" "$CLONE" 2>/dev/null
echo one > "$CLONE/f"; git_quiet -C "$CLONE" add f; git_quiet -C "$CLONE" commit -qm one
git_quiet -C "$CLONE" push -q origin main
# A second worker advances the remote; this clone's origin/main is now behind.
WORKER="$TMP/worker"
git_quiet clone -q "$REMOTE" "$WORKER" 2>/dev/null
echo two > "$WORKER/g"; git_quiet -C "$WORKER" add g; git_quiet -C "$WORKER" commit -qm two
git_quiet -C "$WORKER" push -q origin main
# Force the stale view: reset the remote-tracking ref without fetching.
git_quiet -C "$CLONE" update-ref refs/remotes/origin/main "$(git -C "$CLONE" rev-parse HEAD)"
expect "refuses a stale base" 3 "is STALE" -- \
  bash "$SCRIPT" --agent tester --project "$CLONE" --base origin/main --root "$TMP/wt" --check-only
# ...and accepts it when the operator says so, still stale.
out="$(bash "$SCRIPT" --agent tester --project "$CLONE" --base origin/main --root "$TMP/wt" \
        --allow-stale-base --check-only 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -qF "allow-stale-base"; then
  ok "--allow-stale-base proceeds and records why"
else
  bad "--allow-stale-base should proceed (exit $rc)"; echo "$out" | sed 's/^/      /'
fi
# Positive control: after a real fetch the same invocation must pass, so the
# stale refusal above is not just "this check always fails".
git_quiet -C "$CLONE" fetch -q origin main
out="$(bash "$SCRIPT" --agent tester --project "$CLONE" --base origin/main --root "$TMP/wt" --check-only 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -qF "base freshness verified"; then
  ok "positive control: a freshly fetched base passes"
else
  bad "fetched base should pass (exit $rc)"; echo "$out" | sed 's/^/      /'
fi

echo "== 6. worktree location inherits a foreign dune root"
FOREIGN="$TMP/foreign"
mkdir -p "$FOREIGN/wts"
printf '(lang dune 3.15)\n' > "$FOREIGN/dune-project"
expect "refuses a location under a foreign dune root" 3 "foreign dune root" -- \
  bash "$SCRIPT" --agent tester --project "$SOLO" --base HEAD --root "$FOREIGN/wts" --check-only

echo "== 7. happy path actually materializes the environment"
mkdir -p "$SOLO/briefs"; echo '{}' > "$SOLO/briefs/keep-state.json"
WTROOT="$TMP/wtroot"
out="$(bash "$SCRIPT" --agent alpha --project "$SOLO" --base HEAD --root "$WTROOT" 2>/dev/null)"; rc=$?
if [ "$rc" -ne 0 ]; then
  bad "happy path exited $rc"
else
  eval "$out"
  [ -d "$AGENT_WORKTREE" ] && ok "worktree created at $AGENT_WORKTREE" || bad "no worktree"
  [ -L "$AGENT_WORKTREE/briefs" ] && [ -f "$AGENT_WORKTREE/briefs/keep-state.json" ] \
    && ok "briefs/ materialized (symlink, single ledger)" || bad "briefs/ not materialized"
  [ -d "$AGENT_SCRATCH" ] && ok "scratchpad namespace $AGENT_SCRATCH" || bad "no scratchpad"
  [ -f "$AGENT_WORKTREE/dune-workspace" ] && ok "dune root pinned in the worktree" || bad "no dune-workspace"

  # Two agents must not share a scratchpad namespace.
  out2="$(bash "$SCRIPT" --agent beta --project "$SOLO" --base HEAD --root "$WTROOT" 2>/dev/null)"
  s2="$(printf '%s' "$out2" | grep '^AGENT_SCRATCH=' | cut -d= -f2-)"
  if [ -n "$s2" ] && [ "$s2" != "$AGENT_SCRATCH" ]; then
    ok "distinct agents get distinct scratchpads (no shared pr.md)"
  else
    bad "scratchpad namespaces collided: '$AGENT_SCRATCH' vs '$s2'"
  fi

  # Re-dispatching the same agent onto a live worktree must not silently reuse it.
  expect "refuses to reuse an existing worktree path" 3 "already exists" -- \
    bash "$SCRIPT" --agent alpha --project "$SOLO" --base HEAD --root "$WTROOT"
fi

echo "== 8. emitted block is safe to eval"
# The contract says `eval` this output. A path with shell metacharacters must
# therefore survive it as data, never as code.
NASTY="$TMP/we ird\$(touch $TMP/PWNED)'q"
mkdir -p "$NASTY"
git_quiet init -q "$NASTY"
echo x > "$NASTY/x"; git_quiet -C "$NASTY" add x; git_quiet -C "$NASTY" commit -qm init
out="$(bash "$SCRIPT" --agent quoting --project "$NASTY" --base HEAD --root "$TMP/wtq" 2>/dev/null)"; rc=$?
if [ "$rc" -ne 0 ]; then
  bad "bootstrap failed on a path with metacharacters (exit $rc)"
else
  ( eval "$out" ) >/dev/null 2>&1
  if [ -e "$TMP/PWNED" ]; then
    bad "eval of the emitted block EXECUTED embedded code"
  else
    ok "eval of the emitted block does not execute embedded code"
  fi
  evaled_wt="$(eval "$out"; printf '%s' "$AGENT_WORKTREE")"
  case "$evaled_wt" in
    "$TMP/wtq/"*) ok "the eval'd worktree path round-trips intact" ;;
    *) bad "worktree path mangled by eval: $evaled_wt" ;;
  esac
fi

echo "== 9. base freshness cannot pass by failing to check"
# An unreachable origin must refuse, not read as "confirmed fresh".
UNREACH="$TMP/unreach"
git_quiet init -q "$UNREACH"
echo a > "$UNREACH/a"; git_quiet -C "$UNREACH" add a; git_quiet -C "$UNREACH" commit -qm init
git_quiet -C "$UNREACH" remote add origin "$TMP/does-not-exist.git"
git_quiet -C "$UNREACH" update-ref refs/remotes/origin/main "$(git -C "$UNREACH" rev-parse HEAD)"
expect "refuses when origin cannot be reached" 3 "could not verify base freshness" -- \
  bash "$SCRIPT" --agent tester --project "$UNREACH" --base origin/main --root "$TMP/wtu" --check-only
out="$(bash "$SCRIPT" --agent tester --project "$UNREACH" --base origin/main --root "$TMP/wtu" \
        --allow-stale-base --check-only 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -qF "UNVERIFIED"; then
  ok "--allow-stale-base proceeds and records that freshness was unverified"
else
  bad "--allow-stale-base should proceed with an UNVERIFIED note (exit $rc)"; echo "$out"|sed 's/^/      /'
fi

echo "== 10. no attestation to a check that was not run (F5)"
# A local base has no remote to compare against. Proceeding is fine; claiming
# "base freshness verified" is not — the caller cannot then distinguish
# "checked and fresh" from "never checked".
out="$(bash "$SCRIPT" --agent tester --project "$SOLO" --base HEAD --root "$TMP/wtf5" --check-only 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -qF "NOT CHECKED"; then
  ok "a non-origin base reports NOT CHECKED, never 'verified'"
else
  bad "non-origin base attested to an unrun check (exit $rc)"; echo "$out"|sed 's/^/      /'
fi
if printf '%s' "$out" | grep -qF "base freshness verified"; then
  bad "output still claims 'base freshness verified' for an unchecked base"
else
  ok "the phrase 'base freshness verified' is absent when nothing was checked"
fi
# And the positive control: a real, reachable, fresh origin base DOES say verified.
git_quiet -C "$CLONE" fetch -q origin main
out="$(bash "$SCRIPT" --agent tester --project "$CLONE" --base origin/main --root "$TMP/wtf5b" --check-only 2>&1)"
if printf '%s' "$out" | grep -qF "verified against origin/main"; then
  ok "positive control: a genuinely verified base says so, naming the ref"
else
  bad "a verified base failed to report verification"; echo "$out"|sed 's/^/      /'
fi

echo "== 11. --check-only runs EVERY precondition (F6)"
# The header promises --check-only runs every check. The collision check used
# to live after the early exit, so --check-only cleared a directory another
# agent was live in.
COLLIDE="$TMP/wtcollide"
mkdir -p "$COLLIDE/$(basename "$SOLO")-t1"
echo junk > "$COLLIDE/$(basename "$SOLO")-t1/OTHER_AGENT_IS_HERE"
expect "refuses an occupied worktree path in --check-only mode" 3 "already exists" -- \
  bash "$SCRIPT" --agent t1 --project "$SOLO" --base HEAD --root "$COLLIDE" --check-only

echo "== 12. one stray tracked file must not re-open #101 (F6)"
STRAY="$TMP/strayouter"
git_quiet init -q "$STRAY"
echo outer > "$STRAY/README"; git_quiet -C "$STRAY" add README; git_quiet -C "$STRAY" commit -qm init
mkdir -p "$STRAY/proj/src"
echo 'let () = ()' > "$STRAY/proj/src/main.ml"
echo 'stray' > "$STRAY/proj/STRAY.md"
git_quiet -C "$STRAY" add proj/STRAY.md          # exactly ONE file tracked
git_quiet -C "$STRAY" commit -qm "one stray file"
expect "partial tracking is refused, not downgraded to a warning" 3 "PARTIALLY TRACKED" -- \
  bash "$SCRIPT" --agent tester --project "$STRAY/proj" --base HEAD --check-only
out="$(bash "$SCRIPT" --agent tester --project "$STRAY/proj" --base HEAD --root "$TMP/wtp" \
        --allow-partial-tracking --check-only 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -qF "will be ABSENT"; then
  ok "--allow-partial-tracking proceeds and names what the agent will not have"
else
  bad "--allow-partial-tracking should proceed with a warning (exit $rc)"; echo "$out"|sed 's/^/      /'
fi
# Positive control: fully tracked subdirectory still proceeds with a warning.
git_quiet -C "$STRAY" add proj; git_quiet -C "$STRAY" commit -qm "track all of proj"
out="$(bash "$SCRIPT" --agent tester --project "$STRAY/proj" --base HEAD --root "$TMP/wtp2" --check-only 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -qF "subdirectory of the repo"; then
  ok "positive control: a FULLY tracked subdirectory still proceeds"
else
  bad "fully tracked subdirectory should proceed (exit $rc)"; echo "$out"|sed 's/^/      /'
fi

echo "== 13. a relative --root must not hang (F7)"
out="$(cd "$TMP" && timeout 10 bash "$SCRIPT" --agent tester --project "$SOLO" --base HEAD \
        --root ./relroot --check-only 2>&1)"; rc=$?
if [ "$rc" -eq 124 ]; then
  bad "relative --root still hangs (exit 124)"
elif [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ]; then
  ok "relative --root terminates (exit $rc), no infinite ancestor walk"
else
  bad "relative --root gave unexpected exit $rc"; echo "$out"|sed 's/^/      /'
fi
out="$(cd "$TMP" && timeout 10 env AGENT_WORKTREE_ROOT=./envrelroot bash "$SCRIPT" \
        --agent tester --project "$SOLO" --base HEAD --check-only 2>&1)"; rc=$?
if [ "$rc" -eq 124 ]; then
  bad "relative AGENT_WORKTREE_ROOT still hangs"
else
  ok "relative AGENT_WORKTREE_ROOT terminates (exit $rc)"
fi

echo "== 14. option validation covers more than --agent (F7)"
expect "rejects a --branch git will not accept" 2 "--branch contains" -- \
  bash "$SCRIPT" --agent tester --project "$SOLO" --branch 'bad branch;rm -rf /'
expect "rejects a --base with shell metacharacters" 2 "--base contains" -- \
  bash "$SCRIPT" --agent tester --project "$SOLO" --base 'x$(id)'
expect "rejects an empty --root" 2 "--root must not be empty" -- \
  bash "$SCRIPT" --agent tester --project "$SOLO" --root ''

echo "== 15. no machine-specific default root (F11)"
if grep -q '/mnt/ssd-external-2to' "$SCRIPT"; then
  bad "script still hardcodes a single workstation's mount as a default"
else
  ok "no hardcoded machine-specific path in the script"
fi
out="$(env -u AGENT_WORKTREE_ROOT bash "$SCRIPT" --agent tester --project "$SOLO" --base HEAD --check-only 2>&1)"
if printf '%s' "$out" | grep -qE "worktree=(${XDG_CACHE_HOME:-$HOME/.cache})/agent-worktrees/"; then
  ok "default root falls back to XDG cache, which exists on any host"
else
  bad "unexpected default root"; echo "$out"|sed 's/^/      /'
fi

echo "== 16. usage errors"
expect "requires --agent" 2 "--agent <name> is required" -- bash "$SCRIPT" --project "$SOLO"
expect "rejects a shell-unsafe agent name" 2 "must match" -- bash "$SCRIPT" --agent 'a;rm -rf /'

echo ""
echo "agent-worktree-bootstrap.test: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
