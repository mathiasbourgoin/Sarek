#!/usr/bin/env bash
# Tests for scripts/roster-implement-posthook.sh and the two checkers it runs.
#
# Every check gets a constructed bad input and an asserted refusal, plus a
# positive control that the same check passes on good input — otherwise "it
# failed" and "it always fails" are the same observation.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/roster-implement-posthook.sh"
ALCO="$HERE/check-alcotest-registration.js"

PASS=0; FAIL=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
ok()  { PASS=$((PASS+1)); echo "  ok — $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL — $1"; }

expect() { # <label> <rc> <substr> -- cmd...
  local label="$1" want_rc="$2" want="$3"; shift 4
  local out rc; out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -ne "$want_rc" ]; then bad "$label: exit $rc wanted $want_rc"; echo "$out"|sed 's/^/      /'; return; fi
  if ! printf '%s' "$out" | grep -qF -- "$want"; then bad "$label: missing '$want'"; echo "$out"|sed 's/^/      /'; return; fi
  ok "$label"
}

mkproj() { # <name> -> echoes path, with a valid ledger and a minimal OCaml suite
  local p="$TMP/$1"; mkdir -p "$p/briefs" "$p/t"
  # Every fixture carries sources, because the hook now (correctly) refuses a
  # tree where the Alcotest check would examine nothing.
  cat > "$p/t/test_fixture.ml" <<'ML'
let test_a () = Alcotest.(check int) "a" 1 1
let () = Alcotest.run "S" [ ("g", [ Alcotest.test_case "a" `Quick test_a ]) ]
ML
  cat > "$p/briefs/demo-state.json" <<'JSON'
{ "task": "demo", "mode": "fast", "current_phase": "implement",
  "events": [ { "phase": "implement", "outcome": "COMPLETED", "by": "roster-implement" } ] }
JSON
  echo "$p"
}

echo "== A. ledger schema"
P="$(mkproj good)"
expect "accepts a well-formed ledger (positive control)" 0 "ledger schema OK" -- \
  bash "$HOOK" --task demo --project "$P"

P="$(mkproj missing)"; rm "$P/briefs/demo-state.json"
expect "refuses a missing ledger" 1 "MISSING ledger" -- bash "$HOOK" --task demo --project "$P"

P="$(mkproj unparseable)"; echo '{not json' > "$P/briefs/demo-state.json"
expect "refuses an unparseable ledger" 1 "not parseable JSON" -- bash "$HOOK" --task demo --project "$P"

P="$(mkproj badoutcome)"
cat > "$P/briefs/demo-state.json" <<'JSON'
{ "task": "demo", "mode": "fast", "current_phase": "ship",
  "events": [ { "phase": "ship", "outcome": "PARTIAL" } ] }
JSON
expect "refuses an outcome illegal for its phase" 1 "is not legal for phase ship" -- \
  bash "$HOOK" --task demo --project "$P"

P="$(mkproj badreason)"
cat > "$P/briefs/demo-state.json" <<'JSON'
{ "task": "demo", "mode": "fast", "current_phase": "implement",
  "events": [ { "phase": "implement", "outcome": "PARTIAL", "reason": false } ] }
JSON
expect "refuses a non-string reason" 1 "reason must be a string" -- bash "$HOOK" --task demo --project "$P"

P="$(mkproj phasedrift)"
cat > "$P/briefs/demo-state.json" <<'JSON'
{ "task": "demo", "mode": "fast", "current_phase": "review",
  "events": [ { "phase": "implement", "outcome": "COMPLETED" } ] }
JSON
expect "refuses current_phase not matching the last event" 1 "does not match the last event" -- \
  bash "$HOOK" --task demo --project "$P"

P="$(mkproj wrongmode)"
cat > "$P/briefs/demo-state.json" <<'JSON'
{ "task": "demo", "mode": "express", "current_phase": "qa",
  "events": [ { "phase": "qa", "outcome": "GO" } ] }
JSON
expect "refuses a phase outside the mode's sequence" 1 "is not part of the express sequence" -- \
  bash "$HOOK" --task demo --project "$P"

P="$(mkproj wrongtask)"
mv "$P/briefs/demo-state.json" "$P/briefs/other-state.json"   # filename says "other", content says "demo"
expect "refuses a ledger belonging to another task" 1 "task mismatch" -- \
  bash "$HOOK" --task other --project "$P"

if command -v jq >/dev/null 2>&1; then
  # The tracked schema and the jq predicate the skills execute must agree. If
  # they can disagree without anyone noticing, having two of them is worse than
  # having one.
  write_skill() { # <project> <predicate-body>
    mkdir -p "$1/.harness/skills"
    { printf "LEDGER_SCHEMA='"; printf '%s' "$2"; printf "'\n"; } > "$1/.harness/skills/roster-run.md"
  }
  AGREE='.task == $t and (.events|length) > 0'
  DISAGREE='.task == "definitely-not-this-task"'

  P="$(mkproj jqagree)"; write_skill "$P" "$AGREE"
  expect "reports agreement with the jq predicate" 0 "agrees with the jq predicate" -- \
    bash "$HOOK" --task demo --project "$P"

  P="$(mkproj jqdrift)"; write_skill "$P" "$DISAGREE"
  expect "fails when the jq predicate disagrees with the tracked schema" 1 "SCHEMA DRIFT" -- \
    bash "$HOOK" --task demo --project "$P"
else
  echo "  skip — jq absent, cross-check not exercised (this is a GAP, not a pass)"
fi

echo "== B. worktree briefs/ consolidation"
P="$(mkproj consol)"; WT="$TMP/consol-wt"; mkdir -p "$WT/briefs"
echo "impl notes" > "$WT/briefs/demo-impl.md"
cp "$P/briefs/demo-state.json" "$WT/briefs/demo-state.json"
out="$(bash "$HOOK" --task demo --project "$P" --worktree "$WT" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && [ -f "$P/briefs/demo-impl.md" ]; then
  ok "copies worktree-only briefs into the project"
else
  bad "consolidation did not copy (exit $rc)"; echo "$out"|sed 's/^/      /'
fi

P="$(mkproj conflict)"; WT="$TMP/conflict-wt"; mkdir -p "$WT/briefs"
echo "worktree version" > "$WT/briefs/demo-impl.md"
echo "project version"  > "$P/briefs/demo-impl.md"
cp "$P/briefs/demo-state.json" "$WT/briefs/demo-state.json"
expect "refuses to silently resolve a divergent brief" 1 "CONFLICT briefs/demo-impl.md" -- \
  bash "$HOOK" --task demo --project "$P" --worktree "$WT"
if [ "$(cat "$TMP/conflict/briefs/demo-impl.md")" = "project version" ]; then
  ok "neither side was overwritten on conflict"
else
  bad "conflict overwrote the project copy"
fi

P="$(mkproj symlinked)"; WT="$TMP/symlinked-wt"; mkdir -p "$WT"
ln -s "$P/briefs" "$WT/briefs"
expect "recognizes an already-shared (symlinked) briefs/" 0 "already shared" -- \
  bash "$HOOK" --task demo --project "$P" --worktree "$WT"

echo "== C. Alcotest case registration"
SUITE="$TMP/suite"; mkdir -p "$SUITE"
cat > "$SUITE/test_ok.ml" <<'ML'
let test_alpha () = Alcotest.(check int) "a" 1 1
let test_beta () = Alcotest.(check int) "b" 2 2
let () =
  Alcotest.run "S"
    [ ("g", [ Alcotest.test_case "alpha" `Quick test_alpha;
              Alcotest.test_case "beta" `Quick test_beta ]) ]
ML
expect "passes when every case is registered (positive control)" 0 "1 Alcotest suite(s), every test_* case registered" -- \
  node "$ALCO" "$SUITE"

cat > "$SUITE/test_orphan.ml" <<'ML'
let test_alpha () = Alcotest.(check int) "a" 1 1
(* written, reviewed, merged — and never added to the list below *)
let test_forgotten () = Alcotest.(check int) "never runs" 0 1
let () = Alcotest.run "S" [ ("g", [ Alcotest.test_case "alpha" `Quick test_alpha ]) ]
ML
expect "detects a case that is defined but never registered" 1 "UNREGISTERED" -- node "$ALCO" "$SUITE"
expect "names the orphan" 1 "test_forgotten" -- node "$ALCO" "$SUITE"

# A registration that has been commented out must NOT count as a registration.
rm "$SUITE/test_orphan.ml"
cat > "$SUITE/test_commented.ml" <<'ML'
let test_alpha () = Alcotest.(check int) "a" 1 1
let test_disabled () = Alcotest.(check int) "d" 1 1
let () =
  Alcotest.run "S"
    [ ("g", [ Alcotest.test_case "alpha" `Quick test_alpha
              (* ; Alcotest.test_case "disabled" `Quick test_disabled *) ]) ]
ML
expect "a commented-out registration does not count" 1 "test_disabled" -- node "$ALCO" "$SUITE"

# A name mentioned only inside a string literal must not count either.
rm "$SUITE/test_commented.ml"
cat > "$SUITE/test_stringy.ml" <<'ML'
let test_alpha () = Alcotest.(check int) "a" 1 1
let test_ghost () = Alcotest.(check int) "g" 1 1
let () =
  Alcotest.run "S"
    [ ("g", [ Alcotest.test_case "see test_ghost for details" `Quick test_alpha ]) ]
ML
expect "a name appearing only in a string does not count" 1 "test_ghost" -- node "$ALCO" "$SUITE"

# Non-Alcotest files are out of scope.
rm "$SUITE"/*.ml
cat > "$SUITE/plain.ml" <<'ML'
let test_helper () = ()
let () = ignore test_helper
ML
expect "reports zero suites rather than claiming a vacuous pass" 0 "0 Alcotest suite(s)" -- node "$ALCO" "$SUITE"
expect "errors on a target path that does not exist" 2 "does not exist" -- node "$ALCO" "$TMP/nonexistent"

echo "== D. the phase this hook exists for must actually have run (F9)"
# The scenario: /roster-implement dies before appending its event. The ledger
# is well-formed, the schema is satisfied, and it describes the phase BEFORE
# implement. Validating it and printing OK is how a skipped phase becomes an
# inexplicable failure several phases later.
P="$(mkproj noimpl)"
cat > "$P/briefs/demo-state.json" <<'JSON'
{ "task": "demo", "mode": "full", "current_phase": "plan",
  "events": [ { "phase": "plan", "outcome": "COMPLETED", "by": "roster-plan" } ] }
JSON
expect "refuses a ledger with no implement event at all" 1 "there is no implement event anywhere" -- \
  bash "$HOOK" --task demo --project "$P"
expect "names it as a stale ledger" 1 "STALE ledger" -- bash "$HOOK" --task demo --project "$P"

# Implement ran, but the ledger has moved past it — also not this hook's moment.
P="$(mkproj movedon)"
cat > "$P/briefs/demo-state.json" <<'JSON'
{ "task": "demo", "mode": "fast", "current_phase": "review",
  "events": [ { "phase": "implement", "outcome": "COMPLETED" },
              { "phase": "review", "outcome": "GO" } ] }
JSON
expect "refuses when the latest event has moved past implement" 1 "moved on since" -- \
  bash "$HOOK" --task demo --project "$P"

# Positive control: the ordinary case must still pass, including after a
# loop-back (review NO-GO then implement again), which is legal.
P="$(mkproj loopback)"
cat > "$P/briefs/demo-state.json" <<'JSON'
{ "task": "demo", "mode": "fast", "current_phase": "implement",
  "events": [ { "phase": "implement", "outcome": "COMPLETED" },
              { "phase": "review", "outcome": "NO-GO", "reason": "must-fix" },
              { "phase": "implement", "outcome": "PARTIAL", "reason": "budget" } ] }
JSON
expect "a loop-back ledger ending in implement passes" 0 "latest ledger event is implement" -- \
  bash "$HOOK" --task demo --project "$P"

# --expect-phase makes the mechanism general rather than implement-only.
expect "--expect-phase review refuses the same ledger" 1 "expected \"review\"" -- \
  bash "$HOOK" --task demo --project "$P" --expect-phase review

echo "== E. a check that examined nothing is not a pass (F9)"
# check-alcotest-registration exits 0 on a tree with no .ml files. Consuming
# only its exit code turned "I verified nothing" into a green.
P="$(mkproj noml)"; rm -rf "$P/t"
expect "zero scanned .ml files fails instead of passing green" 1 "scanned ZERO .ml files" -- \
  bash "$HOOK" --task demo --project "$P"
P="$(mkproj withml)"
expect "positive control: a tree with sources passes" 0 "roster-implement-posthook: OK" -- \
  bash "$HOOK" --task demo --project "$P"

echo "== F. the hook says which project it validated (F9)"
P="$(mkproj named)"
out="$(bash "$HOOK" --task demo --project "$P" 2>&1)"
if printf '%s' "$out" | grep -qF "project=$P"; then
  ok "reports the project, task and expected phase it acted on"
else
  bad "hook did not state which project it validated"; echo "$out"|sed 's/^/      /'
fi
# An auto-detected project with no briefs/ must refuse rather than validate the
# enclosing repo's ledger — the wrong-repo topology, reached from the other side.
NOBRIEFS="$TMP/nobriefs"; mkdir -p "$NOBRIEFS/sub"
git_q_init() { git -c init.defaultBranch=main -c user.email=t@t -c user.name=t init -q "$1"; }
git_q_init "$NOBRIEFS"
expect "refuses an auto-detected project with no briefs/" 2 "Pass --project" -- \
  sh -c "cd '$NOBRIEFS/sub' && bash '$HOOK' --task demo"

echo "== G. the cross-check reports when it did not run (F9)"
P="$(mkproj nocross)"
out="$(bash "$HOOK" --task demo --project "$P" 2>&1)"
if printf '%s' "$out" | grep -qF "NOT cross-checked"; then
  ok "absence of the jq/skill cross-check is reported, not silently skipped"
else
  bad "cross-check absence was silent"; echo "$out"|sed 's/^/      /'
fi

echo "== H. usage"
expect "requires --task" 2 "--task <slug> is required" -- bash "$HOOK"
expect "rejects a malformed task slug" 2 "must match" -- bash "$HOOK" --task "Bad Slug"

echo ""
echo "roster-implement-posthook.test: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
