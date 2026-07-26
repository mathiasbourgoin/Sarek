#!/usr/bin/env bash
# roster-implement-posthook.sh — runs after /roster-implement, before review.
#
# Backlog #103. Three things that were left to prose discipline and therefore
# failed silently:
#
#   A. LEDGER SCHEMA. briefs/<task>-state.json drives resume routing. A
#      malformed ledger does not fail here — it fails several phases later as
#      an inexplicable re-run or a skipped phase. Upstream encodes the rules as
#      a jq predicate inside an untracked skill file, so a fresh clone has no
#      schema at all. scripts/lib/ledger-schema.js is the tracked authority;
#      when jq and the skill file are both present the two are cross-checked
#      against each other, so a drift between them is a failure rather than a
#      coin flip about which one you happened to run.
#
#   B. WORKTREE briefs/. An implementer working in a git worktree may have a
#      real briefs/ directory of its own rather than the symlink
#      agent-worktree-bootstrap.sh installs. Its ledger events and artifacts
#      are then invisible to the next phase, which runs in the main checkout.
#      Consolidated here — and a genuine conflict (both sides changed the same
#      file to different content) is reported, never silently resolved.
#
#   C. ALCOTEST REGISTRATION. A case written but never added to the suite list
#      runs never and reports nothing, while the suite still reports green.
#
# Usage: scripts/roster-implement-posthook.sh --task <slug> [--project <dir>]
#                                             [--worktree <dir>] [--no-consolidate]
# Exit: 0 all checks pass; 1 a check failed; 2 usage/environment error.
set -uo pipefail

TASK=""
PROJECT=""
WORKTREE=""
CONSOLIDATE=1
EXPECT_PHASE="implement"

die_usage() { echo "roster-implement-posthook: $1" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK="${2:-}"; shift ;;
    --project) PROJECT="${2:-}"; shift ;;
    --worktree) WORKTREE="${2:-}"; shift ;;
    --no-consolidate) CONSOLIDATE=0 ;;
    --expect-phase) EXPECT_PHASE="${2:-}"; shift ;;
    *) die_usage "unknown option: $1" ;;
  esac
  shift
done

[ -n "$TASK" ] || die_usage "--task <slug> is required"
case "$TASK" in
  *[!a-z0-9-]*) die_usage "--task must match [a-z0-9-]+ : $TASK" ;;
esac

PROJECT_WAS_DEFAULTED=0
if [ -z "$PROJECT" ]; then
  PROJECT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  PROJECT_WAS_DEFAULTED=1
fi
[ -d "$PROJECT" ] || die_usage "--project is not a directory: $PROJECT"
PROJECT="$(cd "$PROJECT" && pwd -P)"

# F9: defaulting to `git rev-parse --show-toplevel` silently resolves to the
# ENCLOSING repo when run from an untracked project inside another one — the
# same wrong-repo topology #101 exists to catch, arrived at from the other
# direction. Assert the project actually looks like the task's home before
# validating anything about it, and always say which project was used: a hook
# that validated the wrong repo's ledger would otherwise report a confident OK.
echo "roster-implement-posthook: project=$PROJECT task=$TASK expect-phase=$EXPECT_PHASE"
if [ ! -d "$PROJECT/briefs" ]; then
  if [ "$PROJECT_WAS_DEFAULTED" -eq 1 ]; then
    die_usage "no briefs/ under the auto-detected project $PROJECT.
  The project was inferred from the current directory, and git resolved it to
  that repository — which may not be the task's repo at all. Pass --project
  explicitly."
  fi
  die_usage "no briefs/ under --project $PROJECT"
fi
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

command -v node >/dev/null 2>&1 || die_usage "node is required"

FAILED=0
fail() { echo "roster-implement-posthook: $1" >&2; FAILED=1; }

# ── B first: consolidate, so A validates the ledger that actually results ────
if [ "$CONSOLIDATE" -eq 1 ] && [ -n "$WORKTREE" ]; then
  [ -d "$WORKTREE" ] || die_usage "--worktree is not a directory: $WORKTREE"
  WORKTREE="$(cd "$WORKTREE" && pwd -P)"
  WT_BRIEFS="$WORKTREE/briefs"
  if [ ! -e "$WT_BRIEFS" ]; then
    echo "roster-implement-posthook: worktree has no briefs/ — nothing to consolidate."
  elif [ -L "$WT_BRIEFS" ]; then
    echo "roster-implement-posthook: worktree briefs/ is a symlink — already shared, nothing to consolidate."
  else
    mkdir -p "$PROJECT/briefs"
    moved=0
    while IFS= read -r -d '' src; do
      rel="${src#"$WT_BRIEFS"/}"
      dst="$PROJECT/briefs/$rel"
      if [ -e "$dst" ] && ! cmp -s "$src" "$dst"; then
        fail "CONFLICT briefs/$rel differs between the worktree and $PROJECT/briefs."
        echo "    worktree: $src" >&2
        echo "    project:  $dst" >&2
        echo "    Reconcile by hand; refusing to overwrite either side." >&2
        continue
      fi
      if [ ! -e "$dst" ]; then
        # A swallowed copy failure is the worst outcome available here: the
        # hook reports "consolidated N files" and exits 0 while the ledger
        # events the next phase depends on never arrived. Full disk, bad
        # permissions and a vanished source all land here.
        if ! mkdir -p "$(dirname "$dst")" 2>/dev/null; then
          fail "CONSOLIDATION FAILED could not create $(dirname "$dst") — briefs/$rel was not carried over."
          continue
        fi
        if ! cp -a "$src" "$dst" 2>/dev/null; then
          fail "CONSOLIDATION FAILED could not copy briefs/$rel from the worktree — the next phase will not see it."
          continue
        fi
        moved=$((moved+1))
      fi
    done < <(find "$WT_BRIEFS" -type f -print0)
    echo "roster-implement-posthook: consolidated $moved file(s) from the worktree briefs/."
  fi
fi

# ── A: ledger schema ────────────────────────────────────────────────────────
LEDGER="$PROJECT/briefs/$TASK-state.json"
if [ ! -f "$LEDGER" ]; then
  fail "MISSING ledger $LEDGER — /roster-implement must append its event before handing off."
else
  if ! node -e '
const { validateLedger } = require(process.argv[1]);
const fs = require("fs");
let data;
try { data = JSON.parse(fs.readFileSync(process.argv[2], "utf8")); }
catch (e) { console.error("  ledger is not parseable JSON: " + e.message); process.exit(1); }
const r = validateLedger(data, process.argv[3]);
if (!r.valid) { for (const m of r.errors) console.error("  " + m); process.exit(1); }
' "$SCRIPTS/lib/ledger-schema.js" "$LEDGER" "$TASK"; then
    fail "INVALID ledger $LEDGER (see above)."
  else
    echo "roster-implement-posthook: ledger schema OK ($LEDGER)."
  fi

  # The schema says the ledger is well-formed. It cannot say the phase this
  # hook exists for actually ran.
  if [ -n "$EXPECT_PHASE" ]; then
    if ! node -e '
const { expectLatestPhase } = require(process.argv[1]);
const fs = require("fs");
let data;
try { data = JSON.parse(fs.readFileSync(process.argv[2], "utf8")); } catch { process.exit(1); }
const r = expectLatestPhase(data, process.argv[3]);
if (!r.valid) { for (const m of r.errors) console.error("  " + m); process.exit(1); }
' "$SCRIPTS/lib/ledger-schema.js" "$LEDGER" "$EXPECT_PHASE"; then
      fail "PHASE NOT RECORDED $EXPECT_PHASE (see above) — refusing to hand a stale ledger to the next phase."
    else
      echo "roster-implement-posthook: latest ledger event is $EXPECT_PHASE, as expected."
    fi
  fi

  # Cross-check against the jq predicate the skills actually execute. A
  # disagreement means one of the two is wrong and neither can be trusted.
  SKILL="$PROJECT/.harness/skills/roster-run.md"
  # The cross-check needs jq AND the untracked roster skill file, so in a fresh
  # clone it is inert. Silently skipping made "ledger schema OK" mean two
  # different things depending on the machine. Say which one this was.
  if ! command -v jq >/dev/null 2>&1; then
    echo "roster-implement-posthook: NOTE — jq absent; the tracked schema was NOT cross-checked against the jq predicate (single-source verdict)."
  elif [ ! -f "$SKILL" ]; then
    echo "roster-implement-posthook: NOTE — $SKILL absent (untracked roster install); the tracked schema was NOT cross-checked (single-source verdict)."
  fi
  if command -v jq >/dev/null 2>&1 && [ -f "$SKILL" ]; then
    PRED="$(node -e '
const fs = require("fs");
const m = /LEDGER_SCHEMA='"'"'([\s\S]*?)'"'"'\n/.exec(fs.readFileSync(process.argv[1], "utf8"));
if (m) process.stdout.write(m[1]);
' "$SKILL")"
    if [ -n "$PRED" ]; then
      if jq -e --arg t "$TASK" "$PRED" "$LEDGER" >/dev/null 2>&1; then jq_ok=0; else jq_ok=1; fi
      node -e '
const { validateLedger } = require(process.argv[1]);
const fs = require("fs");
try {
  const r = validateLedger(JSON.parse(fs.readFileSync(process.argv[2], "utf8")), process.argv[3]);
  process.exit(r.valid ? 0 : 1);
} catch { process.exit(1); }
' "$SCRIPTS/lib/ledger-schema.js" "$LEDGER" "$TASK"
      node_ok=$?
      if [ "$jq_ok" -ne "$node_ok" ]; then
        fail "SCHEMA DRIFT: the tracked ledger schema and the jq predicate in $SKILL disagree (node=$node_ok jq=$jq_ok). One of them is wrong; neither verdict is trustworthy."
      else
        echo "roster-implement-posthook: ledger schema agrees with the jq predicate in roster-run.md."
      fi
    fi
  fi
fi

# ── C: Alcotest case registration ───────────────────────────────────────────
# The checker reports how much it actually scanned. Consuming only its exit
# code throws that away — and "scanned 0 .ml file(s)" exits 0, so a wrong
# --project produced a confident green from a checker that examined nothing.
ALCO_OUT="$(node "$SCRIPTS/check-alcotest-registration.js" "$PROJECT" 2>&1)"
ALCO_RC=$?
printf '%s\n' "$ALCO_OUT"
if [ "$ALCO_RC" -ne 0 ]; then
  fail "unregistered Alcotest case(s) — see above."
elif printf '%s' "$ALCO_OUT" | grep -q "scanned 0 .ml file"; then
  fail "the Alcotest check scanned ZERO .ml files under $PROJECT — it verified nothing. Wrong --project, or the sources are missing."
fi

if [ "$FAILED" -ne 0 ]; then
  echo "roster-implement-posthook: FAIL — do not hand off to /roster-review until the above is fixed." >&2
  exit 1
fi
echo "roster-implement-posthook: OK."
