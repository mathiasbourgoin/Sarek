#!/usr/bin/env bash
# agent-worktree-bootstrap.sh — safe dispatch of a worktree-isolated agent.
#
# Backlog #101. Four independent ways `isolation: "worktree"` silently produces
# an agent that is working on the wrong thing, plus one way two agents clobber
# each other:
#
#   1. WRONG REPO. Worktree isolation resolves the enclosing git repository
#      from the working directory. When the project directory is an UNTRACKED
#      directory inside another repository (`~/dev/SPOC` inside the `~/dev`
#      personal-home repo), the enclosing repository is the *parent*. The agent
#      gets a worktree of the parent, which does not contain the project's
#      sources at all, and reports "no such file" for everything. Nothing warns.
#
#   2. STALE BASE. A worktree branched off a local `origin/main` that has not
#      been fetched applies the agent's work to old sources. The diff looks
#      plausible and the conflict surfaces at merge time.
#
#   3. MISSING briefs/. briefs/ is git-ignored, so it does not exist in a fresh
#      worktree. Pipeline control files, the ledger, and the xruntime journal
#      all live there; an agent silently starts with no pipeline state.
#
#   4. WRONG DUNE ROOT. dune resolves its root by walking UP to the outermost
#      ancestor holding dune-workspace/dune-project. A worktree placed under
#      such an ancestor builds the ancestor's project.
#
#   5. SHARED SCRATCHPAD. Two agents both writing the generic `pr.md` — one
#      published the other's text as a PR description. Each agent gets its own
#      scratchpad namespace here so a generic filename can no longer collide.
#
# Usage:
#   scripts/agent-worktree-bootstrap.sh --agent <name> [options]
#
#   --agent <name>      required; identifies the agent, namespaces the worktree
#                       and the scratchpad
#   --project <dir>     project directory to isolate (default: cwd)
#   --base <ref>        base ref for the new branch (default: origin/<default>)
#   --branch <name>     branch to create (default: agent/<agent>-<timestamp>)
#   --root <dir>        where worktrees are created (default: $AGENT_WORKTREE_ROOT
#                       or /mnt/ssd-external-2to/spoc-pr-wt)
#   --check-only        run every precondition check and exit; create nothing
#   --allow-stale-base  proceed despite a stale base (records why in the report)
#
# On success prints a shell-sourceable block:
#   AGENT_WORKTREE=... AGENT_BRANCH=... AGENT_SCRATCH=... AGENT_DUNE_ROOT=...
#
# Exit: 0 ok; 2 usage; 3 refused (a precondition failed — nothing was created).
set -uo pipefail

AGENT=""
PROJECT=""
BASE=""
BRANCH=""
WT_ROOT="${AGENT_WORKTREE_ROOT:-/mnt/ssd-external-2to/spoc-pr-wt}"
CHECK_ONLY=0
ALLOW_STALE=0

die_usage() { echo "agent-worktree-bootstrap: $1" >&2; exit 2; }
refuse() {
  echo "" >&2
  echo "agent-worktree-bootstrap: REFUSED — $1" >&2
  shift
  for line in "$@"; do echo "  $line" >&2; done
  echo "  Nothing was created. Fix the above and re-run." >&2
  exit 3
}

while [ $# -gt 0 ]; do
  case "$1" in
    --agent) AGENT="${2:-}"; shift ;;
    --project) PROJECT="${2:-}"; shift ;;
    --base) BASE="${2:-}"; shift ;;
    --branch) BRANCH="${2:-}"; shift ;;
    --root) WT_ROOT="${2:-}"; shift ;;
    --check-only) CHECK_ONLY=1 ;;
    --allow-stale-base) ALLOW_STALE=1 ;;
    *) die_usage "unknown option: $1" ;;
  esac
  shift
done

[ -n "$AGENT" ] || die_usage "--agent <name> is required (it namespaces the worktree and scratchpad)"
case "$AGENT" in
  *[!a-zA-Z0-9._-]*) die_usage "--agent must match [A-Za-z0-9._-]+ : $AGENT" ;;
esac

PROJECT="${PROJECT:-$PWD}"
[ -d "$PROJECT" ] || die_usage "--project is not a directory: $PROJECT"
PROJECT="$(cd "$PROJECT" && pwd -P)"

# ── Check 1: the enclosing repository must BE the project ────────────────────
TOPLEVEL="$(git -C "$PROJECT" rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$TOPLEVEL" ]; then
  refuse "$PROJECT is not inside any git repository." \
         "Worktree isolation has nothing to branch from. Run 'git init' in the" \
         "project, or dispatch a non-isolated agent."
fi
TOPLEVEL="$(cd "$TOPLEVEL" && pwd -P)"

if [ "$TOPLEVEL" != "$PROJECT" ]; then
  # The project is a subdirectory of some other repository. That is only
  # tolerable if the outer repository actually TRACKS the project's files. If
  # it does not, a worktree of the outer repo contains no project sources —
  # this is the #101 defect, and it must be a hard refusal, not a warning.
  if git -C "$TOPLEVEL" ls-files --error-unmatch -- "$PROJECT" >/dev/null 2>&1; then
    echo "agent-worktree-bootstrap: warning — $PROJECT is a subdirectory of the repo" >&2
    echo "  at $TOPLEVEL. The worktree will contain the whole outer repository." >&2
  else
    refuse "worktree isolation would target the WRONG REPOSITORY." \
           "project:          $PROJECT" \
           "enclosing repo:   $TOPLEVEL" \
           "$PROJECT is UNTRACKED in $TOPLEVEL, so a worktree of that repo" \
           "would not contain the project's sources — the agent would find" \
           "nothing and report every file as missing." \
           "" \
           "Redirect: make the project its own repository" \
           "  git -C $PROJECT init && git -C $PROJECT add -A && git -C $PROJECT commit" \
           "or dispatch a NON-isolated agent that works in $PROJECT directly."
  fi
fi

# ── Check 2: the base ref must exist and be fresh ────────────────────────────
if [ -z "$BASE" ]; then
  DEFAULT_BRANCH="$(git -C "$PROJECT" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null)"
  DEFAULT_BRANCH="${DEFAULT_BRANCH:-origin/main}"
  BASE="$DEFAULT_BRANCH"
fi

git -C "$PROJECT" rev-parse --verify --quiet "$BASE^{commit}" >/dev/null || \
  refuse "base ref does not resolve: $BASE" \
         "Pass an existing ref with --base, or fetch it first."

STALE_NOTE="base freshness verified"
case "$BASE" in
  origin/*)
    REMOTE_BRANCH="${BASE#origin/}"
    # Deliberately NOT auto-fetching. Two reasons: a silent fetch mutates a
    # checkout that other agents share, and it would paper over exactly the
    # condition being checked — the operator would never learn their base was
    # stale, only that it "worked this time".
    LOCAL_SHA="$(git -C "$PROJECT" rev-parse "$BASE^{commit}")"

    # ls-remote is a blocking network call. Left unbounded it hangs the whole
    # dispatch on a slow or offline network; left unchecked, any failure
    # (no route, auth, DNS) yields an empty REMOTE_SHA that the comparison
    # below reads as "confirmed fresh". That is the same fail-open this check
    # exists to prevent, just relocated from a missing fetch to a missing
    # network — and it is worse, because it is silent.
    LS_OUT="$(mktemp)"
    if timeout 15 git -C "$PROJECT" ls-remote origin "refs/heads/$REMOTE_BRANCH" >"$LS_OUT" 2>&1; then
      REMOTE_SHA="$(cut -f1 <"$LS_OUT")"
    else
      LS_RC=$?
      LS_ERR="$(head -c 500 "$LS_OUT")"
      rm -f "$LS_OUT"
      if [ "$ALLOW_STALE" -eq 1 ]; then
        STALE_NOTE="base freshness UNVERIFIED (ls-remote failed rc=$LS_RC), accepted via --allow-stale-base"
        echo "agent-worktree-bootstrap: warning — $STALE_NOTE" >&2
        REMOTE_SHA=""
      else
        [ "$LS_RC" -eq 124 ] && LS_ERR="timed out after 15s"
        refuse "could not verify base freshness against origin." \
               "ls-remote exit $LS_RC: $LS_ERR" \
               "Unverified is not the same as fresh — refusing rather than" \
               "assuming. Fix connectivity, or pass --allow-stale-base to" \
               "proceed with the base explicitly unverified."
      fi
    fi
    rm -f "$LS_OUT"

    if [ -n "$REMOTE_SHA" ] && [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
      if [ "$ALLOW_STALE" -eq 1 ]; then
        STALE_NOTE="STALE base accepted via --allow-stale-base (local $LOCAL_SHA != remote $REMOTE_SHA)"
        echo "agent-worktree-bootstrap: warning — $STALE_NOTE" >&2
      else
        refuse "base ref $BASE is STALE." \
               "local:  $LOCAL_SHA" \
               "remote: $REMOTE_SHA" \
               "Work branched here applies to old sources; the conflict only" \
               "surfaces at merge time." \
               "Run: git -C $PROJECT fetch origin $REMOTE_BRANCH" \
               "or pass --allow-stale-base if the old base is deliberate."
      fi
    fi
    ;;
esac

# ── Check 3: the worktree location must not inherit a foreign dune root ──────
BRANCH="${BRANCH:-agent/${AGENT}-$(date -u +%Y%m%dT%H%M%SZ)}"
WT_PATH="$WT_ROOT/$(basename "$PROJECT")-$AGENT"

# dune's root is the OUTERMOST ancestor holding dune-workspace, else
# dune-project. Any such ancestor above the worktree silently captures the
# build. Check the ancestors of the intended location, not the worktree itself.
FOREIGN_ROOT=""
probe="$(dirname "$WT_PATH")"
while [ "$probe" != "/" ] && [ -n "$probe" ]; do
  if [ -e "$probe/dune-workspace" ] || [ -e "$probe/dune-project" ]; then
    FOREIGN_ROOT="$probe"
  fi
  probe="$(dirname "$probe")"
done
if [ -n "$FOREIGN_ROOT" ]; then
  refuse "the worktree location inherits a foreign dune root." \
         "intended worktree: $WT_PATH" \
         "dune would root at: $FOREIGN_ROOT" \
         "Builds inside the worktree would build THAT project instead." \
         "Choose a --root with no dune-project/dune-workspace above it."
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  echo "agent-worktree-bootstrap: CHECKS PASS (--check-only, nothing created)"
  echo "  project=$PROJECT base=$BASE branch=$BRANCH worktree=$WT_PATH"
  echo "  $STALE_NOTE"
  exit 0
fi

[ -e "$WT_PATH" ] && refuse "worktree path already exists: $WT_PATH" \
  "Another agent may be using it. Remove it with 'git worktree remove' or pass a different --agent."

mkdir -p "$WT_ROOT" || refuse "cannot create worktree root: $WT_ROOT"
git -C "$PROJECT" worktree add "$WT_PATH" -b "$BRANCH" "$BASE" >&2 || \
  refuse "git worktree add failed for $WT_PATH"

# ── Materialize the environment the agent actually needs ─────────────────────
# _opam: an opam switch is huge and machine-local; symlink rather than copy.
[ -d "$PROJECT/_opam" ] && ln -sfn "$PROJECT/_opam" "$WT_PATH/_opam"

# briefs/: git-ignored, therefore absent from a fresh worktree, yet it holds
# the pipeline ledger, control files and the xruntime journal. Symlink it so
# there is ONE briefs/ — a copy would fork the ledger and need reconciling.
if [ -d "$PROJECT/briefs" ]; then
  ln -sfn "$PROJECT/briefs" "$WT_PATH/briefs"
else
  mkdir -p "$PROJECT/briefs" && ln -sfn "$PROJECT/briefs" "$WT_PATH/briefs"
fi

# Scratchpad namespace: unique per agent AND per invocation, so a generic
# filename ("pr.md", "notes.md") written by two agents can no longer alias.
SCRATCH="$WT_PATH/.agent-scratch/${AGENT}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$SCRATCH"

# Pin the dune root explicitly. `dune build --root .` is the guarantee; the
# marker file makes an accidental `dune build` from a subdirectory root here
# rather than climbing out of the worktree.
[ -e "$WT_PATH/dune-workspace" ] || printf '(lang dune 3.15)\n' > "$WT_PATH/dune-workspace"

# The documented consumption pattern for this block is `eval`, so every value
# is shell-quoted on the way out. A project directory or --root containing a
# space, a quote, or `$(...)` would otherwise be executed as code by any caller
# following the contract — including this script's own test suite.
printf 'AGENT_WORKTREE=%q\n' "$WT_PATH"
printf 'AGENT_BRANCH=%q\n' "$BRANCH"
printf 'AGENT_BASE=%q\n' "$BASE"
printf 'AGENT_SCRATCH=%q\n' "$SCRATCH"
printf 'AGENT_DUNE_ROOT=%q\n' "$WT_PATH"
echo "agent-worktree-bootstrap: ready ($STALE_NOTE)" >&2
echo "  Build with: dune build --root $WT_PATH" >&2
echo "  Write scratch files under \$AGENT_SCRATCH — never a bare pr.md at repo root." >&2
