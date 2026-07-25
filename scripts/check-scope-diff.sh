#!/usr/bin/env bash
# check-scope-diff.sh — deterministic review scope gate (spec: specs/surgical-implementation.md)
#
# Usage: check-scope-diff.sh <manifest-path>
#
# Manifest grammar (pinned — see skills/pipeline/roster-implement.md §1.5):
#   base=<full sha>     one line: HEAD at implement-phase start
#   dirty=<path>        zero or more lines: one pre-task dirty file per line
#   ---                 literal separator
#   <entry>             one per line: exact repo-relative path, or directory prefix ending in /
#
# Changed set = (git diff --name-only <base>...HEAD) ∪ (git status --porcelain -uall),
# renames counting both paths, minus the dirty= files.
#
# Known blind spots (by design, spec FR-011): a task edit to a pre-task-dirty file is excluded;
# a mid-phase third-party file is attributed to the task and must be human-ACCEPTED.
#
# Exit contract (spec FR-041):
#   0 = no violations
#   1 = violations found — findings JSON array on stdout (review finding schema)
#   2 = degraded/unusable input (manifest missing or malformed, base sha unknown)
set -u

MANIFEST="${1:-}"
if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
  echo "check-scope-diff: manifest missing: ${MANIFEST:-<none>}" >&2
  exit 2
fi

BASE=""
DIRTY=()
ENTRIES=()
IN_BODY=0
while IFS= read -r line || [ -n "$line" ]; do
  if [ "$IN_BODY" -eq 0 ]; then
    case "$line" in
      base=*)
        if [ -n "$BASE" ]; then
          echo "check-scope-diff: duplicate base= line" >&2; exit 2
        fi
        BASE="${line#base=}"
        ;;
      dirty=*) DIRTY+=("${line#dirty=}") ;;
      ---)     IN_BODY=1 ;;
      "")      ;;
      *) echo "check-scope-diff: malformed header line: $line" >&2; exit 2 ;;
    esac
  else
    [ -n "$line" ] && ENTRIES+=("$line")
  fi
done < "$MANIFEST"

if [ "$IN_BODY" -ne 1 ] || [ -z "$BASE" ]; then
  echo "check-scope-diff: malformed manifest (missing base= or --- separator)" >&2
  exit 2
fi
# base must be a full 40-hex sha — symbolic refs (HEAD, branch names) would make the
# range degenerate (HEAD...HEAD is empty) and silently pass committed violations
case "$BASE" in
  *[!0-9a-f]*) echo "check-scope-diff: base= must be a full 40-hex sha, got: $BASE" >&2; exit 2 ;;
esac
if [ "${#BASE}" -ne 40 ]; then
  echo "check-scope-diff: base= must be a full 40-hex sha, got: $BASE" >&2
  exit 2
fi
if ! git rev-parse -q --verify "${BASE}^{commit}" >/dev/null 2>&1; then
  echo "check-scope-diff: base sha not found in repo: $BASE" >&2
  exit 2
fi

# --- changed set ---------------------------------------------------------
# Note: paths git C-quotes (control chars / non-ASCII under core.quotePath) keep their
# escaped form and will over-report as violations — fails safe, never silently allows.
declare -A CHANGED=()
declare -A EVIDENCE=()
if ! DIFF_OUT=$(git diff --name-status "${BASE}...HEAD"); then
  echo "check-scope-diff: git diff failed for range ${BASE}...HEAD" >&2
  exit 2
fi
if ! STATUS_OUT=$(git status --porcelain -uall); then
  echo "check-scope-diff: git status failed" >&2
  exit 2
fi
while IFS= read -r dline; do
  [ -z "$dline" ] && continue
  st="${dline%%$'\t'*}"
  rest="${dline#*$'\t'}"
  case "$st" in
    R*|C*)
      old="${rest%%$'\t'*}"
      new="${rest#*$'\t'}"
      CHANGED["$old"]=1; EVIDENCE["$old"]="git diff --name-status ${BASE:0:12}...HEAD: $dline"
      CHANGED["$new"]=1; EVIDENCE["$new"]="git diff --name-status ${BASE:0:12}...HEAD: $dline"
      ;;
    *)
      CHANGED["$rest"]=1; EVIDENCE["$rest"]="git diff --name-status ${BASE:0:12}...HEAD: $dline"
      ;;
  esac
done <<< "$DIFF_OUT"

strip_quotes() {
  local p="$1"
  p="${p#\"}"; p="${p%\"}"
  printf '%s' "$p"
}

while IFS= read -r pline; do
  [ -z "$pline" ] && continue
  status="${pline:0:2}"
  rest="${pline:3}"
  case "$status" in
    R*|C*)
      old="$(strip_quotes "${rest%% -> *}")"
      new="$(strip_quotes "${rest##* -> }")"
      CHANGED["$old"]=1; EVIDENCE["$old"]="git status --porcelain: $pline"
      CHANGED["$new"]=1; EVIDENCE["$new"]="git status --porcelain: $pline"
      ;;
    *)
      f="$(strip_quotes "$rest")"
      CHANGED["$f"]=1; EVIDENCE["$f"]="git status --porcelain: $pline"
      ;;
  esac
done <<< "$STATUS_OUT"

# minus pre-task dirty files
if [ "${#DIRTY[@]}" -gt 0 ]; then
  for d in "${DIRTY[@]}"; do
    [ -n "$d" ] && unset 'CHANGED[$d]' 2>/dev/null
  done
fi

# --- match against manifest entries --------------------------------------
VIOLATIONS=()
for f in "${!CHANGED[@]}"; do
  allowed=0
  if [ "${#ENTRIES[@]}" -gt 0 ]; then
    for e in "${ENTRIES[@]}"; do
      case "$e" in
        */)
          case "$f" in
            "$e"*) allowed=1; break ;;
          esac
          ;;
        *)
          if [ "$f" = "$e" ]; then allowed=1; break; fi
          ;;
      esac
    done
  fi
  [ "$allowed" -eq 0 ] && VIOLATIONS+=("$f")
done

[ "${#VIOLATIONS[@]}" -eq 0 ] && exit 0

{
  for f in "${VIOLATIONS[@]}"; do
    jq -n --arg p "$f" --arg ev "${EVIDENCE[$f]:-changed file not listed in the task manifest: $f}" '{
      severity: "HIGH",
      confidence: 5,
      path: $p,
      line: 0,
      category: "scope",
      summary: ("Out-of-manifest change: " + $p),
      evidence: $ev,
      fix: "Revert the file (git checkout <base> -- <path>) or have the human accept the finding / extend the manifest",
      fingerprint: ($p + ":0:scope"),
      specialist: "scope-gate",
      status: "OPEN"
    }'
  done
} | jq -s .
exit 1
