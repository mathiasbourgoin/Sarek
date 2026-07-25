#!/usr/bin/env bash
# xruntime-exec.sh — standardized cross-runtime (second-model) invocation wrapper.
# Usage: ./scripts/xruntime-exec.sh <codex|opencode> ["<prompt>" | --prompt-file=<path>] [--write] [--timeout <sec>] [--out <file>]
#
# Motivated by: friction cluster "cross-runtime invocation friction" (8+ occurrences —
# codex needs --skip-git-repo-check; hangs reading stdin unless </dev/null; shell wrappers
# swallow stdout unless file-redirected; manual tree-integrity snapshots around
# workspace-write runs; opencode capture timeouts). Health report 2026-07-10 P2.
# Added: 2026-07-10
#
# Behavior:
#   - positional prompts retain the legacy closed-stdin behavior
#   - --prompt-file=<path> streams the prompt with EOF, keeping large content out of argv
#   - codex: `codex exec --skip-git-repo-check`, sandbox read-only unless --write
#     (then --sandbox workspace-write); opencode: `opencode run`
#   - runtime output captured to --out file (default: mktemp), then echoed to stdout
#     (file-capture survives output-mangling shell wrappers)
#   - git tree-integrity snapshot (`git status --porcelain | sha256sum`) before/after;
#     divergence prints TREE-MUTATED on stderr and exits 3
#   - XRUNTIME_BIN overrides the runtime binary (testing hook)
#
# Exit codes: runtime's own exit code; 2 = usage error; 3 = tree mutated; 124 = timeout.
#
# Tested (2026-07-10): nominal via XRUNTIME_BIN=/bin/echo stub (exit 0, output captured &
# echoed); usage error on missing prompt (exit 2, stderr message); unknown runtime (exit 2);
# tree-mutation detection via stub touching a file (exit 3, TREE-MUTATED on stderr).
set -uo pipefail

RUNTIME="${1:-}"
shift 1 2>/dev/null || { echo "usage: xruntime-exec.sh <codex|opencode> [\"<prompt>\" | --prompt-file=<path>] [--write] [--timeout <sec>] [--out <file>]" >&2; exit 2; }

PROMPT=""
PROMPT_FILE=""
WRITE=0
TIMEOUT=480
OUT=""

# Backward compatibility: historically the first post-runtime argument was
# always the prompt, even when it began with `--`. Only the new explicit
# --prompt-file=<path> form displaces that positional slot without stealing
# the previously valid literal prompt `--prompt-file`.
if [ $# -gt 0 ]; then
  case "$1" in
    --prompt-file=*) PROMPT_FILE="${1#--prompt-file=}" ;;
    *) PROMPT="$1" ;;
  esac
  shift
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --write) WRITE=1 ;;
    --timeout) TIMEOUT="${2:-480}"; shift ;;
    --out) OUT="${2:-}"; shift ;;
    *) echo "xruntime-exec: unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

[ -n "$PROMPT" ] && [ -n "$PROMPT_FILE" ] && { echo "xruntime-exec: choose positional prompt or --prompt-file, not both" >&2; exit 2; }
[ -n "$PROMPT_FILE" ] && [ ! -r "$PROMPT_FILE" ] && { echo "xruntime-exec: prompt file not readable: $PROMPT_FILE" >&2; exit 2; }
[ -z "$PROMPT" ] && [ -z "$PROMPT_FILE" ] && { echo "xruntime-exec: empty prompt" >&2; exit 2; }
[ -n "$PROMPT_FILE" ] && [ ! -s "$PROMPT_FILE" ] && { echo "xruntime-exec: empty prompt file" >&2; exit 2; }
OWN_OUT=0
if [ -z "$OUT" ]; then
  OUT="$(mktemp)" || { echo "xruntime-exec: failed to create output file" >&2; exit 2; }
  OWN_OUT=1
fi
cleanup() {
  [ "$OWN_OUT" -eq 1 ] && rm -f -- "$OUT"
}
trap cleanup EXIT

INPUT="/dev/null"
[ -n "$PROMPT_FILE" ] && INPUT="$PROMPT_FILE"

case "$RUNTIME" in
  codex)
    CMD=("${XRUNTIME_BIN:-codex}" exec --skip-git-repo-check)
    if [ "$WRITE" -eq 1 ]; then CMD+=(--sandbox workspace-write); else CMD+=(--sandbox read-only); fi
    if [ -n "$PROMPT_FILE" ]; then CMD+=("-"); else CMD+=("$PROMPT"); fi
    ;;
  opencode)
    CMD=("${XRUNTIME_BIN:-opencode}" run)
    [ -n "$PROMPT" ] && CMD+=("$PROMPT")
    ;;
  *) echo "xruntime-exec: unknown runtime '$RUNTIME' (codex|opencode)" >&2; exit 2 ;;
esac

in_repo() { git rev-parse --is-inside-work-tree >/dev/null 2>&1; }
snapshot() { git status --porcelain -uall 2>/dev/null | sha256sum | cut -d' ' -f1; }

BEFORE=""
in_repo && BEFORE="$(snapshot)"

timeout "$TIMEOUT" "${CMD[@]}" < "$INPUT" > "$OUT" 2>&1
RC=$?

cat "$OUT"

if in_repo && [ -n "$BEFORE" ]; then
  AFTER="$(snapshot)"
  if [ "$BEFORE" != "$AFTER" ]; then
    echo "xruntime-exec: TREE-MUTATED — the $RUNTIME run changed the working tree (before=$BEFORE after=$AFTER). Inspect 'git status' before trusting results." >&2
    exit 3
  fi
fi

exit "$RC"
