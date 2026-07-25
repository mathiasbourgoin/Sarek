#!/usr/bin/env bash
# check-review-bundle-tracked.sh — the review-tool bundle must survive a fresh clone.
#
# Failure mode this guard exists for (backlog #108): every deterministic quality
# gate this project relies on (review normalization, convergence, scope-diff,
# cross-runtime probing, the finding/trace schemas and their fixtures) lived
# ONLY as untracked files on one workstation, hidden behind `.git/info/exclude`
# — a machine-local, unshared, unreviewable ignore file. `git status` was clean,
# `.gitignore` said nothing, and a fresh clone got no gates at all.
#
# `scripts/review-bundle-verify.js` checks that the installed bundle is complete
# and unmodified. It cannot see the tracking problem: an untracked bundle
# verifies perfectly. This guard checks the orthogonal property — that git will
# actually hand the bundle to the next clone.
#
# Checks, in order:
#   1. the manifest exists and is itself tracked
#   2. every manifest-listed path is tracked
#   3. no manifest-listed path is matched by ANY ignore source
#      (.gitignore, .git/info/exclude, core.excludesFile)
#   4. content integrity, delegated to review-bundle-verify.js
#
# Exit: 0 all good; 1 a bundle file is untracked or ignored, or content drifted;
#       2 usage/environment error.
set -uo pipefail

ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$ROOT" ] || { echo "check-review-bundle-tracked: not inside a git repository" >&2; exit 2; }
cd "$ROOT" || exit 2

MANIFEST="scripts/review-bundle.manifest.json"
[ -f "$MANIFEST" ] || {
  echo "check-review-bundle-tracked: FAIL — $MANIFEST is missing." >&2
  echo "  The review-tool bundle is not installed. Install it before relying on any review gate." >&2
  exit 1
}

command -v node >/dev/null 2>&1 || { echo "check-review-bundle-tracked: node is required" >&2; exit 2; }

# The manifest is the sentinel and is not listed inside its own `files` array,
# so it is checked explicitly alongside the paths it names.
PATHS=$(node -e '
const m = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
if (!Array.isArray(m.files)) { process.exit(3); }
for (const f of m.files) { if (f && typeof f.path === "string") console.log(f.path); }
' "$MANIFEST") || { echo "check-review-bundle-tracked: $MANIFEST is unreadable or has no files[]" >&2; exit 2; }

FAILED=0
report() { echo "check-review-bundle-tracked: $1" >&2; FAILED=1; }

while IFS= read -r rel; do
  [ -n "$rel" ] || continue

  if ! git ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
    report "UNTRACKED $rel — present on this workstation only; a fresh clone will not have it."
  fi

  # `git check-ignore` reports the matching rule and its source file, which is
  # what makes a re-introduced .git/info/exclude entry diagnosable rather than
  # just "mysteriously missing".
  if RULE=$(git check-ignore -v --no-index -- "$rel" 2>/dev/null); then
    report "IGNORED $rel — matched by: $RULE"
  fi
done <<EOF
$MANIFEST
$PATHS
EOF

if ! node scripts/review-bundle-verify.js; then
  report "content verification failed (see review-bundle-verify output above)."
fi

if [ "$FAILED" -ne 0 ]; then
  echo "" >&2
  echo "check-review-bundle-tracked: FAIL." >&2
  echo "  Track the listed files (git add), and remove any bundle entry from" >&2
  echo "  .gitignore / .git/info/exclude. Local-only roster infrastructure" >&2
  echo "  (briefs/, skills-meta/, kb/, .harness/, .agents/) stays untracked —" >&2
  echo "  the bundle does not." >&2
  exit 1
fi

echo "check-review-bundle-tracked: OK — bundle tracked, un-ignored, and sha-matched."
