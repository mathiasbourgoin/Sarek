#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Covering test for scripts/check-kb-properties.sh (backlog-100).
#
# The checker's entire subject is "a gate that cannot fail is worse than no
# gate". Accepting its own green on faith would be committing the bug inside
# the fix, so every check type it implements is mutated here and asserted to
# go red WITH ITS OWN MESSAGE -- not merely non-zero. An assertion that accepts
# any non-zero exit where the contract names a specific code is a weakened
# assertion, and this repository has recurred on exactly that inside a fix.
#
# Two kinds of mutation, deliberately both:
#   - doctoring the DECLARATION (a red_path that vanished, a bad type, ...)
#   - doctoring the TARGET (deleting the literal a grep-present names, adding
#     the one grep-absent forbids, unwiring a gate from the carrier)
# The second kind is why the checker takes a `root` argument: without it only
# the declaration half would ever be exercised, and the half that guards the
# source tree would be untested.
#
# Case 0 is the positive control. Without it, "went red" and "is always red"
# are the same observation.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$HERE/check-kb-properties.sh"
[ -x "$CHECKER" ] || { echo "FAIL: $CHECKER not found or not executable"; exit 2; }

TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/kb-properties-test.XXXXXX")"
trap 'rm -rf "$TMPROOT"' EXIT

pass=0
fail=0

# ---------------------------------------------------------------------------
# A synthetic tree that mirrors the real one's shape: two carriers, a bundle
# manifest, a gate with a red-path test, a tool, and a source file carrying one
# literal that must be present and not carrying one that must be absent.
# ---------------------------------------------------------------------------
make_tree() {
  local d="$TMPROOT/$1"
  rm -rf "$d"
  mkdir -p "$d/scripts" "$d/kb" "$d/src" "$d/.github/workflows"

  cat >"$d/.github/workflows/ci.yml" <<'EOF'
jobs:
  fast:
    steps:
      # a comment mentioning scripts/not-really-run.sh must not count
      - run: ./scripts/real-gate.sh
      - run: ./scripts/real-gate.test.sh
EOF

  cat >"$d/Makefile" <<'EOF'
harness:
	node scripts/bundled-tool-test.js
EOF

  cat >"$d/scripts/review-bundle.manifest.json" <<'EOF'
{"files": [{"path": "scripts/bundled.js", "sha256": "x"}]}
EOF

  : >"$d/scripts/real-gate.sh"
  : >"$d/scripts/real-gate.test.sh"
  : >"$d/scripts/manual-tool.sh"
  : >"$d/scripts/manual-tool.test.sh"
  : >"$d/scripts/bundled.js"
  : >"$d/scripts/bundled-tool-test.js"
  : >"$d/scripts/a-plain-tool.sh"

  printf 'let f = function\n  | Unknown _ -> false\n  | TFloat64 -> "double"\n' \
    >"$d/src/target.ml"

  cat >"$d/kb/properties.md" <<'EOF'
# Properties

```code-intel
{"id": "INV-1", "type": "gate-inventory-complete", "description": "d", "check": {"carriers": [".github/workflows/ci.yml", "Makefile"], "exempt_manifest": "scripts/review-bundle.manifest.json", "exempt": []}}
{"id": "INV-2", "type": "gate-red-path", "description": "d", "check": {"tool": "scripts/real-gate.sh", "red_path": "scripts/real-gate.test.sh"}}
{"id": "INV-3", "type": "gate-red-path", "description": "d", "check": {"tool": "scripts/bundled.js", "red_path": "scripts/bundled-tool-test.js", "invocation": "manual", "reason": "phase-driven bundle member"}}
{"id": "INV-4", "type": "grep-present", "description": "Unknown must not permit", "check": {"file": "src/target.ml", "literal": "| Unknown _ -> false"}}
{"id": "INV-5", "type": "grep-absent", "description": "no silent f64 narrowing", "check": {"paths": ["src"], "suffixes": [".ml"], "literal": "| TFloat64 -> \"float\""}}
```
EOF
  echo "$d"
}

# expect <desc> <root> <want-exit> <want-substring>
expect() {
  local desc="$1" root="$2" want="$3" needle="$4"
  local out got
  out="$("$CHECKER" kb/properties.md "$root" 2>&1)"
  got=$?
  if [ "$got" != "$want" ]; then
    echo "  FAIL: $desc -- expected exit $want, got $got"
    echo "$out" | sed 's/^/        /'
    fail=$((fail + 1))
    return
  fi
  if [ -n "$needle" ] && ! printf '%s' "$out" | grep -qF -- "$needle"; then
    echo "  FAIL: $desc -- exit $want as expected, but the message did not"
    echo "        mention: $needle"
    echo "$out" | sed 's/^/        /'
    fail=$((fail + 1))
    return
  fi
  echo "  PASS: $desc (exit $got)"
  pass=$((pass + 1))
}

echo "check-kb-properties.sh covering test"

# --- Case 0: POSITIVE CONTROL -----------------------------------------------
# Everything holds. If this is not green, every red below proves nothing --
# a checker that always fails goes red on every mutation for free.
root="$(make_tree case0)"
expect "case 0 (positive control): an intact tree is GREEN" "$root" 0 "OK: 5"

# --- Case 1: a declared gate loses its red-path test -------------------------
root="$(make_tree case1)"
rm "$root/scripts/real-gate.test.sh"
sed -i 's#^      - run: ./scripts/real-gate.test.sh$##' "$root/.github/workflows/ci.yml"
expect "case 1: red-path test deleted" "$root" 1 "does not exist"

# --- Case 2: the red-path test exists but nothing runs it --------------------
# This is the shape that matters most: a covering test sitting in the tree,
# unwired, is exactly as much proof as no covering test at all.
root="$(make_tree case2)"
sed -i 's#^      - run: ./scripts/real-gate.test.sh$##' "$root/.github/workflows/ci.yml"
expect "case 2: red-path test present but unwired" "$root" 1 \
  "so nothing ever proves"

# --- Case 3: the gate itself is unwired --------------------------------------
root="$(make_tree case3)"
sed -i 's#^      - run: ./scripts/real-gate.sh$##' "$root/.github/workflows/ci.yml"
expect "case 3: gate present but invoked by no carrier" "$root" 1 \
  "indistinguishable from a gate that passes"

# --- Case 4: a new gate reaches CI with no row in properties.md --------------
root="$(make_tree case4)"
printf '      - run: ./scripts/a-plain-tool.sh\n' >>"$root/.github/workflows/ci.yml"
expect "case 4: undeclared script wired into a carrier" "$root" 1 \
  "scripts/a-plain-tool.sh"

# --- Case 5: an exemption that no longer names anything ----------------------
root="$(make_tree case5)"
sed -i 's#"exempt": \[\]#"exempt": ["scripts/gone.sh"]#' "$root/kb/properties.md"
expect "case 5: stale exemption" "$root" 1 "stale entry"

# --- Case 6: grep-present, with the literal removed from the TARGET ----------
root="$(make_tree case6)"
sed -i 's/  | Unknown _ -> false/  | Unknown _ -> true/' "$root/src/target.ml"
expect "case 6: grep-present literal deleted from the source" "$root" 1 \
  "expected at least 1"

# --- Case 7: grep-absent, with the forbidden literal introduced --------------
root="$(make_tree case7)"
sed -i 's/  | TFloat64 -> "double"/  | TFloat64 -> "float"/' "$root/src/target.ml"
expect "case 7: grep-absent literal introduced into the source" "$root" 1 \
  "no silent f64 narrowing"

# --- Case 8: grep-absent whose scan root vanished ----------------------------
# The defect this repository actually shipped: a finder that reads nothing and
# reports success. An absence check over zero files is true and worthless.
root="$(make_tree case8)"
rm -rf "$root/src"
expect "case 8: absence check whose scan root is gone" "$root" 1 \
  "would otherwise have passed having read nothing"

# --- Case 9: `invocation: manual` with no reason -----------------------------
root="$(make_tree case9)"
sed -i 's#, "reason": "phase-driven bundle member"##' "$root/kb/properties.md"
expect "case 9: manual invocation with no stated reason" "$root" 1 \
  "it must be argued"

# --- Case 10: `invocation: manual` on a gate that IS wired -------------------
# The excuse outliving the weakness. Left in place it silently permits the gate
# to be unwired again later.
root="$(make_tree case10)"
printf '      - run: ./scripts/bundled.js\n' >>"$root/.github/workflows/ci.yml"
expect "case 10: manual declaration on a now-wired gate" "$root" 1 \
  "excuses a weakness that no longer exists"

# --- Case 11: MALFORMED, exit 2 not 1 ---------------------------------------
# A typo'd type must be loud. If an unknown type were skipped, the cheapest way
# to silence any declaration here would be to misspell it.
root="$(make_tree case11)"
sed -i 's/"type": "grep-present"/"type": "grep-presnet"/' "$root/kb/properties.md"
expect "case 11: unknown check type is exit 2, not a skip" "$root" 2 \
  "unknown check type"

# --- Case 12: the block is empty --------------------------------------------
root="$(make_tree case12)"
printf '# Properties\n\n```code-intel\n```\n' >"$root/kb/properties.md"
expect "case 12: empty code-intel block is exit 2" "$root" 2 \
  "a gate that cannot fail"

# --- Case 13: no block at all ------------------------------------------------
root="$(make_tree case13)"
printf '# Properties\n\nAll our invariants, in prose.\n' >"$root/kb/properties.md"
expect "case 13: properties.md with no block is exit 2" "$root" 2 \
  'carries no ```code-intel block'

# --- Case 14: two blocks -----------------------------------------------------
root="$(make_tree case14)"
printf '# P\n\n```code-intel\n{"id":"A","type":"grep-present","description":"d","check":{"file":"src/target.ml","literal":"Unknown"}}\n```\n\n```code-intel\n{"id":"B","type":"grep-present","description":"d","check":{"file":"src/target.ml","literal":"Unknown"}}\n```\n' >"$root/kb/properties.md"
expect "case 14: two code-intel blocks is exit 2" "$root" 2 \
  "Exactly one is allowed"

# --- Case 15: duplicate ids --------------------------------------------------
root="$(make_tree case15)"
sed -i 's/"id": "INV-5"/"id": "INV-4"/' "$root/kb/properties.md"
expect "case 15: duplicate declaration id is exit 2" "$root" 2 "duplicate id"

# --- Case 16: a declared carrier that does not exist -------------------------
root="$(make_tree case16)"
rm "$root/Makefile"
expect "case 16: missing carrier is exit 2" "$root" 2 \
  "wrong in the permissive direction"

# --- Case 17: properties.md itself is gone -----------------------------------
root="$(make_tree case17)"
rm "$root/kb/properties.md"
expect "case 17: missing properties.md is exit 2, never a skip" "$root" 2 \
  "a pass would mean nothing"

# --- Case 18: a line missing a required envelope field -----------------------
root="$(make_tree case18)"
sed -i 's/{"id": "INV-4", "type": "grep-present", "description": "Unknown must not permit", /{"id": "INV-4", "type": "grep-present", /' \
  "$root/kb/properties.md"
expect "case 18: declaration missing a description is exit 2" "$root" 2 \
  "missing or empty string field"

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
