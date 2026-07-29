#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Covering test for scripts/check-license-headers.sh (#137).
#
# The gate spent its whole life exiting 1 on `main` and being invoked by
# nothing, so its green was never observed and its red was never trusted.
# Wiring it into CI without a red-path test would just move an unverified
# gate into a place where it blocks merges.
#
# Two distinct failure modes are asserted here:
#
#   * RED on a real problem  -- a covered file loses its header, and the gate
#     must go to exit 1 AND name the file.
#   * RED on a broken gate   -- a declared root disappears, or the candidate
#     set comes out empty. The pre-#137 finder ended in `2>/dev/null`, so this
#     shape exited 0 having inspected nothing. It must now be exit 2.
#
# Every case builds a synthetic project tree in a temp dir, copies the real
# scripts into its scripts/ directory (they resolve PROJECT_ROOT from their
# own location, so this redirects them entirely), and runs them there. No
# case touches the real working tree.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE_SRC="$SCRIPT_DIR/check-license-headers.sh"
FIXER_SRC="$SCRIPT_DIR/add-license-headers.sh"
for f in "$GATE_SRC" "$FIXER_SRC"; do
  [ -x "$f" ] || { echo "FAIL: $f not found or not executable"; exit 2; }
done

TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/license-headers-test.XXXXXX")"
trap 'rm -rf "$TMPROOT"' EXIT

pass=0
fail=0

# make_project <name> — a minimal tree with every declared root present and
# one covered file of each supported kind, all headers already applied by the
# real fixer. Returns the project root on stdout.
make_project() {
  local dir="$TMPROOT/$1"
  mkdir -p "$dir/scripts" "$dir/ci" \
           "$dir/sarek" "$dir/sarek-cuda" "$dir/sarek-opencl" \
           "$dir/sarek-vulkan" "$dir/sarek-metal" "$dir/spoc"

  cp "$GATE_SRC" "$FIXER_SRC" "$dir/scripts/"
  chmod +x "$dir/scripts/check-license-headers.sh" "$dir/scripts/add-license-headers.sh"

  printf 'let x = 1\n' > "$dir/sarek/covered.ml"
  printf 'val x : int\n' > "$dir/sarek/covered.mli"
  printf '#!/usr/bin/env bash\necho hi\n' > "$dir/ci/covered.sh"
  printf '#!/usr/bin/env python3\nprint(1)\n' > "$dir/scripts/covered.py"

  # The two review-bundle members, deliberately header-less. They are named
  # as exact-path exemptions, so the gate must stay green over them.
  printf '#!/usr/bin/env bash\necho scope\n' > "$dir/scripts/check-scope-diff.sh"
  printf '#!/usr/bin/env bash\necho xruntime\n' > "$dir/scripts/xruntime-exec.sh"

  ( cd "$dir" && git init -q . && git config user.email t@t.t && git config user.name t ) >/dev/null 2>&1
  # Apply headers with the real fixer so the "good" state is exactly what the
  # gate expects, rather than a hand-written approximation that could drift.
  ( cd "$dir" && ./scripts/add-license-headers.sh ) >/dev/null 2>&1
  echo "$dir"
}

# expect <desc> <project-dir> <wanted-exit> [substring-that-must-appear]
expect() {
  local desc="$1" dir="$2" want="$3" needle="${4:-}"
  local out got
  out="$(cd "$dir" && ./scripts/check-license-headers.sh 2>&1)"
  got=$?
  if [ "$got" != "$want" ]; then
    echo "  FAIL: $desc -- expected exit $want, got $got"
    echo "$out" | sed 's/^/        /'
    fail=$((fail + 1))
    return
  fi
  if [ -n "$needle" ] && ! printf '%s' "$out" | grep -qF -- "$needle"; then
    echo "  FAIL: $desc -- exit $got as expected, but output never mentioned '$needle'"
    echo "$out" | sed 's/^/        /'
    fail=$((fail + 1))
    return
  fi
  echo "  PASS: $desc (exit $got)"
  pass=$((pass + 1))
}

echo "check-license-headers.sh covering test"

# --- Case 0: positive control ------------------------------------------------
# Without this, "the gate went red" and "the gate is always red" are the same
# observation — which is precisely the state #137 found it in.
d="$(make_project case0)"
expect "case0: fully-headered tree is GREEN" "$d" 0 "up-to-date"

# --- Case 1: a shell script loses its header ---------------------------------
d="$(make_project case1)"
grep -v 'SPDX-' "$d/ci/covered.sh" > "$d/ci/covered.sh.tmp" && mv "$d/ci/covered.sh.tmp" "$d/ci/covered.sh"
expect "case1: stripped header on ci/covered.sh is RED and names the file" \
  "$d" 1 "ci/covered.sh"

# --- Case 2: an OCaml source loses its header --------------------------------
d="$(make_project case2)"
grep -v 'SPDX-\|^(\*\*' "$d/sarek/covered.ml" > "$d/sarek/covered.ml.tmp" && mv "$d/sarek/covered.ml.tmp" "$d/sarek/covered.ml"
expect "case2: stripped header on sarek/covered.ml is RED and names the file" \
  "$d" 1 "sarek/covered.ml"

# --- Case 3: Python coverage is live, not merely declared --------------------
# scripts/*.py carried headers long before anything checked them. If the
# extension to .py were declared and not wired, this case would read green.
d="$(make_project case3)"
grep -v 'SPDX-' "$d/scripts/covered.py" > "$d/scripts/covered.py.tmp" && mv "$d/scripts/covered.py.tmp" "$d/scripts/covered.py"
expect "case3: stripped header on scripts/covered.py is RED and names the file" \
  "$d" 1 "scripts/covered.py"

# --- Case 4: a declared root has vanished ------------------------------------
# The pre-#137 finder swallowed find's error and exited 0 here, reporting
# "all headers up-to-date" about a tree it had not read.
d="$(make_project case4)"
rm -rf "$d/ci"
expect "case4: missing declared root is exit 2, not a silent pass" \
  "$d" 2 "root(s) not found"

# --- Case 5: roots exist but match nothing -----------------------------------
d="$(make_project case5)"
rm -f "$d"/sarek/*.ml "$d"/sarek/*.mli
expect "case5: empty OCaml candidate set is exit 2, not a silent pass" \
  "$d" 2 "matched 0 files"

# --- Case 6: exempted paths really are exempt --------------------------------
# A header-less file under an EXEMPT_GLOBS path must NOT turn the gate red;
# otherwise the exemption list is decorative.
d="$(make_project case6)"
mkdir -p "$d/sarek/dependencies/vendored"
printf 'let vendored = 1\n' > "$d/sarek/dependencies/vendored/thirdparty.ml"
expect "case6: header-less file under an exempt path stays GREEN" "$d" 0 "up-to-date"

# --- Case 7: JavaScript is out of scope, deliberately ------------------------
# Documented in add-license-headers.sh's coverage block. Asserted here so the
# omission stays a decision someone can find and reverse, rather than folklore.
d="$(make_project case7)"
printf 'module.exports = 1;\n' > "$d/scripts/uncovered.js"
expect "case7: header-less .js is out of scope and stays GREEN" "$d" 0 "up-to-date"

# --- Case 8a: the review-bundle exemptions are live --------------------------
# scripts/check-scope-diff.sh and scripts/xruntime-exec.sh are upstream-owned,
# sha256-pinned bundle files that REVIEW-BUNDLE.md forbids hand-editing.
# make_project creates both without headers; case0 above already required the
# tree to be GREEN, so the exemption is asserted there. Here we check the
# fixer does not WRITE to them either — an exemption that only silences the
# report while the fixer still stamps the file is worse than none.
#
# The comparison is against the PRISTINE bytes, not against the file as it
# stands after make_project. make_project runs the fixer itself, so a
# before/after snapshot taken at this point would compare a
# possibly-already-stamped file with itself and pass no matter what — the
# first cut of this case did exactly that and stayed green with the
# exemptions deleted.
d="$(make_project case8a)"
ok=1
for f in check-scope-diff.sh xruntime-exec.sh; do
  if head -1 "$d/scripts/$f" | grep -q '^#!' && \
     ! grep -q 'SPDX-License-Identifier' "$d/scripts/$f"; then
    :
  else
    echo "  FAIL: case8a: scripts/$f was stamped despite being an exempt bundle member"
    ok=0
  fi
done
if [ "$ok" = 1 ]; then
  echo "  PASS: case8a: fixer leaves both review-bundle members unstamped"
  pass=$((pass + 1))
else
  fail=$((fail + 1))
fi

# --- Case 8b: a stale exemption is loud --------------------------------------
# An exact-path exemption for a file that has moved is an exemption nobody is
# reading — the file it was meant to protect is now being stamped under its
# new name, silently.
d="$(make_project case8b)"
rm -f "$d/scripts/xruntime-exec.sh"
expect "case8b: exemption pointing at a missing file is exit 2" \
  "$d" 2 "stale exemption"

# --- Case 9: the fixer must not change file modes ----------------------------
# It used to `mv` a mktemp file (0600) over the target, stripping the
# executable bit from every script it stamped -- so the act of satisfying the
# license gate broke every OTHER gate with "Permission denied". Asserted on
# the fixer directly, since the checker never writes.
d="$(make_project case9)"
printf '#!/usr/bin/env bash\necho later\n' > "$d/ci/newly-added.sh"
chmod 755 "$d/ci/newly-added.sh"
( cd "$d" && ./scripts/add-license-headers.sh ) >/dev/null 2>&1
mode="$(stat -c %a "$d/ci/newly-added.sh")"
if [ "$mode" = "755" ]; then
  echo "  PASS: case9: fixer preserves the executable bit (mode $mode)"
  pass=$((pass + 1))
else
  echo "  FAIL: case9: fixer changed mode 755 -> $mode on a file it stamped"
  fail=$((fail + 1))
fi

# --- Case 10: a duplicate copyright line is named, not a nameless sed crash ---
# Two SPDX-FileCopyrightText lines for the SAME email made the year-update grep
# return two lines, which went into `sed -i "s/$existing_years..."` as a
# multi-line value. sed died with "unterminated `s' command", the fixer exited 1
# partway through its walk having silently skipped every remaining file, and it
# never printed the offending path -- so the operator saw a red run with no
# location. Four files in this repo were in exactly that state.
#
# Both halves are asserted, because the exit status alone is satisfied by the
# old crash too: it must exit 2 (not 1, which is "files need updating") AND name
# the file AND say what is wrong with it.
d="$(make_project case10)"
dup_line='(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)'
# Insert a second, identical copyright line inside the header block.
awk -v line="$dup_line" '
  /SPDX-FileCopyrightText:/ && !done { print; print line; done=1; next }
  { print }
' "$d/sarek/covered.ml" > "$d/sarek/covered.ml.tmp" && mv "$d/sarek/covered.ml.tmp" "$d/sarek/covered.ml"

out="$(cd "$d" && ./scripts/add-license-headers.sh --check 2>&1)"
got=$?
if [ "$got" != 2 ]; then
  echo "  FAIL: case10: duplicate copyright line -- expected exit 2, got $got"
  echo "$out" | sed 's/^/        /'
  fail=$((fail + 1))
elif ! printf '%s' "$out" | grep -qF -- "sarek/covered.ml"; then
  echo "  FAIL: case10: exit 2 as expected, but the output never named sarek/covered.ml"
  echo "$out" | sed 's/^/        /'
  fail=$((fail + 1))
elif ! printf '%s' "$out" | grep -qF -- "DUPLICATE"; then
  echo "  FAIL: case10: exit 2 and named the file, but never said what was wrong"
  echo "$out" | sed 's/^/        /'
  fail=$((fail + 1))
elif printf '%s' "$out" | grep -qF -- "unterminated"; then
  echo "  FAIL: case10: sed still crashed rather than the duplicate being refused"
  echo "$out" | sed 's/^/        /'
  fail=$((fail + 1))
else
  echo "  PASS: case10: duplicate copyright line is refused by path (exit 2)"
  pass=$((pass + 1))
fi

echo ""
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
