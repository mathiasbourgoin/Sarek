#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Covering test for scripts/check-dune-dir-visibility.sh (#147).
#
# The guard's whole value is its red path. A visibility guard that cannot fail
# is indistinguishable from no guard at all -- and this repository's #147 was
# itself an instance of "green because nothing was looked at", so accepting the
# guard's own green on faith would be repeating the bug in the fix.
#
# Each case builds a synthetic git repo in a temp dir, runs the real guard
# against it, and asserts on the exit code. Cases 1-4 must be RED; case 5 must
# be GREEN; case 6 must exit 2 (parser refuses to guess).
set -uo pipefail

GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check-dune-dir-visibility.sh"
[ -x "$GUARD" ] || { echo "FAIL: $GUARD not found or not executable"; exit 2; }

TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/dune-visibility-test.XXXXXX")"
trap 'rm -rf "$TMPROOT"' EXIT

pass=0
fail=0

# make_repo <name> <builder-fn>
make_repo() {
  local name="$1"
  local dir="$TMPROOT/$name"
  mkdir -p "$dir"
  (
    cd "$dir" || exit 1
    git init -q .
    git config user.email t@t.t
    git config user.name t
  ) || return 1
  echo "$dir"
}

commit_all() {
  ( cd "$1" && git add -A && git -c commit.gpgsign=false commit -qm t ) >/dev/null 2>&1
}

expect() {
  local desc="$1" dir="$2" want="$3"
  local out got
  out="$(cd "$dir" && "$GUARD" 2>&1)"
  got=$?
  if [ "$got" = "$want" ]; then
    echo "  PASS: $desc (exit $got)"
    pass=$((pass + 1))
  else
    echo "  FAIL: $desc -- expected exit $want, got $got"
    echo "$out" | sed 's/^/        /'
    fail=$((fail + 1))
  fi
}

echo "check-dune-dir-visibility.sh covering test"

# --- Case 1: the exact #147 shape -- (dirs X) hiding a sibling test dir -------
d="$(make_repo case1)"
mkdir -p "$d/lib/test"
printf '(dirs ir_extract)\n\n(library (name foo))\n' > "$d/lib/dune"
printf '(test (name test_foo) (libraries foo))\n' > "$d/lib/test/dune"
commit_all "$d"
expect "case1: (dirs ir_extract) hides lib/test" "$d" 1

# --- Case 2: set-difference form -- (dirs :standard \ test) -------------------
d="$(make_repo case2)"
mkdir -p "$d/lib/test"
printf '(dirs :standard \\ test)\n\n(library (name foo))\n' > "$d/lib/dune"
printf '(test (name test_foo))\n' > "$d/lib/test/dune"
commit_all "$d"
expect "case2: (dirs :standard \\ test) hides lib/test" "$d" 1

# --- Case 3: data_only_dirs also makes a build dir invisible ------------------
d="$(make_repo case3)"
mkdir -p "$d/lib/test"
printf '(data_only_dirs test)\n\n(library (name foo))\n' > "$d/lib/dune"
printf '(test (name test_foo))\n' > "$d/lib/test/dune"
commit_all "$d"
expect "case3: (data_only_dirs test) hides lib/test" "$d" 1

# --- Case 4: exclusion two levels up, not just the immediate parent -----------
d="$(make_repo case4)"
mkdir -p "$d/a/b/test"
printf '(dirs other)\n' > "$d/a/dune"
printf '(library (name foo))\n' > "$d/a/b/dune"
printf '(test (name test_foo))\n' > "$d/a/b/test/dune"
commit_all "$d"
expect "case4: grandparent (dirs other) hides a/b and a/b/test" "$d" 1

# --- Case 5: no exclusion -- must be GREEN (guard is not a blanket failer) ----
d="$(make_repo case5)"
mkdir -p "$d/lib/test"
printf '(library (name foo))\n' > "$d/lib/dune"
printf '(test (name test_foo))\n' > "$d/lib/test/dune"
commit_all "$d"
expect "case5: unrestricted tree is green" "$d" 0

# --- Case 5b: a scope stanza that ADMITS the dir must stay green --------------
d="$(make_repo case5b)"
mkdir -p "$d/lib/test" "$d/lib/vendor"
printf '(dirs :standard \\ vendor)\n\n(library (name foo))\n' > "$d/lib/dune"
printf '(test (name test_foo))\n' > "$d/lib/test/dune"
printf 'not a dune file\n' > "$d/lib/vendor/README"
commit_all "$d"
expect "case5b: exclusion that misses the test dir is green" "$d" 0

# --- Case 6: unparseable predicate must exit 2, not silently pass -------------
d="$(make_repo case6)"
mkdir -p "$d/lib/test"
printf '(dirs (re_matches "^t.*"))\n\n(library (name foo))\n' > "$d/lib/dune"
printf '(test (name test_foo))\n' > "$d/lib/test/dune"
commit_all "$d"
expect "case6: unsupported predicate exits 2 rather than guessing" "$d" 2

# --- Case 7: comments must not be parsed as stanzas ---------------------------
d="$(make_repo case7)"
mkdir -p "$d/lib/test"
printf '; (dirs ir_extract) -- historical note, not active\n\n(library (name foo))\n' > "$d/lib/dune"
printf '(test (name test_foo))\n' > "$d/lib/test/dune"
commit_all "$d"
expect "case7: commented-out (dirs) does not count as an exclusion" "$d" 0

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
echo "OK: check-dune-dir-visibility.sh fails on all four invisibility shapes and passes clean trees"
