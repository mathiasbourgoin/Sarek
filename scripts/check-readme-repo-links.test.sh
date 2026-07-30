#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Red-path test for scripts/check-readme-repo-links.sh.
#
# WHY A .test.sh AND NOT A prove-red-spec BLOCK. The subject derives the expected
# owner/repo from GITHUB_REPOSITORY or the git remote — never from a hardcoded
# name, since a second copy of that fact is the drift being prevented. Both
# sources are absent in prove-red.sh's bare `mktemp` scratch, so its mandatory
# green baseline could not be established there. This harness supplies each
# source explicitly, which also lets it check that they are BOTH honoured.
#
# Every case asserts an exit code, and each red case its message, because "went
# red" and "went red for the reason claimed" are different observations.

set -uo pipefail

SUBJECT="$(cd "$(dirname "$0")" && pwd)/check-readme-repo-links.sh"
[ -x "$SUBJECT" ] || {
  echo "FAIL: subject not executable: $SUBJECT" >&2
  exit 2
}

pass=0
fail=0

# $1 = case name, $2 = GITHUB_REPOSITORY ("" = unset, use the remote),
# $3 = expected exit, $4 = expected message substring, $5 = README body,
# $6 = remote URL (optional; defaults to the scp-like form)
check() {
  local name="$1" slug="$2" want="$3" msg="$4" body="$5" remote="${6:-git@github.com:owner/repo.git}" d out code
  d="$(mktemp -d)"
  (
    cd "$d" || exit 2
    git init --quiet .
    git remote add origin "$remote"
  ) >/dev/null 2>&1
  printf '%s' "$body" >"$d/README.md"
  if [ -n "$slug" ]; then
    out="$(cd "$d" && GITHUB_REPOSITORY="$slug" bash "$SUBJECT" 2>&1)"
  else
    out="$(cd "$d" && env -u GITHUB_REPOSITORY bash "$SUBJECT" 2>&1)"
  fi
  code=$?
  rm -rf "$d"
  if [ "$code" != "$want" ]; then
    echo "FAIL $name: exit $code, wanted $want"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  if [ -n "$msg" ] && ! printf '%s' "$out" | grep -qF "$msg"; then
    echo "FAIL $name: exit $want as wanted, but message lacked '$msg'"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  echo "PASS $name (exit $code)"
  pass=$((pass + 1))
}

GOOD='[![Build Status](https://github.com/owner/repo/actions/workflows/ci.yml/badge.svg)](https://github.com/owner/repo/actions)
'
BAD_BADGE='[![Build Status](https://github.com/owner/OTHER/actions/workflows/ci.yml/badge.svg)](https://github.com/owner/repo/actions)
'
BAD_LINK='See [CI](https://github.com/someone/else/actions) for status.
'

# --- positive control -------------------------------------------------------
# Green first, from each of the two derivation sources independently. Without
# this a subject that always exits 1 would satisfy every red case below.
check "green: badge matches GITHUB_REPOSITORY" "owner/repo" 0 "OK" "$GOOD"
check "green: badge matches the git remote (no GITHUB_REPOSITORY)" "" 0 "OK" "$GOOD"

# All four GitHub remote spellings must reduce to owner/repo. The ssh:// form
# was NOT normalized: the sed left "ssh://git@github.com/owner/repo", which
# contains a slash and so satisfied the old `*/*` test, after which no CI link
# could match — a correct README failing on a correct repo, on the local
# fallback path only (CodeRabbit, PR #387).
check "green: ssh:// remote" "" 0 "OK" "$GOOD" "ssh://git@github.com/owner/repo.git"
check "green: https:// remote" "" 0 "OK" "$GOOD" "https://github.com/owner/repo.git"
check "green: git:// remote" "" 0 "OK" "$GOOD" "git://github.com/owner/repo.git"

# --- the defect this gate exists for ---------------------------------------
# README.md:5 pointed at mathiasbourgoin/SPOC for the whole Sarek rework. A badge
# is an image: it renders, so a URL naming another repository looks fine while
# reporting a different project's CI status.
check "red: badge names another repository" "owner/repo" 1 "but this repository is" "$BAD_BADGE"
check "red: a plain Actions link names another repository" "owner/repo" 1 "but this repository is" "$BAD_LINK"

# GITHUB_REPOSITORY must WIN over the remote — otherwise the CI-side check is
# silently derived from the wrong source and this gate would pass in Actions on
# a repo it is not describing.
check "red: GITHUB_REPOSITORY disagrees with the README" "someone/else" 1 "but this repository is" "$GOOD"

# --- fails closed ----------------------------------------------------------
check "red: unparseable slug is exit 2, not a pass" "noslash" 2 "could not parse" "$GOOD"
# The slug must match owner/repo EXACTLY. "contains a slash" is what let the
# un-normalized ssh:// URI through as though it were a slug.
check "red: a slug with a space is not owner/repo" "own er/repo" 2 "could not parse" "$GOOD"
check "red: a three-segment slug is not owner/repo" "a/b/c" 2 "could not parse" "$GOOD"

# A missing README must refuse rather than report success on a file it never read
# — the same shape as the license gate's pre-backlog-137 `2>/dev/null` bug.
d="$(mktemp -d)"
(cd "$d" && git init --quiet . && git remote add origin git@github.com:owner/repo.git) >/dev/null 2>&1
out="$(cd "$d" && GITHUB_REPOSITORY=owner/repo bash "$SUBJECT" 2>&1)"
code=$?
rm -rf "$d"
if [ "$code" = 2 ]; then
  echo "PASS red: README absent (exit 2)"
  pass=$((pass + 1))
else
  echo "FAIL red: README absent: exit $code, wanted 2"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "check-readme-repo-links.test.sh: $pass passed, $fail FAILED"
  exit 1
fi
echo "check-readme-repo-links.test.sh: all $pass cases passed"
exit 0
