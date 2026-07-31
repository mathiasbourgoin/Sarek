#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Red-path test for scripts/check-cited-paths-exist.sh.
#
# WHY A .test.sh AND NOT A prove-red-spec BLOCK. prove-red.sh builds its scratch
# tree with `tempfile.mkdtemp` — a bare directory with no `.git`. The subject
# resolves citations against `git ls-files` on purpose (tracked, not merely
# present, is the whole point: `roster/` and `briefs/` exist on a workstation and
# in no clone), so in that scratch it would exit 2 and the mandatory green
# baseline could never be established. This harness therefore builds its own
# throwaway git repository per case, which is the only way to exercise the
# tracked-vs-present distinction at all.
#
# Each case asserts an exit code AND a message, because "went red" and "went red
# for the reason claimed" are different observations.

set -uo pipefail

# CITED_PATHS_SUBJECT exists for ONE purpose: the meta-control at the bottom of
# this file, which points the harness at a stub that always exits 0 and at one
# that always exits 1, and asserts that this harness FAILS on both. Without it,
# "all cases passed" is unfalsifiable -- it is a claim about the subject that a
# harness checking nothing would also print.
SUBJECT="${CITED_PATHS_SUBJECT:-$(cd "$(dirname "$0")" && pwd)/check-cited-paths-exist.sh}"
[ -x "$SUBJECT" ] || {
  echo "FAIL: subject not executable: $SUBJECT" >&2
  exit 2
}

pass=0
fail=0

# Build a minimal tracked repo: one .ml file whose comment/body we vary.
# $1 = file body, $2 = optional docs/Doc.md body. Echoes the repo path.
#
# docs/Index.md is unconditional: the subject exits 2 when a tree has no tracked
# markdown at all (a repository whose prose it cannot see is one it cannot
# report coverage of), so every fixture needs at least one .md file. It cites
# nothing, so it never changes a verdict.
mkfixture() {
  local body="$1" md="${2:-}" d
  d="$(mktemp -d)"
  (
    cd "$d" || exit 2
    git init --quiet .
    git config user.email t@example.invalid
    git config user.name t
    mkdir -p sub scripts docs
    printf '%s' "$body" >sub/Thing.ml
    printf 'let real = 1\n' >sub/Existing.ml
    printf 'let a = 1\nlet b = 2\nlet c = 3\nlet d = 4\nlet e = 5\n' >sub/Multi.ml
    printf '# Index\n\nNothing cited here.\n' >docs/Index.md
    printf '# Real\n\nA real document.\n' >docs/Real.md
    [ -n "$md" ] && printf '%s' "$md" >docs/Doc.md
    git add -A
    git commit --quiet -m fixture
  ) >/dev/null 2>&1
  printf '%s' "$d"
}

# $1 = case name, $2 = expected exit, $3 = expected message substring
# ("" = no message requirement), $4 = .ml body, $5 = optional docs/Doc.md body,
# $6 = optional scripts/cited-lines-exempt.tsv body (tier-1 exemption channel;
# deliberately NOT populated from this repo's own file, so a fixture's
# citations are judged only against what the case itself supplies)
check() {
  local name="$1" want="$2" msg="$3" body="$4" md="${5:-}" lex="${6:-}" d out code
  d="$(mkfixture "$body" "$md")"
  cp "$(dirname "$0")/cited-paths-exempt.tsv" "$d/scripts/" 2>/dev/null || true
  [ -n "$lex" ] && printf '%s' "$lex" >"$d/scripts/cited-lines-exempt.tsv"
  out="$(cd "$d" && bash "$SUBJECT" 2>&1)"
  code=$?
  rm -rf "$d"
  if [ "$code" != "$want" ]; then
    echo "FAIL $name: exit $code, wanted $want"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  if [ -n "$msg" ] && ! printf '%s' "$out" | grep -qF "$msg"; then
    echo "FAIL $name: exit $want as wanted, but message did not contain '$msg'"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  echo "PASS $name (exit $code)"
  pass=$((pass + 1))
}

# --- positive control -------------------------------------------------------
# A green baseline first. Without it, every red below is unfalsifiable: a subject
# that always exits 1 would satisfy the whole rest of this file.
check "green: a citation that resolves" 0 "OK" \
  '(* See sub/Existing.ml for the real thing. *)
let x = 1
'

# --- the defect this gate exists for ---------------------------------------
check "red: citation into an unpublished directory" 1 "is not a tracked file" \
  '(* See roster/gone/L99-note.md for the design. *)
let x = 1
'

# ocamlformat breaks long comments mid-path, and one of the four roster/
# citations that motivated the gate was written exactly this way. A
# line-oriented scan sees no `.md` on either line and reports nothing.
check "red: the same citation WRAPPED across comment lines" 1 "is not a tracked file" \
  '(* See roster/gone/L99-
 * note.md for the design. *)
let x = 1
'

# --- the boundary CodeRabbit raised on PR #387 ------------------------------
# The identical path, in a string literal instead of a comment. Rows 2 and this
# one are the discriminating pair: same bytes, different syntactic position,
# opposite verdicts. If the comment scanner had simply blanked everything, row 2
# would be green and this gate would be inert.
check "green: the same path inside a STRING LITERAL is not a citation" 0 "OK" \
  'let fixture = "roster/gone/L99-note.md"
let x = 1
'

# A path inside a comment is comment text even when quoted — odoc code refs
# legitimately read ["some/path.ml"], and those must still be checked.
check "red: a path quoted INSIDE a comment is still a citation" 1 "is not a tracked file" \
  '(* See ["roster/gone/L99-note.md"] for the design. *)
let x = 1
'

# --- OCaml quoted-string literals (CodeRabbit, PR #387) ---------------------
# 46 files here hold raw PTX/CUDA text in {| ... |}. That text contains "(*"
# sequences and real-looking paths, so an unhandled {| opens a phantom comment
# and swallows the code after it into the scan.
check "green: a path inside a {| |} quoted string is not a citation" 0 "OK" \
  'let ptx = {|.version 8.0
// roster/gone/L99-note.md
|}
let x = 1
'

# The delimited form, and the case that actually bites: a "(*" inside the
# literal must not open a comment that swallows what follows.
check "green: {id| |id} holding an unbalanced (* does not open a comment" 0 "OK" \
  'let src = {ptx|(* this is data, not a comment
roster/gone/L99-note.md
|ptx}
let x = 1
'

# The positive control for the pair above: with the literal closed, a REAL
# citation after it is still found. Without this, "handled {|" would be
# indistinguishable from "stopped scanning at the first {|".
check "red: a real citation AFTER a quoted string is still found" 1 "is not a tracked file" \
  'let ptx = {|.version 8.0
|}
(* See roster/gone/L99-note.md for the design. *)
let x = 1
'

# --- documentation placeholders are not citations --------------------------
check "green: path/to/ placeholder" 0 "OK" \
  '(* Usage: put it at path/to/Thing.ml and go. *)
let x = 1
'

# --- the char-literal hole (found by adversarial review) --------------------
# `Buffer.add_char buf '"'` flipped in_string on and never off, blanking every
# comment for the REST OF THE FILE. Two tracked files already contain that exact
# token, so the gate was measurably blind over their tails while reporting full
# coverage. The citation below sits after one.
check "red: a citation after a '\"' char literal is still found" 1 "is not a tracked file" \
  "let c = '\"'
(* See roster/gone/note.md for the design. *)
let x = 1
"

# --- the hyphen-unwrap must not FABRICATE a citation from prose -------------
# `re.sub(r"-\n\s*\*?\s*", "-", …)` joined ANY comment line ending in "-", so
# "The lookup is device-\n * specific/…" invented `device-specific/notes.md`,
# a path nobody wrote, reported at line "?" because it matches no single line.
# The unwrap now only joins when the trailing token already contains a "/".
check "red: prose join no longer invents device-specific/..." 1 "specific/notes.md" \
  '(* The lookup is device-
 * specific/notes.md aware. *)
let x = 1
'

# --- exemption channel ------------------------------------------------------
# Without one the gate hard-fails a correct, useful, unfixable comment. Every
# sibling gate has an escape hatch; this one had none, so its own failure text
# told users to "say the document is not part of this repository" -- a marker it
# did not honour.
check "green: an exempted path is not a finding" 0 "OK" \
  '(* Mirrors OCaml stdlib typing/typecore.ml behaviour. *)
let x = 1
'

# ===========================================================================
# TIER 1 (backlog-226). CITATION matched only the path -- "foo.ml:147" and
# "foo.ml:150" were the same match to it, so a citation pointing at the wrong
# line, or past the end of the file, passed at exit 0. sub/Multi.ml is a fixed
# 5-line fixture so a citation's line number can be pinned against a known
# length.
# ===========================================================================

# Positive control: a citation whose line number is genuinely in range.
check "green: a cited line number within the file's length" 0 "OK" \
  '(* See sub/Multi.ml:3 for the real thing. *)
let x = 1
'

# The defect itself: the PATH resolves (unlike the roster/ cases above), but
# the cited line does not exist. Before tier 1, this was indistinguishable
# from the case above -- both exit 0.
check "red: a cited line number past the end of the file" 1 \
  "sub/Multi.ml has only 5 line(s)" \
  '(* See sub/Multi.ml:9 for the real thing. *)
let x = 1
'

# The other half of the pair: correcting the line number is what flips this
# back to green -- proves the check is reading the NUMBER, not just noticing
# "some citation of Multi.ml exists".
check "green: the same citation with the line number corrected" 0 "OK" \
  '(* See sub/Multi.ml:5 for the real thing. *)
let x = 1
'

# A RANGE citation (":NNN-MM") is checked on its END, not just its start --
# 565 of the 729 live citations counted in the backlog-226 corpus sweep were
# ranges, so this is the dominant shape, not an edge case.
check "red: a range citation whose END exceeds the file's length" 1 \
  "sub/Multi.ml has only 5 line(s)" \
  '(* See sub/Multi.ml:3-9 for the real thing. *)
let x = 1
'

check "green: a range citation that fits entirely inside the file" 0 "OK" \
  '(* See sub/Multi.ml:3-5 for the real thing. *)
let x = 1
'

# The tier-1 exemption channel (cited-lines-exempt.tsv), scoped to the EXACT
# citation. Without this a dated audit citing a line from a stated historical
# baseline (the actual shape found under kb/research/obj-usage/ in the
# backlog-226 sweep) would have no remedy but to falsify its own record.
check "green: an exact tier-1 exemption is honoured" 0 "OK" \
  '(* See sub/Multi.ml:9 for the real thing. *)
let x = 1
' "" "sub/Thing.ml::sub/Multi.ml:9	dated audit, frozen to a stated historical baseline
"

# ...and the same exemption does not blanket-cover a DIFFERENT wrong line
# number for the same path -- otherwise one row would silence every future
# defect against that path, which is the one thing tier 1 exists to catch.
check "red: a tier-1 exemption does not cover a DIFFERENT wrong line" 1 \
  "sub/Multi.ml has only 5 line(s)" \
  '(* See sub/Multi.ml:11 for the real thing. *)
let x = 1
' "" "sub/Thing.ml::sub/Multi.ml:9	dated audit, frozen to a stated historical baseline
"

# ===========================================================================
# MARKDOWN (backlog-210). The .ml source list left every tracked .md file
# unscanned, and the pass message said "every repo-relative path cited in N
# tracked .ml/.mli files" — a true sentence that reads like whole-repo coverage.
# PR #378 was bounced three times over false prose in markdown, one of those a
# path that resolved to nothing; nothing mechanical caught any of them.
# ===========================================================================

NOOP_ML='let x = 1
'

# The measured defect: a broken inline link. Before this, exit 0.
check "red: markdown inline link to a missing file" 1 "is not a tracked file" \
  "$NOOP_ML" '# Doc

See [the design](gone/design.md) for details.
'

# Its discriminating partner. Without a green link case, "red on a link" is
# indistinguishable from "red on any link".
check "green: markdown inline link to a tracked file" 0 "OK" \
  "$NOOP_ML" '# Doc

See [the real one](Real.md) and [the source](../sub/Existing.ml).
'

# A backticked path — the dominant citation form in this repo'"'"'s docs, and
# invisible to any link-only extractor.
check "red: markdown backticked path that resolves to nothing" 1 "is not a tracked file" \
  "$NOOP_ML" '# Doc

The lowering lives in `sub/Gone.ml` these days.
'

check "green: markdown backticked path that resolves" 0 "OK" \
  "$NOOP_ML" '# Doc

The lowering lives in `sub/Existing.ml` these days.
'

# --- TIER 1, markdown half (backlog-226) ------------------------------------
# The same defect, reached through a backticked `path:NNN` in prose instead of
# an OCaml comment.
check "red: markdown backticked citation with a line past the end" 1 \
  "sub/Multi.ml has only 5 line(s)" \
  "$NOOP_ML" '# Doc

The lowering lives in `sub/Multi.ml:9` these days.
'

check "green: markdown backticked citation with a line in range" 0 "OK" \
  "$NOOP_ML" '# Doc

The lowering lives in `sub/Multi.ml:3` these days.
'

check "green: markdown range citation entirely inside the file" 0 "OK" \
  "$NOOP_ML" '# Doc

The lowering lives in `sub/Multi.ml:3-5` these days.
'

check "red: markdown range citation whose end exceeds the file" 1 \
  "sub/Multi.ml has only 5 line(s)" \
  "$NOOP_ML" '# Doc

The lowering lives in `sub/Multi.ml:3-9` these days.
'

# An anchor is the READER'"'"'s problem; the file is the gate'"'"'s. Stripped, not
# excluded — excluding anchored links would blind the gate to most doc-to-doc
# citations, which is how this class hid in the first place.
check "green: an anchor on a real path is stripped, not a reason to skip" 0 "OK" \
  "$NOOP_ML" '# Doc

See [a section](Real.md#the-part-that-matters).
'

check "red: an anchor does not rescue a missing file" 1 "is not a tracked file" \
  "$NOOP_ML" '# Doc

See [a section](Gone.md#the-part-that-matters).
'

# URLs, mailto:, bare fragments and site-absolute Jekyll permalinks address
# something other than a path in this tree.
check "green: URLs, fragments and site-absolute links are not repo paths" 0 "OK" \
  "$NOOP_ML" '# Doc

See [upstream](https://example.invalid/x.md), [here](#below), [mail](mailto:t@example.invalid)
and [the site page](/backends/).

## below
'

# A fenced block is a specimen on display, not a pointer: shell transcripts,
# `git log` output, generated ledger dumps. Excluded deliberately, which is
# exactly why inline spans OUTSIDE fences are scanned.
check "green: a path inside a fenced code block is not a citation" 0 "OK" \
  "$NOOP_ML" '# Doc

```
$ cat sub/Gone.ml
sub/AlsoGone.ml: No such file
```
'

# The positive control for the pair above: with the fence closed, a real
# citation after it is still found. Otherwise "handled fences" would be
# indistinguishable from "stopped scanning at the first fence".
check "red: a citation AFTER a fenced block is still found" 1 "is not a tracked file" \
  "$NOOP_ML" '# Doc

```
sample output
```

But the code is in `sub/Gone.ml`.
'

# An absolute or home-relative path is not a repo-relative citation: the regex
# cannot start at "/", so `/home/u/repo/sub/Gone.ml` would otherwise be reported
# as a dangling citation of a directory named "home".
check "green: absolute and ~/ paths are not repo-relative citations" 0 "OK" \
  "$NOOP_ML" '# Doc

Pasted from a shell: `/home/u/repo/sub/Gone.ml`, and the skill doc is
`~/.claude/skills/formal-apparatus/SKILL.md`. `$SCRIPT_DIR/helper.sh` runs it.
'

# gh-pages is a Jekyll site: [backends](backends.html) addresses the page built
# from backends.md. Seventeen link targets here resolve only through that rule
# (measured by deleting it and counting the findings) and every one is correct.
check "green: a .html link resolves via its markdown source" 0 "OK" \
  "$NOOP_ML" '# Doc

See [the real page](Real.html).
'

check "red: a .html link with no markdown source behind it" 1 "is not a tracked file" \
  "$NOOP_ML" '# Doc

See [a phantom page](Nowhere.html).
'

# ===========================================================================
# COMMIT SHAS (backlog-204). Five instances in one day. The defect is not a
# missing object — every one EXISTED in the clone that wrote it. It is
# unreachable: `git branch -r --contains` is empty, so no fresh clone resolves
# it. An existence check alone passes all five.
# ===========================================================================

# Build a repo with a remote-tracking ref, a reachable commit and an ORPHANED
# one (committed on a branch that is then deleted: the object survives, and no
# branch contains it — the exact shape of d72a2e6a and 1da95861).
# Echoes "<repo> <reachable-sha> <orphan-sha>".
#
# Every git call goes through `git -C` rather than a `cd` subshell, and the two
# shas are captured into variables rather than passed through a temp file. The
# first version used both, and in CI the orphan sha came back EMPTY: three cases
# then substituted nothing, cited no sha at all, and passed for the wrong reason
# while the local run was green. A fixture builder that can silently produce an
# inert fixture is the same failure this gate exists to prevent, so `checksha`
# below now asserts the substitution actually happened.
# An abbreviation the SUBJECT will actually treat as a sha.
#
# `rev-parse --short=8` returns eight hex characters, and roughly one time in
# forty-three every one of them is a DIGIT. The subject skips an all-digit token
# on purpose -- `tok.isdigit()` in check-cited-paths-exist.sh, the rule that
# stops a line count or a byte size being read as a commit -- so a fixture that
# happens to draw such a sha cites nothing the subject can see, and EVERY
# red-path case built on that fixture reports exit 0 and fails. Measured: a doc
# citing `12345678` makes the subject print "0 cited commit sha(s)" and exit 0;
# and this suite failed once in eight local runs before this helper, on a
# different case each time, which is what a per-fixture 2.3% draw looks like
# across four fixtures.
#
# Widening the abbreviation keeps the sha REAL (the subject accepts 7..40 hex)
# and costs the test nothing. Falls back to the full 40 characters, which cannot
# be all digits for any commit this fixture will ever produce.
abbrev_sha() {
  local repo="$1" rev="$2" tok
  for width in 8 10 12; do
    tok="$(git -C "$repo" rev-parse --short="$width" "$rev" 2>/dev/null)"
    case "$tok" in
      *[a-f]*) printf '%s' "$tok"; return 0 ;;
    esac
  done
  git -C "$repo" rev-parse "$rev" 2>/dev/null
}

mkshafixture() {
  local d ok orphan br
  d="$(mktemp -d)"
  git -C "$d" init --quiet . >/dev/null 2>&1 || return 1
  git -C "$d" config user.email t@example.invalid
  git -C "$d" config user.name t
  git -C "$d" config commit.gpgsign false
  mkdir -p "$d/sub" "$d/scripts" "$d/docs"
  printf 'let real = 1\n' >"$d/sub/Existing.ml"
  printf 'let x = 1\n' >"$d/sub/Thing.ml"
  printf '# Index\n\nNothing cited here.\n' >"$d/docs/Index.md"
  git -C "$d" add -A >/dev/null 2>&1
  git -C "$d" commit --quiet -m reachable >/dev/null 2>&1 || return 1
  ok="$(abbrev_sha "$d" HEAD)"
  # The initial branch name is whatever this git's init.defaultBranch says, so
  # ask rather than assume: `git init -b main` is not portable to every runner,
  # and a failed init is how the first version of this fixture went inert.
  br="$(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  # The ORPHAN: committed on a detached HEAD, which no branch will ever contain.
  # The object survives; `git branch -r --contains` is empty. That is the exact
  # shape of d72a2e6a and 1da95861 -- present, unreachable.
  git -C "$d" checkout --quiet --detach HEAD >/dev/null 2>&1 || return 1
  printf 'let y = 2\n' >>"$d/sub/Thing.ml"
  git -C "$d" add -A >/dev/null 2>&1
  git -C "$d" commit --quiet -m orphan >/dev/null 2>&1 || return 1
  orphan="$(abbrev_sha "$d" HEAD)"
  git -C "$d" checkout --quiet "$br" >/dev/null 2>&1 || return 1
  # A remote-tracking ref: reachability is measured against refs/remotes/*, and
  # this is the only ref the reachable sha needs to be contained in.
  git -C "$d" update-ref refs/remotes/origin/main "$ok" >/dev/null 2>&1 || return 1
  printf '%s %s %s' "$d" "$ok" "$orphan"
}

# AN UNUSABLE FIXTURE IS NOT INERT -- IT WRITES INTO THE HARNESS'S OWN REPO.
# `git -C "" add -A` does not fail: an empty -C argument leaves git in the
# directory it was started from, so a dead `$d` retargets every git call in the
# block at the repository the harness is RUNNING IN. Measured, on this file:
# a fixture whose `br` came back empty made `mkshafixture` return 1, the two
# guards below counted a failure and FELL THROUGH, and the block's
# `git -C "$d" add -A && commit -m doc && update-ref refs/remotes/origin/main HEAD`
# committed the reviewer's working tree onto the branch under review and
# rewrote refs/remotes/origin/main -- a ref shared by every worktree of that
# clone. Six junk commits, and the mutated harness came back committed, so the
# next run was inert too. Counting a failure is therefore NOT enough: the guard
# must ABORT before anything dereferences `$d`.
#
# $1 = repo path, $2.. = shas that must each be exactly 8 hex chars.
usable_fixture() {
  local d="$1" s
  shift
  [ -n "$d" ] && [ -d "$d/.git" ] || return 1
  for s in "$@"; do
    # Each sha must be one the SUBJECT can see, which is what this guard is
    # actually for. It used to demand exactly 8 characters -- a magic width that
    # was true only of `rev-parse --short=8` and that `abbrev_sha` legitimately
    # exceeds when it has to widen past an all-digit draw. So check the three
    # properties the subject cares about instead: lowercase hex, a length inside
    # the subject's 7..40 window, and at least one letter (an all-digit token is
    # skipped by `tok.isdigit()`, so a fixture citing one cites nothing).
    case "$s" in
      "" | *[!0-9a-f]*) return 1 ;;
    esac
    [ ${#s} -ge 7 ] && [ ${#s} -le 40 ] || return 1
    case "$s" in
      *[a-f]*) ;;
      *) return 1 ;;
    esac
  done
  return 0
}

# $1 = case name, $2 = expected exit, $3 = expected message substring,
# $4 = docs/Doc.md body with %REACHABLE% / %ORPHAN% placeholders
checksha() {
  local name="$1" want="$2" msg="$3" body="$4" fx d ok orphan out code
  fx="$(mkshafixture)"
  d="${fx%% *}"
  ok="$(printf '%s' "$fx" | cut -d' ' -f2)"
  orphan="$(printf '%s' "$fx" | cut -d' ' -f3)"
  # A fixture that came back without its shas substitutes the empty string, and
  # then the document cites no sha at all: every case would pass, having checked
  # nothing. That is what happened in CI on the first version of this file, so
  # the builder's output is now asserted rather than assumed.
  if ! usable_fixture "$d" "$ok" "$orphan" || [ "$ok" = "$orphan" ]; then
    echo "FAIL $name: sha fixture is unusable (repo='$d' reachable='$ok' orphan='$orphan')"
    rm -rf "$d"
    fail=$((fail + 1))
    return
  fi
  body="${body//%REACHABLE%/$ok}"
  body="${body//%ORPHAN%/$orphan}"
  case "$body" in
    *%REACHABLE%* | *%ORPHAN%*)
      echo "FAIL $name: a placeholder survived substitution"
      rm -rf "$d"
      fail=$((fail + 1))
      return
      ;;
  esac
  printf '%s' "$body" >"$d/docs/Doc.md"
  git -C "$d" add -A >/dev/null 2>&1
  git -C "$d" commit --quiet -m doc >/dev/null 2>&1
  # The commit above moved main; re-point the remote ref so the reachable sha
  # stays reachable and the orphan stays orphaned.
  git -C "$d" update-ref refs/remotes/origin/main HEAD >/dev/null 2>&1
  out="$(cd "$d" && bash "$SUBJECT" 2>&1)"
  code=$?
  rm -rf "$d"
  if [ "$code" != "$want" ]; then
    echo "FAIL $name: exit $code, wanted $want"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  if [ -n "$msg" ] && ! printf '%s' "$out" | grep -qF "$msg"; then
    echo "FAIL $name: exit $want as wanted, but message did not contain '$msg'"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
    return
  fi
  echo "PASS $name (exit $code)"
  pass=$((pass + 1))
}

# The green baseline for this half. A reachable sha must NOT be a finding, or
# every red below is satisfied by a check that rejects all hex.
checksha "green: a sha reachable from a remote branch" 0 "OK" '# Doc

Fixed in `%REACHABLE%`.
'

# The defect itself: the object exists — `git cat-file -e` succeeds — and no
# remote branch contains it.
checksha "red: a sha that EXISTS but is in zero remote branches" 1 \
  "in ZERO remote branches" '# Doc

Fixed in `%ORPHAN%`.
'

# The other half of the pair: a sha that is not an object at all.
checksha "red: a sha that is no commit in this repository" 1 \
  "no such commit" '# Doc

Fixed in `0000000abcdef1`.
'

# Numbers and digests are not shas. An all-digit run is a count; a 64-hex token
# is a sha256 and fails the 7..40 boundary by construction.
checksha "green: a line count and a sha256 digest are not commit shas" 0 "OK" '# Doc

It grew to 1234567 lines; the artifact hashes to
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
'

# A hex run that is part of a longer token is not a sha either.
checksha "green: hex inside a longer token is not a sha" 0 "OK" '# Doc

Tagged `abc1234-rc1`, built from `deadbee.tar.gz`, cache key `1abcdef.2fedcba`.
'

# A fenced block holds transcripts and lockfile digests. Same exclusion as the
# path half, for the same reason.
checksha "green: shas inside a fenced code block are not citations" 0 "OK" '# Doc

```
commit 0000000abcdef1
  parent deadbeefcafe12
```
'

# The scoped exemption channel: a document that quotes an unreachable sha in
# order to SAY it is unreachable. Four documents here do exactly that, and a
# bare exemption would also wave the sha through somewhere it is cited as
# evidence — so the exemption is keyed on the citing file.
fx="$(mkshafixture)"
d="${fx%% *}"
orphan="$(printf '%s' "$fx" | cut -d" " -f3)"
if ! usable_fixture "$d" "$orphan"; then
  echo "FAIL: sha fixture is unusable (repo='$d' orphan='$orphan'); the two" \
    "exemption-scope cases below are SKIPPED rather than run against the" \
    "harness's own repository"
  fail=$((fail + 1))
  rm -rf "$d"
else
  printf '# Doc\n\nThe old note cited `%s`, which no clone can resolve.\n' \
    "$orphan" >"$d/docs/Doc.md"
  printf 'docs/Doc.md::%s\tquoted in order to say it is unreachable\n' \
    "$orphan" >"$d/scripts/cited-paths-exempt.tsv"
  git -C "$d" add -A >/dev/null 2>&1
  git -C "$d" commit --quiet -m doc >/dev/null 2>&1
  git -C "$d" update-ref refs/remotes/origin/main HEAD >/dev/null 2>&1
  out="$(cd "$d" && bash "$SUBJECT" 2>&1)"
  code=$?
  if [ "$code" = 0 ]; then
    echo "PASS green: a file-scoped sha exemption is honoured (exit 0)"
    pass=$((pass + 1))
  else
    echo "FAIL green: a file-scoped sha exemption is honoured: exit $code, wanted 0"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
  fi
  # ...and the same exemption in the WRONG file does not apply. Without this, a
  # scoped row would be indistinguishable from a bare one. The MESSAGE is
  # asserted as well as the code: this case wanted exit 1, and a `cd` into an
  # unusable fixture also exits 1 without ever running the subject, so the code
  # alone was satisfied by a dead fixture -- observed passing that way.
  printf 'docs/Other.md::%s\tscoped elsewhere on purpose\n' \
    "$orphan" >"$d/scripts/cited-paths-exempt.tsv"
  out="$(cd "$d" && bash "$SUBJECT" 2>&1)"
  code=$?
  rm -rf "$d"
  if [ "$code" = 1 ] && printf '%s' "$out" | grep -qF "in ZERO remote branches"; then
    echo "PASS red: a sha exemption scoped to another file does not apply (exit 1)"
    pass=$((pass + 1))
  else
    echo "FAIL red: a sha exemption scoped to another file does not apply:" \
      "exit $code, wanted 1 reporting ZERO remote branches"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
  fi
fi

# --- the way this gate would pass while checking nothing --------------------
# A SHALLOW clone. `git branch -r --contains` reports nothing for EVERY sha
# there, reachable or not. So a shallow checkout either fails the whole build,
# or — if someone "fixes" that by reading empty as fine — passes everything
# while examining nothing. Exit 2 is the only answer that is neither.
fx="$(mkshafixture)"
d="${fx%% *}"
orphan="$(printf '%s' "$fx" | cut -d" " -f3)"
if ! usable_fixture "$d" "$orphan"; then
  echo "FAIL: sha fixture is unusable (repo='$d' orphan='$orphan'); the" \
    "shallow-clone case below is SKIPPED rather than run against the harness's" \
    "own repository"
  fail=$((fail + 1))
  rm -rf "$d"
else
  printf '# Doc\n\nFixed in `%s`.\n' "$orphan" >"$d/docs/Doc.md"
  git -C "$d" add -A >/dev/null 2>&1
  git -C "$d" commit --quiet -m doc >/dev/null 2>&1
  shallow="$(mktemp -d)/clone"
  git clone --quiet --depth 1 "file://$d" "$shallow" >/dev/null 2>&1
  out="$(cd "$shallow" && bash "$SUBJECT" 2>&1)"
  code=$?
  if [ "$code" = 2 ] && printf '%s' "$out" | grep -qF "SHALLOW"; then
    echo "PASS red: a shallow clone is exit 2, not a pass (exit 2)"
    pass=$((pass + 1))
  else
    echo "FAIL red: a shallow clone must be exit 2 with a SHALLOW message: exit $code"
    printf '%s\n' "$out" | sed 's/^/      /'
    fail=$((fail + 1))
  fi
  rm -rf "$shallow" "$d"
fi

# The other half of the same hole: a tree with history but NO remote-tracking
# refs. This is what `actions/checkout` produces at the default fetch-depth on a
# pull_request event — the backlog-152 shape, where a gate read "no refs" as "no
# differences" and about twenty PRs merged green onto an already-red main.
norefs="$(mkfixture 'let x = 1
' '# Doc

Fixed in `0000000abcdef1`.
')"
out="$(cd "$norefs" && bash "$SUBJECT" 2>&1)"
code=$?
rm -rf "$norefs"
if [ "$code" = 2 ] && printf '%s' "$out" | grep -qF "no remote-tracking branches"; then
  echo "PASS red: no remote-tracking refs is exit 2, not a pass (exit 2)"
  pass=$((pass + 1))
else
  echo "FAIL red: no remote-tracking refs must be exit 2: exit $code"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

# --- fails closed ----------------------------------------------------------
# Not a git tree at all. Exit 2, never 0: a check whose inputs are unavailable
# must refuse rather than report success — the vacuous-green failure mode this
# repo has been bitten by repeatedly.
outside="$(mktemp -d)"
out="$(cd "$outside" && bash "$SUBJECT" 2>&1)"
code=$?
rm -rf "$outside"
if [ "$code" = 2 ]; then
  echo "PASS red: outside a git tree (exit 2)"
  pass=$((pass + 1))
else
  echo "FAIL red: outside a git tree: exit $code, wanted 2"
  printf '%s\n' "$out" | sed 's/^/      /'
  fail=$((fail + 1))
fi

# --- the guard that keeps this file out of its own repository ---------------
# `usable_fixture` is the only thing standing between a dead fixture and
# `git -C "" commit` on the branch under review. Edited to `return 0` it would
# be invisible: every case above still passes on a GOOD fixture. So pin both
# directions -- it must accept a real fixture and refuse each way one can rot.
metafx="$(mkshafixture)"
metad="${metafx%% *}"
metasha="$(printf '%s' "$metafx" | cut -d' ' -f3)"
# Both shas the fixture hands out must be ones the SUBJECT can SEE. An
# all-digit abbreviation is skipped by the subject on purpose, so a fixture that
# draws one cites nothing and turns every red-path case built on it green --
# which is how this suite failed one run in eight before `abbrev_sha`. Checking
# the property here means a regression in `abbrev_sha` fails deterministically,
# instead of once every eleven runs on a random case.
metaok="$(printf '%s' "$metafx" | cut -d' ' -f2)"
# CONCATENATED, this tested neither sha. `case "$metaok$metasha" in *[a-f]*)`
# is satisfied by a single letter anywhere in the join, so one all-digit sha
# beside one containing a letter PASSED -- re-admitting the very draw this case
# exists to pin, and at a rate (one sha in 43, twice) close to the flake it
# replaced. The delimiter is what makes the two halves separate assertions;
# `:` cannot occur in a hex abbreviation, so it cannot be absorbed by either
# side's glob. Caught by CodeRabbit on PR #398.
case "$metaok:$metasha" in
  *[a-f]*:*[a-f]*)
    echo "PASS meta: the fixture's shas are visible to the subject (not all-digit)"
    pass=$((pass + 1))
    ;;
  *)
    echo "FAIL meta: at least one of the fixture shas '$metaok'/'$metasha' is" \
      "all digits; the subject skips those, so every sha case built on it" \
      "would pass vacuously"
    fail=$((fail + 1))
    ;;
esac

if usable_fixture "$metad" "$metasha"; then
  echo "PASS meta: usable_fixture accepts a real fixture (exit 0)"
  pass=$((pass + 1))
else
  echo "FAIL meta: usable_fixture rejected a REAL fixture (repo='$metad'" \
    "sha='$metasha') -- every fixture-backed case above is being skipped"
  fail=$((fail + 1))
fi
refuses() { # $1 = what it is, then the argv usable_fixture must reject
  local what="$1"
  shift
  if usable_fixture "$@"; then
    echo "FAIL meta: usable_fixture ACCEPTED $what -- a dead fixture then runs" \
      "\`git -C\` against the repository this harness is running in"
    fail=$((fail + 1))
  else
    echo "PASS meta: usable_fixture refuses $what"
    pass=$((pass + 1))
  fi
}
refuses "an empty repo path" "" abcd1234
refuses "a directory that is not a git repo" "$(dirname "$(mktemp -u)")" abcd1234
refuses "an empty sha" "$metad" ""
rm -rf "$metad"

# --- meta-control: this harness must be able to FAIL ------------------------
# "All cases passed" is a claim about the subject. A harness that asserted
# nothing would print it too. So run the whole file twice more against stub
# subjects — one that always exits 0, one that always exits 1 — and require both
# runs to fail. A gate whose own red path cannot go red is the class this repo
# keeps closing.
if [ -z "${CITED_PATHS_META:-}" ]; then
  for stub_code in 0 1; do
    stub="$(mktemp -d)/always-$stub_code.sh"
    printf '#!/usr/bin/env bash\nexit %s\n' "$stub_code" >"$stub"
    chmod +x "$stub"
    CITED_PATHS_META=1 CITED_PATHS_SUBJECT="$stub" bash "$0" >/dev/null 2>&1
    meta=$?
    rm -rf "$(dirname "$stub")"
    if [ "$meta" -ne 0 ]; then
      echo "PASS meta: harness rejects a subject that always exits $stub_code"
      pass=$((pass + 1))
    else
      echo "FAIL meta: harness ACCEPTED a subject that always exits $stub_code --" \
        "it is not checking what it claims to check"
      fail=$((fail + 1))
    fi
  done
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "check-cited-paths-exist.test.sh: $pass passed, $fail FAILED"
  exit 1
fi
echo "check-cited-paths-exist.test.sh: all $pass cases passed"
exit 0
