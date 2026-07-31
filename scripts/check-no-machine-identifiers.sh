#!/usr/bin/env bash
# SPDX-License-Identifier: CeCILL-B
# Copyright (c) 2012-2026 Mathias Bourgoin
#
# Refuse machine-identifying data in tracked files.
#
# WHY (backlog-168): benchmark output carried `system.hostname`, `system.kernel`
# and host `system.memory_gb`. Three personal machines ended up named in 263
# committed files -- in the FILENAMES as well as the payloads -- and in the
# published gh-pages dashboard. The producer is fixed; this stops it coming back.
#
# Four independent checks, because the leak had four independent shapes and
# closing one would not have caught the others:
#
#   1. PAYLOAD  -- a benchmark JSON carrying a removed field.
#   2. PATH     -- a tracked filename that looks like <host>_<bench>_<size>_<ts>,
#                  which is exactly the shape the 263 files had.
#   3. PRODUCER -- source that writes a removed field, or shells out to
#                  `hostname` outside the one sanctioned call site.
#   4. HOMEPATH -- an absolute path under /home or /Users naming a specific
#                  user account, anywhere in a tracked text file (backlog-216).
#
# Checks 1-3 are the backlog-168 surfaces. Check 4 is a LATER and NARROWER
# addition and this comment is the whole of its claim: it looks for one lexical
# shape in tracked text, nothing more. It does not look for hostnames in prose,
# usernames outside a path, IP addresses, MAC addresses, serial numbers or
# e-mail addresses, and the file's name promising "no machine identifiers"
# should not be read as saying otherwise.
#
# Scope is `git ls-files`, not the filesystem: an untracked scratch file is not
# a disclosure, and a tracked one is -- regardless of what is on disk.
#
# Exit: 0 clean - 1 identifier found - 2 the check could not run (fails closed).

set -uo pipefail

cd "$(dirname "$0")/.." || { echo "::error::cannot reach repo root" >&2; exit 2; }

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "::error::not a git work tree -- this check reads 'git ls-files'" >&2
  exit 2
fi

DEIDENT="scripts/deidentify-benchmark-results.py"
[ -f "$DEIDENT" ] || { echo "::error::$DEIDENT missing" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "::error::python3 required" >&2; exit 2; }

# The label shape is defined once, in scripts/machine-label-shape.sh, and shared
# with the producer's override validation (benchmarks/machine_label.ml). Sourced
# rather than restated: two independently written regexes for one rule is how
# drift starts. Fails closed -- without the shape this gate cannot tell a label
# from a hostname, and a check that cannot decide is not a pass.
SHAPE="scripts/machine-label-shape.sh"
# shellcheck source=scripts/machine-label-shape.sh
[ -f "$SHAPE" ] && . "$SHAPE" \
  || { echo "::error::$SHAPE missing -- cannot decide what a legal machine label is" >&2; exit 2; }
[ -n "${MACHINE_LABEL_PATH_SHAPE:-}" ] \
  || { echo "::error::$SHAPE defined no MACHINE_LABEL_PATH_SHAPE" >&2; exit 2; }
[ -n "${MACHINE_LABEL_RESULT_TAIL:-}" ] \
  || { echo "::error::$SHAPE defined no MACHINE_LABEL_RESULT_TAIL" >&2; exit 2; }

# A `git grep` whose output is taken with `|| true` cannot be distinguished from
# a `git grep` that found nothing, so an operational failure reads as a clean
# rule -- and the header above promises exit 2 when this check cannot run. Every
# `git grep` goes through here. 0 = matches, 1 = none, anything else is git
# failing and is not a verdict.
#
# Filtering and matching are then done with bash's own `=~` and `case`, so there
# is no other external command whose failure could empty a result: the only
# processes this check spawns are git, python3 for the scrubber, and mktemp.
#
# The result comes back in $GG_OUT rather than on stdout, and that is the whole
# point: called as `x=$(gg ...)` the function would run in a SUBSHELL, where its
# `exit 2` ends the subshell and the script carries on with an empty result --
# the exact vacuous pass this exists to prevent, wearing a fail-closed comment.
GG_OUT=""
gg() {
  local st
  GG_OUT="$(git grep "$@" 2>/dev/null)"
  st=$?
  if [ "$st" -gt 1 ]; then
    echo "::error::git grep failed (exit $st) -- this check cannot report a pass" >&2
    exit 2
  fi
}

failed=0

# --- 1. payloads -----------------------------------------------------------
# Reuse the scrubber's own --check so the gate and the fixer can never disagree
# about what counts as an identifier.
mapfile -t payloads < <(git ls-files -- 'benchmarks/results/*.json' \
                                       'gh-pages/benchmarks/data/*.json' 2>/dev/null)
if [ ${#payloads[@]} -gt 0 ]; then
  if ! python3 "$DEIDENT" --check "${payloads[@]}"; then
    failed=1
  fi
else
  echo "note: no tracked benchmark payloads (that is the expected state after"
  echo "      backlog-168 dropped the historical results)"
fi

# --- 2. paths --------------------------------------------------------------
# The 263 files were named <hostname>_<benchmark>_<size>_<ISO timestamp>.json.
# A machine LABEL is allowed in that position; a hostname is not, and the two
# are not distinguishable by pattern -- so require the label to have the shape
# the producer can actually emit: <os>-<vendor>, optionally plus the bounded
# disambiguating suffix (see $SHAPE for the shape and why it is bounded).
#
# Deliberately NOT a blocklist of the three known hostnames: that would pass a
# fourth machine, which is how this class survives. The enumerated <os>-<vendor>
# prefix is what refuses a bare hostname -- `drangleic` cannot acquire one, and
# the suffix cannot supply it either since the prefix is mandatory.
#
# Both patterns are composed from $SHAPE's ONE definition of the filename tail,
# so the label the allowlist looks for is anchored to the same path component as
# the name that made the file a candidate. A label-only allowlist was satisfied
# by a sibling DIRECTORY (see MACHINE_LABEL_PATH_SHAPE).
# Matched with bash's own `=~`, not a `grep | grep || true` pipeline: shimming
# the first grep to fail made that pipeline yield an empty result and the whole
# rule reported clean. An external filter that can fail is a way for this gate
# to pass without looking, so there is no external filter here at all.
bad_paths=""
while IFS= read -r rp || [ -n "$rp" ]; do
  [ -n "$rp" ] || continue
  [[ "$rp" =~ /[^/]+${MACHINE_LABEL_RESULT_TAIL} ]] || continue
  [[ "$rp" =~ $MACHINE_LABEL_PATH_SHAPE ]] && continue
  bad_paths+="$rp"$'\n'
done < <(git ls-files -- 'benchmarks/results/*' 2>/dev/null)
if [ -n "$bad_paths" ]; then
  echo "::error::tracked result path(s) are not named after a derived machine label:"
  printf '  %s\n' $bad_paths
  echo "  Expected <os>-<vendor>[-<suffix>]_<benchmark>_<size>_<timestamp>.json,"
  echo "  matching $MACHINE_LABEL_SHAPE"
  echo "  If that leading token is a hostname, it must not be committed."
  failed=1
fi

# --- 3. producer -----------------------------------------------------------
# `hostname` may be read at exactly one place: the check that REFUSES an
# operator override equal to the hostname. Anywhere else is a reintroduction.
SANCTIONED='benchmarks/system_info.ml'
gg -nI -E 'open_process_in "hostname"|Unix\.gethostname' -- '*.ml'
producer_hits=""
while IFS= read -r ph || [ -n "$ph" ]; do
  [ -n "$ph" ] || continue
  case "$ph" in "${SANCTIONED}:"*) continue ;; esac
  producer_hits+="$ph"$'\n'
done < <(printf '%s' "$GG_OUT")
if [ -n "$producer_hits" ]; then
  echo "::error::source reads the hostname outside ${SANCTIONED}:"
  printf '  %s\n' "$producer_hits"
  failed=1
fi

# A JSON writer emitting a removed field. Matches the emit shape, not prose, so
# the comments explaining the removal do not trip it.
gg -nI -E '\("(hostname|kernel)", *`String' -- '*.ml'
emit_hits="$GG_OUT"
if [ -n "$emit_hits" ]; then
  echo "::error::source emits a field removed by backlog-168:"
  printf '  %s\n' "$emit_hits"
  failed=1
fi

# The CSV header is a separate surface -- it leaked independently of the JSON.
gg -nI -E '"(benchmark,timestamp,hostname|hostname,timestamp)' -- '*.ml'
csv_hits="$GG_OUT"
if [ -n "$csv_hits" ]; then
  echo "::error::a CSV header still declares a hostname column:"
  printf '  %s\n' "$csv_hits"
  failed=1
fi

# --- 4. absolute home paths (backlog-216) ----------------------------------
# The measured gap: this gate exited 0 with `/home/<user>/dev/SPOC/_opam`
# committed in a kb page. Checks 1-3 look at benchmark payloads, benchmark
# filenames and producer source; none of them reads prose, and the leak that
# reached a published site in the first place was a name in a FILENAME, so
# "prose is harmless" was never the rule -- it was just out of scope.
#
# What is enforced, exactly: a tracked text file may not contain the two-
# component prefix `/home/<account>` or `/Users/<account>`, where <account>
# matches $HOME_ACCOUNT_RE below, at a position where it can begin a path.
# Two harms, and the second is the one that actually bites: the path discloses
# whose machine produced the file, and nobody else can re-run the command it
# appears in. The kb instance was a verification command pinned to one
# developer's opam switch DIRECTORY; the fix was to name the SWITCH, which
# anyone can resolve, and that remedy is in the error message because it is the
# fix in most cases.
#
# The four things this rule does NOT do, each for a reason, none of them an
# oversight -- the gate's file name says "no machine identifiers" and that name
# is wider than these rules, so the limits are written down rather than implied:
#
#  a. `~/`, `$HOME` and `${HOME}` are not findings. Both harms are absent --
#     those spellings name no user and re-run correctly for everyone -- and the
#     sweep behind this rule measured 106 tracked occurrences of them on
#     2026-07-31 (48 tilde-slash, 58 $HOME/${HOME}), every one legitimate. A
#     rule that lands with a hundred known failures gets switched off.
#     So this rule cannot see a portable reach into a home directory, which is
#     exactly the shape it wants people to write.
#
#  b. The account component is $HOME_ACCOUNT_RE, and it must run to a `/` or to
#     the end of its token. A POSIX account may hold characters outside that
#     class, and such a name is not a finding AT ALL -- not a partial one.
#     Widening the class is not free: `/home/<user>/` and `/home/$USER/` are how
#     this repo's own prose, and this very comment, write a home path they do
#     NOT mean literally, and a class admitting `<` or `$` would make the
#     documentation of the rule a violation of it. The class is the trade, and
#     this sentence is the disclosure.
#
#     Running to the end of the token is what stops the rule reporting a
#     TRUNCATED root: an earlier version reported the prefix before an unusual
#     character, so an exemption written for the reported root would also have
#     waived a genuine account of that shorter name. A missed finding is a
#     disclosed limit; an aliasing exemption is a hole.
#
#  c. Only the two leading components are matched, so the finding is the home
#     ROOT and not the whole path. That is deliberate -- one exemption row then
#     covers a file that reaches into the same home twelve times -- but it means
#     a username appearing anywhere other than the second component is not seen.
#     A path component such as `-home-<user>-dev-...` is not a match.
#
#  d. Scope is the INDEX (`--cached`): what would be published, not what is in
#     the editor. A fix must be STAGED before this rule sees it, which is the
#     right way round for a rule about committed content, and it is neither git
#     history nor any file git treats as binary (-I).
HOME_EXEMPT="scripts/home-path-exempt.tsv"
# First char excludes `.` so that `/home/../etc` and `/home/./x` are not read as
# accounts named `..` and `.`.
HOME_ACCOUNT_RE='[A-Za-z0-9_-][A-Za-z0-9._-]*'
HOME_ROOT_RE="(/home|/Users)/${HOME_ACCOUNT_RE}"
# A line is split into TOKENS on the characters that cannot occur inside a path
# anyone would write here, and a token is a finding only if the home root starts
# at its BEGINNING. That yields both boundaries at once, and it is why the
# boundary is not a regex:
#
#  - a regex left boundary CONSUMES the character it matches, so two roots with
#    one space between them yielded ONE finding, not two -- and an exemption for
#    the first then waived
#    the second, which is an aliasing hole rather than a missed line.
#  - starting at the beginning of a token is what refuses a `/srv/home/<user>`
#    mount point, a URL path, and any other embedded home-shaped segment.
#  - the trailing slash is NOT required: a `chown -R opam:opam` argument and a
#    `--switch=` value end at a token boundary, and three such slashless
#    occurrences were already tracked when this rule was written.
#
# Splitting is bash IFS word splitting, so no external process is involved and
# none can fail in a way that quietly empties the result.
HOME_SPLIT_IFS=$' \t"'"'"'`(){}<>[],;:|=*?!&'
# Parenthesised as a whole so BASH_REMATCH[1] is the ROOT and not just the
# /home-or-/Users alternation, which is group 1 inside $HOME_ROOT_RE.
HOME_TOKEN_RE="^(${HOME_ROOT_RE})(/|\$)"

# Exemptions are SCOPED ONLY -- `citing/file::/home/<user>/`. See $HOME_EXEMPT
# for why there is no bare form. A malformed row is exit 2: a row this parser
# cannot read is an exemption nobody can audit, and silently dropping it would
# turn a typo into a surprise finding somewhere else.
#
# A MISSING exemption file is not fatal: with no exemptions the rule can only
# get stricter, and a failure mode that can only produce a FALSE FINDING does
# not need to fail closed. A missing $SHAPE above is different -- that one
# leaves the gate unable to decide, which is not a pass.
declare -A home_exempt=()
declare -A home_exempt_used=()
# Line numbers of well-formed rows, recorded HERE so that the self-scan below
# and this parser cannot disagree about what "well-formed" means.
declare -A home_exempt_row=()
# From the INDEX, like the scan. Read from the working tree, an UNSTAGED
# exemption waived a STAGED home path -- the two halves of the rule disagreeing
# about which version of the repository they are judging.
hp_exempt_body="$(mktemp)" || { echo "::error::mktemp failed" >&2; exit 2; }
# On a trap, not only on the happy path: the three malformed-row exits below are
# `exit 2` and each one used to leave its temporary file behind.
trap 'rm -f "${hp_exempt_body:-}" "${hp_list:-}" "${hp_body:-}"' EXIT
if git show ":0:$HOME_EXEMPT" >"$hp_exempt_body" 2>/dev/null; then
  exempt_lineno=0
  while IFS= read -r line || [ -n "$line" ]; do
    exempt_lineno=$((exempt_lineno + 1))
    line="${line%$'\r'}"   # a CRLF checkout must not turn every row malformed
    case "$line" in ''|'#'*) continue ;; esac
    key="${line%%$'\t'*}"
    reason="${line#*$'\t'}"
    if [ "$reason" = "$line" ] || [ -z "${reason//[[:space:]]/}" ]; then
      echo "::error::$HOME_EXEMPT:$exempt_lineno: exemption has no TAB-separated reason" >&2
      exit 2
    fi
    case "$key" in
      *'::'*) ;;
      *) echo "::error::$HOME_EXEMPT:$exempt_lineno: exemptions must be scoped as <file>::<home-root>/, not bare '$key'" >&2
         exit 2 ;;
    esac
    # LAST `::`, not the first: a tracked path may legally contain `::`, and a
    # home root never can, so the last occurrence is the separator.
    ex_file="${key%::*}"
    ex_root="${key##*::}"
    if ! printf '%s' "$ex_root" | grep -qE "^${HOME_ROOT_RE}/\$"; then
      echo "::error::$HOME_EXEMPT:$exempt_lineno: '$ex_root' is not a home root (want two components and a trailing slash)" >&2
      exit 2
    fi
    home_exempt["$ex_file::$ex_root"]=1
    home_exempt_row["$exempt_lineno"]=1
  done <"$hp_exempt_body"
fi
rm -f "$hp_exempt_body"

# Candidate FILES, NUL-separated: a tracked path may legally contain a colon, so
# `file:line:text` cannot be parsed back apart. Each candidate is then read from
# the index and numbered here.
#
# The selection regex is deliberately LOOSER than the per-token decision above:
# over-selecting a file costs a read, while under-selecting one is a miss, so
# every boundary decision is made by the tokeniser and none of it here.
HOME_CANDIDATE_RE="(/home|/Users)/[A-Za-z0-9_-]"
hp_list="$(mktemp)" || { echo "::error::mktemp failed" >&2; exit 2; }
git grep -lI --cached -z -E "$HOME_CANDIDATE_RE" -- . >"$hp_list" 2>/dev/null
hp_status=$?
# 0 = matches, 1 = none. Anything else is git failing, which is not a pass:
# `|| true` here would turn every operational error into a clean rule.
if [ "$hp_status" -gt 1 ]; then
  echo "::error::git grep failed (exit $hp_status) -- this rule cannot report a pass" >&2
  rm -f "$hp_list"
  exit 2
fi

home_hits=""
while IFS= read -r -d '' hf; do
  [ -n "$hf" ] || continue
  hp_body="$(mktemp)" || { echo "::error::mktemp failed" >&2; rm -f "$hp_list"; exit 2; }
  # `:0:<path>`, not `:<path>`: the short form reads a tracked `0:note.md` as
  # stage 0 of `note.md`, so a leaking file was silently replaced by its clean
  # namesake. The explicit stage number leaves the rest of the string a path.
  if ! git show ":0:$hf" >"$hp_body" 2>/dev/null; then
    echo "::error::cannot read $hf from the index -- this rule cannot report a pass" >&2
    rm -f "$hp_list" "$hp_body"
    exit 2
  fi
  hl=0
  while IFS= read -r text || [ -n "$text" ]; do
    hl=$((hl + 1))
    scan="$text"
    # The exemption file is scanned like any other file. On a WELL-FORMED ROW
    # only the first field is waived -- by scanning the reason alone -- so a
    # real home path in the reason text, or in a comment that happens to look
    # like a row, is still a finding. Waiving the whole line was the first
    # shape of this code and both reviewers broke it in one line.
    if [ "$hf" = "$HOME_EXEMPT" ] && [ -n "${home_exempt_row[$hl]:-}" ]; then
      scan="${text#*$'\t'}"
    fi
    IFS="$HOME_SPLIT_IFS" read -r -a hp_tokens <<<"$scan"
    declare -A hp_seen=()
    for tok in ${hp_tokens[@]+"${hp_tokens[@]}"}; do
      [[ "$tok" =~ $HOME_TOKEN_RE ]] || continue
      # Two components only, canonicalised with a trailing slash: the exemption
      # rows are written that way, so a reported root pastes into one unchanged.
      root="${BASH_REMATCH[1]}/"
      [ -n "${hp_seen[$root]:-}" ] && continue
      hp_seen["$root"]=1
      if [ -n "${home_exempt["$hf::$root"]:-}" ]; then
        home_exempt_used["$hf::$root"]=1
        continue
      fi
      hint=""
      case "$text" in
        *--switch=*|*OPAMSWITCH*|*'opam switch'*)
          hint="  -> name the opam SWITCH, not its directory (\`--switch=.\` from the repo root)" ;;
      esac
      home_hits+="$hf:$hl: $root"$'\n'
      [ -n "$hint" ] && home_hits+="$hint"$'\n'
    done
    unset hp_seen
  done <"$hp_body"
  rm -f "$hp_body"
done <"$hp_list"
rm -f "$hp_list"

if [ -n "$home_hits" ]; then
  echo "::error::tracked file(s) contain an absolute home path naming a user account:"
  printf '%s' "$home_hits" | sed 's/^/  /'
  echo "  Nobody else can re-run a command pinned to one developer's home directory,"
  echo "  and the path says whose machine it was. Write it relative to the repository"
  echo "  root, or with \$HOME / ~ if it must leave the repo. If the path is the"
  echo "  SUBJECT of the text rather than a path being used, add a scoped row to"
  echo "  $HOME_EXEMPT."
  failed=1
fi

# A row that no longer matches anything is an unaudited claim: it says a file
# legitimately contains a path that file no longer contains. Removing it is the
# fix, and it is a one-line one. Rows are marked used only from the CITING file,
# so a row cannot satisfy itself.
stale=""
for k in "${!home_exempt[@]}"; do
  [ -n "${home_exempt_used[$k]:-}" ] || stale+="$k"$'\n'
done
if [ -n "$stale" ]; then
  echo "::error::$HOME_EXEMPT has row(s) that match nothing -- delete them:"
  printf '%s' "$stale" | sed 's/^/  /'
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  echo ""
  echo "Fix: run scripts/deidentify-benchmark-results.py on the payload(s), drop the"
  echo "     identifying field from the producer, or replace the absolute home path"
  echo "     with a repository-relative one. See backlog-168 and backlog-216."
  exit 1
fi

echo "OK -- ${#payloads[@]} tracked payload(s), no machine identifier in payloads, paths"
echo "     or producers, and no absolute home path -- a /home or /Users root whose"
echo "     account matches ${HOME_ACCOUNT_RE} -- outside ${#home_exempt[@]} scoped exemption(s)"
exit 0
