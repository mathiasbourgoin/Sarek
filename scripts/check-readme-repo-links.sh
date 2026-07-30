#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Assert that README's CI badge and Actions links name THIS repository.
#
# WHY THIS EXISTS. README.md line 5 carried
#   https://github.com/mathiasbourgoin/SPOC/actions/workflows/ci.yml/badge.svg
# for the whole life of the Sarek rework, while every other link in the file
# pointed at .../Sarek. The badge therefore rendered the CI status of a
# DIFFERENT repository — the single most-read line in the repo, reporting on
# something else. Nothing caught it because a badge is an image: it renders,
# so it looks fine.
#
# The expected owner/repo is DERIVED from the git remote, never hardcoded here.
# A hardcoded name would be a second source for the same fact and would go
# stale on a rename or a fork — the exact failure being fixed.
#
# NOT checked, deliberately: links into the OLD SPOC repository that point at
# content genuinely hosted there. gh-pages/docs/publications.md links a dozen
# talk PDFs under mathiasbourgoin/SPOC/blob/gh-pages/; those files live there
# and rewriting the URLs would break them. This gate is scoped to CI-status
# surfaces, where naming another repo is always wrong.
#
# Exit codes: 0 = clean, 1 = a link names the wrong repo, 2 = cannot determine
# the repo (no remote / not a git tree) — fail closed, never skip silently.

set -uo pipefail

README="${1:-README.md}"

if [ ! -f "$README" ]; then
  echo "check-readme-repo-links: $README not found" >&2
  exit 2
fi

# GITHUB_REPOSITORY is already owner/repo and is what Actions considers this
# repository to be, so prefer it when running in CI; fall back to the remote
# for a local run. Either way the expected name is DERIVED, never written here.
if [ -n "${GITHUB_REPOSITORY:-}" ]; then
  slug="$GITHUB_REPOSITORY"
else
  remote="$(git config --get remote.origin.url 2>/dev/null || true)"
  if [ -z "$remote" ]; then
    echo "check-readme-repo-links: no GITHUB_REPOSITORY and no remote.origin.url — cannot derive the expected repository" >&2
    exit 2
  fi
  # All four GitHub remote spellings reduce to owner/repo:
  #   git@github.com:owner/repo.git
  #   https://github.com/owner/repo(.git)
  #   ssh://git@github.com/owner/repo.git   <- was NOT handled, see below
  #   git://github.com/owner/repo.git
  #
  # The ssh:// form matters: without its rule the sed left
  # "ssh://git@github.com/owner/repo", which contains a slash and so passed the
  # old `*/*` test, and then no CI link could ever match it -- a correct README
  # failing on a correct repo, during local fallback only. Reported by
  # CodeRabbit on PR #387.
  slug="$(printf '%s' "$remote" \
    | sed -E 's#^ssh://git@github\.com/##; s#^git://github\.com/##;
               s#^git@github\.com:##; s#^https?://github\.com/##; s#\.git$##; s#/$##')"
fi
# EXACT owner/repo, not merely "contains a slash". The weaker test is what let
# the un-normalized ssh:// URI through as if it were a slug.
if ! printf '%s' "$slug" | grep -Eq '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$'; then
  echo "check-readme-repo-links: could not parse owner/repo out of '$slug'" >&2
  exit 2
fi

# CI surfaces, matched PER URL rather than per line, case-insensitively, and
# including the shields.io form.
#
# Four holes the adversarial review found in the previous version:
#   - case-sensitive host: `https://GitHub.com/owner/OTHER/...badge.svg` renders
#     on GitHub, names the wrong repo, and was accepted at exit 0.
#   - shields.io: `img.shields.io/github/actions/workflow/status/owner/OTHER/...`
#     is the most common badge host, and switching to it silently emptied the
#     gate's coverage entirely.
#   - per-LINE scoping broke this gate's own documented promise: a talk-PDF link
#     under mathiasbourgoin/SPOC sharing a line with an Actions link was flagged,
#     though the header says those are deliberately not checked.
#   - a README with NO CI surface at all exited 0 saying every link was fine, so
#     DELETING the badge was invisible.
bad=0
found_any=0
# Normalise to lowercase for host/path matching, but report the original line.
while IFS= read -r pair; do
  [ -n "$pair" ] || continue
  lineno="${pair%%:*}"
  url="${pair#*:}"
  lower="$(printf '%s' "$url" | tr '[:upper:]' '[:lower:]')"
  got=""
  case "$lower" in
    *img.shields.io/github/actions/workflow/status/*)
      got="$(printf '%s' "$url" | sed -E 's#.*[Ss]tatus/([^/]+/[^/]+).*#\1#')"
      ;;
    *github.com/*/actions*)
      got="$(printf '%s' "$url" | sed -E 's#.*github\.com/([^/]+/[^/]+)/actions.*#\1#')"
      ;;
  esac
  [ -n "$got" ] || continue
  found_any=1
  if [ "$got" != "$slug" ]; then
    line="$(sed -n "${lineno}p" "$README")"
    echo "$README:$lineno: CI link names '$got' but this repository is '$slug'"
    echo "    $url"
    bad=1
  fi
done < <(grep -noE 'https?://[^ )"]+' "$README")

if [ "$found_any" -eq 0 ]; then
  echo "check-readme-repo-links: $README has no CI badge or Actions link at all." >&2
  echo "  Exit 2, not 0: a gate that examined no CI surface has not verified one." >&2
  echo "  If the badge was removed on purpose, this gate has nothing left to guard." >&2
  exit 2
fi

if [ "$bad" -ne 0 ]; then
  echo
  echo "A CI badge or Actions link points at a different repository, so it reports"
  echo "another project's build status. Update it to '$slug'."
  exit 1
fi

echo "check-readme-repo-links: OK — every CI link in $README names $slug"
exit 0
