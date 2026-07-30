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

# Every github.com/<owner>/<repo> occurrence on a line that is a CI surface:
# an Actions link, or a workflow badge image.
bad=0
while IFS=: read -r lineno line; do
  [ -n "$lineno" ] || continue
  found="$(printf '%s' "$line" \
    | grep -oE 'github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+' \
    | sed 's#github\.com/##' | sort -u)"
  for got in $found; do
    if [ "$got" != "$slug" ]; then
      echo "$README:$lineno: CI link names '$got' but this repository is '$slug'"
      echo "    $line"
      bad=1
    fi
  done
done < <(grep -nE 'actions/workflows/[^ )]*badge\.svg|github\.com/[^ )]*/actions' "$README")

if [ "$bad" -ne 0 ]; then
  echo
  echo "A CI badge or Actions link points at a different repository, so it reports"
  echo "another project's build status. Update it to '$slug'."
  exit 1
fi

echo "check-readme-repo-links: OK — every CI link in $README names $slug"
exit 0
