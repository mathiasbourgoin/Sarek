#!/usr/bin/env bash
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

remote="$(git config --get remote.origin.url 2>/dev/null || true)"
if [ -z "$remote" ]; then
  echo "check-readme-repo-links: no remote.origin.url — cannot derive the expected repository" >&2
  exit 2
fi

# git@github.com:owner/repo.git and https://github.com/owner/repo(.git) both
# reduce to owner/repo.
slug="$(printf '%s' "$remote" \
  | sed -E 's#^git@github\.com:##; s#^https?://github\.com/##; s#\.git$##')"
case "$slug" in
  */*) : ;;
  *)
    echo "check-readme-repo-links: could not parse owner/repo out of remote '$remote'" >&2
    exit 2
    ;;
esac

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
