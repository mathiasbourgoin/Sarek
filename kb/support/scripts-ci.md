# Scripts and CI Support

## Component Inventory

- Shell scripts: `scripts/coverage-unit.sh`, `coverage-e2e.sh`, `coverage-benchmarks.sh`, `coverage-aggregate.sh`, `check-license-headers.sh`, `add-license-headers.sh`.
- CI image: `ci/Dockerfile`.
- Legacy Docker helpers: `docker_scripts/run.sh`, `run_with_x.sh`, `emacs-pkg-install.sh`, `emacs-pkg-install.el`.
- Workflows: `.github/workflows/ci.yml`, `docs.yml`, `deploy-pr-preview.yml`, `cleanup-pr-preview.yml`, `ghcr-image.yml`, `retarget-prs.yml`, plus workflow docs.
- Agent stubs: `.github/agents/*.agent.md`.

## Per-File Purpose

- `scripts/coverage-unit.sh`: builds unit test targets with `bisect_ppx`, attempts to install bisect if missing, runs unit/core/SPOC tests, emits HTML/summary reports.
- `scripts/coverage-e2e.sh`: instruments and runs selected e2e tests using `--native`.
- `scripts/coverage-benchmarks.sh`: instruments selected e2e binaries and runs small native benchmark-like cases.
- `scripts/coverage-aggregate.sh`: combines unit, framework, SPOC, e2e, and benchmark coverage in one report.
- `scripts/add-license-headers.sh`: mutating fixer that adds/updates SPDX headers using git history.
- `scripts/check-license-headers.sh`: intended checker for SPDX header freshness.
- `ci/Dockerfile`: Ubuntu 22.04 oneAPI/OpenCL CI base with build tools, opam, clang, git, and `OPAMYES=1`.
- `docker_scripts/run.sh`: runs `spoc_docker` interactively.
- `docker_scripts/run_with_x.sh`: runs `spoc_docker` with X11 mount and `--privileged`.
- `docker_scripts/emacs-pkg-install.*`: old command-line Emacs package installer.
- `.github/workflows/ci.yml`: Dockerized opam build, unit/e2e tests, optional coverage artifact upload, generated-code freshness.
- `.github/workflows/docs.yml`: builds odoc, Jekyll, combines docs, deploys to `gh-pages`.
- `.github/workflows/deploy-pr-preview.yml`: same-repo PR preview deployment under `preview/pr-N/`.
- `.github/workflows/cleanup-pr-preview.yml`: removes preview directory on PR close.
- `.github/workflows/ghcr-image.yml`: builds/pushes interactive image to GHCR.
- `.github/workflows/retarget-prs.yml`: retargets stacked PRs after a base branch is merged.
- `.github/agents/*.agent.md`: minimal activation wrappers that instruct the agent to load `_bmad/...` persona files.

## Features and APIs

- Coverage scripts set `BISECT_FILE` and write reports under `_coverage/*-report` (`scripts/coverage-unit.sh:42-76`, `scripts/coverage-e2e.sh:110-138`, `scripts/coverage-benchmarks.sh:169-194`).
- `ci.yml` builds a reusable Docker image from `ci/Dockerfile`, caches `.opam-ci`, runs `dune build @install`, `dune runtest`, `make e2e-fast`, and optional coverage (`.github/workflows/ci.yml:33-115`).
- PR preview workflow converts benchmark results to web JSON, rewrites Jekyll `baseurl`, builds the site, deploys to `gh-pages`, and comments URLs (`.github/workflows/deploy-pr-preview.yml:41-88`).
- Docs workflow appends `docs/odoc_custom.css` to generated `odoc.css` before deployment (`.github/workflows/docs.yml:247-258`).

## Invariants

- Check scripts should either be read-only or clearly named as mutating fixers.
- CI should not require GPUs for normal pull request validation.
- PR preview deployment with write permissions must not execute untrusted fork code; it gates on same-repository PRs (`.github/workflows/deploy-pr-preview.yml:18-20`).
- Cleanup should only remove `preview/pr-<number>` on the `gh-pages` branch (`.github/workflows/cleanup-pr-preview.yml:192-210`).
- Coverage instrumentation assumes a local opam switch under `_opam` in scripts, while CI sets `OPAMROOT=.opam-ci`.

## Potential Invariant Violations or Bugs

- `scripts/check-license-headers.sh` is not a dry-run. It runs the mutating fixer and then checks `git diff` (`scripts/check-license-headers.sh:23-40`), so running the checker can leave source edits.
- `scripts/add-license-headers.sh` processes only `scripts ci` for shell scripts (`scripts/add-license-headers.sh:225-245`), so `benchmarks/run_all_benchmarks.sh` and `docker_scripts/*.sh` are outside the fixer/checker scope.
- `scripts/coverage-aggregate.sh` builds `sarek/framework/test/` (`scripts/coverage-aggregate.sh:22-25`), while the earlier inventory did not show `sarek/framework/test` in the support-scope list; if absent in a checkout, aggregate coverage will fail. Marked uncertain because this path may exist outside the support slice.
- `docker_scripts/run_with_x.sh` uses `--privileged` and mounts the X11 socket (`docker_scripts/run_with_x.sh:2`), which is inappropriate as a default helper on shared hosts.
- `docker_scripts/emacs-pkg-install.el` uses an HTTP MELPA archive URL (`docker_scripts/emacs-pkg-install.el:10-11`), exposing package metadata/downloads to transport tampering.
- PR preview executes PR-controlled Jekyll/JS/benchmark content with `contents: write` and `pull-requests: write`, limited to same-repo PRs (`.github/workflows/deploy-pr-preview.yml:12-20`, `.github/workflows/deploy-pr-preview.yml:73-88`). This is acceptable only if same-repo branch write access is trusted.
- All workflows use action tags rather than immutable SHAs (`.github/workflows/ci.yml:25-46`, `.github/workflows/docs.yml:16-70`, `.github/workflows/ghcr-image.yml:20-30`).

## Performance and Maintainability Risks

- CI still adds the alpha opam repository in workflow setup (`.github/workflows/ci.yml:69-72`, `.github/workflows/ci.yml:164-168`) and local coverage scripts (`scripts/coverage-unit.sh:19-27`), increasing build variance.
- Workflow Docker commands duplicate long opam setup blocks.
- Coverage scripts assume CUDA library paths even for native-only coverage (`scripts/coverage-unit.sh:29-38`, `scripts/coverage-e2e.sh:93-111`).
- Agent stubs reference `_bmad/...` files that are not in this repository; users invoking them without that external tree get broken activation.

## Recently Resolved

- The mandatory CI build no longer installs the removed `bisect_ppx.2.8.3.1~alpha-repo` package. This fix merged through PR #136 as part of `5dffea3`.
- `scripts/coverage-unit.sh` now attempts a generic `bisect_ppx` install instead of pinning the removed alpha-only version, and still skips optional coverage if opam cannot solve it.

## Related Tests and Checks

- Workflows themselves are the primary checks: `ci.yml`, `docs.yml`, `deploy-pr-preview.yml`.
- `make e2e-fast` is the PR-friendly e2e target used by CI (`.github/workflows/ci.yml:85-94`, `Makefile:248-272`).
- Generated-code freshness is enforced in `ci.yml:117-191`.
- License header check exists but is mutating.

## Missing Tests

- A shellcheck/shfmt pass for `scripts/**`, `benchmarks/run_all_benchmarks.sh`, and `docker_scripts/**`.
- A test that `check-license-headers.sh` leaves a clean tree when no changes are needed.
- Workflow policy linting for action SHA pinning and least-privilege permissions.
- PR-preview sanitizer/security tests for benchmark data and docs content.
- CI dry-run for docs build without deploying.

## Concrete Improvement Candidates

- Add a true dry-run mode to `add-license-headers.sh` and make `check-license-headers.sh` use it.
- Extend license-header scope to all shell scripts or document exclusions.
- Replace HTTP MELPA with HTTPS or remove stale Emacs helper scripts.
- Remove `--privileged` from `docker_scripts/run_with_x.sh` or mark it as an explicit unsafe/debug helper.
- Pin workflow actions by SHA and add a scheduled dependency-refresh process.
- Factor repeated CI opam setup into a script or composite action.
