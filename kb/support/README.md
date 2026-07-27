# Repository Support Surface

Reviewed slice: repository support files and generated/static boundaries. This KB covers `benchmarks/**` excluding `benchmarks/results/**`, `tools/**`, `scripts/**`, `ci/**`, `docker_scripts/**`, `Dockerfile`, `Makefile`, root `dune`, root `*.opam`, root `README.md`, `.github/workflows/**`, `.github/agents/**`, `gh-pages/**` excluding generated `gh-pages/spoc_docs/**`, `docs/**`, `notebooks/**`, and `dependencies/**`.

Large third-party/static assets under `gh-pages/static/**`, `gh-pages/pres_resources/**`, `gh-pages/docs/talks/**`, and vendored headers under `dependencies/**` were inventoried and license/wrapping surfaces were reviewed. Minified/vendor JS/CSS/font/binary assets were not semantically audited line by line.

## Component Inventory

- `benchmarks/`: standalone Sarek benchmark executables, shared result schema, result aggregation/conversion tools, generated backend-code exporter, contributor docs, and the all-benchmark runner.
- `tools/`: backend initialization and `sarek-device-info` diagnostic utility.
- `scripts/`, `ci/`, `docker_scripts/`, `.github/workflows/`: coverage, license-header, Docker, CI, docs, preview, release-image, and PR-retarget support.
- `Dockerfile`, `Makefile`, root `dune`, root `*.opam`, `README.md`: packaging, local development, Binder/Jupyter image, opam metadata, and top-level task entry points.
- `gh-pages/`, `docs/`, `notebooks/`: Jekyll documentation site, benchmark dashboard/viewer, static legacy notebooks, talks, images, and odoc style overlay.
- `dependencies/`: vendored OpenCL and CUDA C headers plus license files.
- `.github/agents/`: GitHub Copilot/agent persona activation stubs referencing `_bmad/...` files that are not present in this repository slice.
- **Missing from prior inventory, added 2026-07-02:**
  - `gh-pages/learn/`: interactive GPU course — 9 lesson pages (`01-vector-add.html` through `09-reduction.html`), `index.md`, and `gh-pages/learn/test/` (Playwright acceptance tests driven by `make lessons-gpu-test`/`compose-gpu-test`).
  - `gh-pages/playground.html`, `gh-pages/javascripts/sarek_lesson.js`, `gh-pages/javascripts/sarek_webgpu_runner.js`: the in-browser WebGPU transpiler playground and lesson/runtime harness that drive `gh-pages/learn/`.
  - `scripts/gpu-bench-check.sh` (110 lines) and the `make bench-gpu-check` target (`Makefile:395-400`): correctness gate that runs the self-verifying benchmark suite on available GPU backends and fails on any `"verified": false` result or crash.
  - `Makefile:401-427`: `wgsl-gpu-test`, `lessons-gpu-test`, `webgpu-runtime-test`, `compose-gpu-test` — Playwright + flagged-Chrome (Dawn/Vulkan) GPU acceptance targets for the WGSL backend, the course lessons, the jsoo WebGPU runtime, and kernel composition. All skip when Playwright/Chrome/WebGPU are unavailable.
  - `.github/workflows/block-master-prs.yml`: fails any PR targeting the stale `master` branch (diverged from `main` since 2019) with a guidance message to retarget onto `main`.
  - `SPOC_DISABLE_GPU`/`SPOC_DISABLE_<BACKEND>` environment gating in `benchmarks/backend_{cuda,opencl,vulkan,metal}.available.ml` (e.g. `benchmarks/backend_cuda.available.ml:8-9`) — see [benchmarks.md](benchmarks.md) and [tools.md](tools.md) for the corresponding (and now functionally diverged) `tools/` stubs.

## Subdocs

- [benchmarks.md](benchmarks.md)
- [tools.md](tools.md)
- [scripts-ci.md](scripts-ci.md)
- [docs-site.md](docs-site.md)
- [dependencies.md](dependencies.md)
- [packaging.md](packaging.md)

## Cross-Cutting Features and APIs

- Benchmark JSON shape is emitted by `benchmarks/output.ml`, then consumed by `benchmarks/to_web.ml`, `benchmarks/to_csv.ml`, `.github/workflows/deploy-pr-preview.yml`, and `gh-pages/javascripts/benchmark-viewer.js`.
- Optional GPU backend support in support tools is built through Dune `select` stubs in `benchmarks/dune` and `tools/dune`.
- CI uses Dockerized opam builds for core tests and a separate generated-code freshness job.
- GitHub Pages combines Jekyll docs, static benchmark data, generated benchmark descriptions, legacy notebooks, and generated odoc under `spoc_docs/`.
- Packaging is split across generated opam files, top-level Dune package selection, Make targets, and container images.

## Cross-Cutting Invariants

- Support artifacts should not require GPU drivers for basic build/test paths; optional backends are selected or no-op stubs are used in `benchmarks/dune:25-42` and `tools/dune:15-34`.
- Benchmark result JSON must preserve `benchmark.name`, `benchmark.parameters.size`, `system.hostname`, and per-device `framework`/timing fields because the dashboard and converters depend on them (`benchmarks/output.ml:70-88`, `gh-pages/javascripts/benchmark-viewer.js:974-983`).
- Generated benchmark descriptions should be deterministic; CI rebuilds `benchmarks/generate_backend_code.exe` and fails if `benchmarks/descriptions/generated/` changes (`.github/workflows/ci.yml:150-188`).
- Repository checks should not unexpectedly modify source unless explicitly named as fixers.
- Vendored/static assets should carry clear provenance and license metadata.

## Potential Invariant Violations or Bugs

- **fixed 2026-07-02 (merged)** — the license checker previously mutated the working tree before deciding whether headers were current (`scripts/check-license-headers.sh` ran `add-license-headers.sh`, then checked `git diff`). Commit `5bad179d` gave `add-license-headers.sh` a `--check` dry-run mode (temp-copy diff, never touches the real file) and `check-license-headers.sh` now delegates to it — confirmed non-mutating in current source. See `kb/support/scripts-ci.md` for the full evidence, including the related case-sensitive-email-grep root-cause fix.
- **fixed 2026-07-02 (merged), `benchmark-viewer.js` only:** GitHub Pages benchmark rendering previously trusted data and markdown via `innerHTML` with no escaping. `gh-pages/javascripts/benchmark-viewer.js` now defines `escapeHtml()` (`:45`) and `markdownToHtml()` escapes its input first (`:546-547`, comment: "escapeHtml() ... every substitution below runs on top of that"); links are restricted to http(s)-only (`:543-583`, URL scheme check before rendering as a real `<a>`, else rendered as plain escaped text); system/device fields are now wrapped in `escapeHtml(...)` before interpolation (`:1083-1101`, `:1835`). **Correction, do not conflate with the fix above:** `gh-pages/javascripts/benchmark-dashboard.js` is a **separate, older four-chart dashboard file** loaded by `gh-pages/benchmarks/dashboard.md:210` (`<script src="../javascripts/benchmark-dashboard.js">`). It was **not touched by this fix** — grepping it finds zero `escapeHtml` calls against 12 raw `innerHTML` writes (`benchmark-dashboard.js:71,126,156,170,274,291,381,393,481,497,719,726`). Its XSS sinks remain unfixed and were out of scope for PR #213/#214. Previously this KB (and the general commentary elsewhere in this file) treated `benchmark-dashboard.js` as a candidate for deletion/archival on the assumption it might be dead code; that assumption needs revision — see the orphan-page note below. **Do not mark the dashboard.js sinks as fixed.**
- **New finding (2026-07-02): `benchmark-dashboard.js` is live but orphaned, not dead code.** `gh-pages/benchmarks/dashboard.md` builds a real, URL-addressable Jekyll page (`/benchmarks/dashboard.html` once built) that loads `benchmark-dashboard.js` directly. No file under `gh-pages/_layouts/`, `gh-pages/_config.yml`, or any `.md` page's nav/links was found linking to `dashboard.md`/`dashboard.html` — so the page is reachable by direct URL but not discoverable through normal site navigation. This means it is simultaneously (a) a real attack surface (unescaped `innerHTML` sinks, reachable by anyone who knows or guesses the URL) and (b) not exercised by normal users, which is likely why its XSS gap was not prioritized in #213/#214 alongside the canonical `benchmark-viewer.js` fix. **Open question for a human:** should this orphan page be linked into site navigation (raising its priority for the XSS fix), or removed/archived (eliminating the live-but-unfixed attack surface entirely)? This KB does not recommend one over the other — it is flagging the fact that the page is neither fully dead nor fully maintained, which is a state that should be resolved deliberately rather than left ambiguous.
- Vendored CUDA headers contain restrictive NVIDIA copyright text (`dependencies/Cuda/nvrtc.h:4-10`, `dependencies/Cuda/host_defines.h:2-17`), while `dependencies/Cuda/LICENCE` is GPLv3 text. OpenCL headers carry Khronos permissive notices (`dependencies/CL/opencl.h:2-17`), while `dependencies/CL/LICENCE` is also GPLv3 text. The license files appear mismatched or incomplete.
- The interactive Docker image disables Jupyter token/password authentication by default (`Dockerfile:76`). This is acceptable only for local/Binder-style isolated use, not exposed multi-user hosts.
- GitHub Actions are tag-pinned rather than SHA-pinned (`.github/workflows/ci.yml:25-46`, `.github/workflows/deploy-pr-preview.yml:23-88`, `.github/workflows/docs.yml:16-70`), leaving normal third-party action supply-chain drift risk.

## Performance and Maintainability Risks

- Benchmark code mixes shared runner style with older per-benchmark local runner logic, causing duplicated CLI parsing, output defaults, and statistics choices.
- `gh-pages/javascripts/benchmark-viewer.js` duplicates benchmark metadata and chart configuration across large objects and rendering paths, raising drift risk when adding workloads.
- CI image setup repeatedly installs opam packages inside workflow runs; caches help, but the scripts still hit external package sources and alpha repositories.
- Legacy notebooks and old static site assets mix executable historical WebSPOC examples with modern Sarek docs; the site needs clear static/archive boundaries.

## Related Tests and Checks

- CI build/test: `.github/workflows/ci.yml`.
- Generated-code freshness: `.github/workflows/ci.yml:117-191`.
- Coverage helpers: `scripts/coverage-unit.sh`, `scripts/coverage-e2e.sh`, `scripts/coverage-benchmarks.sh`, `scripts/coverage-aggregate.sh`.
- Benchmark deduplication/check command: `Makefile:396-404`, `benchmarks/deduplicate_results.ml`.
- Docs build/deploy: `.github/workflows/docs.yml`.

## Missing Tests

- No automated tests for benchmark JSON schema compatibility across `output.ml`, `to_web.ml`, `to_csv.ml`, PR comments, and browser dashboard code.
- No sanitizer tests for benchmark descriptions, generated code tabs, benchmark JSON system/device fields, or dashboard matrix rows.
- No check that static/vendored assets have correct source URLs, versions, checksums, and matching license files.
- ~~No CI test for `scripts/check-license-headers.sh` preserving a clean worktree.~~ **Largely moot 2026-07-02** — the script no longer mutates the tree at all; see `kb/support/scripts-ci.md`.
- No workflow policy test for action pinning, least-privilege permissions, or PR-preview trust boundaries.

## Concrete Improvement Candidates

- Add a JSON schema fixture test for benchmark files and web `latest.json`.
- ~~Replace ad hoc markdown conversion and string-built HTML with a sanitizer or DOM node construction for untrusted fields.~~ **DONE 2026-07-02 for `benchmark-viewer.js` only** — `escapeHtml()` plus an escape-first `markdownToHtml()` and http(s)-only href gating. **Still open for `benchmark-dashboard.js`** — the orphaned dashboard page's 12 raw `innerHTML` sinks are unfixed; see the orphan-page note above.
- ~~Split `check-license-headers.sh` into a true dry-run check and a separate fixer invocation.~~ **DONE 2026-07-02.**
- Add `THIRD_PARTY.md` or per-vendor metadata with source URL, version/date, checksum, and license for `dependencies/**` and `gh-pages/static/**`.
- Pin GitHub Actions by SHA or document an accepted update cadence.
- Make `Dockerfile` authentication defaults explicit through build args/env vars instead of hard-coded empty Jupyter credentials.
