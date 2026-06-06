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

- The license checker mutates the working tree before deciding whether headers are current: `scripts/check-license-headers.sh:23-40` runs `add-license-headers.sh`, then checks `git diff`. This is surprising for a command named `check`.
- GitHub Pages benchmark rendering trusts data and markdown via `innerHTML`. Markdown conversion writes unsanitized headings, inline code, links, images, tables, and lists (`gh-pages/javascripts/benchmark-viewer.js:507-558`) into the page (`gh-pages/javascripts/benchmark-viewer.js:491`), and system/device fields from benchmark JSON are interpolated into HTML (`gh-pages/javascripts/benchmark-viewer.js:1054-1067`, `gh-pages/javascripts/benchmark-viewer.js:1837`). This is a stored-XSS risk if benchmark JSON or description markdown can be modified by an untrusted contributor.
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
- No CI test for `scripts/check-license-headers.sh` preserving a clean worktree.
- No workflow policy test for action pinning, least-privilege permissions, or PR-preview trust boundaries.

## Concrete Improvement Candidates

- Add a JSON schema fixture test for benchmark files and web `latest.json`.
- Replace ad hoc markdown conversion and string-built HTML with a sanitizer or DOM node construction for untrusted fields.
- Split `check-license-headers.sh` into a true dry-run check and a separate fixer invocation.
- Add `THIRD_PARTY.md` or per-vendor metadata with source URL, version/date, checksum, and license for `dependencies/**` and `gh-pages/static/**`.
- Pin GitHub Actions by SHA or document an accepted update cadence.
- Make `Dockerfile` authentication defaults explicit through build args/env vars instead of hard-coded empty Jupyter credentials.
