# Packaging and Local Development Support

## Component Inventory

- Root `dune`: top-level Dune directory selection.
- Root opam files: `spoc.opam`, `sarek.opam`, `sarek-cuda.opam`, `sarek-opencl.opam`, `sarek-vulkan.opam`, `sarek-metal.opam`.
- `Makefile`: local build/test/benchmark/release task wrapper.
- `Dockerfile`: interactive Jupyter/Binder image.
- `ci/Dockerfile`: CI image, also covered in [scripts-ci.md](scripts-ci.md).
- `README.md`: top-level architecture, install, verification, testing, and docs guidance.

## Per-File Purpose

- `dune`: includes package directories `spoc`, `sarek`, backend packages, `tools`, `test_verification`, and `benchmarks`.
- `spoc.opam`: SDK/plugin-interface package metadata and deps.
- `sarek.opam`: full DSL/runtime package metadata and deps.
- `sarek-cuda.opam`, `sarek-opencl.opam`, `sarek-vulkan.opam`, `sarek-metal.opam`: optional backend package metadata and deps/depopts.
- `Makefile`: wraps Dune builds, test tiers, coverage, release, benchmarks, preview, and generated-code commands.
- `Dockerfile`: builds an OCaml 5.4 JupyterLab image, installs opam dependencies, installs Sarek, configures OCaml Jupyter kernel, and starts JupyterLab.
- `README.md`: user-facing feature, build, benchmark, docs, troubleshooting, and project-history overview.

## Features and APIs

- Packages require OCaml 5.4.0+ and Dune 3.15+ in opam metadata (`spoc.opam:15-20`, `sarek.opam:16-26`, `sarek-cuda.opam:11-19`, `sarek-opencl.opam:12-20`, `sarek-vulkan.opam:12-20`, `sarek-metal.opam:12-18`).
- Backend packages depend on `sarek` and use optional `conf-*` depopts for CUDA/OpenCL/Vulkan (`sarek-cuda.opam:19`, `sarek-opencl.opam:20`, `sarek-vulkan.opam:20`).
- Docker image installs JupyterLab/Thebe, opam deps, builds and installs the project, then starts Jupyter (`Dockerfile:5-31`, `Dockerfile:74-76`).
- Make benchmark targets build/run standalone benchmark tooling and update `gh-pages/benchmarks/data/latest.json` (`Makefile:389-437`).

## Invariants

- Root opam files are generated from Dune project metadata and should not be hand-edited except through the generator.
- Top-level Dune directories must exist or be intentionally tolerated by Dune.
- `Makefile` targets should not unexpectedly append to generated opam metadata.
- Docker image defaults should be safe for the deployment mode they target.
- README commands should match actual Make targets and public executables.

## Potential Invariant Violations or Bugs

- `Makefile` target `opam` appends `available: [ os = "linux" ]` to generated opam files (`Makefile:21-24`). Repeated runs duplicate lines, and `sarek_ppx.opam` is referenced even though it was not present in the root file inventory.
- Root `dune` lists `test_verification` (`dune:8-10`); this directory was not present in the initial support-scope directory listing. Marked uncertain because Dune may tolerate absent dirs differently by version/config, but it is suspicious.
- `Dockerfile` starts Jupyter with empty token and password (`Dockerfile:76`). Safe for ephemeral local/Binder contexts only.
- `Dockerfile` uses unpinned `pip3 install jupyterlab thebe` (`Dockerfile:9`) and unpinned opam installs (`Dockerfile:22-26`), so interactive image rebuilds are not reproducible.
- `sarek-metal.opam` has a redundant/odd Dune constraint `{>= "3.15" & >= "2.9"}` (`sarek-metal.opam:13-14`), unlike other backend opam files.
- README says `make benchmarks-fast` verifies installation (`README.md:155-168`), but no `benchmarks-fast` target appeared in the reviewed `Makefile`.

## Performance and Maintainability Risks

- Large Makefile mixes modern Sarek tests, legacy sample targets, backend-specific tiers, benchmarks, docs preview, and release commands; target drift is likely.
- Docker image copies the whole repository before build (`Dockerfile:29`), reducing layer-cache efficiency for source changes.
- Unpinned external package installs in Docker and CI make rebuilds time-dependent.
- Generated opam files are edited by Make rather than by Dune project configuration, undermining reproducibility.

## Related Tests and Checks

- `make test`, `make test-all`, `make e2e-fast`, `make bench-deduplicate`, and CI build paths cover parts of packaging behavior.
- `ghcr-image.yml` builds and pushes the interactive Docker image on manual trigger or package/Dockerfile changes (`.github/workflows/ghcr-image.yml:3-36`).
- README verification path includes `dune exec -- sarek-device-info`, `dune runtest`, and benchmark commands (`README.md:155-172`).

## Missing Tests

- Check that every README command and Make target exists.
- Check that `make opam` is idempotent or remove it.
- Docker build smoke test with pinned dependencies or lockfiles.
- Opam lint across all generated opam files.
- Packaging test for optional backend absent/present combinations.

## Concrete Improvement Candidates

- Move opam availability constraints into Dune project metadata and remove mutating `Makefile:21-24`.
- Add `make help` output generated from actual targets and test README command snippets.
- Make Jupyter token/password configurable and default to secure credentials outside Binder.
- Pin Docker Python/opam dependencies or record a lockfile/switch export for the interactive image.
- Normalize opam Dune constraints and add an opam lint target in CI.
