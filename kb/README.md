# SPOC/Sarek Knowledge Base

This knowledge base was built from a clean, up-to-date `main` branch at commit `6242463`.
It was updated after the runtime fixes merged through `06b7d70` on `main`.

The repo is an OCaml/Dune monorepo for the SPOC SDK and Sarek GPU-kernel DSL. It includes the low-level framework interfaces, Sarek runtime and PPX compiler, CPU and GPU backend plugins, benchmark tooling, CI/docs/site support, and vendored C headers/static assets.

## Review Scope

Reviewed source and support components:

- `spoc/`: SDK framework types, IR, registry, and tests.
- `sarek/core/`, `sarek/framework/`, `sarek/sarek/`, `sarek/plugins/`, `sarek/Sarek_stdlib/`, `sarek/Sarek_float64/`, `sarek/Sarek_geometry/`, `sarek/Visibility_lib/`: runtime, CPU/interpreter/native execution, registries, support libraries, and colocated tests.
- `sarek/ppx/`, `sarek/ppx_intrinsic/`, `sarek/tests/`: PPX compiler, intrinsic-authoring PPX, unit/e2e/negative/new-runtime test suites.
- `sarek-cuda/`, `sarek-opencl/`, `sarek-vulkan/`, `sarek-metal/`: optional GPU backend packages and tests.
- `benchmarks/`, `tools/`, `scripts/`, `ci/`, `docker_scripts/`, `.github/`, `gh-pages/`, `docs/`, `notebooks/`, `dependencies/`, root packaging files, and top-level docs.

Large vendored/static assets under `gh-pages/static/**`, `gh-pages/pres_resources/**`, `gh-pages/docs/talks/**`, and C headers under `dependencies/**` are inventoried with license/provenance notes. Minified/vendor JS/CSS/font/binary assets were not treated as first-party source for semantic bug review.

## KB Map

- [spoc/](spoc/README.md): framework, IR, registry, and SPOC tests.
- [sarek/runtime/](sarek/runtime/README.md): core runtime, execution, CPU/interpreter/native plugins, fusion, framework registry, stdlib/support modules.
- [sarek/ppx/](sarek/ppx/README.md): parser, type system, lowering, quote generation, native generation, tail recursion, convergence, and intrinsic PPX.
- [sarek/tests/](sarek/tests/README.md): common helpers, unit tests, e2e tests, negative compile tests, new runtime tests, native placeholder.
- [backends/](backends/README.md): CUDA, OpenCL, Vulkan, Metal, and shared backend patterns.
- [support/](support/README.md): benchmarks, tools, scripts/CI, docs/site, dependencies, packaging.

## Component Inventory

### SDK Layer

- `spoc/framework/`: backend contracts, device records, launch dimensions, typed values, shared backend errors.
- `spoc/ir/`: pure kernel IR, pretty-printer, float64 usage analysis.
- `spoc/registry/`: process-global type/record/variant/function registry.

### Runtime Layer

- `sarek/core/`: device discovery, vectors, transfer state, memory, kernels, profiling, logging.
- `sarek/framework/`: backend registry, intrinsic registry, framework cache, framework errors.
- `sarek/sarek/`: execution dispatcher, KIRC/Sarek IR wrappers, CPU runtime, interpreter, fusion.
- `sarek/plugins/native/`: native CPU backend plugin.
- `sarek/plugins/interpreter/`: sequential/debug interpreter backend plugin.
- `sarek/Sarek_stdlib/`, `sarek/Sarek_float64/`, `sarek/Sarek_geometry/`, `sarek/Visibility_lib/`: kernel stdlib and support packages.

### Compiler and Tests

- `sarek/ppx/`: PPX pipeline from OCaml AST to typed/lowered Sarek IR and runtime wrappers.
- `sarek/ppx_intrinsic/`: helper PPX for intrinsic definitions/extensions.
- `sarek/tests/`: unit, e2e, negative, common, native, and new-runtime test suites.

### Backends

- `sarek-cuda/`: CUDA C generation, NVRTC, CUDA Driver API, plugin.
- `sarek-opencl/`: OpenCL C generation, OpenCL runtime API, plugin.
- `sarek-vulkan/`: GLSL/SPIR-V generation, Vulkan API, Shaderc/glslang paths, plugin.
- `sarek-metal/`: Metal Shading Language generation, Metal API, plugin.

### Support Surface

- `benchmarks/`: benchmark workloads, JSON/CSV/web conversion, generated backend-code snippets.
- `tools/`: optional-backend init and `sarek-device-info`.
- `scripts/`, `ci/`, `docker_scripts/`, `.github/workflows/`: coverage, license, Docker, CI, docs deploy, PR preview, GHCR, PR retarget.
- `gh-pages/`, `docs/`, `notebooks/`: Jekyll site, benchmark dashboard, legacy notebooks/talk assets, odoc styling.
- `dependencies/`: vendored OpenCL/CUDA headers and license files.
- Root `dune`, `*.opam`, `Makefile`, `Dockerfile`, `README.md`, `CONTRIBUTING.md`, `COVERAGE.md`: packaging and developer entry points.

## Repo-Wide Invariants

- Basic builds/tests should not require GPU drivers; optional GPU backends are selected or stubbed through Dune `select`.
- Backend `Kernel.args` and wrapper APIs must preserve argument order, explicit indices, types, sizes, and ownership.
- Vector transfer state must always identify the authoritative host/device copy before reads, writes, frees, and cross-device moves.
- Kernel IR and PPX type information must preserve memory space, scalar width, record/variant layout, convergence safety, and runtime registration data.
- Runtime registries and backend caches must be deterministic under repeated registration, test execution, plugin loading, and concurrent use, or explicitly documented otherwise.
- Benchmark JSON must stay compatible across producers, converters, PR previews, and browser dashboard code.
- Vendored/static assets need clear provenance and license metadata.

## Highest-Priority Findings

- PPX/compiler type handling has confirmed drift for bare `float`, and array unification ignores memory space; see [ppx/README.md](sarek/ppx/README.md).
- Process-global registries silently overwrite entries and short-name lookup can be nondeterministic; see [spoc/registry.md](spoc/registry.md).
- GPU backend argument binding and buffer-copy validation are inconsistent across backends; see [backends/README.md](backends/README.md).
- Framework cache accepts arbitrary key strings as path components; see [runtime/framework.md](sarek/runtime/framework.md).
- Benchmark output schema and statistics are inconsistent across workloads; see [support/benchmarks.md](support/benchmarks.md).
- The benchmark dashboard writes unsanitized markdown/JSON-derived fields through `innerHTML`; see [support/docs-site.md](support/docs-site.md).
- Vendored CUDA/OpenCL license metadata appears mismatched or incomplete; see [support/dependencies.md](support/dependencies.md).

## Recently Merged Fixes

- Runtime transfer-state preservation was fixed by PR #136, merged as `5dffea3` on 2026-05-05. The fix covers cross-device movement from authoritative device data and preserves authoritative device data during buffer cleanup.
- CPU barrier auto-detection side effects were fixed by PR #137, merged as `d30b2ba` on 2026-05-05. The runtime now avoids speculative user-kernel execution when barrier metadata is absent and propagates worker failures in the reviewed path.
- Fusion pipeline preservation was fixed by PR #138, merged as `06b7d70` on 2026-05-05. The fusion APIs now preserve unfused kernels through list-returning pipeline APIs and reject mismatched one-to-one index fusion.
- The CI breakage found while merging these PRs was fixed in PR #136 and then rebased out of PRs #137/#138. The mandatory build no longer depends on the removed `bisect_ppx.2.8.3.1~alpha-repo` package.

## Recommended Next Work

1. Add focused regression tests for the PPX `float` and array-memory-space findings before refactoring.
2. Add backend argument-order/size-validation tests shared across CUDA/OpenCL/Vulkan/Metal.
3. Replace silent registry overwrite and ambiguous short-name lookup with duplicate-aware APIs.
4. Add benchmark JSON schema fixtures that exercise producers, converters, PR preview summaries, and the dashboard.
5. Split mutating license-header fixes from read-only checks, and add third-party provenance metadata for vendored/static assets.

## Verification Notes

- The pre-review dirty worktree was preserved in `stash@{0}` as `pre-review-clean-main-2026-05-05`.
- The repository was fast-forwarded to `origin/main` before KB work started.
- Review changes are confined to `kb/`.
- The repo requires OCaml 5.4.0+; validation should use the repo-local switch, for example `opam exec --switch=/home/mathias/dev/SPOC -- ...`, rather than an ambient global switch.
- Tests/builds were not run for the KB-only Markdown additions; component workers did not modify source files.
- PRs #136, #137, and #138 were reviewed, cleaned to one commit each before merge, and merged one after another with rebase merges. CI was green before each merge.
