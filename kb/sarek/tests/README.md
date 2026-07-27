# Sarek Tests Knowledge Base

## Component Inventory

Top-level Sarek tests live in `sarek/tests/` and are arranged by `sarek/tests/dune:1` (`common` was removed from that `(dirs ...)` stanza on 2026-07-25, task #78, when the orphan `sarek_test_common` library was deleted; re-verified):

- `sarek/tests/unit/`
- `sarek/tests/e2e/`
- `sarek/tests/negative/`
- `sarek/tests/native/`
- `sarek/tests/new_runtime/`
- `sarek/tests/codegen_golden/`

Default `runtest` depends on the `unit`, `e2e`, and `codegen_golden` aliases, plus `../core/test`'s `runtest` alias, in `sarek/tests/dune:1-15`. `negative` and `new_runtime` are not in this default set (see below).

## Per-File Purpose

- `sarek/tests/dune`: top-level test directory declaration and default test alias (`sarek/tests/dune:1-15`).
- (Deleted 2026-07-25, task #78: the whole `sarek/tests/common/` directory — `dune`, `Test_kernels.ml`, `Ir_compare.ml`, `Source_gen.ml`. It declared library `sarek_test_common`, which had zero consumers repo-wide, and its `Ir_compare.ml` compared the now-deleted legacy `Kirc_Ast` trees. `kb/sarek/tests/common.md` was deleted with it.)
- `sarek/tests/unit/dune`: unit test executable list and libraries.
- `sarek/tests/e2e/dune`: e2e helper libraries, backend selection, executable list, and e2e alias.
- `sarek/tests/negative/dune`: profile-gated compile-failure cases and expected error comments.
- `sarek/tests/new_runtime/dune`: native runtime and GPU/runtime comparison executables.
- `sarek/tests/native/dune`: placeholder; no active executables.
- `sarek/tests/codegen_golden/dune`: golden-snapshot harness for the 5 pure-OCaml codegen backends (CUDA/OpenCL/Metal/GLSL/WGSL); builds `test_codegen_golden` (1,444 lines) against `sarek_ir`/`sarek_codegen`, no FFI. In the default `runtest` alias.

## Features And APIs

- Unit tests link `sarek_ppx_lib`, `sarek_stdlib`, `sarek`, `alcotest`, `ppxlib`, and `str` in `sarek/tests/unit/dune:28-30` (re-verified 2026-07-25; the range shifted by one line when `test_lower` was dropped from the `(names ...)` list, task #78).
- E2E tests use optional backend selection via Dune `(select ...)` in `sarek/tests/e2e/dune:18-54`.
- Backend disabling env vars are documented in `sarek/tests/e2e/dune:4-9`.
- Negative tests are enabled only through profile `negative` in `sarek/tests/negative/dune:4-6`.
- New runtime comparison is CUDA-gated by `CUDA_PATH` in `sarek/tests/new_runtime/dune:17-20`.

## Invariants

- Default tests should keep unit and e2e suites passing without requiring a GPU. **Precision note (2026-07-02 audit):** this holds only in a compile-only sense for e2e — `dune runtest`'s `e2e/runtest` alias (`sarek/tests/e2e/dune:147-181`) lists 32 `.exe` files as alias *dependencies* with no `run` action, so `dune runtest` builds all 32 and executes none of them. The only e2e binaries actually *executed* anywhere are the 5 run by `make e2e-fast` in CI (`Makefile:247-262`, `.github/workflows/ci.yml:93-104`). Treat "e2e suite passing" as "e2e suite compiles" unless `make e2e-fast` (or a manual `dune exec`) was also run.
- Negative tests should fail for the expected compiler reason, not any arbitrary compile error.
- E2E backend selection should degrade cleanly when CUDA/OpenCL/Vulkan/Metal libraries are unavailable.
- Shared test helpers should not drift from active PPX syntax and runtime APIs. (The one shared helper library, `sarek_test_common`, had drifted to the point of having no consumers and was deleted on 2026-07-25, task #78; no shared test-helper library exists today.)

## Potential Invariant Violations Or Bugs

- Confirmed (known issue, headline finding of the 2026-07-02 audit): the e2e `runtest` alias (`sarek/tests/e2e/dune:147-181`) is build-only — it depends on 32 `.exe` targets with no `(action (run ...))`, so `dune runtest` compiles but never runs any of them. CI's actual e2e execution is `make e2e-fast`, which runs only 5 tests (`test_vector_add`, `test_matrix_mul`, `test_reduce`, `test_transpose`, `test_math_intrinsics`; `Makefile:247-262`). `test_scan`, `test_sort`, `test_histogram`, `test_convolution`, `test_stencil`, `test_nbody_ppx`, `test_ray_ppx`, and the rest of the 32 never execute in CI.
- Confirmed: `sarek/tests/e2e/dune:68-69`, `sarek/tests/e2e/dune:105-106`, and `sarek/tests/e2e/dune:160-161` disable cross-module type and registered-constant e2e tests due to PPX registration issues.
- Stale (corrected 2026-07-02): the negative suite is not merely comment-documented — `make test_negative` (`Makefile:82-95`) does grep the built output for the exact expected message on 6 of the 8 documented cases (all but `test_warp_diverged` and `test_convention_kernel_fail`, see `kb/sarek/tests/negative.md`). The real gap is narrower: those 2 of 8 cases are never invoked by any Makefile target, and the whole negative suite (`--profile=negative`) is absent from CI (`.github/workflows/ci.yml` has no negative/`test_negative` step).
- Confirmed: `sarek/tests/new_runtime/` is not wired into top-level default `runtest` in `sarek/tests/dune:1-15`.
- Confirmed: `sarek/tests/native/dune:1-2` has no active executables.

## Performance Or Maintainability Risks

- E2E runtime cost is high and backend-dependent; failures may be environment-specific.
- Disabled e2e tests can hide regressions in registration semantics.
- (Retired 2026-07-25, task #78: "common neutral-kernel generators still mention old camlp4 syntax and may become stale if not consumed regularly" — the generators were never consumed and are deleted.)
- Negative compile tests that only expect failure can pass for the wrong reason (still true for the 2 of 8 cases that assert nothing — see `kb/sarek/tests/negative.md`).
- The e2e `runtest` alias's build-only nature (above) means CI green on `dune runtest` says nothing about runtime correctness for 27 of the 32 aliased e2e executables.

## Related Tests

- Unit suite: `kb/sarek/tests/unit.md`.
- E2E suite: `kb/sarek/tests/e2e.md`.
- Negative suite: `kb/sarek/tests/negative.md`.
- New runtime tests: `kb/sarek/tests/new_runtime.md`.
- Native directory status: `kb/sarek/tests/native.md`.
- Codegen golden-snapshot suite: documented inline above and in `kb/sarek/tests/unit.md` (sibling `test_ptx_snapshot`); no dedicated file yet.

## Missing Tests

- Top-level alias or CI job for `sarek/tests/new_runtime`.
- Exact-output negative test harness for `test_warp_diverged` and `test_convention_kernel_fail` (the 2 of 8 cases with no grep check anywhere), and CI wiring for the negative suite as a whole.
- Reactivation or replacement of disabled registration e2e tests.
- Native directory tests or removal of the placeholder.
- Actual `run` actions on the e2e `runtest` alias so `dune runtest` executes what it builds.

## Concrete Improvement/Fix Candidates

- Add cram-style or Dune action tests for the 2 negative cases that currently assert nothing, and wire the negative profile into CI.
- Wire `new_runtime` into CI under a feature/environment gate.
- Track disabled e2e tests with issues and add reduced unit tests for their root causes.
- (Done 2026-07-25, task #78: "retire stale common old-syntax generation if no active test consumes it" — no test consumed it, so `sarek/tests/common/` was deleted outright rather than rewired.)
- Add `(action (run %{exe:...}))` (or equivalent) to the e2e `runtest` alias deps so it stops being build-only.
