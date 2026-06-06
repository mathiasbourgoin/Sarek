# Sarek Tests Knowledge Base

## Component Inventory

Top-level Sarek tests live in `sarek/tests/` and are arranged by `sarek/tests/dune:1-9`:

- `sarek/tests/common/`
- `sarek/tests/unit/`
- `sarek/tests/e2e/`
- `sarek/tests/negative/`
- `sarek/tests/native/`
- `sarek/tests/new_runtime/`

Default `runtest` depends only on unit and e2e aliases in `sarek/tests/dune:3-9`.

## Per-File Purpose

- `sarek/tests/dune`: top-level test directory declaration and default test alias.
- `sarek/tests/common/dune`: declares `sarek_test_common`.
- `sarek/tests/common/Test_kernels.ml`: neutral kernel descriptions for old/new syntax comparison.
- `sarek/tests/common/Ir_compare.ml`: structural diff helpers for legacy `Kirc_Ast`.
- `sarek/tests/common/Source_gen.ml`: old camlp4 and new PPX source generation from neutral kernel descriptions.
- `sarek/tests/unit/dune`: unit test executable list and libraries.
- `sarek/tests/e2e/dune`: e2e helper libraries, backend selection, executable list, and e2e alias.
- `sarek/tests/negative/dune`: profile-gated compile-failure cases and expected error comments.
- `sarek/tests/new_runtime/dune`: native runtime and GPU/runtime comparison executables.
- `sarek/tests/native/dune`: placeholder; no active executables.

## Features And APIs

- Unit tests link `sarek_ppx_lib`, `sarek_stdlib`, `sarek`, `alcotest`, `ppxlib`, and `str` in `sarek/tests/unit/dune:29-32`.
- E2E tests use optional backend selection via Dune `(select ...)` in `sarek/tests/e2e/dune:18-54`.
- Backend disabling env vars are documented in `sarek/tests/e2e/dune:4-9`.
- Negative tests are enabled only through profile `negative` in `sarek/tests/negative/dune:4-6`.
- New runtime comparison is CUDA-gated by `CUDA_PATH` in `sarek/tests/new_runtime/dune:17-20`.

## Invariants

- Default tests should keep unit and e2e suites passing without requiring a GPU.
- Negative tests should fail for the expected compiler reason, not any arbitrary compile error.
- E2E backend selection should degrade cleanly when CUDA/OpenCL/Vulkan/Metal libraries are unavailable.
- Shared test helpers should not drift from active PPX syntax and runtime APIs.

## Potential Invariant Violations Or Bugs

- Confirmed: `sarek/tests/e2e/dune:68-69`, `sarek/tests/e2e/dune:105-106`, and `sarek/tests/e2e/dune:160-161` disable cross-module type and registered-constant e2e tests due to PPX registration issues.
- Confirmed: negative tests document expected messages in comments at `sarek/tests/negative/dune:8-16`, but the Dune file only gates libraries by profile at `sarek/tests/negative/dune:18-104`; there is no local assertion of exact stderr text.
- Confirmed: `sarek/tests/new_runtime/` is not wired into top-level default `runtest` in `sarek/tests/dune:3-9`.
- Confirmed: `sarek/tests/native/dune:1-2` has no active executables.

## Performance Or Maintainability Risks

- E2E runtime cost is high and backend-dependent; failures may be environment-specific.
- Disabled e2e tests can hide regressions in registration semantics.
- Common neutral-kernel generators still mention old camlp4 syntax and may become stale if not consumed regularly.
- Negative compile tests that only expect failure can pass for the wrong reason.

## Related Tests

- Unit suite: `kb/sarek/tests/unit.md`.
- E2E suite: `kb/sarek/tests/e2e.md`.
- Negative suite: `kb/sarek/tests/negative.md`.
- Common helpers: `kb/sarek/tests/common.md`.
- New runtime tests: `kb/sarek/tests/new_runtime.md`.
- Native directory status: `kb/sarek/tests/native.md`.

## Missing Tests

- Top-level alias or CI job for `sarek/tests/new_runtime`.
- Exact-output negative test harness.
- Reactivation or replacement of disabled registration e2e tests.
- Native directory tests or removal of the placeholder.

## Concrete Improvement/Fix Candidates

- Add cram-style or Dune action tests for negative cases that assert expected messages.
- Wire `new_runtime` into CI under a feature/environment gate.
- Track disabled e2e tests with issues and add reduced unit tests for their root causes.
- Retire stale common old-syntax generation if no active test consumes it.
