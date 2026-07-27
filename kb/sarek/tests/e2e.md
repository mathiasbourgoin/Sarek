# Sarek E2E Tests

## Component Inventory

E2E tests live in `sarek/tests/e2e/` and are configured by `sarek/tests/e2e/dune`.

## Per-File Purpose

- `sarek/tests/e2e/dune`: declares helper libraries, optional backend modules, PPX-preprocessed executable list, and e2e alias.
- `Benchmarks.ml`, `test_helpers.ml`, `backend_loader.ml`, `backend_*.ml`: device/backend selection, benchmark running, and verification utilities.
- `registered_defs.ml`: shared registered definitions for PPX tests.
- `test_vector_add.ml`, `test_module_const.ml`, `test_ktype_*`, `test_klet_*`: basic PPX kernel/type/module behavior.
- `test_registered_type.ml`, `test_registered_variant.ml`, `test_visibility_private.ml`: registration and visibility behavior.
- `test_convention*.ml`, `test_pragma.ml`, `test_barrier_converged.ml`, `test_superstep.ml`: compiler contracts and control-flow features.
- `test_stencil.ml`, `test_matrix_mul.ml`, `test_reduce.ml`, `test_scan.ml`, `test_transpose.ml`, `test_sort.ml`, `test_histogram.ml`, `test_convolution.ml`, `test_mandelbrot.ml`: algorithmic kernels.
- `test_math_intrinsics.ml`, `test_bitwise_ops.ml`: intrinsic/operator behavior.
- `test_polymorphism.ml`, `test_module_poly.ml`, `test_bounded_recursion.ml`, `test_inline_pragma.ml`, `test_nested_types.ml`: advanced compiler features.
- `test_debug_native.ml`, `test_external_kernel.ml`: auxiliary/native/external-kernel executables.
- `test_fusion_api.ml`: fusion-API e2e coverage; declared in the main executable/modules stanzas and included in the `runtest` alias (`sarek/tests/e2e/dune:76,113,168`).
- `test_float32_sin_pure.ml`: PR-2 Float32.sin pure-registry GPU e2e proof; declared separately, not preprocessed by `sarek_ppx`, not in the `runtest` alias — run manually via `dune exec ... -- --vulkan` (`sarek/tests/e2e/dune:190-197`).
- `test_float64_math_intrinsics.ml`: fp64 math-intrinsic parity report across every fp64-capable device; **report-only by design** — a mismatch on any (device, function) pair does not fail the test or block CI (documented in the module and in the dune comment); always exits 0; not in the `runtest` alias (`sarek/tests/e2e/dune:199-208`).
- `test_stdlib_meta_proof.ml`: PR-5a proof that `sarek_stdlib_meta` populates `Sarek_ppx_registry` without any Ctypes/`spoc_core` dependency; links only `sarek_frontend`/`sarek_stdlib_meta`; not in the `runtest` alias (`sarek/tests/e2e/dune:218-226`).

## Features And APIs

- Optional GPU backends use Dune `(select ...)` in `sarek/tests/e2e/dune:34-53`.
- The main PPX executable set is declared in `sarek/tests/e2e/dune:56-145`.
- The e2e alias depends on a subset of executables in `sarek/tests/e2e/dune:147-181`.
- `test_debug_native` is declared separately at `sarek/tests/e2e/dune:183-188`.
- `test_external_kernel` is optional and not in the main alias at `sarek/tests/e2e/dune:190-196`.

## Invariants

- E2E tests should run on native/interpreter paths and use GPU backends only when available.
- Backend filtering should respect `SPOC_DISABLE_*` environment variables documented in `sarek/tests/e2e/dune:4-9`.
- Executables listed in the main stanza should either be part of `runtest` or intentionally excluded.
- Disabled tests should have tracked replacement coverage.

## Potential Invariant Violations Or Bugs

- **fixed 2026-07-02 (merged)** — the `runtest`-alias-executes-nothing headline finding is closed. `sarek/tests/e2e/dune` no longer has a bare `(alias (name runtest) (deps test_X.exe ...))` for the executable set; each test now has its own `(rule (alias runtest) (action (run %{dep:test_X.exe})))` stanza (commit `ebbc0b4b`, "make the e2e alias actually execute tests"). Verified by direct count: `grep -c "(run %{dep:" sarek/tests/e2e/dune` = 33 total run actions, of which 32 are attached to `(alias runtest)` (`test_vector_add` through `test_float32_sin_pure`, `sarek/tests/e2e/dune:170-331`) and 1 (`test_external_kernel`) is attached to the separate `e2e-gpu` alias (`sarek/tests/e2e/dune:374-377`) — exactly matching the claimed "32 tests executed." `dune runtest` now actually fails the build on nonzero exit from any of these 32, not just on a compile error.
  - New alias structure, also verified present in the dune file: `compile-only` (`sarek/tests/e2e/dune:347-353`, build-only — `test_pragma`, `test_barrier_converged`, `test_superstep`, `test_inline_pragma`; the same 4 are also gated into `runtest` as build-only deps at `:363-369` so a compile regression still fails CI without executing them), `e2e-gpu` (`:374-377`, `test_external_kernel`, GPU-required with no CPU fallback), `e2e-manual` (`:492-494`, `test_debug_native` + `test_float64_math_intrinsics`, report-only/no-assert tools). **Correction to the original claim wording:** these are **dune aliases**, not `make` targets — grepping `Makefile` for `compile-only`/`e2e-gpu`/`e2e-manual` finds zero references; there is no `make compile-only`, `make e2e-gpu`, or `make e2e-manual` wrapper. To invoke them you must run `dune build @sarek/tests/e2e/compile-only` etc. directly (or add a Makefile wrapper, which does not exist yet).
- **fixed 2026-07-02 (merged)** — the 5 previously-vacuous tests now assert real computed values (commit `3c6c0f98`, verified against the diff): `test_klet_fun.ml` now runs the add_scale kernel via `Execute.run_vectors` and compares every element against the expected formula (previously the `try`/`with` wrapped only a `print_endline`, so the guarded block could never raise); `test_klet_variant.ml` fixed the unconditional-PASSED-after-`None`-branch bug (the `None -> "No IR - SKIPPED"` path now exits 0 without printing PASSED) and builds `Circle`/`Square` from a float input inside the kernel, verifying computed areas against the host-side formula; `test_convention.ml` now asserts `point.x = 1.0` and `point.y = 2.0` instead of printing PASSED unconditionally; `test_convention_kernel.ml` builds real `Geometry_lib.point_custom` vectors and verifies the distance-to-origin kernel against `sqrt(x^2+y^2)`; `test_visibility_private.ml` builds real vectors and verifies `Visibility_lib.public_add` output against `x + y`. Per the commit message, `test_convention_kernel` and `test_visibility_private` had to pin the Native device explicitly rather than `devs.(0)`, because running on this session's OpenCL GPU raised an unrelated genuine `Backend_error` — noted as out of scope for the test-fix task.
- Confirmed, unchanged: `test_cross_module_type` and `test_registered_const` are disabled in names/modules and alias sections due to PPX registration issues (now at `sarek/tests/e2e/dune:68-69`, `:105-106`; the disabled-comment lines shifted slightly with the alias rewrite but the same two tests remain disabled).
- Confirmed, unchanged: `test_module_poly`, `test_bounded_recursion`, `test_inline_pragma`, and `test_nested_types` are declared as executables but only `test_module_poly`/`test_bounded_recursion`/`test_nested_types` are now wired into `runtest` via the per-test `(rule (alias runtest) ...)` stanzas (`sarek/tests/e2e/dune:308-321`) — `test_inline_pragma` remains **build-only**, parked in the `compile-only` alias, per an explicit code comment (`sarek/tests/e2e/dune:338-346`): running it on a real GPU segfaulted reproducibly when `Device.all ()` selected the OpenCL device for a non-tail-recursive `pragma ["sarek.inline N"]` kernel — a genuine backend crash, explicitly kept out of `runtest` to avoid nondeterministic suite crashes on machines where a GPU is the first device. **Keep this entry live** — this is the one item the audit asked to explicitly verify stays parked/unfixed, and it is confirmed still parked as of `618768b7`.
- Confirmed: `test_debug_native` and optional `test_external_kernel` are declared outside the main `runtest` alias (`sarek/tests/e2e/dune:379-462`); `test_float32_sin_pure` is now run inside `runtest` (see above, no longer excluded); `test_float64_math_intrinsics` and `test_stdlib_meta_proof` remain declared outside/partially outside the main alias — `test_stdlib_meta_proof` is now in `runtest` (`:323-326`), but `test_float64_math_intrinsics` remains report-only in `e2e-manual` (`:450-454`, `:492-494`) and must be run manually with `dune exec`.
- **RESOLVED 2026-07-02 (merged)** — the 5 previously-documented "vacuous tests that cannot fail on their own logic" are no longer vacuous; see the fixed-2026-07-02 entry above for per-test evidence.

## Performance Or Maintainability Risks

- Algorithmic E2E tests may be costly and backend-sensitive; failures can mix compiler, runtime, and device issues.
- Alias omissions can leave newer feature tests compiling but not running (now narrower in scope: only `test_cross_module_type`/`test_registered_const`, both intentionally disabled for PPX registration issues).
- Disabled registration tests are directly relevant to PPX include/registry behavior and can mask regressions.
- **Stale, corrected 2026-07-02:** the previous claim that "the alias's build-only nature means most of these risks are latent even when `dune runtest` is green" no longer holds — `runtest` now executes 32 tests and fails on nonzero exit (fixed 2026-07-02, merged; see Potential Invariant Violations above). Remaining latent risk is narrower: the `compile-only`/`e2e-manual` aliases and the still-build-only `test_inline_pragma` are the only tests whose regressions would not surface via a plain `dune runtest`.

## Related Tests

- Unit tests cover individual compiler stages before E2E; see `kb/sarek/tests/unit.md`.
- Negative tests cover expected compile failures; see `kb/sarek/tests/negative.md`.
- New runtime tests are configured separately in `sarek/tests/new_runtime/dune:4-20`.

## Missing Tests

- Active E2E for cross-module type registration and registered constants. Still missing.
- ~~Alias coverage for every executable intended to run.~~ **Largely closed 2026-07-02** — 32 executables now run under `runtest`; `test_inline_pragma` remains intentionally build-only (real segfault, out of scope), and `test_debug_native`/`test_float64_math_intrinsics` remain intentionally report-only/manual.
- E2E for native `downto`, native `create_array`, and simple `global_size_*`. Still missing from this suite (separate standalone regression executables exist elsewhere in `sarek/tests/e2e/dune`, e.g. `test_native_downto`, `test_native_create_array`, but they are not wired into `runtest` either).
- E2E for indirect convergence false negatives. Still missing.
- ~~Actual execution (not just compilation) of the 27 alias-listed executables that `make e2e-fast` doesn't cover.~~ **Closed 2026-07-02** — `dune runtest` now executes 32 of them directly; `make e2e-fast` is no longer the only thing that runs e2e tests in CI (CI runs `dune runtest`, per `scripts-ci.md`).
- ~~Real assertions in `test_klet_fun`, `test_klet_variant`, `test_convention`, `test_convention_kernel`, and `test_visibility_private` (currently print-only, cannot fail).~~ **Closed 2026-07-02** — all 5 now assert real computed values.
- No `make` wrapper exists for the new `compile-only`/`e2e-gpu`/`e2e-manual` dune aliases — a human must know to run `dune build @sarek/tests/e2e/<alias>` directly.

## Concrete Improvement/Fix Candidates

- ~~Add all intended executable tests to the e2e alias or document why they are build-only.~~ **DONE 2026-07-02** — remaining build-only/manual tests (`test_inline_pragma`, `test_debug_native`, `test_float64_math_intrinsics`) are now explicitly documented with reasons in the dune file.
- ~~Add `(action (run %{exe:<name>.exe}))` (or a `diff`/status-check action) to the e2e `runtest` alias so `dune runtest` executes what it builds, not just compiles it.~~ **DONE 2026-07-02** — implemented via per-test `(rule (alias runtest) (action (run ...)))` stanzas.
- Re-enable disabled registration tests after fixing registration, or add reduced failing unit tests now. Still open (`test_cross_module_type`, `test_registered_const`).
- Split long algorithm tests from compiler smoke tests so compiler regressions can be isolated quickly. Still open.
- New: add `make` targets that wrap `dune build @sarek/tests/e2e/compile-only`, `@e2e-gpu`, and `@e2e-manual` so these aliases are discoverable without reading the dune file.
