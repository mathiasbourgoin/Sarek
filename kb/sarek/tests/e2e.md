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

- Confirmed: `test_cross_module_type` and `test_registered_const` are disabled in names/modules and alias sections due to PPX registration issues at `sarek/tests/e2e/dune:68-69`, `sarek/tests/e2e/dune:105-106`, and `sarek/tests/e2e/dune:160-161`.
- Confirmed: `test_module_poly`, `test_bounded_recursion`, `test_inline_pragma`, and `test_nested_types` are built in `sarek/tests/e2e/dune:89-93` but are not included in the alias dependency list ending at `sarek/tests/e2e/dune:181`.
- Confirmed: `test_debug_native` and optional `test_external_kernel` are declared outside the main alias in `sarek/tests/e2e/dune:183-196`.

## Performance Or Maintainability Risks

- Algorithmic E2E tests may be costly and backend-sensitive; failures can mix compiler, runtime, and device issues.
- Alias omissions can leave newer feature tests compiling but not running.
- Disabled registration tests are directly relevant to PPX include/registry behavior and can mask regressions.

## Related Tests

- Unit tests cover individual compiler stages before E2E; see `kb/sarek/tests/unit.md`.
- Negative tests cover expected compile failures; see `kb/sarek/tests/negative.md`.
- New runtime tests are configured separately in `sarek/tests/new_runtime/dune:4-20`.

## Missing Tests

- Active E2E for cross-module type registration and registered constants.
- Alias coverage for every executable intended to run.
- E2E for native `downto`, native `create_array`, and simple `global_size_*`.
- E2E for indirect convergence false negatives.

## Concrete Improvement/Fix Candidates

- Add all intended executable tests to the e2e alias or document why they are build-only.
- Re-enable disabled registration tests after fixing registration, or add reduced failing unit tests now.
- Split long algorithm tests from compiler smoke tests so compiler regressions can be isolated quickly.
