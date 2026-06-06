# Sarek Negative Tests

## Component Inventory

Negative tests live in `sarek/tests/negative/` and are configured by `sarek/tests/negative/dune`.

## Per-File Purpose

- `sarek/tests/negative/dune`: documents expected compiler errors and declares profile-gated libraries.
- `test_barrier_diverged.ml`: barrier in diverged control flow.
- `test_superstep_diverged.ml`: superstep/barrier convergence violation.
- `test_warp_diverged.ml`: warp collective in diverged control flow.
- `test_unbound_function.ml`: unbound function/variable.
- `test_reserved_keyword.ml`: reserved keyword rejection.
- `test_convention_kernel_fail.ml`: field lookup failure.
- `test_convention_kernel_fail2.ml`: type unification failure.
- `test_inline_node_exhaustion.ml`: pragma inlining node-budget failure.

## Features And APIs

- Expected errors are documented in `sarek/tests/negative/dune:8-16`.
- Each case is declared as a Dune library with `sarek_ppx` preprocessing.
- All cases are disabled unless profile is `negative`, for example `sarek/tests/negative/dune:26-27`; the same pattern repeats through `sarek/tests/negative/dune:103-104`.

## Invariants

- Each negative test should fail compilation.
- The failure reason should match the expected diagnostic, not just any compile error.
- Negative tests should be easy to run from CI or `make test_negative`.

## Potential Invariant Violations Or Bugs

- Confirmed: expected diagnostics are comments only at `sarek/tests/negative/dune:8-16`; Dune library stanzas do not assert stderr content.
- Confirmed: the default top-level test alias does not include the negative profile in `sarek/tests/dune:3-9`.
- Confirmed risk: if a negative test fails earlier for an unrelated parse/module/build reason, the profile build could still be considered a failure in the expected direction unless the outer harness checks text.

## Performance Or Maintainability Risks

- Profile-gated compile failures are easy to omit from regular local runs.
- Error message changes can silently desynchronize comments and actual diagnostics.
- Adding new negative cases requires remembering the profile gate pattern.

## Related Tests

- Unit error string tests live in `sarek/ppx/test/test_sarek_error.ml`.
- Unit convergence tests are included by `sarek/tests/unit/dune:22`.
- E2E converged barrier/superstep positive cases are declared in `sarek/tests/e2e/dune:74-75`.

## Missing Tests

- Exact stderr assertions for all expected messages.
- Negative tests for indirect convergence false negatives.
- Negative tests for duplicate fields/constructors and memory-space mismatches.
- Negative tests for invalid `%sarek_extend` or intrinsic PPX declarations.

## Concrete Improvement/Fix Candidates

- Convert each negative case to a cram test or Dune action that asserts expected stderr.
- Keep the current libraries as compile-failure fixtures, but drive them from a checked harness.
- Add a small helper script to reduce duplicated profile-gated stanza boilerplate.
