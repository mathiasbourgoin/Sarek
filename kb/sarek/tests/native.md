# Sarek Native Test Directory

## Component Inventory

The native test directory currently contains only `sarek/tests/native/dune`.

## Per-File Purpose

- `sarek/tests/native/dune`: documents that the legacy SPOC test was moved to `*.spoc_legacy` and that there are no active executables.

## Features And APIs

- No active tests or libraries are declared.
- The directory remains listed in `sarek/tests/dune:1`.

## Invariants

- If the directory stays in the test tree, it should either contain active native tests or clearly remain a placeholder.
- Native compiler/runtime behavior should be covered elsewhere while this directory is inactive.

## Potential Invariant Violations Or Bugs

- Confirmed: `sarek/tests/native/dune:1-2` has no active executables, so native-specific coverage depends on unit, e2e, and `new_runtime` directories.

## Performance Or Maintainability Risks

- Empty test directories can mislead reviewers into assuming a dedicated native suite exists.
- Legacy-file references can accumulate stale expectations if not tracked.

## Related Tests

- Native helper/intrinsic unit tests are declared in `sarek/tests/unit/dune:14-15`.
- E2E `test_debug_native` is declared in `sarek/tests/e2e/dune:183-188`.
- New runtime native tests are documented in `kb/sarek/tests/new_runtime.md`.

## Missing Tests

- Dedicated native fallback tests for loop direction, local arrays, simple-mode intrinsics, and custom scalar arguments.
- CI alias that makes native-only coverage explicit.

## Concrete Improvement/Fix Candidates

- Either remove the placeholder directory from `sarek/tests/dune:1` or add native fallback tests here.
- If kept, add a README or Dune comment pointing to the active native coverage locations.
