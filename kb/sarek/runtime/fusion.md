# Fusion

## Component Inventory

Reviewed fusion files: `sarek/sarek/Sarek_fusion.ml`, `sarek/sarek/FUSION.md`, `sarek/sarek/Fusion_error.ml`, and `sarek/sarek/test/test_fusion_error.ml`.

## Per-File Purpose

- `sarek/sarek/FUSION.md`: design intent, supported fusion patterns, cost model, constraints, and examples.
- `sarek/sarek/Sarek_fusion.ml`: fusion analysis, eligibility checks, map-map fusion, stencil/map fusion, substitution helpers, cost estimation, and auto-fusion.
- `sarek/sarek/Fusion_error.ml`: fusion-specific error variants and formatting.
- `sarek/sarek/test/test_fusion_error.ml`: error formatting tests.

## Features/APIs

- Kernel analysis for reads, writes, side effects, barriers, shared memory, and estimated cost.
- `can_fuse` style eligibility decisions.
- Pairwise fusion and pipeline auto-fusion APIs.
- Map-map fusion and stencil fusion rewrite helpers.
- Fusion diagnostics and cost-model metadata.

## Invariants

- Fusion must preserve every original kernel's effects unless it replaces them with a proven equivalent fused kernel.
- Kernels with atomics, unsafe side effects, incompatible dimensions, or unsupported memory dependencies must not be fused.
- Substitution over IR expressions must recursively cover every expression form that can reference an intermediate value.
- Fusion diagnostics should match the documented constraints in `FUSION.md`.

## Potential Invariant Violations/Bugs

- Atomic detection is effectively absent. `fusion_info.has_atomics` exists at `sarek/sarek/Sarek_fusion.ml:24-30`, but analysis sets it to false with a TODO around `sarek/sarek/Sarek_fusion.ml:348-354`, and `can_fuse` does not reject atomics at `sarek/sarek/Sarek_fusion.ml:401-431`. The docs say atomics prevent fusion at `sarek/sarek/FUSION.md:116-122`.
- Stencil substitution is incomplete. `subst_stencil_reads` recurses through only a subset of expression variants and returns unchanged for the rest at `sarek/sarek/Sarek_fusion.ml:873-908`. Expressions in records, app args, array-read expressions, if/match branches, or other missed forms can retain references to intermediates. Compare with fuller substitution helpers elsewhere in the file. Uncertain exact surface depends on which IR forms stencil fusion admits.

## Recently Resolved

- Pipeline APIs dropping unfused kernels was fixed by PR #138, merged as `06b7d70`. New list-returning APIs preserve unfused kernels, while legacy single-kernel APIs remain strict when a pipeline cannot collapse to one kernel.
- Mismatched one-to-one index fusion was rejected in PR #138, preventing a producer from being dropped when substitution cannot actually replace the consumer read.

## Performance Or Maintainability Risks

- Legacy single-kernel pipeline APIs remain narrow: callers that need to preserve partially fused schedules should use the list-returning APIs added in PR #138.
- Analysis fields can drift from implementation because some fields are populated as constants.
- Rewrite helpers are hand-recursive and differ in expression coverage.
- Fusion diagnostics can become misleading when documented constraints are not enforced.

## Related Tests

- `sarek/sarek/test/test_fusion_error.ml`: error formatting only.
- `sarek/tests/unit/test_fusion.ml`: fusion analysis, pipeline preservation, auto-fusion decisions, and mismatched-index rejection.

## Missing Tests

- Kernels containing atomic intrinsics, verifying fusion rejection.
- Stencil fusion with intermediate reads embedded in each supported expression form.
- Fused versus unfused equivalence tests for map-map and stencil cases.
- Diagnostics tests that ensure documented constraints are emitted.

## Concrete Improvement/Fix Candidates

- Implement atomic detection by scanning intrinsic calls and memory operations, then reject in `can_fuse`.
- Replace partial substitution walkers with a shared full-expression traversal.
- Add semantic regression tests that execute original and fused kernels on small arrays and compare outputs.
