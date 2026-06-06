# Sarek Unit Tests

## Component Inventory

Unit tests live in `sarek/tests/unit/` and are declared in `sarek/tests/unit/dune`.

## Per-File Purpose

- `sarek/tests/unit/dune`: lists PPX/compiler/runtime unit test executables at `sarek/tests/unit/dune:1-32`; declares separate fusion, Float32, and Float64 tests at `sarek/tests/unit/dune:34-52`.
- `test_types.ml`, `test_scheme.ml`: type and scheme behavior.
- `test_env.ml`: environment lookup, shadowing, custom types, and scope levels.
- `test_typer.ml`: type inference and typed AST construction.
- `test_parse.ml`: parser coverage for operators, types, kernels, and expressions.
- `test_lower.ml`, `test_lower_ir.ml`: legacy and Sarek IR lowering.
- `test_quote.ml`, `test_quote_ir.ml`: generated quote helpers and IR quote constructors.
- `test_core_primitives.ml`, `test_ppx_registry.ml`: built-ins and registry behavior.
- `test_convergence.ml`: barrier/warp and execution strategy analysis.
- `test_tailrec*.ml`: tail recursion analysis, elimination, bounded recursion, and pragma inlining.
- `test_native_helpers.ml`, `test_native_intrinsics.ml`: native codegen helpers and intrinsic expressions.
- `test_ir_interp.ml`, `test_execute.ml`, `test_kirc_kernel.ml`, `test_cpu_runtime.ml`: runtime-facing helper behavior.
- `test_error.ml`, `test_reserved.ml`: error and reserved-keyword behavior.
- `test_mono.ml`: monomorphization.
- `test_fusion.ml`, `test_float32.ml`, `test_float64.ml`: runtime/library module tests outside the bulk PPX test stanza.

## Features And APIs

- Bulk PPX unit tests use `ppxlib.metaquot` for AST construction in `sarek/tests/unit/dune:29-32`.
- Parse tests run groups from `sarek/tests/unit/test_parse.ml:501-562`.
- Legacy lower tests run groups from `sarek/tests/unit/test_lower.ml:384-446`.
- Quote IR tests cover most IR constructors in `sarek/tests/unit/test_quote_ir.ml:719-831`.
- Native intrinsic tests cover type conversion and selected constants/functions in `sarek/tests/unit/test_native_intrinsics.ml:186-227`.

## Invariants

- Unit tests should isolate compiler stages before E2E execution.
- Tests that inspect generated AST/IR should assert semantics, not just construction success.
- Unit coverage should include known edge cases found in lower/native/convergence code.

## Potential Invariant Violations Or Bugs

- Confirmed gap: lower tests include `for` coverage at `sarek/tests/unit/test_lower.ml:422-425`, but no observed explicit `downto` coverage; both legacy and native code have loop-direction risks.
- Confirmed gap: parser tests cover `lnot` at `sarek/tests/unit/test_parse.ml:518-521`, but legacy lowering maps `Lnot` incorrectly; lowering tests list `not` at `sarek/tests/unit/test_lower.ml:416` and do not visibly separate bitwise-not semantics.
- Confirmed gap: native intrinsic simple-mode tests are grouped at `sarek/tests/unit/test_native_intrinsics.ml:202-212`, but the `global_size_*` simple-mode mismatch is not documented as covered.

## Performance Or Maintainability Risks

- The unit suite is broad but each file is stage-specific; cross-stage invariants like `float` semantics and memory space can fall between tests.
- Some tests likely assert structure tightly, which is useful for regression but can increase churn when IR shapes intentionally change.

## Related Tests

- E2E tests validate that compiler-stage behavior compiles and runs; see `kb/sarek/tests/e2e.md`.
- Negative tests validate selected compiler failures; see `kb/sarek/tests/negative.md`.
- PPX-local tests under `sarek/ppx/test/` cover reserved/error/debug helpers outside `sarek/tests/unit`.

## Missing Tests

- Cross-stage `float` mapping.
- Array memory-space unification.
- Indirect convergence violations.
- `downto` and bitwise `lnot` through lowering/native execution.
- Native `create_array` int32 size handling.
- Duplicate record field/constructor names.

## Concrete Improvement/Fix Candidates

- Add focused unit tests for each confirmed compiler finding before fixing.
- Add a small cross-stage test helper that parses, types, lowers, quotes, and checks selected invariants.
- Add table-driven tests for every binary/unary op across parser, lowerer, IR quote, and native generation.
