# Sarek Unit Tests

## Component Inventory

Unit tests live in `sarek/tests/unit/` and are declared in `sarek/tests/unit/dune`.

## Per-File Purpose

- `sarek/tests/unit/dune`: lists PPX/compiler/runtime unit test executables at `sarek/tests/unit/dune:1-30`; declares separate `test_cpu_runtime`, fusion, Float32, and Float64 tests at `sarek/tests/unit/dune:32-54`; declares `test_ptx_snapshot` at `sarek/tests/unit/dune:58-60`. (All three ranges re-verified 2026-07-25; each shifted by one line when `test_lower` was dropped from the `(names ...)` list, task #78.)
- `test_types.ml`, `test_scheme.ml`: type and scheme behavior.
- `test_env.ml`: environment lookup, shadowing, custom types, and scope levels.
- `test_typer.ml`: type inference and typed AST construction.
- `test_parse.ml`: parser coverage for operators, types, kernels, and expressions.
- `test_lower_ir.ml`: Sarek IR lowering. (Its sibling `test_lower.ml`, which tested only the legacy `Kirc_Ast` lowering, was deleted 2026-07-25 with that lowering — task #78.)
- `test_quote.ml`, `test_quote_ir.ml`: generated quote helpers and IR quote constructors. `test_quote.ml` lost its `kirc_ast_quoting` group on 2026-07-25 (task #78): its 4 cases (`test_quote_elttype_int32/float32`, `test_quote_memspace_local/shared`) tested the deleted Kirc quoting family and were vacuous anyway — they accepted `Pexp_extension _` as a pass, so they asserted nothing.
- `test_core_primitives.ml`, `test_ppx_registry.ml`: built-ins and registry behavior.
- `test_convergence.ml`: barrier/warp and execution strategy analysis.
- `test_tailrec*.ml`: tail recursion analysis, elimination, bounded recursion, and pragma inlining.
- `test_native_helpers.ml`, `test_native_intrinsics.ml`: native codegen helpers and intrinsic expressions.
- `test_ir_interp.ml`, `test_execute.ml`, `test_kirc_kernel.ml`, `test_cpu_runtime.ml`: runtime-facing helper behavior.
- `test_error.ml`, `test_reserved.ml`: error and reserved-keyword behavior.
- `test_mono.ml`: monomorphization.
- `test_fusion.ml`, `test_float32.ml`, `test_float64.ml`: runtime/library module tests outside the bulk PPX test stanza.
- `test_ptx_snapshot.ml`: CPU-only PTX golden-snapshot test, no CUDA device required; links `sarek.codegen`/`alcotest` (`sarek/tests/unit/dune:58-60`). Sibling in spirit to `sarek/tests/codegen_golden/test_codegen_golden.ml` (1,444 lines), which snapshot-tests the other 4 pure-OCaml codegen backends (CUDA/OpenCL/Metal/GLSL/WGSL text emission) via `codegen_golden_backends` (`sarek/tests/codegen_golden/dune`); that suite lives in its own top-level directory, not under `unit/`, but is likewise CPU-only/FFI-free and is in the default `runtest` alias (`sarek/tests/dune:1-15`). See `kb/sarek/tests/README.md` for the top-level wiring.

## Features And APIs

- Bulk PPX unit tests use `ppxlib.metaquot` for AST construction in `sarek/tests/unit/dune:28-30`.
- Parse tests run groups from `sarek/tests/unit/test_parse.ml:501-562`.
- (Gone 2026-07-25, task #78: "legacy lower tests run groups from `sarek/tests/unit/test_lower.ml:384-446`" — the file was deleted with the legacy lowering it tested.)
- Quote IR tests cover most IR constructors in `sarek/tests/unit/test_quote_ir.ml:719-831`.
- Native intrinsic tests cover type conversion and selected constants/functions in `sarek/tests/unit/test_native_intrinsics.ml:186-227`.

## Invariants

- Unit tests should isolate compiler stages before E2E execution.
- Tests that inspect generated AST/IR should assert semantics, not just construction success.
- Unit coverage should include known edge cases found in lower/native/convergence code.

## Potential Invariant Violations Or Bugs

- **MOOT — code and test deleted 2026-07-25 (task #78).** Historical record: lower tests included `for` coverage at `sarek/tests/unit/test_lower.ml:422-425` with no explicit `downto` coverage. `test_lower.ml` is gone; the loop-direction risk it left uncovered was in the deleted legacy lowering. Native/IR `downto` coverage is unaffected — see `kb/sarek/ppx/native-gen.md`.
- **PARTLY MOOT — legacy half deleted 2026-07-25 (task #78).** Parser tests still cover `lnot` at `sarek/tests/unit/test_parse.ml:518-521`. The rest of this gap ("legacy lowering maps `Lnot` incorrectly; lowering tests list `not` at `sarek/tests/unit/test_lower.ml:416`") referred to the deleted legacy lowering and its deleted test — nothing to separate there anymore. Whether the *live* IR path separates bitwise from logical not is not asserted here and was not re-checked.
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
