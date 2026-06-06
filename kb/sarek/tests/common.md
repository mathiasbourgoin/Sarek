# Sarek Common Test Helpers

## Component Inventory

Common helpers live in `sarek/tests/common/` and are built as `sarek_test_common` by `sarek/tests/common/dune:1-4`.

## Per-File Purpose

- `sarek/tests/common/Test_kernels.ml`: defines neutral `param` and `test_kernel` records plus a small catalog of kernels.
- `sarek/tests/common/Ir_compare.ml`: compares legacy `Kirc_Ast.k_ext` trees and prints diffs.
- `sarek/tests/common/Source_gen.ml`: renders neutral kernels into old camlp4 and new PPX source strings.
- `sarek/tests/common/dune`: library stanza linking `sarek_ppx_lib`.

## Features And APIs

- Neutral test kernel schema is defined at `sarek/tests/common/Test_kernels.ml:13-28`.
- The included kernel catalog covers vector add/scale, int add, if, for, while, thread indices, and Float32 math at `sarek/tests/common/Test_kernels.ml:30-194`.
- `kernels_by_tag` and `find_kernel` are exposed at `sarek/tests/common/Test_kernels.ml:196-200`.
- Structural legacy IR equality is implemented in `sarek/tests/common/Ir_compare.ml:21-118`.
- Detailed diff reporting starts at `sarek/tests/common/Ir_compare.ml:195-247`.
- Source generation for old/new syntax is in `sarek/tests/common/Source_gen.ml:15-65`.

## Invariants

- Neutral kernel strings must be valid for both generated syntaxes.
- `Ir_compare.ir_equal` must compare semantic fields and deliberately ignore only uncomparable closures.
- Source generators should reflect currently supported PPX syntax.

## Potential Invariant Violations Or Bugs

- Confirmed limitation: `Ir_compare.ir_equal` treats `Native`, `GInt`, `GFloat`, `GFloat64`, and `NativeWithFallback` payloads as equal without comparing closures in `sarek/tests/common/Ir_compare.ml:106-115`.
- Confirmed drift risk: `Source_gen` still emits old camlp4 syntax and opens `Spoc`/`Kirc` in `sarek/tests/common/Source_gen.ml:33-65`; if old syntax is no longer active, generated output may be stale.
- Confirmed: the neutral catalog is small and lacks records, variants, shared memory, supersteps, recursion, and native fallback-specific cases.

## Performance Or Maintainability Risks

- The helper library is only useful if tests actively consume it; otherwise it can drift unnoticed.
- String-based source generation makes syntax changes hard to validate without compile tests.

## Related Tests

- The library is available to tests through `sarek/tests/common/dune:1-4`.
- Legacy lower/quote unit tests use related Kirc AST comparison patterns.

## Missing Tests

- Compile generated PPX source from `Source_gen`.
- Round-trip or golden tests for every neutral kernel.
- Common kernels for custom types, convergence, tailrec, and memory-space behavior.

## Concrete Improvement/Fix Candidates

- Add a small unit test that generates PPX source and parses it through `Sarek_parse`.
- Either remove old camlp4 generation or mark it explicitly as legacy-only.
- Expand the neutral catalog only if an active comparison test consumes it.
