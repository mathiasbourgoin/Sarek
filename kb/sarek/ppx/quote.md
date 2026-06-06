# Sarek Quoting

## Component Inventory

Quoting spans `sarek/ppx/Sarek_quote_ir.ml`, `sarek/ppx/Sarek_quote.ml`, and generated code integration in `sarek/ppx/Sarek_ppx.ml`.

## Per-File Purpose

- `sarek/ppx/Sarek_quote_ir.ml`: converts compile-time `Sarek_ir_ppx` values into OCaml AST expressions referencing runtime `Sarek.Sarek_ir`.
- `sarek/ppx/Sarek_quote.ml`: quotes typed/source helper data, collects intrinsic references, builds native fallback wrappers, and creates final runtime `Kirc_types.kernel` expressions.
- `sarek/ppx/Sarek_ppx.ml`: calls quote generation after lowering in `sarek/ppx/Sarek_ppx.ml:1524-1542`.

## Features And APIs

- `Sarek_quote_ir` covers memory spaces, element types, constants, variables, expressions, lvalues, statements, declarations, type declarations, variants, functions, and whole kernels.
- `Sarek_quote.ml` generates intrinsic presence checks and native fallback functions before constructing the final runtime kernel at `sarek/ppx/Sarek_quote.ml:638-682`.
- Runtime constructor/type registration is coordinated with the PPX registration path in `sarek/ppx/Sarek_ppx.ml:1336-1380`.

## Invariants

- Every `Sarek_ir_ppx` constructor reachable from `Sarek_lower_ir` must have an equivalent quote path.
- Quoted native fallback functions must agree with the quoted runtime IR on argument order, type layout, and type declaration metadata.
- Intrinsic references must be retained so runtime/link checks see used intrinsics.

## Potential Invariant Violations Or Bugs

- Uncertain: intrinsic-reference collection in `Sarek_quote.ml` is separate from IR lowering/native generation; newly added intrinsic forms could be emitted without being collected unless tests cover the new constructor.
- Confirmed maintainability issue: quoting bridges compile-time IR types to runtime `Sarek.Sarek_ir`; any runtime IR constructor change requires synchronized edits in `Sarek_ir_ppx`, `Sarek_lower_ir`, `Sarek_quote_ir`, and tests.

## Performance Or Maintainability Risks

- Generated OCaml AST is large for complex kernels and inlined recursion; quoting is directly affected by monomorphization/tailrec expansion.
- Runtime wrapper generation, intrinsic checks, and IR quoting are concentrated in one module, increasing review cost for small semantic changes.

## Related Tests

- `sarek/tests/unit/test_quote.ml:405-556` covers helper quoting and legacy/source quote utilities.
- `sarek/tests/unit/test_quote_ir.ml:719-831` covers most IR quote constructors, including types, expressions, statements, declarations, and kernels.
- E2E tests in `sarek/tests/e2e/dune:56-93` validate generated quoted kernels compile and execute.

## Missing Tests

- Quote coverage guard that fails when `Sarek_ir_ppx` gains a constructor without tests.
- Intrinsic-reference collection for every intrinsic expression/statement form.
- Large generated kernel quoting after monomorphization and pragma inlining.

## Concrete Improvement/Fix Candidates

- Add constructor-coverage tests for `Sarek_ir_ppx` and runtime `Sarek_ir` parity.
- Keep intrinsic collection close to IR traversal or derive it from the same traversal used for quoting.
- Add a small golden generated-code test for a kernel with types, intrinsics, native fallback, and helper functions.
