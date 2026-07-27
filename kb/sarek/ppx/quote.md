# Sarek Quoting

## Component Inventory

Quoting spans `sarek/ppx/Sarek_quote_ir.ml`, `sarek/ppx/Sarek_quote.ml`, and generated code integration in `sarek/ppx/Sarek_ppx.ml`.

## Per-File Purpose

- `sarek/ppx/Sarek_quote_ir.ml`: converts compile-time `Sarek_ir_ppx` values into OCaml AST expressions referencing runtime `Sarek.Sarek_ir`.
- `sarek/ppx/Sarek_quote.ml`: quotes typed/source helper data, collects intrinsic references, builds native fallback wrappers, and creates final runtime `Kirc_types.kernel` expressions. **Trimmed 2026-07-25 (task #78):** the dead Kirc quoting family (`quote_elttype`, `quote_memspace`, `quote_case`, `quote_k_ext`, ~315 lines) was removed with the legacy `Kirc_Ast` path. What survives is live: the generic helpers (`quote_int`, `quote_int32`, `quote_float`, `quote_string`, `quote_bool`, `quote_list`, `quote_array`, `quote_option`, `evar`, `evar_qualified`, `sarek/ppx/Sarek_quote.ml:20-62`), `core_type_of_typ` (`:64`), `kernel_ctor_name` (`:85`), `build_kernel_args` (`:111`), intrinsic-ref collection (`:184-339`), `quote_kernel` (`:341`), and the `Sarek_ast` quoting family (`:397-703`).
- `sarek/ppx/Sarek_ppx.ml`: calls `Sarek_quote.quote_kernel` after lowering at `sarek/ppx/Sarek_ppx.ml:1902` (re-verified 2026-07-25; prior `:1524-1542` range was stale — drift unrelated to task #78, which changed only line `:2189` of this file).

## Features And APIs

- `Sarek_quote_ir` covers memory spaces, element types, constants, variables, expressions, lvalues, statements, declarations, type declarations, variants, functions, and whole kernels.
- `Sarek_quote.ml` generates intrinsic presence checks and native fallback functions before constructing the final runtime kernel in `quote_kernel` at `sarek/ppx/Sarek_quote.ml:341-387` (re-verified 2026-07-25; the prior `:638-682` range moved when ~315 lines of dead Kirc quoting were removed, task #78). Native-wrapper generation is delegated to `Sarek_native_gen_kernel.gen_cpu_kern_native_wrapper` (`:367`) and IR quoting to `Sarek_quote_ir.quote_kernel` (`:374`).
- Runtime constructor/type registration is coordinated with the PPX registration path in `sarek/ppx/Sarek_ppx.ml:1336-1380`.

## Invariants

- Every `Sarek_ir_ppx` constructor reachable from `Sarek_lower_ir` must have an equivalent quote path.
- Quoted native fallback functions must agree with the quoted runtime IR on argument order, type layout, and type declaration metadata.
- Intrinsic references must be retained so runtime/link checks see used intrinsics.

## Potential Invariant Violations Or Bugs

- **NEW (2026-07-02 audit, confirmed):** `collect_intrinsic_refs` drops the value subexpression of `TEVecSet`. At `sarek/ppx/Sarek_quote.ml:206` (re-verified 2026-07-25; was `:504` before the task #78 trim), `TEVecSet (v, i, _)` is grouped with `TEVecGet`/`TEArrGet` and only traverses the vector (`v`) and index (`i`) — the stored value is matched as `_` and never recursed into. By contrast `TEArrSet (a, i, x)` at `:210-215` correctly unions over all three subexpressions (`a`, `i`, and `x`). Consequence: an intrinsic function or constant used only inside the value being stored to a vector (e.g. `vec.(i) <- warp_shuffle x mask`) is never collected into the intrinsic-reference set, so downstream runtime/link "intrinsic used" checks can miss it. This is a hot path (fan-in 43 per the audit's call-graph). Not previously documented in the KB.
- Uncertain: intrinsic-reference collection in `Sarek_quote.ml` is separate from IR lowering/native generation; newly added intrinsic forms could be emitted without being collected unless tests cover the new constructor.
- Confirmed maintainability issue: quoting bridges compile-time IR types to runtime `Sarek.Sarek_ir`; any runtime IR constructor change requires synchronized edits in `Sarek_ir_ppx`, `Sarek_lower_ir`, `Sarek_quote_ir`, and tests.

## Performance Or Maintainability Risks

- Generated OCaml AST is large for complex kernels and inlined recursion; quoting is directly affected by monomorphization/tailrec expansion.
- Runtime wrapper generation, intrinsic checks, and IR quoting are concentrated in one module, increasing review cost for small semantic changes.

## Related Tests

- `sarek/tests/unit/test_quote.ml:368-506` covers helper quoting and `Sarek_ast`/source quote utilities: groups `basic_quoting`, `collection_quoting`, `type_conversion`, `sarek_ast_quoting`, `expr_quoting`, `intrinsic_refs` (re-verified 2026-07-25). The `kirc_ast_quoting` group and its 4 vacuous cases were removed with the legacy path (task #78).
- `sarek/tests/unit/test_quote_ir.ml:719-831` covers most IR quote constructors, including types, expressions, statements, declarations, and kernels.
- E2E tests in `sarek/tests/e2e/dune:56-93` validate generated quoted kernels compile and execute.

## Missing Tests

- Quote coverage guard that fails when `Sarek_ir_ppx` gains a constructor without tests.
- Intrinsic-reference collection for every intrinsic expression/statement form, specifically a regression test for an intrinsic used only in the value position of a vector store (`TEVecSet`) — currently uncollected.
- Large generated kernel quoting after monomorphization and pragma inlining.

## Concrete Improvement/Fix Candidates

- Fix `collect_intrinsic_refs` to also traverse the value subexpression of `TEVecSet` (`Sarek_quote.ml:206`), matching the `TEArrSet` handling at `:210-215`.
- Add constructor-coverage tests for `Sarek_ir_ppx` and runtime `Sarek_ir` parity.
- Keep intrinsic collection close to IR traversal or derive it from the same traversal used for quoting.
- Add a small golden generated-code test for a kernel with types, intrinsics, native fallback, and helper functions.
