# Sarek Lowering

## Component Inventory

Lowering spans the legacy Kirc path and the current Sarek IR path:

- `sarek/ppx/Kirc_Ast.ml`
- `sarek/ppx/Sarek_lower.ml`
- `sarek/ppx/Sarek_ir_ppx.ml`
- `sarek/ppx/Sarek_lower_ir.ml`

## Per-File Purpose

- `Kirc_Ast.ml`: compile-time representation of legacy Kirc nodes used by the old lowerer and compatibility tests.
- `Sarek_lower.ml`: lowers typed AST to `Kirc_Ast`; still useful for legacy compatibility but not the main runtime IR path.
- `Sarek_ir_ppx.ml`: compile-time Sarek IR type mirror that can be quoted into runtime `Sarek.Sarek_ir`.
- `Sarek_lower_ir.ml`: lowers typed AST to `Sarek_ir_ppx` expressions, statements, declarations, helper functions, type declarations, and kernel records.

## Features And APIs

- Legacy type conversion from core type names is in `sarek/ppx/Sarek_lower.ml:116-139`.
- Legacy expression lowering covers constants, variables, arithmetic, control flow, memory access, records, variants, matches, shared memory, and supersteps.
- IR lowering maps core types to element types in `sarek/ppx/Sarek_lower_ir.ml:21-55`.
- IR expression lowering is in `sarek/ppx/Sarek_lower_ir.ml:274-412`.
- IR statement lowering is in `sarek/ppx/Sarek_lower_ir.ml:415-527`.
- Kernel lowering assembles declarations, constructors, type records, helper functions, and body IR in `sarek/ppx/Sarek_lower_ir.ml:594-702`.

## Invariants

- Lowerers should receive typed AST nodes with resolved, backend-compatible types.
- Expression-only typed forms must lower as IR expressions; statement-only forms must lower as statements.
- Loop direction, bitwise/logical operations, and integer widths must preserve source semantics.
- Shared memory sizes must be valid int32 expressions and converted where downstream APIs expect host ints.

## Potential Invariant Violations Or Bugs

- Confirmed: legacy lowering downcasts int64 constants with `Int64.to_int` in `sarek/ppx/Sarek_lower.ml:240-241`.
- Confirmed: legacy lowering ignores `Downto` direction and assumes `to` in `sarek/ppx/Sarek_lower.ml:381-389`.
- Confirmed: legacy lowering maps `Lnot` to logical `Not` in `sarek/ppx/Sarek_lower.ml:565-570`.
- Confirmed: IR type lowering maps unsupported `TTuple`, `TFun`, and `TVar` to `TInt32` in `sarek/ppx/Sarek_lower_ir.ml:52-54`.
- Probable: IR lowering maps arithmetic shift right `Asr` to logical `Shr` in `sarek/ppx/Sarek_lower_ir.ml:201-226`; signed negative values may differ.
- Confirmed limitation: match lowering still has tuple-pattern TODOs in legacy lowering at `sarek/ppx/Sarek_lower.ml:581-617`.
- Confirmed: IR pattern lowering encodes variable and tuple patterns as constructors in `sarek/ppx/Sarek_lower_ir.ml:556-574`, which depends on downstream interpretation.

## Performance Or Maintainability Risks

- Maintaining both legacy Kirc lowering and Sarek IR lowering creates semantic drift risk.
- Defaulting unknown IR types to `TInt32` can hide bugs until runtime or backend codegen.
- Lowering has separate expression and statement paths; adding new typed AST forms requires updating both paths and quote/runtime IR tests.

## Related Tests

- `sarek/tests/unit/test_lower.ml:384-446` covers legacy lowering.
- `sarek/tests/unit/dune:18` includes `test_lower_ir`.
- `sarek/tests/unit/test_quote_ir.ml:719-831` covers quoted IR shapes that the lowerer produces.
- E2E loop/bitwise/shared-memory behavior is exercised by tests listed in `sarek/tests/e2e/dune:73-88`.

## Missing Tests

- Legacy and native/IR `downto` loop semantics.
- Legacy `lnot` versus bitwise-not semantics.
- Int64 literals outside OCaml `int` range through the legacy path.
- Signed arithmetic shift right on negative inputs.
- Lowerer rejection of residual `TVar`, `TTuple`, and `TFun` instead of `TInt32` fallback.

## Concrete Improvement/Fix Candidates

- Make legacy lowering either intentionally frozen or remove it from semantic authority.
- Replace `TInt32` fallback in IR type lowering with structured errors.
- Add shared loop-direction tests across legacy, IR interpreter, native, and E2E paths.
- Split bitwise and logical unary operation tests in every lowering/codegen layer.
