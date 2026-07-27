# Sarek Lowering

## Component Inventory

Lowering is a single path — the Sarek IR path — since 2026-07-25 (task #78):

- `sarek/ppx/Sarek_ir_ppx.ml`
- `sarek/ppx/Sarek_lower_ir.ml`

**Deleted 2026-07-25 (task #78):** `sarek/ppx/Kirc_Ast.ml` (the legacy Kirc AST mirror) and `sarek/ppx/Sarek_lower.ml` (the legacy typed-AST → `Kirc_Ast` lowering). The path was not merely vestigial, it was *unreachable*: the `sarek` runtime library never had a `Kirc_Ast` module, yet `Sarek_quote.quote_elttype` emitted `Sarek.Kirc_Ast.EInt32`, so any code generated through it could never have compiled. Empirically 0 of 85 PPX-expanded artifacts referenced `Kirc_Ast`. The live pipeline is `Sarek_ppx.expand_kernel` → `Sarek_lower_ir.lower_kernel` → `Sarek_quote_ir`; after the deletion the whole repo still builds and the full test suite is unchanged (all PPX-expanded artifacts byte-identical except the three edited source files). `Sarek_lower.ml`'s one live part — the C struct/union/builder string generator for `[@@sarek.type]` registrations — was kept and the file renamed to `sarek/ppx/Sarek_ctype_gen.ml`; it is not a lowering component and is documented in `kb/sarek/ppx/README.md`.

## Per-File Purpose

- `Sarek_ir_ppx.ml`: compile-time Sarek IR type mirror that can be quoted into runtime `Sarek.Sarek_ir`.
- `Sarek_lower_ir.ml`: lowers typed AST to `Sarek_ir_ppx` expressions, statements, declarations, helper functions, type declarations, and kernel records.

## Features And APIs

- Type conversion from core type names (`c_type_of_core_type`/`typ_of_core_type`, formerly cited here as `sarek/ppx/Sarek_lower.ml:116-139`) survives byte-identically, same line numbers, in `sarek/ppx/Sarek_ctype_gen.ml:116-139` (re-verified 2026-07-25). It serves `[@@sarek.type]` C-string generation, not lowering.
- Legacy expression lowering (constants, variables, arithmetic, control flow, memory access, records, variants, matches, shared memory, supersteps) — **gone, code deleted 2026-07-25 (task #78)**.
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

- **MOOT — code deleted 2026-07-25 (task #78).** Historical record: legacy lowering downcast int64 constants with `Int64.to_int` at `sarek/ppx/Sarek_lower.ml:240-241`. That file no longer exists; the defect cannot be reached and there is nothing left to fix. Do not go hunting for it.
- **MOOT — code deleted 2026-07-25 (task #78); the 2026-07-02 fix below went away with the file it guarded.** Historical record, **FIXED 2026-07-02 (merged, PRs #211/#213), commit `c9c1289b`:** legacy `Sarek_lower.ml` lowering ignored `Downto` direction (`let _ = dir in`) and always lowered `TEFor` to a `Kirc_Ast.DoLoop` as if it were `Upto`, silently mis-lowering any `downto` loop that reached this path. Verified fixed at HEAD `618768b7`: `sarek/ppx/Sarek_lower.ml:381-393` now pattern-matches `dir` and raises `Location.raise_errorf ~loc:... "downto is not supported by the legacy lowering path"` for `Sarek_ast.Downto`, only lowering `Upto` to `DoLoop`. This path had no production callers (native and `Sarek_lower_ir` handle `downto` themselves), so the fix was a hard rejection rather than an implementation. Its regression test `test_lower_for_downto_raises` lived in `sarek/tests/unit/test_lower.ml`, deleted with the path.
- **MOOT — code deleted 2026-07-25 (task #78).** Historical record: legacy lowering mapped `Lnot` to logical `Not` at `sarek/ppx/Sarek_lower.ml:565-570`. Gone with the file; the live `Sarek_lower_ir` path is unaffected by this claim and was never covered by it.
- **Confirmed, still live, but narrower than previously stated.** `elttype_of_typ` (`sarek/ppx/Sarek_lower_ir.ml:22-68`) still maps `TTuple _` and `TFun _` to `Ir.TInt32` as a general fallback (`:62-64`) — this is now a documented, intentional placeholder for non-parameter typed values that legitimately carry `TFun`/`TTuple` type but whose "elttype" is never read as data, e.g. local helper-function bindings inside a kernel body (`let make_p x y z = ... in`, see `test_visibility_private.ml`/`test_transpose.ml`/`bench_nbody.ml`). **FIXED 2026-07-02 (merged, PRs #211/#213), commit `fdf53ac3` — kernel *parameters* specifically are no longer silently accepted.** `lower_param` (`sarek/ppx/Sarek_lower_ir.ml:697-720`) now checks `p.tparam_type` before calling `elttype_of_typ` and raises a located PPX error for `TTuple _` ("Tuple-typed kernel parameters are not supported; pass components as separate parameters.") and `TFun _` ("Function-typed kernel parameters are not supported."). Negative-test coverage: `sarek/tests/negative/test_tuple_param.ml` and `test_fun_param.ml` (see `kb/sarek/tests/negative.md`), wired into `make test_negative` (`Makefile:101-104`) and into `sarek/tests/negative/dune:16-17` documentation. The general `elttype_of_typ` fallback for non-parameter `TTuple`/`TFun` values remains live by design and is not itself the defect.
- **Rewritten 2026-07-02 (merged, PRs #211/#213), commits `87864852` + `81a2de48` — the prior text here ("Asr maps to logical Shr") was backwards and is now corrected.** `Ir.Shr` is canonicalized to a single **arithmetic** (sign-extending) shift-right IR constructor, matching what 5 of 7 backend consumers (CUDA/OpenCL/Metal/GLSL/WGSL plain `>>` on a signed type) already emitted; PTX (`shr.u32` → `shr.s32`) and the interpreter (`shift_right_logical` → `shift_right`) were changed to match. `Asr` lowers directly to `Ir.Shr` (`sarek/ppx/Sarek_lower_ir.ml:241`). `Lsr` (logical/unsigned shift) is rewritten by `lower_lsr` (`sarek/ppx/Sarek_lower_ir.ml:359-384`) into a width-aware IR expression tree built only from existing `Shr`/`Shl`/`BitXor`/`Eq`/`EIf` nodes (no new IR constructor — `Sarek_ir_ptx_expr.ml` and formal/codegen-ptx model `Shr` and cannot gain a new shift kind in this task): `if n = 0 then a else (a >>_asr n) lxor ((a >>_asr (width-1)) <<_lsl ((width - n) land (width - 1)))`, where `width` is 32 or 64 from the operand's `Ir.elttype`. `is_trivial_ir_expr` (`:298-299`) restricts this rewrite to `EVar`/`EConst` operands — `Sarek_ir_ppx.expr` has no let-binding form, so a non-trivial operand (e.g. one embedding an atomic intrinsic call) would otherwise be silently evaluated multiple times (`a`/`b` each appear 3x in the tree); `lower_lsr` raises a located PPX error directing the user to hoist the operand into a `let` instead (`sarek/ppx/Sarek_lower_ir.ml:365-372`). Regression coverage: `sarek/tests/unit/test_lower_ir.ml` (non-vacuous negative-operand cases per commit `05690388`).
- **MOOT — code deleted 2026-07-25 (task #78).** Historical record: match lowering carried tuple-pattern TODOs in legacy lowering at `sarek/ppx/Sarek_lower.ml:581-617`. Those TODOs were deleted with the file; nothing was ported. IR-path pattern lowering is the bullet below.
- Confirmed: IR pattern lowering encodes variable and tuple patterns as constructors in `sarek/ppx/Sarek_lower_ir.ml:556-574`, which depends on downstream interpretation.

## Performance Or Maintainability Risks

- (Retired 2026-07-25, task #78: "maintaining both legacy Kirc lowering and Sarek IR lowering creates semantic drift risk" — there is only one lowering path now.)
- Defaulting unknown IR types to `TInt32` can hide bugs until runtime or backend codegen.
- Lowering has separate expression and statement paths; adding new typed AST forms requires updating both paths and quote/runtime IR tests.

## Related Tests

- `sarek/tests/unit/dune:17` includes `test_lower_ir` (re-verified 2026-07-25 after `test_lower` was dropped from the `(names ...)` list).
- (Gone 2026-07-25, task #78: `sarek/tests/unit/test_lower.ml`, which covered only the deleted legacy lowering, was deleted with it. `test_lower_ir` is a different, still-live test.)
- `sarek/tests/unit/test_quote_ir.ml:719-831` covers quoted IR shapes that the lowerer produces.
- E2E loop/bitwise/shared-memory behavior is exercised by tests listed in `sarek/tests/e2e/dune:73-88`.

## Missing Tests

- (No longer needed 2026-07-25, task #78: "legacy `lnot` versus bitwise-not semantics" and "int64 literals outside OCaml `int` range through the legacy path" — both were gaps in the deleted legacy lowering, so there is no code left to cover. Bitwise/shift semantics on the live IR path are covered by `test_lower_ir.ml`, see the `Shr`/`Lsr` entry above.)
- (Now covered, no longer missing: native/IR `downto` — see `kb/sarek/ppx/native-gen.md`; signed arithmetic shift right on negative inputs and `lsr` non-vacuous negative-operand cases — `test_lower_ir.ml`; kernel-parameter `TTuple`/`TFun` rejection — `test_tuple_param.ml`/`test_fun_param.ml`.)

## Concrete Improvement/Fix Candidates

- (Done 2026-07-25, task #78: "make legacy lowering either intentionally frozen or remove it from semantic authority" — resolved by deleting it.)
- Consider whether the general (non-parameter) `elttype_of_typ` `TTuple`/`TFun` → `TInt32` fallback should be replaced with an explicit "don't-care placeholder" type instead of `TInt32`, to reduce the chance a future caller reads it as real data — the parameter case is already fixed and does not depend on this.
- Split bitwise and logical unary operation tests in every lowering/codegen layer.
