# spoc/ir

<!-- last-updated: 2026-07-02 -->

## Component Inventory

- `spoc/ir/README.md`: overview of the GPU kernel IR hierarchy, APIs, analysis, and testing.
- `spoc/ir/dune`: builds public library `spoc.ir` as unwrapped modules `Sarek_ir_types`, `Sarek_ir_pp`, `Sarek_ir_analysis`, `Sarek_ir_codegen`, and `Sarek_pure_registry` (`spoc/ir/dune:4-13`). No external libraries.
- `spoc/ir/Sarek_ir_types.ml`: pure IR type definitions, a functional `Type_id` runtime-witness module, plus native execution helper types.
- `spoc/ir/Sarek_ir_pp.ml`: string conversion and `Format` pretty-printers for IR nodes.
- `spoc/ir/Sarek_ir_analysis.ml`: float64 usage analysis across types, expressions, statements, declarations, helpers, and kernels.
- `spoc/ir/Sarek_ir_codegen.ml` (+ `.mli`): shared GPU code-generation helpers extracted from the backends to avoid duplicated variant/struct emission — `mangle_name`, `gen_variant_def` (C/MSL tagged-union variant emission for CUDA/OpenCL/Metal), and `gen_variant_def_glsl` (GLSL variant emission for Vulkan).
- `spoc/ir/Sarek_pure_registry.ml` (245 lines, down from 377 after the 2026-07-02 dedup — see "Sarek_pure_registry details" below): pure, path-qualified intrinsic device-name registry. Registers a `(module_path : string list, name : string) -> (framework:string -> string)` mapping (e.g. `(["Float32"], "sin") -> "sinf"` on CUDA) with **no** `Device.t` and no ctypes, so GPU code generators can resolve intrinsic device-code names without the FFI-backed `Sarek_registry`. See "Sarek_pure_registry details" below.
- `spoc/ir/test/*`: construction, printing, and float64 analysis tests. **No test file covers `Sarek_pure_registry.ml`** (`spoc/ir/test/dune` lists only `test_sarek_ir_types test_sarek_ir_pp test_sarek_ir_analysis`).

## Per-File Purpose

- `Sarek_ir_types.ml` defines memory spaces, element types, variables, constants, operators, expressions, lvalues, statements, declarations, helper functions, native arguments, native closures, native functions, and kernels (`spoc/ir/Sarek_ir_types.ml:11-245`), plus `module Type_id` (generative extensible GADT witnesses with an `equal : 'a t -> 'b t -> ('a, 'b) eq option` proof — no unsafe casts).
- `Sarek_ir_pp.ml` renders IR to debug/source-like strings (`spoc/ir/Sarek_ir_pp.ml:10-249`).
- `Sarek_ir_analysis.ml` recursively detects `TFloat64` and `CFloat64` use (`spoc/ir/Sarek_ir_analysis.ml:10-111`).
- `Sarek_ir_codegen.ml` emits backend-agnostic variant types: `mangle_name` normalizes type names into valid C/GLSL identifiers (`spoc/ir/Sarek_ir_codegen.mli:12`), `gen_variant_def` emits an enum + tagged-union struct + per-case constructor functions parameterised by `type_of_elttype` and `constructor_prefix` (`spoc/ir/Sarek_ir_codegen.mli:24-29`), and `gen_variant_def_glsl` emits the GLSL equivalent without enum/typedef/union (`spoc/ir/Sarek_ir_codegen.mli:38-42`).
- `Sarek_pure_registry.ml` builds framework-dispatching closures: `float32_math_template` picks the CUDA `f`-suffixed name (`sinf`), the GLSL-overridden name (via `glsl_override_name`: `fabs`/`abs_float`→`abs`, `rsqrt`→`inversesqrt`, `atan2`→`atan`), or a generic un-suffixed name for OpenCL/Metal/WGSL (`:57-66`); `generic_math_template` does the same for float64 math, which has no CUDA suffix form (`:71-75`). **Dedup (commit in #213, merged 2026-07-02):** the four `Float32`/`Float64` name lists that used to be repeated per registration block (`Float32`, `Math.Float32`, `Sarek_stdlib_meta.Float32`, `Sarek_stdlib_meta.Math.Float32`, and the `Float64` equivalents) are now defined once as shared static lists — `float32_list` (`:119`), `float64_list` (`:157`), `math_float64_list` (`:190`) — and each registration block iterates the shared list rather than repeating an inline name literal set. Registration surface (which `(module_path, name)` pairs get registered) is unchanged by the dedup; only the source-level duplication was removed.
- **GLSL `rsqrt` fix (same commit, merged 2026-07-02):** `Float64.rsqrt` now resolves through `glsl_override_name` (`:44-50`) via the same framework-aware template used for float32, so GLSL emits `inversesqrt` instead of the previously-invalid `rsqrt` (GLSL has no `rsqrt` builtin). Verified: `spoc/ir/Sarek_pure_registry.ml:44-50` maps `"rsqrt" -> Some "inversesqrt"` unconditionally for GLSL, independent of the float32/float64 split.
- `pp_kernel` prints a kernel header, params, locals, and body (`spoc/ir/Sarek_ir_pp.ml:236-247`).
- `kernel_uses_float64` combines parameter/local/body/helper/type/variant checks (`spoc/ir/Sarek_ir_analysis.ml:96-111`).

## Sarek_pure_registry details

- **Purpose**: the pure side of intrinsic resolution used exclusively by GPU code generators; `Sarek_registry` (in `spoc/registry/`) remains authoritative for native/interpreter paths (`spoc/ir/Sarek_pure_registry.ml:13-15`).
- **Registration invariant**: `register_fun` uses `Hashtbl.replace` on the `(module_path, name)` key (`spoc/ir/Sarek_pure_registry.ml:32-33`) — same silent-overwrite discipline as `Sarek_registry.register_*` and `Typed_value.Registry.register_*`. A duplicate `register_fun` call for the same path/name pair replaces the earlier device-code closure with no warning.
- **Unqualified names are intentionally absent**: only path-qualified entries (`["Float32"]`, `["Float64"]`, `["Math"; "Float32"]`, `["Math"; "Float64"]`, and their `Sarek_stdlib_meta`-prefixed twins) are registered; unqualified calls are still resolved by hardcoded match arms in each backend generator (`:89-92`).
- **Dedup landed 2026-07-02 (merged, commit in #213) — the 8 intrinsic tables are no longer independently-repeated inline literals.** `float32_list` (`:119`), `float64_list` (`:157`), and `math_float64_list` (`:190`) are now the single shared source for `register_float32_path`/`register_float64_path`/`register_math_float64_path`, each iterated once per module-path alias (`Float32`/`Float64`, `Math.Float32`/`Math.Float64`, and their `Sarek_stdlib_meta`-prefixed twins). This removes the surface-preserving copy/paste that the pre-2026-07-02 audit flagged as "4x drift risk" — there is now exactly one place to add or remove a float32 name, one for float64, and one for the `Math.Float64` subset, so the four aliases per family can no longer diverge from each other by editing mistake.
- **Float64.rsqrt GLSL fix** (same commit): resolved via `glsl_override_name` — see "Per-File Purpose" above. Was emitting the invalid GLSL token `rsqrt`; now emits `inversesqrt`.
- **Still-live, intentional gap — NOT fixed by the dedup: `math_float64_list` is a deliberate 16-entry strict subset of the 27-entry `float64_list`.** Verified against source 2026-07-02 (`spoc/ir/Sarek_pure_registry.ml:188-189` doc comment: *"The 16-entry Math.Float64 list — intentionally a strict subset of `float64_list`"*):
  - `float64_list` (`:157-186`, 27 entries) registers `sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, exp2, log, log2, log10, sqrt, rsqrt, cbrt, floor, ceil, round, trunc, fabs, pow, atan2, fma, min, max` under `["Float64"]` and `["Sarek_stdlib_meta"; "Float64"]`.
  - `math_float64_list` (`:190-206`, 16 entries) registers only `sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, floor, ceil, pow, atan2` under `["Math"; "Float64"]` and `["Sarek_stdlib_meta"; "Math"; "Float64"]` — still missing `exp2, log2, log10, rsqrt, cbrt, round, trunc, fabs, fma, min, max` (11 functions).
  - Practical effect (unchanged from pre-fix behavior): a kernel using `open Math.Float64` and calling e.g. `exp2`/`rsqrt`/`fma` will fail intrinsic lookup in `fun_device_template`, while the same call via the `Float64`-qualified or unqualified path resolves. The dedup commit made this an explicit, single-source, documented design choice (the source comment now states it is intentional) rather than accidental per-block drift — but the 11-function functional gap itself is a **tracked gap, not fixed**: it requires interpreter/stdlib support for those 11 names under the `Math.Float64` path, which does not currently exist.
  - **Correction to an earlier audit note (still valid)**: the four *float32* tables (`["Float32"]`, `["Math"; "Float32"]`, `["Sarek_stdlib_meta"; "Float32"]`, `["Sarek_stdlib_meta"; "Math"; "Float32"]`) all register the identical 32-name `float32_list` set including `expm1`, `log1p`, `hypot`, `copysign` — no drift, no gap.

## Invariants

- `var_id` is intended to be unique for alpha-renaming, while `var_name` remains human-readable (`spoc/ir/README.md:66-72`).
- `DParam` array information should agree with vector/array parameter shape (`spoc/ir/Sarek_ir_types.ml:134-142`).
- `SLet` uses immutable variables and `SLetMut` uses mutable variables by convention (`spoc/ir/Sarek_ir_types.ml:123-124`).
- `NativeFn` receives typed `native_arg array` plus block/grid dimensions and a `parallel` flag (`spoc/ir/Sarek_ir_types.ml:220-227`).
- Float64 analysis should conservatively return true if any reachable type or expression requires double precision.

## Potential Invariant Violations or Bugs

- Lvalue types are not inspected in `stmt_uses_float64`; `SAssign` checks only the right-hand expression (`spoc/ir/Sarek_ir_analysis.ml:53-56`). A statement assigning a non-float64 expression into a `TFloat64` variable could be missed unless the variable is declared elsewhere in the kernel. This matters for isolated `stmt_uses_float64` callers.
- `SNative` is treated as not using float64 (`spoc/ir/Sarek_ir_analysis.ml:72-73`). Marked uncertain because native code may be intentionally opaque, but this can under-report requirements.
- `EMatch` pretty-printing discards actual expression patterns and prints `_` for each case (`spoc/ir/Sarek_ir_pp.ml:111-116`).
- `SFor` pretty-printing uses `<` for `Upto` and `>` for `Downto` (`spoc/ir/Sarek_ir_pp.ml:146-162`), which excludes the stop expression. OCaml-style `to`/`downto` loops are usually inclusive, so the IR semantics need clarification.
- `pp_kernel` does not print `kern_types`, `kern_variants`, `kern_funcs`, or `kern_native_fn` (`spoc/ir/Sarek_ir_pp.ml:236-247`).
- **Stale claim withdrawn**: ~~native helpers use `Obj.magic`~~. `vec_get_custom`, `vec_set_custom`, and `vec_as_vector` now type-check element/underlying identity via `Sarek_ir_types.Type_id.equal` GADT `Refl` proofs (`spoc/ir/Sarek_ir_types.ml:221-255`) and raise `Failure` on witness mismatch; there is no `Obj.magic`/`Obj.` usage anywhere in `Sarek_ir_types.ml`. Type safety now depends on the `Type_id` module's generative-constructor discipline (each backend/PPX-generated custom type creates one fresh `witness` constructor) rather than an unchecked cast.
- The `Sarek_pure_registry.ml` `Math.Float64`-path tables (`math_float64_list`) under-register 11 functions relative to `float64_list` — **fixed 2026-07-02 (merged) as a drift risk** (dedup means the two lists can no longer diverge by accident), but the 11-function functional gap itself remains a **tracked, intentional omission** (see [Sarek_pure_registry details](#sarek_pure_registry-details) above), not a KB staleness issue.

## Performance and Maintainability Risks

- Recursive analysis and pretty-printing are straightforward but not tail-recursive for deeply nested expressions/statements.
- No validator enforces variable uniqueness, declaration consistency, field existence, constructor arity, array memory spaces, or statement expression types.
- Debug pretty-printing resembles C/CUDA syntax in places but is incomplete; maintainers could accidentally rely on it for code generation.
- Unwrapped modules (`spoc/ir/dune:7`) ease access but increase namespace collision risk.

## Related Tests

- `spoc/ir/test/test_sarek_ir_types.ml` covers construction of most variants and helpers.
- `spoc/ir/test/test_sarek_ir_pp.ml` covers primitive string conversions and representative expression/statement/declaration/kernel printing.
- `spoc/ir/test/test_sarek_ir_analysis.ml` covers float64 detection across many type, expression, statement, declaration, helper, and kernel shapes.
- **No test targets `Sarek_pure_registry.ml`** — absent from `spoc/ir/test/dune`, which lists only the three files above.

## Missing Tests

- `SAssign` where the lvalue type is `TFloat64` and expression is not.
- `SNative` analysis policy.
- Pretty-printing for `EArrayReadExpr`, `EMatch` patterns, `SMatch`, `SWhile`, `SFor Downto`, `SLetMut`, `SPragma`, `SBlock`, `SNative`, helper functions, record/variant definitions, and empty/non-empty tuple corner cases.
- Native helper failure paths (`vec_get_custom`, `vec_set_custom`, `vec_as_vector` on non-`NA_Vec`, including the `Type_id` mismatch `Failure` case).
- IR structural validation and semantic invariants.
- `Sarek_pure_registry.fun_device_template` per `(module_path, name, framework)` triple — would have caught the `Math.Float64` gap documented above — plus `register_fun` overwrite behavior and GLSL override coverage.

## Concrete Improvement Candidates

- Add `lvalue_uses_float64` and include it in `SAssign` analysis.
- Define and document exact `SFor` bound semantics; adjust pretty-printer/tests if inclusive loops are intended.
- Make `SNative` carry metadata such as required capabilities or `uses_float64`.
- Add an `Sarek_ir_validate` module returning structured errors for malformed IR.
- Rename or document `Sarek_ir_pp` as debug-only if it must not be used as source generation.
- **Partially done (2026-07-02, merged):** all four float64 tables now derive from shared lists (`float64_list`, `math_float64_list`), preventing re-drift. Still open: backfill `math_float64_list` with the 11 functions missing relative to `float64_list` (requires interpreter/stdlib support for those names under the `Math.Float64` path), or explicitly document the subset as permanent API surface if it will never be backfilled; add a `spoc/ir/test/test_sarek_pure_registry.ml`.
