# SPOC Component Knowledge Base

<!-- last-updated: 2026-07-02 -->

Source slice reviewed completely: `spoc/README.md`, all `spoc/framework/**`, all `spoc/framework_error/**`, all `spoc/ir/**`, all `spoc/registry/**`, and their tests.

## Component Inventory

- `spoc/README.md`: package-level architecture and usage notes for the SPOC SDK layer.
- `spoc/framework/`: backend plugin signatures, shared device/capability types, typed values. `Backend_error.ml` is now a 13-line re-export of `spoc/framework_error/Backend_error.ml` kept for backward compatibility. Also includes `Kernel_args.ml`/`.mli` (indexed kernel-argument container, added 2026-07-02, merged) and `Compile_cache.ml`/`.mli` (standardized compile-cache key builder, added 2026-07-02, merged) — see [framework.md](framework.md).
- `spoc/framework_error/`: pure, ctypes-free backend error model (library `spoc.backend-error` / module `Sarek_backend_error.Backend_error`); no dependency on ctypes, `spoc_core`, or `Device.t`, so it is safe in bytecode-only and jsoo targets. This is the authoritative definition; `spoc/framework/Backend_error.ml` only re-exports it.
- `spoc/ir/`: pure Sarek kernel IR type definitions, pretty-printers, float64 usage analysis, shared codegen helpers, and the pure path-qualified intrinsic registry (`Sarek_pure_registry.ml`).
- `spoc/registry/`: runtime registry for primitive types, user records/variants, and intrinsic/device functions. Depends on no other spoc libraries (`spoc/registry/dune:7` — `libraries` is empty).
- `spoc/**/test/`: standalone unit tests for construction, lookup, pretty-printing, and error formatting.

## Subcomponent Map

- Framework details: [framework.md](framework.md)
- IR details: [ir.md](ir.md)
- Registry details: [registry.md](registry.md)
- Cross-component test coverage: [tests.md](tests.md)

## Key Features and APIs

- Backend abstraction is centralized in `spoc/framework/Framework_sig.ml`, including device capability records, launch dimensions, streams, events, memory, kernels, source generation, direct execution, external source execution, and typed `kargs` wrapping.
- Typed argument/value transport is in `spoc/framework/Typed_value.ml`, with scalar/composite existential wrappers and a separate global registry for scalar/composite type modules.
- Sarek IR is data-only in `spoc/ir/Sarek_ir_types.ml`, plus a functional `Type_id` module (generative GADT witnesses with an `equal` proof, no `Obj.magic`) used for type-safe native vector access; the module remains dependency-free.
- Debug output and source-like renderings are in `spoc/ir/Sarek_ir_pp.ml`.
- Float64 detection is in `spoc/ir/Sarek_ir_analysis.ml`.
- Path-qualified pure intrinsic device-name dispatch (framework-string keyed, no `Device.t`/ctypes) is in `spoc/ir/Sarek_pure_registry.ml`.
- Runtime type/function lookup is in `spoc/registry/Sarek_registry.ml`.
- Structured backend error model is in `spoc/framework_error/Backend_error.ml` (library `spoc.backend-error`), re-exported for compatibility from `spoc/framework/Backend_error.ml`.

## Cross-Cutting Invariants

- `spoc/ir` is intended to be pure and dependency-light: `spoc/ir/dune:4-13` declares only the `sarek_ir` library modules (including `Sarek_pure_registry`) and no external libraries.
- `spoc/framework` is ctypes-free: `spoc/framework/dune:5` lists only `sarek_ir sarek_backend_error` as libraries. This is mechanically enforced by `spoc/framework/ffi_free_gate/gate_framework.ml`, a `.bc`/`.bc.js` executable that depends only on `spoc_framework` and fails to build (or link as jsoo) if `ctypes` or `unix` re-enter the library.
- `spoc/framework_error` is likewise ctypes-free and has no dependency on `spoc_core` or `Device.t` (`spoc/framework_error/dune`).
- `spoc/registry` depends on no other spoc library: `spoc/registry/dune:7` declares an empty `libraries` list (it does **not** depend on `spoc_framework`).
- Device records expose backend identity as strings (`framework`) and capabilities as plain records: `spoc/framework/Framework_sig.ml:43-50`.
- Runtime registries are process-global mutable hash tables: `spoc/framework/Typed_value.ml:158-182`, `spoc/registry/Sarek_registry.ml:78-89`, and `spoc/ir/Sarek_pure_registry.ml:24-38`.

## Potential Invariant Violations or Bugs

- Launch dimensions can represent invalid GPU sizes. `dims_1d`, `dims_2d`, and `dims_3d` accept any `int`, including zero or negative values (`spoc/framework/Framework_sig.ml:17-23`), while tests cover only positive values (`spoc/framework/test/test_framework_sig.ml:16-35`).
- Global registries silently overwrite existing entries via `Hashtbl.replace` (`spoc/framework/Typed_value.ml:165-169`, `spoc/registry/Sarek_registry.ml:91-123`, `spoc/ir/Sarek_pure_registry.ml:32-33`). That may hide duplicate PPX-generated registrations or load-order conflicts. Still live.
- Registry lookup by short record name returns the first hash-table fold match (`spoc/registry/Sarek_registry.ml:202-217`). If two modules register the same short name, resolution is nondeterministic.
- IR analysis does not inspect `SNative` GPU/OCaml bodies (`spoc/ir/Sarek_ir_analysis.ml:72-73`). This may miss float64 usage hidden in native code. Marked uncertain because native snippets may be intentionally opaque.
- Pretty-printer output is not a complete lossless representation of every IR node. For example, `EMatch` ignores concrete patterns in expression cases and prints `_` for each case (`spoc/ir/Sarek_ir_pp.ml:111-116`), and `SNative` prints only a placeholder (`spoc/ir/Sarek_ir_pp.ml:196-197`).
- **Fixed 2026-07-02 (merged, commit in #213) — `Sarek_pure_registry.ml`'s 8 intrinsic tables are now deduplicated into 3 shared static lists** (`float32_list`, `float64_list`, `math_float64_list`; `spoc/ir/Sarek_pure_registry.ml:119,157,190`), removing the copy/paste drift risk across the `Float32`/`Float64`/`Math.*`/`Sarek_stdlib_meta.*` aliases. The same commit fixed `Float64.rsqrt` on GLSL to emit `inversesqrt` (was the invalid token `rsqrt`) via the framework-aware `glsl_override_name` template. **Still a tracked, intentional gap (NOT fixed):** `math_float64_list` remains a deliberate 16-entry strict subset of the 27-entry `float64_list` (source comment: "intentionally a strict subset", `:188-189`) — `["Math";"Float64"]` and `["Sarek_stdlib_meta";"Math";"Float64"]` still lack `exp2`, `log2`, `log10`, `rsqrt`, `cbrt`, `round`, `trunc`, `fabs`, `fma`, `min`, `max` relative to `["Float64"]`/`["Sarek_stdlib_meta";"Float64"]`. Kernels calling `Math.Float64.exp2` (etc.) will still fail intrinsic lookup where the equivalent unqualified/`Float64`-path call succeeds; backfilling requires interpreter/stdlib support for those 11 names under the `Math.Float64` path, which does not currently exist. See [ir.md](ir.md) for full detail. Verified against source 2026-07-02; the four *float32* tables remain **not** drifted — all four register the same 32 names including `expm1`/`log1p`/`hypot`/`copysign`.

## Performance and Maintainability Risks

- Process-global mutable registries are simple but make test isolation, parallel execution, duplicate detection, and dynamic unloading hard.
- The IR has no validation layer for type consistency, variable uniqueness, mutability, lvalue legality, loop bounds, array memory spaces, or kernel parameter shape.
- Pretty-printing currently mixes debugging output with C-like syntax; consumers must not treat it as authoritative backend code without additional validation.
- The framework API is broad and stringly typed in places (`framework` names, source-language dispatch helpers, intrinsic names), so backend compatibility depends on convention.

## Related Tests

- Framework tests: `spoc/framework/test/test_framework_sig.ml`, `test_typed_value.ml`, `test_device_type.ml`, `test_backend_error.ml`.
- IR tests: `spoc/ir/test/test_sarek_ir_types.ml`, `test_sarek_ir_pp.ml`, `test_sarek_ir_analysis.ml`.
- Registry tests: `spoc/registry/test/test_sarek_registry.ml`.
- **No test file exists for `Sarek_pure_registry.ml`** — `spoc/ir/test/dune` names only `test_sarek_ir_types test_sarek_ir_pp test_sarek_ir_analysis`; the dedup landed 2026-07-02 removed the copy/paste drift risk, but the tables (and the still-open `Math.Float64` 11-function gap) remain untested and would not be caught by CI.

## Missing Tests

- Invalid dimension construction and launch capability boundary checks.
- Duplicate registration behavior and deterministic conflict handling.
- Registry short-name ambiguity.
- Error formatting edge cases such as `Device_not_found` with `max_devices = 0`.
- Pretty-printer coverage for `EArrayReadExpr`, `EMatch` patterns, `SMatch`, `SWhile`, `SFor` downto semantics, `SBlock`, `SNative`, helper functions, and type/variant definitions.
- Float64 analysis coverage for lvalue types in assignments and `SNative` behavior.
- `Sarek_pure_registry.fun_device_template` coverage for every registered `module_path`/name pair per framework string (would have caught the `Math.Float64` gap above), plus GLSL override behavior (`fabs`/`abs_float` → `abs`, `rsqrt` → `inversesqrt`, `atan2` → `atan`) and unregistered-path `None` behavior.

## Concrete Improvement Candidates

- Add validated dimension constructors or a `validate_dims` helper used before backend launch.
- Change registry APIs to optionally reject duplicate names, return previous values, or record source/module ownership.
- Make short-name lookup return `Ok unique | Error ambiguous` rather than a nondeterministic first match.
- Add an IR validator module for structural invariants before backend code generation.
- Separate debugging pretty-printing from backend-like source rendering, or explicitly document that `Sarek_ir_pp` is non-authoritative.
- **Partially done (2026-07-02, merged):** all four float64 tables now derive from shared lists, preventing re-drift. Still open: backfill the `Math.Float64`/`Sarek_stdlib_meta.Math.Float64` table gaps in `Sarek_pure_registry.ml` (add `exp2`, `log2`, `log10`, `rsqrt`, `cbrt`, `round`, `trunc`, `fabs`, `fma`, `min`, `max`), which requires interpreter/stdlib support for those names first.
- Add a test file for `Sarek_pure_registry.ml` to `spoc/ir/test/dune`.
