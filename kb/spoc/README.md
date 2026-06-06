# SPOC Component Knowledge Base

Source slice reviewed completely: `spoc/README.md`, all `spoc/framework/**`, all `spoc/ir/**`, all `spoc/registry/**`, and their tests.

## Component Inventory

- `spoc/README.md`: package-level architecture and usage notes for the SPOC SDK layer.
- `spoc/framework/`: backend plugin signatures, shared device/capability types, typed values, and backend error helpers.
- `spoc/ir/`: pure Sarek kernel IR type definitions, pretty-printers, and float64 usage analysis.
- `spoc/registry/`: runtime registry for primitive types, user records/variants, and intrinsic/device functions.
- `spoc/**/test/`: standalone unit tests for construction, lookup, pretty-printing, and error formatting.

## Subcomponent Map

- Framework details: [framework.md](framework.md)
- IR details: [ir.md](ir.md)
- Registry details: [registry.md](registry.md)
- Cross-component test coverage: [tests.md](tests.md)

## Key Features and APIs

- Backend abstraction is centralized in `spoc/framework/Framework_sig.ml`, including device capability records, launch dimensions, streams, events, memory, kernels, source generation, direct execution, external source execution, and typed `kargs` wrapping.
- Typed argument/value transport is in `spoc/framework/Typed_value.ml`, with scalar/composite existential wrappers and a separate global registry for scalar/composite type modules.
- Sarek IR is data-only in `spoc/ir/Sarek_ir_types.ml`, with helper functions for type-erased native vector access.
- Debug output and source-like renderings are in `spoc/ir/Sarek_ir_pp.ml`.
- Float64 detection is in `spoc/ir/Sarek_ir_analysis.ml`.
- Runtime type/function lookup is in `spoc/registry/Sarek_registry.ml`.

## Cross-Cutting Invariants

- `spoc/ir` is intended to be pure and dependency-light: `spoc/ir/dune:4-8` declares only the `sarek_ir` library modules and no external libraries.
- `spoc/framework` depends on `ctypes` and `sarek_ir`, not GPU runtimes: `spoc/framework/dune:1-8`.
- `spoc/registry` depends on `spoc_framework`: `spoc/registry/dune:4-8`.
- Device records expose backend identity as strings (`framework`) and capabilities as plain records: `spoc/framework/Framework_sig.ml:43-50`.
- Runtime registries are process-global mutable hash tables: `spoc/framework/Typed_value.ml:150-174` and `spoc/registry/Sarek_registry.ml:78-89`.

## Potential Invariant Violations or Bugs

- Launch dimensions can represent invalid GPU sizes. `dims_1d`, `dims_2d`, and `dims_3d` accept any `int`, including zero or negative values (`spoc/framework/Framework_sig.ml:17-23`), while tests cover only positive values (`spoc/framework/test/test_framework_sig.ml:16-35`).
- Global registries silently overwrite existing entries via `Hashtbl.replace` (`spoc/framework/Typed_value.ml:157-161`, `spoc/registry/Sarek_registry.ml:91-123`). That may hide duplicate PPX-generated registrations or load-order conflicts.
- Registry lookup by short record name returns the first hash-table fold match (`spoc/registry/Sarek_registry.ml:202-217`). If two modules register the same short name, resolution is nondeterministic.
- IR analysis does not inspect `SNative` GPU/OCaml bodies (`spoc/ir/Sarek_ir_analysis.ml:72-73`). This may miss float64 usage hidden in native code. Marked uncertain because native snippets may be intentionally opaque.
- Pretty-printer output is not a complete lossless representation of every IR node. For example, `EMatch` ignores concrete patterns in expression cases and prints `_` for each case (`spoc/ir/Sarek_ir_pp.ml:111-116`), and `SNative` prints only a placeholder (`spoc/ir/Sarek_ir_pp.ml:196-197`).

## Performance and Maintainability Risks

- Process-global mutable registries are simple but make test isolation, parallel execution, duplicate detection, and dynamic unloading hard.
- The IR has no validation layer for type consistency, variable uniqueness, mutability, lvalue legality, loop bounds, array memory spaces, or kernel parameter shape.
- Pretty-printing currently mixes debugging output with C-like syntax; consumers must not treat it as authoritative backend code without additional validation.
- The framework API is broad and stringly typed in places (`framework` names, source-language dispatch helpers, intrinsic names), so backend compatibility depends on convention.

## Related Tests

- Framework tests: `spoc/framework/test/test_framework_sig.ml`, `test_typed_value.ml`, `test_device_type.ml`, `test_backend_error.ml`.
- IR tests: `spoc/ir/test/test_sarek_ir_types.ml`, `test_sarek_ir_pp.ml`, `test_sarek_ir_analysis.ml`.
- Registry tests: `spoc/registry/test/test_sarek_registry.ml`.

## Missing Tests

- Invalid dimension construction and launch capability boundary checks.
- Duplicate registration behavior and deterministic conflict handling.
- Registry short-name ambiguity.
- Error formatting edge cases such as `Device_not_found` with `max_devices = 0`.
- Pretty-printer coverage for `EArrayReadExpr`, `EMatch` patterns, `SMatch`, `SWhile`, `SFor` downto semantics, `SBlock`, `SNative`, helper functions, and type/variant definitions.
- Float64 analysis coverage for lvalue types in assignments and `SNative` behavior.

## Concrete Improvement Candidates

- Add validated dimension constructors or a `validate_dims` helper used before backend launch.
- Change registry APIs to optionally reject duplicate names, return previous values, or record source/module ownership.
- Make short-name lookup return `Ok unique | Error ambiguous` rather than a nondeterministic first match.
- Add an IR validator module for structural invariants before backend code generation.
- Separate debugging pretty-printing from backend-like source rendering, or explicitly document that `Sarek_ir_pp` is non-authoritative.
