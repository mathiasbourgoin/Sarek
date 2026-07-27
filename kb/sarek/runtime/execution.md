# Execution Layer

## Component Inventory

Reviewed execution-facing files under `sarek/sarek/**` and, following the 2026-07-02 directory move noted below, `sarek/execute/**`: `README.md`, `BSP.md`, `Execute.ml`, `Execute_error.ml`, `Kirc_error.ml`, `Kirc_kernel.ml`, `Kirc_types.ml`, `Kirc_types.mli`, `Sarek_ir.ml`, `Sarek_value.ml`, `Sarek_type_helpers.ml`, `Skeletons.ml`, `Sarek_float32.ml`, `Sarek_float32.mli`, error modules, and related tests. CPU runtime, interpreter, and fusion have separate pages.

**Directory move (2026-07-02 audit):** `Execute.ml`/`Execute_error.ml` now live in their own dune library `sarek_execute` (`public_name sarek.execute`, `sarek/execute/dune`), extracted so a jsoo target can link `sarek_execute` against a ctypes-free `spoc_core` instance. `sarek/sarek/Execute.ml` and `sarek/sarek/Execute_error.ml` are now 6-line forwarding shims (`include Sarek_execute.Execute` / `include Sarek_execute.Execute_error`). `Kirc_types.ml`, `Sarek_ir.ml`, `Sarek_value.ml`, `Sarek_type_helpers.ml`, `Kirc_kernel.ml`, `Kirc_error.ml`, and `Sarek_float32.ml`/`.mli` were **not** moved — they remain live in `sarek/sarek/` (still present in `sarek/sarek/dune`'s `(modules ...)` list). **`Skeletons.ml` removed 2026-07-02 (merged, commit 7ca31a76):** the byte-identical dead copy formerly excluded from `sarek/sarek/dune`'s modules list has been deleted outright — see [interpreter.md](interpreter.md) for the corresponding note.

## Per-File Purpose

- `sarek/sarek/README.md`: KIRC/Sarek runtime overview and usage examples.
- `sarek/sarek/BSP.md`: intended BSP/barrier semantics for CPU execution.
- `sarek/execute/Execute.ml`: runtime dispatch, kernel argument conversion, vector transfer integration, native/interpreter/backend selection, and stale marking. (`sarek/sarek/Execute.ml` is now a 6-line forwarding shim.)
- `sarek/execute/Execute_error.ml`: execution error variants and formatting. (`sarek/sarek/Execute_error.ml` is now a 6-line forwarding shim.)
- `sarek/sarek/Kirc_types.ml` and `Kirc_types.mli`: KIRC type, expression, statement, declaration, and kernel AST definitions.
- `sarek/sarek/Sarek_ir.ml`: lower-level IR representation used by interpreter/fusion paths.
- `sarek/sarek/Sarek_value.ml`: boxed runtime values and conversion helpers.
- `sarek/sarek/Sarek_type_helpers.ml`: type helper registry and existential bridges.
- `sarek/sarek/Kirc_kernel.ml`: kernel object construction, metadata, argument helpers, and run entrypoints.
- `sarek/sarek/Kirc_error.ml`, `Interp_error.ml`, `Fusion_error.ml`: domain error modules.
- `sarek/sarek/Skeletons.ml`: **deleted 2026-07-02 (merged, commit 7ca31a76)** — was a dead, byte-identical unused copy excluded from `sarek/sarek/dune`'s modules list; now removed outright rather than merely excluded.
- `sarek/sarek/Sarek_float32.ml` and `.mli`: runtime float32 helper surface used by Sarek code.
- `sarek/sarek/test/**`: mostly value, float32, type-helper, and error formatting tests.

## Features/APIs

- Kernel construction and metadata around KIRC declarations.
- Execution entrypoints that can dispatch to registered backends, interpreter fallback, or CPU runtime paths.
- Runtime vector-argument handling, host/device transfer before launch, and stale marking after launch.
- Scalar and vector execution argument conversion.
- Runtime value representation for interpreter and helper paths.
- Type-helper registry to bridge generated code and runtime vectors.

## Invariants

- Execution must transfer vector arguments to the chosen backend before passing raw buffers.
- Writable vector arguments must be marked stale on host after a device backend mutates them.
- Argument conversion must preserve element type, shape, and order.
- Kernel parameters and supplied arguments must match without relying on generic `Invalid_argument` exceptions.
- Type-helper lookups must not mix incompatible element types.

## Potential Invariant Violations/Bugs

- `Execute.run_vectors` transfers vectors and dispatches at `sarek/execute/Execute.ml:520-543`, but stale marking only updates vectors in `Both d` state and skips native devices at `sarek/execute/Execute.ml:321-335`. This may be intentional for native host execution, but it is worth validating for vectors already in `GPU` or `Stale_CPU` state. Uncertain.
- **Fixed as of 2026-07-02** (was: custom vector element reads returned a placeholder `Float32 0.0`): the custom-element path now round-trips through `Vector.custom_to_bytes`/`Vector.custom_of_bytes` — `custom_value_to_bytes`/`custom_value_of_bytes` and the `EXEC_VECTOR.get`/`set` cases for `Vector.Custom custom` at `sarek/execute/Execute.ml:37-171` construct a real `Typed_value.COMPOSITE_TYPE` wrapper around the custom type's own byte conversion, instead of returning a stub value or failing on write.
- `Kirc_kernel.run_with_args` calls `Execute.run` directly at `sarek/sarek/Kirc_kernel.ml:306-318`, not `Execute.run_vectors`. If callers use this path with device backends, vectors may not be transferred or stale-marked before backend buffer expansion in `sarek/execute/Execute.ml:170-215`. Uncertain because generated callers may prefer a different entrypoint.
- `Sarek_type_helpers` relies on `Obj` for existential bridges at `sarek/sarek/Sarek_type_helpers.ml:53-67`. This is probably intentional, but there is no runtime type identity check if a helper name is reused for an incompatible element type.

## Performance Or Maintainability Risks

- Execution has several partially overlapping paths: raw `Execute.run`, `run_vectors`, `run_cpu`, interpreter direct execution, and plugin compatibility launches.
- Native-device stale marking is special-cased, so future native-like backends must understand whether they mutate host arrays or backend buffers.
- Type helper registration by string name is flexible but fragile without uniqueness/type guards.

## Related Tests

- `sarek/sarek/test/test_sarek_value.ml`: runtime value behavior.
- `sarek/sarek/test/test_sarek_float32.ml`: float32 helper behavior.
- `sarek/sarek/test/test_sarek_type_helpers.ml`: type helper registration and lookup.
- `sarek/execute/test/test_execute_error.ml` (or colocated equivalent): execution error formatting.
- `sarek/sarek/test/test_kirc_error.ml`, `test_interp_error.ml`, `test_fusion_error.ml`: error modules.

## Missing Tests

- `Execute.run_vectors` transfer and stale-mark behavior for CPU, native, interpreter, CUDA/OpenCL-like dummy devices, and pre-stale vectors.
- `Kirc_kernel.run_with_args` with vector arguments on a backend requiring device buffers.
- Custom vector element read/write behavior through the now-real `custom_to_bytes`/`custom_of_bytes` round trip (regression coverage for the 2026-07-02 fix, e.g. mismatched custom-type name/size).
- Type-helper name collision and wrong-type helper retrieval.
- Parameter count/type mismatch errors that assert structured runtime errors rather than generic exceptions.

## Concrete Improvement/Fix Candidates

- Route vector-aware kernel execution through `Execute.run_vectors` by default, or explicitly document that `Execute.run` and `Kirc_kernel.run_with_args` require pre-transferred arguments.
- Add explicit stale-mark semantics for native devices and test both host-mutating and buffer-mutating backends.
- Include type identity or phantom witness checks in the type-helper registry.
- Normalize argument mismatch failures into `Execute_error` variants.
