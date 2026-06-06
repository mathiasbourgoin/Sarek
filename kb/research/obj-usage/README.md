# Obj Usage Audit

Audit date: 2026-05-05.

Baseline: `main` at `624246376973c586880dc933f6c7e113993604c8`. The worktree was already dirty before this pass with untracked `.agents/`, `.harness/`, and `kb/`; this audit only adds files under `kb/research/obj-usage/`.

Scope: every first-party OCaml `Obj` usage found with `rg "Obj\\.(magic|repr|obj)|Obj\\.t"` under `spoc/`, `sarek/`, backend packages, benchmarks, tools, and scripts. Comment-only "no Obj.t" mentions are not separate findings, but `Obj.t` public or internal API surfaces are covered when they enable executable `Obj` calls.

No active `Obj.field`, `Obj.set_field`, `Obj.tag`, `Obj.size`, or related low-level object-inspection APIs were found. The only active first-party APIs are `Obj.magic`, `Obj.repr`, `Obj.obj`, and explicit `Obj.t` type exposure. There are stale comments mentioning `Obj.set_field` in `Sarek_cpu_runtime`, but no executable use.

## Priority Index

1. [Typed `EXEC_VECTOR` boundary](typed-exec-vector-boundary.md) - P0. Replace the `internal_get_vector_obj : unit -> Obj.t` escape hatch and the callers that reconstruct vectors or backend buffers from it. This removes the broadest erasure boundary and several downstream casts.
2. [Typed native custom-vector witnesses](native-custom-vector-witnesses.md) - P0. Replace `get_any`/`set_any`/`get_vec` `Obj.t` fields and generated `vec_get_custom`/`vec_set_custom`/`vec_as_vector` casts with typed vector witness modules. This removes hot-path casts for custom records and typed native wrappers.
3. [Typed plugin buffer storage](plugin-bigarray-buffer-storage.md) - P1. Keep the Bigarray kind phantom in native/interpreter plugin buffers instead of erasing it and casting during blits.
4. [Interpreter launch typed values](interpreter-launch-typed-values.md) - P1. Replace byte-width type guessing and raw buffer extraction in the interpreter plugin launch path with `EXEC_VECTOR.get` or a dedicated typed buffer argument.
5. [Shared-memory type IDs](shared-memory-type-ids.md) - P1. Replace the generic shared-memory custom array cast with generated type IDs keyed by shared-memory name and type.
6. [Typed helper lookup](typed-helper-lookup.md) - P2. Replace rank-2 helper functions that accept/return arbitrary `'a` through `Obj` with typed helper lookup and dynamic value-only helpers.
7. [Explicit interpreter fallback kinds](interpreter-fallback-kinds.md) - P2. Replace `Char`/`Complex32` catch-all `Obj.magic` fallbacks in the interpreter vector bridge with explicit cases, matching `Execute.ml`.
8. [Legacy/test direct APIs](legacy-test-direct-apis.md) - P3. Remove or quarantine `Obj.t array` legacy native APIs and update tests to use typed vector element access.

## Complete Executable Obj Site Map

- `spoc/framework/Typed_value.ml:130`: `internal_get_vector_obj : unit -> Obj.t`.
- `spoc/ir/Sarek_ir_types.ml:173-176`: `get_any`, `set_any`, and `get_vec` use `Obj.t`.
- `spoc/ir/Sarek_ir_types.ml:194`, `201`, `217`: `Obj.magic` in custom/native vector helpers.
- `sarek/sarek/Execute.ml:53`: `Obj.repr v` when wrapping `Vector.t` as `EXEC_VECTOR`.
- `sarek/plugins/native/Native_plugin_base.ml:431`, `450`, `495`, `498`: `Obj.magic` on Bigarray buffers.
- `sarek/plugins/native/Native_plugin_base.ml:577`: `Obj.repr buf` when wrapping backend buffer as `EXEC_VECTOR`.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml:545`, `588`, `601`, `621`: backend buffer tunnel and byte-width guessing through `Obj`.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml:414`, `430`, `468`, `471`: same Bigarray buffer casts.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml:545`: `Obj.repr buf` when wrapping backend buffer as `EXEC_VECTOR`.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml:588`, `601`, `621`: `Obj.obj` and `Obj.magic` in interpreter launch vector-to-array conversion.
- `sarek/plugins/interpreter/Interpreter_plugin.ml:93`: `Obj.magic` from `EXEC_VECTOR` to `Kernel_arg.Vec`.
- `sarek/plugins/native/Native_plugin.ml:264-269`: `Obj.obj`, `Obj.repr`, and `Obj.obj` for custom vector access.
- `sarek/sarek/Kirc_kernel.ml:251-256`: same custom vector access bridge.
- `sarek/sarek/Sarek_cpu_runtime_types.ml:126` (`alloc_shared_with_key`): former `Obj.obj (Obj.repr arr)` in generic shared-memory custom arrays — now typed via `Type_id.Refl` (resolved; module split out of `Sarek_cpu_runtime.ml`).
- `sarek/sarek/Sarek_type_helpers.ml:65-66`: `Obj.obj (Obj.repr ...)` in untyped helper dispatch.
- `sarek/sarek/Sarek_ir_interp.ml:239` (`vector_to_array`), `274` (`array_to_vector`): former fallback vector element casts — no `Obj` present now (interpreter module split; bridge stayed in main).
- `sarek/plugins/native/Native_plugin.ml:384-385`: legacy `Obj.t array` direct execution type.
- `sarek/tests/new_runtime/test_native_runtime.ml:63`, `71`, `79`: test-only `Obj.obj` extraction.

## Existing Alternatives To Reuse

- GADT scalar and vector kinds in `sarek/core/Vector_types.ml:17-23` and `138-141`.
- Existential kernel arguments and typed folds in `sarek/core/Kernel_arg.ml:13-24` and `60-80`.
- Typed primitive/composite value modules in `spoc/framework/Typed_value.ml:24-101`.
- Existing `EXEC_VECTOR.get`/`set`/`device_ptr` methods in `spoc/framework/Typed_value.ml:108-130`.
- Explicit scalar vector conversion already implemented in `sarek/sarek/Execute.ml:349-437`.
- First-class module helper packing in `sarek/sarek/Sarek_type_helpers.ml:16-35`.
