# spoc/framework

<!-- last-updated: 2026-07-02 -->

## Component Inventory

- `spoc/framework/README.md`: public description of the backend plugin interface, typed values, and launch examples.
- `spoc/framework/Kernel_args.ml`/`.mli`: indexed kernel-argument container. Single shared implementation of "store a value at `idx`, last-set-wins, then validate the full set before launch," replacing each backend's hand-rolled accumulate-by-call-order scheme. Added 2026-07-02 (merged); consumed by `Native_plugin_base.ml` and `Interpreter_plugin_base.ml` (`spoc/framework/Kernel_args.mli:1-57`).
- `spoc/framework/Compile_cache.ml`/`.mli`: standardized compile-cache key builder for GPU backend plugins (CUDA/OpenCL/Metal/Vulkan). Added 2026-07-02 (merged); see "Compile_cache details" below.
- `spoc/framework/dune`: builds public library `spoc.framework` from `Framework_sig`, `Device_type`, `Typed_value`, and `Backend_error`, depending only on `sarek_ir` and `sarek_backend_error` — **ctypes-free** (`spoc/framework/dune:5`).
- `spoc/framework/ffi_free_gate/`: regression gate (`gate_framework.ml`) that mechanically enforces the ctypes-free invariant above. It builds as `.bc` and `.bc.js` (js_of_ocaml) and depends only on `spoc_framework`; if `ctypes` or `unix` re-enter `spoc_framework`, this target fails to build/link. It also pins the `Memory.host_ptr_to_device`/`device_to_host_ptr` boundary signatures to `nativeint` at compile time.
- `spoc/framework/Framework_sig.ml`: common SDK types, the `BACKEND` module type, and the shared low-level `PLUGIN_BASE` module type.
- `spoc/framework/Device_type.ml`: compatibility alias for `Framework_sig.device`.
- `spoc/framework/Typed_value.ml`: primitive storage, scalar/composite type interfaces, existential wrappers, execution arguments, and a typed-value registry.
- `spoc/framework/Backend_error.ml`: **13-line re-export** of `Sarek_backend_error.Backend_error` (`spoc/framework_error/Backend_error.ml`), kept only for backward compatibility with `Spoc_framework.Backend_error` call sites. New code should depend on `sarek_backend_error` directly. See [../spoc/framework_error](../spoc/framework_error) (source, not yet a separate KB page) for the actual error model, constructors, rendering, and the `Make` functor.
- `spoc/framework/Backend_error.md`: usage and migration guide for the shared error model.
- `spoc/framework/test/*`: unit tests for the above modules.

## Per-File Purpose

- `Framework_sig.ml` defines `dims`, device `capabilities`, `device`, minimal plugin signature `S`, execution model/source language enums, extensible `kargs`, external source arguments, intrinsic registry signature, the full `BACKEND` contract (`spoc/framework/Framework_sig.ml:16-350`), and the shared low-level `PLUGIN_BASE` module type covering device/stream/memory/event/kernel FFI bindings on top of which each backend assembles its full `BACKEND` (`spoc/framework/Framework_sig.ml:362-493`).
- `Device_type.ml` preserves older API compatibility with a manifest alias to `Framework_sig.device` (`spoc/framework/Device_type.ml:13-20`).
- `Typed_value.ml` provides typed scalar/composite values and `exec_arg` variants without relying on `Obj.t`/`Obj.magic` anywhere in the module (`spoc/framework/Typed_value.ml:24-145`), plus global scalar/composite registries (`spoc/framework/Typed_value.ml:158-182`).
- `Backend_error.ml` is a 13-line `include Sarek_backend_error.Backend_error`. The real definitions live in `spoc/framework_error/Backend_error.ml` (library `spoc.backend-error`, which has no dependency on ctypes, `spoc_core`, or `Device.t`): codegen/runtime/plugin error categorization (`spoc/framework_error/Backend_error.ml:21-75`), rendering (`:160-268`), exception/result helpers (`:271-289`), and the `Make` backend-name functor (`:291-357`).
- `Kernel_args.ml`/`.mli` stores `'a t` values keyed by non-negative index (`set`/`count`/`to_sorted_list`/`validate_and_extract`, `spoc/framework/Kernel_args.mli:20-57`). `validate_and_extract t ~expected_count` requires exactly indices `0 .. expected_count - 1` with no gaps and none at/beyond `expected_count`, returning `Error msg` (not raising) on a negative `expected_count` or on gaps/duplicates/out-of-range indices.
- `Compile_cache.ml`/`.mli` provides `make_key ~device ~name ~source ?options ()`, returning a `':'`-joined string of four independently-MD5-digested components (device, name, source, canonicalized/sorted options).

## Features and APIs

- Dimension helpers: `dims_1d`, `dims_2d`, `dims_3d` (`spoc/framework/Framework_sig.ml:17-23`).
- Device capability and identity records (`spoc/framework/Framework_sig.ml:25-50`).
- Backend modules must provide device, stream, memory, event, kernel, profiling, source generation, direct execution, intrinsic registry, external source execution, and `kargs` wrapping (`spoc/framework/Framework_sig.ml:137-350`).
- `PLUGIN_BASE` is the shared low-level backend interface implemented by the CUDA, OpenCL, and Metal `*_plugin_base.ml` modules; it factors out the previously per-backend inline device/stream/memory/event/kernel signature (`spoc/framework/Framework_sig.ml:362-493`).
- `Typed_value` built-ins register `int32`, `int64`, `float32`, `float64`, and `bool` on module load (`spoc/framework/Typed_value.ml:188-264`).
- `typed_value_of_exec_arg`, `exec_arg_of_typed_value`, and `type_name_of_exec_arg` bridge runtime arguments and stored typed values (`spoc/framework/Typed_value.ml:269-307`; `typed_value_of_exec_arg` itself is `:274-281`).
- `EXEC_VECTOR.get_typed`/`set_typed`/`type_id`/`underlying_type_id` give type-preserving element access via `Sarek_ir_types.Type_id.t` runtime witnesses (`spoc/framework/Typed_value.ml:105-139`); backends read/write vector elements without `Obj.t`.
- `Backend_error.to_result`, `with_default`, `raise_error`, `print_error`, and `result_to_string` support both exception and result workflows (re-exported from `spoc/framework_error/Backend_error.ml:271-289`).
- `Backend_error.Make(B).check ~is_success ~to_string ctx result` (`spoc/framework_error/Backend_error.ml:344-347`, added 2026-07-02, merged): shared FFI check funnel replacing each backend's hand-rolled `check` function (`Cuda_api.check`, `Opencl_api.check`, `Metal_api.check`, `Vulkan_api_base.check`). Raises canonical `Backend_error` (`context_error ctx (to_string result)`) via `raise_error` when `is_success result` is false; backends supply their own success predicate/stringifier since underlying FFI result types differ (`cu_result`/`cl_error`/`mtl_error`/`vk_result`), but every backend's funnel now raises the same exception shape.
- `Kernel_args.set`/`validate_and_extract` give backends a strict-launch-validation invariant: values may be set in any order or overwritten by index (last-set-wins), but `validate_and_extract` is the single gate that rejects gaps, duplicates (collapsed to last-set), negative indices, and indices `>= expected_count` before a kernel launches, returning `Error msg` rather than raising.

## Compile_cache details

- **Key shape**: `"<device-digest>:<name-digest>:<source-digest>:<options-digest>"`, each field an independent 32-hex-char `Digest.string` MD5 digest (`spoc/framework/Compile_cache.mli:22-26`).
- **Unambiguous by construction**: because every component is digested *before* being joined with `':'`, a raw value containing `':'` (in `device`) or `','`/`'='` (in `options`) cannot shift a byte across the field boundary and collide with an unrelated `(device, name, source, options)` tuple — the digest step, not the delimiter, is what prevents ambiguity (`spoc/framework/Compile_cache.mli:28-33`).
- **Why `name` is required**: a single source file frequently defines more than one kernel entry point; omitting the kernel/entry name from the key means the second kernel compiled from a shared source silently resolves to whatever was compiled first under the same key (`spoc/framework/Compile_cache.mli:8-12`).
- **Options canonicalization**: the options association list is sorted by key and joined into one canonical string before digesting, so option order never affects cache hits (`spoc/framework/Compile_cache.mli:42-45`).
- Does not force a shared `Hashtbl` — each backend keeps its own cache table; the module's only contract is the key *shape* (`spoc/framework/Compile_cache.mli:17-20`).

## Invariants

- `Device_type.t` must remain exactly the same type as `Framework_sig.device` (`spoc/framework/Device_type.ml:13-20`).
- Backend `wrap_kargs`/`unwrap_kargs` are expected to be inverse for the backend's own `Kernel.args` and return `None` for other backend variants (`spoc/framework/Framework_sig.ml:340-349`).
- `SCALAR_TYPE.of_primitive` is expected to accept only the matching primitive representation; built-ins fail on mismatches (`spoc/framework/Typed_value.ml:188-264`).
- `EXEC_VECTOR.get` and `set` traffic values through `typed_value`, while `get_typed`/`set_typed` traffic the underlying element type directly (checked at the call site by matching `type_id` via `Type_id.equal`), and `device_ptr`/`elem_size` expose binding data (`spoc/framework/Typed_value.ml:105-139`).
- `Backend_error.Make` should stamp every constructor with the backend name captured in the functor argument (`spoc/framework_error/Backend_error.ml:291-357`).
- `spoc/framework` must not depend on `ctypes` or `unix`, enforced by `spoc/framework/ffi_free_gate/gate_framework.ml`.

## Potential Invariant Violations or Bugs

- `dims_*` constructors do not reject zero or negative values (`spoc/framework/Framework_sig.ml:17-23`). Invalid dimensions could flow to backend `Kernel.launch` (`spoc/framework/Framework_sig.ml:270-277`).
- `Device_not_found` renders available range as `0-(max_devices - 1)` (`spoc/framework_error/Backend_error.ml:203-208` — moved from `spoc/framework/Backend_error.ml`, same code, still live). With `max_devices = 0`, the message becomes `0--1`; there is no edge-case test.
- `Typed_value.Registry.register_scalar` and `register_composite` silently overwrite existing names (`spoc/framework/Typed_value.ml:165-169`). That can hide duplicate generated type modules. Still live.
- `primitive_type_name` maps any `PFloat` to `"float"` (`spoc/framework/Typed_value.ml:287-292`), losing float32/float64 distinction when only the primitive remains. This is probably intentional for primitive storage, but it is a footgun if used as a type name.
- ~~The module claims no `Obj.t` in normal typed value transport, but `EXEC_VECTOR.internal_get_vector_obj` explicitly exposes `Obj.t`.~~ **Resolved / stale claim withdrawn**: `internal_get_vector_obj` no longer exists. `EXEC_VECTOR` now exposes typed `get_typed : int -> elt`, `set_typed : int -> elt -> unit`, and `type_id : elt Sarek_ir_types.Type_id.t` (`spoc/framework/Typed_value.ml:122-130`); callers prove element-type equality via `Type_id.equal` (GADT `Refl` witness, see `spoc/ir/Sarek_ir_types.ml:221-255`) instead of an `Obj.t` escape hatch. No `Obj.` usage exists anywhere in `Typed_value.ml` or `Sarek_ir_types.ml`.
- `typed_value_of_exec_arg` raises on `EA_Vec` (`spoc/framework/Typed_value.ml:274-281`). That is explicit behavior but currently untested.

## Performance and Maintainability Risks

- The `BACKEND` signature is broad; any change is a cross-backend migration.
- Global mutable registries are not synchronized and have no reset API, which complicates parallel tests and long-lived processes.
- Backend identity uses free-form strings (`device.framework`, `cuda_or_opencl`-style code elsewhere), so typos are not caught by the type system.
- Error rendering embeds source previews in `Compilation_failed` (`spoc/framework_error/Backend_error.ml:209-218`); useful for debugging but potentially noisy or sensitive.

## Related Tests

- `spoc/framework/test/test_framework_sig.ml` checks positive dimension helpers, capability/device construction, enum distinctness, and basic `exec_arg`/`run_source_arg` variants.
- `spoc/framework/test/test_device_type.ml` verifies alias compatibility and representative device records.
- `spoc/framework/test/test_typed_value.ml` covers primitive variants, built-in scalar round trips, scalar registry lookup/listing, custom scalar registration, scalar typed values, and selected exec-arg conversions.
- `spoc/framework/test/test_backend_error.ml` covers representative formatting and exception/result helpers.

## Missing Tests

- Invalid dimensions and capability boundary validation.
- `typed_value_of_exec_arg (EA_Vec _)` failure behavior.
- Composite type registration/listing and duplicate scalar/composite registrations.
- `primitive_type_name` expectations for `PFloat`.
- `Backend_error.result_to_string`, `print_error`, all error constructors, source preview truncation, and zero-device ranges.
- `BACKEND` mock implementation compile test that exercises every required member.

## Concrete Improvement Candidates

- Introduce `validate_dims : dims -> (unit, string) result` or checked constructors for launch grid/block dimensions.
- Add duplicate-aware registry APIs such as `register_scalar_exn` or `register_scalar_result`.
- Document `PFloat` as intentionally ambiguous or split it into `PFloat32` and `PFloat64`.
- Add a small mock backend in tests to compile against the full `BACKEND` signature.
- Improve `Device_not_found` rendering for `max_devices <= 0`.
