# spoc/framework

## Component Inventory

- `spoc/framework/README.md`: public description of the backend plugin interface, typed values, and launch examples.
- `spoc/framework/dune`: builds public library `spoc.framework` from `Framework_sig`, `Device_type`, `Typed_value`, and `Backend_error`, depending on `ctypes` and `sarek_ir` (`spoc/framework/dune:1-8`).
- `spoc/framework/Framework_sig.ml`: common SDK types, the `BACKEND` module type, and the shared low-level `PLUGIN_BASE` module type.
- `spoc/framework/Device_type.ml`: compatibility alias for `Framework_sig.device`.
- `spoc/framework/Typed_value.ml`: primitive storage, scalar/composite type interfaces, existential wrappers, execution arguments, and a typed-value registry.
- `spoc/framework/Backend_error.ml`: structured backend error model, constructors, rendering, exception helpers, and a backend-name functor.
- `spoc/framework/Backend_error.md`: usage and migration guide for the shared error model.
- `spoc/framework/test/*`: unit tests for the above modules.

## Per-File Purpose

- `Framework_sig.ml` defines `dims`, device `capabilities`, `device`, minimal plugin signature `S`, execution model/source language enums, extensible `kargs`, external source arguments, intrinsic registry signature, the full `BACKEND` contract (`spoc/framework/Framework_sig.ml:16-350`), and the shared low-level `PLUGIN_BASE` module type covering device/stream/memory/event/kernel FFI bindings on top of which each backend assembles its full `BACKEND` (`spoc/framework/Framework_sig.ml:362-493`).
- `Device_type.ml` preserves older API compatibility with a manifest alias to `Framework_sig.device` (`spoc/framework/Device_type.ml:13-20`).
- `Typed_value.ml` provides typed scalar/composite values and `exec_arg` variants without relying on `Obj.t` for normal scalar/composite transport (`spoc/framework/Typed_value.ml:24-145`), plus global scalar/composite registries (`spoc/framework/Typed_value.ml:150-174`).
- `Backend_error.ml` categorizes backend failures into codegen, runtime, and plugin errors (`spoc/framework/Backend_error.ml:20-75`), renders them (`spoc/framework/Backend_error.ml:159-268`), and provides `Make` for backend-specific constructors (`spoc/framework/Backend_error.ml:291-360`).

## Features and APIs

- Dimension helpers: `dims_1d`, `dims_2d`, `dims_3d` (`spoc/framework/Framework_sig.ml:17-23`).
- Device capability and identity records (`spoc/framework/Framework_sig.ml:25-50`).
- Backend modules must provide device, stream, memory, event, kernel, profiling, source generation, direct execution, intrinsic registry, external source execution, and `kargs` wrapping (`spoc/framework/Framework_sig.ml:137-350`).
- `PLUGIN_BASE` is the shared low-level backend interface implemented by the CUDA, OpenCL, and Metal `*_plugin_base.ml` modules; it factors out the previously per-backend inline device/stream/memory/event/kernel signature (`spoc/framework/Framework_sig.ml:362-493`).
- `Typed_value` built-ins register `int32`, `int64`, `float32`, `float64`, and `bool` on module load (`spoc/framework/Typed_value.ml:180-267`).
- `typed_value_of_exec_arg`, `exec_arg_of_typed_value`, and `type_name_of_exec_arg` bridge runtime arguments and stored typed values (`spoc/framework/Typed_value.ml:270-309`).
- `Backend_error.to_result`, `with_default`, `raise_error`, `print_error`, and `result_to_string` support both exception and result workflows (`spoc/framework/Backend_error.ml:270-285`).

## Invariants

- `Device_type.t` must remain exactly the same type as `Framework_sig.device` (`spoc/framework/Device_type.ml:13-20`).
- Backend `wrap_kargs`/`unwrap_kargs` are expected to be inverse for the backend's own `Kernel.args` and return `None` for other backend variants (`spoc/framework/Framework_sig.ml:340-349`).
- `SCALAR_TYPE.of_primitive` is expected to accept only the matching primitive representation; built-ins fail on mismatches (`spoc/framework/Typed_value.ml:191-257`).
- `EXEC_VECTOR.get` and `set` traffic values through `typed_value`, while `device_ptr` and `elem_size` expose binding data (`spoc/framework/Typed_value.ml:108-130`).
- `Backend_error.Make` should stamp every constructor with the backend name captured in the functor argument (`spoc/framework/Backend_error.ml:291-347`).

## Potential Invariant Violations or Bugs

- `dims_*` constructors do not reject zero or negative values (`spoc/framework/Framework_sig.ml:17-23`). Invalid dimensions could flow to backend `Kernel.launch` (`spoc/framework/Framework_sig.ml:270-277`).
- `Device_not_found` renders available range as `0-(max_devices - 1)` (`spoc/framework/Backend_error.ml:203-208`). With `max_devices = 0`, the message becomes `0--1`; there is no edge-case test.
- `Typed_value.Registry.register_scalar` and `register_composite` silently overwrite existing names (`spoc/framework/Typed_value.ml:157-161`). That can hide duplicate generated type modules.
- `primitive_type_name` maps any `PFloat` to `"float"` (`spoc/framework/Typed_value.ml:289-294`), losing float32/float64 distinction when only the primitive remains. This is probably intentional for primitive storage, but it is a footgun if used as a type name.
- The module claims no `Obj.t` in normal typed value transport, but `EXEC_VECTOR.internal_get_vector_obj` explicitly exposes `Obj.t` for interpreter internals (`spoc/framework/Typed_value.ml:127-130`). Marked uncertain because the comment documents it as an escape hatch.
- `typed_value_of_exec_arg` raises on `EA_Vec` (`spoc/framework/Typed_value.ml:275-283`). That is explicit behavior but currently untested.

## Performance and Maintainability Risks

- The `BACKEND` signature is broad; any change is a cross-backend migration.
- Global mutable registries are not synchronized and have no reset API, which complicates parallel tests and long-lived processes.
- Backend identity uses free-form strings (`device.framework`, `cuda_or_opencl`-style code elsewhere), so typos are not caught by the type system.
- Error rendering embeds source previews in `Compilation_failed` (`spoc/framework/Backend_error.ml:209-218`); useful for debugging but potentially noisy or sensitive.

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
