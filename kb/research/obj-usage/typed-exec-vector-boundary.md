# Typed EXEC_VECTOR Boundary

Priority: P0.

## Obj Sites

- `spoc/framework/Typed_value.ml:130`: `val internal_get_vector_obj : unit -> Obj.t`.
- `sarek/sarek/Execute.ml:53`: wraps a typed `('a, 'b) Vector.t` with `Obj.repr`.
- `sarek/plugins/native/Native_plugin_base.ml:577`: wraps a backend `Memory.buffer` with `Obj.repr`.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml:545`: same for interpreter buffers.
- `sarek/plugins/interpreter/Interpreter_plugin.ml:93`: casts that erased value back to `Spoc_core.Kernel_arg.Vec`.
- Downstream consumers in `sarek/plugins/native/Native_plugin.ml:264-269`, `sarek/sarek/Kirc_kernel.ml:251-256`, `sarek/plugins/interpreter/Interpreter_plugin_base.ml:588`, and tests depend on this escape hatch.

## Assumed Invariant

Every `EA_Vec (module V)` must secretly wrap the concrete value expected by its eventual consumer: either a `('a, 'b) Spoc_core.Vector.t`, a native/interpreter backend `Memory.buffer`, or a Bigarray in tests. The type name and element size are trusted but not checked against the recovered value.

This is the weakest boundary in the current design because the same method is used for incompatible payload families.

## Existing Project Pattern

The project already has better options:

- `Kernel_arg.Vec : ('a, 'b) Vector.t -> Kernel_arg.t` keeps vectors existentially typed without `Obj` (`sarek/core/Kernel_arg.ml:13-24`).
- `EXEC_VECTOR.get` and `set` expose element access through `typed_value` (`spoc/framework/Typed_value.ml:115-119`).
- Backend buffers are already first-class modules in `sarek/core/Vector_types.ml:205-216`; they should not be disguised as vectors.

## Viable Replacement

Split the erased vector escape hatch into two typed paths.

1. Remove `internal_get_vector_obj` from `Typed_value.EXEC_VECTOR`.
2. Keep `EA_Vec` as the framework-level argument for element and device-pointer access only.
3. Add a Sarek-local direct execution argument for paths that genuinely need `Vector.t`:

```ocaml
type direct_arg =
  | DVec : ('a, 'b) Spoc_core.Vector.t -> direct_arg
  | DExec of Spoc_framework.Framework_sig.exec_arg
```

`Execute.vector_args_to_exec_array` can produce `direct_arg array` for native/interpreter direct backends, while JIT backends continue to receive `exec_arg array`.

4. Add a separate buffer argument for plugin low-level `Kernel.set_arg_buffer` by extending the closed `exec_arg` definition or, preferably, by adding a backend-local kernel-argument type that is not confused with a vector:

```ocaml
module type EXEC_BUFFER = sig
  type elt
  val length : int
  val elem_size : int
  val device_ptr : unit -> nativeint
  val get : int -> Spoc_framework.Typed_value.typed_value
  val set : int -> Spoc_framework.Typed_value.typed_value -> unit
end

type backend_arg =
  | BA_Exec of Spoc_framework.Framework_sig.exec_arg
  | BA_Buffer of (module EXEC_BUFFER)
```

Keeping `BA_Buffer` backend-local avoids framework-wide API churn. If the public `exec_arg` type is changed instead, update the closed variant in both `Typed_value.ml` and the alias in `Framework_sig.ml`.

5. Change `Interpreter_plugin.exec_args_to_kernel_args` to operate on `direct_arg array` or take a `Kernel_arg.t list` directly. It should never recover a `Vector.t` from `exec_arg`.

## Expected Impact

Code quality improves substantially: "vector" and "backend buffer" become separate concepts again. This also makes type errors local to argument construction instead of deferred until a backend casts an `Obj.t`.

Performance improves for native/interpreter paths that currently call `V.internal_get_vector_obj` inside per-element accessors. The replacement can close over the typed vector once and call `Vector.get`/`kernel_set` directly.

## Tests

- Add a negative direct-backend test that passes a backend buffer argument where a `DVec`/`Kernel_arg.Vec` is expected and checks for a structured type error.
- Add native and interpreter E2E tests for custom vector kernels to ensure direct vector access does not use `Obj`.
- Update `sarek/tests/new_runtime/test_native_runtime.ml` to use `V.get`/`V.set` or the new `DVec` wrapper instead of `Obj.obj`.
