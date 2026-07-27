# Legacy And Test Direct APIs

Priority: P3.

## Obj Sites

- `sarek/plugins/native/Native_plugin.ml:384-385`: `run_kernel_direct` accepts `Obj.t array`.
- `sarek/plugins/native/Native_plugin_base.ml:646`: stale documentation references `Obj.t array`; the actual registry type at `sarek/plugins/native/Native_plugin_base.ml:30-35` is already `Framework_sig.exec_arg array`.
- `sarek/tests/new_runtime/test_native_runtime.ml:63`, `71`, `79`: test-only `Obj.obj (V.internal_get_vector_obj ())`.

## Assumed Invariant

Legacy callers know the exact runtime representation of each argument. In tests, each `EA_Vec` wraps a Bigarray of `(float, Bigarray.float32_elt, Bigarray.c_layout)`.

## Existing Project Pattern

The native kernel registry already uses typed `Framework_sig.exec_arg array` (`sarek/plugins/native/Native_plugin_base.ml:30-35`). The test can use `EXEC_VECTOR.get`/`set` with `Typed_value.Float32_type` instead of raw Bigarray extraction.

## Viable Replacement

Deprecate or delete `run_kernel_direct`. If compatibility is required, move it to a clearly named `Unsafe` submodule and keep it out of normal backend APIs:

```ocaml
module Unsafe : sig
  val run_kernel_direct :
    name:string ->
    native_fn:(Obj.t array -> dims -> dims -> unit) ->
    args:Obj.t array -> grid:dims -> block:dims -> unit
end
```

The better default API should be:

```ocaml
val run_kernel_direct :
  name:string ->
  native_fn:(Framework_sig.exec_arg array -> dims -> dims -> unit) ->
  args:Framework_sig.exec_arg array -> grid:dims -> block:dims -> unit
```

Update `test_native_runtime.ml` to unpack vectors through `V.get` and `V.set`, or through a local test helper:

```ocaml
let get_float32 (module V : EXEC_VECTOR) i =
  match V.get i with
  | TV_Scalar (SV ((module S), x)) -> (
      match S.to_primitive x with PFloat f -> f | _ -> failwith "float32")
  | _ -> failwith "scalar"
```

## Expected Impact

This is low risk and mostly cleans up public surface area and tests. It prevents new code from copying the `Obj.t array` pattern and makes tests exercise the typed API they are meant to validate.

## Tests

- Update `test_native_runtime.ml` to avoid `Obj`.
- Add a compile-time or API test that the normal native plugin direct API accepts `exec_arg array`.
- If `Unsafe` remains, document that it is outside the type-safe execution path.

