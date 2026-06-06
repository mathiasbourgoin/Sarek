# Interpreter Launch Typed Values

Priority: P1.

## Obj Sites

- `sarek/plugins/interpreter/Interpreter_plugin_base.ml:588`: `Obj.obj (V.internal_get_vector_obj ())`.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml:601`: casts storage to `float32` Bigarray when `elem_size = 4`.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml:621`: casts storage to `float64` Bigarray otherwise.

## Assumed Invariant

The argument is a backend buffer with Bigarray storage, and element size uniquely identifies the scalar type. That is false: 4 bytes can be `float32` or `int32`, and 8 bytes can be `float64`, `int64`, or `complex32`.

## Existing Project Pattern

`EXEC_VECTOR.get` already returns a `Typed_value.typed_value` (`spoc/framework/Typed_value.ml:115-119`). `Kirc_kernel.exec_arg_to_native_arg` and `Native_plugin.exec_arg_to_native` already convert `typed_value` primitives into typed runtime values without looking at raw storage.

## Viable Replacement

Use `V.get` instead of recovering the raw buffer.

```ocaml
let interp_value_of_typed_value = function
  | TV_Scalar (SV ((module S), x)) -> (
      match S.to_primitive x with
      | PInt32 n -> VInt32 n
      | PInt64 n -> VInt64 n
      | PFloat f ->
          if S.name = "float64" then VFloat64 f else VFloat32 f
      | PBool b -> VBool b
      | PBytes _ -> unsupported)
  | TV_Composite (CV ((module C), x)) ->
      (* Use typed helper lookup for C.name, see typed-helper-lookup.md. *)
```

Then build `ArgArray (Array.init V.length (fun i -> interp_value_of_typed_value (V.get i)))`.

If this launch path is intended to receive raw backend buffers rather than vectors, introduce the backend-local `BA_Buffer` shape described in `typed-exec-vector-boundary.md` and give the buffer wrapper typed `get`/`set` methods. Do not cast through `internal_get_vector_obj`.

## Expected Impact

This is both a safety and correctness improvement. It removes three Obj calls and fixes the current integer/float ambiguity in interpreter launch conversion. It may be slower than raw Bigarray blit for primitive arrays, but this path is the interpreter backend and correctness/debuggability matter more. A later optimization can add typed fast paths by matching `V.type_name`.

## Tests

- Add interpreter launch tests for `int32`, `int64`, `float32`, and `float64` vectors with the same byte widths.
- Add a regression test proving an `int32` vector is interpreted as `VInt32`, not `VFloat32`.
- Add custom record vector interpreter launch coverage once typed helper lookup is fixed.
