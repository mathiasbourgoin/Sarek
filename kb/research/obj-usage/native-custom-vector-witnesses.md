# Native Custom Vector Witnesses

Priority: P0.

## Obj Sites

- `spoc/ir/Sarek_ir_types.ml:173-176`: `get_any : int -> Obj.t`, `set_any : int -> Obj.t -> unit`, `get_vec : unit -> Obj.t`.
- `spoc/ir/Sarek_ir_types.ml:194`: `Obj.magic (v.get_any i)`.
- `spoc/ir/Sarek_ir_types.ml:201`: `v.set_any i (Obj.magic x)`.
- `spoc/ir/Sarek_ir_types.ml:217`: `Obj.magic (v.get_vec ())`.
- `sarek/plugins/native/Native_plugin.ml:264-269`: casts `EXEC_VECTOR` to a vector and custom values to/from `Obj.t`.
- `sarek/sarek/Kirc_kernel.ml:251-256`: same bridge in the KIRC execution path.
- `sarek/ppx/Sarek_native_gen_kernel.ml:32` (`gen_arg_cast`): generated custom vector wrappers call `vec_get_custom` (`:143`), `vec_set_custom` (`:155`), and `vec_as_vector` (`:57`). (Relocated from `Sarek_native_gen.ml:1121-1127` when that module was split — see [native-gen](../../sarek/ppx/native-gen.md).)

## Assumed Invariant

The PPX-generated accessor object has a statically known element type, and the runtime `NA_Vec` actually carries a vector with the same element type. The invariant is real in normal generated code, but it is represented only by naming discipline and `type_name`.

## Existing Project Pattern

`Kernel_arg.t` and `Vector.kind` already use GADTs to keep existential values typed. `Sarek_type_helpers.AnyHelpers` also shows the intended first-class module pattern, but its final lookup API erases too much.

## Viable Replacement

Replace `get_any`/`set_any`/`get_vec` with typed native vector witnesses.

```ocaml
module type NATIVE_VECTOR = sig
  type elt
  type storage

  val type_name : string
  val length : int
  val elem_size : int
  val type_id : elt Type_id.t
  val get : int -> elt
  val set : int -> elt -> unit
  val underlying : storage option
end

type native_vec = NV : (module NATIVE_VECTOR with type elt = 'a) -> native_vec

type native_arg =
  | NA_Int32 of int32
  | NA_Int64 of int64
  | NA_Float32 of float
  | NA_Float64 of float
  | NA_Vec of native_vec
```

Generated code should match once, check `Type_id.equal expected V.type_id`, then build an accessor object using `V.get` and `V.set`. For scalar vectors, the existing specialized `get_f32`/`set_f32` methods can stay as a fast path during migration. For custom records, the PPX already knows the expected type and can emit the `type_id` check near `gen_arg_cast`.

`underlying` should be removed from ordinary generated element access. When an intrinsic truly needs the full vector, model that intrinsic as accepting a `native_vec` witness or as a dedicated `Vector_arg` API, not as a polymorphic cast.

## Expected Impact

This removes casts from the native custom-vector hot path and makes custom vector type mismatch fail at wrapper construction instead of producing undefined behavior during element access. It should be faster for custom record kernels because `Obj.repr`/`Obj.obj` disappear from every `get` and `set`.

The migration is ambitious but tractable because the surface is concentrated in `Sarek_ir_types`, `Native_plugin.exec_arg_to_native`, `Kirc_kernel.exec_arg_to_native_arg`, and `Sarek_native_gen.gen_arg_cast`. It should also share one conversion helper between `Native_plugin` and `Kirc_kernel`; today they duplicate the same unsafe bridge.

## Tests

- Add generated native tests for `point vector` get/set on both `Native_plugin` and KIRC paths.
- Add a deliberate mismatched custom-vector test using two records with identical layout but different type IDs; it should fail before execution.
- Add a microbenchmark or allocation counter for custom vector kernels to confirm `Obj.repr` allocation pressure disappears.
