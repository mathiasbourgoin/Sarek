# Plugin Bigarray Buffer Storage

Priority: P1.

## Obj Sites

- `sarek/plugins/native/Native_plugin_base.ml:431`, `450`, `495`, `498`.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml:414`, `430`, `468`, `471`.

These casts recover the erased Bigarray kind parameter from `Bigarray_storage`.

## Assumed Invariant

An `'a buffer` with `Bigarray_storage arr` has an array whose hidden Bigarray element kind matches the source or destination Bigarray passed to `host_to_device`, `device_to_host`, or `device_to_device`.

The invariant is usually true because `alloc` and `alloc_zero_copy` construct the buffer from the same kind, but the type representation forgets the phantom Bigarray kind and cannot prove it later.

## Existing Project Pattern

`sarek/core/Vector_types.ml:17-23` keeps scalar value type and Bigarray kind together:

```ocaml
type (_, _) scalar_kind =
  | Float32 : (float, Bigarray.float32_elt) scalar_kind
  | Int32 : (int32, Bigarray.int32_elt) scalar_kind
```

`Vector_types.host_storage` also keeps both parameters for vectors (`sarek/core/Vector_types.ml:184-193`). The plugin memory modules should copy that pattern.

## Viable Replacement

Make plugin buffers existential over the Bigarray kind instead of erasing it completely.

```ocaml
type ('a, 'b) element_kind =
  | Scalar_kind : ('a, 'b) Vector_types.scalar_kind -> ('a, 'b) element_kind
  | Custom_kind : 'a Vector_types.custom_type -> ('a, unit) element_kind

type ('a, 'b) typed_buffer = {
  storage : ('a, 'b) storage;
  kind : ('a, 'b) element_kind;
  size : int;
  device : Device.t;
}

and ('a, 'b) storage =
  | Bigarray_storage : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> ('a, 'b) storage
  | Ctypes_storage : unit Ctypes.ptr -> ('a, unit) storage

type 'a buffer = B : ('a, 'b) typed_buffer -> 'a buffer
```

For `host_to_device`, pattern match `B dst`, then compare the source Bigarray kind with `dst.kind`. Add a small equality helper for scalar kinds:

```ocaml
type (_, _) eq = Refl : ('a, 'a) eq
val scalar_kind_eq :
  ('a, 'b) scalar_kind -> ('a, 'c) scalar_kind -> ('b, 'c) eq option
```

After `Some Refl`, the blit type-checks without `Obj.magic`. For `device_to_device`, either require equal scalar kinds for Bigarray storage or use the existing byte-copy path when kinds differ.

## Expected Impact

Code quality improves because buffer element kind mismatches become explicit errors. Runtime cost is a single kind comparison per transfer, which is negligible compared with a Bigarray blit. It may improve debugging performance by failing before a bad blit corrupts memory.

## Tests

- Unit-test native and interpreter `host_to_device`/`device_to_host` for all supported scalar kinds.
- Add a negative test that attempts to copy an `Int32` Bigarray into a `Float32` buffer and checks for a structured type error.
- Add a `device_to_device` same-kind test and a different-kind rejection or byte-copy test, depending on chosen semantics.

