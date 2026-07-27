# Shared Memory Type IDs

Priority: P1. **Status (2026-06-02): RESOLVED.** The proposed `Type_id` replacement
below was implemented — `alloc_shared_with_key` now keys custom arrays by
`Sarek_ir_types.Type_id.t` and matches with `Type_id.Refl` (no `Obj`). The CPU runtime
was also split (see [cpu-runtime](../../sarek/runtime/cpu-runtime.md)); the
shared-memory allocators now live in `Sarek_cpu_runtime_types.ml`. References below are
repointed to current locations and retained for history.

## Obj Sites (former)

- `sarek/sarek/Sarek_cpu_runtime_types.ml:126` (`alloc_shared_with_key`): the former
  `Obj.obj (Obj.repr arr)` cast is gone — now a typed `Type_id.equal`/`Refl` match.
- `sarek/sarek/Sarek_cpu_runtime.mli:84`: now documents typed custom allocation (no
  `Obj.t`).

## Assumed Invariant

Each shared-memory name is allocated with exactly one OCaml element type for the lifetime of the block. The caller must also use the same size and default representation each time.

The code has typed primitive tables for `int`, `float`, `int32`, and `int64`; only custom arrays fall back to untyped erasure.

## Existing Project Pattern

The typed primitive shared-memory allocators avoid casts by splitting storage by type (`sarek/sarek/Sarek_cpu_runtime_types.ml:88-124`). `Sarek_type_helpers.AnyHelpers` already packs a type-specific module with an existential.

## Viable Replacement

Give custom shared arrays a generated type identity and check it on lookup.

```ocaml
module Type_id : sig
  type 'a t
  type (_, _) eq = Refl : ('a, 'a) eq
  val create : string -> 'a t
  val equal : 'a t -> 'b t -> ('a, 'b) eq option
end

type custom_array =
  | CustomArray : {
      type_id : 'a Type_id.t;
      type_name : string;
      data : 'a array;
      length : int;
    } -> custom_array

val alloc_shared_custom :
  shared_mem -> 'a Type_id.t -> string -> int -> 'a -> 'a array
```

The PPX that generates custom type descriptors can also generate `let point_type_id : point Type_id.t`. `alloc_shared_custom` checks `Type_id.equal`; after `Some Refl`, returning `data` is type-safe.

As a smaller first step, key `custom_arrays` by `(name, type_name)` and fail if the same name appears with a different type. The full type-ID version is stronger and still implementable.

## Expected Impact

This removes a latent type confusion bug in block shared memory. Runtime overhead is one hash lookup and one integer/name comparison at allocation time, not per element, so performance should be unchanged in kernels.

## Tests

- Add a custom shared-memory test that allocates the same name twice with the same generated type ID and confirms the same array is reused.
- Add a negative test that allocates the same name with two different custom type IDs and expects a structured runtime error.
- Add a size mismatch test for the same name and type.
- Preserve or update existing custom shared-memory coverage if present, especially tests named around `alloc_shared_custom`.
