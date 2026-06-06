# Typed Helper Lookup

Priority: P2.

## Obj Sites

- `sarek/sarek/Sarek_type_helpers.ml:65`: `H.to_value (Obj.obj (Obj.repr record))`.
- `sarek/sarek/Sarek_type_helpers.ml:66`: `Obj.obj (Obj.repr (H.from_value v))`.

## Assumed Invariant

The caller of `helpers.to_value` passes the native record type hidden inside `AnyHelpers`, and the caller of `helpers.from_value` expects that same type. The current API exposes these as polymorphic functions:

```ocaml
to_value : 'a. 'a -> value
from_value : 'a. value -> 'a
```

That signature is not actually type-safe; it can only be implemented with a cast.

## Existing Project Pattern

The registry stores helpers as `AnyHelpers : (module HELPERS with type t = 'a) -> any_helpers` (`sarek/sarek/Sarek_type_helpers.ml:32-35`). That is a good existential representation. The unsafe part is only the untyped `lookup` result.

## Viable Replacement

Expose two lookup APIs instead of one polymorphic helper record.

```ocaml
type dynamic_helpers = {
  from_values : value array -> value;
  to_values : value -> value array;
  get_field : value -> string -> value;
}

val lookup_dynamic : string -> dynamic_helpers option
val lookup_typed :
  'a Type_id.t -> (module HELPERS with type t = 'a) option
```

Interpreter code that only manipulates `VRecord` uses `lookup_dynamic` and never sees native OCaml records. Generated code and custom vector bridges use `lookup_typed` with the PPX-generated type ID and get a first-class module whose `to_value : t -> value` and `from_value : value -> t` are genuinely typed.

During migration, keep `lookup` as deprecated and implement new callers first. Once custom vector witnesses are in place, delete the polymorphic `to_value`/`from_value` fields.

## Expected Impact

This removes a hidden universal cast from custom record conversion and makes helper misuse fail at lookup. Performance should improve slightly in typed generated paths because the helper module can be resolved once and closed over, instead of bouncing through polymorphic record fields.

## Tests

- Add unit tests for `lookup_typed` success and type-ID mismatch.
- Add interpreter custom record tests that use only `lookup_dynamic`.
- Add generated `[@@sarek.type]` tests proving helper modules expose a type ID and round-trip a native record without `Obj`.

