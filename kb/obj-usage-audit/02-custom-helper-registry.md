# Custom Helper Registry

## Original Risk

`Sarek_type_helpers` stored custom conversion helpers behind untyped conversion
closures. Lookup recovered the requested helper type through raw runtime
representation casts.

## Implemented Alternative

Each helper module now exposes a `type_id` witness. Dynamic lookup uses
`Sarek_ir_types.Type_id.equal` and returns a typed first-class module only when
the witness proves equality.

## Why This Improves The Code

Custom record and variant conversion now fails deterministically on mismatched
types. The registry still supports dynamic lookup, but successful lookups carry
real type evidence.
