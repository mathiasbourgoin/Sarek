# Native Vector Execution Boundary

## Original Risk

Kernel execution moved vectors through native arguments as raw runtime values.
Backends recovered the expected vector type with representation casts, so a
wrong descriptor or stale generated code could reinterpret a vector as another
element type.

## Implemented Alternative

> **Verified in the tree 2026-07-27 at `c92a9a28`**, not merely proposed:
> `Sarek_ir_types.ml:195` (`NA_Vec of native_vec`), `:197` (the existential
> `NV : ('elt, 'underlying) native_vec_ops -> native_vec`), `:199`
> (`native_vec_ops`), `:226` `vec_get_custom`, `:236` `vec_set_custom`; eight
> call sites in `sarek/ppx/Sarek_native_gen_kernel.ml`. `internal_get_vector_obj`
> has zero hits tree-wide. This note exists because the section's own wording
> ("now carries") cannot distinguish a landed change from a proposed one, and a
> reviewer reading it flagged it as possibly aspirational — reasonably, since the
> parent README's global claim above it *was*.

`Sarek_ir_types.native_arg` now carries an existential `native_vec` record with
typed element accessors, typed scalar accessors, a vector length, an underlying
value, and `Type_id` witnesses for both element and underlying vector type.

Generated native code calls `vec_get_custom`, `vec_set_custom`, and
`vec_as_vector` with an expected witness. The operation succeeds only when the
witness matches.

## Why This Improves The Code

The dynamic boundary remains flexible, but type recovery is now guarded by a
GADT equality proof instead of unchecked representation conversion. Scalar
vectors use direct typed closures, which avoids extra boxing through generic
values in the common native path.
