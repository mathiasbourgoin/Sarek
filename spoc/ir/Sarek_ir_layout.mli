(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_ir_layout - Aligned (C-ABI-compatible) aggregate byte layout for GPU
    codegen.

    Pure layout computation for record/variant element types, mirroring the host
    PPX layout exactly (aligned C struct rules: each field rounded up to its
    natural alignment with padding, total size rounded up to the struct's max
    member alignment; the variant
    [[tag:int32@0][payload@max(4, max payload align)]] encoding — see
    [calc_offsets] in sarek/ppx/Sarek_ppx.ml). This is byte-for-byte identical
    to the [typedef struct {...}] the C-family backends emit, so host and every
    device agree. Every backend that stores aggregates in vector elements must
    take its offsets from this module.

    Because placement is alignment-derived, mixed-alignment aggregates
    ([{i32;f64}] records, f64/i64-payload variants) are now laid out correctly
    on all backends rather than rejected. The [Misaligned_field] error is
    retained only as a defensive internal invariant (it can no longer fire for
    well-formed input); variants nested below top level and array/vector fields
    are still rejected. *)

open Sarek_ir_types

(** {1 Errors} *)

(** Typed layout rejection. [type_name] is the aggregate being laid out; [field]
    is the offending field path (dotted for nested records, [_N] for variant
    payload slots). *)
type layout_error =
  | Misaligned_field of {
      type_name : string;
      field : string;
      offset : int;  (** byte offset of the leaf *)
      required_align : int;  (** natural alignment of the leaf's scalar type *)
    }
      (** Defensive internal invariant: a scalar leaf landed at an offset that
          is not a multiple of its natural alignment. With the aligned layout
          every field is padded to its boundary, so this can no longer be
          produced by well-formed input; kept as an assertion guard. *)
  | Nested_variant of {type_name : string; field : string}
      (** A variant occurs below top level (record field or variant payload). *)
  | Unsupported_field of {type_name : string; field : string; what : string}
      (** A field type has no byte layout (arrays, vectors, ...). *)

exception Layout_error of layout_error

(** [layout_error_message e] renders [e] as a human-readable message naming the
    type, field, offset and required alignment. *)
val layout_error_message : layout_error -> string

(** {1 Scalar size and alignment} *)

(** [scalar_size t] is the byte size of scalar type [t], identical to the host
    [field_byte_size] mapping (sarek/ppx/Sarek_ppx.ml,
    [get_type_size_from_core_type]): 4 for int32/float32/bool/unit, 8 for
    int64/float64. Raises [Invalid_argument] on aggregate/array types. *)
val scalar_size : elttype -> int

(** [scalar_align t] is the natural alignment of scalar type [t]: 4 for 32-bit
    scalars, 8 for int64/float64. Raises [Invalid_argument] on aggregate/array
    types. *)
val scalar_align : elttype -> int

(** {1 Layout results} *)

(** One scalar leaf of a flattened aggregate. *)
type leaf = {
  leaf_path : string;
      (** Field path from the aggregate root: ["x"], ["inner.b"], or ["_0"] for
          positional variant payload slots. *)
  leaf_type : elttype;  (** Scalar type of the leaf. *)
  leaf_offset : int;  (** Byte offset from the start of the element. *)
  leaf_size : int;  (** Byte size ([scalar_size leaf_type]). *)
  leaf_align : int;  (** Natural alignment ([scalar_align leaf_type]). *)
}

(** Aligned record layout. *)
type record_layout = {
  rl_fields : (string * int) list;
      (** Aligned byte offset of each immediate field (declaration order),
          including nested-record fields as a whole. *)
  rl_leaves : leaf list;
      (** All scalar leaves, flattened recursively, declaration order. *)
  rl_size : int;
      (** Total byte size, padded to the struct's maximum member alignment. *)
}

(** Layout of one variant constructor's payload. *)
type ctor_layout = {
  ctor_name : string;
  ctor_tag : int;  (** Constructor declaration index (host tag value). *)
  ctor_leaves : leaf list;
      (** Payload scalar leaves; offsets are absolute from the element start
          (i.e. [>= vl_payload_offset]), paths are constructor-qualified
          positional slots ([Value._0], [Pair._1], ...). *)
  ctor_payload_size : int;
      (** Aligned (padded) byte size of this payload — its C union member size.
      *)
}

(** Variant layout: [[tag:int32@0][payload@P]] with
    [P = max(4, max payload-member alignment)]. *)
type variant_layout = {
  vl_tag_offset : int;  (** Always 0. *)
  vl_payload_offset : int;
      (** [max(4, max payload-member alignment)] — 4 when every payload is
          4-byte-aligned, 8 when any payload is 8-byte-aligned. *)
  vl_ctors : ctor_layout list;  (** Declaration order. *)
  vl_size : int;
      (** [round_up(vl_payload_offset + max payload size, max_align)]. *)
}

(** Layout of any element type, as dispatched by {!elttype_layout}. *)
type layout =
  | LScalar of {size : int; align : int}
  | LRecord of record_layout
  | LVariant of variant_layout

(** {1 Layout computation} *)

(** [record_layout ~type_name fields] computes the aligned layout of a record:
    each field is placed at the next offset satisfying its natural alignment
    (padding inserted), the total size is rounded up to the struct's max member
    alignment, nested records are flattened recursively with dotted leaf paths.
    Rejects variant fields and array/vector fields. *)
val record_layout :
  type_name:string ->
  (string * elttype) list ->
  (record_layout, layout_error) result

(** [variant_layout ~type_name ctors] computes the aligned layout of a variant:
    int32 tag at offset 0, payload region at offset
    [max(4, max payload-member alignment)], per-constructor arg offsets aligned
    within the payload, total size =
    [round_up(payload_offset + max payload, max_align)]. Mixed-alignment and
    8-byte-scalar payloads are now supported (aligned naturally). Rejects nested
    variants and array/vector payloads. *)
val variant_layout :
  type_name:string ->
  (string * elttype list) list ->
  (variant_layout, layout_error) result

(** [elttype_layout t] dispatches on [t]: scalars yield [LScalar], [TRecord] and
    [TVariant] yield their validated aggregate layouts. [TArray]/[TVec] are
    rejected with [Unsupported_field]. *)
val elttype_layout : elttype -> (layout, layout_error) result
