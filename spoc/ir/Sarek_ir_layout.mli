(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_ir_layout - Packed aggregate byte layout for GPU codegen.

    Pure layout computation for record/variant element types, mirroring the host
    PPX layout exactly (packed cumulative offsets, no padding — see
    [calc_offsets] and the variant [[tag:int32@0][payload@4]] encoding in
    sarek/ppx/Sarek_ppx.ml). Every backend that stores aggregates in vector
    elements must take its offsets from this module so host and device agree
    byte-for-byte.

    Layouts whose packed placement would put a scalar leaf at a
    non-naturally-aligned offset are rejected with a typed error, as are
    variants nested below top level and array/vector fields. *)

open Sarek_ir_types

(** {1 Errors} *)

(** Typed layout rejection. [type_name] is the aggregate being laid out; [field]
    is the offending field path (dotted for nested records, [_N] for variant
    payload slots). *)
type layout_error =
  | Misaligned_field of {
      type_name : string;
      field : string;
      offset : int;  (** packed byte offset of the leaf *)
      required_align : int;  (** natural alignment of the leaf's scalar type *)
    }
      (** The packed layout places a scalar leaf at an offset that is not a
          multiple of its natural alignment. *)
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

(** Packed record layout. *)
type record_layout = {
  rl_fields : (string * int) list;
      (** Byte offset of each immediate field (declaration order), including
          nested-record fields as a whole. *)
  rl_leaves : leaf list;
      (** All scalar leaves, flattened recursively, declaration order. *)
  rl_size : int;  (** Total packed byte size. *)
}

(** Layout of one variant constructor's payload. *)
type ctor_layout = {
  ctor_name : string;
  ctor_tag : int;  (** Constructor declaration index (host tag value). *)
  ctor_leaves : leaf list;
      (** Payload scalar leaves; offsets are absolute from the element start
          (i.e. [>= 4]), paths are constructor-qualified positional slots
          ([Value._0], [Pair._1], ...). *)
  ctor_payload_size : int;  (** Packed byte size of this payload. *)
}

(** Variant layout: [[tag:int32@0][payload@4]]. *)
type variant_layout = {
  vl_tag_offset : int;  (** Always 0. *)
  vl_payload_offset : int;  (** Always 4. *)
  vl_ctors : ctor_layout list;  (** Declaration order. *)
  vl_size : int;  (** [4 + max payload size] over all constructors. *)
}

(** Layout of any element type, as dispatched by {!elttype_layout}. *)
type layout =
  | LScalar of {size : int; align : int}
  | LRecord of record_layout
  | LVariant of variant_layout

(** {1 Layout computation} *)

(** [record_layout ~type_name fields] computes the packed layout of a record:
    offsets are cumulative field sizes with no padding, nested records are
    flattened recursively with dotted leaf paths. Rejects misaligned leaves,
    variant fields, and array/vector fields. *)
val record_layout :
  type_name:string ->
  (string * elttype) list ->
  (record_layout, layout_error) result

(** [variant_layout ~type_name ctors] computes the packed layout of a variant:
    int32 tag at offset 0, payload region at offset 4, per-constructor arg
    offsets = 4 + packed cumulative sizes, total size = 4 + max payload. Rejects
    misaligned payload leaves (hence every 8-byte scalar payload, which would
    sit at a non-8-aligned offset), nested variants, and array/vector payloads.
*)
val variant_layout :
  type_name:string ->
  (string * elttype list) list ->
  (variant_layout, layout_error) result

(** [elttype_layout t] dispatches on [t]: scalars yield [LScalar], [TRecord] and
    [TVariant] yield their validated aggregate layouts. [TArray]/[TVec] are
    rejected with [Unsupported_field]. *)
val elttype_layout : elttype -> (layout, layout_error) result
