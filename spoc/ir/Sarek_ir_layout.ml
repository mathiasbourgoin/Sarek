(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_ir_layout - Packed aggregate byte layout for GPU codegen.

    Mirrors the host PPX layout exactly:

    - Records: packed cumulative offsets, NO padding — sarek/ppx/Sarek_ppx.ml
      [calc_offsets] (lines 616-623), field sizes from [field_byte_size] /
      [get_type_size_from_core_type].
    - Variants: [[tag:int32@0][payload@4]], element size
      [4 + max_payload_bytes], tag = constructor declaration index —
      sarek/ppx/Sarek_ppx.ml lines 750-755.

    See the .mli for the API contract. *)

open Sarek_ir_types

(** {1 Errors} *)

type layout_error =
  | Misaligned_field of {
      type_name : string;
      field : string;
      offset : int;
      required_align : int;
    }
  | Nested_variant of {type_name : string; field : string}
  | Unsupported_field of {type_name : string; field : string; what : string}

exception Layout_error of layout_error

let layout_error_message = function
  | Misaligned_field {type_name; field; offset; required_align} ->
      Printf.sprintf
        "layout of '%s': field '%s' at packed byte offset %d is misaligned \
         (its type requires %d-byte alignment). The host layout is packed with \
         no padding, so this aggregate cannot be represented; reorder or split \
         the fields so every %d-byte scalar lands on a %d-byte boundary."
        type_name
        field
        offset
        required_align
        required_align
        required_align
  | Nested_variant {type_name; field} ->
      Printf.sprintf
        "layout of '%s': field '%s' is a variant nested below top level; \
         variants are only supported as the element type itself. Hoist the \
         variant to its own vector or flatten its payload into the enclosing \
         record."
        type_name
        field
  | Unsupported_field {type_name; field; what} ->
      Printf.sprintf
        "layout of '%s': field '%s' has type %s, which has no byte layout in a \
         vector element; pass it as a separate kernel parameter instead."
        type_name
        field
        what

(** {1 Scalar size and alignment} *)

(* Byte sizes MUST equal the host [field_byte_size] mapping
   (sarek/ppx/Sarek_ppx.ml:472 -> get_type_size_from_core_type, lines
   116-126): int32 -> 4 (:118), int64 -> 8 (:119), float32 -> 4 (:120),
   float -> 4 "GPU float32" (:121), int -> 4 (:122), and everything else --
   including [bool], which has no explicit case -- falls to the catch-all
   [4] (:125-126). Hence TBool = 4 here. TFloat64 has no host case (the host
   PPX cannot marshal float64 record fields today); its natural size 8 is
   used, and the alignment rule rejects any packed placement that would
   misalign it. *)
let scalar_size = function
  | TInt32 | TFloat32 | TBool | TUnit -> 4
  | TInt64 | TFloat64 -> 8
  | TRecord (n, _) ->
      invalid_arg
        ("Sarek_ir_layout.scalar_size: not a scalar type: TRecord " ^ n)
  | TVariant (n, _) ->
      invalid_arg
        ("Sarek_ir_layout.scalar_size: not a scalar type: TVariant " ^ n)
  | TArray _ ->
      invalid_arg "Sarek_ir_layout.scalar_size: not a scalar type: TArray"
  | TVec _ -> invalid_arg "Sarek_ir_layout.scalar_size: not a scalar type: TVec"

let scalar_align = function
  | TInt32 | TFloat32 | TBool | TUnit -> 4
  | TInt64 | TFloat64 -> 8
  | TRecord _ | TVariant _ | TArray _ | TVec _ ->
      invalid_arg "Sarek_ir_layout.scalar_align: not a scalar type"

(** {1 Layout results} *)

type leaf = {
  leaf_path : string;
  leaf_type : elttype;
  leaf_offset : int;
  leaf_size : int;
  leaf_align : int;
}

type record_layout = {
  rl_fields : (string * int) list;
  rl_leaves : leaf list;
  rl_size : int;
}

type ctor_layout = {
  ctor_name : string;
  ctor_tag : int;
  ctor_leaves : leaf list;
  ctor_payload_size : int;
}

type variant_layout = {
  vl_tag_offset : int;
  vl_payload_offset : int;
  vl_ctors : ctor_layout list;
  vl_size : int;
}

type layout =
  | LScalar of {size : int; align : int}
  | LRecord of record_layout
  | LVariant of variant_layout

(** {1 Internal flattening} *)

let ( let* ) = Result.bind

(** Flatten one field into scalar leaves at absolute byte [offset], validating
    natural alignment of every leaf. [path] is the field's dotted path from the
    aggregate root; [type_name] names the root aggregate for error reporting.
    Returns the leaves (declaration order) and the field's packed byte size. *)
let rec flatten_field ~type_name ~path ~offset (t : elttype) :
    (leaf list * int, layout_error) result =
  match t with
  | TInt32 | TInt64 | TFloat32 | TFloat64 | TBool | TUnit ->
      let size = scalar_size t in
      let align = scalar_align t in
      if offset mod align <> 0 then
        Error
          (Misaligned_field
             {type_name; field = path; offset; required_align = align})
      else
        Ok
          ( [
              {
                leaf_path = path;
                leaf_type = t;
                leaf_offset = offset;
                leaf_size = size;
                leaf_align = align;
              };
            ],
            size )
  | TRecord (_, fields) ->
      (* Nested record: recurse with a dotted path prefix (FR-005). *)
      flatten_fields ~type_name ~prefix:(path ^ ".") ~offset fields
  | TVariant _ -> Error (Nested_variant {type_name; field = path})
  | TArray _ ->
      Error (Unsupported_field {type_name; field = path; what = "TArray"})
  | TVec _ -> Error (Unsupported_field {type_name; field = path; what = "TVec"})

(** Flatten a packed sequence of named fields starting at absolute byte [offset]
    (host [calc_offsets] rule: cumulative sums, no padding). Returns all leaves
    and the total packed size of the sequence. *)
and flatten_fields ~type_name ~prefix ~offset fields :
    (leaf list * int, layout_error) result =
  List.fold_left
    (fun acc (name, ftype) ->
      let* leaves_acc, size_acc = acc in
      let* leaves, size =
        flatten_field
          ~type_name
          ~path:(prefix ^ name)
          ~offset:(offset + size_acc)
          ftype
      in
      Ok (leaves_acc @ leaves, size_acc + size))
    (Ok ([], 0))
    fields

(** Packed byte size of a field type. Only called on types already validated by
    [flatten_field], so aggregates can only be records. *)
let rec packed_size = function
  | TRecord (_, fields) ->
      List.fold_left (fun acc (_, t) -> acc + packed_size t) 0 fields
  | t -> scalar_size t

(** {1 Layout computation} *)

let record_layout ~type_name (fields : (string * elttype) list) :
    (record_layout, layout_error) result =
  let* leaves, size = flatten_fields ~type_name ~prefix:"" ~offset:0 fields in
  let field_offsets =
    List.rev
      (fst
         (List.fold_left
            (fun (acc, off) (name, ftype) ->
              ((name, off) :: acc, off + packed_size ftype))
            ([], 0)
            fields))
  in
  Ok {rl_fields = field_offsets; rl_leaves = leaves; rl_size = size}

let variant_layout ~type_name (ctors : (string * elttype list) list) :
    (variant_layout, layout_error) result =
  let* rev_ctors, max_payload =
    List.fold_left
      (fun acc (tag, (name, args)) ->
        let* ctors_acc, max_acc = acc in
        let named_args =
          List.mapi (fun j t -> (Printf.sprintf "_%d" j, t)) args
        in
        (* Payload region starts at fixed offset 4, after the int32 tag. *)
        let* leaves, payload_size =
          flatten_fields ~type_name ~prefix:(name ^ ".") ~offset:4 named_args
        in
        let ctor =
          {
            ctor_name = name;
            ctor_tag = tag;
            ctor_leaves = leaves;
            ctor_payload_size = payload_size;
          }
        in
        Ok (ctor :: ctors_acc, max max_acc payload_size))
      (Ok ([], 0))
      (List.mapi (fun i c -> (i, c)) ctors)
  in
  Ok
    {
      vl_tag_offset = 0;
      vl_payload_offset = 4;
      vl_ctors = List.rev rev_ctors;
      (* Host rule: 4-byte int32 tag + max payload (Sarek_ppx.ml:750-755). *)
      vl_size = 4 + max_payload;
    }

let elttype_layout (t : elttype) : (layout, layout_error) result =
  match t with
  | TInt32 | TInt64 | TFloat32 | TFloat64 | TBool | TUnit ->
      Ok (LScalar {size = scalar_size t; align = scalar_align t})
  | TRecord (name, fields) ->
      let* rl = record_layout ~type_name:name fields in
      Ok (LRecord rl)
  | TVariant (name, ctors) ->
      let* vl = variant_layout ~type_name:name ctors in
      Ok (LVariant vl)
  | TArray _ ->
      Error
        (Unsupported_field
           {type_name = "<element>"; field = "<self>"; what = "TArray"})
  | TVec _ ->
      Error
        (Unsupported_field
           {type_name = "<element>"; field = "<self>"; what = "TVec"})
