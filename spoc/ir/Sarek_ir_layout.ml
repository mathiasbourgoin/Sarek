(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_ir_layout - Aligned (C-ABI-compatible) aggregate byte layout for GPU
    codegen.

    Mirrors the host PPX layout exactly (both migrated from packed to aligned by
    campaign item L8):

    - Records: each field is placed at the lowest offset >= the running size
      that satisfies the field's natural alignment (padding inserted as needed);
      the struct's total size is rounded up to the struct's maximum member
      alignment — sarek/ppx/Sarek_ppx.ml [aligned_record_offsets], field
      sizes/alignments from [get_type_size_from_core_type] /
      [get_type_align_from_core_type].
    - Variants: [[tag:int32@0][payload@P]] where
      [P = max(4, max payload-member alignment)]; element size =
      [round_up(P + max_payload_size, max_align)], tag = constructor declaration
      index.

    This is the standard C struct-layout ABI, so it agrees byte-for-byte with
    the real [typedef struct {...}] the C-family backends (CUDA-C, OpenCL,
    Metal) emit and let the C compiler align — resolving the former
    host-vs-C-compiler divergence for mixed-alignment aggregates.

    Zero-breakage: for a homogeneous 4-byte aggregate (every currently shipped
    [[@@sarek.type]]), every offset is already a multiple of 4 and every
    [round_up(_, 4)] is a no-op, so aligned == packed byte-for-byte.

    Performance: aligned layout costs padding bandwidth for mixed-alignment
    types ([{i32;f64}] = 16B aligned vs 12B packed = +33%). To minimise inserted
    padding, order struct fields largest-alignment-first (put f64/i64 fields
    before i32/f32 fields) — standard C struct-packing guidance.

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
      (* Defensive: with the aligned layout every leaf lands on its natural
         boundary by construction, so this can no longer be produced by
         well-formed input (see [flatten_field]). Kept as an internal invariant
         guard only. *)
      Printf.sprintf
        "layout of '%s': field '%s' at byte offset %d is misaligned (its type \
         requires %d-byte alignment). This is an internal invariant violation: \
         the aligned layout should have padded the field to a %d-byte \
         boundary."
        type_name
        field
        offset
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

(* Byte sizes MUST equal the host [get_type_size_from_core_type] mapping in
   sarek/ppx/Sarek_ppx.ml: int32 -> 4, int64 -> 8, float32 -> 4,
   float -> 4 "GPU float32", float64 -> 8, int -> 4, and everything else --
   including [bool], which has no explicit case -- falls to the catch-all 4.
   Hence TBool = 4 here. Since L8, float64 record fields ARE marshalled by the
   host PPX (via read_float64/write_float64) and placed on their natural 8-byte
   boundary by the aligned layout. *)
let scalar_size = function
  | TFloat16 -> 2
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
  | TFloat16 -> 2
  | TInt32 | TFloat32 | TBool | TUnit -> 4
  | TInt64 | TFloat64 -> 8
  | TRecord _ | TVariant _ | TArray _ | TVec _ ->
      invalid_arg "Sarek_ir_layout.scalar_align: not a scalar type"

(** [align_up off a] rounds [off] up to the next multiple of [a] (the C ABI
    padding rule). [a] is always a positive power of two (4 or 8) here; [a <= 1]
    is the identity. *)
let align_up off a = if a <= 1 then off else (off + a - 1) / a * a

(** Natural alignment of any element type (total; never raises). Scalars use
    {!scalar_align}; a record's alignment is the maximum of its fields'
    alignments (min 1 for the empty record); variant/array/vector types are
    rejected below top level, so their value here is a harmless placeholder used
    only before {!flatten_field} produces the typed rejection. *)
let rec elttype_align = function
  | (TInt32 | TFloat32 | TBool | TUnit | TInt64 | TFloat64 | TFloat16) as t ->
      scalar_align t
  | TRecord (_, fields) -> record_align fields
  | TVariant _ -> 4 (* rejected below top level; placeholder *)
  | TArray _ | TVec _ -> 1 (* rejected; placeholder *)

and record_align fields =
  List.fold_left (fun m (_, t) -> max m (elttype_align t)) 1 fields

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

(** Flatten one field into scalar leaves. The field is placed at absolute byte
    [offset], which the caller has already rounded up to the field's natural
    alignment ([elttype_align]); every scalar leaf therefore lands on its
    natural boundary by construction. [path] is the field's dotted path from the
    aggregate root; [type_name] names the root aggregate for error reporting.
    Returns the leaves (declaration order) and the field's aligned (padded) byte
    size. The [Misaligned_field] guard is a defensive internal invariant — with
    aligned placement it can never fire for well-formed input. *)
let rec flatten_field ~type_name ~path ~offset (t : elttype) :
    (leaf list * int, layout_error) result =
  match t with
  | TInt32 | TInt64 | TFloat32 | TFloat64 | TBool | TUnit ->
      let size = scalar_size t in
      let align = scalar_align t in
      if offset mod align <> 0 then
        (* Unreachable for well-formed input: [flatten_fields] aligns [offset]
           to [align] before dispatching here. Kept as an invariant assertion. *)
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
      (* Nested record: lay out its own fields (aligned) from [offset], which is
         a multiple of the record's alignment, then pad the whole to the record's
         alignment (its C-ABI size). *)
      let* leaves, _foffs, endoff =
        flatten_fields ~type_name ~prefix:(path ^ ".") ~offset fields
      in
      let size = align_up (endoff - offset) (record_align fields) in
      Ok (leaves, size)
  | TFloat16 ->
      (* f16 as a record/variant FIELD is deliberately out of scope: the host
         PPX marshaller has no read_float16/write_float16 (the byte sizes here
         must agree with [get_type_size_from_core_type] in sarek/ppx/Sarek_ppx.ml,
         which knows nothing of f16). f16 is supported as a *vector* element
         type; aggregate fields are a follow-on. Reject rather than lay out a
         field the host cannot marshal. *)
      Error (Unsupported_field {type_name; field = path; what = "TFloat16"})
  | TVariant _ -> Error (Nested_variant {type_name; field = path})
  | TArray _ ->
      Error (Unsupported_field {type_name; field = path; what = "TArray"})
  | TVec _ -> Error (Unsupported_field {type_name; field = path; what = "TVec"})

(** Lay out a sequence of named fields starting at absolute byte [offset] (the
    aligned C-ABI rule: each field is rounded up to its natural alignment, no
    trailing struct padding applied here — the caller pads). Returns all leaves
    (declaration order), the immediate-field offset table, and the end offset
    (one past the last field, before trailing padding). *)
and flatten_fields ~type_name ~prefix ~offset fields :
    (leaf list * (string * int) list * int, layout_error) result =
  let* leaves, offsets, endoff =
    List.fold_left
      (fun acc (name, ftype) ->
        let* leaves_acc, offsets_acc, running = acc in
        let field_off = align_up running (elttype_align ftype) in
        let* leaves, fsize =
          flatten_field ~type_name ~path:(prefix ^ name) ~offset:field_off ftype
        in
        Ok
          ( leaves_acc @ leaves,
            (name, field_off) :: offsets_acc,
            field_off + fsize ))
      (Ok ([], [], offset))
      fields
  in
  Ok (leaves, List.rev offsets, endoff)

(** {1 Layout computation} *)

let record_layout ~type_name (fields : (string * elttype) list) :
    (record_layout, layout_error) result =
  (* Single validated traversal: leaves, per-field offsets and end offset all
     derive from the same [flatten_fields] fold, so the aligned-offset rule is
     encoded exactly once. Total size is padded to the struct's alignment. *)
  let* leaves, field_offsets, endoff =
    flatten_fields ~type_name ~prefix:"" ~offset:0 fields
  in
  let size = align_up endoff (record_align fields) in
  Ok {rl_fields = field_offsets; rl_leaves = leaves; rl_size = size}

let variant_layout ~type_name (ctors : (string * elttype list) list) :
    (variant_layout, layout_error) result =
  (* Payload region starts at the union's natural alignment boundary after the
     int32 tag: max(4, max payload-member alignment). *)
  let payload_align =
    List.fold_left
      (fun m (_, args) ->
        List.fold_left (fun m t -> max m (elttype_align t)) m args)
      4
      ctors
  in
  let payload_offset = payload_align in
  let* rev_ctors, max_payload =
    List.fold_left
      (fun acc (tag, (name, args)) ->
        let* ctors_acc, max_acc = acc in
        let named_args =
          List.mapi (fun j t -> (Printf.sprintf "_%d" j, t)) args
        in
        let* leaves, _foffs, endoff =
          flatten_fields
            ~type_name
            ~prefix:(name ^ ".")
            ~offset:payload_offset
            named_args
        in
        (* Padded payload size (this constructor's C union member size). *)
        let this_align =
          List.fold_left (fun m (_, t) -> max m (elttype_align t)) 1 named_args
        in
        let payload_size = align_up (endoff - payload_offset) this_align in
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
      vl_payload_offset = payload_offset;
      vl_ctors = List.rev rev_ctors;
      (* Aligned rule: [round_up(payload_offset + max payload, max_align)],
         mirroring the C [struct { int tag; union {...} data; }] trailing pad. *)
      vl_size = align_up (payload_offset + max_payload) payload_align;
    }

let elttype_layout (t : elttype) : (layout, layout_error) result =
  match t with
  | TInt32 | TInt64 | TFloat16 | TFloat32 | TFloat64 | TBool | TUnit ->
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
