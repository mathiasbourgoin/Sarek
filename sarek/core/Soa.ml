(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* See Soa.mli for the design rationale. *)

module Layout = Sarek_ir_layout
module Helpers = Vector_types.Custom_helpers

type leaf = {
  path : string;
  ty : Sarek_ir_types.elttype;
  aos_offset : int;
  size : int;
}

type plan = {name : string; leaves : leaf list; aos_stride : int}

exception Unsupported of string

let reject_non_flat name fields =
  List.iter
    (fun (fname, ty) ->
      match (ty : Sarek_ir_types.elttype) with
      | TInt32 | TInt64 | TFloat32 | TFloat64 | TBool | TUnit -> ()
      | TFloat16 ->
          (* Consistent with Sarek_ir_layout.flatten_field: f16 aggregate
             fields are out of scope for #57 slice 1 (the host PPX has no
             read_float16/write_float16 marshaller), so an f16 field cannot be
             SoA-split either. *)
          raise
            (Unsupported
               (Printf.sprintf
                  "float16 field %S in %S: f16 record fields unsupported (#57 \
                   slice 1 supports f16 vectors, not aggregate fields)"
                  fname
                  name))
      | TRecord _ ->
          raise
            (Unsupported
               (Printf.sprintf
                  "nested-record field %S in %S: v1 SoA supports flat records \
                   only"
                  fname
                  name))
      | TVariant _ ->
          raise
            (Unsupported
               (Printf.sprintf
                  "variant field %S in %S: variants have no well-defined SoA \
                   split"
                  fname
                  name))
      | TArray _ | TVec _ ->
          raise
            (Unsupported
               (Printf.sprintf
                  "array/vector field %S in %S unsupported"
                  fname
                  name)))
    fields

let plan ~name fields =
  reject_non_flat name fields ;
  match Layout.record_layout ~type_name:name fields with
  | Error e -> raise (Unsupported (Layout.layout_error_message e))
  | Ok rl ->
      let leaves =
        List.map
          (fun (l : Layout.leaf) ->
            {
              path = l.leaf_path;
              ty = l.leaf_type;
              aos_offset = l.leaf_offset;
              size = l.leaf_size;
            })
          rl.rl_leaves
      in
      {name; leaves; aos_stride = rl.rl_size}

let plan_of_elttype (t : Sarek_ir_types.elttype) =
  match t with
  | TRecord (name, fields) -> plan ~name fields
  | _ -> raise (Unsupported "SoA plan requires a record (TRecord) element type")

let num_leaves p = List.length p.leaves

let aos_bytes p ~length = length * p.aos_stride

(* Bit-preserving copy of one [size]-byte scalar. All Sarek scalar leaves are
   4 or 8 bytes; the byte-loop is a defensive fallback. *)
let copy_word ~size ~src ~src_off ~dst ~dst_off =
  match size with
  | 4 -> Helpers.write_int32 dst dst_off (Helpers.read_int32 src src_off)
  | 8 -> Helpers.write_int64 dst dst_off (Helpers.read_int64 src src_off)
  | _ ->
      (* Defensive fallback (no Sarek scalar hits this): plain byte copy. *)
      for b = 0 to size - 1 do
        let s = Ctypes.(from_voidp uint8_t src +@ (src_off + b)) in
        let d = Ctypes.(from_voidp uint8_t dst +@ (dst_off + b)) in
        Ctypes.(d <-@ !@s)
      done

let scatter p ~aos ~length ~leaves =
  List.iteri
    (fun k leaf ->
      let dst = leaves.(k) in
      for i = 0 to length - 1 do
        copy_word
          ~size:leaf.size
          ~src:aos
          ~src_off:((i * p.aos_stride) + leaf.aos_offset)
          ~dst
          ~dst_off:(i * leaf.size)
      done)
    p.leaves

let gather p ~leaves ~length ~aos =
  List.iteri
    (fun k leaf ->
      let src = leaves.(k) in
      for i = 0 to length - 1 do
        copy_word
          ~size:leaf.size
          ~src
          ~src_off:(i * leaf.size)
          ~dst:aos
          ~dst_off:((i * p.aos_stride) + leaf.aos_offset)
      done)
    p.leaves
