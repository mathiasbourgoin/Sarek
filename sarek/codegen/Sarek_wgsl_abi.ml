(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_wgsl_abi - WGSL kernel ABI descriptor
 *
 * Plain data type plus hand-rolled JSON serializer. No IR dependencies.
 * No yojson/ctypes — sarek_codegen stays FFI-free.
 ******************************************************************************)

type element_type =
  | F32
  | I32
  | U32

type buffer = {
  name : string;
  binding : int;
  element_type : element_type;
  access : string;
}

type field_kind =
  | Length of string
  | Scalar

type field = {
  name : string;
  field_type : element_type;
  offset : int;
  kind : field_kind;
}

type params = {
  binding : int;
  byte_size : int;
  fields : field list;
}

type t = {
  kernel_name : string;
  workgroup_size : int * int * int;
  buffers : buffer list;
  params : params option;
}

(** {1 JSON helpers} *)

let string_of_element_type = function
  | F32 -> "f32"
  | I32 -> "i32"
  | U32 -> "u32"

let json_string s =
  (* Minimal JSON string quoting: escape backslash, double-quote, and the
     common control characters. ABI names are WGSL identifiers so in practice
     only ASCII alphanumerics and underscores appear; we still quote correctly
     for correctness. *)
  let buf = Buffer.create (String.length s + 2) in
  Buffer.add_char buf '"' ;
  String.iter
    (fun c ->
      match c with
      | '"' -> Buffer.add_string buf "\\\""
      | '\\' -> Buffer.add_string buf "\\\\"
      | '\n' -> Buffer.add_string buf "\\n"
      | '\r' -> Buffer.add_string buf "\\r"
      | '\t' -> Buffer.add_string buf "\\t"
      | c -> Buffer.add_char buf c)
    s ;
  Buffer.add_char buf '"' ;
  Buffer.contents buf

let json_buffer (b : buffer) =
  Printf.sprintf
    {|{"name":%s,"binding":%d,"elementType":%s,"access":%s}|}
    (json_string b.name)
    b.binding
    (json_string (string_of_element_type b.element_type))
    (json_string b.access)

let json_field (f : field) =
  let kind_part =
    match f.kind with
    | Length vec_name ->
        Printf.sprintf
          {|,"kind":"length","of":%s|}
          (json_string vec_name)
    | Scalar -> {|,"kind":"scalar"|}
  in
  Printf.sprintf
    {|{"name":%s,"type":%s,"offset":%d%s}|}
    (json_string f.name)
    (json_string (string_of_element_type f.field_type))
    f.offset
    kind_part

let json_params (p : params) =
  let fields_json =
    String.concat "," (List.map json_field p.fields)
  in
  Printf.sprintf
    {|{"binding":%d,"byteSize":%d,"fields":[%s]}|}
    p.binding
    p.byte_size
    fields_json

let to_json (abi : t) =
  let bx, by, bz = abi.workgroup_size in
  let buffers_json =
    String.concat "," (List.map json_buffer abi.buffers)
  in
  let params_json =
    match abi.params with
    | None -> "null"
    | Some p -> json_params p
  in
  Printf.sprintf
    {|{"kernelName":%s,"workgroupSize":[%d,%d,%d],"buffers":[%s],"params":%s}|}
    (json_string abi.kernel_name)
    bx
    by
    bz
    buffers_json
    params_json
