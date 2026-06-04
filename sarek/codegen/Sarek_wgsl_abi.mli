(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** WGSL kernel ABI descriptor.

    Describes the binding layout of a Sarek-generated WGSL compute shader in a
    plain data structure that can be serialized to JSON. The shape mirrors the
    binding assignments produced by {!Sarek_ir_wgsl.gen_bindings} exactly, so
    the JavaScript WebGPU runner can bind buffers and pack the uniform struct
    without parsing the shader source.

    All field names go through the same [escape_wgsl_name] transformation used
    by the code generator, so ABI names are identical to the names in the
    emitted WGSL.

    JSON serialization is hand-rolled (no yojson) so [sarek_codegen] stays
    FFI-free. *)

(** WGSL scalar type of a storage-buffer element or uniform field. *)
type element_type =
  | F32
  | I32
  | U32

(** A storage buffer binding (one per vector parameter, in declaration order). *)
type buffer = {
  name : string;
      (** Escaped WGSL identifier. *)
  binding : int;
      (** [@binding] index (0-based, vector order). *)
  element_type : element_type;
  access : string;
      (** Always ["read_write"] for storage buffers. *)
}

(** Kind of a field inside the Params uniform struct. *)
type field_kind =
  | Length of string
      (** [sarek_<vec>_length] — carries the element count of vector [<vec>]. *)
  | Scalar  (** A user scalar parameter. *)

(** A single field inside the [Params] uniform struct. *)
type field = {
  name : string;  (** Escaped WGSL identifier. *)
  field_type : element_type;
  offset : int;  (** Byte offset inside the struct (4 bytes per field). *)
  kind : field_kind;
}

(** The [Params] uniform struct binding.

    Present whenever there is at least one vector or scalar parameter (mirrors
    [gen_bindings]). The [binding] index equals the number of vector buffers.
    [byte_size] is the total number of fields × 4 rounded up to 16 (WebGPU
    minimum uniform buffer binding-size alignment). *)
type params = {
  binding : int;
  byte_size : int;
  fields : field list;
}

(** Complete ABI descriptor for one kernel. *)
type t = {
  kernel_name : string;
  workgroup_size : int * int * int;
  buffers : buffer list;
  params : params option;
}

(** [to_json abi] returns a JSON string representing [abi].

    Hand-rolled output — no external dependencies. The shape matches the ABI
    JSON contract documented in the implementer sub-brief. *)
val to_json : t -> string
