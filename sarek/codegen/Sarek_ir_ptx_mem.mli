(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX array load/store helpers.

    Emits typed [ld.global]/[st.global] or [ld.shared]/[st.shared] instruction
    sequences into a {!Buffer.t}. Global paths use 64-bit pointer arithmetic;
    shared paths use 32-bit. The element type is either supplied explicitly or
    inferred from the allocator's element-type table. *)

open Sarek_ir_types
open Sarek_ir_ptx_types

(** [elt_shift t] returns the log2 byte stride for element type [t] (2 for
    4-byte types, 3 for 8-byte types). Raises {!Ptx_codegen_error} for
    unsupported element types. *)
val elt_shift : elttype -> int

(** [emit_array_read buf alloc r_base r_idx elt_type ~is_shared] emits a
    pointer-arithmetic sequence followed by a typed load and returns the
    destination register name. When [~is_shared:true], uses 32-bit pointer
    arithmetic and [ld.shared.*]; otherwise uses 64-bit and [ld.global.*].

    Ownership: [buf] is mutated; [alloc] counters are incremented. *)
val emit_array_read :
  Buffer.t ->
  reg_alloc ->
  string ->
  string ->
  elttype ->
  is_shared:bool ->
  string

(** [emit_array_write buf alloc r_base r_idx r_val elt_type ~is_shared] emits a
    pointer-arithmetic sequence followed by a typed store. When
    [~is_shared:true], uses 32-bit pointer arithmetic and [st.shared.*];
    otherwise uses 64-bit and [st.global.*].

    Ownership: [buf] is mutated; [alloc] counters are incremented. *)
val emit_array_write :
  Buffer.t ->
  reg_alloc ->
  string ->
  string ->
  string ->
  elttype ->
  is_shared:bool ->
  unit

(** [infer_elt_type alloc arr_name] returns the element type registered for
    [arr_name] in [alloc.arr_elt_types], or raises {!Ptx_codegen_error} if none
    is recorded. *)
val infer_elt_type : reg_alloc -> string -> elttype
