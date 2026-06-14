(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX array load/store helpers.

    Emits typed [ld.global] / [st.global] instruction sequences into a
    {!Buffer.t}. All operations use 64-bit pointer arithmetic with shift-based
    stride calculation. The element type is either supplied explicitly or
    inferred from the allocator's element-type table. *)

open Sarek_ir_types
open Sarek_ir_ptx_types

(** [elt_shift t] returns the log2 byte stride for element type [t] (2 for
    4-byte types, 3 for 8-byte types). Raises {!Ptx_codegen_error} for
    unsupported element types. *)
val elt_shift : elttype -> int

(** [emit_array_read buf alloc r_base r_idx elt_type] emits a pointer-arithmetic
    sequence followed by a typed [ld.global] instruction and returns the
    destination register name.

    Ownership: [buf] is mutated; [alloc] counters are incremented. *)
val emit_array_read :
  Buffer.t -> reg_alloc -> string -> string -> elttype -> string

(** [emit_array_write buf alloc r_base r_idx r_val elt_type] emits a
    pointer-arithmetic sequence followed by a typed [st.global] instruction.

    Ownership: [buf] is mutated; [alloc] counters are incremented. *)
val emit_array_write :
  Buffer.t -> reg_alloc -> string -> string -> string -> elttype -> unit

(** [infer_elt_type alloc arr_name] returns the element type registered for
    [arr_name] in [alloc.arr_elt_types], or [TFloat32] if none is recorded. *)
val infer_elt_type : reg_alloc -> string -> elttype
