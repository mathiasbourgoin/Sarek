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

(** {1 Aggregate (record/variant) vector elements}

    Element addressing uses general byte-stride multiplication ([mul.wide.u32] +
    [add.u64] — FR-010); fields are typed [ld.global] / [st.global] at immediate
    offsets from the element base (FR-011). All offsets/strides come from
    {!Sarek_ir_layout} (FR-001). *)

(** [elt_is_aggregate alloc arr_name] is true when [arr_name]'s registered
    element type is a record or variant. *)
val elt_is_aggregate : reg_alloc -> string -> bool

(** [elt_stride t] is the byte stride of a vector element of type [t], from the
    validated {!Sarek_ir_layout} layout. Raises {!Ptx_codegen_error} on layout
    rejection. *)
val elt_stride : elttype -> int

(** [agg_field_path t path] folds field path [path] (outermost first) over
    aggregate element type [t]: returns the byte offset from the element base
    and the projected field's type. Raises {!Ptx_codegen_error} on unknown
    fields, variant projection, or layout rejection. *)
val agg_field_path : elttype -> string list -> int * elttype

(** [emit_agg_elem_addr buf alloc r_base r_idx ~stride ~is_shared ~arr_name]
    emits [mul.wide.u32 r_idx, stride] + [add.u64 r_base] and returns the u64
    element base address register. Raises {!Ptx_codegen_error} for shared-memory
    aggregate arrays (unsupported). *)
val emit_agg_elem_addr :
  Buffer.t ->
  reg_alloc ->
  string ->
  string ->
  stride:int ->
  is_shared:bool ->
  arr_name:string ->
  string

(** [emit_field_load buf alloc r_addr ~offset ty] emits one typed [ld.global] of
    the scalar field at [offset] from [r_addr]; returns the loaded register. *)
val emit_field_load :
  Buffer.t -> reg_alloc -> string -> offset:int -> elttype -> string

(** [emit_field_store buf r_addr ~offset ty r_val] emits one typed [st.global]
    of [r_val] at [offset] from [r_addr]. *)
val emit_field_store :
  Buffer.t -> string -> offset:int -> elttype -> string -> unit

(** [emit_agg_elem_load buf alloc r_addr ~offset t] materializes the SROA
    binding of a whole aggregate element: one typed [ld.global] per scalar leaf,
    in layout order (FR-012). Variant elements load the tag and every
    constructor's payload slots (FR-013 — never past the element size). *)
val emit_agg_elem_load :
  Buffer.t -> reg_alloc -> string -> offset:int -> elttype -> binding

(** [emit_agg_elem_store buf alloc r_addr ~offset t b] emits one typed
    [st.global] per scalar leaf of [b]. Callers must materialize [b] fully
    before calling so all loads precede the first store (EC-1). Variant elements
    store the tag then only the active constructor's slots via a tag branch
    chain. *)
val emit_agg_elem_store :
  Buffer.t -> reg_alloc -> string -> offset:int -> elttype -> binding -> unit
