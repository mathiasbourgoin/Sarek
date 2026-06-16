(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX array load/store helpers: element stride, typed ld.global/st.global
    instruction emission, and element-type inference from the allocator table.
*)

open Sarek_ir_types
open Sarek_ir_ptx_types

(** {1 Array load/store helpers}

    Emit a typed array read (ld.global) or write (st.global) into [buf]. The
    element type determines the stride (2 = 4 bytes, 3 = 8 bytes) and the PTX
    load/store qualifier. All other element types raise [unsupported]. *)

let elt_shift = function
  | TFloat32 | TInt32 -> 2
  | TFloat64 | TInt64 -> 3
  | t -> unsupported ("array element type " ^ ptx_reg_type_of t)

let emit_array_read buf alloc r_base r_idx elt_type ~is_shared =
  if is_shared then begin
    let r_off = new_u32 alloc in
    emit buf "shl.b32 %s, %s, %d;" r_off r_idx (elt_shift elt_type) ;
    let r_addr = new_u32 alloc in
    emit buf "add.u32 %s, %s, %s;" r_addr r_base r_off ;
    match elt_type with
    | TFloat32 ->
        let r = new_f32 alloc in
        emit buf "ld.shared.f32 %s, [%s];" r r_addr ;
        r
    | TInt32 ->
        let r = new_u32 alloc in
        emit buf "ld.shared.s32 %s, [%s];" r r_addr ;
        r
    | TFloat64 ->
        let r = new_f64 alloc in
        emit buf "ld.shared.f64 %s, [%s];" r r_addr ;
        r
    | TInt64 ->
        let r = new_u64 alloc in
        emit buf "ld.shared.s64 %s, [%s];" r r_addr ;
        r
    | t -> unsupported ("shared array read of element type " ^ ptx_reg_type_of t)
  end
  else begin
    let r_idx64 = new_u64 alloc in
    emit buf "cvt.u64.u32 %s, %s;" r_idx64 r_idx ;
    let r_off = new_u64 alloc in
    emit buf "shl.b64 %s, %s, %d;" r_off r_idx64 (elt_shift elt_type) ;
    let r_addr = new_u64 alloc in
    emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
    match elt_type with
    | TFloat32 ->
        let r = new_f32 alloc in
        emit buf "ld.global.f32 %s, [%s];" r r_addr ;
        r
    | TInt32 ->
        let r = new_u32 alloc in
        emit buf "ld.global.s32 %s, [%s];" r r_addr ;
        r
    | TFloat64 ->
        let r = new_f64 alloc in
        emit buf "ld.global.f64 %s, [%s];" r r_addr ;
        r
    | TInt64 ->
        let r = new_u64 alloc in
        emit buf "ld.global.s64 %s, [%s];" r r_addr ;
        r
    | t -> unsupported ("array read of element type " ^ ptx_reg_type_of t)
  end

let emit_array_write buf alloc r_base r_idx r_val elt_type ~is_shared =
  if is_shared then begin
    let r_off = new_u32 alloc in
    emit buf "shl.b32 %s, %s, %d;" r_off r_idx (elt_shift elt_type) ;
    let r_addr = new_u32 alloc in
    emit buf "add.u32 %s, %s, %s;" r_addr r_base r_off ;
    match elt_type with
    | TFloat32 -> emit buf "st.shared.f32 [%s], %s;" r_addr r_val
    | TInt32 -> emit buf "st.shared.s32 [%s], %s;" r_addr r_val
    | TFloat64 -> emit buf "st.shared.f64 [%s], %s;" r_addr r_val
    | TInt64 -> emit buf "st.shared.s64 [%s], %s;" r_addr r_val
    | t ->
        unsupported ("shared array write of element type " ^ ptx_reg_type_of t)
  end
  else begin
    let r_idx64 = new_u64 alloc in
    emit buf "cvt.u64.u32 %s, %s;" r_idx64 r_idx ;
    let r_off = new_u64 alloc in
    emit buf "shl.b64 %s, %s, %d;" r_off r_idx64 (elt_shift elt_type) ;
    let r_addr = new_u64 alloc in
    emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
    match elt_type with
    | TFloat32 -> emit buf "st.global.f32 [%s], %s;" r_addr r_val
    | TInt32 -> emit buf "st.global.s32 [%s], %s;" r_addr r_val
    | TFloat64 -> emit buf "st.global.f64 [%s], %s;" r_addr r_val
    | TInt64 -> emit buf "st.global.s64 [%s], %s;" r_addr r_val
    | t -> unsupported ("array write of element type " ^ ptx_reg_type_of t)
  end

let infer_elt_type alloc arr_name =
  match Hashtbl.find_opt alloc.arr_elt_types arr_name with
  | Some t -> t
  | None ->
      fail
        (Printf.sprintf "missing element-type metadata for array '%s'" arr_name)
