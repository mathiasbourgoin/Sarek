(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX array load/store helpers: element stride, typed ld.global/st.global,
    ld.shared/st.shared and ld.local/st.local instruction emission, and
    element-type inference from the allocator table. *)

open Sarek_ir_types
open Sarek_ir_ptx_types

(** {1 Array load/store helpers}

    Emit a typed array read or write into [buf]. The element type determines the
    stride (shift 2 = 4 bytes, shift 3 = 8 bytes) and the PTX load/store
    qualifier. [~space] selects the state space: [Some SpaceShared] uses 32-bit
    pointer arithmetic and [ld/st.shared.*]; [Some SpaceLocal] uses 64-bit
    arithmetic and [ld/st.local.*] (per-thread stack memory); [None] (global)
    uses 64-bit and [ld/st.global.*]. All other element types raise
    [unsupported]. *)

let elt_shift = function
  | TFloat32 | TInt32 -> 2
  | TFloat64 | TInt64 -> 3
  | TRecord (n, _) | TVariant (n, _) ->
      fail
        ("PTX codegen: internal error: aggregate element type '" ^ n
       ^ "' reached the scalar array path (use the aggregate element helpers)")
  | t -> unsupported ("array element type " ^ ptx_reg_type_of t)

(** PTX space qualifier of an array's loads/stores. *)
let space_qualifier = function
  | Some SpaceShared -> "shared"
  | Some SpaceLocal -> "local"
  | None -> "global"

(** Element byte address: shared arrays use 32-bit pointer arithmetic (their
    base is a 32-bit window offset); local and global arrays use 64-bit. *)
let emit_elt_addr buf alloc r_base r_idx elt_type ~space =
  match space with
  | Some SpaceShared ->
      let r_off = new_u32 alloc in
      emit buf "shl.b32 %s, %s, %d;" r_off r_idx (elt_shift elt_type) ;
      let r_addr = new_u32 alloc in
      emit buf "add.u32 %s, %s, %s;" r_addr r_base r_off ;
      r_addr
  | Some SpaceLocal | None ->
      let r_idx64 = new_u64 alloc in
      emit buf "cvt.u64.u32 %s, %s;" r_idx64 r_idx ;
      let r_off = new_u64 alloc in
      emit buf "shl.b64 %s, %s, %d;" r_off r_idx64 (elt_shift elt_type) ;
      let r_addr = new_u64 alloc in
      emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
      r_addr

let emit_array_read buf alloc r_base r_idx elt_type ~space =
  let sp = space_qualifier space in
  let r_addr = emit_elt_addr buf alloc r_base r_idx elt_type ~space in
  match elt_type with
  | TFloat32 ->
      let r = new_f32 alloc in
      emit buf "ld.%s.f32 %s, [%s];" sp r r_addr ;
      r
  | TInt32 ->
      let r = new_u32 alloc in
      emit buf "ld.%s.s32 %s, [%s];" sp r r_addr ;
      r
  | TFloat64 ->
      let r = new_f64 alloc in
      emit buf "ld.%s.f64 %s, [%s];" sp r r_addr ;
      r
  | TInt64 ->
      let r = new_u64 alloc in
      emit buf "ld.%s.s64 %s, [%s];" sp r r_addr ;
      r
  | t -> unsupported (sp ^ " array read of element type " ^ ptx_reg_type_of t)

let emit_array_write buf alloc r_base r_idx r_val elt_type ~space =
  let sp = space_qualifier space in
  let r_addr = emit_elt_addr buf alloc r_base r_idx elt_type ~space in
  match elt_type with
  | TFloat32 -> emit buf "st.%s.f32 [%s], %s;" sp r_addr r_val
  | TInt32 -> emit buf "st.%s.s32 [%s], %s;" sp r_addr r_val
  | TFloat64 -> emit buf "st.%s.f64 [%s], %s;" sp r_addr r_val
  | TInt64 -> emit buf "st.%s.s64 [%s], %s;" sp r_addr r_val
  | t -> unsupported (sp ^ " array write of element type " ^ ptx_reg_type_of t)

let infer_elt_type alloc arr_name =
  match Hashtbl.find_opt alloc.arr_elt_types arr_name with
  | Some t -> t
  | None ->
      fail
        (Printf.sprintf "missing element-type metadata for array '%s'" arr_name)

(** {1 Aggregate (record/variant) vector elements}

    Element addressing uses general byte-stride multiplication
    ([mul.wide.u32 idx, stride] widening to the 64-bit offset, then [add.u64]
    with the base pointer — FR-010, the shape pinned on hardware by
    sarek-cuda/test/test_ptx_stride_spike.ml); field access is a typed
    [ld.global]/[st.global] at an immediate byte offset from the element base
    (FR-011). Every offset, size and stride comes from {!Sarek_ir_layout}
    (FR-001) — no offset arithmetic is duplicated here. *)

module L = Sarek_ir_layout

let layout_exn (r : ('a, L.layout_error) result) : 'a =
  match r with
  | Ok l -> l
  | Error e -> fail ("PTX codegen: " ^ L.layout_error_message e)

let record_layout_exn ~type_name fields =
  layout_exn (L.record_layout ~type_name fields)

let variant_layout_exn ~type_name ctors =
  layout_exn (L.variant_layout ~type_name ctors)

(** [elt_is_aggregate alloc arr_name] is true when [arr_name]'s registered
    element type is a record or variant. *)
let elt_is_aggregate alloc arr_name =
  match Hashtbl.find_opt alloc.arr_elt_types arr_name with
  | Some (TRecord _ | TVariant _) -> true
  | _ -> false

(** [elt_stride t] is the byte stride of one vector element of type [t], from
    the validated layout. *)
let elt_stride (t : elttype) : int =
  match layout_exn (L.elttype_layout t) with
  | L.LScalar {size; _} -> size
  | L.LRecord rl -> rl.L.rl_size
  | L.LVariant vl -> vl.L.vl_size

(** [agg_field_path t path] folds field path [path] (outermost field first) over
    aggregate element type [t]: returns the accumulated byte offset from the
    element base and the projected field's type. *)
let agg_field_path (t : elttype) (path : string list) : int * elttype =
  let rec go offset t = function
    | [] -> (offset, t)
    | f :: rest -> (
        match t with
        | TRecord (name, fields) -> (
            let rl = record_layout_exn ~type_name:name fields in
            match
              (List.assoc_opt f rl.L.rl_fields, List.assoc_opt f fields)
            with
            | Some foff, Some fty -> go (offset + foff) fty rest
            | _ ->
                fail
                  (Printf.sprintf
                     "PTX codegen: record '%s' has no field '%s' (available: \
                      %s)"
                     name
                     f
                     (String.concat ", " (List.map fst fields))))
        | TVariant (name, _) ->
            fail
              ("PTX codegen: field access '." ^ f
             ^ "' on a vector element of variant type '" ^ name
             ^ "'; use match to inspect a variant")
        | _ -> fail ("PTX codegen: field access '." ^ f ^ "' on a non-record"))
  in
  go 0 t path

(** [emit_agg_elem_addr buf alloc r_base r_idx ~stride ~space ~arr_name] emits
    the element base address of an aggregate vector element: [mul.wide.u32] of
    the u32 index by the byte stride, then [add.u64] with the base pointer.
    Shared- and local-memory aggregate arrays are not supported. *)
let emit_agg_elem_addr buf alloc r_base r_idx ~stride ~space ~arr_name =
  if space <> None then
    unsupported
      (Printf.sprintf
         "%s-memory array '%s' with record/variant elements (aggregate \
          elements are supported in global vectors only; use a vector \
          parameter or scalar %s arrays)"
         (space_qualifier space)
         arr_name
         (space_qualifier space))
  else begin
    let r_off = new_u64 alloc in
    emit buf "mul.wide.u32 %s, %s, %d;" r_off r_idx stride ;
    let r_addr = new_u64 alloc in
    emit buf "add.u64 %s, %s, %s;" r_addr r_base r_off ;
    r_addr
  end

(** Immediate-offset address operand: [[%rd]] at offset 0, [[%rd+8]] otherwise
    (the exact shape proven by the stride spike). *)
let addr_operand r_addr offset =
  if offset = 0 then Printf.sprintf "[%s]" r_addr
  else Printf.sprintf "[%s+%d]" r_addr offset

(** [emit_field_load buf alloc r_addr ~offset ty] emits one typed [ld.global] of
    the scalar field at [offset] from the element base [r_addr]. *)
let emit_field_load buf alloc r_addr ~offset ty =
  let ld suffix fresh =
    let r = fresh alloc in
    emit buf "ld.global.%s %s, %s;" suffix r (addr_operand r_addr offset) ;
    r
  in
  match ty with
  | TFloat32 -> ld "f32" new_f32
  | TInt32 -> ld "s32" new_u32
  | TBool -> ld "u32" new_u32
  | TFloat64 -> ld "f64" new_f64
  | TInt64 -> ld "s64" new_u64
  | _ -> unsupported "aggregate field load of non-scalar leaf type"

(** [emit_field_store buf r_addr ~offset ty r_val] emits one typed [st.global]
    of [r_val] into the scalar field at [offset] from [r_addr]. *)
let emit_field_store buf r_addr ~offset ty r_val =
  let suffix =
    match ty with
    | TFloat32 -> "f32"
    | TInt32 -> "s32"
    | TBool -> "u32"
    | TFloat64 -> "f64"
    | TInt64 -> "s64"
    | _ -> unsupported "aggregate field store of non-scalar leaf type"
  in
  emit buf "st.global.%s %s, %s;" suffix (addr_operand r_addr offset) r_val

(** Load a variant element: tag at [vl_tag_offset], then EVERY constructor's
    payload slots at their layout offsets. Loading all slots (FR-013,
    implementer's choice) keeps the loaded binding shape-uniform with locally
    constructed variants; every slot offset is < [vl_size], so no load reads
    past the element, and slots of non-active constructors are never selected by
    the tag at runtime. Aggregate payload args are rejected (untested,
    layout-flattened shape would be ambiguous). *)
let emit_variant_elem_load buf alloc r_addr ~offset name ctors =
  let vl = variant_layout_exn ~type_name:name ctors in
  let tag_reg = new_u32 alloc in
  emit
    buf
    "ld.global.u32 %s, %s;"
    tag_reg
    (addr_operand r_addr (offset + vl.L.vl_tag_offset)) ;
  let load_ctor (cn, tys) (cl : L.ctor_layout) =
    if List.length tys <> List.length cl.L.ctor_leaves then
      unsupported
        (Printf.sprintf
           "constructor '%s' of variant '%s' with an aggregate payload \
            argument in a vector element (use scalar payload arguments)"
           cn
           name)
    else
      ( cn,
        List.map
          (fun (leaf : L.leaf) ->
            Scalar
              (emit_field_load
                 buf
                 alloc
                 r_addr
                 ~offset:(offset + leaf.L.leaf_offset)
                 leaf.L.leaf_type))
          cl.L.ctor_leaves )
  in
  Agg
    (AVariant
       {vname = name; tag_reg; ctors = List.map2 load_ctor ctors vl.L.vl_ctors})

(** [emit_agg_elem_load buf alloc r_addr ~offset t] materializes the SROA
    binding of a whole aggregate element (FR-012): one typed [ld.global] per
    scalar leaf, in layout (declaration) order. *)
let rec emit_agg_elem_load buf alloc r_addr ~offset (t : elttype) : binding =
  match t with
  | TRecord (name, fields) ->
      let rl = record_layout_exn ~type_name:name fields in
      Agg
        (ARecord
           (List.map
              (fun (fname, fty) ->
                let foff = List.assoc fname rl.L.rl_fields in
                ( fname,
                  emit_agg_elem_load
                    buf
                    alloc
                    r_addr
                    ~offset:(offset + foff)
                    fty ))
              fields))
  | TVariant (name, ctors) ->
      emit_variant_elem_load buf alloc r_addr ~offset name ctors
  | t -> Scalar (emit_field_load buf alloc r_addr ~offset t)

(** Store a variant element: tag at [vl_tag_offset] unconditionally, then a tag
    branch chain storing ONLY the active constructor's payload slots — the
    payload regions of all constructors overlap at the payload offset, and the
    non-active constructors' slot registers are never-written (undefined), so an
    unconditional store of every slot would clobber the live payload. *)
let store_variant_elem buf alloc ~store_ctor r_addr ~offset name ctors v_ctors
    tag_reg =
  let vl = variant_layout_exn ~type_name:name ctors in
  (* The layout ctor list (from the element type) and the binding's ctor list
     (from the kernel's variant declarations) come from independent sources;
     they agree within one kernel but can diverge if fusion ever concatenates
     two kernels declaring a variant under the same name with different
     constructors. Validate name-wise agreement before the positional zip so
     that case fails loudly instead of writing mispaired bytes. *)
  (if
     List.length vl.L.vl_ctors <> List.length v_ctors
     || not
          (List.for_all2
             (fun (cl : L.ctor_layout) (cn, _) -> cl.L.ctor_name = cn)
             vl.L.vl_ctors
             v_ctors)
   then
     let names l = String.concat ", " l in
     unsupported
       (Printf.sprintf
          "variant '%s': element-type constructors [%s] disagree with the \
           kernel's declaration [%s] (two kernels declaring '%s' differently \
           were probably fused); rename one of the variant types"
          name
          (names
             (List.map
                (fun (cl : L.ctor_layout) -> cl.L.ctor_name)
                vl.L.vl_ctors))
          (names (List.map fst v_ctors))
          name)) ;
  emit
    buf
    "st.global.u32 %s, %s;"
    (addr_operand r_addr (offset + vl.L.vl_tag_offset))
    tag_reg ;
  List.iter2
    (fun (cl : L.ctor_layout) (cn, bs) ->
      if bs <> [] then begin
        let l_skip = new_label alloc in
        let p = new_pred alloc in
        emit buf "setp.ne.u32 %s, %s, %d;" p tag_reg cl.L.ctor_tag ;
        emit buf "@%s bra %s;" p l_skip ;
        store_ctor cn cl bs ;
        emit_label buf l_skip
      end)
    vl.L.vl_ctors
    v_ctors

(** [emit_agg_elem_store buf alloc r_addr ~offset t b] stores binding [b] into
    an aggregate element: one typed [st.global] per scalar leaf. The caller must
    have fully materialized [b] beforehand so that every load precedes the first
    store (EC-1 / FR-012). *)
let rec emit_agg_elem_store buf alloc r_addr ~offset (t : elttype) (b : binding)
    : unit =
  match (t, b) with
  | TRecord (name, fields), Agg (ARecord fbs) ->
      let rl = record_layout_exn ~type_name:name fields in
      List.iter
        (fun (fname, fty) ->
          let foff = List.assoc fname rl.L.rl_fields in
          match List.assoc_opt fname fbs with
          | Some fb ->
              emit_agg_elem_store
                buf
                alloc
                r_addr
                ~offset:(offset + foff)
                fty
                fb
          | None ->
              fail
                ("PTX codegen: internal error: record shape mismatch storing \
                  element of '" ^ name ^ "' (missing field '" ^ fname ^ "')"))
        fields
  | TVariant (name, ctors), Agg (AVariant v) ->
      let store_ctor cn (cl : L.ctor_layout) bs =
        if List.length bs <> List.length cl.L.ctor_leaves then
          unsupported
            (Printf.sprintf
               "constructor '%s' of variant '%s' with an aggregate payload \
                argument in a vector element (use scalar payload arguments)"
               cn
               name)
        else
          List.iter2
            (fun (leaf : L.leaf) b ->
              match b with
              | Scalar r ->
                  emit_field_store
                    buf
                    r_addr
                    ~offset:(offset + leaf.L.leaf_offset)
                    leaf.L.leaf_type
                    r
              | Agg _ ->
                  unsupported
                    ("aggregate payload slot in variant '" ^ name
                   ^ "' vector element"))
            cl.L.ctor_leaves
            bs
      in
      store_variant_elem
        buf
        alloc
        ~store_ctor
        r_addr
        ~offset
        name
        ctors
        v.ctors
        v.tag_reg
  | (TRecord _ | TVariant _), _ | _, Agg _ ->
      fail
        "PTX codegen: internal error: aggregate shape mismatch storing a \
         vector element (scalar/record/variant kinds differ)"
  | t, Scalar r -> emit_field_store buf r_addr ~offset t r

(** {1 Structure-of-Arrays (SoA) custom-vector element access}

    A SoA custom-vector parameter (selected via [~soa_params]) stores each
    scalar leaf of its record type in its own contiguous device buffer, bound to
    its own base-pointer register ([soa_leaf.sl_base]). Every field access is
    then a plain coalesced scalar-array access at that leaf's base and index —
    [mul.wide]/[shl] of the shared index by the leaf's own width, then a typed
    [ld.global]/[st.global] — exactly what
    {!emit_array_read}/{!emit_array_write} already emit for scalar vectors. This
    is strictly less work than the AoS aggregate path (no packed element stride,
    no byte-offset folding), and, being per-leaf-contiguous, it is what restores
    full memory coalescing for single-field access over a wide record (the Tier
    1b headline win).

    v1 supports flat records only (validated at parameter time in
    [Sarek_ir_ptx_kernel.emit_params]); leaf paths are therefore always a single
    field name. A [TBool] leaf is addressed as its 4-byte [u32] storage. *)

(* Addressing/load width of a leaf: bool is stored as its 4-byte u32 form; the
   scalar-array path has no dedicated bool case. Bit-preserving for a 0/1 bool. *)
let soa_leaf_ld_type (l : soa_leaf) : elttype =
  match l.sl_type with TBool -> TInt32 | t -> t

let soa_field_or_fail alloc arr_name field : soa_leaf =
  match soa_leaf_of_field alloc arr_name field with
  | Some l -> l
  | None ->
      fail
        (Printf.sprintf
           "PTX codegen: SoA vector '%s' has no scalar leaf for field '%s'"
           arr_name
           field)

(* A flat-record SoA field path is a single field name; nested paths cannot
   occur (nested-record SoA params are rejected at parameter time), but guard
   defensively so a shape bug fails loudly rather than mis-addressing. *)
let soa_single_field arr_name (path : string list) : string =
  match path with
  | [f] -> f
  | _ ->
      fail
        (Printf.sprintf
           "PTX codegen: nested field path %s on SoA vector '%s' (v1 SoA is \
            flat records only)"
           (String.concat "." path)
           arr_name)

(** Whole-element SoA read [v.(i)]: one coalesced scalar [ld.global] per leaf
    from its own base, assembled into the same [ARecord] binding shape the AoS
    whole-element load produces (so every downstream consumer is unchanged). *)
let emit_soa_elem_load buf alloc r_idx arr_name : binding =
  Agg
    (ARecord
       (List.map
          (fun (l : soa_leaf) ->
            ( l.sl_field,
              Scalar
                (emit_array_read
                   buf
                   alloc
                   l.sl_base
                   r_idx
                   (soa_leaf_ld_type l)
                   ~space:None) ))
          (soa_leaves alloc arr_name)))

(** Single-field SoA read [v.(i).field]: one coalesced scalar [ld.global] at
    that leaf's base — the untouched leaves are never loaded. *)
let emit_soa_field_load buf alloc r_idx arr_name (path : string list) : binding
    =
  let field = soa_single_field arr_name path in
  let l = soa_field_or_fail alloc arr_name field in
  Scalar
    (emit_array_read buf alloc l.sl_base r_idx (soa_leaf_ld_type l) ~space:None)

(** Whole-element SoA write [v.(i) <- e]: one coalesced scalar [st.global] per
    leaf. The value binding must be fully materialized by the caller first
    (EC-1: every load precedes the first store). *)
let emit_soa_elem_store buf alloc r_idx arr_name (b : binding) : unit =
  match b with
  | Agg (ARecord fbs) ->
      List.iter
        (fun (l : soa_leaf) ->
          match List.assoc_opt l.sl_field fbs with
          | Some (Scalar r) ->
              emit_array_write
                buf
                alloc
                l.sl_base
                r_idx
                r
                (soa_leaf_ld_type l)
                ~space:None
          | Some (Agg _) ->
              fail
                (Printf.sprintf
                   "PTX codegen: internal error: nested field '%s' in SoA \
                    element store of '%s' (flat records only)"
                   l.sl_field
                   arr_name)
          | None ->
              fail
                (Printf.sprintf
                   "PTX codegen: internal error: SoA element store of '%s' \
                    missing field '%s'"
                   arr_name
                   l.sl_field))
        (soa_leaves alloc arr_name)
  | Agg (AVariant _) ->
      fail
        (Printf.sprintf
           "PTX codegen: variant value storing a SoA element of '%s' (v1 SoA \
            is flat records only)"
           arr_name)
  | Scalar _ ->
      fail
        (Printf.sprintf
           "PTX codegen: internal error: scalar binding storing whole SoA \
            record element of '%s'"
           arr_name)

(** Single-field SoA write [v.(i).field <- e]: one coalesced scalar [st.global]
    at that leaf's base. *)
let emit_soa_field_store buf alloc r_idx arr_name (path : string list)
    (b : binding) : unit =
  let field = soa_single_field arr_name path in
  let l = soa_field_or_fail alloc arr_name field in
  match b with
  | Scalar r ->
      emit_array_write
        buf
        alloc
        l.sl_base
        r_idx
        r
        (soa_leaf_ld_type l)
        ~space:None
  | Agg _ ->
      fail
        (Printf.sprintf
           "PTX codegen: aggregate value stored into scalar SoA field '%s' of \
            '%s'"
           field
           arr_name)
