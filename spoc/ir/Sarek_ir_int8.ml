(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** See the interface for the semantics. *)

open Sarek_ir_types

let is_8bit = Sarek_ir_int8_check.is_8bit

let byte_elt_of_type = Sarek_ir_int8_check.byte_elt_of_type

let byte_arrays = Sarek_ir_int8_check.byte_arrays

let refusal = Sarek_ir_int8_check.refusal

(** {1 Semantics} *)

let narrow t (v : int32) =
  match t with
  | TUint8 -> Int32.logand v 0xffl
  | TInt8 -> Int32.shift_right (Int32.shift_left v 24) 24
  | _ -> v

(** {1 Normalisation} *)

let rec map_expr bytes e =
  let r = map_expr bytes in
  let elt_of_base = function
    | EVar v -> byte_elt_of_type v.var_type
    | _ -> None
  in
  match e with
  | EConst _ | EVar _ | EArrayLen _ -> e
  | EArrayRead (a, i) -> (
      let e' = EArrayRead (a, r i) in
      match List.assoc_opt a bytes with Some t -> ECast (t, e') | None -> e')
  | EArrayReadExpr (b, i) -> (
      let e' = EArrayReadExpr (r b, r i) in
      match elt_of_base b with Some t -> ECast (t, e') | None -> e')
  | EBinop (op, a, b) -> EBinop (op, r a, r b)
  | EUnop (op, a) -> EUnop (op, r a)
  | ERecordField (a, f) -> ERecordField (r a, f)
  | EIntrinsic (p, n, args) -> EIntrinsic (p, n, List.map r args)
  | ECast (t, a) -> ECast (t, r a)
  | ETuple es -> ETuple (List.map r es)
  | EApp (f, args) -> EApp (r f, List.map r args)
  | ERecord (n, fs) -> ERecord (n, List.map (fun (f, a) -> (f, r a)) fs)
  | EVariant (n, c, args) -> EVariant (n, c, List.map r args)
  | EArrayCreate (t, sz, ms) -> EArrayCreate (t, r sz, ms)
  | EIf (c, a, b) -> EIf (r c, r a, r b)
  | EMatch (s, cases) -> EMatch (r s, List.map (fun (p, a) -> (p, r a)) cases)

let rec map_lvalue bytes = function
  | LVar _ as lv -> lv
  | LArrayElem (a, i) -> LArrayElem (a, map_expr bytes i)
  | LArrayElemExpr (b, i) -> LArrayElemExpr (map_expr bytes b, map_expr bytes i)
  | LRecordField (b, f) -> LRecordField (map_lvalue bytes b, f)

(** The 8-bit element type a store into [lv] narrows to, if any. *)
let store_elt bytes = function
  | LArrayElem (a, _) -> List.assoc_opt a bytes
  | LArrayElemExpr (EVar v, _) -> byte_elt_of_type v.var_type
  | LArrayElemExpr _ | LVar _ | LRecordField _ -> None

let rec map_stmt bytes s =
  let e = map_expr bytes and r = map_stmt bytes in
  match s with
  | SBarrier | SWarpBarrier | SEmpty | SMemFence | SNative _ -> s
  | SAssign (lv, x) -> (
      let x' = e x in
      let lv' = map_lvalue bytes lv in
      match (store_elt bytes lv, x') with
      | Some t, ECast (t', _) when t = t' -> SAssign (lv', x')
      | Some t, _ -> SAssign (lv', ECast (t, x'))
      | None, _ -> SAssign (lv', x'))
  | SSeq ss -> SSeq (List.map r ss)
  | SIf (c, t, el) -> SIf (e c, r t, Option.map r el)
  | SWhile (c, b) -> SWhile (e c, r b)
  | SFor (v, lo, hi, d, b) -> SFor (v, e lo, e hi, d, r b)
  | SMatch (sc, cases) -> SMatch (e sc, List.map (fun (p, b) -> (p, r b)) cases)
  | SReturn x -> SReturn (e x)
  | SExpr x -> SExpr (e x)
  | SLet (v, x, b) -> SLet (v, e x, r b)
  | SLetMut (v, x, b) -> SLetMut (v, e x, r b)
  | SPragma (h, b) -> SPragma (h, r b)
  | SBlock b -> SBlock (r b)
  | SCoopmat op ->
      SCoopmat
        (match op with
        | CM_decl _ | CM_muladd _ -> op
        | CM_load c -> CM_load {c with index = e c.index; stride = e c.stride}
        | CM_store c -> CM_store {c with index = e c.index; stride = e c.stride})

(** {1 8-bit local scalars}

    A [let] of an 8-bit type holds the narrowed value, as a C [unsigned char]
    local does: [SLet (v : TUint8, e, b)] means
    [SLet (v : TInt32, ECast (TUint8, e), b)], and an assignment to a mutable
    one narrows too. {!lower_scalars} rewrites the former into the latter, for
    the targets with no implicit conversion between 8- and 32-bit integers
    (GLSL, PTX) and for the interpreter. The C-family emitters do not need it:
    their [unsigned char] local is that semantics, and leaving them alone keeps
    their output for existing kernels (the byte planes of the int32 matrix
    rewrite bind one) byte-identical. *)

let retype (v : var) = {v with var_type = TInt32}

let narrowed t e =
  match e with ECast (t', _) when t' = t -> e | _ -> ECast (t, e)

(** Every [EVar] of a retyped binder carries the binder's record, so it is
    retyped too (matched by [var_id], which the IR keeps unique per binder). *)
let rec retype_expr ids e =
  let r = retype_expr ids in
  match e with
  | EVar v when List.mem v.var_id ids -> EVar (retype v)
  | EConst _ | EVar _ | EArrayLen _ -> e
  | EArrayRead (a, i) -> EArrayRead (a, r i)
  | EArrayReadExpr (b, i) -> EArrayReadExpr (r b, r i)
  | EBinop (op, a, b) -> EBinop (op, r a, r b)
  | EUnop (op, a) -> EUnop (op, r a)
  | ERecordField (a, f) -> ERecordField (r a, f)
  | EIntrinsic (p, n, args) -> EIntrinsic (p, n, List.map r args)
  | ECast (t, a) -> ECast (t, r a)
  | ETuple es -> ETuple (List.map r es)
  | EApp (f, args) -> EApp (r f, List.map r args)
  | ERecord (n, fs) -> ERecord (n, List.map (fun (f, a) -> (f, r a)) fs)
  | EVariant (n, c, args) -> EVariant (n, c, List.map r args)
  | EArrayCreate (t, sz, ms) -> EArrayCreate (t, r sz, ms)
  | EIf (c, a, b) -> EIf (r c, r a, r b)
  | EMatch (s, cases) -> EMatch (r s, List.map (fun (p, a) -> (p, r a)) cases)

let rec retype_lvalue ids = function
  | LVar v when List.mem v.var_id ids -> LVar (retype v)
  | LVar _ as lv -> lv
  | LArrayElem (a, i) -> LArrayElem (a, retype_expr ids i)
  | LArrayElemExpr (b, i) ->
      LArrayElemExpr (retype_expr ids b, retype_expr ids i)
  | LRecordField (b, f) -> LRecordField (retype_lvalue ids b, f)

(** The kernel's 8-bit local scalars whose type satisfies [keep], by id, so that
    a use anywhere in their scope is retyped with them. *)
let rec local_scalars keep acc = function
  | SLet (v, _, b) | SLetMut (v, _, b) ->
      local_scalars
        keep
        (if keep v.var_type then (v.var_id, v.var_type) :: acc else acc)
        b
  | SSeq ss -> List.fold_left (local_scalars keep) acc ss
  | SIf (_, t, e) ->
      let acc = local_scalars keep acc t in
      Option.fold ~none:acc ~some:(local_scalars keep acc) e
  | SFor (_, _, _, _, b) | SWhile (_, b) | SPragma (_, b) | SBlock b ->
      local_scalars keep acc b
  | SMatch (_, cases) ->
      List.fold_left (fun a (_, s) -> local_scalars keep a s) acc cases
  | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty
  | SMemFence | SNative _ | SCoopmat _ ->
      acc

(** [types] maps a lowered binder's id to its 8-bit type. *)
let rec lower_stmt types s =
  let ids = List.map fst types in
  let e = retype_expr ids and r = lower_stmt types in
  let narrow_of (v : var) = List.assoc_opt v.var_id types in
  let bind v x =
    match narrow_of v with
    | Some t -> (retype v, narrowed t (e x))
    | None -> (v, e x)
  in
  match s with
  | SBarrier | SWarpBarrier | SEmpty | SMemFence | SNative _ -> s
  | SAssign ((LVar v as lv), x) -> (
      match narrow_of v with
      | Some t -> SAssign (retype_lvalue ids lv, narrowed t (e x))
      | None -> SAssign (retype_lvalue ids lv, e x))
  | SAssign (lv, x) -> SAssign (retype_lvalue ids lv, e x)
  | SLet (v, x, b) ->
      let v, x = bind v x in
      SLet (v, x, r b)
  | SLetMut (v, x, b) ->
      let v, x = bind v x in
      SLetMut (v, x, r b)
  | SSeq ss -> SSeq (List.map r ss)
  | SIf (c, t, el) -> SIf (e c, r t, Option.map r el)
  | SWhile (c, b) -> SWhile (e c, r b)
  | SFor (v, lo, hi, d, b) -> SFor (v, e lo, e hi, d, r b)
  | SMatch (sc, cases) -> SMatch (e sc, List.map (fun (p, b) -> (p, r b)) cases)
  | SReturn x -> SReturn (e x)
  | SExpr x -> SExpr (e x)
  | SPragma (h, b) -> SPragma (h, r b)
  | SBlock b -> SBlock (r b)
  | SCoopmat op ->
      SCoopmat
        (match op with
        | CM_decl _ | CM_muladd _ -> op
        | CM_load c -> CM_load {c with index = e c.index; stride = e c.stride}
        | CM_store c -> CM_store {c with index = e c.index; stride = e c.stride})

let lower_body keep stmt locals =
  let decls =
    List.filter_map
      (function
        | DLocal (v, _) when keep v.var_type -> Some (v.var_id, v.var_type)
        | _ -> None)
      locals
  in
  let types = local_scalars keep decls stmt in
  (types, lower_stmt types stmt)

let lower_scalars_where keep (k : kernel) : kernel =
  let types, body = lower_body keep k.kern_body k.kern_locals in
  let ids = List.map fst types in
  {
    k with
    kern_body = body;
    kern_locals =
      List.map
        (function
          | DLocal (v, init) when keep v.var_type ->
              let t = v.var_type in
              DLocal
                ( retype v,
                  Option.map (fun x -> narrowed t (retype_expr ids x)) init )
          | d -> d)
        k.kern_locals;
    kern_funcs =
      List.map
        (fun hf -> {hf with hf_body = snd (lower_body keep hf.hf_body [])})
        k.kern_funcs;
  }

let lower_scalars k =
  if Sarek_ir_analysis.kernel_uses Sarek_ir_analysis.Int8 k then
    lower_scalars_where is_8bit k
  else k

(** {1 Signed narrowing on the C-family targets}

    C (before C23 for this case), C++ before C++20, OpenCL C and MSL leave the
    conversion of an out-of-range [int] to [signed char] implementation-defined.
    The conversion to [unsigned char] is defined (modulo 2^8). So the C-family
    emitters must never convert an int that may be out of range to a signed
    8-bit type: {!define_signed_narrowing} makes every such conversion an
    explicit [ECast (TInt8, _)], which they spell as the int expression
    [((e & 0xFF) ^ 0x80) - 0x80]; its value is in [-128, 127], so the store that
    follows converts an in-range value. *)

let is_int8 = function TInt8 -> true | _ -> false

let define_signed_narrowing (k : kernel) : kernel =
  if not (Sarek_ir_analysis.kernel_uses Sarek_ir_analysis.Int8 k) then k
  else
    let k = lower_scalars_where is_int8 k in
    let signed_arrays = List.filter (fun (_, t) -> is_int8 t) (byte_arrays k) in
    if signed_arrays = [] then k
    else
      let rec wrap s =
        match s with
        | SAssign (lv, x) -> (
            match store_elt signed_arrays lv with
            | Some t -> SAssign (lv, narrowed t x)
            | None -> s)
        | SSeq ss -> SSeq (List.map wrap ss)
        | SIf (c, t, el) -> SIf (c, wrap t, Option.map wrap el)
        | SWhile (c, b) -> SWhile (c, wrap b)
        | SFor (v, lo, hi, d, b) -> SFor (v, lo, hi, d, wrap b)
        | SMatch (sc, cases) ->
            SMatch (sc, List.map (fun (p, b) -> (p, wrap b)) cases)
        | SLet (v, x, b) -> SLet (v, x, wrap b)
        | SLetMut (v, x, b) -> SLetMut (v, x, wrap b)
        | SPragma (h, b) -> SPragma (h, wrap b)
        | SBlock b -> SBlock (wrap b)
        | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty | SMemFence
        | SNative _ | SCoopmat _ ->
            s
      in
      {
        k with
        kern_body = wrap k.kern_body;
        kern_funcs =
          List.map (fun hf -> {hf with hf_body = wrap hf.hf_body}) k.kern_funcs;
      }

(** The C spelling of [ECast (TInt8, e)] given [e]'s spelling: standard-defined
    on every C dialect, an [int] in [-128, 127]. Shared by the CUDA/HIP, OpenCL
    and Metal emitters. *)
let c_signed_narrowing operand =
  Printf.sprintf "((((%s) & 0xFF) ^ 0x80) - 0x80)" operand

let normalise (k : kernel) : kernel =
  let k = lower_scalars k in
  match byte_arrays k with
  | [] -> k
  | bytes ->
      {
        k with
        kern_body = map_stmt bytes k.kern_body;
        kern_funcs =
          List.map
            (fun hf -> {hf with hf_body = map_stmt bytes hf.hf_body})
            k.kern_funcs;
      }
