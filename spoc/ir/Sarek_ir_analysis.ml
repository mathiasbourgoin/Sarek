(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_ir_analysis - Analysis functions for GPU kernel IR *)

open Sarek_ir_types

(** Check if an element type is or contains float64 *)
let rec elttype_uses_float64 = function
  | TFloat64 -> true
  | TRecord (_, fields) ->
      List.exists (fun (_, t) -> elttype_uses_float64 t) fields
  | TVariant (_, constrs) ->
      List.exists
        (fun (_, args) -> List.exists elttype_uses_float64 args)
        constrs
  | TArray (elt, _) | TVec elt -> elttype_uses_float64 elt
  | TInt32 | TInt64 | TFloat32 | TBool | TUnit -> false

(** Check if a constant is float64 *)
let const_uses_float64 = function CFloat64 _ -> true | _ -> false

(** Check if an expression uses float64 *)
let rec expr_uses_float64 = function
  | EConst c -> const_uses_float64 c
  | EVar v -> elttype_uses_float64 v.var_type
  | EBinop (_, e1, e2) -> expr_uses_float64 e1 || expr_uses_float64 e2
  | EUnop (_, e) -> expr_uses_float64 e
  | EArrayRead (_, idx) -> expr_uses_float64 idx
  | EArrayReadExpr (base, idx) ->
      expr_uses_float64 base || expr_uses_float64 idx
  | ERecordField (e, _) -> expr_uses_float64 e
  | EIntrinsic (_, _, args) -> List.exists expr_uses_float64 args
  | ECast (ty, e) -> elttype_uses_float64 ty || expr_uses_float64 e
  | ETuple exprs -> List.exists expr_uses_float64 exprs
  | EApp (fn, args) ->
      expr_uses_float64 fn || List.exists expr_uses_float64 args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_float64 e) fields
  | EVariant (_, _, args) -> List.exists expr_uses_float64 args
  | EArrayLen _ -> false
  | EArrayCreate (ty, size, _) ->
      elttype_uses_float64 ty || expr_uses_float64 size
  | EIf (cond, then_, else_) ->
      expr_uses_float64 cond || expr_uses_float64 then_
      || expr_uses_float64 else_
  | EMatch (scrutinee, cases) ->
      expr_uses_float64 scrutinee
      || List.exists (fun (_, e) -> expr_uses_float64 e) cases

(** Check if a statement uses float64 *)
let rec stmt_uses_float64 = function
  | SAssign (_, e) -> expr_uses_float64 e
  | SSeq stmts -> List.exists stmt_uses_float64 stmts
  | SIf (cond, then_, else_) ->
      expr_uses_float64 cond || stmt_uses_float64 then_
      || Option.fold ~none:false ~some:stmt_uses_float64 else_
  | SWhile (cond, body) -> expr_uses_float64 cond || stmt_uses_float64 body
  | SFor (v, lo, hi, _, body) ->
      elttype_uses_float64 v.var_type
      || expr_uses_float64 lo || expr_uses_float64 hi || stmt_uses_float64 body
  | SMatch (scrutinee, cases) ->
      expr_uses_float64 scrutinee
      || List.exists (fun (_, s) -> stmt_uses_float64 s) cases
  | SReturn e | SExpr e -> expr_uses_float64 e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> false
  | SLet (v, e, body) | SLetMut (v, e, body) ->
      elttype_uses_float64 v.var_type
      || expr_uses_float64 e || stmt_uses_float64 body
  | SPragma (_, body) | SBlock body -> stmt_uses_float64 body
  (* Deliberately asymmetric vs. stmt_uses_atomics's SNative arm: native-block float64 usage is a separate, not-yet-decided question - see KB / review notes; do not change this arm as part of the atomics fix. *)
  | SNative _ -> false

(** Check if a declaration uses float64 *)
let decl_uses_float64 = function
  | DParam (v, arr_info) ->
      elttype_uses_float64 v.var_type
      || Option.fold
           ~none:false
           ~some:(fun ai -> elttype_uses_float64 ai.arr_elttype)
           arr_info
  | DLocal (v, init) ->
      elttype_uses_float64 v.var_type
      || Option.fold ~none:false ~some:expr_uses_float64 init
  | DShared (_, ty, size) ->
      elttype_uses_float64 ty
      || Option.fold ~none:false ~some:expr_uses_float64 size

(** Check if a helper function uses float64 *)
let helper_uses_float64 hf =
  elttype_uses_float64 hf.hf_ret_type
  || List.exists (fun v -> elttype_uses_float64 v.var_type) hf.hf_params
  || stmt_uses_float64 hf.hf_body

(** Check if a kernel uses float64 anywhere *)
let kernel_uses_float64 k =
  List.exists decl_uses_float64 k.kern_params
  || List.exists decl_uses_float64 k.kern_locals
  || stmt_uses_float64 k.kern_body
  || List.exists helper_uses_float64 k.kern_funcs
  || List.exists
       (fun (_, fields) ->
         List.exists (fun (_, t) -> elttype_uses_float64 t) fields)
       k.kern_types
  || List.exists
       (fun (_, constrs) ->
         List.exists
           (fun (_, args) -> List.exists elttype_uses_float64 args)
           constrs)
       k.kern_variants

(** {1 Atomic-operation detection}

    Atomic intrinsics have no dedicated IR constructor: the PPX lowers every
    atomic primitive (see the [category = "atomic"] entries registered in
    [sarek/ppx/Sarek_core_primitives.ml], and the [%sarek_intrinsic] atomics in
    [sarek/Sarek_stdlib/Gpu.ml]) to a plain [EIntrinsic (path, name, args)]
    node, e.g. ["atomic_add_int32"], ["atomic_cas_int32"],
    ["atomic_add_global_int32"], ... All such names share the ["atomic_"] prefix
    by registration convention.

    REGISTRATION POINT: this is the single source of truth for recognizing an
    atomic intrinsic from IR. If a future atomic primitive is registered under a
    name that does not start with ["atomic_"], update [is_atomic_intrinsic_name]
    below (and consider exporting the name list from Sarek_core_primitives.ml
    instead of relying on the prefix convention). Do not duplicate this check
    elsewhere. *)
let is_atomic_intrinsic_name name =
  let prefix = "atomic_" in
  String.length name >= String.length prefix
  && String.sub name 0 (String.length prefix) = prefix

(** Check if an expression contains an atomic intrinsic call *)
let rec expr_uses_atomics = function
  | EIntrinsic (_, name, args) ->
      is_atomic_intrinsic_name name || List.exists expr_uses_atomics args
  | EConst _ | EVar _ -> false
  | EBinop (_, e1, e2) -> expr_uses_atomics e1 || expr_uses_atomics e2
  | EUnop (_, e) -> expr_uses_atomics e
  | EArrayRead (_, idx) -> expr_uses_atomics idx
  | EArrayReadExpr (base, idx) ->
      expr_uses_atomics base || expr_uses_atomics idx
  | ERecordField (e, _) -> expr_uses_atomics e
  | ECast (_, e) -> expr_uses_atomics e
  | ETuple exprs -> List.exists expr_uses_atomics exprs
  | EApp (fn, args) ->
      expr_uses_atomics fn || List.exists expr_uses_atomics args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_atomics e) fields
  | EVariant (_, _, args) -> List.exists expr_uses_atomics args
  | EArrayLen _ -> false
  | EArrayCreate (_, size, _) -> expr_uses_atomics size
  | EIf (cond, then_, else_) ->
      expr_uses_atomics cond || expr_uses_atomics then_
      || expr_uses_atomics else_
  | EMatch (scrutinee, cases) ->
      expr_uses_atomics scrutinee
      || List.exists (fun (_, e) -> expr_uses_atomics e) cases

(** Check if an lvalue contains an atomic intrinsic call (in its index/base
    expression). LVar has no sub-expression; LRecordField recurses into the
    inner lvalue. *)
let rec lvalue_uses_atomics = function
  | LVar _ -> false
  | LArrayElem (_, idx) -> expr_uses_atomics idx
  | LArrayElemExpr (base, idx) ->
      expr_uses_atomics base || expr_uses_atomics idx
  | LRecordField (lv, _) -> lvalue_uses_atomics lv

(** Check if a statement contains an atomic intrinsic call *)
let rec stmt_uses_atomics = function
  | SAssign (lv, e) -> lvalue_uses_atomics lv || expr_uses_atomics e
  | SSeq stmts -> List.exists stmt_uses_atomics stmts
  | SIf (cond, then_, else_) ->
      expr_uses_atomics cond || stmt_uses_atomics then_
      || Option.fold ~none:false ~some:stmt_uses_atomics else_
  | SWhile (cond, body) -> expr_uses_atomics cond || stmt_uses_atomics body
  | SFor (_, lo, hi, _, body) ->
      expr_uses_atomics lo || expr_uses_atomics hi || stmt_uses_atomics body
  | SMatch (scrutinee, cases) ->
      expr_uses_atomics scrutinee
      || List.exists (fun (_, s) -> stmt_uses_atomics s) cases
  | SReturn e | SExpr e -> expr_uses_atomics e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> false
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      expr_uses_atomics e || stmt_uses_atomics body
  | SPragma (_, body) | SBlock body -> stmt_uses_atomics body
  | SNative _ ->
      (* Conservative: inline native GPU code is opaque; fusion must not
         assume it is atomic-free. *)
      true

(** Check if a declaration contains an atomic intrinsic call (in its
    initializer/size expression, if any) *)
let decl_uses_atomics = function
  | DParam _ -> false
  | DLocal (_, init) -> Option.fold ~none:false ~some:expr_uses_atomics init
  | DShared (_, _, size) -> Option.fold ~none:false ~some:expr_uses_atomics size

(** Check if a helper function contains an atomic intrinsic call *)
let helper_uses_atomics hf = stmt_uses_atomics hf.hf_body

(** Check if a kernel uses atomic operations anywhere: params/locals
    initializers, body, and helper functions called from the kernel. Helper
    bodies must be walked explicitly — a body-only check would miss atomics
    hidden inside a called helper function. *)
let kernel_uses_atomics k =
  List.exists decl_uses_atomics k.kern_params
  || List.exists decl_uses_atomics k.kern_locals
  || stmt_uses_atomics k.kern_body
  || List.exists helper_uses_atomics k.kern_funcs

(** {1 Integer-remainder detection}

    [EBinop (Mod, _, _)] is always integer remainder — float [mod] is lowered to
    the [fmod]/[mod] intrinsic (an [EIntrinsic]), never to [Ir.Mod]. Backends
    that cannot lower [%] directly (e.g. GLSL, whose [%] is undefined for
    negative operands) use this to decide whether to emit a remainder helper. *)
let rec expr_uses_int_mod = function
  | EBinop (Mod, _, _) -> true
  | EConst _ | EVar _ -> false
  | EBinop (_, e1, e2) -> expr_uses_int_mod e1 || expr_uses_int_mod e2
  | EUnop (_, e) -> expr_uses_int_mod e
  | EArrayRead (_, idx) -> expr_uses_int_mod idx
  | EArrayReadExpr (base, idx) ->
      expr_uses_int_mod base || expr_uses_int_mod idx
  | ERecordField (e, _) -> expr_uses_int_mod e
  | EIntrinsic (_, _, args) -> List.exists expr_uses_int_mod args
  | ECast (_, e) -> expr_uses_int_mod e
  | ETuple exprs -> List.exists expr_uses_int_mod exprs
  | EApp (fn, args) ->
      expr_uses_int_mod fn || List.exists expr_uses_int_mod args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_int_mod e) fields
  | EVariant (_, _, args) -> List.exists expr_uses_int_mod args
  | EArrayLen _ -> false
  | EArrayCreate (_, size, _) -> expr_uses_int_mod size
  | EIf (cond, then_, else_) ->
      expr_uses_int_mod cond || expr_uses_int_mod then_
      || expr_uses_int_mod else_
  | EMatch (scrutinee, cases) ->
      expr_uses_int_mod scrutinee
      || List.exists (fun (_, e) -> expr_uses_int_mod e) cases

let rec lvalue_uses_int_mod = function
  | LVar _ -> false
  | LArrayElem (_, idx) -> expr_uses_int_mod idx
  | LArrayElemExpr (base, idx) ->
      expr_uses_int_mod base || expr_uses_int_mod idx
  (* Recurse into the nested lvalue: its array index may carry a [mod], e.g.
     [arr.(j mod n).field <- v]. A non-recursive arm would miss it and skip
     emitting the [sarek_smod] helper the emitted index references. Mirrors
     [lvalue_uses_atomics]. *)
  | LRecordField (lv, _) -> lvalue_uses_int_mod lv

let rec stmt_uses_int_mod = function
  | SAssign (lv, e) -> lvalue_uses_int_mod lv || expr_uses_int_mod e
  | SSeq stmts -> List.exists stmt_uses_int_mod stmts
  | SIf (cond, then_, else_) ->
      expr_uses_int_mod cond || stmt_uses_int_mod then_
      || Option.fold ~none:false ~some:stmt_uses_int_mod else_
  | SWhile (cond, body) -> expr_uses_int_mod cond || stmt_uses_int_mod body
  | SFor (_, lo, hi, _, body) ->
      expr_uses_int_mod lo || expr_uses_int_mod hi || stmt_uses_int_mod body
  | SMatch (scrutinee, cases) ->
      expr_uses_int_mod scrutinee
      || List.exists (fun (_, s) -> stmt_uses_int_mod s) cases
  | SReturn e | SExpr e -> expr_uses_int_mod e
  | SBarrier | SWarpBarrier | SEmpty | SMemFence -> false
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      expr_uses_int_mod e || stmt_uses_int_mod body
  | SPragma (_, body) | SBlock body -> stmt_uses_int_mod body
  (* Inline native GPU code is opaque text; assume it may contain a remainder
     so a helper it references is still emitted. *)
  | SNative _ -> true

let decl_uses_int_mod = function
  | DParam _ -> false
  | DLocal (_, init) -> Option.fold ~none:false ~some:expr_uses_int_mod init
  | DShared (_, _, size) -> Option.fold ~none:false ~some:expr_uses_int_mod size

let helper_uses_int_mod hf = stmt_uses_int_mod hf.hf_body

(** Check if a kernel uses integer remainder anywhere: locals initializers,
    body, and helper functions. Helper bodies are walked explicitly (a helper
    may use [mod] even when the top-level body does not). *)
let kernel_uses_int_mod k =
  List.exists decl_uses_int_mod k.kern_params
  || List.exists decl_uses_int_mod k.kern_locals
  || stmt_uses_int_mod k.kern_body
  || List.exists helper_uses_int_mod k.kern_funcs
