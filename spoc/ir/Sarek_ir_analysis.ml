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
