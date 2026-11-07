(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Renaming of a kernel's names, shared by the fusion rewrites
    (specs/launch-window.md, specs/epilogue-fusion.md).

    Every variable, array and pattern name goes through [name], every var id
    through [id], and every argument-less [Sarek_stdlib.Gpu] intrinsic through
    [position] (which may replace it). Helper functions are called by their own
    names, which are left alone. Record fields and constructors are not names of
    the kernel and are left alone too. *)

open Sarek_ir_types

type ctx = {
  name : string -> string;
  id : int -> int;
  position : string -> expr option;
}

let var ctx (v : var) =
  {v with var_name = ctx.name v.var_name; var_id = ctx.id v.var_id}

let rec expr ctx e =
  let go = expr ctx and n = ctx.name in
  match e with
  | EConst _ -> e
  | EVar v -> EVar (var ctx v)
  | EBinop (op, a, b) -> EBinop (op, go a, go b)
  | EUnop (op, a) -> EUnop (op, go a)
  | EArrayRead (a, i) -> EArrayRead (n a, go i)
  | EArrayReadExpr (a, i) -> EArrayReadExpr (go a, go i)
  | ERecordField (a, f) -> ERecordField (go a, f)
  | EIntrinsic ((["Sarek_stdlib"; "Gpu"] as p), name, args) -> (
      match ctx.position name with
      | Some r when args = [] -> r
      | _ -> EIntrinsic (p, name, List.map go args))
  | EIntrinsic (p, name, args) -> EIntrinsic (p, name, List.map go args)
  | ECast (t, a) -> ECast (t, go a)
  | ETuple es -> ETuple (List.map go es)
  | EApp ((EVar _ as helper), es) ->
      (* A helper function, called by its own (shared, unrenamed) name. *)
      EApp (helper, List.map go es)
  | EApp (f, es) -> EApp (go f, List.map go es)
  | ERecord (t, fs) -> ERecord (t, List.map (fun (f, x) -> (f, go x)) fs)
  | EVariant (t, c, es) -> EVariant (t, c, List.map go es)
  | EArrayLen a -> EArrayLen (n a)
  | EArrayCreate (t, s, m) -> EArrayCreate (t, go s, m)
  | EIf (c, a, b) -> EIf (go c, go a, go b)
  | EMatch (s, cs) ->
      EMatch (go s, List.map (fun (p, x) -> (pattern ctx p, go x)) cs)

and pattern ctx = function
  | PConstr (c, vs) -> PConstr (c, List.map ctx.name vs)
  | PWild -> PWild

let rec lvalue ctx = function
  | LVar v -> LVar (var ctx v)
  | LArrayElem (a, i) -> LArrayElem (ctx.name a, expr ctx i)
  | LArrayElemExpr (a, i) -> LArrayElemExpr (expr ctx a, expr ctx i)
  | LRecordField (lv, f) -> LRecordField (lvalue ctx lv, f)

let rec stmt ctx s =
  let e = expr ctx and st = stmt ctx and v = var ctx in
  match s with
  | SAssign (lv, x) -> SAssign (lvalue ctx lv, e x)
  | SSeq ss -> SSeq (List.map st ss)
  | SIf (c, t, f) -> SIf (e c, st t, Option.map st f)
  | SWhile (c, b) -> SWhile (e c, st b)
  | SFor (i, lo, hi, d, b) -> SFor (v i, e lo, e hi, d, st b)
  | SMatch (x, cs) ->
      SMatch (e x, List.map (fun (p, b) -> (pattern ctx p, st b)) cs)
  | SReturn x -> SReturn (e x)
  | SExpr x -> SExpr (e x)
  | SLet (x, y, b) -> SLet (v x, e y, st b)
  | SLetMut (x, y, b) -> SLetMut (v x, e y, st b)
  | SPragma (p, b) -> SPragma (p, st b)
  | SBlock b -> SBlock (st b)
  | SBarrier | SWarpBarrier | SMemFence | SEmpty | SNative _ | SCoopmat _ -> s

let decl ctx = function
  | DParam (v, info) -> DParam (var ctx v, info)
  | DLocal (v, init) -> DLocal (var ctx v, Option.map (expr ctx) init)
  | DShared (name, t, size) ->
      DShared (ctx.name name, t, Option.map (expr ctx) size)

(** {1 Variable ids} *)

(* The largest var id a kernel uses (params, locals, body, helpers), so a
   rewrite can place new or shifted ids above it. The interpreter and
   [Index_slice] key variables by id. *)
let rec max_id_expr m = function
  | EVar v -> max m v.var_id
  | EConst _ | EArrayLen _ -> m
  | EBinop (_, a, b) | EArrayReadExpr (a, b) -> max_id_expr (max_id_expr m a) b
  | EUnop (_, a) | ECast (_, a) | ERecordField (a, _) | EArrayRead (_, a) ->
      max_id_expr m a
  | EIntrinsic (_, _, es) | ETuple es | EVariant (_, _, es) ->
      List.fold_left max_id_expr m es
  | EApp (f, es) -> List.fold_left max_id_expr m (f :: es)
  | ERecord (_, fs) -> List.fold_left (fun m (_, e) -> max_id_expr m e) m fs
  | EArrayCreate (_, s, _) -> max_id_expr m s
  | EIf (a, b, c) -> List.fold_left max_id_expr m [a; b; c]
  | EMatch (s, cs) ->
      List.fold_left (fun m (_, e) -> max_id_expr m e) (max_id_expr m s) cs

let rec max_id_lvalue m = function
  | LVar v -> max m v.var_id
  | LArrayElem (_, i) -> max_id_expr m i
  | LArrayElemExpr (a, i) -> max_id_expr (max_id_expr m a) i
  | LRecordField (lv, _) -> max_id_lvalue m lv

let rec max_id_stmt m = function
  | SAssign (lv, e) -> max_id_expr (max_id_lvalue m lv) e
  | SSeq ss -> List.fold_left max_id_stmt m ss
  | SIf (c, t, f) ->
      let m = max_id_stmt (max_id_expr m c) t in
      Option.fold ~none:m ~some:(max_id_stmt m) f
  | SWhile (c, b) -> max_id_stmt (max_id_expr m c) b
  | SFor (v, lo, hi, _, b) ->
      max_id_stmt (max_id_expr (max_id_expr (max m v.var_id) lo) hi) b
  | SMatch (e, cs) ->
      List.fold_left (fun m (_, b) -> max_id_stmt m b) (max_id_expr m e) cs
  | SReturn e | SExpr e -> max_id_expr m e
  | SLet (v, e, b) | SLetMut (v, e, b) ->
      max_id_stmt (max_id_expr (max m v.var_id) e) b
  | SPragma (_, b) | SBlock b -> max_id_stmt m b
  | SCoopmat (CM_load {index; stride; _})
  | SCoopmat (CM_store {index; stride; _}) ->
      max_id_expr (max_id_expr m index) stride
  | SCoopmat (CM_scale_add _ as op) ->
      Sarek_coopmat_scale.fold_exprs max_id_expr m op
  | SCoopmat _ | SBarrier | SWarpBarrier | SMemFence | SEmpty | SNative _ -> m

let max_var_id (k : kernel) : int =
  let decl m = function
    | DParam (v, _) -> max m v.var_id
    | DLocal (v, e) ->
        Option.fold
          ~none:(max m v.var_id)
          ~some:(max_id_expr (max m v.var_id))
          e
    | DShared (_, _, e) -> Option.fold ~none:m ~some:(max_id_expr m) e
  in
  let m = List.fold_left decl 0 (k.kern_params @ k.kern_locals) in
  let m = max_id_stmt m k.kern_body in
  List.fold_left
    (fun m hf ->
      max_id_stmt
        (List.fold_left (fun m (v : var) -> max m v.var_id) m hf.hf_params)
        hf.hf_body)
    m
    k.kern_funcs
