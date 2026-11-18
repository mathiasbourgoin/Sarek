(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Index lets: immutable lets of an integer whose every use is index
   arithmetic (Sarek_ir_c_arith, "Index arithmetic"). One traversal records the
   uses, then a worklist propagates exclusions: linear in the kernel size. *)

open Sarek_ir_types
module T = Sarek_ir_c_types

(* ---- candidates: names bound exactly once, by an immutable integer let ---- *)

let bind_patterns bind cases =
  List.iter
    (function PConstr (_, names), _ -> List.iter bind names | PWild, _ -> ())
    cases

(* A fold that calls [bind] on every binder and [int_let] on every immutable
   integer let. *)
let binder_folder ~bind ~int_let =
  let on_stmt () = function
    | SLet (v, EArrayCreate _, _) -> bind v.var_name
    | SLet (v, _, _) ->
        bind v.var_name ;
        if T.width_of_type v.var_type <> None then int_let v.var_name
    | SLetMut (v, _, _) | SFor (v, _, _, _, _) -> bind v.var_name
    | SMatch (_, cases) -> bind_patterns bind cases
    | _ -> ()
  in
  let on_expr () = function
    | EMatch (_, cases) -> bind_patterns bind cases
    | _ -> ()
  in
  {
    Sarek_ir_analysis.fe = on_expr;
    ft = (fun () _ -> ());
    fs = on_stmt;
    fnative = (fun () -> ());
    visit_lvalue = false;
  }

let candidates bodies params =
  let count = Hashtbl.create 32 in
  let lets = Hashtbl.create 16 in
  let bind name =
    Hashtbl.replace
      count
      name
      (1 + Option.value ~default:0 (Hashtbl.find_opt count name))
  in
  List.iter bind params ;
  let folder =
    binder_folder ~bind ~int_let:(fun name -> Hashtbl.replace lets name ())
  in
  List.iter (Sarek_ir_analysis.stmt_fold folder ()) bodies ;
  Hashtbl.filter_map_inplace
    (fun name () -> if Hashtbl.find count name = 1 then Some () else None)
    lets ;
  lets

(* ---- uses ---- *)

(* Where a variable occurrence sits: index arithmetic, the root chain of the
   definition of candidate [Def l] (index arithmetic while [l] stays an index
   let), or anywhere else. *)
type ctx = Index | Def of string | Other

type uses = {
  other : (string, unit) Hashtbl.t;  (** names with a use outside any index *)
  deps : (string, string) Hashtbl.t;
      (** [l -> v]: [v] is used in the root chain of [l]'s definition *)
}

(* [e] with its annotation [a]; [ctx] applies while inside the root chain. *)
let rec use_expr env u ctx e (a : T.annot) =
  let sub ctx i x = use_expr env u ctx x a.kids.(i) in
  match e with
  | EVar v -> (
      match ctx with
      | Index -> ()
      | Def l -> Hashtbl.add u.deps l v.var_name
      | Other -> Hashtbl.replace u.other v.var_name ())
  | EBinop (((Add | Sub | Mul) as op), x, y)
    when ctx <> Other && T.binop_width_a op a.kids.(0) a.kids.(1) <> None ->
      sub ctx 0 x ;
      sub ctx 1 y
  | EBinop (Shl, x, y)
    when ctx <> Other && T.binop_width_a Shl a.kids.(0) a.kids.(1) <> None ->
      sub ctx 0 x ;
      sub Other 1 y
  | EArrayRead (_, i) -> sub Index 0 i
  | EArrayReadExpr (base, i) ->
      sub Other 0 base ;
      sub Index 1 i
  | _ -> List.iteri (fun i x -> sub Other i x) (T.children e)

let use_root env u ctx e = use_expr env u ctx e (T.annotate env e)

let rec use_lvalue env u = function
  | LVar _ -> ()
  | LArrayElem (_, i) -> use_root env u Index i
  | LArrayElemExpr (base, i) ->
      use_root env u Other base ;
      use_root env u Index i
  | LRecordField (lv, _) -> use_lvalue env u lv

let use_coopmat env u = function
  | CM_load {index; stride; _} | CM_store {index; stride; _} ->
      use_root env u Index index ;
      use_root env u Index stride
  | CM_scale_add _ as op ->
      Sarek_coopmat_scale.fold_exprs (fun () e -> use_root env u Index e) () op
  | CM_decl _ | CM_muladd _ -> ()

let rec use_stmt env lets u = function
  | SAssign (lv, e) ->
      use_lvalue env u lv ;
      use_root env u Other e
  | SSeq l -> List.iter (use_stmt env lets u) l
  | SIf (c, t, f) ->
      use_root env u Other c ;
      use_stmt env lets u t ;
      Option.iter (use_stmt env lets u) f
  | SWhile (c, b) ->
      use_root env u Other c ;
      use_stmt env lets u b
  | SFor (_, lo, hi, _, b) ->
      use_root env u Index lo ;
      use_root env u Index hi ;
      use_stmt env lets u b
  | SMatch (e, cases) ->
      use_root env u Other e ;
      List.iter (fun (_, s) -> use_stmt env lets u s) cases
  | SReturn e | SExpr e -> use_root env u Other e
  | SLet (v, e, b) ->
      let ctx = if Hashtbl.mem lets v.var_name then Def v.var_name else Other in
      use_root env u ctx e ;
      use_stmt env lets u b
  | SLetMut (_, e, b) ->
      use_root env u Other e ;
      use_stmt env lets u b
  | SPragma (_, b) | SBlock b -> use_stmt env lets u b
  | SCoopmat op -> use_coopmat env u op
  | SBarrier | SWarpBarrier | SEmpty | SMemFence | SNative _ -> ()

(* ---- the fixpoint ---- *)

(* [index_steps] counts worklist pops and dependency visits. Drop every
   candidate with an [Other] use; dropping [l] turns the uses in
   its definition into [Other] uses, so its dependencies are dropped in turn.
   Each candidate enters the worklist at most once. *)
let propagate (env : T.env) lets u =
  let work = Queue.create () in
  let drop name =
    incr env.T.index_steps ;
    if Hashtbl.mem lets name then begin
      Hashtbl.remove lets name ;
      Queue.add name work
    end
  in
  Hashtbl.iter (fun name () -> drop name) u.other ;
  while not (Queue.is_empty work) do
    incr env.T.index_steps ;
    List.iter drop (Hashtbl.find_all u.deps (Queue.pop work))
  done

(* [scopes]: each function body with the environment typing it; [extra]:
   expressions outside any body (local initialisers), all [Other] uses. *)
let compute (env : T.env) ~scopes ~extra ~params =
  let lets = candidates (List.map fst scopes) params in
  let u = {other = Hashtbl.create 32; deps = Hashtbl.create 32} in
  List.iter (use_root env u Other) extra ;
  List.iter (fun (body, scope_env) -> use_stmt scope_env lets u body) scopes ;
  propagate env lets u ;
  Hashtbl.iter (fun name () -> Hashtbl.replace env.T.index_lets name ()) lets
