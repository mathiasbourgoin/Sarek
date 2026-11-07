(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Unrolling of small innermost runtime-trip loops for the PTX generator. See
   the .mli for the rule and why the iteration order is preserved. *)

open Sarek_ir_types
module A = Sarek_ir_analysis

let target_loads_in_flight = 32

let max_factor = 16

(* Size bounds on the IR of a body, in nodes (statements and expressions): a
   body above [max_body_nodes] is not small, and [factor * body] stays under
   [max_expanded_nodes] so the code growth is bounded. *)
let max_body_nodes = 96

let max_expanded_nodes = 768

let enabled () = not (Sarek_policy.restricted Sarek_policy.Auto_unroll)

(** {1 Constants} *)

let rec is_constant = function
  | EConst _ -> true
  | ECast (_, e) | EUnop (_, e) -> is_constant e
  | EBinop (_, a, b) -> is_constant a && is_constant b
  | _ -> false

(** {1 Scans over a loop body} *)

let folder ~fe ~fs ~native : 'a A.folder =
  {
    A.fe;
    ft = (fun acc _ -> acc);
    fs;
    fnative = (fun acc -> native acc);
    visit_lvalue = true;
  }

let count_nodes (body : stmt) =
  A.stmt_fold
    (folder
       ~fe:(fun n _ -> n + 1)
       ~fs:(fun n _ -> n + 1)
       ~native:(fun n -> n + 1))
    0
    body

let barrier_intrinsics = ["block_barrier"; "warp_barrier"; "memory_fence"]

(* What a copy of the body must not contain: constructs whose duplication would
   change what is executed (a return, a barrier, a warp operation), declare
   storage (an array), hide control flow or a loop (a helper call, a nested
   loop, a match), or that is opaque (native code). *)
let blocks_unrolling_leaf = function
  | EIntrinsic (_, name, _) -> List.mem name barrier_intrinsics
  | EApp _ | EArrayCreate _ -> true
  | _ -> false

let blocks_unrolling_stmt = function
  | SWhile _ | SFor _ | SReturn _ | SBarrier | SWarpBarrier | SMemFence
  | SMatch _ | SCoopmat _ ->
      true
  | _ -> false

let duplicable (body : stmt) =
  let blocked =
    A.stmt_fold
      (A.exists_folder
         ~leaf:blocks_unrolling_leaf
         ~stmt_leaf:blocks_unrolling_stmt
         ~native:true
         ())
      false
      body
  in
  (not blocked) && not (A.stmt_uses A.Warp_shuffle body)

(* Number of global array reads in [body]: reads of arrays that are not
   declared shared or local. The unroll factor aims at keeping about
   [target_loads_in_flight] of them in flight. *)
let count_global_loads ~non_global (body : stmt) =
  let fe n = function
    | EArrayRead (a, _) when not (List.mem a non_global) -> n + 1
    | EArrayReadExpr (EVar v, _) when not (List.mem v.var_name non_global) ->
        n + 1
    | EArrayReadExpr _ -> n + 1
    | _ -> n
  in
  A.stmt_fold (folder ~fe ~fs:(fun n _ -> n) ~native:(fun n -> n)) 0 body

let rec lvalue_root = function
  | LVar v -> Some v
  | LRecordField (inner, _) -> lvalue_root inner
  | LArrayElem _ | LArrayElemExpr _ -> None

(* Variables assigned or (re)bound in [body]: one entry per assignment and per
   binder, by id and name so a shadowing binder is caught either way. *)
let assigned_in (body : stmt) : var list =
  let fs acc = function
    | SAssign (lv, _) -> (
        match lvalue_root lv with Some v -> v :: acc | None -> acc)
    | SLet (v, _, _) | SLetMut (v, _, _) | SFor (v, _, _, _, _) -> v :: acc
    | _ -> acc
  in
  A.stmt_fold (folder ~fe:(fun acc _ -> acc) ~fs ~native:(fun a -> a)) [] body

let same_var (a : var) (b : var) =
  a.var_id = b.var_id || String.equal a.var_name b.var_name

(* A loop-invariant bound: built from constants and variables that the loop
   body never assigns, so it can be evaluated once before the loop. *)
let rec invariant ~assigned = function
  | EConst _ | EArrayLen _ -> true
  | EVar v -> not (List.exists (same_var v) assigned)
  | ECast (_, e) | EUnop (_, e) -> invariant ~assigned e
  | EBinop (_, a, b) -> invariant ~assigned a && invariant ~assigned b
  | _ -> false

(* The constant [c] of the last statement [k := k + c], looking through
   sequences, blocks and let scopes, none of which can skip it. *)
let rec trailing_increment (k : var) (s : stmt) : int32 option =
  match s with
  | SSeq stmts -> (
      match
        List.rev (List.filter (function SEmpty -> false | _ -> true) stmts)
      with
      | last :: _ -> trailing_increment k last
      | [] -> None)
  | SBlock s | SLet (_, _, s) | SLetMut (_, _, s) -> trailing_increment k s
  | SAssign
      ( LVar v,
        ( EBinop (Add, EVar v', EConst (CInt32 c))
        | EBinop (Add, EConst (CInt32 c), EVar v') ) )
    when same_var v k && same_var v' k && Int32.compare c 0l > 0 ->
      Some c
  | _ -> None

(** {1 Kernel context} *)

type ctx = {
  non_global : string list;  (** shared and local arrays: not global loads *)
  mutable next_id : int;  (** last variable id handed out *)
  mutable fresh : int;  (** counter of generated names *)
  mutable rewritten : int;  (** loops unrolled *)
}

let fresh_var ctx base ty =
  ctx.next_id <- ctx.next_id + 1 ;
  ctx.fresh <- ctx.fresh + 1 ;
  {
    var_name = Printf.sprintf "__sarek_unroll_%s%d" base ctx.fresh;
    var_id = ctx.next_id;
    var_type = ty;
    var_mutable = false;
  }

let max_var_id (k : kernel) =
  let note acc (v : var) = max acc v.var_id in
  let fe acc = function EVar v -> note acc v | _ -> acc in
  let fs acc = function
    | SLet (v, _, _) | SLetMut (v, _, _) | SFor (v, _, _, _, _) -> note acc v
    | _ -> acc
  in
  let f = folder ~fe ~fs ~native:(fun a -> a) in
  let acc =
    List.fold_left
      (fun acc -> function
        | DParam (v, _) | DLocal (v, _) -> note acc v | DShared _ -> acc)
      0
      (k.kern_params @ k.kern_locals)
  in
  let acc = A.stmt_fold f acc k.kern_body in
  List.fold_left
    (fun acc hf ->
      let acc = List.fold_left note acc hf.hf_params in
      A.stmt_fold f acc hf.hf_body)
    acc
    k.kern_funcs

let non_global_arrays (k : kernel) =
  let fs acc = function
    | SLet (v, EArrayCreate _, _) | SLetMut (v, EArrayCreate _, _) ->
        v.var_name :: acc
    | _ -> acc
  in
  let from_body =
    A.stmt_fold
      (folder ~fe:(fun acc _ -> acc) ~fs ~native:(fun a -> a))
      []
      k.kern_body
  in
  List.fold_left
    (fun acc -> function DShared (name, _, _) -> name :: acc | _ -> acc)
    from_body
    k.kern_locals

(** {1 The rewrite} *)

(* Unroll factor of a body, [None] when it would not reach 2. *)
let factor ~loads ~nodes =
  let by_loads = target_loads_in_flight / max 1 loads in
  let by_size = max_expanded_nodes / max 1 nodes in
  let u = min max_factor (min by_loads by_size) in
  if u >= 2 then Some u else None

let qualifying_body ctx body =
  let nodes = count_nodes body in
  let loads = count_global_loads ~non_global:ctx.non_global body in
  if nodes <= max_body_nodes && loads >= 1 && duplicable body then
    factor ~loads ~nodes
  else None

let i64 e = ECast (TInt64, e)

let i64_const n = EConst (CInt64 (Int64.of_int n))

let copies u body = SSeq (List.init u (fun _ -> body))

(* [while k < bound do body done], [body] ending in [k := k + c]. The main loop
   runs while [k + (u - 1) * c < bound], i.e. [k < bound - (u - 1) * c] in 64
   bits: every one of its [u] iterations would have passed the original test. *)
let unroll_while ctx cond body =
  match cond with
  | EBinop (((Lt | Le) as op), EVar k, bound) when k.var_type = TInt32 -> (
      let assigned = assigned_in body in
      let k_assignments = List.length (List.filter (same_var k) assigned) in
      match trailing_increment k body with
      | Some c
        when k_assignments = 1
             && (not (is_constant bound))
             && invariant ~assigned:(k :: assigned) bound -> (
          match qualifying_body ctx body with
          | None -> None
          | Some u ->
              let lim = fresh_var ctx "lim" TInt64 in
              let delta = (u - 1) * Int32.to_int c in
              ctx.rewritten <- ctx.rewritten + 1 ;
              Some
                (SLet
                   ( lim,
                     EBinop (Sub, i64 bound, i64_const delta),
                     SSeq
                       [
                         SWhile
                           (EBinop (op, i64 (EVar k), EVar lim), copies u body);
                         SWhile (cond, body);
                       ] )))
      | _ -> None)
  | _ -> None

(* [for v = start to stop do body done] (or [downto]): the bounds are evaluated
   once, in the original order and before [v] is bound, then [v] becomes a
   mutable counter shared by the main loop and the remainder loop. *)
let unroll_for ctx (v : var) start stop dir body =
  let assigned = assigned_in body in
  if
    v.var_type <> TInt32
    || (is_constant start && is_constant stop)
    || List.exists (same_var v) assigned
  then None
  else
    match qualifying_body ctx body with
    | None -> None
    | Some u ->
        let start_v = fresh_var ctx "start" TInt32 in
        let stop_v = fresh_var ctx "stop" TInt32 in
        let lim = fresh_var ctx "lim" TInt64 in
        let counter = {v with var_mutable = true} in
        let step_op, main_cmp, tail_cmp, delta =
          match dir with
          | Upto -> (Add, Le, Le, -(u - 1))
          | Downto -> (Sub, Ge, Ge, u - 1)
        in
        let step =
          SAssign
            (LVar counter, EBinop (step_op, EVar counter, EConst (CInt32 1l)))
        in
        let iteration = SSeq [body; step] in
        ctx.rewritten <- ctx.rewritten + 1 ;
        Some
          (SLet
             ( start_v,
               start,
               SLet
                 ( stop_v,
                   stop,
                   SLet
                     ( lim,
                       EBinop (Add, i64 (EVar stop_v), i64_const delta),
                       SLetMut
                         ( counter,
                           EVar start_v,
                           SSeq
                             [
                               SWhile
                                 ( EBinop
                                     (main_cmp, i64 (EVar counter), EVar lim),
                                   copies u iteration );
                               SWhile
                                 ( EBinop (tail_cmp, EVar counter, EVar stop_v),
                                   iteration );
                             ] ) ) ) ))

let is_sarek_directive = function
  | h :: _ -> String.length h >= 6 && String.sub h 0 6 = "sarek."
  | [] -> false

(* Rewrites innermost loops, bottom-up. A loop under an explicit pragma keeps
   its own shape; loops inside it are still considered. *)
let rec rewrite_stmt ctx (s : stmt) : stmt =
  let go = rewrite_stmt ctx in
  match s with
  | SWhile (cond, body) -> (
      match unroll_while ctx cond body with
      | Some unrolled -> unrolled
      | None -> SWhile (cond, go body))
  | SFor (v, start, stop, dir, body) -> (
      match unroll_for ctx v start stop dir body with
      | Some unrolled -> unrolled
      | None -> SFor (v, start, stop, dir, go body))
  | SPragma (h, ((SWhile _ | SFor _) as loop)) when not (is_sarek_directive h)
    -> (
      match loop with
      | SWhile (cond, body) -> SPragma (h, SWhile (cond, go body))
      | SFor (v, a, b, d, body) -> SPragma (h, SFor (v, a, b, d, go body))
      | _ -> s)
  | SPragma (h, body) -> SPragma (h, go body)
  | SSeq stmts -> SSeq (List.map go stmts)
  | SIf (c, t, e) -> SIf (c, go t, Option.map go e)
  | SMatch (e, cases) -> SMatch (e, List.map (fun (p, b) -> (p, go b)) cases)
  | SLet (v, e, body) -> SLet (v, e, go body)
  | SLetMut (v, e, body) -> SLetMut (v, e, go body)
  | SBlock body -> SBlock (go body)
  | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty
  | SMemFence | SNative _ | SCoopmat _ ->
      s

let apply (k : kernel) : kernel * int =
  let ctx =
    {
      non_global = non_global_arrays k;
      next_id = max_var_id k;
      fresh = 0;
      rewritten = 0;
    }
  in
  let k' =
    {
      k with
      kern_body = rewrite_stmt ctx k.kern_body;
      kern_funcs =
        List.map
          (fun hf -> {hf with hf_body = rewrite_stmt ctx hf.hf_body})
          k.kern_funcs;
    }
  in
  (k', ctx.rewritten)

let report (k : kernel) =
  Sarek_policy.report
    {
      kernel = k.kern_name;
      target = "gen:ptx";
      optimisation = Sarek_policy.Auto_unroll;
      decision =
        Sarek_policy.decide
          (Sarek_policy.effective ~kernel:k.kern_name)
          Sarek_policy.Auto_unroll
          ~backend:"ptx";
    }

let rewrite (k : kernel) : kernel =
  if (not (enabled ())) && not (Sarek_policy.reporting ()) then k
  else
    let unrolled, loops = apply k in
    if loops > 0 then report k ;
    if enabled () then unrolled else k
