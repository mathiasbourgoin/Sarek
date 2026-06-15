(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Tail Recursion Analysis
 *
 * This module analyzes recursive functions to determine:
 * - Whether they are tail-recursive
 * - Whether recursive calls occur inside loops
 * - The number of recursive call sites
 * - Whether bounded depth can be detected (currently conservative)
 ******************************************************************************)

open Sarek_typed_ast

(** {1 Recursion Analysis} *)

(** Information about a recursive function *)
type recursion_info = {
  ri_name : string;  (** Function name *)
  ri_is_tail : bool;  (** All recursive calls are in tail position *)
  ri_max_depth : int option;  (** Bounded depth if detectable *)
  ri_call_count : int;  (** Number of recursive call sites *)
  ri_in_loop : bool;  (** Has recursive calls inside loops *)
}

(** Check if an expression is a call to the named function *)
let is_self_call (fname : string) (expr : texpr) : bool =
  match expr.te with TEVar (name, _) -> name = fname | _ -> false

(** Check if a function application is a recursive call *)
let is_recursive_call (fname : string) (expr : texpr) : bool =
  match expr.te with TEApp (fn, _) -> is_self_call fname fn | _ -> false

(** Count recursive calls in an expression *)
let rec count_recursive_calls (fname : string) (expr : texpr) : int =
  match expr.te with
  | TEApp (fn, args) ->
      let self = if is_self_call fname fn then 1 else 0 in
      let in_fn = count_recursive_calls fname fn in
      let in_args =
        List.fold_left (fun acc a -> acc + count_recursive_calls fname a) 0 args
      in
      self + in_fn + in_args
  | TELet (_, _, value, body) | TELetMut (_, _, value, body) ->
      count_recursive_calls fname value + count_recursive_calls fname body
  | TELetShared (_, _, _, _, body) -> count_recursive_calls fname body
  | TEIf (cond, then_, else_opt) -> (
      count_recursive_calls fname cond
      + count_recursive_calls fname then_
      +
      match else_opt with
      | Some e -> count_recursive_calls fname e
      | None -> 0)
  | TEFor (_, _, lo, hi, _, body) ->
      count_recursive_calls fname lo
      + count_recursive_calls fname hi
      + count_recursive_calls fname body
  | TEWhile (cond, body) ->
      count_recursive_calls fname cond + count_recursive_calls fname body
  | TESeq exprs ->
      List.fold_left (fun acc e -> acc + count_recursive_calls fname e) 0 exprs
  | TEBinop (_, a, b) ->
      count_recursive_calls fname a + count_recursive_calls fname b
  | TEUnop (_, e) -> count_recursive_calls fname e
  | TEVecGet (v, i) | TEArrGet (v, i) ->
      count_recursive_calls fname v + count_recursive_calls fname i
  | TEVecSet (v, i, x) | TEArrSet (v, i, x) ->
      count_recursive_calls fname v
      + count_recursive_calls fname i
      + count_recursive_calls fname x
  | TEFieldGet (r, _, _) -> count_recursive_calls fname r
  | TEFieldSet (r, _, _, v) ->
      count_recursive_calls fname r + count_recursive_calls fname v
  | TEMatch (scrut, cases) ->
      count_recursive_calls fname scrut
      + List.fold_left
          (fun acc (_, body) -> acc + count_recursive_calls fname body)
          0
          cases
  | TETuple es ->
      List.fold_left (fun acc e -> acc + count_recursive_calls fname e) 0 es
  | TERecord (_, fields) ->
      List.fold_left
        (fun acc (_, e) -> acc + count_recursive_calls fname e)
        0
        fields
  | TEConstr (_, _, arg) -> (
      match arg with Some e -> count_recursive_calls fname e | None -> 0)
  | TEAssign (_, _, v) -> count_recursive_calls fname v
  | TEReturn e -> count_recursive_calls fname e
  | TESuperstep (_, _, step, cont) ->
      count_recursive_calls fname step + count_recursive_calls fname cont
  | TEPragma (_, body) -> count_recursive_calls fname body
  | TEOpen (_, body) -> count_recursive_calls fname body
  | TECreateArray (size, _, _) -> count_recursive_calls fname size
  | TEIntrinsicFun (_, _, args) ->
      List.fold_left (fun acc a -> acc + count_recursive_calls fname a) 0 args
  | TELetRec (_, _, _params, fn_body, cont) ->
      (* Count calls in function body and continuation *)
      let in_body = count_recursive_calls fname fn_body in
      let in_cont = count_recursive_calls fname cont in
      in_body + in_cont
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      0

(** Check if all recursive calls are in tail position *)
let is_tail_recursive (fname : string) (expr : texpr) : bool =
  (* Helper: check if expression is in tail position *)
  let rec check_tail expr =
    match expr.te with
    | TEApp (fn, args) when is_self_call fname fn ->
        (* Recursive call in tail position - but args must not contain recursive calls! *)
        let in_args =
          List.fold_left
            (fun acc a -> acc + count_recursive_calls fname a)
            0
            args
        in
        in_args = 0
    | TEApp (fn, args) ->
        (* Non-self call: recursive calls in args are NOT tail *)
        let in_fn = count_recursive_calls fname fn in
        let in_args =
          List.fold_left
            (fun acc a -> acc + count_recursive_calls fname a)
            0
            args
        in
        in_fn = 0 && in_args = 0
    | TEIf (cond, then_, else_opt) -> (
        (* Cond is not tail, branches are tail *)
        count_recursive_calls fname cond = 0
        && check_tail then_
        && match else_opt with Some e -> check_tail e | None -> true)
    | TELet (_, _, value, body) | TELetMut (_, _, value, body) ->
        (* Value is not tail, body is tail *)
        count_recursive_calls fname value = 0 && check_tail body
    | TELetShared (_, _, _, _, body) -> check_tail body
    | TESeq exprs -> (
        match List.rev exprs with
        | [] -> true
        | last :: rest ->
            (* Only last expression is in tail position *)
            List.for_all (fun e -> count_recursive_calls fname e = 0) rest
            && check_tail last)
    | TEFor (_, _, lo, hi, _, body) ->
        (* Recursive calls in loops are never tail *)
        count_recursive_calls fname lo = 0
        && count_recursive_calls fname hi = 0
        && count_recursive_calls fname body = 0
    | TEWhile (cond, body) ->
        count_recursive_calls fname cond = 0
        && count_recursive_calls fname body = 0
    | TEMatch (scrut, cases) ->
        count_recursive_calls fname scrut = 0
        && List.for_all (fun (_, body) -> check_tail body) cases
    | TEReturn e -> check_tail e
    | TESuperstep (_, _, step, cont) ->
        count_recursive_calls fname step = 0 && check_tail cont
    | TEPragma (_, body) -> check_tail body
    | TEOpen (_, body) -> check_tail body
    | _ ->
        (* Other expressions: no recursive calls allowed for tail recursion *)
        count_recursive_calls fname expr = 0
  in
  check_tail expr

(** Check if recursive calls occur inside loops *)
let rec has_recursion_in_loops (fname : string) (expr : texpr) : bool =
  match expr.te with
  | TEFor (_, _, lo, hi, _, body) ->
      count_recursive_calls fname body > 0
      || has_recursion_in_loops fname lo
      || has_recursion_in_loops fname hi
      || has_recursion_in_loops fname body
  | TEWhile (cond, body) ->
      count_recursive_calls fname body > 0
      || has_recursion_in_loops fname cond
      || has_recursion_in_loops fname body
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      has_recursion_in_loops fname v || has_recursion_in_loops fname b
  | TELetShared (_, _, _, _, b) -> has_recursion_in_loops fname b
  | TEIf (c, t, e) -> (
      has_recursion_in_loops fname c
      || has_recursion_in_loops fname t
      || match e with Some x -> has_recursion_in_loops fname x | None -> false)
  | TESeq es -> List.exists (has_recursion_in_loops fname) es
  | TEApp (fn, args) ->
      has_recursion_in_loops fname fn
      || List.exists (has_recursion_in_loops fname) args
  | TEBinop (_, a, b) ->
      has_recursion_in_loops fname a || has_recursion_in_loops fname b
  | TEUnop (_, e) -> has_recursion_in_loops fname e
  | TEMatch (s, cases) ->
      has_recursion_in_loops fname s
      || List.exists (fun (_, b) -> has_recursion_in_loops fname b) cases
  | TESuperstep (_, _, step, cont) ->
      has_recursion_in_loops fname step || has_recursion_in_loops fname cont
  | _ -> false

(** Detect bounded recursion depth from the code structure.

    This is a conservative analysis that only returns Some when we can prove the
    recursion is bounded. Currently disabled as the analysis is too imprecise.

    TODO: Implement proper termination analysis that tracks:
    - Which parameter is the "depth" parameter (must increase towards bound)
    - That recursive calls increment that parameter
    - That the bound is reached before max_inline_limit

    For now, we only support tail recursion which is transformed to loops. *)
let detect_bounded_depth (_fname : string) (_body : texpr) : int option =
  (* Disabled - the simple heuristic was too imprecise and caused infinite
     inlining for patterns like fibonacci where n decreases rather than
     a depth parameter that increases. *)
  None

(** Analyze a function for recursion patterns *)
let analyze_recursion (fname : string) (body : texpr) : recursion_info =
  let call_count = count_recursive_calls fname body in
  let is_tail = call_count > 0 && is_tail_recursive fname body in
  let in_loop = has_recursion_in_loops fname body in
  let max_depth =
    if call_count > 0 && not is_tail then detect_bounded_depth fname body
    else None
  in
  {
    ri_name = fname;
    ri_is_tail = is_tail;
    ri_max_depth = max_depth;
    ri_call_count = call_count;
    ri_in_loop = in_loop;
  }
