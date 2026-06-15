(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Tail Recursion Elimination
 *
 * This module transforms tail-recursive functions into while loops suitable
 * for GPU execution. It uses a continue-flag approach with mutable loop variables.
 ******************************************************************************)

open Sarek_typed_ast
open Sarek_types

(** {1 Tail Recursion Elimination} *)

(** Fresh variable ID generator for transformation (thread-safe) *)
let transform_var_counter = Atomic.make 0

let fresh_transform_id () = Atomic.fetch_and_add transform_var_counter 1

(** Transform a tail-recursive function into a loop.

    Uses a simple continue-flag approach suitable for GPU code:
    - _continue: bool mutable, starts true
    - _result: mutable of result type, stores the return value
    - Loop while _continue
    - Base case: _result := value; _continue := false
    - Recursive call: update loop vars (continue stays true) *)
let eliminate_tail_recursion (fname : string) (params : tparam list)
    (body : texpr) (loc : Sarek_ast.loc) : texpr =
  (* Create loop variable IDs for each parameter - use __name to avoid C redeclaration *)
  let loop_vars =
    List.map
      (fun p ->
        (p.tparam_name, p.tparam_id, fresh_transform_id (), p.tparam_type))
      params
  in

  (* Create result and continue variables *)
  let result_id = fresh_transform_id () in
  let continue_id = fresh_transform_id () in
  let result_ty = body.ty in

  (* Helper: substitute original param references with loop var references in an expression *)
  let rec substitute_params expr =
    match expr.te with
    | TEVar (name, id) -> (
        (* Check if this is a reference to one of our function parameters *)
        match
          List.find_opt
            (fun (n, orig_id, _, _) -> n = name && orig_id = id)
            loop_vars
        with
        | Some (n, _, loop_id, _) ->
            (* Replace with reference to loop variable *)
            {expr with te = TEVar ("__" ^ n, loop_id)}
        | None -> expr)
    | TELet (n, id, v, b) ->
        {expr with te = TELet (n, id, substitute_params v, substitute_params b)}
    | TELetMut (n, id, v, b) ->
        {
          expr with
          te = TELetMut (n, id, substitute_params v, substitute_params b);
        }
    | TEIf (c, t, e) ->
        {
          expr with
          te =
            TEIf
              ( substitute_params c,
                substitute_params t,
                Option.map substitute_params e );
        }
    | TESeq es -> {expr with te = TESeq (List.map substitute_params es)}
    | TEApp (fn, args) ->
        {
          expr with
          te = TEApp (substitute_params fn, List.map substitute_params args);
        }
    | TEBinop (op, a, b) ->
        {expr with te = TEBinop (op, substitute_params a, substitute_params b)}
    | TEUnop (op, e) -> {expr with te = TEUnop (op, substitute_params e)}
    | TEWhile (c, b) ->
        {expr with te = TEWhile (substitute_params c, substitute_params b)}
    | TEFor (v, id, lo, hi, dir, b) ->
        {
          expr with
          te =
            TEFor
              ( v,
                id,
                substitute_params lo,
                substitute_params hi,
                dir,
                substitute_params b );
        }
    | TEVecGet (v, i) ->
        {expr with te = TEVecGet (substitute_params v, substitute_params i)}
    | TEVecSet (v, i, x) ->
        {
          expr with
          te =
            TEVecSet
              (substitute_params v, substitute_params i, substitute_params x);
        }
    | TEArrGet (v, i) ->
        {expr with te = TEArrGet (substitute_params v, substitute_params i)}
    | TEArrSet (v, i, x) ->
        {
          expr with
          te =
            TEArrSet
              (substitute_params v, substitute_params i, substitute_params x);
        }
    | TEMatch (s, cases) ->
        {
          expr with
          te =
            TEMatch
              ( substitute_params s,
                List.map (fun (p, b) -> (p, substitute_params b)) cases );
        }
    | TEReturn e -> {expr with te = TEReturn (substitute_params e)}
    | TEAssign (n, id, v) ->
        {expr with te = TEAssign (n, id, substitute_params v)}
    | TETuple es -> {expr with te = TETuple (List.map substitute_params es)}
    | TERecord (n, fs) ->
        {
          expr with
          te = TERecord (n, List.map (fun (f, e) -> (f, substitute_params e)) fs);
        }
    | TEConstr (t, c, arg) ->
        {expr with te = TEConstr (t, c, Option.map substitute_params arg)}
    | TEFieldGet (r, t, f) ->
        {expr with te = TEFieldGet (substitute_params r, t, f)}
    | TEFieldSet (r, t, f, v) ->
        {
          expr with
          te = TEFieldSet (substitute_params r, t, f, substitute_params v);
        }
    | TEOpen (m, b) -> {expr with te = TEOpen (m, substitute_params b)}
    | TEPragma (p, b) -> {expr with te = TEPragma (p, substitute_params b)}
    | TECreateArray (sz, ty, mem) ->
        {expr with te = TECreateArray (substitute_params sz, ty, mem)}
    | TEIntrinsicFun (r, c, args) ->
        {expr with te = TEIntrinsicFun (r, c, List.map substitute_params args)}
    | TELetShared (n, id, sz, ty, b) ->
        {expr with te = TELetShared (n, id, sz, ty, substitute_params b)}
    | TESuperstep (s, n, step, cont) ->
        {
          expr with
          te = TESuperstep (s, n, substitute_params step, substitute_params cont);
        }
    | TELetRec (n, id, ps, fb, c) ->
        {
          expr with
          te = TELetRec (n, id, ps, substitute_params fb, substitute_params c);
        }
    | _ -> expr
  in

  (* Helper: create assignment to loop variable *)
  let mk_loop_assign name id value =
    {te = TEAssign ("__" ^ name, id, value); ty = t_unit; te_loc = loc}
  in

  (* Helper: create assignment to _continue := false (to break from loop) *)
  let mk_break () =
    {
      te =
        TEAssign
          ( "_continue",
            continue_id,
            {te = TEBool false; ty = t_bool; te_loc = loc} );
      ty = t_unit;
      te_loc = loc;
    }
  in

  (* Helper: create assignment to _result *)
  let mk_result_assign value =
    {te = TEAssign ("_result", result_id, value); ty = t_unit; te_loc = loc}
  in

  (* Transform the body: replace recursive calls with assignments *)
  let rec transform expr =
    match expr.te with
    | TEApp (fn, args) when Sarek_tailrec_analysis.is_self_call fname fn ->
        (* Replace recursive call with assignment to loop variables.
           _continue stays true, so loop continues.

           IMPORTANT: We must evaluate all arguments BEFORE assigning them,
           otherwise `gcd b (a mod b)` would become:
             __a := __b;
             __b := __a mod __b;  // Wrong! Uses new __a, not old
           Instead we generate:
             let _tmp_0 = __b in
             let _tmp_1 = __a mod __b in
             __a := _tmp_0;
             __b := _tmp_1;
        *)
        (* Create temporary variables for each argument *)
        let temps =
          List.mapi
            (fun i arg ->
              let tmp_id = fresh_transform_id () in
              let tmp_name = "_tmp_" ^ string_of_int i in
              (tmp_name, tmp_id, arg))
            args
        in
        (* Create assignments from temps to loop vars *)
        let assigns =
          List.map2
            (fun (name, _orig_id, loop_id, _ty) (tmp_name, tmp_id, arg) ->
              let tmp_ref =
                {te = TEVar (tmp_name, tmp_id); ty = arg.ty; te_loc = loc}
              in
              mk_loop_assign name loop_id tmp_ref)
            loop_vars
            temps
        in
        (* Wrap assigns in let bindings for temps *)
        let body = {te = TESeq assigns; ty = t_unit; te_loc = expr.te_loc} in
        List.fold_right
          (fun (tmp_name, tmp_id, arg) acc ->
            {te = TELet (tmp_name, tmp_id, arg, acc); ty = t_unit; te_loc = loc})
          temps
          body
    | TEIf (cond, then_, else_opt) ->
        (* Only transform branches - condition is NOT in tail position *)
        {
          expr with
          te =
            TEIf
              ( cond,
                (* Keep condition as-is *)
                transform then_,
                Option.map transform else_opt );
        }
    | TELet (name, id, value, body_expr) ->
        (* Only transform body - value binding is NOT in tail position *)
        {expr with te = TELet (name, id, value, transform body_expr)}
    | TELetMut (name, id, value, body_expr) ->
        {expr with te = TELetMut (name, id, value, transform body_expr)}
    | TESeq exprs -> (
        (* Only the last expression is in tail position *)
        match List.rev exprs with
        | [] -> expr
        | last :: rest ->
            let transformed_last = transform last in
            {expr with te = TESeq (List.rev (transformed_last :: rest))})
    | TEMatch (scrut, cases) ->
        (* Only transform branch bodies - scrutinee is NOT in tail position *)
        {
          expr with
          te =
            TEMatch
              ( scrut,
                (* Keep scrutinee as-is *)
                List.map
                  (fun (pat, body_expr) -> (pat, transform body_expr))
                  cases );
        }
    | TEReturn e ->
        (* Return becomes: _result := e; _continue := false *)
        {
          te = TESeq [mk_result_assign (transform e); mk_break ()];
          ty = t_unit;
          te_loc = loc;
        }
    | _ ->
        (* For non-tail expressions that return a value, wrap as result assignment *)
        if Sarek_tailrec_analysis.count_recursive_calls fname expr = 0 then
          (* Base case: _result := expr; _continue := false *)
          {
            te = TESeq [mk_result_assign expr; mk_break ()];
            ty = t_unit;
            te_loc = loc;
          }
        else expr (* Keep as-is, will be handled by recursion *)
  in

  (* Build the loop structure *)

  (* 1. Initialize loop variables from parameters - use __name prefix to avoid C redeclaration *)
  let init_loop_vars =
    List.map
      (fun (name, orig_id, loop_id, ty) ->
        let param_ref = {te = TEVar (name, orig_id); ty; te_loc = loc} in
        ("__" ^ name, loop_id, param_ref, ty))
      loop_vars
  in

  (* 2. Initialize _continue to true *)
  let init_continue =
    ( "_continue",
      continue_id,
      {te = TEBool true; ty = t_bool; te_loc = loc},
      t_bool )
  in

  (* 3. Initialize _result to a default value of the appropriate type.
     This value will be overwritten before use, but we need a valid initializer. *)
  let init_result =
    let rec default_for_type ty =
      match ty with
      | TPrim TInt32 ->
          (* int/int32 use TEInt with t_int32 type *)
          {te = TEInt 0; ty = result_ty; te_loc = loc}
      | TReg Int -> {te = TEInt 0; ty = result_ty; te_loc = loc}
      | TReg Int64 -> {te = TEInt64 0L; ty = result_ty; te_loc = loc}
      | TReg Float32 -> {te = TEFloat 0.0; ty = result_ty; te_loc = loc}
      | TReg Float64 -> {te = TEDouble 0.0; ty = result_ty; te_loc = loc}
      | TReg Char -> {te = TEInt 0; ty = result_ty; te_loc = loc}
      | TReg (Custom _) -> {te = TEInt 0; ty = result_ty; te_loc = loc}
      | TPrim TBool -> {te = TEBool false; ty = result_ty; te_loc = loc}
      | TPrim TUnit -> {te = TEUnit; ty = result_ty; te_loc = loc}
      | TVar {contents = Link t} -> default_for_type t
      | _ ->
          (* For other types, use TEInt 0 as a fallback - it's just a placeholder *)
          {te = TEInt 0; ty = result_ty; te_loc = loc}
    in
    let init_val = default_for_type result_ty in
    ("_result", result_id, init_val, result_ty)
  in

  (* 4. Transform body - first substitute param refs, then transform recursive calls *)
  let transformed_body = substitute_params (transform body) in

  (* 5. While loop: while _continue do transformed_body done *)
  let loop_cond =
    {te = TEVar ("_continue", continue_id); ty = t_bool; te_loc = loc}
  in

  let while_loop =
    {te = TEWhile (loop_cond, transformed_body); ty = t_unit; te_loc = loc}
  in

  (* 6. Return _result *)
  let return_result =
    {te = TEVar ("_result", result_id); ty = result_ty; te_loc = loc}
  in

  (* 7. Wrap everything in let bindings *)
  let wrap_lets body_expr bindings =
    List.fold_right
      (fun (name, id, init, _ty) acc ->
        {te = TELetMut (name, id, init, acc); ty = acc.ty; te_loc = loc})
      bindings
      body_expr
  in

  (* Also need spoc_prof_cond for Kirc's while loop generation *)
  let spoc_prof_cond_id = fresh_transform_id () in
  let init_spoc_prof_cond =
    ( "spoc_prof_cond",
      spoc_prof_cond_id,
      {te = TEBool true; ty = t_bool; te_loc = loc},
      t_bool )
  in

  let final_body =
    {te = TESeq [while_loop; return_result]; ty = result_ty; te_loc = loc}
  in
  (* Order: loop vars first, then continue, then result, then spoc_prof_cond (innermost) *)
  wrap_lets
    final_body
    (init_spoc_prof_cond :: init_result :: init_continue :: init_loop_vars)
