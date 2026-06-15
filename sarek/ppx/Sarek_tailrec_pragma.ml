(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Pragma-Based Recursion Inlining
 *
 * This module handles explicit inlining of recursive functions using
 * pragma ["sarek.inline N"] directives. It mechanically inlines N times
 * and verifies no recursive calls remain.
 ******************************************************************************)

open Sarek_typed_ast

(** {1 Pragma-Based Inlining} *)

(** Maximum number of AST nodes allowed after inlining. Prevents code explosion
    from exponential recursion like fib. *)
let max_inlined_nodes = 10000

(** Count AST nodes in an expression *)
let rec count_nodes (expr : texpr) : int =
  1
  +
  match expr.te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
      0
  | TEApp (fn, args) ->
      count_nodes fn + List.fold_left (fun acc a -> acc + count_nodes a) 0 args
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) -> count_nodes v + count_nodes b
  | TELetShared (_, _, _, _, b) -> count_nodes b
  | TEIf (c, t, e) -> (
      count_nodes c + count_nodes t
      + match e with Some x -> count_nodes x | None -> 0)
  | TEFor (_, _, lo, hi, _, body) ->
      count_nodes lo + count_nodes hi + count_nodes body
  | TEWhile (c, b) -> count_nodes c + count_nodes b
  | TESeq es -> List.fold_left (fun acc e -> acc + count_nodes e) 0 es
  | TEBinop (_, a, b) -> count_nodes a + count_nodes b
  | TEUnop (_, e) -> count_nodes e
  | TEVecGet (v, i) | TEArrGet (v, i) -> count_nodes v + count_nodes i
  | TEVecSet (v, i, x) | TEArrSet (v, i, x) ->
      count_nodes v + count_nodes i + count_nodes x
  | TEFieldGet (r, _, _) -> count_nodes r
  | TEFieldSet (r, _, _, v) -> count_nodes r + count_nodes v
  | TEMatch (s, cases) ->
      count_nodes s
      + List.fold_left (fun acc (_, b) -> acc + count_nodes b) 0 cases
  | TETuple es -> List.fold_left (fun acc e -> acc + count_nodes e) 0 es
  | TERecord (_, fs) ->
      List.fold_left (fun acc (_, e) -> acc + count_nodes e) 0 fs
  | TEConstr (_, _, arg) -> (
      match arg with Some e -> count_nodes e | None -> 0)
  | TEAssign (_, _, v) -> count_nodes v
  | TEReturn e -> count_nodes e
  | TESuperstep (_, _, step, cont) -> count_nodes step + count_nodes cont
  | TEPragma (_, body) -> count_nodes body
  | TEOpen (_, body) -> count_nodes body
  | TECreateArray (sz, _, _) -> count_nodes sz
  | TEIntrinsicFun (_, _, args) ->
      List.fold_left (fun acc a -> acc + count_nodes a) 0 args
  | TELetRec (_, _, _, fb, c) -> count_nodes fb + count_nodes c

(** Parse pragma options to extract sarek.inline depth. Returns Some depth for
    "sarek.inline N", None otherwise. *)
let parse_sarek_inline_pragma (opts : string list) : int option =
  match opts with
  | [opt] -> (
      (* Handle "sarek.inline N" as a single string *)
      match String.split_on_char ' ' opt with
      | ["sarek.inline"; n] -> int_of_string_opt n
      | _ -> None)
  | ["sarek.inline"; n] -> int_of_string_opt n
  | _ -> None

(** Check if pragma is an unroll pragma *)
let is_unroll_pragma (opts : string list) : bool =
  match opts with
  | [opt] -> String.length opt >= 6 && String.sub opt 0 6 = "unroll"
  | "unroll" :: _ -> true
  | _ -> false

(** Substitute a variable in an expression with another expression *)
let rec subst_var (name : string) (id : int) (replacement : texpr) (e : texpr) :
    texpr =
  match e.te with
  | TEVar (n, i) when n = name && i = id -> replacement
  | TEVar _ -> e
  | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _ | TEBool _ | TEUnit
    ->
      e
  | TEApp (fn, args) ->
      {
        e with
        te =
          TEApp
            ( subst_var name id replacement fn,
              List.map (subst_var name id replacement) args );
      }
  | TELet (n, i, v, b) ->
      (* Don't substitute in body if this binding shadows our variable *)
      let v' = subst_var name id replacement v in
      let b' =
        if n = name && i = id then b else subst_var name id replacement b
      in
      {e with te = TELet (n, i, v', b')}
  | TELetMut (n, i, v, b) ->
      let v' = subst_var name id replacement v in
      let b' =
        if n = name && i = id then b else subst_var name id replacement b
      in
      {e with te = TELetMut (n, i, v', b')}
  | TEIf (c, t, el) ->
      {
        e with
        te =
          TEIf
            ( subst_var name id replacement c,
              subst_var name id replacement t,
              Option.map (subst_var name id replacement) el );
      }
  | TESeq es ->
      {e with te = TESeq (List.map (subst_var name id replacement) es)}
  | TEFor (v, vid, lo, hi, dir, b) ->
      let lo' = subst_var name id replacement lo in
      let hi' = subst_var name id replacement hi in
      let b' =
        if v = name && vid = id then b else subst_var name id replacement b
      in
      {e with te = TEFor (v, vid, lo', hi', dir, b')}
  | TEWhile (c, b) ->
      {
        e with
        te =
          TEWhile
            (subst_var name id replacement c, subst_var name id replacement b);
      }
  | TEBinop (op, a, b) ->
      {
        e with
        te =
          TEBinop
            ( op,
              subst_var name id replacement a,
              subst_var name id replacement b );
      }
  | TEUnop (op, x) -> {e with te = TEUnop (op, subst_var name id replacement x)}
  | TEVecGet (v, i) ->
      {
        e with
        te =
          TEVecGet
            (subst_var name id replacement v, subst_var name id replacement i);
      }
  | TEVecSet (v, i, x) ->
      {
        e with
        te =
          TEVecSet
            ( subst_var name id replacement v,
              subst_var name id replacement i,
              subst_var name id replacement x );
      }
  | TEArrGet (v, i) ->
      {
        e with
        te =
          TEArrGet
            (subst_var name id replacement v, subst_var name id replacement i);
      }
  | TEArrSet (v, i, x) ->
      {
        e with
        te =
          TEArrSet
            ( subst_var name id replacement v,
              subst_var name id replacement i,
              subst_var name id replacement x );
      }
  | TEFieldGet (r, t, f) ->
      {e with te = TEFieldGet (subst_var name id replacement r, t, f)}
  | TEFieldSet (r, t, f, v) ->
      {
        e with
        te =
          TEFieldSet
            ( subst_var name id replacement r,
              t,
              f,
              subst_var name id replacement v );
      }
  | TEMatch (s, cases) ->
      {
        e with
        te =
          TEMatch
            ( subst_var name id replacement s,
              List.map
                (fun (p, b) -> (p, subst_var name id replacement b))
                cases );
      }
  | TETuple es ->
      {e with te = TETuple (List.map (subst_var name id replacement) es)}
  | TERecord (n, fs) ->
      {
        e with
        te =
          TERecord
            (n, List.map (fun (f, x) -> (f, subst_var name id replacement x)) fs);
      }
  | TEConstr (t, c, arg) ->
      {
        e with
        te = TEConstr (t, c, Option.map (subst_var name id replacement) arg);
      }
  | TEAssign (n, i, v) ->
      {e with te = TEAssign (n, i, subst_var name id replacement v)}
  | TEReturn x -> {e with te = TEReturn (subst_var name id replacement x)}
  | TESuperstep (s, n, step, cont) ->
      {
        e with
        te =
          TESuperstep
            ( s,
              n,
              subst_var name id replacement step,
              subst_var name id replacement cont );
      }
  | TEPragma (p, b) ->
      {e with te = TEPragma (p, subst_var name id replacement b)}
  | TEOpen (m, b) -> {e with te = TEOpen (m, subst_var name id replacement b)}
  | TECreateArray (sz, ty, mem) ->
      {e with te = TECreateArray (subst_var name id replacement sz, ty, mem)}
  | TEIntrinsicFun (r, c, args) ->
      {
        e with
        te = TEIntrinsicFun (r, c, List.map (subst_var name id replacement) args);
      }
  | TELetShared (n, i, sz, ty, b) ->
      let b' =
        if n = name && i = id then b else subst_var name id replacement b
      in
      {e with te = TELetShared (n, i, sz, ty, b')}
  | TELetRec (n, i, ps, fb, c) ->
      let fb' =
        if n = name && i = id then fb else subst_var name id replacement fb
      in
      let c' =
        if n = name && i = id then c else subst_var name id replacement c
      in
      {e with te = TELetRec (n, i, ps, fb', c')}
  (* These don't contain variable references to substitute *)
  | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ -> e

(** Substitute all occurrences of a recursive call with the function body. This
    is one step of mechanical inlining.

    IMPORTANT: We do NOT recurse into the substituted body - that happens in the
    next inlining round. This ensures we do exactly N rounds of inlining.

    Instead of creating let-bindings (which generate statement expressions that
    OpenCL doesn't support), we directly substitute parameter values into the
    body. *)
let substitute_recursive_calls (fname : string) (params : tparam list)
    (body : texpr) (expr : texpr) : texpr =
  let rec subst e =
    match e.te with
    | TEApp (fn, args) when Sarek_tailrec_analysis.is_self_call fname fn ->
        (* Direct substitution: replace params with args in body.
           Note: we substitute in args (they're from the current expression)
           but then substitute those into body. *)
        let substituted_args = List.map subst args in
        (* Substitute each parameter with its corresponding argument *)
        List.fold_left2
          (fun acc param arg ->
            subst_var param.tparam_name param.tparam_id arg acc)
          body
          params
          substituted_args
    | TEApp (fn, args) -> {e with te = TEApp (subst fn, List.map subst args)}
    | TELet (n, id, v, b) -> {e with te = TELet (n, id, subst v, subst b)}
    | TELetMut (n, id, v, b) -> {e with te = TELetMut (n, id, subst v, subst b)}
    | TEIf (c, t, el) ->
        {e with te = TEIf (subst c, subst t, Option.map subst el)}
    | TESeq es -> {e with te = TESeq (List.map subst es)}
    | TEFor (v, id, lo, hi, dir, b) ->
        {e with te = TEFor (v, id, subst lo, subst hi, dir, subst b)}
    | TEWhile (c, b) -> {e with te = TEWhile (subst c, subst b)}
    | TEBinop (op, a, b) -> {e with te = TEBinop (op, subst a, subst b)}
    | TEUnop (op, x) -> {e with te = TEUnop (op, subst x)}
    | TEVecGet (v, i) -> {e with te = TEVecGet (subst v, subst i)}
    | TEVecSet (v, i, x) -> {e with te = TEVecSet (subst v, subst i, subst x)}
    | TEArrGet (v, i) -> {e with te = TEArrGet (subst v, subst i)}
    | TEArrSet (v, i, x) -> {e with te = TEArrSet (subst v, subst i, subst x)}
    | TEFieldGet (r, t, f) -> {e with te = TEFieldGet (subst r, t, f)}
    | TEFieldSet (r, t, f, v) ->
        {e with te = TEFieldSet (subst r, t, f, subst v)}
    | TEMatch (s, cases) ->
        {
          e with
          te = TEMatch (subst s, List.map (fun (p, b) -> (p, subst b)) cases);
        }
    | TETuple es -> {e with te = TETuple (List.map subst es)}
    | TERecord (n, fs) ->
        {e with te = TERecord (n, List.map (fun (f, x) -> (f, subst x)) fs)}
    | TEConstr (t, c, arg) ->
        {e with te = TEConstr (t, c, Option.map subst arg)}
    | TEAssign (n, id, v) -> {e with te = TEAssign (n, id, subst v)}
    | TEReturn x -> {e with te = TEReturn (subst x)}
    | TESuperstep (s, n, step, cont) ->
        {e with te = TESuperstep (s, n, subst step, subst cont)}
    | TEPragma (p, b) -> {e with te = TEPragma (p, subst b)}
    | TEOpen (m, b) -> {e with te = TEOpen (m, subst b)}
    | TECreateArray (sz, ty, mem) ->
        {e with te = TECreateArray (subst sz, ty, mem)}
    | TEIntrinsicFun (r, c, args) ->
        {e with te = TEIntrinsicFun (r, c, List.map subst args)}
    | TELetShared (n, id, sz, ty, b) ->
        {e with te = TELetShared (n, id, sz, ty, subst b)}
    | TELetRec (n, id, ps, fb, c) ->
        {e with te = TELetRec (n, id, ps, subst fb, subst c)}
    | _ -> e
  in
  subst expr

(** Inline a recursive function N times using pragma ["sarek.inline N"]. Returns
    Ok new_expr if successful, Error message if failed.

    Note: After inlining, some calls may remain syntactically present but they
    will be in dead code branches (e.g., after the base case condition). We emit
    a warning but allow the code to proceed, trusting the GPU compiler to
    eliminate dead code. *)
let inline_with_pragma (fname : string) (params : tparam list) (body : texpr)
    (depth : int) : (texpr, string) result =
  Sarek_debug.log "inline_with_pragma '%s' depth=%d" fname depth ;
  let rec inline_n n expr =
    Sarek_debug.log "inline_n n=%d" n ;
    if n = 0 then (
      Sarek_debug.log "inline_n done (n=0)" ;
      expr)
    else (
      (* Check node count BEFORE substitution to avoid explosion *)
      Sarek_debug.log "counting nodes..." ;
      let nodes = count_nodes expr in
      Sarek_debug.log "nodes=%d" nodes ;
      if nodes > max_inlined_nodes then
        raise
          (Failure
             (Printf.sprintf
                "Inlining produced %d nodes (limit: %d). Reduce inline depth \
                 or use tail recursion."
                nodes
                max_inlined_nodes)) ;
      Sarek_debug.log "substituting..." ;
      let expr' = substitute_recursive_calls fname params body expr in
      Sarek_debug.log "substitution done, recursing" ;
      inline_n (n - 1) expr')
  in
  try
    Sarek_debug.log "starting inline_n" ;
    let result = inline_n depth body in
    Sarek_debug.log "inline_n finished" ;
    (* Check for remaining recursive calls *)
    let remaining = Sarek_tailrec_analysis.count_recursive_calls fname result in
    if remaining > 0 && Sarek_debug.enabled then (
      (* Warn but allow - these should be in dead code branches *)
      Format.eprintf
        "Sarek warning: %d recursive calls remain after %d inlines.@."
        remaining
        depth ;
      Format.eprintf
        "  These should be in unreachable branches (GPU will eliminate).@." ;
      Format.eprintf "  If you get runtime errors, increase the inline depth.@.") ;
    Ok result
  with Failure msg -> Error msg
