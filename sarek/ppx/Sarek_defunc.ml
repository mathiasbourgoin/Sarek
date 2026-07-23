(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Defunctionalization pass (Tier 0)
 *
 * Typed-AST -> typed-AST rewrite that runs between monomorphization
 * (Sarek_mono) and lowering (Sarek_lower_ir). It eliminates first-class
 * function *values* that are bound to `let` variables inside a kernel or
 * helper body, so that lowering never has to emit an `EApp` whose callee is
 * anything other than a directly-named helper.
 *
 * Design: L12-defunctionalization.md (roster/ptx-limits-campaign). This
 * implements the capture-free Tier 0 subset via *application distribution*
 * rather than a synthesized Reynolds tag variant:
 *
 *   - `let f = g in ... f x ...`         (single static candidate, L12 4.1)
 *       -> the binding is dropped and every `f a...` becomes `g a...`,
 *          i.e. a direct named call that lowering already handles.
 *
 *   - `let f = if c then g else h in f x` (genuinely runtime-dynamic, L12 4.2)
 *       -> the application is *distributed into the selector's leaves*:
 *          `if c then g x else h x`. Only direct named calls remain; no
 *          function value ever reaches lowering and no tag variant is
 *          synthesized, so every backend's existing `if`/`match` + direct
 *          call paths handle it with zero emitter changes.
 *
 * This covers the two headline examples in L12 1 (the ones that fail today
 * with "EApp to unknown function" / "EApp with non-variable callee"). It is
 * capture-free: the candidate leaves must be named helpers (module
 * `[@sarek.module]` functions or kernel-local `let rec` helpers), never a
 * lambda closing over kernel-local state. Scalar captures (L12 Tier 1),
 * aggregate captures (Tier 2) and specialization of higher-order *helper
 * parameters* (`sort ~cmp:my_less`) are out of scope for this milestone and
 * documented as follow-ups in the design doc.
 *
 * Escape rules (L12 5): after rewriting, any residual function-typed value
 * sitting in a position that has no runtime representation (record field,
 * tuple/vector element, variant payload, created-array element, or an
 * assignment/return value) is reported as `Function_value_escapes` with a
 * source location, instead of being left to fail with an opaque codegen
 * message three passes downstream.
 ******************************************************************************)

open Sarek_ast
open Sarek_types
open Sarek_typed_ast

let is_fun_ty (t : typ) : bool = match repr t with TFun _ -> true | _ -> false

(** Is the top constructor of this expression's type a function type? *)
let is_fun_typed (e : texpr) : bool = is_fun_ty e.ty

(* -------------------------------------------------------------------------- *)
(* Core rewrite                                                               *)
(* -------------------------------------------------------------------------- *)

(* [fenv] maps the variable id of a function-typed `let` binding to the
   (already-rewritten) selector expression it was bound to. A selector is a
   texpr whose "leaves" (reached through if/match/let/open/seq spines) are the
   concrete callee expressions to apply the arguments to. *)

let rec rewrite (fenv : (int, texpr) Hashtbl.t) (e : texpr) : texpr =
  match e.te with
  (* Drop a function-typed let binding, recording its selector so uses of the
     bound variable resolve to it. *)
  | TELet (_name, id, value, body) when is_fun_typed value ->
      let sel = rewrite_fexpr fenv value in
      Hashtbl.replace fenv id sel ;
      let body' = rewrite fenv body in
      Hashtbl.remove fenv id ;
      body'
  (* Application: resolve the callee to a selector and distribute. *)
  | TEApp (callee, args) ->
      let args' = List.map (rewrite fenv) args in
      let sel = resolve_callee fenv callee in
      distribute_app sel args' e.ty e.te_loc
  (* A bare use of a function-typed local (not in callee position) has no
     runtime representation under Tier 0 -- report it as an escape. *)
  | TEVar (name, id) when is_fun_typed e && Hashtbl.mem fenv id ->
      Sarek_error.raise_error
        (Sarek_error.Function_value_escapes
           ( Printf.sprintf
               "function value '%s' is used where no runtime representation \
                exists (it may only be applied to arguments)"
               name,
             e.ty,
             e.te_loc ))
  | _ -> {e with te = rewrite_desc fenv e.te}

(* Structural recursion for every constructor that is not special-cased in
   [rewrite]. Mirrors Sarek_mono.apply_subst_expr's shape. *)
and rewrite_desc (fenv : (int, texpr) Hashtbl.t) (te : texpr_desc) : texpr_desc
    =
  let r = rewrite fenv in
  match te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ | TEVar _ ->
      te
  | TEVecGet (v, i) -> TEVecGet (r v, r i)
  | TEVecSet (v, i, x) -> TEVecSet (r v, r i, r x)
  | TEArrGet (a, i) -> TEArrGet (r a, r i)
  | TEArrSet (a, i, x) -> TEArrSet (r a, r i, r x)
  | TEFieldGet (rec_, f, idx) -> TEFieldGet (r rec_, f, idx)
  | TEFieldSet (rec_, f, idx, v) -> TEFieldSet (r rec_, f, idx, r v)
  | TEBinop (op, a, b) -> TEBinop (op, r a, r b)
  | TEUnop (op, a) -> TEUnop (op, r a)
  | TEApp (fn, args) -> TEApp (r fn, List.map r args)
  | TEAssign (n, id, v) -> TEAssign (n, id, r v)
  | TELet (n, id, v, b) -> TELet (n, id, r v, r b)
  | TELetRec (n, id, params, fn_body, cont) ->
      TELetRec (n, id, params, r fn_body, r cont)
  | TELetMut (n, id, v, b) -> TELetMut (n, id, r v, r b)
  | TEIf (c, t, e) -> TEIf (r c, r t, Option.map r e)
  | TEFor (v, id, lo, hi, dir, body) -> TEFor (v, id, r lo, r hi, dir, r body)
  | TEWhile (c, b) -> TEWhile (r c, r b)
  | TESeq es -> TESeq (List.map r es)
  | TEMatch (s, cases) -> TEMatch (r s, List.map (fun (p, b) -> (p, r b)) cases)
  | TERecord (n, fields) ->
      TERecord (n, List.map (fun (f, e) -> (f, r e)) fields)
  | TEConstr (tn, cn, arg) -> TEConstr (tn, cn, Option.map r arg)
  | TETuple es -> TETuple (List.map r es)
  | TEReturn e -> TEReturn (r e)
  | TECreateArray (size, t, m) -> TECreateArray (r size, t, m)
  | TEPragma (opts, body) -> TEPragma (opts, r body)
  | TEIntrinsicFun (ref, c, args) -> TEIntrinsicFun (ref, c, List.map r args)
  | TELetShared (n, id, t, size, b) ->
      TELetShared (n, id, t, Option.map r size, r b)
  | TESuperstep (n, d, step, cont) -> TESuperstep (n, d, r step, r cont)
  | TEOpen (path, body) -> TEOpen (path, r body)

(* Resolve the callee of an application to a fully-rewritten selector. *)
and resolve_callee (fenv : (int, texpr) Hashtbl.t) (callee : texpr) : texpr =
  match callee.te with
  | TEVar (_, id) when Hashtbl.mem fenv id -> Hashtbl.find fenv id
  | (TEIf _ | TEMatch _ | TELet _ | TEOpen _ | TESeq _) when is_fun_typed callee
    ->
      rewrite_fexpr fenv callee
  | _ -> rewrite fenv callee

(* Rewrite a function-valued expression into a selector: conditions and
   scrutinees are rewritten normally; the function-typed result positions are
   rewritten recursively; TEVar leaves are resolved through [fenv]. *)
and rewrite_fexpr (fenv : (int, texpr) Hashtbl.t) (fe : texpr) : texpr =
  match fe.te with
  | TEVar (_, id) when Hashtbl.mem fenv id -> Hashtbl.find fenv id
  | TEVar _ -> fe (* named helper leaf *)
  | TEIf (c, t, Some el) ->
      {
        fe with
        te =
          TEIf
            (rewrite fenv c, rewrite_fexpr fenv t, Some (rewrite_fexpr fenv el));
      }
  | TEMatch (s, cases) ->
      {
        fe with
        te =
          TEMatch
            ( rewrite fenv s,
              List.map (fun (p, b) -> (p, rewrite_fexpr fenv b)) cases );
      }
  | TELet (n, id, v, b) when not (is_fun_typed v) ->
      {fe with te = TELet (n, id, rewrite fenv v, rewrite_fexpr fenv b)}
  | TELet (_n, id, v, b) ->
      (* nested function-typed let inside a selector: register and drop *)
      let sel = rewrite_fexpr fenv v in
      Hashtbl.replace fenv id sel ;
      let b' = rewrite_fexpr fenv b in
      Hashtbl.remove fenv id ;
      b'
  | TEOpen (path, b) -> {fe with te = TEOpen (path, rewrite_fexpr fenv b)}
  | TESeq es -> (
      match List.rev es with
      | [] -> fe
      | last :: rest_rev ->
          let init = List.rev_map (rewrite fenv) rest_rev in
          {fe with te = TESeq (init @ [rewrite_fexpr fenv last])})
  | _ ->
      (* Not a recognized function-value shape (e.g. a helper parameter used
         directly). Leave it to the normal rewrite; distribution will keep it
         as a leaf and lowering/codegen will handle or reject it as today. *)
      rewrite fenv fe

(* Push [args] down to every leaf of a selector. The selector's branches
   (if/match arms) are mutually exclusive, so at runtime exactly one leaf
   executes and each argument is evaluated exactly as many times as in the
   source `f args` -- distribution duplicates arguments only *textually*, never
   at runtime, so no fresh temporaries are needed. This also keeps the result a
   plain expression (introducing `let` bindings here would place a
   statement-only form in expression position and break lowering). *)
and distribute_app (sel : texpr) (args : texpr list) (ty : typ) (loc : loc) :
    texpr =
  push_args sel args ty loc

(* Recursively descend a selector's spine, applying [args] at each leaf. *)
and push_args (sel : texpr) (args : texpr list) (ty : typ) (loc : loc) : texpr =
  match sel.te with
  | TEIf (c, t, Some el) ->
      mk_texpr
        (TEIf (c, push_args t args ty loc, Some (push_args el args ty loc)))
        ty
        loc
  | TEMatch (s, cases) ->
      mk_texpr
        (TEMatch (s, List.map (fun (p, b) -> (p, push_args b args ty loc)) cases))
        ty
        loc
  | TELet (n, id, v, b) ->
      mk_texpr (TELet (n, id, v, push_args b args ty loc)) ty loc
  | TEOpen (path, b) -> mk_texpr (TEOpen (path, push_args b args ty loc)) ty loc
  | TESeq es -> (
      match List.rev es with
      | [] -> mk_texpr (TEApp (sel, args)) ty loc
      | last :: rest_rev ->
          let init = List.rev rest_rev in
          mk_texpr (TESeq (init @ [push_args last args ty loc])) ty loc)
  | _ ->
      (* Leaf: a directly-named callee. *)
      mk_texpr (TEApp (sel, args)) ty loc

(* -------------------------------------------------------------------------- *)
(* Escape-rule validation (L12 5)                                            *)
(* -------------------------------------------------------------------------- *)

let escape (what : string) (t : typ) (loc : loc) : 'a =
  Sarek_error.raise_error
    (Sarek_error.Function_value_escapes
       ( Printf.sprintf
           "a function value cannot appear as %s (no runtime representation)"
           what,
         t,
         loc ))

(* Walk the rewritten AST and reject any function-typed value that survives in
   a position with no runtime representation. A function-typed node is only
   legal as the immediate callee of an application. *)
let rec check_expr (e : texpr) : unit =
  (match e.te with
  | TERecord (_, fields) ->
      List.iter
        (fun (_, fe) ->
          if is_fun_typed fe then escape "a record field" fe.ty fe.te_loc)
        fields
  | TETuple es ->
      List.iter
        (fun te ->
          if is_fun_typed te then escape "a tuple element" te.ty te.te_loc)
        es
  | TEConstr (_, _, Some arg) ->
      if is_fun_typed arg then escape "a variant payload" arg.ty arg.te_loc
  | TEReturn re ->
      if is_fun_typed re then escape "a return value" re.ty re.te_loc
  | TEVecSet (_, _, v) ->
      if is_fun_typed v then escape "a vector element" v.ty v.te_loc
  | TEArrSet (_, _, v) ->
      if is_fun_typed v then escape "an array element" v.ty v.te_loc
  | TEAssign (_, _, v) ->
      if is_fun_typed v then escape "an assigned value" v.ty v.te_loc
  | TECreateArray (_, t, _) ->
      if is_fun_ty t then escape "an array element type" t e.te_loc
  | _ -> ()) ;
  (* Recurse. The callee child of an application is legally function-typed, so
     skip it and only descend into the arguments. *)
  match e.te with
  | TEApp (_callee, args) -> List.iter check_expr args
  | _ -> iter_children check_expr e

and iter_children (f : texpr -> unit) (e : texpr) : unit =
  match e.te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ | TEVar _ ->
      ()
  | TEVecGet (a, b) | TEArrGet (a, b) | TEBinop (_, a, b) | TEWhile (a, b) ->
      f a ;
      f b
  | TEVecSet (a, b, c) | TEArrSet (a, b, c) ->
      f a ;
      f b ;
      f c
  | TEFieldGet (a, _, _)
  | TEUnop (_, a)
  | TEReturn a
  | TEPragma (_, a)
  | TEOpen (_, a) ->
      f a
  | TEFieldSet (a, _, _, b) ->
      f a ;
      f b
  | TEApp (fn, args) ->
      f fn ;
      List.iter f args
  | TEAssign (_, _, v) -> f v
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      f v ;
      f b
  | TELetRec (_, _, _, fn_body, cont) ->
      f fn_body ;
      f cont
  | TEIf (c, t, e) ->
      f c ;
      f t ;
      Option.iter f e
  | TEFor (_, _, lo, hi, _, body) ->
      f lo ;
      f hi ;
      f body
  | TESeq es | TETuple es -> List.iter f es
  | TEMatch (s, cases) ->
      f s ;
      List.iter (fun (_, b) -> f b) cases
  | TERecord (_, fields) -> List.iter (fun (_, e) -> f e) fields
  | TEConstr (_, _, arg) -> Option.iter f arg
  | TECreateArray (size, _, _) -> f size
  | TEIntrinsicFun (_, _, args) -> List.iter f args
  | TELetShared (_, _, _, size, b) ->
      Option.iter f size ;
      f b
  | TESuperstep (_, _, step, cont) ->
      f step ;
      f cont

(* -------------------------------------------------------------------------- *)
(* Entry point                                                                *)
(* -------------------------------------------------------------------------- *)

let rewrite_body (body : texpr) : texpr =
  let fenv : (int, texpr) Hashtbl.t = Hashtbl.create 8 in
  let body' = rewrite fenv body in
  check_expr body' ;
  body'

let defunctionalize (kernel : tkernel) : tkernel =
  let module_items' =
    List.map
      (function
        | TMFun (name, is_rec, params, body) ->
            (* A helper whose parameter is itself function-typed is a
               higher-order helper (specialization case, L12 4.3 / 4.1). That
               is out of scope for this Tier-0 milestone; leave its body
               untouched so the existing lowering behavior is unchanged. *)
            if List.exists (fun p -> is_fun_ty p.tparam_type) params then
              TMFun (name, is_rec, params, body)
            else TMFun (name, is_rec, params, rewrite_body body)
        | TMConst _ as item -> item)
      kernel.tkern_module_items
  in
  let body' = rewrite_body kernel.tkern_body in
  if is_fun_ty kernel.tkern_return_type then
    escape "a kernel return type" kernel.tkern_return_type kernel.tkern_loc ;
  {kernel with tkern_module_items = module_items'; tkern_body = body'}
