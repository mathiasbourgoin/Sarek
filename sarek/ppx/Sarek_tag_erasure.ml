(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Static tag erasure (campaign item L14, scoped tier)
 *
 * Typed-AST -> typed-AST rewrite that runs after defunctionalization
 * (Sarek_defunc) / tail-recursion elimination and BEFORE lowering
 * (Sarek_lower_ir), on the DEVICE-lowered kernel only: the un-erased
 * [native_kernel] captured earlier in Sarek_ppx keeps the tag and stays the
 * OCaml reference path. See roster/ptx-limits-campaign/L14-static-tag-erasure.md.
 *
 * The tag stays in the typed AST for typing / OCaml-side exhaustiveness
 * checking; this pass erases it for code generation wherever the live
 * constructor of a variant-typed *storage slot* is statically determined -
 * the slot is written exactly once by a literal constructor application
 * ([TEConstr]) and is only ever read as the scrutinee of a [match]. This is
 * case-of-known-constructor / iota-reduction applied to the storage
 * representation of a value whose constructor identity is invariant, which is
 * a standard, decades-old compiler technique (design doc SS7).
 *
 * Scoped tier implemented here (S1): a variant-typed plain (immutable)
 * kernel-local `let` binding, so there is exactly one write site by
 * construction:
 *
 *     let s = C payload in ... match s with .. | C x -> rhs | .. ...
 *  ~> let s = payload   in ... let x = s in rhs ...            (unary)
 *     let s = C         in ... match s with .. | C -> rhs | .. ...
 *  ~>                       ... rhs ...                         (nullary, drop)
 *
 * The tag store and the whole branch chain disappear from every backend's
 * emitted code (observable in the IR / CUDA-C / PTX), and behaviour is
 * unchanged: the reduced form runs the C-arm unconditionally, binding the
 * payload for the unary case. Anything with a runtime-selected constructor - a
 * mutable slot, a value from a `TEIf`/`TEMatch`/helper call, or a slot read as
 * a whole variant value rather than only as a `match` scrutinee - is
 * conservatively left tagged, exactly today's behaviour.
 *
 * NOT implemented (deferred to a follow-up tier, see the design doc's
 * "Implementation findings" section): erasing a variant-typed *record field*
 * (the layout-unblock headline of SS0/SS8). Empirically that requires threading
 * a rewritten record type through every occurrence AND reconciling with the
 * `[@@sarek.type]` declaration that the Interpreter and host marshalling
 * consume by field position - substantially more than an expression rewrite,
 * and it regresses a currently-working backend if attempted as a pure
 * expression rewrite.
 ******************************************************************************)

open Sarek_types
open Sarek_typed_ast

let is_variant_ty (t : typ) : bool =
  match repr t with TVariant _ -> true | _ -> false

(* -------------------------------------------------------------------------- *)
(* Generic structural map / iter over the typed AST (mirrors the shape of     *)
(* Sarek_defunc.rewrite_desc / iter_children).                                *)
(* -------------------------------------------------------------------------- *)

let map_desc (f : texpr -> texpr) (te : texpr_desc) : texpr_desc =
  match te with
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ | TEVar _ ->
      te
  | TEVecGet (v, i) -> TEVecGet (f v, f i)
  | TEVecSet (v, i, x) -> TEVecSet (f v, f i, f x)
  | TEArrGet (a, i) -> TEArrGet (f a, f i)
  | TEArrSet (a, i, x) -> TEArrSet (f a, f i, f x)
  | TEFieldGet (r, fld, idx) -> TEFieldGet (f r, fld, idx)
  | TEFieldSet (r, fld, idx, v) -> TEFieldSet (f r, fld, idx, f v)
  | TEBinop (op, a, b) -> TEBinop (op, f a, f b)
  | TEUnop (op, a) -> TEUnop (op, f a)
  | TEApp (fn, args) -> TEApp (f fn, List.map f args)
  | TEAssign (n, id, v) -> TEAssign (n, id, f v)
  | TELet (n, id, v, b) -> TELet (n, id, f v, f b)
  | TELetRec (n, id, ps, fn_body, cont) ->
      TELetRec (n, id, ps, f fn_body, f cont)
  | TELetMut (n, id, v, b) -> TELetMut (n, id, f v, f b)
  | TEIf (c, t, e) -> TEIf (f c, f t, Option.map f e)
  | TEFor (v, id, lo, hi, dir, body) -> TEFor (v, id, f lo, f hi, dir, f body)
  | TEWhile (c, b) -> TEWhile (f c, f b)
  | TESeq es -> TESeq (List.map f es)
  | TEMatch (s, cases) -> TEMatch (f s, List.map (fun (p, b) -> (p, f b)) cases)
  | TERecord (n, fields) ->
      TERecord (n, List.map (fun (fl, e) -> (fl, f e)) fields)
  | TEConstr (tn, cn, arg) -> TEConstr (tn, cn, Option.map f arg)
  | TETuple es -> TETuple (List.map f es)
  | TEReturn e -> TEReturn (f e)
  | TECreateArray (size, t, m) -> TECreateArray (f size, t, m)
  | TEPragma (opts, body) -> TEPragma (opts, f body)
  | TEIntrinsicFun (r, c, args) -> TEIntrinsicFun (r, c, List.map f args)
  | TELetShared (n, id, t, size, b) ->
      TELetShared (n, id, t, Option.map f size, f b)
  | TESuperstep (n, d, step, cont) -> TESuperstep (n, d, f step, f cont)
  | TEOpen (path, body) -> TEOpen (path, f body)

let iter_children (f : texpr -> unit) (e : texpr) : unit =
  ignore
    (map_desc
       (fun c ->
         f c ;
         c)
       e.te)

let map_children (f : texpr -> texpr) (e : texpr) : texpr =
  {e with te = map_desc f e.te}

(* -------------------------------------------------------------------------- *)
(* Eligibility + reduction, keyed on a slot variable id                       *)
(* -------------------------------------------------------------------------- *)

let is_slot_var (id : int) (e : texpr) : bool =
  match e.te with TEVar (_, i) -> i = id | _ -> false

(* A constructor arm is reducible only when its payload pattern binds through
   the forms [reduce_matches] actually substitutes: no payload (nullary), a
   single [TPVar], or a binder-free [TPAny] (`C _ -> ...`). Compound payload
   patterns (a multi-arg constructor's [TPTuple], nested [TPConstr], ...) would
   have their binders silently dropped by the reduction, so they make the arm -
   and therefore the slot - ineligible and the tag is retained. *)
let is_reducible_payload (arg : tpattern option) : bool =
  match arg with
  | None -> true
  | Some {tpat = TPVar _; _} | Some {tpat = TPAny; _} -> true
  | Some _ -> false

let is_ctor_case (cname : string) ((p, _) : tpattern * texpr) : bool =
  match p.tpat with
  | TPConstr (_, cn, arg) -> cn = cname && is_reducible_payload arg
  | _ -> false

(* Every read of the slot variable is the scrutinee of a `match` that has an
   explicit arm for [cname]. Any other use of the variable (which would need
   the whole tagged value) makes the slot ineligible. Variable ids are globally
   unique (Sarek_typed_ast.fresh_var_id), so there is no shadowing to handle. *)
let uses_all_reducible (id : int) (cname : string) (body : texpr) : bool =
  let ok = ref true in
  let rec go (e : texpr) : unit =
    match e.te with
    | TEVar (_, i) when i = id -> ok := false (* bare use outside scrutinee *)
    | TEMatch (scrut, cases) when is_slot_var id scrut ->
        if not (List.exists (is_ctor_case cname) cases) then ok := false ;
        List.iter (fun (_, rhs) -> go rhs) cases
    | _ -> iter_children go e
  in
  go body ;
  !ok

(* Substitute [repl] for every occurrence of variable [pid]. Used to inline the
   payload-typed slot variable in place of a constructor pattern's bound
   variable, so no fresh `let` is introduced in what may be an expression
   position (Sarek treats `let` as statement-only in some contexts - see the
   note in Sarek_defunc.distribute_app). [repl] is always a bare variable read,
   so substitution never duplicates work. *)
let subst_var (pid : int) (repl : texpr) (e : texpr) : texpr =
  let rec go (e : texpr) : texpr =
    match e.te with TEVar (_, i) when i = pid -> repl | _ -> map_children go e
  in
  go e

(* Replace every `match <slot> with .. | C pat -> rhs | ..` by the C-arm: [rhs]
   with the bound pattern variable substituted by [accessor] (the slot re-read
   at the payload type) for a unary constructor, or [rhs] alone for a nullary
   one. Recurses into the kept arm so nested reads reduce too. *)
let reduce_matches (id : int) (cname : string)
    (accessor : (unit -> texpr) option) (body : texpr) : texpr =
  let rec go (e : texpr) : texpr =
    match e.te with
    | TEMatch (scrut, cases) when is_slot_var id scrut -> (
        let pat, rhs =
          match List.find_opt (is_ctor_case cname) cases with
          | Some (p, rhs) -> (
              match p.tpat with
              | TPConstr (_, _, arg) -> (arg, rhs)
              | _ -> (None, rhs))
          | None ->
              (* Guarded by uses_all_reducible; if a future eligibility-check
                 change breaks that invariant, fail loudly rather than
                 re-entering this same branch forever via [go e]. *)
              failwith
                "Sarek_tag_erasure.reduce_matches: no reducible arm for the \
                 slot's constructor (invariant violated: uses_all_reducible \
                 should have made this slot ineligible)"
        in
        let rhs' = go rhs in
        match (pat, accessor) with
        | Some {tpat = TPVar (_, pid); _}, Some mk -> subst_var pid (mk ()) rhs'
        | _ -> {rhs' with ty = e.ty})
    | _ -> map_children go e
  in
  go body

(* -------------------------------------------------------------------------- *)
(* Core rewrite                                                               *)
(* -------------------------------------------------------------------------- *)

let rewrite_body (body : texpr) : texpr =
  let rec rewrite (e : texpr) : texpr =
    match e.te with
    | TELet (name, id, value, body)
      when is_variant_ty value.ty
           && match value.te with TEConstr _ -> true | _ -> false -> (
        match value.te with
        | TEConstr (_tyname, cname, arg) when uses_all_reducible id cname body
          -> (
            match arg with
            | None ->
                (* nullary: drop the binding, reduce every match to its arm *)
                reduce_matches id cname None (rewrite body)
            | Some payload ->
                (* unary: retype the binding to the payload; each match arm
                   binds its pattern var to the (now payload-typed) slot. *)
                let payload' = rewrite payload in
                let accessor () = {payload' with te = TEVar (name, id)} in
                let body' =
                  reduce_matches id cname (Some accessor) (rewrite body)
                in
                {
                  te = TELet (name, id, payload', body');
                  ty = body'.ty;
                  te_loc = e.te_loc;
                })
        | _ -> map_children rewrite e)
    | _ -> map_children rewrite e
  in
  rewrite body

(* -------------------------------------------------------------------------- *)
(* Entry point                                                                *)
(* -------------------------------------------------------------------------- *)

let erase_tags (kernel : tkernel) : tkernel =
  let module_items' =
    List.map
      (function
        | TMFun (n, r, ps, b) -> TMFun (n, r, ps, rewrite_body b)
        | TMConst _ as it -> it)
      kernel.tkern_module_items
  in
  {
    kernel with
    tkern_module_items = module_items';
    tkern_body = rewrite_body kernel.tkern_body;
  }
