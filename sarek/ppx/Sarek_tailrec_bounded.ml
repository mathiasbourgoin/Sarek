(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Bounded Recursion Inlining
 *
 * This module handles inlining of non-tail-recursive functions with known
 * bounded depth. Currently disabled as termination analysis is too imprecise.
 ******************************************************************************)

open Sarek_typed_ast

(** {1 Bounded Recursion Inlining} *)

(** Inline bounded recursion up to a maximum depth. This is used for
    non-tail-recursive functions with known bounds.

    Note: max_depth is the value from the termination condition (e.g., if depth
    >= 4). We inline one extra level to ensure the base case is properly
    expanded. *)
let inline_bounded_recursion (fname : string) (params : tparam list)
    (body : texpr) (max_depth : int) (_loc : Sarek_ast.loc) : texpr =
  (* Add some buffer to ensure base cases are fully expanded *)
  let effective_depth = max_depth + 2 in
  let rec inline depth expr =
    if depth > effective_depth then
      (* Exceeded depth - this should not happen if depth detection is correct.
         Return the expression unchanged as a safety fallback. *)
      expr
    else
      match expr.te with
      | TEApp (fn, args) when Sarek_tailrec_analysis.is_self_call fname fn ->
          (* Inline: substitute params with args and recurse *)
          let subst = List.map2 (fun p a -> (p.tparam_name, a)) params args in
          let inlined = substitute subst body in
          inline (depth + 1) inlined
      | TEIf (c, t, e) ->
          {
            expr with
            te =
              TEIf (inline depth c, inline depth t, Option.map (inline depth) e);
          }
      | TELet (n, id, v, b) ->
          {expr with te = TELet (n, id, inline depth v, inline depth b)}
      | TELetMut (n, id, v, b) ->
          {expr with te = TELetMut (n, id, inline depth v, inline depth b)}
      | TESeq es -> {expr with te = TESeq (List.map (inline depth) es)}
      | TEApp (fn, args) ->
          {expr with te = TEApp (inline depth fn, List.map (inline depth) args)}
      | TEBinop (op, a, b) ->
          {expr with te = TEBinop (op, inline depth a, inline depth b)}
      | TEUnop (op, e) -> {expr with te = TEUnop (op, inline depth e)}
      | _ -> expr
  and substitute subst expr =
    match expr.te with
    | TEVar (name, _id) -> (
        match List.assoc_opt name subst with
        | Some replacement -> replacement
        | None -> expr)
    | TELet (n, id, v, b) ->
        let subst' = List.filter (fun (name, _) -> name <> n) subst in
        {expr with te = TELet (n, id, substitute subst v, substitute subst' b)}
    | TELetMut (n, id, v, b) ->
        let subst' = List.filter (fun (name, _) -> name <> n) subst in
        {
          expr with
          te = TELetMut (n, id, substitute subst v, substitute subst' b);
        }
    | TEIf (c, t, e) ->
        {
          expr with
          te =
            TEIf
              ( substitute subst c,
                substitute subst t,
                Option.map (substitute subst) e );
        }
    | TESeq es -> {expr with te = TESeq (List.map (substitute subst) es)}
    | TEApp (fn, args) ->
        {
          expr with
          te = TEApp (substitute subst fn, List.map (substitute subst) args);
        }
    | TEBinop (op, a, b) ->
        {expr with te = TEBinop (op, substitute subst a, substitute subst b)}
    | TEUnop (op, e) -> {expr with te = TEUnop (op, substitute subst e)}
    | TEFor (v, id, lo, hi, dir, body_expr) ->
        let subst' = List.filter (fun (name, _) -> name <> v) subst in
        {
          expr with
          te =
            TEFor
              ( v,
                id,
                substitute subst lo,
                substitute subst hi,
                dir,
                substitute subst' body_expr );
        }
    | _ -> expr
  in

  inline 0 body
