(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Type Schemes for Let-Polymorphism
 *
 * Extends the type system with type schemes (forall-quantified types)
 * to support polymorphic local functions in Sarek kernels.
 *
 * This is a separate module to avoid modifying Sarek_types.ml directly.
 ******************************************************************************)

open Sarek_types

(** {1 Type Schemes} *)

(** A type scheme represents a polymorphic type with quantified type variables.
    For example, the identity function has scheme: forall a. a -> a *)
type scheme = {
  quantified : int list;
      (** IDs of generalized (forall-bound) type variables *)
  body : typ;  (** The type body with those variables *)
}

(** Create a monomorphic scheme (no quantified variables) *)
let mono (t : typ) : scheme = {quantified = []; body = t}

(** {1 Free Type Variables} *)

(** Collect all free (unbound) type variable IDs in a type *)
let rec free_tvars (t : typ) : int list =
  match repr t with
  | TVar {contents = Unbound (id, _)} -> [id]
  | TVar {contents = Link t'} -> free_tvars t'
  | TVec t -> free_tvars t
  | TArr (t, _) -> free_tvars t
  | TFun (args, ret) -> List.concat_map free_tvars args @ free_tvars ret
  | TRecord (_, fields) -> List.concat_map (fun (_, t) -> free_tvars t) fields
  | TVariant (_, constrs) ->
      List.concat_map
        (function _, None -> [] | _, Some t -> free_tvars t)
        constrs
  | TTuple ts -> List.concat_map free_tvars ts
  | TPrim _ | TReg _ -> []

(** Remove duplicates from a list *)
let unique (lst : int list) : int list = List.sort_uniq compare lst

(** {1 Generalization} *)

(** Deep copy a type, creating fresh type variables with NEW ids for those that
    will be quantified. This ensures the scheme's body is completely independent
    of unification on the original or instantiated types.

    CONSTRAINT PROPAGATION. [Sarek_types] keeps two registries keyed by tvar ID
    — "this tvar came from a bare float literal" and "this tvar stood where a
    numeric type was required, so it may never become float16". Minting a fresh
    id silently DROPS both, which turns generalization into a hole in the type
    system: the body of [let[@sarek.module] twice (x : 'a) : 'a = x +. x]
    records the original [x] as numeric-required, but its generalized copy did
    not, so a call site could unify the fresh variable with a float16 element
    and get half arithmetic past the guard.

    Confirmed by construction: that exact helper applied to a [float16 vector]
    load produced NO Sarek diagnostic, while the identical shape at float32
    compiled cleanly — i.e. the two differed only in the guard that should have
    fired and did not. Both registries are therefore carried onto every fresh
    variable here and in {!instantiate}. *)
let copy_for_scheme (level : int) (t : typ) : typ * int list =
  (* Map from old tvar id to (new_id, new tvar ref) *)
  let tvar_map : (int, int * typ) Hashtbl.t = Hashtbl.create 8 in
  let quantified = ref [] in

  (* Carry the ID-keyed constraints of [old_id] onto [fresh]. Without this a
     fresh id is unconstrained and generalization loses the property. *)
  let inherit_constraints old_id fresh =
    if is_float_literal_id old_id then register_float_literal fresh ;
    if is_numeric_required_id old_id then register_numeric_required fresh
  in
  let rec copy t =
    match repr t with
    | TVar {contents = Unbound (id, l)} when l > level -> (
        (* This variable should be quantified - create fresh copy with NEW id *)
        match Hashtbl.find_opt tvar_map id with
        | Some (_, tv) -> tv
        | None ->
            let new_id = fresh_tvar_id () in
            let fresh = TVar (ref (Unbound (new_id, level))) in
            Hashtbl.add tvar_map id (new_id, fresh) ;
            inherit_constraints id fresh ;
            quantified := new_id :: !quantified ;
            fresh)
    | TVar {contents = Unbound (id, l)} -> (
        (* Not quantified - keep as-is but create fresh ref to isolate *)
        match Hashtbl.find_opt tvar_map id with
        | Some (_, tv) -> tv
        | None ->
            let new_id = fresh_tvar_id () in
            let fresh = TVar (ref (Unbound (new_id, l))) in
            Hashtbl.add tvar_map id (new_id, fresh) ;
            inherit_constraints id fresh ;
            fresh)
    | TVar {contents = Link t'} -> copy t'
    | TVec t -> TVec (copy t)
    | TArr (t, m) -> TArr (copy t, m)
    | TFun (args, ret) -> TFun (List.map copy args, copy ret)
    | TRecord (n, fields) ->
        TRecord (n, List.map (fun (f, t) -> (f, copy t)) fields)
    | TVariant (n, constrs) ->
        TVariant (n, List.map (fun (c, t) -> (c, Option.map copy t)) constrs)
    | TTuple ts -> TTuple (List.map copy ts)
    | t -> t
  in
  let copied = copy t in
  (copied, !quantified)

(** Generalize a type at a given level. Type variables at levels greater than
    the given level are quantified.

    @param level The current binding level (from the environment)
    @param t The type to generalize
    @return A type scheme with free variables at higher levels quantified *)
let generalize (level : int) (t : typ) : scheme =
  let body, quantified = copy_for_scheme level t in
  {quantified; body}

(** {1 Instantiation} *)

(** Instantiate a type scheme by replacing quantified variables with fresh ones.

    Each fresh variable inherits the quantified variable's ID-keyed constraints
    (see {!copy_for_scheme}); this is the second half of the propagation, and
    the one a CALL SITE depends on. Skipping it would let [twice a_f16.(tid)]
    unify the instance with float16 even though the helper's body required a
    numeric type.

    @param s The type scheme to instantiate
    @return A fresh type with all quantified variables replaced *)
let instantiate (s : scheme) : typ =
  if s.quantified = [] then s.body
  else begin
    (* Create fresh type variables for each quantified variable, carrying the
       constraints attached to the quantified id. *)
    let subst =
      List.map
        (fun id ->
          let fresh = fresh_tvar () in
          if is_float_literal_id id then register_float_literal fresh ;
          if is_numeric_required_id id then register_numeric_required fresh ;
          (id, fresh))
        s.quantified
    in

    let rec inst t =
      match repr t with
      | TVar {contents = Unbound (id, _)} -> (
          match List.assoc_opt id subst with Some fresh -> fresh | None -> t)
      | TVar {contents = Link t'} -> inst t'
      | TVec t -> TVec (inst t)
      | TArr (t, m) -> TArr (inst t, m)
      | TFun (args, ret) -> TFun (List.map inst args, inst ret)
      | TRecord (n, fields) ->
          TRecord (n, List.map (fun (f, t) -> (f, inst t)) fields)
      | TVariant (n, constrs) ->
          TVariant (n, List.map (fun (c, t) -> (c, Option.map inst t)) constrs)
      | TTuple ts -> TTuple (List.map inst ts)
      | t -> t
    in
    inst s.body
  end

(** {1 Pretty Printing} *)

let pp_scheme fmt (s : scheme) =
  if s.quantified = [] then pp_typ fmt s.body
  else begin
    Format.fprintf fmt "forall" ;
    List.iter (fun id -> Format.fprintf fmt " 't%d" id) s.quantified ;
    Format.fprintf fmt ". %a" pp_typ s.body
  end

let scheme_to_string (s : scheme) : string = Format.asprintf "%a" pp_scheme s

(** {1 Scheme Utilities} *)

(** Check if a scheme is monomorphic (no quantified variables) *)
let is_mono (s : scheme) : bool = s.quantified = []

(** Check if a scheme is polymorphic (has quantified variables) *)
let is_poly (s : scheme) : bool = s.quantified <> []

(** Get the arity of a function scheme *)
let function_arity (s : scheme) : int option =
  match repr s.body with TFun (args, _) -> Some (List.length args) | _ -> None

(** Check if two schemes are equivalent (up to alpha-renaming) *)
let schemes_equivalent (s1 : scheme) (s2 : scheme) : bool =
  (* Simple check: same number of quantified vars and bodies unify *)
  List.length s1.quantified = List.length s2.quantified
  &&
  let t1 = instantiate s1 in
  let t2 = instantiate s2 in
  match unify t1 t2 with Ok () -> true | Error _ -> false
