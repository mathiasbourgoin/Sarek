(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Monomorphization
 *
 * This module implements monomorphization for polymorphic Sarek kernels.
 * Polymorphic functions are specialized to concrete types at each call site.
 ******************************************************************************)

open Sarek_typed_ast
open Sarek_types

(** {1 Types} *)

(** A monomorphic instance of a polymorphic function *)
type mono_instance = {
  mi_name : string;  (** Original function name *)
  mi_mangled : string;  (** Mangled name for this instance *)
  mi_types : typ list;  (** Concrete types for type parameters *)
  mi_params : tparam list;  (** Specialized parameters *)
  mi_body : texpr;  (** Specialized body *)
}

(** Collection of all instances found during monomorphization *)
type mono_env = {
  (* Map from (function_name, type_list) to mangled name *)
  instances : (string * typ list, string) Hashtbl.t;
  (* Generated specialized functions *)
  specialized : mono_instance list ref;
  (* Counter for generating unique names *)
  counter : int ref;
}

(** {1 Type Utilities} *)

(** Check if a type contains any unresolved type variables *)
let rec has_type_vars (t : typ) : bool =
  match repr t with
  | TVar {contents = Unbound _} -> true
  | TVar {contents = Link t'} -> has_type_vars t'
  | TVec t -> has_type_vars t
  | TArr (t, _) -> has_type_vars t
  | TFun (args, ret) -> List.exists has_type_vars args || has_type_vars ret
  | TRecord (_, fields) -> List.exists (fun (_, t) -> has_type_vars t) fields
  | TVariant (_, constrs) ->
      List.exists
        (function _, None -> false | _, Some t -> has_type_vars t)
        constrs
  | TTuple ts -> List.exists has_type_vars ts
  | TPrim _ | TReg _ -> false

(** Normalize a type by following all links *)
let rec normalize_type (t : typ) : typ =
  match repr t with
  | TVar {contents = Link t'} -> normalize_type t'
  | TVec t -> TVec (normalize_type t)
  | TArr (t, m) -> TArr (normalize_type t, m)
  | TFun (args, ret) -> TFun (List.map normalize_type args, normalize_type ret)
  | TRecord (n, fields) ->
      TRecord (n, List.map (fun (f, t) -> (f, normalize_type t)) fields)
  | TVariant (n, constrs) ->
      TVariant
        (n, List.map (fun (c, t) -> (c, Option.map normalize_type t)) constrs)
  | TTuple ts -> TTuple (List.map normalize_type ts)
  | t -> t

(** Compare two types for equality (after normalization) *)
let types_equal (t1 : typ) (t2 : typ) : bool =
  let rec eq t1 t2 =
    match (normalize_type t1, normalize_type t2) with
    | TPrim p1, TPrim p2 -> p1 = p2
    | TReg r1, TReg r2 -> r1 = r2
    | TVec t1, TVec t2 -> eq t1 t2
    | TArr (t1, m1), TArr (t2, m2) -> m1 = m2 && eq t1 t2
    | TFun (a1, r1), TFun (a2, r2) ->
        List.length a1 = List.length a2 && List.for_all2 eq a1 a2 && eq r1 r2
    | TRecord (n1, f1), TRecord (n2, f2) ->
        n1 = n2
        && List.length f1 = List.length f2
        && List.for_all2
             (fun (fn1, t1) (fn2, t2) -> fn1 = fn2 && eq t1 t2)
             f1
             f2
    | TTuple ts1, TTuple ts2 ->
        List.length ts1 = List.length ts2 && List.for_all2 eq ts1 ts2
    | TVar {contents = Unbound (id1, _)}, TVar {contents = Unbound (id2, _)} ->
        id1 = id2
    | _, _ -> false
  in
  eq t1 t2

(** {1 Name Mangling} *)

(** Create a mangled name for a type *)
let rec mangle_type (t : typ) : string =
  match normalize_type t with
  | TPrim TUnit -> "u"
  | TPrim TBool -> "b"
  | TPrim TInt32 -> "i32"
  | TReg Int -> "i"
  | TReg Int64 -> "i64"
  | TReg Float32 -> "f32"
  | TReg Float64 -> "f64"
  | TReg Char -> "c"
  | TReg (Custom name) -> String.map (fun c -> if c = '.' then '_' else c) name
  | TVec t -> "v" ^ mangle_type t
  | TArr (t, _) -> "a" ^ mangle_type t
  | TFun (args, ret) ->
      "F" ^ String.concat "" (List.map mangle_type args) ^ "_" ^ mangle_type ret
  | TRecord (name, _) ->
      "R" ^ String.map (fun c -> if c = '.' then '_' else c) name
  | TVariant (name, _) ->
      "V" ^ String.map (fun c -> if c = '.' then '_' else c) name
  | TTuple ts ->
      "T"
      ^ string_of_int (List.length ts)
      ^ String.concat "" (List.map mangle_type ts)
  | TVar _ -> "X" (* Should not happen after monomorphization *)

(** Create a mangled name for a function with specific type arguments *)
let mangle_name (base_name : string) (types : typ list) : string =
  if types = [] then base_name
  else
    let type_suffix = String.concat "_" (List.map mangle_type types) in
    Printf.sprintf "%s__%s" base_name type_suffix

(** {1 Instance Collection} *)

(** Create a fresh monomorphization environment *)
let create_mono_env () : mono_env =
  {instances = Hashtbl.create 16; specialized = ref []; counter = ref 0}

(** Get or create a specialized instance name *)
let get_or_create_instance (env : mono_env) (name : string) (types : typ list) :
    string =
  let normalized = List.map normalize_type types in
  let key = (name, normalized) in
  match Hashtbl.find_opt env.instances key with
  | Some mangled -> mangled
  | None ->
      let mangled = mangle_name name normalized in
      Hashtbl.add env.instances key mangled ;
      mangled

(** {1 Type Substitution} *)

(** Substitution from type variable IDs to concrete types *)
type type_subst = (int * typ) list

(** Apply type substitution to a type *)
let rec apply_subst (subst : type_subst) (t : typ) : typ =
  match repr t with
  | TVar {contents = Unbound (id, _)} -> (
      match List.assoc_opt id subst with Some t' -> t' | None -> t)
  | TVar {contents = Link t'} -> apply_subst subst t'
  | TVec t -> TVec (apply_subst subst t)
  | TArr (t, m) -> TArr (apply_subst subst t, m)
  | TFun (args, ret) ->
      TFun (List.map (apply_subst subst) args, apply_subst subst ret)
  | TRecord (n, fields) ->
      TRecord (n, List.map (fun (f, t) -> (f, apply_subst subst t)) fields)
  | TVariant (n, constrs) ->
      TVariant
        ( n,
          List.map (fun (c, t) -> (c, Option.map (apply_subst subst) t)) constrs
        )
  | TTuple ts -> TTuple (List.map (apply_subst subst) ts)
  | t -> t

(** Apply type substitution to a typed expression *)
let rec apply_subst_expr (subst : type_subst) (expr : texpr) : texpr =
  let ty = apply_subst subst expr.ty in
  let te =
    match expr.te with
    | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _
    | TEDouble _ | TEGlobalRef _ | TENative _ | TEIntrinsicConst _ ->
        expr.te
    | TEVar (n, id) -> TEVar (n, id)
    | TEVecGet (v, i) ->
        TEVecGet (apply_subst_expr subst v, apply_subst_expr subst i)
    | TEVecSet (v, i, x) ->
        TEVecSet
          ( apply_subst_expr subst v,
            apply_subst_expr subst i,
            apply_subst_expr subst x )
    | TEArrGet (a, i) ->
        TEArrGet (apply_subst_expr subst a, apply_subst_expr subst i)
    | TEArrSet (a, i, x) ->
        TEArrSet
          ( apply_subst_expr subst a,
            apply_subst_expr subst i,
            apply_subst_expr subst x )
    | TEFieldGet (r, f, idx) -> TEFieldGet (apply_subst_expr subst r, f, idx)
    | TEFieldSet (r, f, idx, v) ->
        TEFieldSet (apply_subst_expr subst r, f, idx, apply_subst_expr subst v)
    | TEBinop (op, a, b) ->
        TEBinop (op, apply_subst_expr subst a, apply_subst_expr subst b)
    | TEUnop (op, e) -> TEUnop (op, apply_subst_expr subst e)
    | TEApp (fn, args) ->
        TEApp (apply_subst_expr subst fn, List.map (apply_subst_expr subst) args)
    | TELet (n, id, v, b) ->
        TELet (n, id, apply_subst_expr subst v, apply_subst_expr subst b)
    | TELetMut (n, id, v, b) ->
        TELetMut (n, id, apply_subst_expr subst v, apply_subst_expr subst b)
    | TELetShared (n, id, t, size, b) ->
        TELetShared
          ( n,
            id,
            apply_subst subst t,
            Option.map (apply_subst_expr subst) size,
            apply_subst_expr subst b )
    | TEIf (c, t, e) ->
        TEIf
          ( apply_subst_expr subst c,
            apply_subst_expr subst t,
            Option.map (apply_subst_expr subst) e )
    | TEFor (v, id, lo, hi, dir, body) ->
        TEFor
          ( v,
            id,
            apply_subst_expr subst lo,
            apply_subst_expr subst hi,
            dir,
            apply_subst_expr subst body )
    | TEWhile (c, b) ->
        TEWhile (apply_subst_expr subst c, apply_subst_expr subst b)
    | TESeq es -> TESeq (List.map (apply_subst_expr subst) es)
    | TEMatch (s, cases) ->
        TEMatch
          ( apply_subst_expr subst s,
            List.map
              (fun (p, b) ->
                (apply_subst_pat subst p, apply_subst_expr subst b))
              cases )
    | TETuple es -> TETuple (List.map (apply_subst_expr subst) es)
    | TERecord (n, fields) ->
        TERecord
          (n, List.map (fun (f, e) -> (f, apply_subst_expr subst e)) fields)
    | TEConstr (tn, cn, arg) ->
        TEConstr (tn, cn, Option.map (apply_subst_expr subst) arg)
    | TEAssign (n, id, v) -> TEAssign (n, id, apply_subst_expr subst v)
    | TEReturn e -> TEReturn (apply_subst_expr subst e)
    | TECreateArray (size, t, m) ->
        TECreateArray (apply_subst_expr subst size, apply_subst subst t, m)
    | TESuperstep (n, d, step, cont) ->
        TESuperstep
          (n, d, apply_subst_expr subst step, apply_subst_expr subst cont)
    | TEPragma (opts, body) -> TEPragma (opts, apply_subst_expr subst body)
    | TEOpen (path, body) -> TEOpen (path, apply_subst_expr subst body)
    | TEIntrinsicFun (r, c, args) ->
        TEIntrinsicFun (r, c, List.map (apply_subst_expr subst) args)
    | TELetRec (name, id, params, fn_body, cont) ->
        let params' =
          List.map
            (fun p -> {p with tparam_type = apply_subst subst p.tparam_type})
            params
        in
        TELetRec
          ( name,
            id,
            params',
            apply_subst_expr subst fn_body,
            apply_subst_expr subst cont )
  in
  {te; ty; te_loc = expr.te_loc}

and apply_subst_pat (subst : type_subst) (pat : tpattern) : tpattern =
  let tpat_ty = apply_subst subst pat.tpat_ty in
  let tpat =
    match pat.tpat with
    | TPAny -> TPAny
    | TPVar (n, id) -> TPVar (n, id)
    | TPConstr (tn, cn, arg) ->
        TPConstr (tn, cn, Option.map (apply_subst_pat subst) arg)
    | TPTuple ps -> TPTuple (List.map (apply_subst_pat subst) ps)
  in
  {tpat; tpat_ty; tpat_loc = pat.tpat_loc}

(** {1 Monomorphization Pass} *)

(** Collect all polymorphic function definitions and their call sites *)
let collect_poly_functions (kernel : tkernel) :
    (string * bool * tparam list * texpr) list =
  (* For now, module functions are candidates for polymorphism *)
  List.filter_map
    (function
      | TMFun (name, is_rec, params, body) ->
          (* Check if any parameter has type variables *)
          let is_poly =
            List.exists (fun p -> has_type_vars p.tparam_type) params
          in
          Sarek_debug.log "collect_poly: %s is_poly=%b" name is_poly ;
          if is_poly then Some (name, is_rec, params, body) else None
      | TMConst _ -> None)
    kernel.tkern_module_items

(** Collect call sites with their concrete types *)
let rec collect_call_sites (poly_names : string list) (expr : texpr) :
    (string * typ list) list =
  match expr.te with
  | TEApp ({te = TEVar (name, _); _}, args) when List.mem name poly_names ->
      (* Found a call to a polymorphic function *)
      let arg_types = List.map (fun a -> normalize_type a.ty) args in
      (name, arg_types) :: List.concat_map (collect_call_sites poly_names) args
  | TEApp (fn, args) ->
      collect_call_sites poly_names fn
      @ List.concat_map (collect_call_sites poly_names) args
  | TELet (_, _, v, b) | TELetMut (_, _, v, b) ->
      collect_call_sites poly_names v @ collect_call_sites poly_names b
  | TELetShared (_, _, _, size_opt, b) ->
      (match size_opt with
        | Some s -> collect_call_sites poly_names s
        | None -> [])
      @ collect_call_sites poly_names b
  | TEIf (c, t, e) -> (
      collect_call_sites poly_names c
      @ collect_call_sites poly_names t
      @ match e with Some x -> collect_call_sites poly_names x | None -> [])
  | TEFor (_, _, lo, hi, _, body) ->
      collect_call_sites poly_names lo
      @ collect_call_sites poly_names hi
      @ collect_call_sites poly_names body
  | TEWhile (c, b) ->
      collect_call_sites poly_names c @ collect_call_sites poly_names b
  | TESeq es -> List.concat_map (collect_call_sites poly_names) es
  | TEBinop (_, a, b) ->
      collect_call_sites poly_names a @ collect_call_sites poly_names b
  | TEUnop (_, e) -> collect_call_sites poly_names e
  | TEMatch (s, cases) ->
      collect_call_sites poly_names s
      @ List.concat_map (fun (_, b) -> collect_call_sites poly_names b) cases
  | TETuple es -> List.concat_map (collect_call_sites poly_names) es
  | TERecord (_, fields) ->
      List.concat_map (fun (_, e) -> collect_call_sites poly_names e) fields
  | TEConstr (_, _, arg) -> (
      match arg with Some e -> collect_call_sites poly_names e | None -> [])
  | TEVecGet (v, i) | TEArrGet (v, i) ->
      collect_call_sites poly_names v @ collect_call_sites poly_names i
  | TEVecSet (v, i, x) | TEArrSet (v, i, x) ->
      collect_call_sites poly_names v
      @ collect_call_sites poly_names i
      @ collect_call_sites poly_names x
  | TEFieldGet (r, _, _) -> collect_call_sites poly_names r
  | TEFieldSet (r, _, _, v) ->
      collect_call_sites poly_names r @ collect_call_sites poly_names v
  | TEAssign (_, _, v) -> collect_call_sites poly_names v
  | TEReturn e -> collect_call_sites poly_names e
  | TESuperstep (_, _, step, cont) ->
      collect_call_sites poly_names step @ collect_call_sites poly_names cont
  | TEPragma (_, body) -> collect_call_sites poly_names body
  | TEOpen (_, body) -> collect_call_sites poly_names body
  | TECreateArray (size, _, _) -> collect_call_sites poly_names size
  | TEIntrinsicFun (_, _, args) ->
      List.concat_map (collect_call_sites poly_names) args
  | _ -> []

(** Rewrite calls to polymorphic functions with mangled names *)
let rec rewrite_calls (env : mono_env) (poly_names : string list) (expr : texpr)
    : texpr =
  match expr.te with
  | TEApp (({te = TEVar (name, id); ty = _fn_ty; te_loc = _fn_loc} as fn), args)
    when List.mem name poly_names ->
      let arg_types = List.map (fun a -> normalize_type a.ty) args in
      let mangled = get_or_create_instance env name arg_types in
      let new_fn = {fn with te = TEVar (mangled, id)} in
      let new_args = List.map (rewrite_calls env poly_names) args in
      {expr with te = TEApp (new_fn, new_args)}
  | TEApp (fn, args) ->
      {
        expr with
        te =
          TEApp
            ( rewrite_calls env poly_names fn,
              List.map (rewrite_calls env poly_names) args );
      }
  | TELet (n, id, v, b) ->
      {
        expr with
        te =
          TELet
            ( n,
              id,
              rewrite_calls env poly_names v,
              rewrite_calls env poly_names b );
      }
  | TELetMut (n, id, v, b) ->
      {
        expr with
        te =
          TELetMut
            ( n,
              id,
              rewrite_calls env poly_names v,
              rewrite_calls env poly_names b );
      }
  | TELetShared (n, id, t, size, b) ->
      {
        expr with
        te =
          TELetShared
            ( n,
              id,
              t,
              Option.map (rewrite_calls env poly_names) size,
              rewrite_calls env poly_names b );
      }
  | TEIf (c, t, e) ->
      {
        expr with
        te =
          TEIf
            ( rewrite_calls env poly_names c,
              rewrite_calls env poly_names t,
              Option.map (rewrite_calls env poly_names) e );
      }
  | TEFor (v, id, lo, hi, dir, body) ->
      {
        expr with
        te =
          TEFor
            ( v,
              id,
              rewrite_calls env poly_names lo,
              rewrite_calls env poly_names hi,
              dir,
              rewrite_calls env poly_names body );
      }
  | TEWhile (c, b) ->
      {
        expr with
        te =
          TEWhile
            (rewrite_calls env poly_names c, rewrite_calls env poly_names b);
      }
  | TESeq es ->
      {expr with te = TESeq (List.map (rewrite_calls env poly_names) es)}
  | TEBinop (op, a, b) ->
      {
        expr with
        te =
          TEBinop
            (op, rewrite_calls env poly_names a, rewrite_calls env poly_names b);
      }
  | TEUnop (op, e) ->
      {expr with te = TEUnop (op, rewrite_calls env poly_names e)}
  | TEMatch (s, cases) ->
      {
        expr with
        te =
          TEMatch
            ( rewrite_calls env poly_names s,
              List.map (fun (p, b) -> (p, rewrite_calls env poly_names b)) cases
            );
      }
  | TETuple es ->
      {expr with te = TETuple (List.map (rewrite_calls env poly_names) es)}
  | TERecord (n, fields) ->
      {
        expr with
        te =
          TERecord
            ( n,
              List.map
                (fun (f, e) -> (f, rewrite_calls env poly_names e))
                fields );
      }
  | TEConstr (tn, cn, arg) ->
      {
        expr with
        te = TEConstr (tn, cn, Option.map (rewrite_calls env poly_names) arg);
      }
  | TEVecGet (v, i) ->
      {
        expr with
        te =
          TEVecGet
            (rewrite_calls env poly_names v, rewrite_calls env poly_names i);
      }
  | TEVecSet (v, i, x) ->
      {
        expr with
        te =
          TEVecSet
            ( rewrite_calls env poly_names v,
              rewrite_calls env poly_names i,
              rewrite_calls env poly_names x );
      }
  | TEArrGet (a, i) ->
      {
        expr with
        te =
          TEArrGet
            (rewrite_calls env poly_names a, rewrite_calls env poly_names i);
      }
  | TEArrSet (a, i, x) ->
      {
        expr with
        te =
          TEArrSet
            ( rewrite_calls env poly_names a,
              rewrite_calls env poly_names i,
              rewrite_calls env poly_names x );
      }
  | TEFieldGet (r, f, idx) ->
      {expr with te = TEFieldGet (rewrite_calls env poly_names r, f, idx)}
  | TEFieldSet (r, f, idx, v) ->
      {
        expr with
        te =
          TEFieldSet
            ( rewrite_calls env poly_names r,
              f,
              idx,
              rewrite_calls env poly_names v );
      }
  | TEAssign (n, id, v) ->
      {expr with te = TEAssign (n, id, rewrite_calls env poly_names v)}
  | TEReturn e -> {expr with te = TEReturn (rewrite_calls env poly_names e)}
  | TECreateArray (size, t, m) ->
      {expr with te = TECreateArray (rewrite_calls env poly_names size, t, m)}
  | TESuperstep (n, d, step, cont) ->
      {
        expr with
        te =
          TESuperstep
            ( n,
              d,
              rewrite_calls env poly_names step,
              rewrite_calls env poly_names cont );
      }
  | TEPragma (opts, body) ->
      {expr with te = TEPragma (opts, rewrite_calls env poly_names body)}
  | TEOpen (path, body) ->
      {expr with te = TEOpen (path, rewrite_calls env poly_names body)}
  | TEIntrinsicFun (r, c, args) ->
      {
        expr with
        te = TEIntrinsicFun (r, c, List.map (rewrite_calls env poly_names) args);
      }
  | _ -> expr

(** Main monomorphization entry point *)
let monomorphize (kernel : tkernel) : tkernel =
  Sarek_debug.log_enter "monomorphize" ;
  (* 1. Find polymorphic functions *)
  let poly_funs = collect_poly_functions kernel in
  Sarek_debug.log "poly_funs count=%d" (List.length poly_funs) ;
  if poly_funs = [] then (
    Sarek_debug.log_exit "monomorphize (no poly funs)" ;
    kernel (* No polymorphic functions *))
  else
    let poly_names = List.map (fun (n, _, _, _) -> n) poly_funs in
    let env = create_mono_env () in

    (* 2. Collect all call sites in kernel body and module items *)
    let body_sites = collect_call_sites poly_names kernel.tkern_body in
    let module_sites =
      List.concat_map
        (function
          | TMFun (_, _, _, body) -> collect_call_sites poly_names body
          | TMConst (_, _, _, body) -> collect_call_sites poly_names body)
        kernel.tkern_module_items
    in

    let all_sites = body_sites @ module_sites in

    (* 3. Create instances for each unique (name, types) pair *)
    List.iter
      (fun (name, types) ->
        let _ = get_or_create_instance env name types in
        ())
      all_sites ;

    (* 4. Generate specialized function copies *)
    let specialized_funs =
      Hashtbl.fold
        (fun (name, types) mangled acc ->
          match List.find_opt (fun (n, _, _, _) -> n = name) poly_funs with
          | None -> acc
          | Some (_, is_rec, params, body) ->
              (* Build substitution by extracting type variables from param types
                 and matching them with the corresponding concrete types.
                 This handles nested type variables like TVec (TVar 'a). *)
              let rec extract_tvar_id ty =
                match repr ty with
                | TVar {contents = Unbound (id, _)} -> Some id
                | TVec t | TArr (t, _) -> extract_tvar_id t
                | _ -> None
              in
              let rec extract_concrete_elem ty =
                match repr ty with
                | TVec t | TArr (t, _) -> extract_concrete_elem t
                | t -> t
              in
              let subst =
                List.mapi
                  (fun i p ->
                    match extract_tvar_id p.tparam_type with
                    | Some id ->
                        let concrete =
                          extract_concrete_elem (List.nth types i)
                        in
                        Some (id, concrete)
                    | None -> None)
                  params
                |> List.filter_map Fun.id
              in

              let spec_params =
                List.map
                  (fun p ->
                    {p with tparam_type = apply_subst subst p.tparam_type})
                  params
              in
              let spec_body = apply_subst_expr subst body in

              TMFun (mangled, is_rec, spec_params, spec_body) :: acc)
        env.instances
        []
    in

    (* 5. Rewrite kernel body and non-polymorphic module items *)
    let new_body = rewrite_calls env poly_names kernel.tkern_body in
    let new_items =
      List.filter_map
        (function
          | TMFun (name, _, _, _) when List.mem name poly_names ->
              None (* Remove original polymorphic function *)
          | TMFun (name, is_rec, params, body) ->
              Some
                (TMFun (name, is_rec, params, rewrite_calls env poly_names body))
          | TMConst (name, id, ty, body) ->
              Some (TMConst (name, id, ty, rewrite_calls env poly_names body)))
        kernel.tkern_module_items
    in

    (* Count how many external items remain after removing polymorphic ones.
       The specialized functions are NOT external - they're generated inline. *)
    let original_external_count = kernel.tkern_external_item_count in
    let removed_external_count =
      List.length
        (List.filter
           (fun (name, _, _, _) -> List.mem name poly_names)
           (List.filter_map
              (function TMFun (n, r, p, b) -> Some (n, r, p, b) | _ -> None)
              (List.filteri
                 (fun i _ -> i < original_external_count)
                 kernel.tkern_module_items)))
    in
    let new_external_count = original_external_count - removed_external_count in

    {
      kernel with
      tkern_module_items = new_items @ specialized_funs;
      tkern_external_item_count = new_external_count;
      tkern_body = new_body;
    }
