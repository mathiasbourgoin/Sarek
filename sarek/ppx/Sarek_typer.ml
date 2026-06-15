(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * Type Inference Module
 * =====================
 *
 * This module implements type inference for Sarek kernels using constraint-based
 * Hindley-Milner style inference with let-polymorphism.
 *
 * Organization:
 * - Type validators (Result-returning with error messages)
 * - Type inference helpers (literals, operators, memory access, etc.)
 * - Main inference function [infer]
 * - Pattern matching and auxiliary functions
 *
 * See also:
 * - Sarek_types.ml for type predicates (bool-returning)
 * - Sarek_error.ml for error type definitions
 * - Sarek_scheme.ml for type generalization and instantiation
 ******************************************************************************)

open Sarek_ast
open Sarek_types
open Sarek_env
open Sarek_error
open Sarek_typed_ast
open Sarek_scheme

(** {1 Type Unification with Error Reporting} *)

(** Unify two types and convert unification errors to typed errors with location
*)
let unify_or_error t1 t2 loc =
  match unify t1 t2 with
  | Ok () -> Ok ()
  | Error (Cannot_unify (t1, t2)) -> Error [Cannot_unify (t1, t2, loc)]
  | Error (Occurs_check (_, t)) -> Error [Recursive_type (t, loc)]

(** {1 Type Validators}

    Result-returning validators for type checking with error messages.

    These functions validate types and return detailed error messages on
    failure. For simple boolean checks without errors, use predicates from
    [Sarek_types]:
    - [Sarek_types.is_numeric]
    - [Sarek_types.is_integer]
    - [Sarek_types.is_boolean]
    - [Sarek_types.is_float]
    - [Sarek_types.is_tvar] *)

(** Check that a type is numeric (int32, int64, float32, float64).

    @param t The type to check
    @param loc Source location for error reporting
    @return Ok () if numeric, Error with type mismatch otherwise

    See also: {!Sarek_types.is_numeric} for bool predicate version. *)
let check_numeric t loc =
  match repr t with
  | TPrim TInt32 -> Ok ()
  | TReg (Int64 | Int | Float32 | Float64) -> Ok ()
  | TVar _ -> Ok () (* Will be constrained later *)
  | t -> Error [Type_mismatch {expected = t_int32; got = t; loc}]

(** Check that a type is integer (int32, int64, int).

    @param t The type to check
    @param loc Source location for error reporting
    @return Ok () if integer, Error with type mismatch otherwise

    See also: {!Sarek_types.is_integer} for bool predicate version. *)
let check_integer t loc =
  match repr t with
  | TPrim TInt32 -> Ok ()
  | TReg (Int64 | Int) -> Ok ()
  | TVar _ -> Ok ()
  | t -> Error [Type_mismatch {expected = t_int32; got = t; loc}]

(** Check that a type is boolean.

    @param t The type to check
    @param loc Source location for error reporting
    @return Ok () if boolean, Error with type mismatch otherwise

    See also: {!Sarek_types.is_boolean} for bool predicate version. *)
let check_boolean t loc =
  match repr t with
  | TPrim TBool -> Ok ()
  | TVar _ -> unify_or_error t t_bool loc
  | t -> Error [Type_mismatch {expected = t_bool; got = t; loc}]

(** Type variable context for tracking named type variables like 'a, 'b *)
module TvarCtx = Map.Make (String)

type tvar_ctx = {
  tvars : typ TvarCtx.t ref;
  level : int;  (** Level at which to create new type variables *)
}

(** Create a fresh type variable context at a given level *)
let fresh_tvar_ctx ?(level = 0) () : tvar_ctx =
  {tvars = ref TvarCtx.empty; level}

(** Convert a parsed type expression using the current typing environment and a
    type variable context for polymorphic type variables *)
let rec type_of_type_expr_ctx env (ctx : tvar_ctx) te =
  match te with
  | Sarek_ast.TEVar name -> (
      (* Named type variable like 'a - look up or create at the context's level *)
      match TvarCtx.find_opt name !(ctx.tvars) with
      | Some tv -> tv
      | None ->
          let tv = fresh_tvar ~level:ctx.level () in
          ctx.tvars := TvarCtx.add name tv !(ctx.tvars) ;
          tv)
  | Sarek_ast.TEConstr ("vector", [elem]) ->
      TVec (type_of_type_expr_ctx env ctx elem)
  | Sarek_ast.TEConstr (name, [elem])
    when String.ends_with ~suffix:"vector" name ->
      TVec (type_of_type_expr_ctx env ctx elem)
  | Sarek_ast.TEConstr ("array", [elem]) ->
      TArr (type_of_type_expr_ctx env ctx elem, Local)
  | Sarek_ast.TEArrow (a, b) ->
      TFun ([type_of_type_expr_ctx env ctx a], type_of_type_expr_ctx env ctx b)
  | Sarek_ast.TETuple ts -> TTuple (List.map (type_of_type_expr_ctx env ctx) ts)
  | Sarek_ast.TEConstr (name, _) -> (
      match find_type name env with
      | Some (TIRecord {ti_fields; ti_name}) ->
          TRecord (ti_name, List.map (fun (f, t, _) -> (f, t)) ti_fields)
      | Some (TIVariant {ti_constrs; ti_name}) -> TVariant (ti_name, ti_constrs)
      | None -> type_of_type_expr te)

(** Convert a parsed type expression using the current typing environment.
    Creates fresh type variables for each TEVar - use type_of_type_expr_ctx when
    type variable names need to be preserved across multiple types. *)
let type_of_type_expr_env env te =
  type_of_type_expr_ctx env (fresh_tvar_ctx ~level:env.current_level ()) te

(** Infer type of a binary operation *)
let infer_binop op t1 t2 loc =
  match op with
  | Add | Sub | Mul | Div ->
      (* Numeric ops: both operands same type, result same type *)
      let* () = unify_or_error t1 t2 loc in
      let* () = check_numeric t1 loc in
      Ok t1
  | Mod ->
      (* Integer only *)
      let* () = unify_or_error t1 t2 loc in
      let* () = check_integer t1 loc in
      Ok t1
  | Eq | Ne ->
      (* Equality: both same type, result bool *)
      let* () = unify_or_error t1 t2 loc in
      Ok t_bool
  | Lt | Le | Gt | Ge ->
      (* Comparison: both same numeric type, result bool *)
      let* () = unify_or_error t1 t2 loc in
      let* () = check_numeric t1 loc in
      Ok t_bool
  | And | Or ->
      (* Boolean ops *)
      let* () = check_boolean t1 loc in
      let* () = check_boolean t2 loc in
      Ok t_bool
  | Land | Lor | Lxor | Lsl | Lsr | Asr ->
      (* Bitwise ops: integer only *)
      let* () = unify_or_error t1 t2 loc in
      let* () = check_integer t1 loc in
      Ok t1

(** Infer type of a unary operation *)
let infer_unop op t loc =
  match op with
  | Neg ->
      let* () = check_numeric t loc in
      Ok t
  | Not ->
      let* () = check_boolean t loc in
      Ok t_bool
  | Lnot ->
      let* () = check_integer t loc in
      Ok t

(** {1 Type Inference Helpers}

    These helpers partition the main [infer] function by expression category. *)

(** Infer type of literal expressions (unit, bool, int, float constants). *)
let infer_literal (loc : Sarek_ast.loc) (expr : expr_desc) : texpr result =
  match expr with
  | EUnit -> Ok (mk_texpr TEUnit t_unit loc)
  | EBool b -> Ok (mk_texpr (TEBool b) t_bool loc)
  | EInt i -> Ok (mk_texpr (TEInt i) t_int32 loc)
  | EInt32 i -> Ok (mk_texpr (TEInt32 i) t_int32 loc)
  | EInt64 i -> Ok (mk_texpr (TEInt64 i) t_int64 loc)
  | EFloat f -> Ok (mk_texpr (TEFloat f) t_float32 loc)
  | EDouble f -> Ok (mk_texpr (TEDouble f) t_float64 loc)
  | _ ->
      error
        (Parse_error ("infer_literal called with non-literal expression", loc))

(** Infer type of binary and unary operations (arithmetic, logical, bitwise). *)
let infer_binop_unop ~infer (env : t) (loc : Sarek_ast.loc) (expr : expr_desc) :
    (texpr * t) result =
  match expr with
  | EBinop (op, e1, e2) ->
      let* t1, env = infer env e1 in
      let* t2, env = infer env e2 in
      let* result_ty = infer_binop op t1.ty t2.ty loc in
      Ok (mk_texpr (TEBinop (op, t1, t2)) result_ty loc, env)
  | EUnop (op, e) ->
      let* te, env = infer env e in
      let* result_ty = infer_unop op te.ty loc in
      Ok (mk_texpr (TEUnop (op, te)) result_ty loc, env)
  | _ ->
      error
        (Parse_error
           ("infer_binop_unop called with non-binop/unop expression", loc))

(** Infer type of memory access operations (vectors, arrays, record fields).
    Handles field resolution for both known and external record types. *)
let infer_memory_access ~infer (env : t) (loc : Sarek_ast.loc)
    (expr : expr_desc) : (texpr * t) result =
  match expr with
  | EVecGet (vec, idx) ->
      let* tv, env = infer env vec in
      let* ti, env = infer env idx in
      let* () = unify_or_error ti.ty t_int32 idx.expr_loc in
      let elem_ty = fresh_tvar () in
      let* () = unify_or_error tv.ty (TVec elem_ty) vec.expr_loc in
      let resolved_elem = repr elem_ty in
      Ok (mk_texpr (TEVecGet (tv, ti)) resolved_elem loc, env)
  | EVecSet (vec, idx, value) ->
      let* tv, env = infer env vec in
      let* ti, env = infer env idx in
      let* tx, env = infer env value in
      let* () = unify_or_error ti.ty t_int32 idx.expr_loc in
      let elem_ty = fresh_tvar () in
      let* () = unify_or_error tv.ty (TVec elem_ty) vec.expr_loc in
      let* () = unify_or_error tx.ty elem_ty value.expr_loc in
      Ok (mk_texpr (TEVecSet (tv, ti, tx)) t_unit loc, env)
  | EArrGet (arr, idx) -> (
      let* ta, env = infer env arr in
      let* ti, env = infer env idx in
      let* () = unify_or_error ti.ty t_int32 idx.expr_loc in
      let elem_ty = fresh_tvar () in
      (* Try to unify with vector first, then array *)
      match unify ta.ty (TVec elem_ty) with
      | Ok () ->
          let resolved_elem = repr elem_ty in
          Ok (mk_texpr (TEVecGet (ta, ti)) resolved_elem loc, env)
      | Error _ ->
          let mem = Local in
          let* () = unify_or_error ta.ty (TArr (elem_ty, mem)) arr.expr_loc in
          let resolved_elem = repr elem_ty in
          Ok (mk_texpr (TEArrGet (ta, ti)) resolved_elem loc, env))
  | EArrSet (arr, idx, value) -> (
      let* ta, env = infer env arr in
      let* ti, env = infer env idx in
      let* tx, env = infer env value in
      let* () = unify_or_error ti.ty t_int32 idx.expr_loc in
      let elem_ty = fresh_tvar () in
      (* Try to unify with vector first, then array *)
      match unify ta.ty (TVec elem_ty) with
      | Ok () ->
          let* () = unify_or_error tx.ty elem_ty value.expr_loc in
          Ok (mk_texpr (TEVecSet (ta, ti, tx)) t_unit loc, env)
      | Error _ ->
          let mem = Local in
          let* () = unify_or_error ta.ty (TArr (elem_ty, mem)) arr.expr_loc in
          let* () = unify_or_error tx.ty elem_ty value.expr_loc in
          Ok (mk_texpr (TEArrSet (ta, ti, tx)) t_unit loc, env))
  | EFieldGet (record, field) -> (
      let* tr, env = infer env record in
      match repr tr.ty with
      | TRecord (_type_name, fields) when fields <> [] -> (
          (* Known record type with field info - validate at compile time *)
          match List.assoc_opt field fields with
          | Some field_ty ->
              let idx =
                match
                  List.mapi (fun i (f, _) -> (f, i)) fields
                  |> List.assoc_opt field
                with
                | Some i -> i
                | None -> 0
              in
              Ok (mk_texpr (TEFieldGet (tr, field, idx)) field_ty loc, env)
          | None -> Error [Field_not_found (field, tr.ty, loc)])
      | TRecord (_type_name, []) ->
          (* External record type (from another module) - defer field lookup to runtime.
             This follows the ppx_deriving composability pattern: we trust that the
             type exists and will be registered at runtime by the dependent library.
             The field type and index will be resolved during JIT compilation. *)
          let field_ty = fresh_tvar () in
          Ok (mk_texpr (TEFieldGet (tr, field, 0)) field_ty loc, env)
      | t
        when match repr t with
             | TVar {contents = Unbound _} -> true
             | _ -> false ->
          (* Type not yet known, defer *)
          let field_ty = fresh_tvar () in
          Ok (mk_texpr (TEFieldGet (tr, field, 0)) field_ty loc, env)
      | t -> Error [Not_a_record (t, loc)])
  | EFieldSet (record, field, value) -> (
      let* tr, env = infer env record in
      let* tx, env = infer env value in
      match repr tr.ty with
      | TRecord (_type_name, fields) when fields <> [] -> (
          (* Known record type with field info - validate at compile time *)
          match List.assoc_opt field fields with
          | Some field_ty ->
              let* () = unify_or_error tx.ty field_ty value.expr_loc in
              let idx =
                match
                  List.mapi (fun i (f, _) -> (f, i)) fields
                  |> List.assoc_opt field
                with
                | Some i -> i
                | None -> 0
              in
              Ok (mk_texpr (TEFieldSet (tr, field, idx, tx)) t_unit loc, env)
          | None -> Error [Field_not_found (field, tr.ty, loc)])
      | TRecord (_type_name, []) ->
          (* External record type - defer to runtime (ppx_deriving pattern) *)
          Ok (mk_texpr (TEFieldSet (tr, field, 0, tx)) t_unit loc, env)
      | t -> Error [Not_a_record (t, loc)])
  | _ ->
      error
        (Parse_error
           ("infer_memory_access called with non-memory-access expression", loc))

(** Infer type of control flow expressions (if, for, while, sequence). Ensures
    loop bounds are int32 and conditions are boolean. *)
let infer_control_flow ~infer (env : t) (loc : Sarek_ast.loc) (expr : expr_desc)
    : (texpr * t) result =
  match expr with
  | EIf (cond, then_e, else_opt) -> (
      let* tc, env = infer env cond in
      let* () = check_boolean tc.ty cond.expr_loc in
      let* tt, env = infer env then_e in
      match else_opt with
      | None ->
          let* () = unify_or_error tt.ty t_unit then_e.expr_loc in
          Ok (mk_texpr (TEIf (tc, tt, None)) t_unit loc, env)
      | Some else_e ->
          let* te, env = infer env else_e in
          let* () = unify_or_error tt.ty te.ty else_e.expr_loc in
          Ok (mk_texpr (TEIf (tc, tt, Some te)) tt.ty loc, env))
  | EFor (var, lo, hi, dir, body) ->
      let* tlo, env = infer env lo in
      let* thi, env = infer env hi in
      let* () = unify_or_error tlo.ty t_int32 lo.expr_loc in
      let* () = unify_or_error thi.ty t_int32 hi.expr_loc in
      let var_id = fresh_var_id () in
      let vi =
        {
          vi_type = t_int32;
          vi_mutable = false;
          vi_is_param = false;
          vi_index = var_id;
          vi_is_vec = false;
        }
      in
      let env' = add_var var vi env in
      let* tbody, _ = infer env' body in
      Ok (mk_texpr (TEFor (var, var_id, tlo, thi, dir, tbody)) t_unit loc, env)
  | EWhile (cond, body) ->
      let* tc, env = infer env cond in
      let* () = check_boolean tc.ty cond.expr_loc in
      let* tbody, _ = infer env body in
      Ok (mk_texpr (TEWhile (tc, tbody)) t_unit loc, env)
  | ESeq (e1, e2) ->
      let* t1, env = infer env e1 in
      let* t2, env = infer env e2 in
      Ok (mk_texpr (TESeq [t1; t2]) t2.ty loc, env)
  | _ ->
      assert
        false (* infer_control_flow: called with non-control-flow expression *)

(** Infer type of data structures (records, variants, tuples, arrays). Uses
    mutual recursion helpers for complex nested structures. *)
let infer_data_structure ~infer ~infer_record_fields ~infer_list (env : t)
    (loc : Sarek_ast.loc) (expr : expr_desc) : (texpr * t) result =
  match expr with
  | ERecord (name_opt, fields) ->
      let* tfields, env = infer_record_fields env fields in
      let field_tys = List.map (fun (f, te) -> (f, te.ty)) tfields in
      let inferred_name, ty =
        match name_opt with
        | Some n -> (
            match find_type n env with
            | Some (TIRecord {ti_fields; ti_name}) ->
                ( ti_name,
                  TRecord (ti_name, List.map (fun (f, t, _) -> (f, t)) ti_fields)
                )
            | Some (TIVariant _) ->
                (* Name provided but not a record *)
                ("anon_record", TRecord ("anon_record", field_tys))
            | None -> (n, TRecord (n, field_tys)))
        | None -> (
            (* Try to match an existing record type by field names *)
            let matches =
              StringMap.fold
                (fun _name info acc ->
                  match info with
                  | TIRecord {ti_fields; ti_name} ->
                      let names_match =
                        List.map (fun (f, _, _) -> f) ti_fields
                        = List.map (fun (f, _) -> f) field_tys
                      in
                      if names_match then
                        (ti_name, List.map (fun (f, t, _) -> (f, t)) ti_fields)
                        :: acc
                      else acc
                  | _ -> acc)
                env.types
                []
            in
            match matches with
            | [] -> ("anon_record", TRecord ("anon_record", field_tys))
            | (n, tys) :: _ ->
                (* Use first match - prefer local types but also accept registered types *)
                (n, TRecord (n, tys)))
      in
      Ok (mk_texpr (TERecord (inferred_name, tfields)) ty loc, env)
  | EConstr (name, arg_opt) -> (
      match find_constructor name env with
      | Some (type_name, expected_arg) -> (
          (* Get the full variant type with ALL constructors from the environment *)
          let full_variant_ty =
            match find_type type_name env with
            | Some (TIVariant {ti_constrs; ti_name}) ->
                TVariant (ti_name, ti_constrs)
            | _ ->
                (* Fallback: shouldn't happen for registered variants *)
                TVariant (type_name, [(name, expected_arg)])
          in
          match (arg_opt, expected_arg) with
          | None, None ->
              Ok
                ( mk_texpr (TEConstr (type_name, name, None)) full_variant_ty loc,
                  env )
          | Some arg, Some expected_ty ->
              let* targ, env = infer env arg in
              let* () = unify_or_error targ.ty expected_ty arg.expr_loc in
              Ok
                ( mk_texpr
                    (TEConstr (type_name, name, Some targ))
                    full_variant_ty
                    loc,
                  env )
          | None, Some _ -> Error [Wrong_arity {expected = 1; got = 0; loc}]
          | Some _, None -> Error [Wrong_arity {expected = 0; got = 1; loc}])
      | None -> Error [Unbound_constructor (name, loc)])
  | ETuple es ->
      let* tes, env = infer_list env es in
      let ty = TTuple (List.map (fun te -> te.ty) tes) in
      Ok (mk_texpr (TETuple tes) ty loc, env)
  | EReturn e ->
      let* te, env = infer env e in
      Ok (mk_texpr (TEReturn te) te.ty loc, env)
  | ECreateArray (size, elem_ty, mem) ->
      let* tsize, env = infer env size in
      let* () = unify_or_error tsize.ty t_int32 size.expr_loc in
      let elem_t = type_of_type_expr_env env elem_ty in
      let arr_ty = TArr (elem_t, memspace_of_ast mem) in
      Ok
        ( mk_texpr
            (TECreateArray (tsize, elem_t, memspace_of_ast mem))
            arr_ty
            loc,
          env )
  | _ ->
      error
        (Parse_error
           ( "infer_data_structure called with non-data-structure expression",
             loc ))

(** Infer type of special expressions (global refs, native blocks, pragmas, type
    annotations, open). These handle meta-level constructs and environment
    manipulation. *)
let infer_special ~infer (env : t) (loc : Sarek_ast.loc) (expr : expr_desc) :
    (texpr * t) result =
  match expr with
  | EGlobalRef name ->
      (* Type will be inferred from context or annotated *)
      let ty = fresh_tvar () in
      Ok (mk_texpr (TEGlobalRef (name, ty)) ty loc, env)
  | ENative {gpu; ocaml} ->
      let ty = fresh_tvar () in
      Ok (mk_texpr (TENative {gpu; ocaml}) ty loc, env)
  | EPragma (opts, body) ->
      Sarek_debug.log "EPragma opts=%s" (String.concat "," opts) ;
      let* tbody, env = infer env body in
      Sarek_debug.log "EPragma done" ;
      Ok (mk_texpr (TEPragma (opts, tbody)) tbody.ty loc, env)
  | ETyped (e, ty_expr) ->
      let* te, env = infer env e in
      let ty = type_of_type_expr_env env ty_expr in
      let* () = unify_or_error te.ty ty loc in
      Ok ({te with ty = repr ty}, env)
  | EOpen (path, e) ->
      (* Bring module's bindings into scope for the inner expression only.
         We discard the modified environment to prevent the open from leaking
         to sibling expressions in sequences.
         We preserve the TEOpen in the typed AST so native code gen can
         generate `let open M in ...` *)
      let env' = Sarek_env.open_module path env in
      let* te, _env_inner = infer env' e in
      Ok ({te = TEOpen (path, te); ty = te.ty; te_loc = loc}, env)
  | _ ->
      error
        (Parse_error ("infer_special called with non-special expression", loc))

(** Infer type of let bindings (assign, let, let mut, let rec). Handles
    mutability constraints and recursive binding generalization. *)
let infer_let_binding ~infer (env : t) (loc : Sarek_ast.loc) (expr : expr_desc)
    : (texpr * t) result =
  match expr with
  | EAssign (name, value) -> (
      let* tv, env = infer env value in
      match find_var name env with
      | None -> Error [Unbound_variable (name, loc)]
      | Some vi ->
          if (not vi.vi_mutable) || vi.vi_is_param then
            Error [Immutable_variable (name, loc)]
          else
            let* () = unify_or_error tv.ty vi.vi_type value.expr_loc in
            Ok (mk_texpr (TEAssign (name, vi.vi_index, tv)) t_unit loc, env))
  | ELet (name, ty_annot, value, body) ->
      let env' = enter_level env in
      let* tv, env' = infer env' value in
      let* () =
        match ty_annot with
        | None -> Ok ()
        | Some te ->
            let t = type_of_type_expr_env env te in
            unify_or_error tv.ty t value.expr_loc
      in
      let var_id = fresh_var_id () in
      let vi =
        {
          vi_type = tv.ty;
          vi_mutable = false;
          vi_is_param = false;
          vi_index = var_id;
          vi_is_vec = false;
        }
      in
      let env'' = add_var name vi (exit_level env') in
      let* tb, env'' = infer env'' body in
      Ok (mk_texpr (TELet (name, var_id, tv, tb)) tb.ty loc, env'')
  | ELetMut (name, ty_annot, value, body) ->
      let* tv, env = infer env value in
      let* () =
        match ty_annot with
        | None -> Ok ()
        | Some te ->
            let t = type_of_type_expr_env env te in
            unify_or_error tv.ty t value.expr_loc
      in
      let var_id = fresh_var_id () in
      let vi =
        {
          vi_type = tv.ty;
          vi_mutable = true;
          vi_is_param = false;
          vi_index = var_id;
          vi_is_vec = false;
        }
      in
      let env' = add_var name vi env in
      let* tb, env' = infer env' body in
      Ok (mk_texpr (TELetMut (name, var_id, tv, tb)) tb.ty loc, env')
  | ELetRec (name, params, ret_ty_opt, fn_body, cont) ->
      Sarek_debug.log "ELetRec '%s'" name ;
      (* Enter a level for the function's type variables *)
      let env_inner = enter_level env in
      (* Type parameters using shared type variable context at inner level *)
      let tvar_ctx = fresh_tvar_ctx ~level:env_inner.current_level () in
      let param_tys =
        List.map
          (fun p -> type_of_type_expr_ctx env tvar_ctx p.param_type)
          params
      in
      let fn_id = fresh_var_id () in
      (* Create function type and add to environment for recursive calls *)
      let ret_ty =
        match ret_ty_opt with
        | Some ty -> type_of_type_expr_ctx env tvar_ctx ty
        | None -> fresh_tvar ~level:env_inner.current_level ()
      in
      let fn_ty = TFun (param_tys, ret_ty) in
      let env_with_fn = add_local_fun name (mono fn_ty) env_inner in
      (* Add parameters to environment *)
      let tparams, env_with_params =
        List.fold_left2
          (fun (acc, env) p param_ty ->
            let pid = fresh_var_id () in
            let tparam =
              {
                tparam_name = p.param_name;
                tparam_type = param_ty;
                tparam_index = List.length acc;
                tparam_is_vec = false;
                tparam_id = pid;
              }
            in
            let param_vi =
              {
                vi_type = param_ty;
                vi_mutable = false;
                vi_is_param = false;
                vi_index = 0;
                vi_is_vec = false;
              }
            in
            (tparam :: acc, add_var p.param_name param_vi env))
          ([], env_with_fn)
          params
          param_tys
      in
      let tparams = List.rev tparams in
      (* Type the function body *)
      let* tfn_body, _ = infer env_with_params fn_body in
      let* () = unify_or_error ret_ty tfn_body.ty fn_body.expr_loc in
      (* Generalize function type for use in continuation *)
      let fn_scheme = generalize env.current_level fn_ty in
      let env_for_cont = add_local_fun name fn_scheme env in
      (* Type the continuation with the function in scope *)
      let* tcont, env = infer env_for_cont cont in
      Ok
        ( mk_texpr
            (TELetRec (name, fn_id, tparams, tfn_body, tcont))
            tcont.ty
            loc,
          env )
  | _ ->
      assert
        false (* infer_let_binding: called with non-let-binding expression *)

(** Main type inference function *)
let rec infer (env : t) (expr : expr) : (texpr * t) result =
  let loc = expr.expr_loc in
  match expr.e with
  (* Literals *)
  | EUnit | EBool _ | EInt _ | EInt32 _ | EInt64 _ | EFloat _ | EDouble _ ->
      let* te = infer_literal loc expr.e in
      Ok (te, env)
  (* Variables *)
  | EVar name -> (
      match lookup name env with
      | LVar info ->
          let id = info.vi_index in
          (* Use the index as ID *)
          Ok (mk_texpr (TEVar (name, id)) info.vi_type loc, env)
      | LIntrinsicConst info ->
          Ok
            (mk_texpr (TEIntrinsicConst info.const_ref) info.const_type loc, env)
      | LIntrinsicFun info ->
          (* Return a function value that will be applied *)
          let id = fresh_var_id () in
          Ok (mk_texpr (TEVar (name, id)) info.intr_type loc, env)
      | LConstructor (type_name, arg_type) -> (
          match arg_type with
          | None ->
              let ty = TVariant (type_name, [(name, None)]) in
              Ok (mk_texpr (TEConstr (type_name, name, None)) ty loc, env)
          | Some _ ->
              (* Constructor that takes an argument - return as function *)
              let ty = fresh_tvar () in
              (* Will be resolved when applied *)
              let id = fresh_var_id () in
              Ok (mk_texpr (TEVar (name, id)) ty loc, env))
      | LLocalFun (_, scheme) ->
          (* Instantiate the polymorphic scheme to get a fresh type *)
          let typ = instantiate scheme in
          let id = fresh_var_id () in
          Ok (mk_texpr (TEVar (name, id)) typ loc, env)
      | LNotFound ->
          (* Check if this is a qualified name (Module.func) from another module.
             If so, treat it as an external function that will be resolved at runtime
             via the Sarek_registry. This follows the ppx_deriving composability pattern. *)
          if String.contains name '.' then
            let id = fresh_var_id () in
            let ty = fresh_tvar () in
            Ok (mk_texpr (TEVar (name, id)) ty loc, env)
          else Error [Unbound_variable (name, loc)])
  (* Memory access *)
  | EVecGet _ | EVecSet _ | EArrGet _ | EArrSet _ | EFieldGet _ | EFieldSet _ ->
      infer_memory_access ~infer env loc expr.e
  (* Binary and unary operations *)
  | EBinop _ | EUnop _ -> infer_binop_unop ~infer env loc expr.e
  (* Function application *)
  | EApp (fn, args) -> (
      let* tfn, env = infer env fn in
      let* targs, env = infer_list env args in
      match fn.e with
      | EVar name when is_intrinsic_fun name env -> (
          (* Intrinsic function call *)
          match find_intrinsic_fun name env with
          | Some info -> (
              match info.intr_type with
              | TFun (param_tys, ret_ty) ->
                  if List.length param_tys <> List.length targs then
                    Error
                      [
                        Wrong_arity
                          {
                            expected = List.length param_tys;
                            got = List.length targs;
                            loc;
                          };
                      ]
                  else begin
                    let* () = unify_args param_tys targs loc in
                    Ok
                      ( mk_texpr
                          (TEIntrinsicFun
                             (info.intr_ref, info.intr_convergence, targs))
                          ret_ty
                          loc,
                        env )
                  end
              | t -> Error [Not_a_function (t, fn.expr_loc)])
          | None -> Error [Unbound_variable (name, fn.expr_loc)])
      | _ ->
          (* Regular function application *)
          let ret_ty = fresh_tvar () in
          let expected_fn_ty = TFun (List.map (fun t -> t.ty) targs, ret_ty) in
          let* () = unify_or_error tfn.ty expected_fn_ty fn.expr_loc in
          Ok (mk_texpr (TEApp (tfn, targs)) (repr ret_ty) loc, env))
  (* Let bindings *)
  | EAssign _ | ELet _ | ELetMut _ | ELetRec _ ->
      infer_let_binding ~infer env loc expr.e
  (* Control flow *)
  | EIf _ | EFor _ | EWhile _ | ESeq _ ->
      infer_control_flow ~infer env loc expr.e
  (* Match *)
  | EMatch (scrutinee, cases) ->
      let* ts, env = infer env scrutinee in
      let* tcases, result_ty, env = infer_match_cases env ts.ty cases loc in
      Ok (mk_texpr (TEMatch (ts, tcases)) result_ty loc, env)
  (* Data structures *)
  | ERecord _ | EConstr _ | ETuple _ | EReturn _ | ECreateArray _ ->
      infer_data_structure
        ~infer
        ~infer_record_fields
        ~infer_list
        env
        loc
        expr.e
  (* Special expressions *)
  | EGlobalRef _ | ENative _ | EPragma _ | ETyped _ | EOpen _ ->
      infer_special ~infer env loc expr.e
  (* BSP constructs: let%shared and let%superstep *)
  | ELetShared (name, elem_ty, size_opt, body) ->
      (* Type the optional size expression *)
      let* tsize_opt, env =
        match size_opt with
        | None -> Ok (None, env)
        | Some size ->
            let* tsize, env = infer env size in
            let* () = unify_or_error tsize.ty t_int32 size.expr_loc in
            Ok (Some tsize, env)
      in
      (* Create the shared array type *)
      let elem_t = type_of_type_expr_env env elem_ty in
      let arr_ty = TArr (elem_t, Shared) in
      let var_id = fresh_var_id () in
      let vi =
        {
          vi_type = arr_ty;
          vi_mutable = false;
          vi_is_param = false;
          vi_index = var_id;
          vi_is_vec = false;
        }
      in
      let env' = add_var name vi env in
      let* tbody, env' = infer env' body in
      Ok
        ( mk_texpr
            (TELetShared (name, var_id, elem_t, tsize_opt, tbody))
            tbody.ty
            loc,
          env' )
  | ESuperstep (name, divergent, step_body, cont) ->
      (* Type the superstep body - should be unit *)
      let* tstep_body, env = infer env step_body in
      let* () = unify_or_error tstep_body.ty t_unit step_body.expr_loc in
      (* Type the continuation *)
      let* tcont, env = infer env cont in
      Ok
        ( mk_texpr
            (TESuperstep (name, divergent, tstep_body, tcont))
            tcont.ty
            loc,
          env )

and infer_list env exprs =
  let rec aux env acc = function
    | [] -> Ok (List.rev acc, env)
    | e :: rest ->
        let* te, env = infer env e in
        aux env (te :: acc) rest
  in
  aux env [] exprs

and unify_args param_tys targs loc =
  let rec aux = function
    | [], [] -> Ok ()
    | pt :: pts, ta :: tas ->
        let* () = unify_or_error pt ta.ty ta.te_loc in
        aux (pts, tas)
    | _, _ ->
        Error
          [
            Wrong_arity
              {expected = List.length param_tys; got = List.length targs; loc};
          ]
  in
  aux (param_tys, targs)

and infer_record_fields env fields =
  let rec aux env acc = function
    | [] -> Ok (List.rev acc, env)
    | (name, e) :: rest ->
        let* te, env = infer env e in
        aux env ((name, te) :: acc) rest
  in
  aux env [] fields

and infer_match_cases env scrutinee_ty cases loc =
  match cases with
  | [] -> Error [Invalid_kernel ("empty match", loc)]
  | (pat, body) :: rest ->
      let* tpat, env' = infer_pattern env scrutinee_ty pat in
      let* tbody, _ = infer env' body in
      let result_ty = tbody.ty in
      let* tcases = infer_remaining_cases env scrutinee_ty result_ty rest in
      Ok ((tpat, tbody) :: tcases, result_ty, env)

and infer_remaining_cases env scrutinee_ty result_ty cases =
  let rec aux acc = function
    | [] -> Ok (List.rev acc)
    | (pat, body) :: rest ->
        let* tpat, env' = infer_pattern env scrutinee_ty pat in
        let* tbody, _ = infer env' body in
        let* () = unify_or_error tbody.ty result_ty body.expr_loc in
        aux ((tpat, tbody) :: acc) rest
  in
  aux [] cases

and infer_pattern env scrutinee_ty pat =
  let loc = pat.pat_loc in
  match pat.pat with
  | PAny -> Ok ({tpat = TPAny; tpat_ty = scrutinee_ty; tpat_loc = loc}, env)
  | PVar name ->
      let var_id = fresh_var_id () in
      let vi =
        {
          vi_type = scrutinee_ty;
          vi_mutable = false;
          vi_is_param = false;
          vi_index = var_id;
          vi_is_vec = false;
        }
      in
      let env' = add_var name vi env in
      Ok
        ( {tpat = TPVar (name, var_id); tpat_ty = scrutinee_ty; tpat_loc = loc},
          env' )
  | PConstr (name, arg_pat) -> (
      match find_constructor name env with
      | Some (type_name, arg_ty_opt) -> (
          match (arg_pat, arg_ty_opt) with
          | None, None ->
              Ok
                ( {
                    tpat = TPConstr (type_name, name, None);
                    tpat_ty = scrutinee_ty;
                    tpat_loc = loc;
                  },
                  env )
          | Some ap, Some arg_ty ->
              let* tap, env' = infer_pattern env arg_ty ap in
              Ok
                ( {
                    tpat = TPConstr (type_name, name, Some tap);
                    tpat_ty = scrutinee_ty;
                    tpat_loc = loc;
                  },
                  env' )
          | None, Some _ -> Error [Wrong_arity {expected = 1; got = 0; loc}]
          | Some _, None -> Error [Wrong_arity {expected = 0; got = 1; loc}])
      | None -> Error [Unbound_constructor (name, loc)])
  | PTuple pats ->
      let* ty_list =
        match repr scrutinee_ty with
        | TTuple ts when List.length ts = List.length pats -> Ok ts
        | _ ->
            let ts = List.map (fun _ -> fresh_tvar ()) pats in
            let* () = unify_or_error scrutinee_ty (TTuple ts) loc in
            Ok ts
      in
      let* tpats, env' = infer_patterns env ty_list pats in
      Ok ({tpat = TPTuple tpats; tpat_ty = scrutinee_ty; tpat_loc = loc}, env')

and infer_patterns env tys pats =
  let rec aux env acc = function
    | [], [] -> Ok (List.rev acc, env)
    | ty :: tys, pat :: pats ->
        let* tpat, env' = infer_pattern env ty pat in
        aux env' (tpat :: acc) (tys, pats)
    | tys, pats ->
        (* Length mismatch between types and patterns *)
        Error
          [
            Wrong_arity
              {
                expected = List.length tys;
                got = List.length pats;
                loc = dummy_loc;
              };
          ]
  in
  aux env [] (tys, pats)

(** Check if a name is an intrinsic function in the environment *)
and is_intrinsic_fun name env =
  match find_intrinsic_fun name env with Some _ -> true | None -> false

(** Type a complete kernel *)
let infer_kernel (env : t) (kernel : Sarek_ast.kernel) : tkernel result =
  (* Register type declarations first *)
  let rec add_type_decls env acc = function
    | [] -> Ok (List.rev acc, env)
    | decl :: rest -> (
        match decl with
        | Sarek_ast.Type_record
            {tdecl_name; tdecl_module; tdecl_fields; tdecl_loc} ->
            let full_name =
              match tdecl_module with
              | Some m -> m ^ "." ^ tdecl_name
              | None -> tdecl_name
            in
            let tfields =
              List.map
                (fun (fname, fmut, fty) ->
                  (fname, type_of_type_expr_env env fty, fmut))
                tdecl_fields
            in
            let env' =
              add_type
                full_name
                (TIRecord {ti_name = full_name; ti_fields = tfields})
                env
            in
            let env'' =
              add_type
                tdecl_name
                (TIRecord {ti_name = full_name; ti_fields = tfields})
                env'
            in
            let acc_decl =
              TTypeRecord
                {
                  tdecl_name = full_name;
                  tdecl_module;
                  tdecl_fields = tfields;
                  tdecl_loc;
                }
            in
            add_type_decls env'' (acc_decl :: acc) rest
        | Sarek_ast.Type_variant
            {tdecl_name; tdecl_module; tdecl_constructors; tdecl_loc} ->
            let full_name =
              match tdecl_module with
              | Some m -> m ^ "." ^ tdecl_name
              | None -> tdecl_name
            in
            let constrs =
              List.map
                (fun (cname, carg) ->
                  (cname, Option.map (type_of_type_expr_env env) carg))
                tdecl_constructors
            in
            let env' =
              add_type
                full_name
                (TIVariant {ti_name = full_name; ti_constrs = constrs})
                env
            in
            let env'' =
              add_type
                tdecl_name
                (TIVariant {ti_name = full_name; ti_constrs = constrs})
                env'
            in
            let acc_decl =
              TTypeVariant
                {
                  tdecl_name = full_name;
                  tdecl_module;
                  tdecl_constructors = constrs;
                  tdecl_loc;
                }
            in
            add_type_decls env'' (acc_decl :: acc) rest)
  in
  let* ttypes, env_with_types = add_type_decls env [] kernel.kern_types in

  (* Type module items next to extend the environment *)
  let rec add_module_items env acc = function
    | [] -> Ok (List.rev acc, env)
    | item :: rest -> (
        match item with
        | Sarek_ast.MConst (name, ty_expr, value) ->
            let ty = type_of_type_expr_env env ty_expr in
            let* tvalue, env' = infer env value in
            let* () = unify_or_error tvalue.ty ty value.expr_loc in
            let var_id = fresh_var_id () in
            let vi =
              {
                vi_type = ty;
                vi_mutable = false;
                vi_is_param = false;
                vi_index = var_id;
                vi_is_vec = false;
              }
            in
            let env'' = add_var name vi env' in
            add_module_items
              env''
              (TMConst (name, var_id, ty, tvalue) :: acc)
              rest
        | Sarek_ast.MFun (name, is_rec, params, body) ->
            (* Check for reserved keywords *)
            if Sarek_reserved.is_reserved name then
              Error [Reserved_keyword (name, body.expr_loc)]
            else begin
              (* Enter a new level for let-polymorphism *)
              let env_inner = enter_level env in
              (* Create a shared type variable context for all params at inner level *)
              let tvar_ctx = fresh_tvar_ctx ~level:env_inner.current_level () in
              let rec add_fun_params env idx acc = function
                | [] -> Ok (List.rev acc, env)
                | p :: rest ->
                    let ty = type_of_type_expr_ctx env tvar_ctx p.param_type in
                    let is_vec = match ty with TVec _ -> true | _ -> false in
                    let var_id = fresh_var_id () in
                    let vi =
                      {
                        vi_type = ty;
                        vi_mutable = false;
                        vi_is_param = false;
                        vi_index = var_id;
                        vi_is_vec = is_vec;
                      }
                    in
                    let env' = add_var p.param_name vi env in
                    let tparam =
                      {
                        tparam_name = p.param_name;
                        tparam_type = ty;
                        tparam_index = idx;
                        tparam_is_vec = is_vec;
                        tparam_id = var_id;
                      }
                    in
                    add_fun_params env' (idx + 1) (tparam :: acc) rest
              in
              let* tparams, env_fun = add_fun_params env_inner 0 [] params in
              (* For recursive functions, add a placeholder type for the function
                 so it can reference itself in its body *)
              let param_types = List.map (fun p -> p.tparam_type) tparams in
              let ret_type_var =
                TVar (ref (Unbound (fresh_var_id (), env_inner.current_level)))
              in
              let fn_ty_placeholder = TFun (param_types, ret_type_var) in
              let env_body =
                if is_rec then
                  add_local_fun name (mono fn_ty_placeholder) env_fun
                else env_fun
              in
              let* tbody, _ = infer env_body body in
              let fn_ty = TFun (param_types, tbody.ty) in
              (* Generalize the function type to create a polymorphic scheme *)
              let fn_scheme = generalize env.current_level fn_ty in
              if Sarek_debug.enabled then
                Sarek_debug.log
                  "MFun %s : %s"
                  name
                  (Sarek_types.typ_to_string fn_ty) ;
              let env'' = add_local_fun name fn_scheme env in
              add_module_items
                env''
                (TMFun (name, is_rec, tparams, tbody) :: acc)
                rest
            end)
  in
  let* tmodule_items, env_after_mods =
    add_module_items env_with_types [] kernel.kern_module_items
  in
  (* Add parameters to environment *)
  let rec add_params env idx acc = function
    | [] -> Ok (List.rev acc, env)
    | p :: rest ->
        let ty = type_of_type_expr_env env p.param_type in
        let is_vec = match ty with TVec _ -> true | _ -> false in
        let var_id = fresh_var_id () in
        let vi =
          {
            vi_type = ty;
            vi_mutable = false;
            vi_is_param = true;
            vi_index = var_id;
            vi_is_vec = is_vec;
          }
        in
        let env' = add_var p.param_name vi env in
        let tparam =
          {
            tparam_name = p.param_name;
            tparam_type = ty;
            tparam_index = idx;
            tparam_is_vec = is_vec;
            tparam_id = var_id;
          }
        in
        add_params env' (idx + 1) (tparam :: acc) rest
  in
  let* tparams, env' = add_params env_after_mods 0 [] kernel.kern_params in
  let* tbody, _ = infer env' kernel.kern_body in
  Ok
    {
      tkern_name = kernel.kern_name;
      tkern_type_decls = ttypes;
      tkern_module_items = tmodule_items;
      tkern_external_item_count = kernel.kern_external_item_count;
      tkern_params = tparams;
      tkern_body = tbody;
      tkern_return_type = tbody.ty;
      tkern_loc = kernel.kern_loc;
    }
