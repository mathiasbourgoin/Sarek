(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module lowers the typed AST directly to Sarek_ir_ppx, bypassing
 * the legacy Kirc_Ast. This provides a cleaner IR for the V2 execution path.
 ******************************************************************************)

open Sarek_ast
open Sarek_types
open Sarek_typed_ast
module Ir = Sarek_ir_ppx

(** Mangle type names for C compatibility *)
let mangle_type_name name = String.map (function '.' -> '_' | c -> c) name

(** Convert Sarek_types.typ to Sarek_ir_ppx.elttype *)
let rec elttype_of_typ (ty : typ) : Ir.elttype =
  match repr ty with
  | TPrim TInt32 -> Ir.TInt32
  | TPrim TBool -> Ir.TBool
  | TPrim TUnit -> Ir.TUnit
  | TReg Int64 -> Ir.TInt64
  | TReg Int -> Ir.TInt32 (* int maps to int32 on GPU *)
  | TReg Float32 -> Ir.TFloat32
  | TReg Float64 -> Ir.TFloat64
  | TReg Char -> Ir.TInt32 (* char represented as int32 *)
  | TReg (Custom _) -> Ir.TInt32 (* Custom types handled separately *)
  | TVec elem_ty -> Ir.TVec (elttype_of_typ elem_ty)
  | TArr (elem_ty, mem) ->
      let ir_mem =
        match mem with
        | Sarek_types.Global -> Ir.Global
        | Sarek_types.Shared -> Ir.Shared
        | Sarek_types.Local -> Ir.Local
      in
      Ir.TArray (elttype_of_typ elem_ty, ir_mem)
  | TRecord (name, fields) ->
      Ir.TRecord (name, List.map (fun (n, ty) -> (n, elttype_of_typ ty)) fields)
  | TVariant (name, constrs) ->
      Ir.TVariant
        ( name,
          List.map
            (fun (n, ty_opt) ->
              ( n,
                match ty_opt with None -> [] | Some ty -> [elttype_of_typ ty] ))
            constrs )
  | TTuple _ -> Ir.TInt32
  | TFun _ -> Ir.TInt32
  | TVar _ ->
      Ppxlib.Location.raise_errorf
        ~loc:Ppxlib.Location.none
        "Kernel parameter type is a type variable — polymorphic kernels are \
         not supported. Annotate the parameter with a concrete type (e.g. \
         float32, int32)."

(** Convert Sarek_types.memspace to Sarek_ir_ppx.memspace *)
let memspace_of_memspace (mem : Sarek_types.memspace) : Ir.memspace =
  match mem with
  | Sarek_types.Global -> Ir.Global
  | Sarek_types.Shared -> Ir.Shared
  | Sarek_types.Local -> Ir.Local

(** Get C type string for a typ *)
let rec c_type_of_typ ty =
  match repr ty with
  | TPrim TInt32 -> "int"
  | TPrim TBool -> "int"
  | TPrim TUnit -> "void"
  | TReg Int -> "int"
  | TReg Int64 -> "long"
  | TReg Float32 -> "float"
  | TReg Float64 -> "double"
  | TReg Char -> "char"
  | TReg (Custom name) -> mangle_type_name name
  | TRecord (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVariant (name, _) -> "struct " ^ mangle_type_name name ^ "_sarek"
  | TVec t -> c_type_of_typ t ^ " *"
  | TArr (t, _) -> c_type_of_typ t ^ " *"
  | _ -> "int"

(** Generate C struct definition and builder for record types *)
let record_constructor_strings name (fields : (string * typ) list) =
  let name = mangle_type_name name in
  let struct_name = name ^ "_sarek" in
  let struct_fields =
    List.map
      (fun (fname, fty) -> "  " ^ c_type_of_typ fty ^ " " ^ fname ^ ";")
      fields
  in
  let struct_def =
    "struct " ^ struct_name ^ " {\n" ^ String.concat "\n" struct_fields ^ "\n};"
  in
  let params =
    String.concat
      ", "
      (List.map (fun (fname, fty) -> c_type_of_typ fty ^ " " ^ fname) fields)
  in
  let assigns =
    String.concat
      "\n"
      (List.map
         (fun (fname, _) -> "  res." ^ fname ^ " = " ^ fname ^ ";")
         fields)
  in
  let builder =
    "struct " ^ struct_name ^ " build_" ^ struct_name ^ "(" ^ params ^ ") {\n"
    ^ "  struct " ^ struct_name ^ " res;\n" ^ assigns ^ "\n  return res;\n}"
  in
  [struct_def; builder]

(** Generate C struct definitions and builders for variant types *)
let variant_constructor_strings name constrs =
  let name = mangle_type_name name in
  let struct_name = name ^ "_sarek" in
  let constr_structs =
    List.map
      (fun (cname, carg) ->
        let field =
          match carg with
          | None -> "  int " ^ name ^ "_sarek_" ^ cname ^ "_t;"
          | Some ty ->
              "  " ^ c_type_of_typ ty ^ " " ^ name ^ "_sarek_" ^ cname ^ "_t;"
        in
        "struct " ^ name ^ "_sarek_" ^ cname ^ " {\n" ^ field ^ "\n};")
      constrs
  in
  let union_fields =
    List.map
      (fun (cname, _) ->
        "  struct " ^ name ^ "_sarek_" ^ cname ^ " " ^ name ^ "_sarek_" ^ cname
        ^ ";")
      constrs
  in
  let union_def =
    "union " ^ name ^ "_sarek_union {\n"
    ^ String.concat "\n" union_fields
    ^ "\n};"
  in
  let main_struct =
    "struct " ^ struct_name ^ " {\n" ^ "  int " ^ name ^ "_sarek_tag;\n"
    ^ "  union " ^ name ^ "_sarek_union " ^ name ^ "_sarek_union;\n" ^ "};"
  in
  let builders =
    List.mapi
      (fun idx (cname, carg) ->
        let params, assign =
          match carg with
          | None -> ("", "  /* no payload */")
          | Some ty ->
              let pname = "v" in
              ( c_type_of_typ ty ^ " " ^ pname,
                "  res." ^ name ^ "_sarek_union." ^ name ^ "_sarek_" ^ cname
                ^ "." ^ name ^ "_sarek_" ^ cname ^ "_t = " ^ pname ^ ";" )
        in
        "struct " ^ struct_name ^ " build_" ^ name ^ "_" ^ cname ^ "(" ^ params
        ^ ") {\n" ^ "  struct " ^ struct_name ^ " res;\n" ^ "  res." ^ name
        ^ "_sarek_tag = " ^ string_of_int idx ^ ";\n" ^ assign ^ "\n"
        ^ "  return res;\n}")
      constrs
  in
  constr_structs @ (union_def :: main_struct :: builders)

(* Debug counters for IR lowering *)
let ir_lower_expr_count = ref 0

let ir_lower_stmt_count = ref 0

(** Lowering state *)
type state = {
  mutable next_var_id : int;
  fun_map : (string, tparam list * texpr) Hashtbl.t;
  lowering_stack : (string, unit) Hashtbl.t;
  lowered_funs : (string, Ir.helper_func) Hashtbl.t;
      (** Lowered helper functions: name -> helper_func *)
  mutable lowered_funs_order : string list;
      (** Order in which functions were lowered (for dependency ordering) *)
  types : (string, (string * Ir.elttype) list) Hashtbl.t;
      (** Collected record types: type_name -> [(field_name, field_type); ...]
      *)
  variants : (string, (string * Ir.elttype list) list) Hashtbl.t;
      (** Collected variant types: type_name ->
          [(constructor_name, payload_types); ...] *)
}

let create_state fun_map =
  {
    next_var_id = 0;
    fun_map;
    lowering_stack = Hashtbl.create 8;
    lowered_funs = Hashtbl.create 8;
    lowered_funs_order = [];
    types = Hashtbl.create 8;
    variants = Hashtbl.create 8;
  }

let fresh_id state =
  let id = state.next_var_id in
  state.next_var_id <- id + 1 ;
  id

(** Convert Sarek_ast.binop to Sarek_ir_ppx.binop *)
let ir_binop (op : binop) (_ty : typ) : Ir.binop =
  match op with
  | Add -> Ir.Add
  | Sub -> Ir.Sub
  | Mul -> Ir.Mul
  | Div -> Ir.Div
  | Mod -> Ir.Mod
  | Eq -> Ir.Eq
  | Ne -> Ir.Ne
  | Lt -> Ir.Lt
  | Le -> Ir.Le
  | Gt -> Ir.Gt
  | Ge -> Ir.Ge
  | And -> Ir.And
  | Or -> Ir.Or
  | Lsl -> Ir.Shl
  | Lsr -> Ir.Shr
  | Asr -> Ir.Shr
  | Land -> Ir.BitAnd
  | Lor -> Ir.BitOr
  | Lxor -> Ir.BitXor

(** Convert Sarek_ast.unop to Sarek_ir_ppx.unop *)
let ir_unop (op : unop) : Ir.unop =
  match op with Neg -> Ir.Neg | Not -> Ir.Not | Lnot -> Ir.BitNot

(** Convert memspace *)
let lower_memspace = function
  | Local -> Ir.Local
  | Shared -> Ir.Shared
  | Global -> Ir.Global

(** Create a var from typed var info *)
let make_var name id ty mutable_ : Ir.var =
  {
    var_name = name;
    var_id = id;
    var_type = elttype_of_typ ty;
    var_mutable = mutable_;
  }

(** Lower a declaration *)
let lower_decl ~mutable_ id name ty : Ir.decl =
  let v = make_var name id ty mutable_ in
  Ir.DLocal (v, None)

(** Transform a statement to ensure it returns a value. This adds SReturn to
    leaf statements without re-traversing the original AST. *)
let rec make_returning stmt =
  match stmt with
  | Ir.SReturn _ -> stmt (* Already returns *)
  | Ir.SExpr e -> Ir.SReturn e (* Expression -> return it *)
  | Ir.SIf (c, t, Some e) ->
      Ir.SIf (c, make_returning t, Some (make_returning e))
  | Ir.SIf (c, t, None) ->
      Ir.SIf (c, make_returning t, Some (Ir.SReturn (Ir.EConst Ir.CUnit)))
  | Ir.SMatch (e, cases) ->
      Ir.SMatch (e, List.map (fun (p, b) -> (p, make_returning b)) cases)
  | Ir.SSeq stmts -> (
      match List.rev stmts with
      | [] -> Ir.SReturn (Ir.EConst Ir.CUnit)
      | last :: rest -> Ir.SSeq (List.rev (make_returning last :: rest)))
  | Ir.SLet (v, e, body) -> Ir.SLet (v, e, make_returning body)
  | Ir.SLetMut (v, e, body) -> Ir.SLetMut (v, e, make_returning body)
  | Ir.SPragma (opts, body) -> Ir.SPragma (opts, make_returning body)
  | Ir.SFor _ | Ir.SWhile _ | Ir.SAssign _ | Ir.SBarrier | Ir.SWarpBarrier
  | Ir.SMemFence | Ir.SEmpty | Ir.SNative _ ->
      (* These are side-effect statements; return unit after *)
      Ir.SSeq [stmt; Ir.SReturn (Ir.EConst Ir.CUnit)]
  | Ir.SBlock body -> Ir.SBlock (make_returning body)

(** Convert a typed expression to IR expression *)
let rec lower_expr (state : state) (te : texpr) : Ir.expr =
  incr ir_lower_expr_count ;
  (* Log progress every 10000 calls *)
  if !ir_lower_expr_count mod 10000 = 0 then
    Sarek_debug.log_to_file
      (Printf.sprintf "    [IR] lower_expr progress: %d" !ir_lower_expr_count) ;
  match te.te with
  | TEUnit -> Ir.EConst Ir.CUnit
  | TEBool b -> Ir.EConst (Ir.CBool b)
  | TEInt i -> Ir.EConst (Ir.CInt32 (Int32.of_int i))
  | TEInt32 i -> Ir.EConst (Ir.CInt32 i)
  | TEInt64 i -> Ir.EConst (Ir.CInt64 i)
  | TEFloat f -> Ir.EConst (Ir.CFloat32 f)
  | TEDouble f -> Ir.EConst (Ir.CFloat64 f)
  | TEVar (name, id) -> (
      match repr te.ty with
      | TFun _ ->
          (* Function reference - just use the name *)
          let v = make_var name id te.ty false in
          Ir.EVar v
      | _ ->
          let v = make_var name id te.ty false in
          Ir.EVar v)
  | TEVecGet (vec, idx) -> (
      match vec.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr state idx)
      | _ -> Ir.EArrayReadExpr (lower_expr state vec, lower_expr state idx))
  | TEArrGet (arr, idx) -> (
      match arr.te with
      | TEVar (name, _) -> Ir.EArrayRead (name, lower_expr state idx)
      | _ -> Ir.EArrayReadExpr (lower_expr state arr, lower_expr state idx))
  | TEFieldGet (r, field, _) -> Ir.ERecordField (lower_expr state r, field)
  | TEBinop (op, a, b) ->
      Ir.EBinop (ir_binop op te.ty, lower_expr state a, lower_expr state b)
  | TEUnop (op, a) -> Ir.EUnop (ir_unop op, lower_expr state a)
  | TEApp (fn, args) -> (
      let args_ir = List.map (lower_expr state) args in
      match fn.te with
      | TEVar (name, _) when Hashtbl.mem state.fun_map name ->
          (* Module-level function call *)
          if Hashtbl.mem state.lowered_funs name then
            (* Already lowered - use cached *)
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
          else if Hashtbl.mem state.lowering_stack name then
            (* Recursive call - emit by name *)
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
          else
            (* First time - lower the function *)
            let params, body = Hashtbl.find state.fun_map name in
            let ret_ty = repr body.ty in
            Hashtbl.add state.lowering_stack name () ;
            let fun_body_ir = lower_stmt state body in
            Hashtbl.remove state.lowering_stack name ;
            (* Use make_returning to add return statements without re-traversing *)
            let fun_body_ir = make_returning fun_body_ir in
            (* Convert tparam list to var list *)
            let hf_params =
              List.mapi
                (fun i (p : tparam) ->
                  make_var p.tparam_name i p.tparam_type false)
                params
            in
            let helper_func : Ir.helper_func =
              {
                hf_name = name;
                hf_params;
                hf_ret_type = elttype_of_typ ret_ty;
                hf_body = fun_body_ir;
              }
            in
            Hashtbl.add state.lowered_funs name helper_func ;
            state.lowered_funs_order <- name :: state.lowered_funs_order ;
            Ir.EApp (Ir.EVar (make_var name 0 fn.ty false), args_ir)
      | _ -> Ir.EApp (lower_expr state fn, args_ir))
  | TERecord (name, fields) ->
      (* Register the record type if not already registered *)
      if not (Hashtbl.mem state.types name) then begin
        let field_types =
          List.map (fun (n, e) -> (n, elttype_of_typ e.ty)) fields
        in
        Hashtbl.add state.types name field_types
      end ;
      Ir.ERecord (name, List.map (fun (n, e) -> (n, lower_expr state e)) fields)
  | TEConstr (ty_name, constr, arg) ->
      (* Register the variant type if not already registered.
         Get constructors from the expression's type (which has full variant info) *)
      if not (Hashtbl.mem state.variants ty_name) then begin
        match repr te.ty with
        | TVariant (_, constrs) ->
            let constr_types =
              List.map
                (fun (cname, ty_opt) ->
                  ( cname,
                    match ty_opt with
                    | None -> []
                    | Some ty -> [elttype_of_typ ty] ))
                constrs
            in
            Hashtbl.add state.variants ty_name constr_types
        | _ -> ()
      end ;
      let args = match arg with None -> [] | Some e -> [lower_expr state e] in
      Ir.EVariant (ty_name, constr, args)
  | TETuple exprs -> Ir.ETuple (List.map (lower_expr state) exprs)
  | TEGlobalRef (name, ty) ->
      let v = make_var name 0 ty false in
      Ir.EVar v
  | TEIntrinsicConst ref -> (
      match ref with
      | Sarek_env.IntrinsicRef (path, name) -> Ir.EIntrinsic (path, name, [])
      | Sarek_env.CorePrimitiveRef name -> Ir.EIntrinsic ([], name, []))
  | TEIntrinsicFun (ref, _conv, args) -> (
      match ref with
      | Sarek_env.IntrinsicRef (path, name) ->
          Ir.EIntrinsic (path, name, List.map (lower_expr state) args)
      | Sarek_env.CorePrimitiveRef name ->
          Ir.EIntrinsic ([], name, List.map (lower_expr state) args))
  (* If-then-else as expression (returns a value) *)
  | TEIf (cond, then_, Some else_) ->
      Ir.EIf
        (lower_expr state cond, lower_expr state then_, lower_expr state else_)
  | TEIf (cond, then_, None) ->
      (* No else branch - only valid for unit-returning expressions *)
      Ir.EIf (lower_expr state cond, lower_expr state then_, Ir.EConst Ir.CUnit)
  (* Match as expression *)
  | TEMatch (e, cases) ->
      let ir_cases =
        List.map
          (fun (pat, body) -> (lower_pattern pat, lower_expr state body))
          cases
      in
      Ir.EMatch (lower_expr state e, ir_cases)
  (* These expression forms require statement context - should be caught by typer *)
  | TEVecSet _ | TEArrSet _ | TEFieldSet _ | TEAssign _ | TELet _ | TELetRec _
  | TELetMut _ | TEFor _ | TEWhile _ | TESeq _ | TEReturn _ | TECreateArray _
  | TENative _ | TEPragma _ | TELetShared _ | TESuperstep _ | TEOpen _ ->
      failwith
        "Internal error: lower_expr called with statement-only expression. \
         This should have been caught by the type checker."

(** Convert a typed expression to IR statement *)
and lower_stmt (state : state) (te : texpr) : Ir.stmt =
  incr ir_lower_stmt_count ;
  match te.te with
  | TEUnit -> Ir.SEmpty
  | TESeq [] -> Ir.SEmpty
  | TESeq [e] -> lower_stmt state e
  | TESeq es -> Ir.SSeq (List.map (lower_stmt state) es)
  | TEVecSet (vec, idx, value) -> (
      match vec.te with
      | TEVar (name, _id) ->
          Ir.SAssign
            (Ir.LArrayElem (name, lower_expr state idx), lower_expr state value)
      | _ ->
          (* Complex base expression - use LArrayElemExpr *)
          Ir.SAssign
            ( Ir.LArrayElemExpr (lower_expr state vec, lower_expr state idx),
              lower_expr state value ))
  | TEArrSet (arr, idx, value) -> (
      match arr.te with
      | TEVar (name, _id) ->
          Ir.SAssign
            (Ir.LArrayElem (name, lower_expr state idx), lower_expr state value)
      | _ ->
          (* Complex base expression - use LArrayElemExpr *)
          Ir.SAssign
            ( Ir.LArrayElemExpr (lower_expr state arr, lower_expr state idx),
              lower_expr state value ))
  | TEFieldSet (r, field, _, value) ->
      let lv = lower_lvalue state r field in
      Ir.SAssign (lv, lower_expr state value)
  | TEAssign (name, id, value) ->
      let v = make_var name id value.ty true in
      Ir.SAssign (Ir.LVar v, lower_expr state value)
  | TELet (name, id, value, body) -> (
      match value.te with
      (* Special case: create_array - need proper array declaration *)
      | TECreateArray (size, elem_ty, mem) ->
          let size_ir = lower_expr state size in
          let v = make_var name id (TArr (elem_ty, mem)) false in
          let body_ir = lower_stmt state body in
          Ir.SLet
            ( v,
              Ir.EArrayCreate
                (elttype_of_typ elem_ty, size_ir, memspace_of_memspace mem),
              body_ir )
      (* Normal let binding *)
      | _ ->
          let v = make_var name id value.ty false in
          Ir.SLet (v, lower_expr state value, lower_stmt state body))
  | TELetMut (name, id, value, body) ->
      let v = make_var name id value.ty true in
      Ir.SLetMut (v, lower_expr state value, lower_stmt state body)
  | TELetRec (name, _id, params, fn_body, cont) ->
      (* Register function in fun_map for later inlining when called *)
      Hashtbl.add state.fun_map name (params, fn_body) ;
      lower_stmt state cont
  | TEIf (cond, then_, else_opt) ->
      Ir.SIf
        ( lower_expr state cond,
          lower_stmt state then_,
          Option.map (lower_stmt state) else_opt )
  | TEFor (var, id, lo, hi, dir, body) ->
      let v = make_var var id (TPrim TInt32) true in
      let ir_dir = match dir with Upto -> Ir.Upto | Downto -> Ir.Downto in
      Ir.SFor
        ( v,
          lower_expr state lo,
          lower_expr state hi,
          ir_dir,
          lower_stmt state body )
  | TEWhile (cond, body) ->
      Ir.SWhile (lower_expr state cond, lower_stmt state body)
  | TEMatch (e, cases) ->
      let ir_cases =
        List.map
          (fun (pat, body) -> (lower_pattern pat, lower_stmt state body))
          cases
      in
      Ir.SMatch (lower_expr state e, ir_cases)
  | TEReturn e -> Ir.SReturn (lower_expr state e)
  | TEPragma (opts, body) -> Ir.SPragma (opts, lower_stmt state body)
  | TELetShared (name, id, elem_ty, size_opt, body) ->
      (* Shared memory declaration: __shared__ type name[size]; or __local type name[size]; *)
      let size_ir =
        match size_opt with
        | Some size -> lower_expr state size
        | None -> Ir.EIntrinsic (["Sarek_stdlib"; "Gpu"], "block_dim_x", [])
      in
      let v = make_var name id (TArr (elem_ty, Sarek_types.Shared)) false in
      let elem_ir = elttype_of_typ elem_ty in
      (* Use EArrayCreate with Shared memspace - codegen will emit proper declaration *)
      Ir.SLet
        (v, Ir.EArrayCreate (elem_ir, size_ir, Ir.Shared), lower_stmt state body)
  | TESuperstep (_name, _divergent, step_body, cont) ->
      (* Wrap step_body in SBlock to create C scope for variable isolation *)
      Ir.SSeq
        [
          Ir.SBlock (lower_stmt state step_body);
          Ir.SBarrier;
          lower_stmt state cont;
        ]
  | TEOpen (_path, body) -> lower_stmt state body
  | TECreateArray (_size, _elem_ty, _mem) ->
      (* Standalone array creation - just emit unit *)
      Ir.SExpr (Ir.EConst Ir.CUnit)
  | TENative {gpu; ocaml} -> Ir.SNative {gpu; ocaml}
  (* Pure expressions as statements *)
  | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
  | TEVar _ | TEVecGet _ | TEArrGet _ | TEFieldGet _ | TEBinop _ | TEUnop _
  | TEApp _ | TERecord _ | TEConstr _ | TETuple _ | TEGlobalRef _
  | TEIntrinsicConst _ | TEIntrinsicFun _ ->
      Ir.SExpr (lower_expr state te)

and lower_lvalue (state : state) (r : texpr) (field : string) : Ir.lvalue =
  match r.te with
  | TEVar (name, id) ->
      let v = make_var name id r.ty false in
      Ir.LRecordField (Ir.LVar v, field)
  | TEFieldGet (base, inner_field, _) ->
      Ir.LRecordField (lower_lvalue state base inner_field, field)
  | TEVecGet (vec, idx) -> (
      match vec.te with
      | TEVar (name, _) ->
          Ir.LRecordField (Ir.LArrayElem (name, lower_expr state idx), field)
      | _ ->
          Ir.LRecordField
            ( Ir.LArrayElemExpr (lower_expr state vec, lower_expr state idx),
              field ))
  | TEArrGet (arr, idx) -> (
      match arr.te with
      | TEVar (name, _) ->
          Ir.LRecordField (Ir.LArrayElem (name, lower_expr state idx), field)
      | _ ->
          Ir.LRecordField
            ( Ir.LArrayElemExpr (lower_expr state arr, lower_expr state idx),
              field ))
  | _ ->
      failwith
        "Internal error: lower_lvalue called with non-lvalue expression. This \
         should have been caught by the type checker."

and lower_pattern (pat : tpattern) : Ir.pattern =
  match pat.tpat with
  | TPAny -> Ir.PWild
  | TPVar (name, _) -> Ir.PConstr ("", [name])
  | TPConstr (_ty_name, constr, arg_pat) ->
      let vars =
        match arg_pat with None -> [] | Some p -> extract_pattern_vars p
      in
      Ir.PConstr (constr, vars)
  | TPTuple pats ->
      Ir.PConstr ("tuple", List.concat_map extract_pattern_vars pats)

and extract_pattern_vars (pat : tpattern) : string list =
  match pat.tpat with
  | TPAny -> ["_"]
  | TPVar (name, _) -> [name]
  | TPConstr (_, _, Some p) -> extract_pattern_vars p
  | TPConstr (_, _, None) -> []
  | TPTuple pats -> List.concat_map extract_pattern_vars pats

(** Convert a kernel parameter to IR declaration *)
let lower_param (p : tparam) : Ir.decl =
  let elt = elttype_of_typ p.tparam_type in
  let v =
    {
      Ir.var_name = p.tparam_name;
      var_id = p.tparam_id;
      var_type = elt;
      var_mutable = false;
    }
  in
  if p.tparam_is_vec then
    let elem_ty =
      match repr p.tparam_type with TVec t -> elttype_of_typ t | _ -> elt
    in
    Ir.DParam (v, Some {arr_elttype = elem_ty; arr_memspace = Ir.Global})
  else Ir.DParam (v, None)

(** Lower a complete kernel *)
let lower_kernel (kernel : tkernel) : Ir.kernel * string list =
  (* Reset and log counters *)
  ir_lower_expr_count := 0 ;
  ir_lower_stmt_count := 0 ;
  let kern_name = Option.value kernel.tkern_name ~default:"anon" in
  Sarek_debug.log_to_file
    (Printf.sprintf "  [IR] lower_kernel start: %s" kern_name) ;
  (* Build fun_map from module items (local + registry) *)
  let fun_map = Hashtbl.create 8 in
  let all_mod_items = kernel.tkern_module_items in
  List.iter
    (function
      | TMFun (name, _, params, body) ->
          Hashtbl.replace fun_map name (params, body)
      | _ -> ())
    all_mod_items ;
  let state = create_state fun_map in

  (* Register record types from parameter types (especially vector element types) *)
  let rec register_types_from_typ ty =
    match repr ty with
    | TRecord (name, fields) ->
        if not (Hashtbl.mem state.types name) then begin
          let field_types =
            List.map (fun (n, t) -> (n, elttype_of_typ t)) fields
          in
          Hashtbl.add state.types name field_types
        end ;
        List.iter (fun (_, t) -> register_types_from_typ t) fields
    | TVec elem_ty -> register_types_from_typ elem_ty
    | TArr (elem_ty, _) -> register_types_from_typ elem_ty
    | TTuple tys -> List.iter register_types_from_typ tys
    | _ -> ()
  in
  List.iter
    (fun (p : tparam) -> register_types_from_typ p.tparam_type)
    kernel.tkern_params ;

  (* Lower module-level constants *)
  let module_items_ir =
    List.fold_right
      (fun item acc ->
        match item with
        | TMConst (name, id, ty, expr) ->
            let v = make_var name id ty false in
            Ir.SSeq [Ir.SLet (v, lower_expr state expr, Ir.SEmpty); acc]
        | TMFun _ -> acc)
      all_mod_items
      Ir.SEmpty
  in

  (* Generate type constructors *)
  let constructors =
    List.concat
      (List.map
         (function
           | TTypeRecord {tdecl_name; tdecl_fields; _} ->
               (* Strip mutability flag from fields *)
               let fields = List.map (fun (n, ty, _) -> (n, ty)) tdecl_fields in
               record_constructor_strings tdecl_name fields
           | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
               variant_constructor_strings tdecl_name tdecl_constructors)
         kernel.tkern_type_decls)
  in

  (* Lower body *)
  let body_ir = lower_stmt state kernel.tkern_body in
  Sarek_debug.log_to_file
    (Printf.sprintf
       "  [IR] lower_kernel done: %s (expr=%d, stmt=%d)"
       kern_name
       !ir_lower_expr_count
       !ir_lower_stmt_count) ;
  let full_body =
    match module_items_ir with
    | Ir.SEmpty -> body_ir
    | _ -> Ir.SSeq [module_items_ir; body_ir]
  in

  (* Collect types from state *)
  let types_list =
    Hashtbl.fold (fun name fields acc -> (name, fields) :: acc) state.types []
  in
  (* Collect variant types from state *)
  let variants_list =
    Hashtbl.fold
      (fun name constrs acc -> (name, constrs) :: acc)
      state.variants
      []
  in
  (* Collect helper functions from state, in dependency order *)
  let funcs_list =
    List.rev state.lowered_funs_order
    |> List.filter_map (fun name -> Hashtbl.find_opt state.lowered_funs name)
  in
  ( {
      Ir.kern_name = Option.value kernel.tkern_name ~default:"sarek_kern";
      (* "kernel" is reserved in OpenCL *)
      kern_params = List.map lower_param kernel.tkern_params;
      kern_locals = [];
      kern_body = full_body;
      kern_types = types_list;
      kern_variants = variants_list;
      kern_funcs = funcs_list;
      kern_native_fn = None;
      (* Native fn is added during quoting *)
    },
    constructors )

(** Get the return value declaration for a kernel *)
let lower_return_value (kernel : tkernel) : Ir.decl option =
  match repr kernel.tkern_return_type with
  | TPrim TUnit -> None
  | ty ->
      let v = make_var "result" 0 ty true in
      Some (Ir.DLocal (v, None))
