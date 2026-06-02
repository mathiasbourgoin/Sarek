(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * Native Code Generation - Main Module
 * ====================================
 *
 * Generates native OCaml code from Sarek's typed AST.
 * The generated code runs on CPU via Sarek_cpu_runtime.
 *
 * Organization:
 * - Expression generation (literals, operators, memory access, control flow)
 * - Module and type declaration generation  
 * - Kernel generation and wrapper functions
 *
 * See also:
 * - Sarek_native_helpers for utility functions and default values
 * - Sarek_native_intrinsics for type/intrinsic mapping
 ******************************************************************************)

open Ppxlib
open Sarek_typed_ast
open Sarek_types

(* Import helpers and intrinsics *)
open Sarek_native_helpers
open Sarek_native_intrinsics
open Sarek_native_gen_base
open Sarek_native_gen_expr

(** Generate OCaml expression from typed Sarek expression.
    @param ctx Generation context with mutable vars and inline types *)
let rec gen_expr_impl ~loc:_ ~ctx (te : texpr) : expression =
  let loc = ppxlib_loc_of_sarek te.te_loc in
  (* Helper to recursively generate, passing ctx *)
  let gen_expr ~loc e = gen_expr_impl ~loc ~ctx e in
  match te.te with
  (* Literals *)
  | TEUnit | TEBool _ | TEInt _ | TEInt32 _ | TEInt64 _ | TEFloat _ | TEDouble _
    ->
      gen_literal ~loc te
  (* Variables *)
  | TEVar (name, id) -> gen_variable ~loc ~ctx name id
  (* Memory access *)
  | TEVecGet _ | TEVecSet _ | TEArrGet _ | TEArrSet _ | TEFieldGet _
  | TEFieldSet _ ->
      gen_memory_access ~loc ~ctx ~gen_expr te
  (* Control flow *)
  | TEIf _ | TEFor _ | TEWhile _ -> gen_control_flow ~loc ~gen_expr te
  (* Binary operations *)
  | TEBinop (op, a, b) ->
      let a_e = gen_expr ~loc a in
      let b_e = gen_expr ~loc b in
      gen_binop ~loc op a_e b_e a.ty
  (* Unary operations *)
  | TEUnop (op, a) ->
      let a_e = gen_expr ~loc a in
      gen_unop ~loc op a_e a.ty
  (* Function application *)
  | TEApp (fn, args) ->
      let fn_e = gen_expr ~loc fn in
      (* When use_native_arg is true, vector arguments need #underlying
         to get the actual Vector.t for functions that expect it *)
      let gen_arg arg =
        let arg_e = gen_expr ~loc arg in
        if ctx.use_native_arg then
          match repr arg.ty with
          | TVec _ -> [%expr [%e arg_e]#underlying]
          | _ -> arg_e
        else arg_e
      in
      let args_e = List.map gen_arg args in
      Ast_builder.Default.pexp_apply
        ~loc
        fn_e
        (List.map (fun a -> (Nolabel, a)) args_e)
  (* Let bindings and assignment *)
  | TEAssign _ | TELet _ | TELetMut _ ->
      gen_let_binding ~loc ~ctx ~gen_expr ~gen_expr_impl te
  (* Sequence *)
  | TESeq exprs -> (
      let exprs_e = List.map (gen_expr ~loc) exprs in
      match exprs_e with
      | [] -> [%expr ()]
      | [e] -> e
      | es ->
          List.fold_right
            (fun e acc ->
              [%expr
                [%e e] ;
                [%e acc]])
            (List.rev (List.tl (List.rev es)))
            (List.hd (List.rev es)))
  (* Match *)
  | TEMatch (scrutinee, cases) ->
      let scrut_e = gen_expr ~loc scrutinee in
      let cases_e =
        List.map
          (fun (pat, body) ->
            let pat_e = gen_pattern_impl ~loc ~ctx pat in
            let body_e = gen_expr ~loc body in
            Ast_builder.Default.case ~lhs:pat_e ~guard:None ~rhs:body_e)
          cases
      in
      Ast_builder.Default.pexp_match ~loc scrut_e cases_e
  (* Data structures *)
  | TERecord _ | TEConstr _ | TETuple _ | TECreateArray _ ->
      gen_data_structure ~loc ~ctx ~gen_expr te
  (* Special expressions *)
  | TEReturn _ | TEGlobalRef _ | TENative _ | TEPragma _ | TEOpen _ ->
      gen_special_expr ~loc ~gen_expr te
  (* Intrinsic constant - thread indices, etc. *)
  | TEIntrinsicConst ref -> gen_intrinsic_const ~loc ~gen_mode:ctx.gen_mode ref
  (* Intrinsic function - math functions, barriers, etc. *)
  | TEIntrinsicFun (ref, _convergence, args) ->
      (* When use_native_arg is true, vector arguments need #underlying *)
      let gen_arg arg =
        let arg_e = gen_expr ~loc arg in
        if ctx.use_native_arg then
          match repr arg.ty with
          | TVec _ -> [%expr [%e arg_e]#underlying]
          | _ -> arg_e
        else arg_e
      in
      let args_e = List.map gen_arg args in
      gen_intrinsic_fun ~loc ~gen_mode:ctx.gen_mode ref args_e
  (* Parallel constructs *)
  | TELetShared _ | TESuperstep _ | TELetRec _ ->
      gen_parallel_construct
        ?current_module:ctx.current_module
        ~inline_types:ctx.inline_types
        ~loc
        ~gen_expr
        te

(** Generate pattern from typed pattern. Takes context to detect same-module
    types that shouldn't be qualified. *)
and gen_pattern_impl ~loc:_ ~ctx (tpat : tpattern) : pattern =
  let loc = ppxlib_loc_of_sarek tpat.tpat_loc in
  match tpat.tpat with
  | TPAny -> Ast_builder.Default.ppat_any ~loc
  | TPVar (name, _id) ->
      (* Use original name for pattern variables *)
      Ast_builder.Default.ppat_var ~loc {txt = name; loc}
  | TPConstr (type_name, constr_name, arg) ->
      let arg_p = Option.map (gen_pattern_impl ~loc ~ctx) arg in
      (* Qualify constructor with module path from type_name if present.
         For "Geometry_lib.shape", we need Geometry_lib.Circle, not Circle.
         For inline types or same-module types, use unqualified name. *)
      let constr_lid =
        match String.rindex_opt type_name '.' with
        | Some idx ->
            (* Check if this is from the current module - use unqualified if so *)
            if is_same_module ctx type_name then Lident constr_name
            else
              let module_path = String.sub type_name 0 idx in
              let parts = String.split_on_char '.' module_path in
              let rec build_lid = function
                | [] -> Lident constr_name
                | [m] -> Ldot (Lident m, constr_name)
                | m :: rest -> Ldot (build_lid rest, m)
              in
              build_lid (List.rev parts)
        | None ->
            (* Inline type - use unqualified name *)
            Lident constr_name
      in
      Ast_builder.Default.ppat_construct ~loc {txt = constr_lid; loc} arg_p
  | TPTuple pats ->
      let pats_p = List.map (gen_pattern_impl ~loc ~ctx) pats in
      Ast_builder.Default.ppat_tuple ~loc pats_p

(** Generate binary operation *)
and gen_binop ~loc op a b ty : expression =
  let ty_repr = repr ty in
  match op with
  | Sarek_ast.Add -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr [%e a] +. [%e b]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.add [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.add [%e a] [%e b]]
      | _ -> [%expr [%e a] + [%e b]])
  | Sarek_ast.Sub -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr [%e a] -. [%e b]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.sub [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.sub [%e a] [%e b]]
      | _ -> [%expr [%e a] - [%e b]])
  | Sarek_ast.Mul -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr [%e a] *. [%e b]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.mul [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.mul [%e a] [%e b]]
      | _ -> [%expr [%e a] * [%e b]])
  | Sarek_ast.Div -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr [%e a] /. [%e b]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.div [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.div [%e a] [%e b]]
      | _ -> [%expr [%e a] / [%e b]])
  | Sarek_ast.Mod -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.rem [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.rem [%e a] [%e b]]
      | _ -> [%expr [%e a] mod [%e b]])
  | Sarek_ast.And -> [%expr [%e a] && [%e b]]
  | Sarek_ast.Or -> [%expr [%e a] || [%e b]]
  (* Comparison operators work polymorphically in OCaml *)
  | Sarek_ast.Eq -> [%expr [%e a] = [%e b]]
  | Sarek_ast.Ne -> [%expr [%e a] <> [%e b]]
  | Sarek_ast.Lt -> [%expr [%e a] < [%e b]]
  | Sarek_ast.Gt -> [%expr [%e a] > [%e b]]
  | Sarek_ast.Le -> [%expr [%e a] <= [%e b]]
  | Sarek_ast.Ge -> [%expr [%e a] >= [%e b]]
  (* Bitwise operations *)
  | Sarek_ast.Land -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.logand [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.logand [%e a] [%e b]]
      | _ -> [%expr [%e a] land [%e b]])
  | Sarek_ast.Lor -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.logor [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.logor [%e a] [%e b]]
      | _ -> [%expr [%e a] lor [%e b]])
  | Sarek_ast.Lxor -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.logxor [%e a] [%e b]]
      | TReg Int64 -> [%expr Int64.logxor [%e a] [%e b]]
      | _ -> [%expr [%e a] lxor [%e b]])
  | Sarek_ast.Lsl -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 ->
          [%expr Int32.shift_left [%e a] (Int32.to_int [%e b])]
      | TReg Int64 -> [%expr Int64.shift_left [%e a] (Int64.to_int [%e b])]
      | _ -> [%expr [%e a] lsl [%e b]])
  | Sarek_ast.Lsr -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 ->
          [%expr Int32.shift_right_logical [%e a] (Int32.to_int [%e b])]
      | TReg Int64 ->
          [%expr Int64.shift_right_logical [%e a] (Int64.to_int [%e b])]
      | _ -> [%expr [%e a] lsr [%e b]])
  | Sarek_ast.Asr -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 ->
          [%expr Int32.shift_right [%e a] (Int32.to_int [%e b])]
      | TReg Int64 -> [%expr Int64.shift_right [%e a] (Int64.to_int [%e b])]
      | _ -> [%expr [%e a] asr [%e b]])

(** Generate unary operation *)
and gen_unop ~loc op a ty : expression =
  let ty_repr = repr ty in
  match op with
  | Sarek_ast.Neg -> (
      match ty_repr with
      | TReg Float32 | TReg Float64 -> [%expr -.[%e a]]
      | TReg Int | TPrim TInt32 -> [%expr Int32.neg [%e a]]
      | TReg Int64 -> [%expr Int64.neg [%e a]]
      | _ -> [%expr -[%e a]])
  | Sarek_ast.Not -> [%expr not [%e a]]
  | Sarek_ast.Lnot -> (
      match ty_repr with
      | TReg Int | TPrim TInt32 -> [%expr Int32.lognot [%e a]]
      | TReg Int64 -> [%expr Int64.lognot [%e a]]
      | _ -> [%expr lnot [%e a]])

(** Extract module name from a Sarek location (file path). For
    "/path/to/test_registered_variant.ml", returns "Test_registered_variant". *)
let module_name_of_sarek_loc (loc : Sarek_ast.loc) : string =
  let file = loc.loc_file in
  let base = Filename.(remove_extension (basename file)) in
  String.capitalize_ascii base

(** Top-level entry point for generating expressions. Starts with empty context.
*)
let gen_expr ~loc e : expression = gen_expr_impl ~loc ~ctx:empty_ctx e

(** Generate expression with inline types context for first-class modules. *)
let gen_expr_with_inline_types ~loc ~inline_type_names ~current_module e :
    expression =
  let ctx = {empty_ctx with inline_types = inline_type_names; current_module} in
  gen_expr_impl ~loc ~ctx e

(** {1 Module Item Generation} *)

(** Generate a module-level function (TMFun) as a let binding. *)
let gen_module_fun ~loc (name : string) (params : tparam list) (body : texpr) :
    pattern * expression =
  let fn_pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
  let body_e = gen_expr ~loc body in

  (* Build function from parameters - use parameter names with type constraints
     to help OCaml resolve record fields and other types *)
  let fn_e =
    List.fold_right
      (fun param acc ->
        let var_pat =
          Ast_builder.Default.ppat_var ~loc {txt = param.tparam_name; loc}
        in
        let ty = core_type_of_typ ~loc param.tparam_type in
        let param_pat = Ast_builder.Default.ppat_constraint ~loc var_pat ty in
        [%expr fun [%p param_pat] -> [%e acc]])
      params
      body_e
  in
  (fn_pat, fn_e)

(** Generate a module-level constant (TMConst) as a let binding. *)
let gen_module_const ~loc (name : string) (_id : int) (_typ : typ)
    (value : texpr) : pattern * expression =
  (* Use the constant name, not the ID *)
  let const_pat = Ast_builder.Default.ppat_var ~loc {txt = name; loc} in
  let value_e = gen_expr ~loc value in
  (const_pat, value_e)

(** {1 Type Declaration Generation} *)

(** Generate a type declaration for a record type *)
let gen_type_decl_record ~loc (name : string)
    (fields : (string * typ * bool) list) : structure_item =
  let field_decls =
    List.map
      (fun (fname, fty, is_mutable) ->
        let ty = core_type_of_typ ~loc fty in
        Ast_builder.Default.label_declaration
          ~loc
          ~name:{txt = fname; loc}
          ~mutable_:(if is_mutable then Mutable else Immutable)
          ~type_:ty)
      fields
  in
  Ast_builder.Default.pstr_type
    ~loc
    Recursive
    [
      Ast_builder.Default.type_declaration
        ~loc
        ~name:{txt = name; loc}
        ~params:[]
        ~cstrs:[]
        ~kind:(Ptype_record field_decls)
        ~private_:Public
        ~manifest:None;
    ]

(** Generate a type declaration for a variant type *)
let gen_type_decl_variant ~loc (name : string)
    (constrs : (string * typ option) list) : structure_item =
  let constr_decls =
    List.map
      (fun (cname, arg_opt) ->
        let args =
          match arg_opt with
          | None -> Pcstr_tuple []
          | Some ty -> Pcstr_tuple [core_type_of_typ ~loc ty]
        in
        Ast_builder.Default.constructor_declaration
          ~loc
          ~name:{txt = cname; loc}
          ~args
          ~res:None)
      constrs
  in
  Ast_builder.Default.pstr_type
    ~loc
    Recursive
    [
      Ast_builder.Default.type_declaration
        ~loc
        ~name:{txt = name; loc}
        ~params:[]
        ~cstrs:[]
        ~kind:(Ptype_variant constr_decls)
        ~private_:Public
        ~manifest:None;
    ]

(** Generate a structure item from a typed type declaration *)
let gen_type_decl_item ~loc (decl : ttype_decl) : structure_item =
  match decl with
  | TTypeRecord {tdecl_name; tdecl_fields; _} ->
      gen_type_decl_record ~loc tdecl_name tdecl_fields
  | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
      gen_type_decl_variant ~loc tdecl_name tdecl_constructors

(** Wrap expression with module item bindings. *)
let wrap_module_items ~loc (items : tmodule_item list) (body : expression) :
    expression =
  List.fold_right
    (fun item acc ->
      match item with
      | TMFun (name, is_rec, params, item_body) ->
          let pat, expr = gen_module_fun ~loc name params item_body in
          if is_rec then
            (* Generate let rec for recursive functions *)
            [%expr
              let rec [%p pat] = [%e expr] in
              [%e acc]]
          else
            [%expr
              let [%p pat] = [%e expr] in
              [%e acc]]
      | TMConst (name, id, typ, value) ->
          let pat, expr = gen_module_const ~loc name id typ value in
          [%expr
            let [%p pat] = [%e expr] in
            [%e acc]])
    items
    body

(** {1 First-Class Module Approach for Inline Types}

    To avoid "type escapes its scope" errors, we use first-class modules to
    encapsulate inline types. The approach:

    1. Generate a module signature KERNEL_TYPES with abstract types and
    accessors 2. Generate a concrete module implementation with the actual types
    3. Transform the kernel body to use T.get_field and T.make_type accessors 4.
    Pass (module T : KERNEL_TYPES) as a parameter to the kernel

    This keeps the concrete types hidden behind an existential type. *)

(** Generate a concrete module implementation for inline types (types only).
    Example for record type point with fields x and y: struct type point =
    record with fields x: float and y: float end

    Note: We only generate type declarations here, not accessor functions. The
    accessor functions are generated as object methods by gen_types_object. *)
let gen_module_impl ~loc (decls : ttype_decl list) : module_expr =
  let struct_items =
    List.map
      (fun decl ->
        match decl with
        | TTypeRecord {tdecl_name; tdecl_fields; _} ->
            gen_type_decl_record ~loc tdecl_name tdecl_fields
        | TTypeVariant {tdecl_name; tdecl_constructors; _} ->
            gen_type_decl_variant ~loc tdecl_name tdecl_constructors)
      decls
  in
  Ast_builder.Default.pmod_structure ~loc struct_items

(** Get only inline type declarations (those without module prefix).
    External/registered types have qualified names like "Module.type". *)
let inline_type_decls (decls : ttype_decl list) : ttype_decl list =
  List.filter
    (fun decl ->
      let name =
        match decl with
        | TTypeRecord {tdecl_name; _} -> tdecl_name
        | TTypeVariant {tdecl_name; _} -> tdecl_name
      in
      (* Only include types without '.' - these are inline definitions *)
      not (String.contains name '.'))
    decls

(** Check if a kernel has inline types that need first-class module handling *)
let has_inline_types (kernel : tkernel) : bool =
  inline_type_decls kernel.tkern_type_decls <> []

(* Kernel generation moved to Sarek_native_gen_kernel *)
