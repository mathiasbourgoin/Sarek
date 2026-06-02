(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This module parses OCaml AST (from ppxlib) into Sarek_ast.
 ******************************************************************************)

open Ppxlib
open Sarek_parse_helpers

(** Re-export helpers used by external callers *)
exception Parse_error_exn = Sarek_parse_helpers.Parse_error_exn

let extract_name_from_pattern = Sarek_parse_helpers.extract_name_from_pattern

let extract_param_from_pattern = Sarek_parse_helpers.extract_param_from_pattern

let extract_type_from_pattern = Sarek_parse_helpers.extract_type_from_pattern

let collect_fun_params = Sarek_parse_helpers.collect_fun_params

let pattern_of_param = Sarek_parse_helpers.pattern_of_param

let parse_binop = Sarek_parse_helpers.parse_binop

let parse_unop = Sarek_parse_helpers.parse_unop

let parse_type = Sarek_parse_helpers.parse_type

(** Parse let%shared: let%shared name : type [= size] in body Syntax: let%shared
    tile : float32 array in body let%shared tile : float32 array = 64 in body *)
let rec parse_let_shared parse_expr (expr : expression) : Sarek_ast.expr_desc =
  match expr.pexp_desc with
  (* Pattern: let name : type = size in body
     Note: when size is (), we treat it as no size specified *)
  | Pexp_let
      ( Nonrecursive,
        [
          {
            pvb_pat = {ppat_desc = Ppat_constraint (name_pat, elem_type); _};
            pvb_expr = size_expr;
            _;
          };
        ],
        body_expr ) ->
      let name =
        match name_pat.ppat_desc with
        | Ppat_var {txt; _} -> txt
        | _ ->
            raise
              (Parse_error_exn ("Expected variable name", name_pat.ppat_loc))
      in
      let elem_ty = parse_type elem_type in
      (* Check if size is unit - if so, no size specified *)
      let size =
        match size_expr.pexp_desc with
        | Pexp_construct ({txt = Lident "()"; _}, None) -> None
        | _ -> Some (parse_expr size_expr)
      in
      let body = parse_expr body_expr in
      Sarek_ast.ELetShared (name, elem_ty, size, body)
  (* Shorthand: name : type in body (no explicit let) *)
  | Pexp_constraint
      ({pexp_desc = Pexp_sequence (name_expr, body_expr); _}, elem_type) -> (
      match name_expr.pexp_desc with
      | Pexp_ident {txt = Lident name; _} ->
          let elem_ty = parse_type elem_type in
          let body = parse_expr body_expr in
          Sarek_ast.ELetShared (name, elem_ty, None, body)
      | _ ->
          raise
            (Parse_error_exn
               ("Expected identifier for shared array name", expr.pexp_loc)))
  | _ ->
      raise
        (Parse_error_exn
           ("Expected 'let%shared name : type [= size] in body'", expr.pexp_loc))

(** Parse let%superstep: let%superstep [~divergent] name = body in cont Syntax:
    let%superstep load = tile.(i) <- v in cont let%superstep ~divergent final =
    ... in cont *)
and parse_superstep parse_expr (expr : expression) : Sarek_ast.expr_desc =
  match expr.pexp_desc with
  (* Pattern: let name = body in cont *)
  | Pexp_let
      ( Nonrecursive,
        [{pvb_pat; pvb_expr = step_body; pvb_attributes; _}],
        cont_expr ) ->
      let name =
        match extract_name_from_pattern pvb_pat with
        | Some n -> n
        | None ->
            raise
              (Parse_error_exn ("Expected superstep name", pvb_pat.ppat_loc))
      in
      (* Check for ~divergent attribute *)
      let divergent =
        List.exists
          (fun (attr : attribute) -> attr.attr_name.txt = "divergent")
          pvb_attributes
      in
      let body = parse_expr step_body in
      let cont = parse_expr cont_expr in
      Sarek_ast.ESuperstep (name, divergent, body, cont)
  | _ ->
      raise
        (Parse_error_exn
           ("Expected 'let%superstep name = body in cont'", expr.pexp_loc))

(** Parse an expression *)
and parse_expression (expr : expression) : Sarek_ast.expr =
  let loc = loc_of_ppxlib expr.pexp_loc in
  let e =
    match expr.pexp_desc with
    (* Unit *)
    | Pexp_construct ({txt = Lident "()"; _}, None) -> Sarek_ast.EUnit
    (* Boolean literals *)
    | Pexp_construct ({txt = Lident "true"; _}, None) -> Sarek_ast.EBool true
    | Pexp_construct ({txt = Lident "false"; _}, None) -> Sarek_ast.EBool false
    (* Integer literals *)
    | Pexp_constant (Pconst_integer (s, Some 'l')) ->
        Sarek_ast.EInt32 (Int32.of_string s)
    | Pexp_constant (Pconst_integer (s, Some 'L')) ->
        Sarek_ast.EInt64 (Int64.of_string s)
    | Pexp_constant (Pconst_integer (s, None)) ->
        Sarek_ast.EInt (int_of_string s)
    (* Float literals *)
    | Pexp_constant (Pconst_float (s, _)) ->
        Sarek_ast.EFloat (float_of_string s)
    (* Variables *)
    | Pexp_ident {txt = Lident name; _} -> Sarek_ast.EVar name
    (* Module-qualified identifiers: Module.name -> "Module.name"
       This preserves the qualified name for cross-module function lookup.
       The typer will look up "Module.name" in the environment/registry. *)
    | Pexp_ident {txt = Ldot (Lident modname, name); _} ->
        Sarek_ast.EVar (modname ^ "." ^ name)
    (* Vector/array access: e.(i) or e.[i] *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident "Array.get"; _}; _},
          [(Nolabel, arr); (Nolabel, idx)] ) ->
        Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Ldot (Lident "Array", "get"); _}; _},
          [(Nolabel, arr); (Nolabel, idx)] ) ->
        Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)
    (* Vector/array set: e.(i) <- x *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Ldot (Lident "Array", "set"); _}; _},
          [(Nolabel, arr); (Nolabel, idx); (Nolabel, value)] ) ->
        Sarek_ast.EArrSet
          (parse_expression arr, parse_expression idx, parse_expression value)
    (* Custom indexing: v.%[i] -> EArrGet *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident ".%[]"; _}; _},
          [(Nolabel, arr); (Nolabel, idx)] ) ->
        Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)
    (* Custom indexing: v.%[i] <- x -> EArrSet *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident ".%[]<-"; _}; _},
          [(Nolabel, arr); (Nolabel, idx); (Nolabel, value)] ) ->
        Sarek_ast.EArrSet
          (parse_expression arr, parse_expression idx, parse_expression value)
    (* Mutable assignment: x := v *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident ":="; _}; _},
          [(Nolabel, lhs); (Nolabel, rhs)] ) -> (
        match lhs.pexp_desc with
        | Pexp_ident {txt = Lident name; _} ->
            Sarek_ast.EAssign (name, parse_expression rhs)
        | _ ->
            raise
              (Parse_error_exn
                 ("Expected variable on left-hand side of :=", lhs.pexp_loc)))
    (* a.(i) syntax - array access *)
    | Pexp_apply (arr, [(Nolabel, idx)]) when is_array_access expr ->
        Sarek_ast.EArrGet (parse_expression arr, parse_expression idx)
    (* Pragma - pragma ["opt1"; "opt2"] body - must come before binary ops *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident "pragma"; _}; _},
          [(Nolabel, opts_expr); (Nolabel, body)] ) ->
        let rec collect_strings acc expr =
          match expr.pexp_desc with
          | Pexp_construct ({txt = Lident "[]"; _}, None) -> List.rev acc
          | Pexp_construct
              ({txt = Lident "::"; _}, Some {pexp_desc = Pexp_tuple [hd; tl]; _})
            -> (
              match hd.pexp_desc with
              | Pexp_constant (Pconst_string (s, _, _)) ->
                  collect_strings (s :: acc) tl
              | _ ->
                  raise
                    (Parse_error_exn
                       ("pragma options must be strings", hd.pexp_loc)))
          | _ ->
              raise
                (Parse_error_exn
                   ("pragma expects a list of strings", opts_expr.pexp_loc))
        in
        let opts = collect_strings [] opts_expr in
        Sarek_ast.EPragma (opts, parse_expression body)
    (* create_array size memspace - special form for local/shared arrays
       Must come before binary operators since it has 2 arguments *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident "create_array"; _}; _},
          [(Nolabel, size_expr); (Nolabel, mem_expr)] ) ->
        let size = parse_expression size_expr in
        let mem =
          match mem_expr.pexp_desc with
          | Pexp_construct ({txt = Lident "Local"; _}, None) -> Sarek_ast.Local
          | Pexp_construct ({txt = Lident "Shared"; _}, None) ->
              Sarek_ast.Shared
          | Pexp_construct ({txt = Lident "Global"; _}, None) ->
              Sarek_ast.Global
          | _ ->
              raise
                (Parse_error_exn
                   ( "create_array expects Local, Shared, or Global as memspace",
                     mem_expr.pexp_loc ))
        in
        (* Type comes from let binding annotation, use type variable for inference *)
        Sarek_ast.ECreateArray (size, Sarek_ast.TEVar "_infer", mem)
    (* Binary operators - exclude create_array which is handled above *)
    | Pexp_apply
        ( {pexp_desc = Pexp_ident {txt = Lident op; _}; _},
          [(Nolabel, e1); (Nolabel, e2)] )
      when op <> "create_array" -> (
        match parse_binop op with
        | Some binop ->
            Sarek_ast.EBinop (binop, parse_expression e1, parse_expression e2)
        | None ->
            (* Regular function application with infix *)
            Sarek_ast.EApp
              ( parse_expression
                  {
                    expr with
                    pexp_desc =
                      Pexp_ident {txt = Lident op; loc = expr.pexp_loc};
                  },
                [parse_expression e1; parse_expression e2] ))
    (* Unary operators *)
    | Pexp_apply
        ({pexp_desc = Pexp_ident {txt = Lident op; _}; _}, [(Nolabel, e)])
      when parse_unop op <> None -> (
        match parse_unop op with
        | Some unop -> Sarek_ast.EUnop (unop, parse_expression e)
        | None ->
            (* Should be unreachable due to when guard, but handle gracefully *)
            raise
              (Parse_error_exn
                 ( "Internal error: unary operator check inconsistency",
                   expr.pexp_loc )))
    (* Function application *)
    | Pexp_apply (fn, args) ->
        let fn_expr = parse_expression fn in
        let arg_exprs = List.map (fun (_, e) -> parse_expression e) args in
        Sarek_ast.EApp (fn_expr, arg_exprs)
    (* Let binding *)
    | Pexp_let (Nonrecursive, [{pvb_pat; pvb_expr; _}], body) ->
        let name =
          match extract_name_from_pattern pvb_pat with
          | Some n -> n
          | None ->
              raise
                (Parse_error_exn ("Expected variable pattern", pvb_pat.ppat_loc))
        in
        let ty = extract_type_from_pattern pvb_pat in
        (* Detect function definitions and emit ELetRec for local functions *)
        let fun_params, fun_body = collect_fun_params pvb_expr in
        if fun_params <> [] then
          match fun_body with
          | Some (Fun_body body_expr) ->
              let parsed_params =
                List.map
                  (fun p -> extract_param_from_pattern (pattern_of_param p))
                  fun_params
              in
              let fn_body = parse_expression body_expr in
              Sarek_ast.ELetRec
                (name, parsed_params, ty, fn_body, parse_expression body)
          | Some (Fun_cases _) ->
              raise
                (Parse_error_exn
                   ( "Pattern-matching functions not supported in let bindings",
                     pvb_expr.pexp_loc ))
          | None ->
              raise
                (Parse_error_exn ("Expected function body", pvb_expr.pexp_loc))
        else
          let mut_expr =
            match pvb_expr.pexp_desc with
            | Pexp_apply
                ( {pexp_desc = Pexp_ident {txt = Lident "mut"; _}; _},
                  [(Nolabel, inner)] ) ->
                Some inner
            | _ -> None
          in
          let is_mutable = Option.is_some mut_expr in
          let value_expr =
            match mut_expr with Some inner -> inner | None -> pvb_expr
          in
          if is_mutable then
            Sarek_ast.ELetMut
              (name, ty, parse_expression value_expr, parse_expression body)
          else
            Sarek_ast.ELet
              (name, ty, parse_expression value_expr, parse_expression body)
    (* If-then-else *)
    | Pexp_ifthenelse (cond, then_e, else_opt) ->
        Sarek_ast.EIf
          ( parse_expression cond,
            parse_expression then_e,
            Option.map parse_expression else_opt )
    (* For loop *)
    | Pexp_for ({ppat_desc = Ppat_var {txt = var; _}; _}, lo, hi, dir, body) ->
        let d =
          match dir with Upto -> Sarek_ast.Upto | Downto -> Sarek_ast.Downto
        in
        Sarek_ast.EFor
          ( var,
            parse_expression lo,
            parse_expression hi,
            d,
            parse_expression body )
    (* While loop *)
    | Pexp_while (cond, body) ->
        Sarek_ast.EWhile (parse_expression cond, parse_expression body)
    (* Sequence *)
    | Pexp_sequence (e1, e2) ->
        Sarek_ast.ESeq (parse_expression e1, parse_expression e2)
    (* Match *)
    | Pexp_match (scrutinee, cases) ->
        let parsed_cases =
          List.map
            (fun case ->
              let pat = parse_pattern case.pc_lhs in
              let body = parse_expression case.pc_rhs in
              (pat, body))
            cases
        in
        Sarek_ast.EMatch (parse_expression scrutinee, parsed_cases)
    (* Record construction *)
    | Pexp_record (fields, _base) ->
        let parsed_fields =
          List.map
            (fun ({txt; _}, e) ->
              let name =
                match txt with Lident n -> n | Ldot (_, n) -> n | _ -> "field"
              in
              (name, parse_expression e))
            fields
        in
        Sarek_ast.ERecord (None, parsed_fields)
    (* Field access *)
    | Pexp_field (record, {txt = Lident field; _}) ->
        Sarek_ast.EFieldGet (parse_expression record, field)
    (* Field set (via setfield) *)
    | Pexp_setfield (record, {txt = Lident field; _}, value) ->
        Sarek_ast.EFieldSet
          (parse_expression record, field, parse_expression value)
    (* Constructor application *)
    | Pexp_construct ({txt = Lident name; _}, arg_opt) ->
        Sarek_ast.EConstr (name, Option.map parse_expression arg_opt)
    (* Tuple *)
    | Pexp_tuple es -> Sarek_ast.ETuple (List.map parse_expression es)
    (* Type annotation *)
    | Pexp_constraint (e, ty) ->
        Sarek_ast.ETyped (parse_expression e, parse_type ty)
    (* Open expression *)
    | Pexp_open ({popen_expr = {pmod_desc = Pmod_ident {txt; _}; _}; _}, e) ->
        let path =
          match txt with
          | Lident n -> [n]
          | Ldot (Lident m, n) -> [m; n]
          | _ -> []
        in
        Sarek_ast.EOpen (path, parse_expression e)
    (* Lambda - for local functions in kernels *)
    | _ when is_function_expression expr ->
        raise
          (Parse_error_exn
             ( "Standalone lambda expressions are not supported in kernels; \
                use let-bound functions",
               expr.pexp_loc ))
    (* Extension point: [%global name] - reference to OCaml value *)
    | Pexp_extension
        ( {txt = "global"; _},
          PStr
            [
              {
                pstr_desc =
                  Pstr_eval
                    ({pexp_desc = Pexp_ident {txt = Lident name; _}; _}, _);
                _;
              };
            ] ) ->
        Sarek_ast.EGlobalRef name
    (* Extension point: [%native gpu_fun, ocaml_expr]
       Inline device code with OCaml fallback for interpreter/native runtimes.
       gpu_fun: (fun dev -> "cuda/opencl code")
       ocaml_expr: OCaml expression to execute on interpreter/native *)
    | Pexp_extension
        ({txt = "native"; _}, PStr [{pstr_desc = Pstr_eval (inner_expr, _); _}])
      -> (
        (* Parse tuple (gpu_fun, ocaml_expr) *)
        match inner_expr.pexp_desc with
        | Pexp_tuple [gpu; ocaml] -> Sarek_ast.ENative {gpu; ocaml}
        | _ ->
            raise
              (Parse_error_exn
                 ( "[%native] requires a tuple: (fun dev -> ..., ocaml_fallback)",
                   expr.pexp_loc )))
    (* Extension: let%shared name : type [= size] in body *)
    | Pexp_extension
        ({txt = "shared"; _}, PStr [{pstr_desc = Pstr_eval (inner_expr, _); _}])
      ->
        parse_let_shared parse_expression inner_expr
    (* Extension: let%superstep [~divergent] name = body in cont *)
    | Pexp_extension
        ( {txt = "superstep"; _},
          PStr [{pstr_desc = Pstr_eval (inner_expr, _); _}] ) ->
        parse_superstep parse_expression inner_expr
    | _ -> raise (Parse_error_exn ("Unsupported expression", expr.pexp_loc))
  in
  {Sarek_ast.e; Sarek_ast.expr_loc = loc}

(** Check if an expression is an array access *)
and is_array_access (_expr : expression) : bool =
  (* This is a simplified check - real implementation would need more context *)
  false

(** Parse a function expression into a kernel *)
let parse_kernel_function (expr : expression) : Sarek_ast.kernel =
  let loc = loc_of_ppxlib expr.pexp_loc in
  let params, body = collect_fun_params expr in
  match body with
  | Some (Fun_cases _) ->
      raise
        (Parse_error_exn
           ("Pattern-matching functions not supported as kernels", expr.pexp_loc))
  | Some (Fun_body body_expr) ->
      if params = [] then
        raise
          (Parse_error_exn
             ("Kernel must have at least one parameter", expr.pexp_loc)) ;
      let parsed_params =
        List.map
          (fun p -> extract_param_from_pattern (pattern_of_param p))
          params
      in
      let body = parse_expression body_expr in
      {
        Sarek_ast.kern_name = None;
        kern_types = [];
        kern_module_items = [];
        kern_external_item_count = 0;
        kern_params = parsed_params;
        kern_body = body;
        kern_loc = loc;
      }
  | None -> raise (Parse_error_exn ("Kernel must be a function", expr.pexp_loc))

(** Parse from ppxlib payload *)
let parse_payload (payload : expression) : Sarek_ast.kernel =
  let parse_module_items_from_structure items =
    List.fold_left
      (fun (types_acc, mods_acc) (item : structure_item) ->
        match item.pstr_desc with
        | Pstr_type (_, decls) ->
            let tdecls =
              List.map
                (fun (td : type_declaration) ->
                  let loc = td.ptype_loc in
                  match td.ptype_kind with
                  | Ptype_record labels ->
                      Sarek_ast.Type_record
                        {
                          tdecl_name = td.ptype_name.txt;
                          tdecl_module = None;
                          tdecl_fields = parse_record_fields labels;
                          tdecl_loc = loc_of_ppxlib loc;
                        }
                  | Ptype_variant constrs ->
                      Sarek_ast.Type_variant
                        {
                          tdecl_name = td.ptype_name.txt;
                          tdecl_module = None;
                          tdecl_constructors =
                            parse_variant_constructors constrs;
                          tdecl_loc = loc_of_ppxlib loc;
                        }
                  | _ ->
                      raise
                        (Parse_error_exn
                           ( "Unsupported type declaration in kernel payload",
                             loc )))
                decls
            in
            (List.rev_append tdecls types_acc, mods_acc)
        | Pstr_value (rec_flag, vbs) ->
            let is_recursive = rec_flag = Recursive in
            let mods =
              List.fold_left
                (fun acc vb ->
                  let name =
                    match extract_name_from_pattern vb.pvb_pat with
                    | Some n -> n
                    | None ->
                        raise
                          (Parse_error_exn
                             ("Expected variable pattern", vb.pvb_pat.ppat_loc))
                  in
                  let ty = extract_type_from_pattern vb.pvb_pat in
                  let params, body =
                    match collect_fun_params vb.pvb_expr with
                    | params, Some (Fun_body fn_body) when params <> [] ->
                        (params, fn_body)
                    | _, Some (Fun_cases _) ->
                        raise
                          (Parse_error_exn
                             ( "Pattern-matching functions not supported in \
                                module items",
                               vb.pvb_expr.pexp_loc ))
                    | _ -> ([], vb.pvb_expr)
                  in
                  if params <> [] then
                    let parsed_params =
                      List.map
                        (fun p ->
                          extract_param_from_pattern (pattern_of_param p))
                        params
                    in
                    let fn_body = parse_expression body in
                    Sarek_ast.MFun (name, is_recursive, parsed_params, fn_body)
                    :: acc
                  else
                    let value = parse_expression vb.pvb_expr in
                    match ty with
                    | Some t -> Sarek_ast.MConst (name, t, value) :: acc
                    | None -> acc)
                mods_acc
                vbs
            in
            (types_acc, mods)
        | _ -> (types_acc, mods_acc))
      ([], [])
      items
  in

  let rec collect_mods types_acc mods_acc e =
    match e.pexp_desc with
    | Pexp_letmodule ({txt = Some _name; _}, mod_expr, body) ->
        let inner_types, inner_mods =
          match mod_expr.pmod_desc with
          | Pmod_structure items -> parse_module_items_from_structure items
          | _ -> ([], [])
        in
        collect_mods
          (List.rev_append inner_types types_acc)
          (List.rev_append inner_mods mods_acc)
          body
    | Pexp_open (_, body) ->
        (* Skip past 'let open M in' and continue collecting *)
        collect_mods types_acc mods_acc body
    | Pexp_let (rec_flag, [{pvb_pat; pvb_expr; _}], body) ->
        (* Capture top-level let as module const/fun *)
        let is_recursive = rec_flag = Recursive in
        let name =
          match extract_name_from_pattern pvb_pat with
          | Some n -> n
          | None ->
              raise
                (Parse_error_exn ("Expected variable pattern", pvb_pat.ppat_loc))
        in
        let ty = extract_type_from_pattern pvb_pat in
        let module_items =
          match collect_fun_params pvb_expr with
          | params, Some (Fun_body fn_body) when params <> [] ->
              let parsed_params =
                List.map
                  (fun p -> extract_param_from_pattern (pattern_of_param p))
                  params
              in
              let fn_body = parse_expression fn_body in
              Sarek_ast.MFun (name, is_recursive, parsed_params, fn_body)
              :: mods_acc
          | _, Some (Fun_cases _) ->
              raise
                (Parse_error_exn
                   ( "Pattern-matching functions not supported in module items",
                     pvb_expr.pexp_loc ))
          | _ ->
              let value = parse_expression pvb_expr in
              let ty =
                match ty with
                | Some t -> t
                | None ->
                    raise
                      (Parse_error_exn
                         ( "Module constants must have type annotations",
                           pvb_pat.ppat_loc ))
              in
              Sarek_ast.MConst (name, ty, value) :: mods_acc
        in
        collect_mods types_acc module_items body
    | _ -> (List.rev types_acc, List.rev mods_acc, e)
  in
  let type_decls, module_items, core = collect_mods [] [] payload in
  let kern = parse_kernel_function core in
  {kern with kern_types = type_decls; kern_module_items = module_items}
