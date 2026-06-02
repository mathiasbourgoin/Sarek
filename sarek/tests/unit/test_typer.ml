(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_typer
 *
 * Tests type inference for expressions.
 ******************************************************************************)

(* Force stdlib initialization to register intrinsics *)
let () = Sarek_stdlib.force_init ()

open Sarek_ast
open Sarek_types
open Sarek_env
open Sarek_typer
open Sarek_typed_ast

let dummy_loc =
  {
    loc_file = "test";
    loc_line = 1;
    loc_col = 0;
    loc_end_line = 1;
    loc_end_col = 0;
  }

(* Helper to create expressions *)
let mk_expr e = {e; expr_loc = dummy_loc}

let int_expr i = mk_expr (EInt i)

let int32_expr i = mk_expr (EInt32 i)

let float_expr f = mk_expr (EFloat f)

let bool_expr b = mk_expr (EBool b)

let var_expr s = mk_expr (EVar s)

let unit_expr = mk_expr EUnit

let binop_expr op e1 e2 = mk_expr (EBinop (op, e1, e2))

let unop_expr op e = mk_expr (EUnop (op, e))

let if_expr c t e = mk_expr (EIf (c, t, e))

let let_expr name ty value body = mk_expr (ELet (name, ty, value, body))

(* Helper to check if two types are equivalent *)
let rec types_equal t1 t2 =
  match (repr t1, repr t2) with
  | TPrim p1, TPrim p2 -> p1 = p2
  | TReg r1, TReg r2 -> r1 = r2
  | TVec t1, TVec t2 -> types_equal t1 t2
  | TArr (t1, _), TArr (t2, _) -> types_equal t1 t2
  | TFun (args1, ret1), TFun (args2, ret2) ->
      List.length args1 = List.length args2
      && List.for_all2 types_equal args1 args2
      && types_equal ret1 ret2
  | TTuple ts1, TTuple ts2 ->
      List.length ts1 = List.length ts2 && List.for_all2 types_equal ts1 ts2
  | TRecord (n1, _), TRecord (n2, _) -> n1 = n2
  | TVariant (n1, _), TVariant (n2, _) -> n1 = n2
  | _ -> false

(* Helper to check inference result *)
let check_infer_ok msg env expr expected_typ =
  reset_tvar_counter () ;
  reset_var_id_counter () ;
  match infer env expr with
  | Ok (te, _) ->
      if types_equal te.ty expected_typ then ()
      else
        Alcotest.failf
          "%s: expected %s, got %s"
          msg
          (typ_to_string expected_typ)
          (typ_to_string te.ty)
  | Error errors ->
      Alcotest.failf
        "%s: type error: %s"
        msg
        (String.concat
           ", "
           (List.map Sarek_error.error_to_string errors))

let check_infer_error msg env expr =
  reset_tvar_counter () ;
  reset_var_id_counter () ;
  match infer env expr with
  | Ok (te, _) ->
      Alcotest.failf
        "%s: expected error but got type %s"
        msg
        (typ_to_string te.ty)
  | Error _ -> ()

(* Test literal type inference *)
let test_literal_int () =
  let env = with_stdlib empty in
  check_infer_ok "int literal" env (int_expr 42) t_int32

let test_literal_int32 () =
  let env = with_stdlib empty in
  check_infer_ok "int32 literal" env (int32_expr 42l) t_int32

let test_literal_float () =
  let env = with_stdlib empty in
  check_infer_ok "float literal" env (float_expr 3.14) t_float32

let test_literal_bool () =
  let env = with_stdlib empty in
  check_infer_ok "bool literal true" env (bool_expr true) t_bool ;
  check_infer_ok "bool literal false" env (bool_expr false) t_bool

let test_literal_unit () =
  let env = with_stdlib empty in
  check_infer_ok "unit literal" env unit_expr t_unit

let test_kernel_module_const () =
  reset_tvar_counter () ;
  reset_var_id_counter () ;
  let env = with_stdlib empty in
  let module_item = MConst ("c", TEConstr ("int32", []), int_expr 1) in
  let param =
    {
      param_name = "x";
      param_type = TEConstr ("int32", []);
      param_loc = dummy_loc;
    }
  in
  let body = binop_expr Add (var_expr "x") (var_expr "c") in
  let kernel =
    {
      kern_name = Some "k";
      kern_types = [];
      kern_module_items = [module_item];
      kern_external_item_count = 0;
      kern_params = [param];
      kern_body = body;
      kern_loc = dummy_loc;
    }
  in
  match infer_kernel env kernel with
  | Ok tk -> (
      Alcotest.(check int)
        "module items count"
        1
        (List.length tk.tkern_module_items) ;
      (match tk.tkern_module_items with
      | TMConst (_name, _id, ty, _value) :: _ -> (
          match repr ty with
          | TPrim TInt32 -> ()
          | _ -> Alcotest.fail "module const should be int32")
      | _ -> Alcotest.fail "expected module const") ;
      match repr tk.tkern_return_type with
      | TPrim TInt32 -> ()
      | _ -> Alcotest.fail "kernel body should type to int32")
  | Error errs ->
      Alcotest.failf
        "kernel with module const failed: %s"
        (String.concat
           ", "
           (List.map Sarek_error.error_to_string errs))

let test_kernel_module_fun () =
  reset_tvar_counter () ;
  reset_var_id_counter () ;
  let env = with_stdlib empty in
  let module_fun =
    MFun
      ( "add1",
        false,
        [
          {
            param_name = "x";
            param_type = TEConstr ("int32", []);
            param_loc = dummy_loc;
          };
        ],
        mk_expr (EBinop (Add, var_expr "x", int_expr 1)) )
  in
  let param =
    {
      param_name = "v";
      param_type = TEConstr ("int32", []);
      param_loc = dummy_loc;
    }
  in
  let body =
    mk_expr
      (ELet
         ( "r",
           None,
           mk_expr (EApp (var_expr "add1", [var_expr "v"])),
           var_expr "r" ))
  in
  let kernel =
    {
      kern_name = Some "k";
      kern_types = [];
      kern_module_items = [module_fun];
      kern_external_item_count = 0;
      kern_params = [param];
      kern_body = body;
      kern_loc = dummy_loc;
    }
  in
  match infer_kernel env kernel with
  | Ok tk -> (
      (match tk.tkern_module_items with
      | TMFun (name, _, _, _) :: _ ->
          Alcotest.(check string) "fun name" "add1" name
      | _ -> Alcotest.fail "expected TMFun") ;
      match repr tk.tkern_return_type with
      | TPrim TInt32 -> ()
      | _ -> Alcotest.fail "return type should be int32")
  | Error errs ->
      Alcotest.failf
        "kernel with module fun failed: %s"
        (String.concat
           ", "
           (List.map Sarek_error.error_to_string errs))

let test_kernel_module_fun_with_variant () =
  reset_tvar_counter () ;
  reset_var_id_counter () ;
  let env = with_stdlib empty in
  let type_decl =
    Type_variant
      {
        tdecl_name = "shape";
        tdecl_constructors =
          [
            ("Circle", Some (TEConstr ("float32", [])));
            ("Square", Some (TEConstr ("float32", [])));
          ];
        tdecl_module = None;
        tdecl_loc = dummy_loc;
      }
  in
  let module_fun =
    MFun
      ( "area",
        false,
        [
          {
            param_name = "s";
            param_type = TEConstr ("shape", []);
            param_loc = dummy_loc;
          };
        ],
        mk_expr
          (EMatch
             ( var_expr "s",
               [
                 ( {
                     pat =
                       PConstr
                         ("Circle", Some {pat = PVar "r"; pat_loc = dummy_loc});
                     pat_loc = dummy_loc;
                   },
                   mk_expr
                     (EBinop
                        ( Mul,
                          float_expr 3.14,
                          mk_expr (EBinop (Mul, var_expr "r", var_expr "r")) ))
                 );
                 ( {
                     pat =
                       PConstr
                         ("Square", Some {pat = PVar "x"; pat_loc = dummy_loc});
                     pat_loc = dummy_loc;
                   },
                   mk_expr (EBinop (Mul, var_expr "x", var_expr "x")) );
               ] )) )
  in
  let param =
    {
      param_name = "v";
      param_type = TEConstr ("shape", []);
      param_loc = dummy_loc;
    }
  in
  let body =
    mk_expr
      (ELet
         ( "a",
           None,
           mk_expr (EApp (var_expr "area", [var_expr "v"])),
           var_expr "a" ))
  in
  let kernel =
    {
      kern_name = Some "k";
      kern_types = [type_decl];
      kern_module_items = [module_fun];
      kern_external_item_count = 0;
      kern_params = [param];
      kern_body = body;
      kern_loc = dummy_loc;
    }
  in
  match infer_kernel env kernel with
  | Ok tk -> (
      Alcotest.(check int)
        "variant type decl"
        1
        (List.length tk.tkern_type_decls) ;
      Alcotest.(check int)
        "module fun count"
        1
        (List.length tk.tkern_module_items) ;
      match repr tk.tkern_return_type with
      | TReg Float32 -> ()
      | other ->
          Alcotest.failf "expected float32 return, got %s" (typ_to_string other)
      )
  | Error errs ->
      Alcotest.failf
        "kernel with variant helper failed: %s"
        (String.concat
           ", "
           (List.map Sarek_error.error_to_string errs))

let test_kernel_module_fun_with_record () =
  reset_tvar_counter () ;
  reset_var_id_counter () ;
  let env = with_stdlib empty in
  let type_decl =
    Type_record
      {
        tdecl_name = "point";
        tdecl_fields =
          [
            ("x", false, TEConstr ("float32", []));
            ("y", false, TEConstr ("float32", []));
          ];
        tdecl_module = None;
        tdecl_loc = dummy_loc;
      }
  in
  let module_fun =
    MFun
      ( "make_point",
        false,
        [
          {
            param_name = "a";
            param_type = TEConstr ("float32", []);
            param_loc = dummy_loc;
          };
          {
            param_name = "b";
            param_type = TEConstr ("float32", []);
            param_loc = dummy_loc;
          };
        ],
        mk_expr
          (ERecord (Some "point", [("x", var_expr "a"); ("y", var_expr "b")]))
      )
  in
  let param =
    {
      param_name = "dst";
      param_type = TEConstr ("point", []);
      param_loc = dummy_loc;
    }
  in
  let body =
    mk_expr
      (ELet
         ( "p",
           None,
           mk_expr
             (EApp (var_expr "make_point", [float_expr 1.0; float_expr 2.0])),
           mk_expr (EFieldGet (var_expr "p", "x")) ))
  in
  let kernel =
    {
      kern_name = Some "k";
      kern_types = [type_decl];
      kern_module_items = [module_fun];
      kern_external_item_count = 0;
      kern_params = [param];
      kern_body = body;
      kern_loc = dummy_loc;
    }
  in
  match infer_kernel env kernel with
  | Ok tk -> (
      Alcotest.(check int) "type decl count" 1 (List.length tk.tkern_type_decls) ;
      Alcotest.(check int)
        "module item count"
        1
        (List.length tk.tkern_module_items) ;
      match repr tk.tkern_return_type with
      | TReg Float32 -> ()
      | other ->
          Alcotest.failf "expected float32 return, got %s" (typ_to_string other)
      )
  | Error errs ->
      Alcotest.failf
        "kernel with record helper failed: %s"
        (String.concat
           ", "
           (List.map Sarek_error.error_to_string errs))

let test_kernel_type_decl_record () =
  reset_tvar_counter () ;
  reset_var_id_counter () ;
  let env = with_stdlib empty in
  let type_decl =
    Type_record
      {
        tdecl_name = "point";
        tdecl_fields =
          [
            ("x", false, TEConstr ("float32", []));
            ("y", false, TEConstr ("float32", []));
          ];
        tdecl_module = None;
        tdecl_loc = dummy_loc;
      }
  in
  let record_expr =
    mk_expr
      (ERecord (Some "point", [("x", float_expr 1.0); ("y", float_expr 2.0)]))
  in
  let body =
    mk_expr
      (ELet ("p", None, record_expr, mk_expr (EFieldGet (var_expr "p", "x"))))
  in
  let kernel =
    {
      kern_name = Some "k";
      kern_types = [type_decl];
      kern_module_items = [];
      kern_external_item_count = 0;
      kern_params = [];
      kern_body = body;
      kern_loc = dummy_loc;
    }
  in
  match infer_kernel env kernel with
  | Ok tk -> (
      Alcotest.(check int) "one type decl" 1 (List.length tk.tkern_type_decls) ;
      (match tk.tkern_type_decls with
      | TTypeRecord {tdecl_name; _} :: _ ->
          Alcotest.(check string) "type name" "point" tdecl_name
      | _ -> Alcotest.fail "expected record type decl") ;
      match repr tk.tkern_return_type with
      | TReg Float32 -> ()
      | other ->
          Alcotest.failf "expected float32 return, got %s" (typ_to_string other)
      )
  | Error errs ->
      Alcotest.failf
        "kernel with type decl failed: %s"
        (String.concat
           ", "
           (List.map Sarek_error.error_to_string errs))

(* Test variable type inference *)
let test_var_lookup () =
  let env = with_stdlib empty in
  let info =
    {
      vi_type = t_float32;
      vi_mutable = false;
      vi_is_param = true;
      vi_index = 0;
      vi_is_vec = false;
    }
  in
  let env = add_var "x" info env in
  check_infer_ok "variable lookup" env (var_expr "x") t_float32

let test_var_unbound () =
  let env = with_stdlib empty in
  check_infer_error "unbound variable" env (var_expr "undefined_var")

let test_unbound_function () =
  let env = with_stdlib empty in
  (* Test calling a non-existent function *)
  let expr = mk_expr (EApp (var_expr "nonexistent_func", [int_expr 42])) in
  check_infer_error "unbound function call" env expr

let test_mistyped_intrinsic () =
  let env = with_stdlib empty in
  (* sin expects float32, not int32 *)
  let expr = mk_expr (EApp (var_expr "sin", [int_expr 42])) in
  check_infer_error "sin with wrong type" env expr

let test_intrinsic_const () =
  let env = with_stdlib empty in
  check_infer_ok "thread_idx_x" env (var_expr "thread_idx_x") t_int32 ;
  check_infer_ok "global_thread_id" env (var_expr "global_thread_id") t_int32

(* Test binary operation type inference *)
let test_binop_add_int () =
  let env = with_stdlib empty in
  let expr = binop_expr Add (int_expr 1) (int_expr 2) in
  check_infer_ok "int + int" env expr t_int32

let test_binop_add_float () =
  let env = with_stdlib empty in
  let expr = binop_expr Add (float_expr 1.0) (float_expr 2.0) in
  check_infer_ok "float + float" env expr t_float32

let test_binop_add_mismatch () =
  let env = with_stdlib empty in
  let expr = binop_expr Add (int_expr 1) (float_expr 2.0) in
  check_infer_error "int + float should fail" env expr

let test_binop_comparison () =
  let env = with_stdlib empty in
  let expr = binop_expr Lt (int_expr 1) (int_expr 2) in
  check_infer_ok "int < int gives bool" env expr t_bool

let test_binop_logical () =
  let env = with_stdlib empty in
  let expr = binop_expr And (bool_expr true) (bool_expr false) in
  check_infer_ok "bool && bool" env expr t_bool

let test_binop_logical_mismatch () =
  let env = with_stdlib empty in
  let expr = binop_expr And (int_expr 1) (bool_expr true) in
  check_infer_error "int && bool should fail" env expr

(* Test unary operation type inference *)
let test_unop_neg_int () =
  let env = with_stdlib empty in
  let expr = unop_expr Neg (int_expr 42) in
  check_infer_ok "-int" env expr t_int32

let test_unop_neg_float () =
  let env = with_stdlib empty in
  let expr = unop_expr Neg (float_expr 3.14) in
  check_infer_ok "-float" env expr t_float32

let test_unop_not () =
  let env = with_stdlib empty in
  let expr = unop_expr Not (bool_expr true) in
  check_infer_ok "not bool" env expr t_bool

let test_unop_not_mismatch () =
  let env = with_stdlib empty in
  let expr = unop_expr Not (int_expr 1) in
  check_infer_error "not int should fail" env expr

(* Test if expression type inference *)
let test_if_simple () =
  let env = with_stdlib empty in
  let expr = if_expr (bool_expr true) (int_expr 1) (Some (int_expr 2)) in
  check_infer_ok "if bool then int else int" env expr t_int32

let test_if_branch_mismatch () =
  let env = with_stdlib empty in
  let expr = if_expr (bool_expr true) (int_expr 1) (Some (float_expr 2.0)) in
  check_infer_error "if branches with different types should fail" env expr

let test_if_condition_not_bool () =
  let env = with_stdlib empty in
  let expr = if_expr (int_expr 1) (int_expr 2) (Some (int_expr 3)) in
  check_infer_error "if with non-bool condition should fail" env expr

let test_if_no_else () =
  let env = with_stdlib empty in
  let expr = if_expr (bool_expr true) unit_expr None in
  check_infer_ok "if bool then unit (no else)" env expr t_unit

(* Test let expression type inference *)
let test_let_simple () =
  let env = with_stdlib empty in
  let expr = let_expr "x" None (int_expr 42) (var_expr "x") in
  check_infer_ok "let x = 42 in x" env expr t_int32

let test_let_with_annotation () =
  let env = with_stdlib empty in
  let ty_annot = Sarek_ast.TEConstr ("int32", []) in
  let expr = let_expr "x" (Some ty_annot) (int_expr 42) (var_expr "x") in
  check_infer_ok "let x : int32 = 42 in x" env expr t_int32

let test_let_annotation_mismatch () =
  let env = with_stdlib empty in
  let ty_annot = Sarek_ast.TEConstr ("float32", []) in
  let expr = let_expr "x" (Some ty_annot) (int_expr 42) (var_expr "x") in
  check_infer_error "let x : float32 = 42 should fail" env expr

let test_let_body_uses_binding () =
  let env = with_stdlib empty in
  let expr =
    let_expr "x" None (int_expr 1) (binop_expr Add (var_expr "x") (int_expr 2))
  in
  check_infer_ok "let x = 1 in x + 2" env expr t_int32

(* Test vector access type inference *)
let test_vec_access () =
  let env = with_stdlib empty in
  let info =
    {
      vi_type = TVec t_float32;
      vi_mutable = false;
      vi_is_param = true;
      vi_index = 0;
      vi_is_vec = true;
    }
  in
  let env = add_var "v" info env in
  let expr = mk_expr (EVecGet (var_expr "v", int_expr 0)) in
  check_infer_ok "v.[0]" env expr t_float32

let test_vec_access_wrong_index () =
  let env = with_stdlib empty in
  let info =
    {
      vi_type = TVec t_float32;
      vi_mutable = false;
      vi_is_param = true;
      vi_index = 0;
      vi_is_vec = true;
    }
  in
  let env = add_var "v" info env in
  let expr = mk_expr (EVecGet (var_expr "v", float_expr 0.0)) in
  check_infer_error "v.[0.0] should fail (index must be int)" env expr

(* Test sequence type inference *)
let test_seq () =
  let env = with_stdlib empty in
  let expr = mk_expr (ESeq (int_expr 1, float_expr 2.0)) in
  check_infer_ok "1; 2.0 has type of last expr" env expr t_float32

(* Test suite *)
let () =
  Alcotest.run
    "Sarek_typer"
    [
      ( "literals",
        [
          Alcotest.test_case "int" `Quick test_literal_int;
          Alcotest.test_case "int32" `Quick test_literal_int32;
          Alcotest.test_case "float" `Quick test_literal_float;
          Alcotest.test_case "bool" `Quick test_literal_bool;
          Alcotest.test_case "unit" `Quick test_literal_unit;
        ] );
      ( "variables",
        [
          Alcotest.test_case "lookup" `Quick test_var_lookup;
          Alcotest.test_case "unbound" `Quick test_var_unbound;
          Alcotest.test_case "unbound function" `Quick test_unbound_function;
          Alcotest.test_case "mistyped intrinsic" `Quick test_mistyped_intrinsic;
          Alcotest.test_case "intrinsic const" `Quick test_intrinsic_const;
        ] );
      ( "binop",
        [
          Alcotest.test_case "add int" `Quick test_binop_add_int;
          Alcotest.test_case "add float" `Quick test_binop_add_float;
          Alcotest.test_case "add mismatch" `Quick test_binop_add_mismatch;
          Alcotest.test_case "comparison" `Quick test_binop_comparison;
          Alcotest.test_case "logical" `Quick test_binop_logical;
          Alcotest.test_case
            "logical mismatch"
            `Quick
            test_binop_logical_mismatch;
        ] );
      ( "unop",
        [
          Alcotest.test_case "neg int" `Quick test_unop_neg_int;
          Alcotest.test_case "neg float" `Quick test_unop_neg_float;
          Alcotest.test_case "not" `Quick test_unop_not;
          Alcotest.test_case "not mismatch" `Quick test_unop_not_mismatch;
        ] );
      ( "if",
        [
          Alcotest.test_case "simple" `Quick test_if_simple;
          Alcotest.test_case "branch mismatch" `Quick test_if_branch_mismatch;
          Alcotest.test_case
            "condition not bool"
            `Quick
            test_if_condition_not_bool;
          Alcotest.test_case "no else" `Quick test_if_no_else;
          Alcotest.test_case
            "kernel module const"
            `Quick
            test_kernel_module_const;
          Alcotest.test_case "kernel module fun" `Quick test_kernel_module_fun;
          Alcotest.test_case
            "kernel module fun with record"
            `Quick
            test_kernel_module_fun_with_record;
          Alcotest.test_case
            "kernel module fun with variant"
            `Quick
            test_kernel_module_fun_with_variant;
          Alcotest.test_case
            "kernel module fun with record"
            `Quick
            test_kernel_module_fun_with_record;
          Alcotest.test_case
            "kernel type decl (record)"
            `Quick
            test_kernel_type_decl_record;
        ] );
      ( "let",
        [
          Alcotest.test_case "simple" `Quick test_let_simple;
          Alcotest.test_case "with annotation" `Quick test_let_with_annotation;
          Alcotest.test_case
            "annotation mismatch"
            `Quick
            test_let_annotation_mismatch;
          Alcotest.test_case
            "body uses binding"
            `Quick
            test_let_body_uses_binding;
        ] );
      ( "vector",
        [
          Alcotest.test_case "access" `Quick test_vec_access;
          Alcotest.test_case
            "wrong index type"
            `Quick
            test_vec_access_wrong_index;
        ] );
      ("sequence", [Alcotest.test_case "seq" `Quick test_seq]);
    ]
