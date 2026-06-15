(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_parse
 *
 * Tests parsing OCaml expressions to Sarek AST.
 * Note: These tests use ppxlib's Ast_builder to create test expressions.
 ******************************************************************************)

open Sarek_ast
open Sarek_parse

(* We'll test the parsing functions indirectly by checking the output AST *)

(* Helper to check binop parsing *)
let check_binop msg op expected =
  match parse_binop op with
  | Some b when b = expected -> ()
  | Some _ -> Alcotest.failf "%s: expected %s, got different binop" msg op
  | None -> Alcotest.failf "%s: parse_binop returned None" msg

let check_binop_none msg op =
  match parse_binop op with
  | None -> ()
  | Some _ -> Alcotest.failf "%s: expected None but got Some" msg

(* Test parse_binop *)
let test_parse_binop_add () =
  check_binop "+" "+" Add ;
  check_binop "+." "+." Add

let test_parse_binop_sub () =
  check_binop "-" "-" Sub ;
  check_binop "-." "-." Sub

let test_parse_binop_mul () =
  check_binop "*" "*" Mul ;
  check_binop "*." "*." Mul

let test_parse_binop_div () =
  check_binop "/" "/" Div ;
  check_binop "/." "/." Div

let test_parse_binop_mod () = check_binop "mod" "mod" Mod

let test_parse_binop_comparison () =
  check_binop "=" "=" Eq ;
  check_binop "<>" "<>" Ne ;
  check_binop "<" "<" Lt ;
  check_binop "<=" "<=" Le ;
  check_binop ">" ">" Gt ;
  check_binop ">=" ">=" Ge

let test_parse_binop_logical () =
  check_binop "&&" "&&" And ;
  check_binop "||" "||" Or

let test_parse_binop_bitwise () =
  check_binop "land" "land" Land ;
  check_binop "lor" "lor" Lor ;
  check_binop "lxor" "lxor" Lxor ;
  check_binop "lsl" "lsl" Lsl ;
  check_binop "lsr" "lsr" Lsr ;
  check_binop "asr" "asr" Asr

let test_parse_binop_unknown () = check_binop_none "unknown" "foo"

(* Helper to check unop parsing *)
let check_unop msg op expected =
  match parse_unop op with
  | Some u when u = expected -> ()
  | Some _ -> Alcotest.failf "%s: expected %s, got different unop" msg op
  | None -> Alcotest.failf "%s: parse_unop returned None" msg

let check_unop_none msg op =
  match parse_unop op with
  | None -> ()
  | Some _ -> Alcotest.failf "%s: expected None but got Some" msg

(* Test parse_unop *)
let test_parse_unop_neg () =
  check_unop "-" "-" Neg ;
  check_unop "-." "-." Neg ;
  check_unop "~-" "~-" Neg ;
  check_unop "~-." "~-." Neg

let test_parse_unop_not () = check_unop "not" "not" Not

let test_parse_unop_lnot () = check_unop "lnot" "lnot" Lnot

let test_parse_unop_unknown () = check_unop_none "unknown" "foo"

(* Test parse_type - we need to create Ppxlib types to test this *)
let test_parse_type_primitives () =
  (* Create a simple type using Ppxlib *)
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let int32_ty = ptyp_constr ~loc {txt = Lident "int32"; loc} [] in
  let parsed = parse_type int32_ty in
  (match parsed with
  | TEConstr ("int32", []) -> Alcotest.(check pass) "int32 parses" () ()
  | _ -> Alcotest.fail "int32 should parse to TEConstr") ;

  let float32_ty = ptyp_constr ~loc {txt = Lident "float32"; loc} [] in
  let parsed = parse_type float32_ty in
  match parsed with
  | TEConstr ("float32", []) -> Alcotest.(check pass) "float32 parses" () ()
  | _ -> Alcotest.fail "float32 should parse to TEConstr"

let test_parse_type_vector () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let elem_ty = ptyp_constr ~loc {txt = Lident "float32"; loc} [] in
  let vec_ty = ptyp_constr ~loc {txt = Lident "vector"; loc} [elem_ty] in
  let parsed = parse_type vec_ty in
  match parsed with
  | TEConstr ("vector", [TEConstr ("float32", [])]) ->
      Alcotest.(check pass) "float32 vector parses" () ()
  | _ -> Alcotest.fail "float32 vector should parse correctly"

let test_parse_type_arrow () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let int_ty = ptyp_constr ~loc {txt = Lident "int32"; loc} [] in
  let bool_ty = ptyp_constr ~loc {txt = Lident "bool"; loc} [] in
  let arrow_ty = ptyp_arrow ~loc Nolabel int_ty bool_ty in
  let parsed = parse_type arrow_ty in
  match parsed with
  | TEArrow (TEConstr ("int32", []), TEConstr ("bool", [])) ->
      Alcotest.(check pass) "int32 -> bool parses" () ()
  | _ -> Alcotest.fail "int32 -> bool should parse correctly"

let test_parse_type_tuple () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let int_ty = ptyp_constr ~loc {txt = Lident "int32"; loc} [] in
  let float_ty = ptyp_constr ~loc {txt = Lident "float32"; loc} [] in
  let tuple_ty = ptyp_tuple ~loc [int_ty; float_ty] in
  let parsed = parse_type tuple_ty in
  match parsed with
  | TETuple [TEConstr ("int32", []); TEConstr ("float32", [])] ->
      Alcotest.(check pass) "int32 * float32 parses" () ()
  | _ -> Alcotest.fail "int32 * float32 should parse correctly"

let test_parse_type_var () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let var_ty = ptyp_var ~loc "a" in
  let parsed = parse_type var_ty in
  match parsed with
  | TEVar "a" -> Alcotest.(check pass) "'a parses" () ()
  | _ -> Alcotest.fail "'a should parse to TEVar"

(* Test parsing payload that starts with a let-module before the kernel function *)
let test_parse_payload_letmodule () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  (* module M = struct let c = 1 end in fun (v : int32 vector) -> () *)
  let module_struct =
    pmod_structure
      ~loc
      [
        pstr_value
          ~loc
          Nonrecursive
          [
            value_binding
              ~loc
              ~pat:
                (ppat_constraint
                   ~loc
                   (ppat_var ~loc {txt = "c"; loc})
                   (ptyp_constr ~loc {txt = Lident "int32"; loc} []))
              ~expr:(eint ~loc 1);
          ];
      ]
  in
  let vec_ty =
    ptyp_constr
      ~loc
      {txt = Lident "vector"; loc}
      [ptyp_constr ~loc {txt = Lident "int32"; loc} []]
  in
  let param_pat =
    ppat_constraint ~loc (ppat_var ~loc {txt = "v"; loc}) vec_ty
  in
  let fun_expr =
    pexp_fun
      ~loc
      Nolabel
      None
      param_pat
      (pexp_construct ~loc {txt = Lident "()"; loc} None)
  in
  let payload =
    pexp_letmodule ~loc {txt = Some "M"; loc} module_struct fun_expr
  in
  let kernel = parse_payload payload in
  match kernel.kern_params with
  | [p] ->
      if p.param_name <> "v" then
        Alcotest.failf "expected param name v, got %s" p.param_name
  | _ -> Alcotest.fail "expected exactly one parameter in kernel"

let test_parse_payload_letmodule_with_type () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let int32_ty = ptyp_constr ~loc {txt = Lident "int32"; loc} [] in
  let float32_ty = ptyp_constr ~loc {txt = Lident "float32"; loc} [] in
  let fields =
    [
      label_declaration
        ~loc
        ~mutable_:Mutable
        ~name:{txt = "x"; loc}
        ~type_:int32_ty;
      label_declaration
        ~loc
        ~mutable_:Immutable
        ~name:{txt = "y"; loc}
        ~type_:float32_ty;
    ]
  in
  let type_decl =
    type_declaration
      ~loc
      ~name:{txt = "point"; loc}
      ~params:[]
      ~cstrs:[]
      ~kind:(Ptype_record fields)
      ~private_:Public
      ~manifest:None
  in
  let module_struct =
    pmod_structure ~loc [pstr_type ~loc Nonrecursive [type_decl]]
  in
  let vec_ty =
    ptyp_constr
      ~loc
      {txt = Lident "vector"; loc}
      [ptyp_constr ~loc {txt = Lident "point"; loc} []]
  in
  let param_pat =
    ppat_constraint ~loc (ppat_var ~loc {txt = "v"; loc}) vec_ty
  in
  let fun_expr =
    pexp_fun
      ~loc
      Nolabel
      None
      param_pat
      (pexp_construct ~loc {txt = Lident "()"; loc} None)
  in
  let payload =
    pexp_letmodule ~loc {txt = Some "Types"; loc} module_struct fun_expr
  in
  let kernel = parse_payload payload in
  Alcotest.(check int) "one type decl" 1 (List.length kernel.kern_types) ;
  (match List.hd kernel.kern_types with
  | Type_record {tdecl_name; _} ->
      Alcotest.(check string) "type name" "point" tdecl_name
  | _ -> Alcotest.fail "expected record type decl") ;
  match kernel.kern_params with
  | [p] -> Alcotest.(check string) "param uses type" "v" p.param_name
  | _ -> Alcotest.fail "expected single parameter"

let test_parse_payload_module_fun () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let vec_ty =
    ptyp_constr
      ~loc
      {txt = Lident "vector"; loc}
      [ptyp_constr ~loc {txt = Lident "int32"; loc} []]
  in
  let fun_binding =
    value_binding
      ~loc
      ~pat:(ppat_var ~loc {txt = "add1"; loc})
      ~expr:
        (pexp_fun
           ~loc
           Nolabel
           None
           (ppat_constraint
              ~loc
              (ppat_var ~loc {txt = "x"; loc})
              (ptyp_constr ~loc {txt = Lident "int32"; loc} []))
           (pexp_apply
              ~loc
              (evar ~loc "Int32.add")
              [(Nolabel, evar ~loc "x"); (Nolabel, eint ~loc 1)]))
  in
  let module_struct =
    pmod_structure ~loc [pstr_value ~loc Nonrecursive [fun_binding]]
  in
  let param_pat =
    ppat_constraint ~loc (ppat_var ~loc {txt = "v"; loc}) vec_ty
  in
  let fun_expr =
    pexp_fun
      ~loc
      Nolabel
      None
      param_pat
      [%expr
        let open Std in
        v.(0) <- add1 0l]
  in
  let payload =
    pexp_letmodule ~loc {txt = Some "M"; loc} module_struct fun_expr
  in
  let kernel = parse_payload payload in
  Alcotest.(check int) "one module fun" 1 (List.length kernel.kern_module_items) ;
  match List.hd kernel.kern_module_items with
  | MFun (name, _, _, _) -> Alcotest.(check string) "fun name" "add1" name
  | _ -> Alcotest.fail "expected module fun"

(* Integration test - parse a complete kernel function *)
let test_parse_kernel_simple () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  (* Build: fun (x : int32) -> x *)
  let param_pat =
    ppat_constraint
      ~loc
      (ppat_var ~loc {txt = "x"; loc})
      (ptyp_constr ~loc {txt = Lident "int32"; loc} [])
  in
  let body = pexp_ident ~loc {txt = Lident "x"; loc} in
  let kernel_expr = pexp_fun ~loc Nolabel None param_pat body in

  let kernel = parse_kernel_function kernel_expr in

  Alcotest.(check int) "kernel has 1 param" 1 (List.length kernel.kern_params) ;
  Alcotest.(check string)
    "param name is x"
    "x"
    (List.hd kernel.kern_params).param_name ;
  match kernel.kern_body.e with
  | EVar "x" -> Alcotest.(check pass) "body is EVar x" () ()
  | _ -> Alcotest.fail "body should be EVar x"

let test_parse_kernel_two_params () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  (* Build: fun (a : float32 vector) (b : float32 vector) -> () *)
  let vec_ty elem =
    ptyp_constr
      ~loc
      {txt = Lident "vector"; loc}
      [ptyp_constr ~loc {txt = Lident elem; loc} []]
  in
  let param_a =
    ppat_constraint ~loc (ppat_var ~loc {txt = "a"; loc}) (vec_ty "float32")
  in
  let param_b =
    ppat_constraint ~loc (ppat_var ~loc {txt = "b"; loc}) (vec_ty "float32")
  in
  let body = pexp_construct ~loc {txt = Lident "()"; loc} None in
  let kernel_expr =
    pexp_fun ~loc Nolabel None param_a (pexp_fun ~loc Nolabel None param_b body)
  in

  let kernel = parse_kernel_function kernel_expr in

  Alcotest.(check int) "kernel has 2 params" 2 (List.length kernel.kern_params) ;
  Alcotest.(check string)
    "first param is a"
    "a"
    (List.nth kernel.kern_params 0).param_name ;
  Alcotest.(check string)
    "second param is b"
    "b"
    (List.nth kernel.kern_params 1).param_name ;
  match kernel.kern_body.e with
  | EUnit -> Alcotest.(check pass) "body is EUnit" () ()
  | _ -> Alcotest.fail "body should be EUnit"

let test_parse_kernel_shared_body () =
  let kernel_expr =
    Ppxlib.Parse.expression
      (Lexing.from_string
         "fun (input : float32 vector) -> let%shared (tile : float32) = () in \
          ()")
  in
  let kernel = parse_kernel_function kernel_expr in
  match kernel.kern_body.e with
  | ELetShared ("tile", TEConstr ("float32", []), None, {e = EUnit; _}) ->
      Alcotest.(check pass) "shared body preserved" () ()
  | _ -> Alcotest.fail "body should be ELetShared"

let test_parse_kernel_no_annotation () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  (* Build: fun x -> x (no type annotation - should fail) *)
  let param_pat = ppat_var ~loc {txt = "x"; loc} in
  let body = pexp_ident ~loc {txt = Lident "x"; loc} in
  let kernel_expr = pexp_fun ~loc Nolabel None param_pat body in

  try
    let _ = parse_kernel_function kernel_expr in
    Alcotest.fail "should fail without type annotation"
  with Parse_error_exn _ ->
    Alcotest.(check pass) "fails without annotation" () ()

(* Test parsing expressions *)
let test_parse_expr_int () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let expr = pexp_constant ~loc (Pconst_integer ("42", None)) in
  let parsed = parse_expression expr in
  match parsed.e with
  | EInt 42 -> Alcotest.(check pass) "42 parses" () ()
  | _ -> Alcotest.fail "42 should parse to EInt 42"

let test_parse_expr_int32 () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let expr = pexp_constant ~loc (Pconst_integer ("42", Some 'l')) in
  let parsed = parse_expression expr in
  match parsed.e with
  | EInt32 42l -> Alcotest.(check pass) "42l parses" () ()
  | _ -> Alcotest.fail "42l should parse to EInt32 42l"

let test_parse_expr_float () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let expr = pexp_constant ~loc (Pconst_float ("3.14", None)) in
  let parsed = parse_expression expr in
  match parsed.e with
  | EFloat f when abs_float (f -. 3.14) < 0.001 ->
      Alcotest.(check pass) "3.14 parses" () ()
  | _ -> Alcotest.fail "3.14 should parse to EFloat"

let test_parse_expr_bool () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let true_expr = pexp_construct ~loc {txt = Lident "true"; loc} None in
  let parsed = parse_expression true_expr in
  (match parsed.e with
  | EBool true -> Alcotest.(check pass) "true parses" () ()
  | _ -> Alcotest.fail "true should parse to EBool true") ;

  let false_expr = pexp_construct ~loc {txt = Lident "false"; loc} None in
  let parsed = parse_expression false_expr in
  match parsed.e with
  | EBool false -> Alcotest.(check pass) "false parses" () ()
  | _ -> Alcotest.fail "false should parse to EBool false"

let test_parse_expr_unit () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let expr = pexp_construct ~loc {txt = Lident "()"; loc} None in
  let parsed = parse_expression expr in
  match parsed.e with
  | EUnit -> Alcotest.(check pass) "() parses" () ()
  | _ -> Alcotest.fail "() should parse to EUnit"

let test_parse_expr_var () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let expr = pexp_ident ~loc {txt = Lident "foo"; loc} in
  let parsed = parse_expression expr in
  match parsed.e with
  | EVar "foo" -> Alcotest.(check pass) "foo parses" () ()
  | _ -> Alcotest.fail "foo should parse to EVar"

let test_parse_expr_binop () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let left = pexp_constant ~loc (Pconst_integer ("1", None)) in
  let right = pexp_constant ~loc (Pconst_integer ("2", None)) in
  let op = pexp_ident ~loc {txt = Lident "+"; loc} in
  let expr = pexp_apply ~loc op [(Nolabel, left); (Nolabel, right)] in
  let parsed = parse_expression expr in
  match parsed.e with
  | EBinop (Add, {e = EInt 1; _}, {e = EInt 2; _}) ->
      Alcotest.(check pass) "1 + 2 parses" () ()
  | _ -> Alcotest.fail "1 + 2 should parse to EBinop"

let test_parse_expr_if () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let cond = pexp_construct ~loc {txt = Lident "true"; loc} None in
  let then_e = pexp_constant ~loc (Pconst_integer ("1", None)) in
  let else_e = pexp_constant ~loc (Pconst_integer ("2", None)) in
  let expr = pexp_ifthenelse ~loc cond then_e (Some else_e) in
  let parsed = parse_expression expr in
  match parsed.e with
  | EIf ({e = EBool true; _}, {e = EInt 1; _}, Some {e = EInt 2; _}) ->
      Alcotest.(check pass) "if true then 1 else 2 parses" () ()
  | _ -> Alcotest.fail "if-then-else should parse correctly"

let test_parse_expr_let () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let pat = ppat_var ~loc {txt = "x"; loc} in
  let value = pexp_constant ~loc (Pconst_integer ("42", None)) in
  let body = pexp_ident ~loc {txt = Lident "x"; loc} in
  let binding = value_binding ~loc ~pat ~expr:value in
  let expr = pexp_let ~loc Nonrecursive [binding] body in
  let parsed = parse_expression expr in
  match parsed.e with
  | ELet ("x", None, {e = EInt 42; _}, {e = EVar "x"; _}) ->
      Alcotest.(check pass) "let x = 42 in x parses" () ()
  | _ -> Alcotest.fail "let should parse correctly"

let test_parse_expr_lambda_rejected () =
  let loc = Location.none in
  let open Ppxlib.Ast_builder.Default in
  let param =
    ppat_constraint
      ~loc
      (ppat_var ~loc {txt = "x"; loc})
      (ptyp_constr ~loc {txt = Lident "int32"; loc} [])
  in
  let expr = pexp_fun ~loc Nolabel None param (evar ~loc "x") in
  try
    ignore (parse_expression expr) ;
    Alcotest.fail "standalone lambda should be rejected"
  with Parse_error_exn (msg, _) ->
    Alcotest.(check bool) "has useful error message" true (String.length msg > 0)

(* Test suite *)
let () =
  Alcotest.run
    "Sarek_parse"
    [
      ( "binop",
        [
          Alcotest.test_case "add" `Quick test_parse_binop_add;
          Alcotest.test_case "sub" `Quick test_parse_binop_sub;
          Alcotest.test_case "mul" `Quick test_parse_binop_mul;
          Alcotest.test_case "div" `Quick test_parse_binop_div;
          Alcotest.test_case "mod" `Quick test_parse_binop_mod;
          Alcotest.test_case "comparison" `Quick test_parse_binop_comparison;
          Alcotest.test_case "logical" `Quick test_parse_binop_logical;
          Alcotest.test_case "bitwise" `Quick test_parse_binop_bitwise;
          Alcotest.test_case "unknown" `Quick test_parse_binop_unknown;
        ] );
      ( "unop",
        [
          Alcotest.test_case "neg" `Quick test_parse_unop_neg;
          Alcotest.test_case "not" `Quick test_parse_unop_not;
          Alcotest.test_case "lnot" `Quick test_parse_unop_lnot;
          Alcotest.test_case "unknown" `Quick test_parse_unop_unknown;
        ] );
      ( "type",
        [
          Alcotest.test_case "primitives" `Quick test_parse_type_primitives;
          Alcotest.test_case "vector" `Quick test_parse_type_vector;
          Alcotest.test_case "arrow" `Quick test_parse_type_arrow;
          Alcotest.test_case "tuple" `Quick test_parse_type_tuple;
          Alcotest.test_case "var" `Quick test_parse_type_var;
        ] );
      ( "kernel",
        [
          Alcotest.test_case "simple" `Quick test_parse_kernel_simple;
          Alcotest.test_case "two params" `Quick test_parse_kernel_two_params;
          Alcotest.test_case "shared body" `Quick test_parse_kernel_shared_body;
          Alcotest.test_case
            "no annotation"
            `Quick
            test_parse_kernel_no_annotation;
          Alcotest.test_case
            "payload letmodule"
            `Quick
            test_parse_payload_letmodule;
          Alcotest.test_case
            "payload type decl"
            `Quick
            test_parse_payload_letmodule_with_type;
          Alcotest.test_case
            "payload module fun"
            `Quick
            test_parse_payload_module_fun;
        ] );
      ( "expr",
        [
          Alcotest.test_case "int" `Quick test_parse_expr_int;
          Alcotest.test_case "int32" `Quick test_parse_expr_int32;
          Alcotest.test_case "float" `Quick test_parse_expr_float;
          Alcotest.test_case "bool" `Quick test_parse_expr_bool;
          Alcotest.test_case "unit" `Quick test_parse_expr_unit;
          Alcotest.test_case "var" `Quick test_parse_expr_var;
          Alcotest.test_case "binop" `Quick test_parse_expr_binop;
          Alcotest.test_case "if" `Quick test_parse_expr_if;
          Alcotest.test_case "let" `Quick test_parse_expr_let;
          Alcotest.test_case
            "lambda rejected"
            `Quick
            test_parse_expr_lambda_rejected;
        ] );
    ]
