[@@@warning "-32-34"]

(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_quote module
 *
 * Tests AST quoting functions that convert Sarek types to ppxlib expressions
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Ppxlib

let dummy_loc : Location.t = Location.none

let dummy_sarek_loc : Sarek_ast.loc =
  {
    loc_file = "test.ml";
    loc_line = 42;
    loc_col = 10;
    loc_end_line = 42;
    loc_end_col = 20;
  }

(* Helper to check expression structure - handles extension nodes and nested expressions *)
let expr_contains_ident e name =
  let rec check expr =
    match expr.pexp_desc with
    | Pexp_ident {txt = Lident id; _} when id = name -> true
    | Pexp_ident {txt = Ldot (_, id); _} when id = name -> true
    | Pexp_apply (f, args) ->
        check f || List.exists (fun (_, arg) -> check arg) args
    | Pexp_construct ({txt = Lident id; _}, _) when id = name -> true
    | Pexp_construct ({txt = Ldot (_, id); _}, _) when id = name -> true
    | Pexp_construct (_, Some arg) -> check arg
    | Pexp_tuple exprs -> List.exists check exprs
    | Pexp_record (fields, _) -> List.exists (fun (_, e) -> check e) fields
    | Pexp_extension (_, payload) -> check_payload payload
    | _ -> false
  and check_payload = function
    | PStr [{pstr_desc = Pstr_eval (e, _); _}] -> check e
    | _ -> false
  in
  check e

(* Helper to check if expression contains a string literal *)
let expr_contains_string e target =
  let rec check expr =
    match expr.pexp_desc with
    | Pexp_constant (Pconst_string (s, _, _)) when s = target -> true
    | Pexp_apply (_, args) -> List.exists (fun (_, arg) -> check arg) args
    | Pexp_construct (_, Some arg) -> check arg
    | Pexp_tuple exprs -> List.exists check exprs
    | Pexp_record (fields, _) -> List.exists (fun (_, e) -> check e) fields
    | Pexp_extension (_, payload) -> check_payload payload
    | _ -> false
  and check_payload = function
    | PStr [{pstr_desc = Pstr_eval (e, _); _}] -> check e
    | _ -> false
  in
  check e

(* Test: evar creates simple identifier *)
let test_evar () =
  let e = Sarek_quote.evar ~loc:dummy_loc "foo" in
  Alcotest.(check bool) "is identifier" true (expr_contains_ident e "foo")

(* Test: evar_qualified creates qualified identifier *)
let test_evar_qualified () =
  let e = Sarek_quote.evar_qualified ~loc:dummy_loc ["Foo"; "Bar"] "baz" in
  Alcotest.(check bool) "contains name" true (expr_contains_ident e "baz")

(* Test: quote_int generates integer constant *)
let test_quote_int () =
  let e = Sarek_quote.quote_int ~loc:dummy_loc 42 in
  match e.pexp_desc with
  | Pexp_constant (Pconst_integer (s, None)) ->
      Alcotest.(check string) "value" "42" s
  | _ -> Alcotest.fail "expected integer constant"

(* Test: quote_float generates float constant *)
let test_quote_float () =
  let e = Sarek_quote.quote_float ~loc:dummy_loc 3.14 in
  match e.pexp_desc with
  | Pexp_constant (Pconst_float _) -> ()
  | _ -> Alcotest.fail "expected float constant"

(* Test: quote_string generates string constant *)
let test_quote_string () =
  let e = Sarek_quote.quote_string ~loc:dummy_loc "hello" in
  match e.pexp_desc with
  | Pexp_constant (Pconst_string (s, _, _)) ->
      Alcotest.(check string) "value" "hello" s
  | _ -> Alcotest.fail "expected string constant"

(* Test: quote_bool generates bool constant *)
let test_quote_bool_true () =
  let e = Sarek_quote.quote_bool ~loc:dummy_loc true in
  (* [%expr true] generates an extension node *)
  match e.pexp_desc with
  | Pexp_construct ({txt = Lident "true"; _}, None) -> ()
  | Pexp_extension _ -> () (* Extension nodes are valid too *)
  | _ -> Alcotest.fail "expected true constant"

let test_quote_bool_false () =
  let e = Sarek_quote.quote_bool ~loc:dummy_loc false in
  match e.pexp_desc with
  | Pexp_construct ({txt = Lident "false"; _}, None) -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected false constant"

(* Test: quote_elttype generates Kirc_Ast constructors *)
let test_quote_elttype_int32 () =
  let e = Sarek_quote.quote_elttype ~loc:dummy_loc Kirc_Ast.EInt32 in
  (* [%expr Sarek.Kirc_Ast.EInt32] expands to Pexp_construct *)
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

let test_quote_elttype_float32 () =
  let e = Sarek_quote.quote_elttype ~loc:dummy_loc Kirc_Ast.EFloat32 in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

(* Test: quote_memspace generates Kirc_Ast constructors *)
let test_quote_memspace_local () =
  let e = Sarek_quote.quote_memspace ~loc:dummy_loc Kirc_Ast.LocalSpace in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

let test_quote_memspace_shared () =
  let e = Sarek_quote.quote_memspace ~loc:dummy_loc Kirc_Ast.Shared in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

(* Test: quote_list generates list construction *)
let test_quote_list_empty () =
  let e = Sarek_quote.quote_list ~loc:dummy_loc Sarek_quote.quote_int [] in
  match e.pexp_desc with
  | Pexp_construct ({txt = Lident "[]"; _}, None) -> ()
  | _ -> Alcotest.fail "expected empty list"

let test_quote_list_nonempty () =
  let e =
    Sarek_quote.quote_list ~loc:dummy_loc Sarek_quote.quote_int [1; 2; 3]
  in
  (* Should generate: 1 :: 2 :: 3 :: [] *)
  match e.pexp_desc with
  | Pexp_construct ({txt = Lident "::"; _}, Some _) -> ()
  | _ -> Alcotest.fail "expected cons cell"

(* Test: quote_option generates option construction *)
let test_quote_option_none () =
  let e = Sarek_quote.quote_option ~loc:dummy_loc Sarek_quote.quote_int None in
  match e.pexp_desc with
  | Pexp_construct ({txt = Lident "None"; _}, None) -> ()
  | _ -> Alcotest.fail "expected None"

let test_quote_option_some () =
  let e =
    Sarek_quote.quote_option ~loc:dummy_loc Sarek_quote.quote_int (Some 42)
  in
  match e.pexp_desc with
  | Pexp_construct ({txt = Lident "Some"; _}, Some _) -> ()
  | _ -> Alcotest.fail "expected Some"

(* Test: core_type_of_typ converts types *)
let test_core_type_of_typ_unit () =
  let ct =
    Sarek_quote.core_type_of_typ ~loc:dummy_loc Sarek_types.(TPrim TUnit)
  in
  match ct with
  | Some _ -> ()
  | None -> Alcotest.fail "expected core type for unit"

let test_core_type_of_typ_int () =
  let ct =
    Sarek_quote.core_type_of_typ ~loc:dummy_loc Sarek_types.(TPrim TInt32)
  in
  match ct with
  | Some _ -> ()
  | None -> Alcotest.fail "expected core type for int"

let test_core_type_of_typ_vec_int () =
  let ct =
    Sarek_quote.core_type_of_typ
      ~loc:dummy_loc
      Sarek_types.(TVec (TPrim TInt32))
  in
  match ct with
  | Some _ -> ()
  | None -> Alcotest.fail "expected core type for int vector"

let test_core_type_of_typ_custom () =
  (* Custom types should return None to let OCaml infer *)
  let ct =
    Sarek_quote.core_type_of_typ
      ~loc:dummy_loc
      Sarek_types.(TRecord ("Point", [("x", TReg Int); ("y", TReg Int)]))
  in
  match ct with
  | None -> ()
  | Some _ -> Alcotest.fail "expected None for custom record type"

(* Test: kernel_ctor_name returns constructor names *)
let test_kernel_ctor_name_int32 () =
  let name = Sarek_quote.kernel_ctor_name Sarek_types.(TPrim TInt32) in
  Alcotest.(check string) "constructor name" "Int32" name

let test_kernel_ctor_name_float64 () =
  let name = Sarek_quote.kernel_ctor_name Sarek_types.(TReg Float64) in
  Alcotest.(check string) "constructor name" "Float64" name

let test_kernel_ctor_name_vec_int32 () =
  let name = Sarek_quote.kernel_ctor_name Sarek_types.(TVec (TPrim TInt32)) in
  Alcotest.(check string) "constructor name" "VInt32" name

let test_kernel_ctor_name_vec_custom () =
  let name =
    Sarek_quote.kernel_ctor_name
      Sarek_types.(TVec (TRecord ("Point", [("x", TReg Int)])))
  in
  Alcotest.(check string) "constructor name" "VCustom" name

let test_kernel_ctor_name_custom () =
  let name =
    Sarek_quote.kernel_ctor_name
      Sarek_types.(TRecord ("Point", [("x", TReg Int)]))
  in
  Alcotest.(check string) "constructor name" "Custom" name

(* Test: quote_sarek_loc quotes location records *)
let test_quote_sarek_loc () =
  let e = Sarek_quote.quote_sarek_loc ~loc:dummy_loc dummy_sarek_loc in
  match e.pexp_desc with
  | Pexp_record _ -> ()
  | _ -> Alcotest.fail "expected record expression"

(* Test: quote_sarek_memspace quotes memspace *)
let test_quote_sarek_memspace_local () =
  let e = Sarek_quote.quote_sarek_memspace ~loc:dummy_loc Sarek_ast.Local in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

let test_quote_sarek_memspace_shared () =
  let e = Sarek_quote.quote_sarek_memspace ~loc:dummy_loc Sarek_ast.Shared in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

(* Test: quote_sarek_binop quotes binary operators *)
let test_quote_sarek_binop_add () =
  let e = Sarek_quote.quote_sarek_binop ~loc:dummy_loc Sarek_ast.Add in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

let test_quote_sarek_binop_eq () =
  let e = Sarek_quote.quote_sarek_binop ~loc:dummy_loc Sarek_ast.Eq in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

(* Test: quote_sarek_unop quotes unary operators *)
let test_quote_sarek_unop_neg () =
  let e = Sarek_quote.quote_sarek_unop ~loc:dummy_loc Sarek_ast.Neg in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

let test_quote_sarek_unop_not () =
  let e = Sarek_quote.quote_sarek_unop ~loc:dummy_loc Sarek_ast.Not in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

(* Test: quote_sarek_for_dir quotes for directions *)
let test_quote_sarek_for_dir_upto () =
  let e = Sarek_quote.quote_sarek_for_dir ~loc:dummy_loc Sarek_ast.Upto in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

let test_quote_sarek_for_dir_downto () =
  let e = Sarek_quote.quote_sarek_for_dir ~loc:dummy_loc Sarek_ast.Downto in
  match e.pexp_desc with
  | Pexp_construct _ -> ()
  | Pexp_ident _ -> ()
  | Pexp_extension _ -> ()
  | _ -> Alcotest.fail "expected constructor, identifier or extension node"

(* Test: quote_sarek_pattern quotes patterns *)
let test_quote_sarek_pattern_any () =
  let p = Sarek_ast.{pat = PAny; pat_loc = dummy_sarek_loc} in
  let e = Sarek_quote.quote_sarek_pattern ~loc:dummy_loc p in
  match e.pexp_desc with
  | Pexp_extension _ -> ()
  | Pexp_record _ -> () (* Record with quoted fields *)
  | _ -> Alcotest.fail "expected extension or record"

let test_quote_sarek_pattern_var () =
  let p = Sarek_ast.{pat = PVar "x"; pat_loc = dummy_sarek_loc} in
  let e = Sarek_quote.quote_sarek_pattern ~loc:dummy_loc p in
  match e.pexp_desc with
  | Pexp_extension _ -> ()
  | Pexp_record _ -> ()
  | _ -> Alcotest.fail "expected extension or record"

(* Test: quote_sarek_param quotes parameters *)
let test_quote_sarek_param () =
  let param =
    Sarek_ast.
      {
        param_name = "x";
        param_type = TEConstr ("int", []);
        param_loc = dummy_sarek_loc;
      }
  in
  let e = Sarek_quote.quote_sarek_param ~loc:dummy_loc param in
  match e.pexp_desc with
  | Pexp_record _ -> ()
  | _ -> Alcotest.fail "expected record expression"

(* Test: quote_sarek_expr quotes simple expressions *)
let test_quote_sarek_expr_unit () =
  let expr = Sarek_ast.{e = EUnit; expr_loc = dummy_sarek_loc} in
  let e = Sarek_quote.quote_sarek_expr ~loc:dummy_loc expr in
  match e.pexp_desc with
  | Pexp_extension _ -> ()
  | Pexp_record _ -> ()
  | _ -> Alcotest.fail "expected extension or record"

let test_quote_sarek_expr_bool () =
  let expr = Sarek_ast.{e = EBool true; expr_loc = dummy_sarek_loc} in
  let e = Sarek_quote.quote_sarek_expr ~loc:dummy_loc expr in
  match e.pexp_desc with
  | Pexp_extension _ -> ()
  | Pexp_record _ -> ()
  | _ -> Alcotest.fail "expected extension or record"

let test_quote_sarek_expr_int () =
  let expr = Sarek_ast.{e = EInt 42; expr_loc = dummy_sarek_loc} in
  let e = Sarek_quote.quote_sarek_expr ~loc:dummy_loc expr in
  match e.pexp_desc with
  | Pexp_extension _ -> ()
  | Pexp_record _ -> ()
  | _ -> Alcotest.fail "expected extension or record"

let test_quote_sarek_expr_var () =
  let expr = Sarek_ast.{e = EVar "x"; expr_loc = dummy_sarek_loc} in
  let e = Sarek_quote.quote_sarek_expr ~loc:dummy_loc expr in
  match e.pexp_desc with
  | Pexp_extension _ -> ()
  | Pexp_record _ -> ()
  | _ -> Alcotest.fail "expected extension or record"

(* Test: expr_of_intrinsic_ref generates intrinsic references *)
let test_expr_of_intrinsic_ref_module () =
  let ref = Sarek_env.IntrinsicRef (["Float32"], "sin") in
  let e = Sarek_quote.expr_of_intrinsic_ref ~loc:dummy_loc ref in
  Alcotest.(check bool) "contains sin" true (expr_contains_ident e "sin")

let test_expr_of_intrinsic_ref_core () =
  let ref = Sarek_env.CorePrimitiveRef "thread_idx_x" in
  let e = Sarek_quote.expr_of_intrinsic_ref ~loc:dummy_loc ref in
  Alcotest.(check bool)
    "contains thread_idx_x"
    true
    (expr_contains_ident e "thread_idx_x")

(* Test suite *)
let () =
  Alcotest.run
    "Sarek_quote"
    [
      ( "basic_quoting",
        [
          Alcotest.test_case "evar" `Quick test_evar;
          Alcotest.test_case "evar_qualified" `Quick test_evar_qualified;
          Alcotest.test_case "quote_int" `Quick test_quote_int;
          Alcotest.test_case "quote_float" `Quick test_quote_float;
          Alcotest.test_case "quote_string" `Quick test_quote_string;
          Alcotest.test_case "quote_bool true" `Quick test_quote_bool_true;
          Alcotest.test_case "quote_bool false" `Quick test_quote_bool_false;
        ] );
      ( "kirc_ast_quoting",
        [
          Alcotest.test_case
            "quote_elttype int32"
            `Quick
            test_quote_elttype_int32;
          Alcotest.test_case
            "quote_elttype float32"
            `Quick
            test_quote_elttype_float32;
          Alcotest.test_case
            "quote_memspace local"
            `Quick
            test_quote_memspace_local;
          Alcotest.test_case
            "quote_memspace shared"
            `Quick
            test_quote_memspace_shared;
        ] );
      ( "collection_quoting",
        [
          Alcotest.test_case "quote_list empty" `Quick test_quote_list_empty;
          Alcotest.test_case
            "quote_list nonempty"
            `Quick
            test_quote_list_nonempty;
          Alcotest.test_case "quote_option none" `Quick test_quote_option_none;
          Alcotest.test_case "quote_option some" `Quick test_quote_option_some;
        ] );
      ( "type_conversion",
        [
          Alcotest.test_case
            "core_type_of_typ unit"
            `Quick
            test_core_type_of_typ_unit;
          Alcotest.test_case
            "core_type_of_typ int"
            `Quick
            test_core_type_of_typ_int;
          Alcotest.test_case
            "core_type_of_typ vec_int"
            `Quick
            test_core_type_of_typ_vec_int;
          Alcotest.test_case
            "core_type_of_typ custom"
            `Quick
            test_core_type_of_typ_custom;
          Alcotest.test_case
            "kernel_ctor_name int32"
            `Quick
            test_kernel_ctor_name_int32;
          Alcotest.test_case
            "kernel_ctor_name float64"
            `Quick
            test_kernel_ctor_name_float64;
          Alcotest.test_case
            "kernel_ctor_name vec_int32"
            `Quick
            test_kernel_ctor_name_vec_int32;
          Alcotest.test_case
            "kernel_ctor_name vec_custom"
            `Quick
            test_kernel_ctor_name_vec_custom;
          Alcotest.test_case
            "kernel_ctor_name custom"
            `Quick
            test_kernel_ctor_name_custom;
        ] );
      ( "sarek_ast_quoting",
        [
          Alcotest.test_case "quote_sarek_loc" `Quick test_quote_sarek_loc;
          Alcotest.test_case
            "quote_sarek_memspace local"
            `Quick
            test_quote_sarek_memspace_local;
          Alcotest.test_case
            "quote_sarek_memspace shared"
            `Quick
            test_quote_sarek_memspace_shared;
          Alcotest.test_case
            "quote_sarek_binop add"
            `Quick
            test_quote_sarek_binop_add;
          Alcotest.test_case
            "quote_sarek_binop eq"
            `Quick
            test_quote_sarek_binop_eq;
          Alcotest.test_case
            "quote_sarek_unop neg"
            `Quick
            test_quote_sarek_unop_neg;
          Alcotest.test_case
            "quote_sarek_unop not"
            `Quick
            test_quote_sarek_unop_not;
          Alcotest.test_case
            "quote_sarek_for_dir upto"
            `Quick
            test_quote_sarek_for_dir_upto;
          Alcotest.test_case
            "quote_sarek_for_dir downto"
            `Quick
            test_quote_sarek_for_dir_downto;
          Alcotest.test_case
            "quote_sarek_pattern any"
            `Quick
            test_quote_sarek_pattern_any;
          Alcotest.test_case
            "quote_sarek_pattern var"
            `Quick
            test_quote_sarek_pattern_var;
          Alcotest.test_case "quote_sarek_param" `Quick test_quote_sarek_param;
        ] );
      ( "expr_quoting",
        [
          Alcotest.test_case
            "quote_sarek_expr unit"
            `Quick
            test_quote_sarek_expr_unit;
          Alcotest.test_case
            "quote_sarek_expr bool"
            `Quick
            test_quote_sarek_expr_bool;
          Alcotest.test_case
            "quote_sarek_expr int"
            `Quick
            test_quote_sarek_expr_int;
          Alcotest.test_case
            "quote_sarek_expr var"
            `Quick
            test_quote_sarek_expr_var;
        ] );
      ( "intrinsic_refs",
        [
          Alcotest.test_case
            "expr_of_intrinsic_ref module"
            `Quick
            test_expr_of_intrinsic_ref_module;
          Alcotest.test_case
            "expr_of_intrinsic_ref core"
            `Quick
            test_expr_of_intrinsic_ref_core;
        ] );
    ]
