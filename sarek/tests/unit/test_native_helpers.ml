(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_native_helpers
 *
 * Tests helper functions for native code generation.
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Ppxlib
open Sarek_types
open Sarek_native_helpers

let dummy_sarek_loc : Sarek_ast.loc =
  {
    loc_file = "test.ml";
    loc_line = 42;
    loc_col = 10;
    loc_end_line = 42;
    loc_end_col = 20;
  }

let dummy_loc = Location.none

(* Helper to check if two expressions are equal (simplified comparison) *)
let expr_to_string expr = Format.asprintf "%a" Pprintast.expression expr

(* ===== Test: ppxlib_loc_of_sarek ===== *)

let test_ppxlib_loc_conversion () =
  let loc = ppxlib_loc_of_sarek dummy_sarek_loc in
  Alcotest.(check string) "filename" "test.ml" loc.loc_start.pos_fname ;
  Alcotest.(check int) "start line" 42 loc.loc_start.pos_lnum ;
  Alcotest.(check int) "start col" 10 loc.loc_start.pos_cnum ;
  Alcotest.(check int) "end line" 42 loc.loc_end.pos_lnum ;
  Alcotest.(check int) "end col" 20 loc.loc_end.pos_cnum ;
  Alcotest.(check bool) "not ghost" false loc.loc_ghost

(* ===== Test: evar ===== *)

let test_evar_simple () =
  let expr = evar ~loc:dummy_loc "foo" in
  let str = expr_to_string expr in
  Alcotest.(check bool) "contains identifier" true (String.length str > 0) ;
  (* Just check it doesn't crash and produces some output *)
  ()

let test_evar_with_underscore () =
  let expr = evar ~loc:dummy_loc "__state" in
  let str = expr_to_string expr in
  Alcotest.(check bool) "creates expression" true (String.length str > 0)

(* ===== Test: evar_qualified ===== *)

let test_evar_qualified_single () =
  let expr = evar_qualified ~loc:dummy_loc ["Foo"] "bar" in
  let str = expr_to_string expr in
  (* Should produce something like Foo.bar *)
  Alcotest.(check bool) "creates qualified expr" true (String.length str > 0)

let test_evar_qualified_nested () =
  let expr = evar_qualified ~loc:dummy_loc ["Sarek"; "Gpu"] "block_barrier" in
  let str = expr_to_string expr in
  (* Should produce something like Sarek.Gpu.block_barrier *)
  Alcotest.(check bool)
    "creates nested qualified expr"
    true
    (String.length str > 0)

(* ===== Test: var_name ===== *)

let test_var_name_generates_unique () =
  let name1 = var_name 0 in
  let name2 = var_name 1 in
  let name3 = var_name 42 in
  Alcotest.(check string) "var 0" "__v0" name1 ;
  Alcotest.(check string) "var 1" "__v1" name2 ;
  Alcotest.(check string) "var 42" "__v42" name3

let test_var_name_different_ids () =
  let name1 = var_name 10 in
  let name2 = var_name 20 in
  Alcotest.(check bool) "different names" true (name1 <> name2)

(* ===== Test: mut_var_name ===== *)

let test_mut_var_name_generates_unique () =
  let name1 = mut_var_name 0 in
  let name2 = mut_var_name 1 in
  Alcotest.(check string) "mutable var 0" "__m0" name1 ;
  Alcotest.(check string) "mutable var 1" "__m1" name2

let test_mut_var_name_differs_from_var_name () =
  let var = var_name 5 in
  let mut_var = mut_var_name 5 in
  Alcotest.(check bool) "var and mut_var differ" true (var <> mut_var)

(* ===== Test: state_var and shared_var ===== *)

let test_state_var_constant () =
  Alcotest.(check string) "state var name" "__state" state_var

let test_shared_var_constant () =
  Alcotest.(check string) "shared var name" "__shared" shared_var

(* ===== Test: default_value_for_type ===== *)

let test_default_value_unit () =
  let expr = default_value_for_type ~loc:dummy_loc t_unit in
  let str = expr_to_string expr in
  Alcotest.(check bool) "unit default exists" true (String.length str > 0)

let test_default_value_bool () =
  let expr = default_value_for_type ~loc:dummy_loc t_bool in
  let str = expr_to_string expr in
  (* Should be false *)
  Alcotest.(check bool) "bool default exists" true (String.length str > 0)

let test_default_value_int32 () =
  let expr = default_value_for_type ~loc:dummy_loc t_int32 in
  let str = expr_to_string expr in
  (* Should be 0l *)
  Alcotest.(check bool) "int32 default exists" true (String.length str > 0)

let test_default_value_float32 () =
  let expr = default_value_for_type ~loc:dummy_loc t_float32 in
  let str = expr_to_string expr in
  (* Should be 0.0 *)
  Alcotest.(check bool) "float32 default exists" true (String.length str > 0)

let test_default_value_float64 () =
  let expr = default_value_for_type ~loc:dummy_loc t_float64 in
  let str = expr_to_string expr in
  (* Should be 0.0 *)
  Alcotest.(check bool) "float64 default exists" true (String.length str > 0)

let test_default_value_int64 () =
  let ty = TReg Int64 in
  let expr = default_value_for_type ~loc:dummy_loc ty in
  let str = expr_to_string expr in
  (* Should be 0L *)
  Alcotest.(check bool) "int64 default exists" true (String.length str > 0)

let test_default_value_char () =
  let ty = TReg Char in
  let expr = default_value_for_type ~loc:dummy_loc ty in
  let str = expr_to_string expr in
  (* Should be '\000' *)
  Alcotest.(check bool) "char default exists" true (String.length str > 0)

let test_default_value_record () =
  (* Create a simple record type: { x: int32; y: float32 } *)
  let fields = [("x", t_int32); ("y", t_float32)] in
  let ty = TRecord ("point", fields) in
  let expr = default_value_for_type ~loc:dummy_loc ty in
  let str = expr_to_string expr in
  (* Should generate { x = 0l; y = 0.0 } or similar *)
  Alcotest.(check bool) "record default exists" true (String.length str > 0)

let test_default_value_variant_nullary () =
  (* Create a variant with nullary constructor: type color = Red | Green | Blue *)
  let constrs = [("Red", None); ("Green", None); ("Blue", None)] in
  let ty = TVariant ("color", constrs) in
  let expr = default_value_for_type ~loc:dummy_loc ty in
  let str = expr_to_string expr in
  (* Should generate Red (first constructor) *)
  Alcotest.(check bool) "variant default exists" true (String.length str > 0)

let test_default_value_variant_with_arg () =
  (* Create a variant with argument: type option = None | Some of int32 *)
  let constrs = [("None", None); ("Some", Some t_int32)] in
  let ty = TVariant ("option", constrs) in
  let expr = default_value_for_type ~loc:dummy_loc ty in
  let str = expr_to_string expr in
  (* Should generate None (nullary constructor preferred) *)
  Alcotest.(check bool) "variant default exists" true (String.length str > 0)

let test_default_value_variant_no_nullary () =
  (* Create a variant with only constructors that have arguments *)
  let constrs =
    [("Some", Some t_int32); ("Pair", Some (TTuple [t_int32; t_bool]))]
  in
  let ty = TVariant ("non_nullary", constrs) in
  let expr = default_value_for_type ~loc:dummy_loc ty in
  let str = expr_to_string expr in
  (* Should generate Some 0l (first constructor with default arg) *)
  Alcotest.(check bool)
    "variant non-nullary default exists"
    true
    (String.length str > 0)

(* ===== Test Suite ===== *)

let location_tests =
  [Alcotest.test_case "conversion" `Quick test_ppxlib_loc_conversion]

let evar_tests =
  [
    Alcotest.test_case "simple" `Quick test_evar_simple;
    Alcotest.test_case "with underscore" `Quick test_evar_with_underscore;
  ]

let evar_qualified_tests =
  [
    Alcotest.test_case "single module" `Quick test_evar_qualified_single;
    Alcotest.test_case "nested modules" `Quick test_evar_qualified_nested;
  ]

let var_name_tests =
  [
    Alcotest.test_case "generates unique" `Quick test_var_name_generates_unique;
    Alcotest.test_case "different ids" `Quick test_var_name_different_ids;
  ]

let mut_var_name_tests =
  [
    Alcotest.test_case
      "generates unique"
      `Quick
      test_mut_var_name_generates_unique;
    Alcotest.test_case
      "differs from var_name"
      `Quick
      test_mut_var_name_differs_from_var_name;
  ]

let constant_tests =
  [
    Alcotest.test_case "state_var" `Quick test_state_var_constant;
    Alcotest.test_case "shared_var" `Quick test_shared_var_constant;
  ]

let default_value_tests =
  [
    Alcotest.test_case "unit" `Quick test_default_value_unit;
    Alcotest.test_case "bool" `Quick test_default_value_bool;
    Alcotest.test_case "int32" `Quick test_default_value_int32;
    Alcotest.test_case "float32" `Quick test_default_value_float32;
    Alcotest.test_case "float64" `Quick test_default_value_float64;
    Alcotest.test_case "int64" `Quick test_default_value_int64;
    Alcotest.test_case "char" `Quick test_default_value_char;
    Alcotest.test_case "record" `Quick test_default_value_record;
    Alcotest.test_case
      "variant nullary"
      `Quick
      test_default_value_variant_nullary;
    Alcotest.test_case
      "variant with arg"
      `Quick
      test_default_value_variant_with_arg;
    Alcotest.test_case
      "variant no nullary"
      `Quick
      test_default_value_variant_no_nullary;
  ]

let () =
  Alcotest.run
    "Sarek_native_helpers"
    [
      ("ppxlib_loc_of_sarek", location_tests);
      ("evar", evar_tests);
      ("evar_qualified", evar_qualified_tests);
      ("var_name", var_name_tests);
      ("mut_var_name", mut_var_name_tests);
      ("constants", constant_tests);
      ("default_value_for_type", default_value_tests);
    ]
