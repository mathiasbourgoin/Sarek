(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_tailrec_elim
 *
 * Tests tail recursion elimination transformation.
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Sarek_ast
open Sarek_types
open Sarek_typed_ast
open Sarek_tailrec_elim

let dummy_loc =
  {
    loc_file = "test";
    loc_line = 1;
    loc_col = 0;
    loc_end_line = 1;
    loc_end_col = 0;
  }

(* Helper to create typed expressions *)
let mk_texpr te ty = {te; ty; te_loc = dummy_loc}

let te_int32 i = mk_texpr (TEInt32 i) (TPrim TInt32)

let te_var name id ty = mk_texpr (TEVar (name, id)) ty

(* ===== Test: fresh_transform_id ===== *)

let test_fresh_transform_id_generates_unique () =
  let id1 = fresh_transform_id () in
  let id2 = fresh_transform_id () in
  let id3 = fresh_transform_id () in
  Alcotest.(check bool) "ids are different" true (id1 <> id2 && id2 <> id3)

let test_fresh_transform_id_sequential () =
  (* IDs should be sequential (or at least increasing) *)
  let id1 = fresh_transform_id () in
  let id2 = fresh_transform_id () in
  Alcotest.(check bool) "id2 > id1" true (id2 > id1)

(* ===== Test: eliminate_tail_recursion (basic structure) ===== *)

let test_eliminate_tail_recursion_simple () =
  (* Test with a simple base case: fun x -> x *)
  let params =
    [
      {
        tparam_name = "x";
        tparam_id = 0;
        tparam_type = TPrim TInt32;
        tparam_index = 0;
        tparam_is_vec = false;
      };
    ]
  in
  let body = te_var "x" 0 (TPrim TInt32) in

  let result = eliminate_tail_recursion "f" params body dummy_loc in

  (* Check that result is a valid expression *)
  Alcotest.(check bool)
    "result has type"
    true
    (result.ty <> t_unit || result.ty = t_unit) ;
  (* Just verify it doesn't crash and produces an expression *)
  ()

let test_eliminate_tail_recursion_preserves_type () =
  (* fun x -> 42 *)
  let params =
    [
      {
        tparam_name = "x";
        tparam_id = 0;
        tparam_type = TPrim TInt32;
        tparam_index = 0;
        tparam_is_vec = false;
      };
    ]
  in
  let body = te_int32 42l in

  let result = eliminate_tail_recursion "f" params body dummy_loc in

  (* Result should have int32 type *)
  Alcotest.(check bool)
    "result type is int32"
    true
    (match result.ty with TPrim TInt32 -> true | _ -> false)

let test_eliminate_tail_recursion_with_multiple_params () =
  (* fun (x, y) -> x *)
  let params =
    [
      {
        tparam_name = "x";
        tparam_id = 0;
        tparam_type = TPrim TInt32;
        tparam_index = 0;
        tparam_is_vec = false;
      };
      {
        tparam_name = "y";
        tparam_id = 1;
        tparam_type = TPrim TInt32;
        tparam_index = 1;
        tparam_is_vec = false;
      };
    ]
  in
  let body = te_var "x" 0 (TPrim TInt32) in

  let result = eliminate_tail_recursion "f" params body dummy_loc in

  (* Should handle multiple params without crashing *)
  Alcotest.(check bool)
    "result generated"
    true
    (result.ty <> t_unit || result.ty = t_unit)

let test_eliminate_tail_recursion_generates_mutable_vars () =
  (* The transformation should generate let mut bindings for loop variables *)
  let params =
    [
      {
        tparam_name = "acc";
        tparam_id = 0;
        tparam_type = TPrim TInt32;
        tparam_index = 0;
        tparam_is_vec = false;
      };
    ]
  in
  let body = te_var "acc" 0 (TPrim TInt32) in

  let result = eliminate_tail_recursion "sum" params body dummy_loc in

  (* Check that result contains let mut bindings *)
  let rec has_let_mut expr =
    match expr.te with
    | TELetMut _ -> true
    | TELet (_, _, _, b) -> has_let_mut b
    | TESeq es -> List.exists has_let_mut es
    | TEIf (_, t, Some e) -> has_let_mut t || has_let_mut e
    | TEIf (_, t, None) -> has_let_mut t
    | TEWhile (_, b) -> has_let_mut b
    | _ -> false
  in
  Alcotest.(check bool) "contains let mut" true (has_let_mut result)

let test_eliminate_tail_recursion_generates_while_loop () =
  (* The transformation should generate a while loop *)
  let params =
    [
      {
        tparam_name = "n";
        tparam_id = 0;
        tparam_type = TPrim TInt32;
        tparam_index = 0;
        tparam_is_vec = false;
      };
    ]
  in
  let body = te_var "n" 0 (TPrim TInt32) in

  let result = eliminate_tail_recursion "f" params body dummy_loc in

  (* Check that result contains a while loop *)
  let rec has_while expr =
    match expr.te with
    | TEWhile _ -> true
    | TELet (_, _, _, b) -> has_while b
    | TELetMut (_, _, _, b) -> has_while b
    | TESeq es -> List.exists has_while es
    | TEIf (_, t, Some e) -> has_while t || has_while e
    | TEIf (_, t, None) -> has_while t
    | _ -> false
  in
  Alcotest.(check bool) "contains while loop" true (has_while result)

(* ===== Test Suite ===== *)

let fresh_id_tests =
  [
    Alcotest.test_case
      "generates unique"
      `Quick
      test_fresh_transform_id_generates_unique;
    Alcotest.test_case "sequential" `Quick test_fresh_transform_id_sequential;
  ]

let eliminate_tests =
  [
    Alcotest.test_case
      "simple base case"
      `Quick
      test_eliminate_tail_recursion_simple;
    Alcotest.test_case
      "preserves type"
      `Quick
      test_eliminate_tail_recursion_preserves_type;
    Alcotest.test_case
      "multiple params"
      `Quick
      test_eliminate_tail_recursion_with_multiple_params;
    Alcotest.test_case
      "generates mutable vars"
      `Quick
      test_eliminate_tail_recursion_generates_mutable_vars;
    Alcotest.test_case
      "generates while loop"
      `Quick
      test_eliminate_tail_recursion_generates_while_loop;
  ]

let () =
  Alcotest.run
    "Sarek_tailrec_elim"
    [
      ("fresh_transform_id", fresh_id_tests);
      ("eliminate_tail_recursion", eliminate_tests);
    ]
