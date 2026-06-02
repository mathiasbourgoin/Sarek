(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_error module
 * Tests error types, location extraction, formatting, and monadic operations
 ******************************************************************************)

let dummy_loc =
  Sarek_ast.
    {
      loc_file = "test.ml";
      loc_line = 10;
      loc_col = 5;
      loc_end_line = 10;
      loc_end_col = 10;
    }

(* Helper to check if string contains substring *)
let contains s sub =
  try
    let _ = Str.search_forward (Str.regexp_string sub) s 0 in
    true
  with Not_found -> false

(* Test: error_loc extracts locations correctly *)
let test_error_loc_simple () =
  let err = Sarek_error.Unbound_variable ("x", dummy_loc) in
  let loc = Sarek_error.error_loc err in
  Alcotest.(check string) "file" "test.ml" loc.loc_file ;
  Alcotest.(check int) "line" 10 loc.loc_line ;
  Alcotest.(check int) "col" 5 loc.loc_col

let test_error_loc_record () =
  let err =
    Sarek_error.Type_mismatch
      {
        expected = Sarek_types.(TReg Int);
        got = Sarek_types.(TReg Float32);
        loc = dummy_loc;
      }
  in
  let loc = Sarek_error.error_loc err in
  Alcotest.(check string) "file" "test.ml" loc.loc_file

let test_error_loc_without_name () =
  let err = Sarek_error.Barrier_in_diverged_flow dummy_loc in
  let loc = Sarek_error.error_loc err in
  Alcotest.(check int) "line" 10 loc.loc_line

(* Test: error_to_string formats errors correctly *)
let test_error_to_string_unbound () =
  let err = Sarek_error.Unbound_variable ("foo", dummy_loc) in
  let s = Sarek_error.error_to_string err in
  Alcotest.(check bool) "contains variable name" true (contains s "foo") ;
  Alcotest.(check bool) "contains 'Unbound'" true (contains s "Unbound")

let test_error_to_string_type_mismatch () =
  let err =
    Sarek_error.Type_mismatch
      {
        expected = Sarek_types.(TReg Int);
        got = Sarek_types.(TReg Float32);
        loc = dummy_loc;
      }
  in
  let s = Sarek_error.error_to_string err in
  Alcotest.(check bool) "contains 'expected'" true (contains s "expected") ;
  Alcotest.(check bool)
    "contains type"
    true
    (contains s "int" || contains s "Int")

let test_error_to_string_barrier () =
  let err = Sarek_error.Barrier_in_diverged_flow dummy_loc in
  let s = Sarek_error.error_to_string err in
  Alcotest.(check bool) "contains 'Barrier'" true (contains s "Barrier") ;
  Alcotest.(check bool) "contains 'diverged'" true (contains s "diverged")

(* Test: pp_error_with_loc includes location *)
let test_pp_error_with_loc () =
  let err = Sarek_error.Unbound_variable ("x", dummy_loc) in
  let s = Format.asprintf "%a" Sarek_error.pp_error_with_loc err in
  Alcotest.(check bool) "contains file" true (contains s "test.ml") ;
  Alcotest.(check bool) "contains line" true (contains s "10") ;
  Alcotest.(check bool) "contains col" true (contains s "5")

(* Test: Monadic operations - ok *)
let test_ok () =
  let r = Sarek_error.ok 42 in
  match r with
  | Ok x -> Alcotest.(check int) "value" 42 x
  | Error _ -> Alcotest.fail "expected Ok"

(* Test: Monadic operations - error *)
let test_error () =
  let err = Sarek_error.Unbound_variable ("x", dummy_loc) in
  let r = Sarek_error.error err in
  match r with
  | Ok _ -> Alcotest.fail "expected Error"
  | Error es ->
      Alcotest.(check int) "single error" 1 (List.length es) ;
      Alcotest.(check bool)
        "correct error"
        true
        (match List.hd es with
        | Sarek_error.Unbound_variable ("x", _) -> true
        | _ -> false)

(* Test: Monadic operations - errors *)
let test_errors () =
  let err1 = Sarek_error.Unbound_variable ("x", dummy_loc) in
  let err2 = Sarek_error.Unbound_variable ("y", dummy_loc) in
  let r = Sarek_error.errors [err1; err2] in
  match r with
  | Ok _ -> Alcotest.fail "expected Error"
  | Error es -> Alcotest.(check int) "two errors" 2 (List.length es)

(* Test: Monadic operations - bind *)
let test_bind_ok () =
  let open Sarek_error in
  let r =
    let* x = ok 10 in
    ok (x * 2)
  in
  match r with
  | Ok x -> Alcotest.(check int) "bound value" 20 x
  | Error _ -> Alcotest.fail "expected Ok"

let test_bind_error () =
  let open Sarek_error in
  let err = Unbound_variable ("x", dummy_loc) in
  let r =
    let* _ = error err in
    ok 42
  in
  match r with Ok _ -> Alcotest.fail "expected Error" | Error _ -> ()

(* Test: map_result *)
let test_map_result_ok () =
  let r = Sarek_error.ok 5 in
  let r' = Sarek_error.map_result (fun x -> x * 2) r in
  match r' with
  | Ok x -> Alcotest.(check int) "mapped value" 10 x
  | Error _ -> Alcotest.fail "expected Ok"

let test_map_result_error () =
  let err = Sarek_error.Unbound_variable ("x", dummy_loc) in
  let r = Sarek_error.error err in
  let r' = Sarek_error.map_result (fun x -> x * 2) r in
  match r' with
  | Ok _ -> Alcotest.fail "expected Error"
  | Error es -> Alcotest.(check int) "preserved error" 1 (List.length es)

(* Test: combine_results - all ok *)
let test_combine_results_all_ok () =
  let results = [Sarek_error.ok 1; Sarek_error.ok 2; Sarek_error.ok 3] in
  let r = Sarek_error.combine_results results in
  match r with
  | Ok xs -> Alcotest.(check (list int)) "combined values" [1; 2; 3] xs
  | Error _ -> Alcotest.fail "expected Ok"

(* Test: combine_results - some errors *)
let test_combine_results_with_errors () =
  let err1 = Sarek_error.Unbound_variable ("x", dummy_loc) in
  let err2 = Sarek_error.Unbound_variable ("y", dummy_loc) in
  let results =
    [
      Sarek_error.ok 1;
      Sarek_error.error err1;
      Sarek_error.ok 2;
      Sarek_error.error err2;
    ]
  in
  let r = Sarek_error.combine_results results in
  match r with
  | Ok _ -> Alcotest.fail "expected Error"
  | Error es -> Alcotest.(check int) "accumulated errors" 2 (List.length es)

(* Test: combine_results - empty list *)
let test_combine_results_empty () =
  let results = [] in
  let r = Sarek_error.combine_results results in
  match r with
  | Ok xs -> Alcotest.(check (list int)) "empty list" [] xs
  | Error _ -> Alcotest.fail "expected Ok"

(* Test: All error constructors have working error_loc *)
let test_all_error_constructors_have_loc () =
  let errors =
    [
      Sarek_error.Unbound_variable ("x", dummy_loc);
      Sarek_error.Unbound_constructor ("C", dummy_loc);
      Sarek_error.Unbound_field ("f", dummy_loc);
      Sarek_error.Unbound_type ("T", dummy_loc);
      Sarek_error.Type_mismatch
        {
          expected = Sarek_types.(TReg Int);
          got = Sarek_types.(TReg Float32);
          loc = dummy_loc;
        };
      Sarek_error.Cannot_unify
        (Sarek_types.(TReg Int), Sarek_types.(TReg Float32), dummy_loc);
      Sarek_error.Not_a_function (Sarek_types.(TReg Int), dummy_loc);
      Sarek_error.Wrong_arity {expected = 2; got = 3; loc = dummy_loc};
      Sarek_error.Not_a_vector (Sarek_types.(TReg Int), dummy_loc);
      Sarek_error.Not_an_array (Sarek_types.(TReg Int), dummy_loc);
      Sarek_error.Not_a_record (Sarek_types.(TReg Int), dummy_loc);
      Sarek_error.Field_not_found ("f", Sarek_types.(TReg Int), dummy_loc);
      Sarek_error.Immutable_variable ("x", dummy_loc);
      Sarek_error.Recursive_type (Sarek_types.(TReg Int), dummy_loc);
      Sarek_error.Unsupported_expression ("match", dummy_loc);
      Sarek_error.Parse_error ("syntax", dummy_loc);
      Sarek_error.Invalid_kernel ("bad", dummy_loc);
      Sarek_error.Duplicate_field ("f", dummy_loc);
      Sarek_error.Missing_type_annotation ("x", dummy_loc);
      Sarek_error.Invalid_intrinsic ("bad_intrinsic", dummy_loc);
      Sarek_error.Barrier_in_diverged_flow dummy_loc;
      Sarek_error.Warp_collective_in_diverged_flow ("warp_reduce", dummy_loc);
      Sarek_error.Reserved_keyword ("void", dummy_loc);
      Sarek_error.Unsupported_type_in_registration ("T", dummy_loc);
      Sarek_error.Unsupported_constructor_form dummy_loc;
      Sarek_error.Unsupported_registration_form dummy_loc;
      Sarek_error.Unsupported_tuple_in_variant dummy_loc;
      Sarek_error.Unsupported_function_in_variant dummy_loc;
      Sarek_error.Unknown_variant_type ("T", dummy_loc);
      Sarek_error.Expression_needs_statement_context ("unit", dummy_loc);
      Sarek_error.Invalid_lvalue dummy_loc;
    ]
  in
  List.iter
    (fun err ->
      let loc = Sarek_error.error_loc err in
      Alcotest.(check string) "location extracted" "test.ml" loc.loc_file)
    errors

(* Test: Error formatting contains expected keywords *)
let test_error_messages_contain_keywords () =
  let test_cases =
    [
      (Sarek_error.Unbound_variable ("x", dummy_loc), "Unbound");
      ( Sarek_error.Wrong_arity {expected = 2; got = 3; loc = dummy_loc},
        "arguments" );
      (Sarek_error.Reserved_keyword ("void", dummy_loc), "reserved");
      (Sarek_error.Barrier_in_diverged_flow dummy_loc, "workgroup");
      (Sarek_error.Unsupported_expression ("match", dummy_loc), "mutable");
    ]
  in
  List.iter
    (fun (err, keyword) ->
      let s = Sarek_error.error_to_string err in
      Alcotest.(check bool)
        (Format.sprintf "contains '%s'" keyword)
        true
        (contains (String.lowercase_ascii s) (String.lowercase_ascii keyword)))
    test_cases

(* Test suite *)
let () =
  Alcotest.run
    "Sarek_error"
    [
      ( "error_loc",
        [
          Alcotest.test_case "simple error" `Quick test_error_loc_simple;
          Alcotest.test_case "record error" `Quick test_error_loc_record;
          Alcotest.test_case "no name error" `Quick test_error_loc_without_name;
        ] );
      ( "formatting",
        [
          Alcotest.test_case
            "unbound variable"
            `Quick
            test_error_to_string_unbound;
          Alcotest.test_case
            "type mismatch"
            `Quick
            test_error_to_string_type_mismatch;
          Alcotest.test_case "barrier" `Quick test_error_to_string_barrier;
          Alcotest.test_case "with location" `Quick test_pp_error_with_loc;
        ] );
      ( "monadic",
        [
          Alcotest.test_case "ok" `Quick test_ok;
          Alcotest.test_case "error" `Quick test_error;
          Alcotest.test_case "errors" `Quick test_errors;
          Alcotest.test_case "bind ok" `Quick test_bind_ok;
          Alcotest.test_case "bind error" `Quick test_bind_error;
          Alcotest.test_case "map_result ok" `Quick test_map_result_ok;
          Alcotest.test_case "map_result error" `Quick test_map_result_error;
        ] );
      ( "combine_results",
        [
          Alcotest.test_case "all ok" `Quick test_combine_results_all_ok;
          Alcotest.test_case
            "with errors"
            `Quick
            test_combine_results_with_errors;
          Alcotest.test_case "empty list" `Quick test_combine_results_empty;
        ] );
      ( "comprehensive",
        [
          Alcotest.test_case
            "all constructors"
            `Quick
            test_all_error_constructors_have_loc;
          Alcotest.test_case
            "message keywords"
            `Quick
            test_error_messages_contain_keywords;
        ] );
    ]
