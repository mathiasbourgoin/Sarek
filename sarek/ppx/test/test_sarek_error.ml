(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_error module *)

open Alcotest
open Sarek_ast
open Sarek_types
open Sarek_error

(** Helper to create a dummy location *)
let dummy_loc =
  {
    loc_file = "test.ml";
    loc_line = 1;
    loc_col = 0;
    loc_end_line = 1;
    loc_end_col = 0;
  }

(** Test error location extraction *)
let test_error_loc_unbound_variable () =
  let err = Unbound_variable ("x", dummy_loc) in
  let loc = error_loc err in
  check string "location file" "test.ml" loc.loc_file

let test_error_loc_type_mismatch () =
  let err =
    Type_mismatch {expected = TPrim TInt32; got = TReg Float32; loc = dummy_loc}
  in
  let loc = error_loc err in
  check int "location line" 1 loc.loc_line

(** Test error to string conversion *)
let test_unbound_variable_string () =
  let err = Unbound_variable ("my_var", dummy_loc) in
  let s = error_to_string err in
  check bool "contains variable name" true (String.contains s 'y')

let test_type_mismatch_string () =
  let err =
    Type_mismatch {expected = TPrim TInt32; got = TReg Float32; loc = dummy_loc}
  in
  let s = error_to_string err in
  check bool "non-empty" true (String.length s > 0)

let test_unbound_constructor_string () =
  let err = Unbound_constructor ("Some", dummy_loc) in
  let s = error_to_string err in
  check bool "contains Some" true (Str.string_match (Str.regexp ".*Some.*") s 0)

let test_parse_error_string () =
  let err = Parse_error ("unexpected token", dummy_loc) in
  let s = error_to_string err in
  check
    bool
    "contains message"
    true
    (Str.string_match (Str.regexp ".*unexpected.*") s 0)

(** Test error construction *)
let test_construct_unbound_field () =
  let err = Unbound_field ("x", dummy_loc) in
  match err with
  | Unbound_field (name, _) -> check string "field name" "x" name
  | _ -> fail "wrong error type"

let test_construct_wrong_arity () =
  let err = Wrong_arity {expected = 2; got = 3; loc = dummy_loc} in
  match err with
  | Wrong_arity {expected; got; _} ->
      check int "expected" 2 expected ;
      check int "got" 3 got
  | _ -> fail "wrong error type"

let test_construct_not_a_function () =
  let err = Not_a_function (TPrim TInt32, dummy_loc) in
  match err with Not_a_function (_, _) -> () | _ -> fail "wrong error type"

(** Test barrier-related errors *)
let test_barrier_in_diverged_flow () =
  let err = Barrier_in_diverged_flow dummy_loc in
  let s = error_to_string err in
  check
    bool
    "mentions barrier"
    true
    (Str.string_match (Str.regexp ".*[Bb]arrier.*") s 0)

let test_warp_collective_in_diverged_flow () =
  let err = Warp_collective_in_diverged_flow ("shuffle", dummy_loc) in
  let s = error_to_string err in
  check
    bool
    "mentions shuffle"
    true
    (Str.string_match (Str.regexp ".*shuffle.*") s 0)

(** Test reserved keyword error *)
let test_reserved_keyword () =
  let err = Reserved_keyword ("float", dummy_loc) in
  let s = error_to_string err in
  check
    bool
    "mentions float"
    true
    (Str.string_match (Str.regexp ".*float.*") s 0)

(** Test lowering errors *)
let test_unsupported_type_in_registration () =
  let err = Unsupported_type_in_registration ("function type", dummy_loc) in
  let s = error_to_string err in
  check
    bool
    "mentions function type"
    true
    (Str.string_match (Str.regexp ".*function.*") s 0)

let test_unsupported_constructor_form () =
  let err = Unsupported_constructor_form dummy_loc in
  let s = error_to_string err in
  check bool "non-empty" true (String.length s > 0)

let test_unknown_variant_type () =
  let err = Unknown_variant_type ("MyType", dummy_loc) in
  let s = error_to_string err in
  check
    bool
    "contains MyType"
    true
    (Str.string_match (Str.regexp ".*MyType.*") s 0)

(** Test invalid lvalue *)
let test_invalid_lvalue () =
  let err = Invalid_lvalue dummy_loc in
  let s = error_to_string err in
  check
    bool
    "mentions assignment"
    true
    (Str.string_match (Str.regexp ".*assignment.*") s 0)

(** Test duplicate field *)
let test_duplicate_field () =
  let err = Duplicate_field ("id", dummy_loc) in
  let s = error_to_string err in
  check
    bool
    "contains field name"
    true
    (Str.string_match (Str.regexp ".*id.*") s 0)

(** Test missing type annotation *)
let test_missing_type_annotation () =
  let err = Missing_type_annotation ("param", dummy_loc) in
  let s = error_to_string err in
  check
    bool
    "contains param"
    true
    (Str.string_match (Str.regexp ".*param.*") s 0)

(** Test result monad *)
let test_ok_result () =
  let r = ok 42 in
  match r with Ok v -> check int "value" 42 v | Error _ -> fail "expected Ok"

let test_error_result () =
  let err = Unbound_variable ("x", dummy_loc) in
  let r = error err in
  match r with
  | Error es -> check int "error count" 1 (List.length es)
  | Ok _ -> fail "expected Error"

let test_combine_results_all_ok () =
  let results = [ok 1; ok 2; ok 3] in
  let combined = combine_results results in
  match combined with
  | Ok vs -> check int "result count" 3 (List.length vs)
  | Error _ -> fail "expected Ok"

let test_combine_results_with_error () =
  let err = Unbound_variable ("x", dummy_loc) in
  let results = [ok 1; error err; ok 3] in
  let combined = combine_results results in
  match combined with
  | Error es -> check int "error count" 1 (List.length es)
  | Ok _ -> fail "expected Error"

(** Test all error types construct properly *)
let test_all_error_types () =
  let errors =
    [
      Unbound_variable ("x", dummy_loc);
      Unbound_constructor ("Some", dummy_loc);
      Unbound_field ("f", dummy_loc);
      Unbound_type ("t", dummy_loc);
      Type_mismatch
        {expected = TPrim TInt32; got = TReg Float32; loc = dummy_loc};
      Wrong_arity {expected = 1; got = 2; loc = dummy_loc};
      Not_a_function (TPrim TInt32, dummy_loc);
      Not_a_vector (TPrim TInt32, dummy_loc);
      Not_an_array (TPrim TInt32, dummy_loc);
      Reserved_keyword ("int", dummy_loc);
      Invalid_lvalue dummy_loc;
    ]
  in
  List.iter
    (fun err ->
      let s = error_to_string err in
      check bool "error string non-empty" true (String.length s > 0))
    errors

let () =
  run
    "Sarek_error"
    [
      ( "error_location",
        [
          test_case "unbound_variable_loc" `Quick test_error_loc_unbound_variable;
          test_case "type_mismatch_loc" `Quick test_error_loc_type_mismatch;
        ] );
      ( "error_to_string",
        [
          test_case "unbound_variable" `Quick test_unbound_variable_string;
          test_case "type_mismatch" `Quick test_type_mismatch_string;
          test_case "unbound_constructor" `Quick test_unbound_constructor_string;
          test_case "parse_error" `Quick test_parse_error_string;
        ] );
      ( "construction",
        [
          test_case "unbound_field" `Quick test_construct_unbound_field;
          test_case "wrong_arity" `Quick test_construct_wrong_arity;
          test_case "not_a_function" `Quick test_construct_not_a_function;
        ] );
      ( "barrier_errors",
        [
          test_case "barrier_diverged" `Quick test_barrier_in_diverged_flow;
          test_case
            "warp_collective"
            `Quick
            test_warp_collective_in_diverged_flow;
        ] );
      ( "lowering_errors",
        [
          test_case
            "unsupported_type"
            `Quick
            test_unsupported_type_in_registration;
          test_case
            "unsupported_constructor"
            `Quick
            test_unsupported_constructor_form;
          test_case "unknown_variant" `Quick test_unknown_variant_type;
        ] );
      ( "misc_errors",
        [
          test_case "reserved_keyword" `Quick test_reserved_keyword;
          test_case "invalid_lvalue" `Quick test_invalid_lvalue;
          test_case "duplicate_field" `Quick test_duplicate_field;
          test_case "missing_annotation" `Quick test_missing_type_annotation;
        ] );
      ( "result_monad",
        [
          test_case "ok" `Quick test_ok_result;
          test_case "error" `Quick test_error_result;
          test_case "combine_all_ok" `Quick test_combine_results_all_ok;
          test_case "combine_with_error" `Quick test_combine_results_with_error;
        ] );
      ( "comprehensive",
        [test_case "all_error_types" `Quick test_all_error_types] );
    ]
