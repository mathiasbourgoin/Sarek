(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Interp_error module *)

open Alcotest
open Interp_error

(** Test error to string conversions *)
let test_unbound_variable_error () =
  let err = Unbound_variable {name = "x"; context = "eval_expr"} in
  let msg = error_to_string err in
  check bool "contains variable name" true (Str.string_match (Str.regexp ".*x.*") msg 0) ;
  check bool "contains context" true (Str.string_match (Str.regexp ".*eval_expr.*") msg 0)

let test_type_conversion_error () =
  let err =
    Type_conversion_error
      {from_type = "float"; to_type = "int32"; context = "binary_op"}
  in
  let msg = error_to_string err in
  check bool "contains from_type" true (Str.string_match (Str.regexp ".*float.*") msg 0) ;
  check bool "contains to_type" true (Str.string_match (Str.regexp ".*int32.*") msg 0)

let test_array_bounds_error () =
  let err = Array_bounds_error {array_name = "data"; index = 10; length = 5} in
  let msg = error_to_string err in
  check bool "contains array name" true (Str.string_match (Str.regexp ".*data.*") msg 0) ;
  check bool "contains index" true (Str.string_match (Str.regexp ".*10.*") msg 0) ;
  check bool "contains length" true (Str.string_match (Str.regexp ".*5.*") msg 0)

let test_unknown_intrinsic_error () =
  let err = Unknown_intrinsic {name = "my_func"} in
  let msg = error_to_string err in
  check bool "contains function name" true (Str.string_match (Str.regexp ".*my_func.*") msg 0)

let test_unknown_function_error () =
  let err = Unknown_function {name = "helper"} in
  let msg = error_to_string err in
  check bool "contains function name" true (Str.string_match (Str.regexp ".*helper.*") msg 0)

let test_pattern_match_failure_error () =
  let err = Pattern_match_failure {context = "eval_binop"} in
  let msg = error_to_string err in
  check bool "contains context" true (Str.string_match (Str.regexp ".*eval_binop.*") msg 0)

let test_not_an_array_error () =
  let err = Not_an_array {expr = "scalar_value"} in
  let msg = error_to_string err in
  check bool "contains expr" true (Str.string_match (Str.regexp ".*scalar_value.*") msg 0) ;
  check bool "mentions array" true (Str.string_match (Str.regexp ".*array.*") msg 0)

let test_not_a_record_error () =
  let err = Not_a_record {expr = "primitive"} in
  let msg = error_to_string err in
  check bool "contains expr" true (Str.string_match (Str.regexp ".*primitive.*") msg 0) ;
  check bool "mentions record" true (Str.string_match (Str.regexp ".*record.*") msg 0)

let test_unsupported_operation_error () =
  let err =
    Unsupported_operation {operation = "tensor_mul"; reason = "not implemented"}
  in
  let msg = error_to_string err in
  check bool "contains operation" true (Str.string_match (Str.regexp ".*tensor_mul.*") msg 0)

let test_bsp_deadlock_error () =
  let err = BSP_deadlock {message = "threads stuck at barrier"} in
  let msg = error_to_string err in
  check bool "contains message" true (Str.string_match (Str.regexp ".*barrier.*") msg 0) ;
  check bool "mentions deadlock" true (Str.string_match (Str.regexp ".*deadlock.*") msg 0)

(** Test raising errors *)
let test_raise_interpreter_error () =
  try
    raise_error (Unknown_intrinsic {name = "test"}) ;
    fail "Expected exception"
  with Interpreter_error _ -> ()

let test_exception_contains_error () =
  try
    raise_error (Array_bounds_error {array_name = "arr"; index = 5; length = 3}) ;
    fail "Expected exception"
  with Interpreter_error err -> (
    match err with
    | Array_bounds_error {array_name; index; length} ->
        check string "array name" "arr" array_name ;
        check int "index" 5 index ;
        check int "length" 3 length
    | _ -> fail "wrong error type")

(** Test error pattern matching *)
let test_pattern_match_unbound () =
  let err = Unbound_variable {name = "y"; context = "test"} in
  match err with
  | Unbound_variable {name; context} ->
      check string "variable name" "y" name ;
      check string "context" "test" context
  | _ -> fail "wrong pattern"

let test_pattern_match_type_conversion () =
  let err =
    Type_conversion_error
      {from_type = "string"; to_type = "int"; context = "parse"}
  in
  match err with
  | Type_conversion_error {from_type; to_type; context} ->
      check string "from_type" "string" from_type ;
      check string "to_type" "int" to_type ;
      check string "context" "parse" context
  | _ -> fail "wrong pattern"

(** Test error construction *)
let test_construct_bounds_error () =
  let err = Array_bounds_error {array_name = "test"; index = 0; length = 0} in
  match err with
  | Array_bounds_error _ -> ()
  | _ -> fail "construction failed"

let test_construct_unsupported_op_error () =
  let err = Unsupported_operation {operation = "test"; reason = "not ready"} in
  match err with
  | Unsupported_operation _ -> ()
  | _ -> fail "construction failed"

(** Test error message formatting *)
let test_error_messages_non_empty () =
  let errors =
    [
      Unbound_variable {name = "x"; context = "test"};
      Type_conversion_error {from_type = "a"; to_type = "b"; context = "c"};
      Unknown_intrinsic {name = "func"};
      Pattern_match_failure {context = "test"};
      BSP_deadlock {message = "stuck"};
    ]
  in
  List.iter
    (fun err ->
      let msg = error_to_string err in
      check bool "message non-empty" true (String.length msg > 0))
    errors

let () =
  run
    "Interp_error"
    [
      ( "error_to_string",
        [
          test_case "unbound_variable" `Quick test_unbound_variable_error;
          test_case "type_conversion" `Quick test_type_conversion_error;
          test_case "array_bounds" `Quick test_array_bounds_error;
          test_case "unknown_intrinsic" `Quick test_unknown_intrinsic_error;
          test_case "unknown_function" `Quick test_unknown_function_error;
          test_case "pattern_match_failure" `Quick
            test_pattern_match_failure_error;
          test_case "not_an_array" `Quick test_not_an_array_error;
          test_case "not_a_record" `Quick test_not_a_record_error;
          test_case "unsupported_operation" `Quick
            test_unsupported_operation_error;
          test_case "bsp_deadlock" `Quick test_bsp_deadlock_error;
        ] );
      ( "raise_error",
        [
          test_case "raise_interpreter_error" `Quick test_raise_interpreter_error;
          test_case "exception_contains_error" `Quick
            test_exception_contains_error;
        ] );
      ( "pattern_matching",
        [
          test_case "match_unbound" `Quick test_pattern_match_unbound;
          test_case "match_type_conversion" `Quick
            test_pattern_match_type_conversion;
        ] );
      ( "construction",
        [
          test_case "bounds_error" `Quick test_construct_bounds_error;
          test_case "unsupported_op" `Quick test_construct_unsupported_op_error;
        ] );
      ( "formatting",
        [
          test_case "messages_non_empty" `Quick test_error_messages_non_empty;
        ] );
    ]
