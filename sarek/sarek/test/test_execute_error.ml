(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Execute_error module *)

open Alcotest
open Execute_error

(** Test error to string conversions *)
let test_unbound_variable_error () =
  let err = Unbound_variable "foo" in
  let msg = error_to_string err in
  check bool "contains 'foo'" true (String.length msg > 0) ;
  check bool "contains 'Unbound'" true (String.contains msg 'U')

let test_type_mismatch_error () =
  let err =
    Type_mismatch {expected = "int"; actual = "float"; context = "kernel_args"}
  in
  let msg = error_to_string err in
  check bool "contains expected type" true (Str.string_match (Str.regexp ".*int.*") msg 0) ;
  check bool "contains actual type" true (Str.string_match (Str.regexp ".*float.*") msg 0)

let test_unsupported_argument_error () =
  let err = Unsupported_argument {arg_type = "custom_type"; context = "execute"} in
  let msg = error_to_string err in
  check bool "contains arg_type" true (Str.string_match (Str.regexp ".*custom_type.*") msg 0)

let test_backend_error () =
  let err = Backend_error {backend = "cuda"; message = "device not found"} in
  let msg = error_to_string err in
  check bool "contains backend name" true (Str.string_match (Str.regexp ".*cuda.*") msg 0) ;
  check bool "contains message" true (Str.string_match (Str.regexp ".*device.*") msg 0)

let test_compilation_failed_error () =
  let err = Compilation_failed {kernel = "vector_add"; reason = "syntax error"} in
  let msg = error_to_string err in
  check bool "contains kernel name" true (Str.string_match (Str.regexp ".*vector_add.*") msg 0)

let test_invalid_dimensions_error () =
  let err =
    Invalid_dimensions
      {grid = "(1,1,1)"; block = "(0,0,0)"; reason = "block size cannot be zero"}
  in
  let msg = error_to_string err in
  check bool "contains reason" true (Str.string_match (Str.regexp ".*zero.*") msg 0)

let test_missing_ir_error () =
  let err = Missing_ir {kernel = "my_kernel"} in
  let msg = error_to_string err in
  check bool "contains kernel name" true (Str.string_match (Str.regexp ".*my_kernel.*") msg 0) ;
  check bool "mentions JIT" true (Str.string_match (Str.regexp ".*JIT.*") msg 0)

let test_missing_native_fn_error () =
  let err = Missing_native_fn {kernel = "native_kernel"} in
  let msg = error_to_string err in
  check bool "contains kernel name" true (Str.string_match (Str.regexp ".*native_kernel.*") msg 0)

let test_transfer_failed_error () =
  let err = Transfer_failed {vector = "vec_a"; reason = "out of memory"} in
  let msg = error_to_string err in
  check bool "contains vector name" true (Str.string_match (Str.regexp ".*vec_a.*") msg 0)

let test_interp_error () =
  let err = Interp_error "interpreter crashed" in
  let msg = error_to_string err in
  check bool "contains message" true (Str.string_match (Str.regexp ".*crashed.*") msg 0)

let test_invalid_file_error () =
  let err = Invalid_file {path = "/tmp/test.cl"; reason = "not found"} in
  let msg = error_to_string err in
  check bool "contains path" true (Str.string_match (Str.regexp ".*/tmp/test.cl.*") msg 0)

let test_type_helper_not_found_error () =
  let err =
    Type_helper_not_found {type_name = "MyRecord"; context = "interpreter"}
  in
  let msg = error_to_string err in
  check bool "contains type name" true (Str.string_match (Str.regexp ".*MyRecord.*") msg 0) ;
  check bool "mentions sarek.type" true (Str.string_match (Str.regexp ".*sarek.type.*") msg 0)

(** Test raising errors *)
let test_raise_execution_error () =
  try
    raise_error (Unbound_variable "test_var") ;
    fail "Expected exception"
  with Execution_error _ -> ()

let test_exception_contains_error () =
  try
    raise_error (Backend_error {backend = "test"; message = "test error"}) ;
    fail "Expected exception"
  with Execution_error err -> (
    match err with
    | Backend_error {backend; message} ->
        check string "backend is test" "test" backend ;
        check string "message is test error" "test error" message
    | _ -> fail "wrong error type")

(** Test error pattern matching *)
let test_pattern_match_unbound () =
  let err = Unbound_variable "x" in
  match err with
  | Unbound_variable name -> check string "variable name" "x" name
  | _ -> fail "wrong pattern"

let test_pattern_match_type_mismatch () =
  let err =
    Type_mismatch {expected = "bool"; actual = "int"; context = "condition"}
  in
  match err with
  | Type_mismatch {expected; actual; context} ->
      check string "expected" "bool" expected ;
      check string "actual" "int" actual ;
      check string "context" "condition" context
  | _ -> fail "wrong pattern"

(** Test error construction *)
let test_construct_backend_error () =
  let err = Backend_error {backend = "opencl"; message = "platform init failed"} in
  match err with
  | Backend_error _ -> ()
  | _ -> fail "construction failed"

let test_construct_compilation_error () =
  let err = Compilation_failed {kernel = "test"; reason = "type error"} in
  match err with
  | Compilation_failed _ -> ()
  | _ -> fail "construction failed"

(** Test error message formatting *)
let test_error_message_non_empty () =
  let errors =
    [
      Unbound_variable "x";
      Type_mismatch {expected = "int"; actual = "float"; context = "test"};
      Backend_error {backend = "cuda"; message = "error"};
      Missing_ir {kernel = "k"};
    ]
  in
  List.iter
    (fun err ->
      let msg = error_to_string err in
      check bool "message non-empty" true (String.length msg > 0))
    errors

let () =
  run
    "Execute_error"
    [
      ( "error_to_string",
        [
          test_case "unbound_variable" `Quick test_unbound_variable_error;
          test_case "type_mismatch" `Quick test_type_mismatch_error;
          test_case "unsupported_argument" `Quick test_unsupported_argument_error;
          test_case "backend_error" `Quick test_backend_error;
          test_case "compilation_failed" `Quick test_compilation_failed_error;
          test_case "invalid_dimensions" `Quick test_invalid_dimensions_error;
          test_case "missing_ir" `Quick test_missing_ir_error;
          test_case "missing_native_fn" `Quick test_missing_native_fn_error;
          test_case "transfer_failed" `Quick test_transfer_failed_error;
          test_case "interp_error" `Quick test_interp_error;
          test_case "invalid_file" `Quick test_invalid_file_error;
          test_case "type_helper_not_found" `Quick
            test_type_helper_not_found_error;
        ] );
      ( "raise_error",
        [
          test_case "raise_execution_error" `Quick test_raise_execution_error;
          test_case "exception_contains_error" `Quick
            test_exception_contains_error;
        ] );
      ( "pattern_matching",
        [
          test_case "match_unbound" `Quick test_pattern_match_unbound;
          test_case "match_type_mismatch" `Quick test_pattern_match_type_mismatch;
        ] );
      ( "construction",
        [
          test_case "backend_error" `Quick test_construct_backend_error;
          test_case "compilation_error" `Quick test_construct_compilation_error;
        ] );
      ( "formatting",
        [
          test_case "messages_non_empty" `Quick test_error_message_non_empty;
        ] );
    ]
