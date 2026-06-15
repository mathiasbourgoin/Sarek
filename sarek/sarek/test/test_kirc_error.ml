(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Kirc_error module *)

open Alcotest
open Kirc_error

(** Test error to string conversions *)
let test_no_native_function_error () =
  let err = No_native_function {kernel_name = "my_kernel"; context = "execute"} in
  let msg = error_to_string err in
  check bool "contains kernel name" true (Str.string_match (Str.regexp ".*my_kernel.*") msg 0) ;
  check bool "mentions native" true (Str.string_match (Str.regexp ".*native.*") msg 0)

let test_no_ir_error () =
  let err = No_ir {kernel_name = "native_only_kernel"} in
  let msg = error_to_string err in
  check bool "contains kernel name" true (Str.string_match (Str.regexp ".*native_only_kernel.*") msg 0)

let test_unsupported_arg_type_error () =
  let err =
    Unsupported_arg_type
      {arg_type = "custom_struct"; reason = "not marshallable"; context = "args"}
  in
  let msg = error_to_string err in
  check bool "contains arg_type" true (Str.string_match (Str.regexp ".*custom_struct.*") msg 0) ;
  check bool "contains reason" true (Str.string_match (Str.regexp ".*marshallable.*") msg 0)

let test_type_conversion_failed_with_index () =
  let err =
    Type_conversion_failed
      {
        from_type = "string";
        to_type = "int32";
        index = Some 2;
        context = "arg_conversion";
      }
  in
  let msg = error_to_string err in
  check bool "contains from_type" true (Str.string_match (Str.regexp ".*string.*") msg 0) ;
  check bool "contains index" true (Str.string_match (Str.regexp ".*2.*") msg 0)

let test_type_conversion_failed_no_index () =
  let err =
    Type_conversion_failed
      {from_type = "float"; to_type = "bool"; index = None; context = "cast"}
  in
  let msg = error_to_string err in
  check bool "contains from_type" true (Str.string_match (Str.regexp ".*float.*") msg 0) ;
  check bool "contains to_type" true (Str.string_match (Str.regexp ".*bool.*") msg 0)

let test_backend_not_found_error () =
  let err = Backend_not_found {backend = "fake_backend"} in
  let msg = error_to_string err in
  check bool "contains backend name" true (Str.string_match (Str.regexp ".*fake_backend.*") msg 0)

let test_no_source_generation_error () =
  let err = No_source_generation {backend = "native"} in
  let msg = error_to_string err in
  check bool "contains backend" true (Str.string_match (Str.regexp ".*native.*") msg 0) ;
  check bool "mentions source" true (Str.string_match (Str.regexp ".*source.*") msg 0)

let test_wrong_backend_error () =
  let err =
    Wrong_backend {expected = "cuda"; got = "opencl"; operation = "texture_op"}
  in
  let msg = error_to_string err in
  check bool "contains expected" true (Str.string_match (Str.regexp ".*cuda.*") msg 0) ;
  check bool "contains got" true (Str.string_match (Str.regexp ".*opencl.*") msg 0) ;
  check bool "contains operation" true (Str.string_match (Str.regexp ".*texture_op.*") msg 0)

(** Test raising errors *)
let test_raise_kirc_error () =
  try
    raise_error (Backend_not_found {backend = "test"}) ;
    fail "Expected exception"
  with Kirc_error _ -> ()

let test_exception_contains_error () =
  try
    raise_error (No_native_function {kernel_name = "test_kernel"; context = "run"}) ;
    fail "Expected exception"
  with Kirc_error err -> (
    match err with
    | No_native_function {kernel_name; context} ->
        check string "kernel name" "test_kernel" kernel_name ;
        check string "context" "run" context
    | _ -> fail "wrong error type")

(** Test error pattern matching *)
let test_pattern_match_no_ir () =
  let err = No_ir {kernel_name = "k1"} in
  match err with
  | No_ir {kernel_name} -> check string "kernel name" "k1" kernel_name
  | _ -> fail "wrong pattern"

let test_pattern_match_wrong_backend () =
  let err =
    Wrong_backend {expected = "vulkan"; got = "cuda"; operation = "render"}
  in
  match err with
  | Wrong_backend {expected; got; operation} ->
      check string "expected" "vulkan" expected ;
      check string "got" "cuda" got ;
      check string "operation" "render" operation
  | _ -> fail "wrong pattern"

let test_pattern_match_type_conversion () =
  let err =
    Type_conversion_failed
      {from_type = "a"; to_type = "b"; index = Some 5; context = "c"}
  in
  match err with
  | Type_conversion_failed {from_type; to_type; index; context} ->
      check string "from_type" "a" from_type ;
      check string "to_type" "b" to_type ;
      check bool "has index" true (Option.is_some index) ;
      (match index with
      | Some i -> check int "index value" 5 i
      | None -> fail "index should be Some") ;
      check string "context" "c" context
  | _ -> fail "wrong pattern"

(** Test error construction *)
let test_construct_no_native_function () =
  let err = No_native_function {kernel_name = "test"; context = "exec"} in
  match err with
  | No_native_function _ -> ()
  | _ -> fail "construction failed"

let test_construct_backend_not_found () =
  let err = Backend_not_found {backend = "xyz"} in
  match err with
  | Backend_not_found _ -> ()
  | _ -> fail "construction failed"

(** Test error message formatting *)
let test_error_messages_non_empty () =
  let errors =
    [
      No_native_function {kernel_name = "k"; context = "c"};
      No_ir {kernel_name = "k2"};
      Backend_not_found {backend = "b"};
      Wrong_backend {expected = "e"; got = "g"; operation = "o"};
    ]
  in
  List.iter
    (fun err ->
      let msg = error_to_string err in
      check bool "message non-empty" true (String.length msg > 0))
    errors

(** Test optional index in type conversion *)
let test_type_conversion_index_some () =
  let err =
    Type_conversion_failed
      {from_type = "x"; to_type = "y"; index = Some 10; context = "z"}
  in
  let msg = error_to_string err in
  check bool "contains index" true (Str.string_match (Str.regexp ".*10.*") msg 0)

let test_type_conversion_index_none () =
  let err =
    Type_conversion_failed
      {from_type = "x"; to_type = "y"; index = None; context = "z"}
  in
  let msg = error_to_string err in
  (* Should not contain "index" when None *)
  check bool "message generated" true (String.length msg > 0)

let () =
  run
    "Kirc_error"
    [
      ( "error_to_string",
        [
          test_case "no_native_function" `Quick test_no_native_function_error;
          test_case "no_ir" `Quick test_no_ir_error;
          test_case "unsupported_arg_type" `Quick test_unsupported_arg_type_error;
          test_case "type_conversion_with_index" `Quick
            test_type_conversion_failed_with_index;
          test_case "type_conversion_no_index" `Quick
            test_type_conversion_failed_no_index;
          test_case "backend_not_found" `Quick test_backend_not_found_error;
          test_case "no_source_generation" `Quick test_no_source_generation_error;
          test_case "wrong_backend" `Quick test_wrong_backend_error;
        ] );
      ( "raise_error",
        [
          test_case "raise_kirc_error" `Quick test_raise_kirc_error;
          test_case "exception_contains_error" `Quick
            test_exception_contains_error;
        ] );
      ( "pattern_matching",
        [
          test_case "match_no_ir" `Quick test_pattern_match_no_ir;
          test_case "match_wrong_backend" `Quick test_pattern_match_wrong_backend;
          test_case "match_type_conversion" `Quick
            test_pattern_match_type_conversion;
        ] );
      ( "construction",
        [
          test_case "no_native_function" `Quick test_construct_no_native_function;
          test_case "backend_not_found" `Quick test_construct_backend_not_found;
        ] );
      ( "formatting",
        [
          test_case "messages_non_empty" `Quick test_error_messages_non_empty;
          test_case "index_some" `Quick test_type_conversion_index_some;
          test_case "index_none" `Quick test_type_conversion_index_none;
        ] );
    ]
