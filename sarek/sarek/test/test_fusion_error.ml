(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Fusion_error module *)

open Alcotest
open Fusion_error

(** Test error to string conversions *)
let test_empty_pipeline_error () =
  let err = Empty_pipeline {function_name = "fuse_all"} in
  let msg = error_to_string err in
  check bool "contains function name" true (Str.string_match (Str.regexp ".*fuse_all.*") msg 0) ;
  check bool "mentions empty" true (Str.string_match (Str.regexp ".*empty.*") msg 0)

let test_fusion_incompatible_error () =
  let err =
    Fusion_incompatible
      {
        producer = "kernel_a";
        consumer = "kernel_b";
        reason = "different data types";
      }
  in
  let msg = error_to_string err in
  check bool "contains producer" true (Str.string_match (Str.regexp ".*kernel_a.*") msg 0) ;
  check bool "contains consumer" true (Str.string_match (Str.regexp ".*kernel_b.*") msg 0) ;
  check bool "contains reason" true (Str.string_match (Str.regexp ".*data types.*") msg 0)

let test_invalid_fusion_error () =
  let err = Invalid_fusion {kernel = "fused_kernel"; reason = "missing outputs"} in
  let msg = error_to_string err in
  check bool "contains kernel name" true (Str.string_match (Str.regexp ".*fused_kernel.*") msg 0) ;
  check bool "contains reason" true (Str.string_match (Str.regexp ".*outputs.*") msg 0)

(** Test raising errors *)
let test_raise_fusion_error () =
  try
    raise_error (Empty_pipeline {function_name = "test"}) ;
    fail "Expected exception"
  with Fusion_error _ -> ()

let test_exception_contains_error () =
  try
    raise_error
      (Fusion_incompatible
         {producer = "p"; consumer = "c"; reason = "incompatible"}) ;
    fail "Expected exception"
  with Fusion_error err -> (
    match err with
    | Fusion_incompatible {producer; consumer; reason} ->
        check string "producer" "p" producer ;
        check string "consumer" "c" consumer ;
        check string "reason" "incompatible" reason
    | _ -> fail "wrong error type")

(** Test error pattern matching *)
let test_pattern_match_empty_pipeline () =
  let err = Empty_pipeline {function_name = "my_fuse"} in
  match err with
  | Empty_pipeline {function_name} ->
      check string "function name" "my_fuse" function_name
  | _ -> fail "wrong pattern"

let test_pattern_match_incompatible () =
  let err =
    Fusion_incompatible {producer = "k1"; consumer = "k2"; reason = "mismatch"}
  in
  match err with
  | Fusion_incompatible {producer; consumer; reason} ->
      check string "producer" "k1" producer ;
      check string "consumer" "k2" consumer ;
      check string "reason" "mismatch" reason
  | _ -> fail "wrong pattern"

let test_pattern_match_invalid_fusion () =
  let err = Invalid_fusion {kernel = "k"; reason = "r"} in
  match err with
  | Invalid_fusion {kernel; reason} ->
      check string "kernel" "k" kernel ;
      check string "reason" "r" reason
  | _ -> fail "wrong pattern"

(** Test error construction *)
let test_construct_empty_pipeline () =
  let err = Empty_pipeline {function_name = "test"} in
  match err with
  | Empty_pipeline _ -> ()
  | _ -> fail "construction failed"

let test_construct_incompatible () =
  let err = Fusion_incompatible {producer = "a"; consumer = "b"; reason = "c"} in
  match err with
  | Fusion_incompatible _ -> ()
  | _ -> fail "construction failed"

let test_construct_invalid_fusion () =
  let err = Invalid_fusion {kernel = "k"; reason = "r"} in
  match err with
  | Invalid_fusion _ -> ()
  | _ -> fail "construction failed"

(** Test error message formatting *)
let test_error_messages_non_empty () =
  let errors =
    [
      Empty_pipeline {function_name = "f"};
      Fusion_incompatible {producer = "p"; consumer = "c"; reason = "r"};
      Invalid_fusion {kernel = "k"; reason = "r"};
    ]
  in
  List.iter
    (fun err ->
      let msg = error_to_string err in
      check bool "message non-empty" true (String.length msg > 0))
    errors

(** Test error messages contain key information *)
let test_empty_pipeline_mentions_function () =
  let err = Empty_pipeline {function_name = "fuse_pipeline"} in
  let msg = error_to_string err in
  check bool "mentions function" true
    (Str.string_match (Str.regexp ".*fuse_pipeline.*") msg 0)

let test_incompatible_shows_both_kernels () =
  let err =
    Fusion_incompatible {producer = "map"; consumer = "reduce"; reason = "types"}
  in
  let msg = error_to_string err in
  check bool "shows producer" true (Str.string_match (Str.regexp ".*map.*") msg 0) ;
  check bool "shows consumer" true (Str.string_match (Str.regexp ".*reduce.*") msg 0)

let test_invalid_fusion_shows_reason () =
  let err =
    Invalid_fusion {kernel = "bad_fuse"; reason = "circular dependency"}
  in
  let msg = error_to_string err in
  check bool "shows reason" true (Str.string_match (Str.regexp ".*circular.*") msg 0)

(** Test error type completeness *)
let test_all_error_types_covered () =
  let errors =
    [
      Empty_pipeline {function_name = "test"};
      Fusion_incompatible {producer = "p"; consumer = "c"; reason = "r"};
      Invalid_fusion {kernel = "k"; reason = "r"};
    ]
  in
  (* Just ensure all can be constructed and converted to strings *)
  List.iter
    (fun err ->
      let _ = error_to_string err in
      ())
    errors ;
  check bool "all types covered" true true

let () =
  run
    "Fusion_error"
    [
      ( "error_to_string",
        [
          test_case "empty_pipeline" `Quick test_empty_pipeline_error;
          test_case "fusion_incompatible" `Quick test_fusion_incompatible_error;
          test_case "invalid_fusion" `Quick test_invalid_fusion_error;
        ] );
      ( "raise_error",
        [
          test_case "raise_fusion_error" `Quick test_raise_fusion_error;
          test_case "exception_contains_error" `Quick
            test_exception_contains_error;
        ] );
      ( "pattern_matching",
        [
          test_case "match_empty_pipeline" `Quick test_pattern_match_empty_pipeline;
          test_case "match_incompatible" `Quick test_pattern_match_incompatible;
          test_case "match_invalid_fusion" `Quick test_pattern_match_invalid_fusion;
        ] );
      ( "construction",
        [
          test_case "empty_pipeline" `Quick test_construct_empty_pipeline;
          test_case "incompatible" `Quick test_construct_incompatible;
          test_case "invalid_fusion" `Quick test_construct_invalid_fusion;
        ] );
      ( "formatting",
        [
          test_case "messages_non_empty" `Quick test_error_messages_non_empty;
          test_case "empty_mentions_function" `Quick
            test_empty_pipeline_mentions_function;
          test_case "incompatible_shows_kernels" `Quick
            test_incompatible_shows_both_kernels;
          test_case "invalid_shows_reason" `Quick test_invalid_fusion_shows_reason;
          test_case "all_types_covered" `Quick test_all_error_types_covered;
        ] );
    ]
