(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * OpenCL Error Tests - Verify Shared Error Module
 ******************************************************************************)

open Sarek_opencl

[@@@warning "-21"]

(** Test error construction and formatting *)
let test_codegen_errors () =
  (* Test unsupported_construct *)
  let e1 = Opencl_error.unsupported_construct "EArrayCreate" "test reason" in
  let s1 = Opencl_error.to_string e1 in
  Alcotest.(check bool)
    "unsupported_construct contains construct name"
    true
    (Str.string_match (Str.regexp ".*EArrayCreate.*") s1 0) ;

  (* Test invalid_arg_count *)
  let e2 = Opencl_error.invalid_arg_count "atomic_add" 2 3 in
  let s2 = Opencl_error.to_string e2 in
  Alcotest.(check bool)
    "invalid_arg_count mentions expected and got"
    true
    (Str.string_match (Str.regexp ".*2.*3.*") s2 0) ;

  (* Test unknown_intrinsic *)
  let e3 = Opencl_error.unknown_intrinsic "unknown_func" in
  let s3 = Opencl_error.to_string e3 in
  Alcotest.(check bool)
    "unknown_intrinsic contains function name"
    true
    (Str.string_match (Str.regexp ".*unknown_func.*") s3 0)

let test_runtime_errors () =
  (* Test device_not_found *)
  let e1 = Opencl_error.device_not_found 5 3 in
  let s1 = Opencl_error.to_string e1 in
  Alcotest.(check bool)
    "device_not_found shows range"
    true
    (Str.string_match (Str.regexp ".*0-2.*") s1 0) ;

  (* Test compilation_failed *)
  let e2 = Opencl_error.compilation_failed "kernel code" "syntax error" in
  let s2 = Opencl_error.to_string e2 in
  (* Use search_forward to handle multiline strings *)
  let contains_compilation =
    try
      ignore (Str.search_forward (Str.regexp "[Cc]ompilation") s2 0) ;
      true
    with Not_found -> false
  in
  Alcotest.(check bool)
    "compilation_failed contains 'compilation'"
    true
    contains_compilation ;

  (* Test no_device_selected *)
  let e3 = Opencl_error.no_device_selected "kernel_launch" in
  let s3 = Opencl_error.to_string e3 in
  Alcotest.(check bool)
    "no_device_selected mentions operation"
    true
    (Str.string_match (Str.regexp ".*kernel_launch.*") s3 0)

let test_plugin_errors () =
  (* Test unsupported_source_lang *)
  let e1 = Opencl_error.unsupported_source_lang "HLSL" in
  let s1 = Opencl_error.to_string e1 in
  Alcotest.(check bool)
    "unsupported_source_lang contains language"
    true
    (Str.string_match (Str.regexp ".*HLSL.*") s1 0) ;

  (* Test library_not_found *)
  let e2 = Opencl_error.library_not_found "libOpenCL.so" ["/usr/lib"] in
  let s2 = Opencl_error.to_string e2 in
  Alcotest.(check bool)
    "library_not_found contains library name"
    true
    (Str.string_match (Str.regexp ".*libOpenCL.*") s2 0) ;

  (* Test feature_not_supported *)
  let e3 = Opencl_error.feature_not_supported "D2D copy" in
  let s3 = Opencl_error.to_string e3 in
  Alcotest.(check bool)
    "feature_not_supported contains feature"
    true
    (Str.string_match (Str.regexp ".*D2D.*") s3 0)

let test_with_default () =
  (* Test with_default helper *)
  let result =
    Opencl_error.with_default ~default:42 (fun () ->
        Opencl_error.raise_error
          (Opencl_error.backend_unavailable "test reason") ;
        0)
  in
  Alcotest.(check int) "with_default returns default on error" 42 result

let test_to_result () =
  (* Test to_result for success *)
  let result_ok = Opencl_error.to_result (fun () -> 123) in
  Alcotest.(check bool)
    "to_result returns Ok on success"
    true
    (match result_ok with Ok 123 -> true | _ -> false) ;

  (* Test to_result for error *)
  let result_err =
    Opencl_error.to_result (fun () ->
        Opencl_error.raise_error
          (Opencl_error.backend_unavailable "test reason"))
  in
  Alcotest.(check bool)
    "to_result returns Error on failure"
    true
    (match result_err with Error _ -> true | _ -> false)

let test_error_equality () =
  (* Same errors should be equal *)
  let e1 = Opencl_error.unknown_intrinsic "func1" in
  let e2 = Opencl_error.unknown_intrinsic "func1" in
  Alcotest.(check bool) "Same errors are equal" true (e1 = e2) ;

  (* Different errors should not be equal *)
  let e3 = Opencl_error.unknown_intrinsic "func2" in
  let e4 = Opencl_error.unsupported_source_lang "CUDA" in
  Alcotest.(check bool) "Different errors are not equal" false (e3 = e4)

(** Test suite *)
let () =
  Alcotest.run
    "Opencl_error"
    [
      ( "codegen_errors",
        [Alcotest.test_case "unsupported_construct" `Quick test_codegen_errors]
      );
      ( "runtime_errors",
        [Alcotest.test_case "device operations" `Quick test_runtime_errors] );
      ( "plugin_errors",
        [Alcotest.test_case "unsupported operations" `Quick test_plugin_errors]
      );
      ( "utilities",
        [
          Alcotest.test_case "with_default" `Quick test_with_default;
          Alcotest.test_case "to_result" `Quick test_to_result;
          Alcotest.test_case "error_equality" `Quick test_error_equality;
        ] );
    ]
