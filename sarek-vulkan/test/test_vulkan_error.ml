(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Vulkan Error Tests - Verify Shared Error Module
 ******************************************************************************)

open Sarek_vulkan

[@@@warning "-21"]

(** Test error construction and formatting *)
let test_codegen_errors () =
  (* Test unsupported_construct *)
  let e1 = Vulkan_error.unsupported_construct "EArrayCreate" "test reason" in
  let s1 = Vulkan_error.to_string e1 in
  Alcotest.(check bool)
    "unsupported_construct contains construct name"
    true
    (Str.string_match (Str.regexp ".*EArrayCreate.*") s1 0) ;

  (* Test invalid_arg_count *)
  let e2 = Vulkan_error.invalid_arg_count "atomic_add" 2 3 in
  let s2 = Vulkan_error.to_string e2 in
  Alcotest.(check bool)
    "invalid_arg_count mentions expected and got"
    true
    (Str.string_match (Str.regexp ".*2.*3.*") s2 0) ;

  (* Test unknown_intrinsic *)
  let e3 = Vulkan_error.unknown_intrinsic "unknown_func" in
  let s3 = Vulkan_error.to_string e3 in
  Alcotest.(check bool)
    "unknown_intrinsic contains function name"
    true
    (Str.string_match (Str.regexp ".*unknown_func.*") s3 0)

let test_runtime_errors () =
  (* Test device_not_found *)
  let e1 = Vulkan_error.device_not_found 5 3 in
  let s1 = Vulkan_error.to_string e1 in
  Alcotest.(check bool)
    "device_not_found shows range"
    true
    (Str.string_match (Str.regexp ".*0-2.*") s1 0) ;

  (* Test compilation_failed *)
  let e2 = Vulkan_error.compilation_failed "shader code" "syntax error" in
  let s2 = Vulkan_error.to_string e2 in
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
  let e3 = Vulkan_error.no_device_selected "kernel_launch" in
  let s3 = Vulkan_error.to_string e3 in
  Alcotest.(check bool)
    "no_device_selected contains operation"
    true
    (Str.string_match (Str.regexp ".*kernel_launch.*") s3 0)

let test_plugin_errors () =
  (* Test library_not_found *)
  let e1 =
    Vulkan_error.library_not_found "vulkan" ["/usr/lib"; "/opt/vulkan"]
  in
  let s1 = Vulkan_error.to_string e1 in
  Alcotest.(check bool)
    "library_not_found contains library name"
    true
    (Str.string_match (Str.regexp ".*vulkan.*") s1 0) ;

  (* Test unsupported_source_lang *)
  let e2 = Vulkan_error.unsupported_source_lang "CUDA" in
  let s2 = Vulkan_error.to_string e2 in
  Alcotest.(check bool)
    "unsupported_source_lang contains language"
    true
    (Str.string_match (Str.regexp ".*CUDA.*") s2 0) ;

  (* Test feature_not_supported *)
  let e3 = Vulkan_error.feature_not_supported "raw pointers" in
  let s3 = Vulkan_error.to_string e3 in
  Alcotest.(check bool)
    "feature_not_supported contains feature"
    true
    (Str.string_match (Str.regexp ".*raw pointers.*") s3 0)

let test_backend_error_exception () =
  (* Test that raise_error actually raises Backend_error *)
  let e = Vulkan_error.unknown_intrinsic "test" in
  Alcotest.check_raises
    "raise_error raises Backend_error"
    (Spoc_framework.Backend_error.Backend_error e)
    (fun () -> Vulkan_error.raise_error e)

let test_error_prefix () =
  (* Verify all errors have [Vulkan ...] prefix *)
  let e = Vulkan_error.unknown_intrinsic "test" in
  let s = Vulkan_error.to_string e in
  Alcotest.(check bool)
    "error string starts with [Vulkan"
    true
    (Str.string_match (Str.regexp "^\\[Vulkan.*") s 0)

let test_utilities () =
  (* Test with_default *)
  let result1 =
    Vulkan_error.with_default ~default:"default" (fun () ->
        Vulkan_error.raise_error (Vulkan_error.unknown_intrinsic "test"))
  in
  Alcotest.(check string) "with_default returns default" "default" result1 ;

  let result2 =
    Vulkan_error.with_default ~default:"default" (fun () -> "success")
  in
  Alcotest.(check string) "with_default returns result" "success" result2 ;

  (* Test to_result *)
  let result3 =
    Vulkan_error.to_result (fun () ->
        Vulkan_error.raise_error (Vulkan_error.unknown_intrinsic "test"))
  in
  (match result3 with
  | Ok _ -> Alcotest.fail "to_result should return Error"
  | Error _ -> ()) ;

  let result4 = Vulkan_error.to_result (fun () -> "success") in
  match result4 with
  | Ok s -> Alcotest.(check string) "to_result returns Ok" "success" s
  | Error _ -> Alcotest.fail "to_result should return Ok"

(** Test suite *)
let () =
  Alcotest.run
    "Vulkan_error"
    [
      ( "codegen",
        [Alcotest.test_case "codegen errors" `Quick test_codegen_errors] );
      ( "runtime",
        [Alcotest.test_case "runtime errors" `Quick test_runtime_errors] );
      ("plugin", [Alcotest.test_case "plugin errors" `Quick test_plugin_errors]);
      ( "exception",
        [
          Alcotest.test_case
            "Backend_error exception"
            `Quick
            test_backend_error_exception;
        ] );
      ("prefix", [Alcotest.test_case "error prefix" `Quick test_error_prefix]);
      ("utilities", [Alcotest.test_case "utilities" `Quick test_utilities]);
    ]
