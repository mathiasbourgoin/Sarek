(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal Error Tests - Verify Shared Error Module
 ******************************************************************************)

open Sarek_metal

[@@@warning "-21"]

let test_codegen_errors () =
  let e1 = Metal_error.unsupported_construct "EArrayCreate" "test reason" in
  let s1 = Metal_error.to_string e1 in
  Alcotest.(check bool)
    "unsupported_construct contains construct name"
    true
    (Str.string_match (Str.regexp ".*EArrayCreate.*") s1 0) ;

  let e2 = Metal_error.invalid_arg_count "atomic_add" 2 3 in
  let s2 = Metal_error.to_string e2 in
  Alcotest.(check bool)
    "invalid_arg_count mentions expected and got"
    true
    (Str.string_match (Str.regexp ".*2.*3.*") s2 0) ;

  let e3 = Metal_error.unknown_intrinsic "unknown_func" in
  let s3 = Metal_error.to_string e3 in
  Alcotest.(check bool)
    "unknown_intrinsic contains function name"
    true
    (Str.string_match (Str.regexp ".*unknown_func.*") s3 0)

let test_runtime_errors () =
  let e1 = Metal_error.device_not_found 5 3 in
  let s1 = Metal_error.to_string e1 in
  Alcotest.(check bool)
    "device_not_found shows range"
    true
    (Str.string_match (Str.regexp ".*0-2.*") s1 0) ;

  let e2 = Metal_error.backend_unavailable "No Metal device found" in
  let s2 = Metal_error.to_string e2 in
  let contains_unavailable =
    try
      ignore (Str.search_forward (Str.regexp "[Uu]navailable") s2 0) ;
      true
    with Not_found -> false
  in
  Alcotest.(check bool)
    "backend_unavailable contains 'unavailable'"
    true
    contains_unavailable ;

  let e3 = Metal_error.no_device_selected "kernel_launch" in
  let s3 = Metal_error.to_string e3 in
  Alcotest.(check bool)
    "no_device_selected contains operation"
    true
    (Str.string_match (Str.regexp ".*kernel_launch.*") s3 0)

let test_plugin_errors () =
  let e1 = Metal_error.library_not_found "Metal" [] in
  let s1 = Metal_error.to_string e1 in
  Alcotest.(check bool)
    "library_not_found contains library name"
    true
    (Str.string_match (Str.regexp ".*Metal.*") s1 0) ;

  let e2 = Metal_error.feature_not_supported "raw pointers" in
  let s2 = Metal_error.to_string e2 in
  Alcotest.(check bool)
    "feature_not_supported contains feature"
    true
    (Str.string_match (Str.regexp ".*raw pointers.*") s2 0)

let test_backend_error_exception () =
  let e = Metal_error.unknown_intrinsic "test" in
  Alcotest.check_raises
    "raise_error raises Backend_error"
    (Spoc_framework.Backend_error.Backend_error e)
    (fun () -> Metal_error.raise_error e)

let test_error_prefix () =
  let e = Metal_error.unknown_intrinsic "test" in
  let s = Metal_error.to_string e in
  Alcotest.(check bool)
    "error string starts with [Metal"
    true
    (Str.string_match (Str.regexp "^\\[Metal.*") s 0)

let test_utilities () =
  let result1 =
    Metal_error.with_default ~default:"default" (fun () ->
        Metal_error.raise_error (Metal_error.unknown_intrinsic "test"))
  in
  Alcotest.(check string) "with_default returns default" "default" result1 ;

  let result2 =
    Metal_error.with_default ~default:"default" (fun () -> "success")
  in
  Alcotest.(check string) "with_default returns result" "success" result2 ;

  let result3 =
    Metal_error.to_result (fun () ->
        Metal_error.raise_error (Metal_error.unknown_intrinsic "test"))
  in
  (match result3 with
  | Ok _ -> Alcotest.fail "to_result should return Error"
  | Error _ -> ()) ;

  let result4 = Metal_error.to_result (fun () -> "success") in
  match result4 with
  | Ok s -> Alcotest.(check string) "to_result returns Ok" "success" s
  | Error _ -> Alcotest.fail "to_result should return Ok"

let () =
  Alcotest.run
    "Metal_error"
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
