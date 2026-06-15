(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Cuda_error module *)

open Sarek_cuda

(** Test error type construction and to_string conversion *)
let test_codegen_errors () =
  (* Test unsupported_construct *)
  let e1 = Cuda_error.unsupported_construct "SNative" "test reason" in
  let s1 = Cuda_error.to_string e1 in
  Alcotest.(check bool)
    "unsupported_construct contains construct name"
    true
    (String.length s1 > 0 && Str.(string_match (regexp ".*SNative.*") s1 0)) ;

  (* Test invalid_memory_space *)
  let e2 = Cuda_error.invalid_memory_space "gen_param" "DLocal or DShared" in
  let s2 = Cuda_error.to_string e2 in
  Alcotest.(check bool)
    "invalid_memory_space contains context"
    true
    (String.length s2 > 0 && Str.(string_match (regexp ".*gen_param.*") s2 0)) ;

  (* Test type_error *)
  let e3 =
    Cuda_error.type_error "pattern match" "matching bindings" "mismatch"
  in
  let s3 = Cuda_error.to_string e3 in
  Alcotest.(check bool) "type_error is formatted" true (String.length s3 > 0)

let test_runtime_errors () =
  (* Test no_device_selected *)
  let e1 = Cuda_error.no_device_selected "kernel_execution" in
  let s1 = Cuda_error.to_string e1 in
  Alcotest.(check bool)
    "no_device_selected contains operation"
    true
    Str.(string_match (regexp ".*kernel_execution.*") s1 0) ;

  (* Test compilation_failed *)
  let e2 =
    Cuda_error.compilation_failed "test kernel" "syntax error at line 5"
  in
  let s2 = Cuda_error.to_string e2 in
  (* Use search_forward to find substring (handles newlines) *)
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

  (* Test device_not_found *)
  let e3 = Cuda_error.device_not_found 5 2 in
  let s3 = Cuda_error.to_string e3 in
  Alcotest.(check bool)
    "device_not_found has message"
    true
    (String.length s3 > 0)

let test_plugin_errors () =
  (* Test unsupported_source_lang *)
  let e1 = Cuda_error.unsupported_source_lang "GLSL" in
  let s1 = Cuda_error.to_string e1 in
  Alcotest.(check bool)
    "unsupported_source_lang contains language"
    true
    Str.(string_match (regexp ".*GLSL.*") s1 0) ;

  (* Test library_not_found *)
  let e2 = Cuda_error.library_not_found "libcuda.so" ["path1"; "path2"] in
  let s2 = Cuda_error.to_string e2 in
  Alcotest.(check bool)
    "library_not_found contains library name"
    true
    Str.(string_match (regexp ".*libcuda.*") s2 0)

let test_with_default () =
  (* Test with_default helper *)
  let err = Cuda_error.no_device_selected "test_op" in
  let result =
    Cuda_error.with_default ~default:99 (fun () ->
        raise (Cuda_error.Cuda_error err))
  in
  Alcotest.(check int) "with_default returns default on error" 99 result

let test_to_result () =
  (* Test to_result helper *)
  let err = Cuda_error.no_device_selected "test_op" in
  let result =
    Cuda_error.to_result (fun () -> raise (Cuda_error.Cuda_error err))
  in
  match result with
  | Ok _ -> Alcotest.fail "Expected Error, got Ok"
  | Error e ->
      let s = Cuda_error.to_string e in
      Alcotest.(check bool) "to_result captures error" true (String.length s > 0)

let test_error_equality () =
  (* Test that same error constructors produce equal errors *)
  let e1 = Cuda_error.no_device_selected "op1" in
  let e2 = Cuda_error.no_device_selected "op1" in
  Alcotest.(check bool) "Same error constructors are equal" true (e1 = e2) ;

  (* Different errors should not be equal *)
  let e3 = Cuda_error.unsupported_source_lang "PTX" in
  let e4 = Cuda_error.unsupported_source_lang "GLSL" in
  Alcotest.(check bool) "Different errors are not equal" false (e3 = e4)

(** Test suite *)
let () =
  Alcotest.run
    "Cuda_error"
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
