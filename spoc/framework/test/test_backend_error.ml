(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Backend_error Tests - Verify Shared Error Module
 ******************************************************************************)

[@@@warning "-21"]

(** Test error construction and formatting *)
let test_codegen_errors () =
  let module E = Spoc_framework.Backend_error.Make (struct
    let name = "TestBackend"
  end) in
  (* Unknown intrinsic *)
  let err1 = E.unknown_intrinsic "unknown_func" in
  let str1 = Spoc_framework.Backend_error.to_string err1 in
  Alcotest.(check bool)
    "unknown intrinsic contains backend name"
    true
    (String.contains str1 'T') ;
  Alcotest.(check bool)
    "unknown intrinsic contains function name"
    true
    (Str.string_match (Str.regexp ".*unknown_func.*") str1 0) ;
  (* Invalid arg count *)
  let err2 = E.invalid_arg_count "add" 2 3 in
  let str2 = Spoc_framework.Backend_error.to_string err2 in
  Alcotest.(check bool)
    "invalid arg count mentions expected and got"
    true
    (Str.string_match (Str.regexp ".*2.*3.*") str2 0) ;
  (* Unsupported construct *)
  let err3 = E.unsupported_construct "goto" "not allowed in GPU kernels" in
  let str3 = Spoc_framework.Backend_error.to_string err3 in
  Alcotest.(check bool)
    "unsupported construct contains reason"
    true
    (Str.string_match (Str.regexp ".*not allowed.*") str3 0)

let test_runtime_errors () =
  let module E = Spoc_framework.Backend_error.Make (struct
    let name = "CUDA"
  end) in
  (* No device selected *)
  let err1 = E.no_device_selected "kernel_launch" in
  let str1 = Spoc_framework.Backend_error.to_string err1 in
  Alcotest.(check bool)
    "no device error mentions operation"
    true
    (Str.string_match (Str.regexp ".*kernel_launch.*") str1 0) ;
  (* Device not found *)
  let err2 = E.device_not_found 5 3 in
  let str2 = Spoc_framework.Backend_error.to_string err2 in
  Alcotest.(check bool)
    "device not found shows range"
    true
    (Str.string_match (Str.regexp ".*0-2.*") str2 0) ;
  (* Compilation failed *)
  let err3 = E.compilation_failed "int main() {}" "syntax error" in
  let str3 = Spoc_framework.Backend_error.to_string err3 in
  (* Check that the log message appears in the string *)
  let contains_log =
    try
      ignore (Str.search_forward (Str.regexp "syntax error") str3 0) ;
      true
    with Not_found -> false
  in
  Alcotest.(check bool) "compilation error includes log" true contains_log ;
  (* Memory allocation failed *)
  let err4 = E.memory_allocation_failed 1024L "out of memory" in
  let str4 = Spoc_framework.Backend_error.to_string err4 in
  Alcotest.(check bool)
    "memory error shows bytes"
    true
    (Str.string_match (Str.regexp ".*1024.*") str4 0)

let test_plugin_errors () =
  let module E = Spoc_framework.Backend_error.Make (struct
    let name = "OpenCL"
  end) in
  (* Unsupported source lang *)
  let err1 = E.unsupported_source_lang "HLSL" in
  let str1 = Spoc_framework.Backend_error.to_string err1 in
  Alcotest.(check bool)
    "unsupported lang mentions language"
    true
    (Str.string_match (Str.regexp ".*HLSL.*") str1 0) ;
  (* Backend unavailable *)
  let err2 = E.backend_unavailable "no drivers installed" in
  let str2 = Spoc_framework.Backend_error.to_string err2 in
  Alcotest.(check bool)
    "backend unavailable includes reason"
    true
    (Str.string_match (Str.regexp ".*no drivers.*") str2 0) ;
  (* Library not found *)
  let err3 =
    E.library_not_found "libOpenCL.so" ["/usr/lib"; "/usr/local/lib"]
  in
  let str3 = Spoc_framework.Backend_error.to_string err3 in
  Alcotest.(check bool)
    "library not found shows paths"
    true
    (Str.string_match (Str.regexp ".*/usr/lib.*") str3 0)

let test_multiple_backends () =
  (* Create errors for different backends *)
  let module CUDA = Spoc_framework.Backend_error.Make (struct
    let name = "CUDA"
  end) in
  let module OpenCL = Spoc_framework.Backend_error.Make (struct
    let name = "OpenCL"
  end) in
  let cuda_err = CUDA.unknown_intrinsic "cuda_func" in
  let opencl_err = OpenCL.unknown_intrinsic "opencl_func" in
  let cuda_str = Spoc_framework.Backend_error.to_string cuda_err in
  let opencl_str = Spoc_framework.Backend_error.to_string opencl_err in
  (* Verify backend names are different *)
  Alcotest.(check bool)
    "CUDA error contains CUDA"
    true
    (Str.string_match (Str.regexp ".*CUDA.*") cuda_str 0) ;
  Alcotest.(check bool)
    "OpenCL error contains OpenCL"
    true
    (Str.string_match (Str.regexp ".*OpenCL.*") opencl_str 0) ;
  (* Verify function names are preserved *)
  Alcotest.(check bool)
    "CUDA error contains cuda_func"
    true
    (Str.string_match (Str.regexp ".*cuda_func.*") cuda_str 0) ;
  Alcotest.(check bool)
    "OpenCL error contains opencl_func"
    true
    (Str.string_match (Str.regexp ".*opencl_func.*") opencl_str 0)

let test_exception_handling () =
  let module E = Spoc_framework.Backend_error.Make (struct
    let name = "Vulkan"
  end) in
  (* Test raise_error *)
  let raised =
    try
      Spoc_framework.Backend_error.raise_error (E.no_device_selected "test_op") ;
      false
    with Spoc_framework.Backend_error.Backend_error _ -> true
  in
  Alcotest.(check bool) "raise_error raises exception" true raised ;
  (* Test with_default *)
  let result =
    Spoc_framework.Backend_error.with_default ~default:42 (fun () ->
        Spoc_framework.Backend_error.raise_error (E.backend_unavailable "test") ;
        0)
  in
  Alcotest.(check int) "with_default returns default on error" 42 result ;
  (* Test to_result *)
  let result_ok = Spoc_framework.Backend_error.to_result (fun () -> 123) in
  Alcotest.(check bool)
    "to_result returns Ok on success"
    true
    (match result_ok with Ok 123 -> true | _ -> false) ;
  let result_err =
    Spoc_framework.Backend_error.to_result (fun () ->
        Spoc_framework.Backend_error.raise_error (E.backend_unavailable "test"))
  in
  Alcotest.(check bool)
    "to_result returns Error on failure"
    true
    (match result_err with Error _ -> true | _ -> false)

(** Test suite *)
let () =
  Alcotest.run
    "Backend_error"
    [
      ( "codegen",
        [Alcotest.test_case "codegen errors" `Quick test_codegen_errors] );
      ( "runtime",
        [Alcotest.test_case "runtime errors" `Quick test_runtime_errors] );
      ("plugin", [Alcotest.test_case "plugin errors" `Quick test_plugin_errors]);
      ( "backends",
        [Alcotest.test_case "multiple backends" `Quick test_multiple_backends]
      );
      ( "exceptions",
        [Alcotest.test_case "exception handling" `Quick test_exception_handling]
      );
    ]
