(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_reserved module
 *
 * Tests reserved keyword checking for C, CUDA, OpenCL identifiers
 ******************************************************************************)

[@@@warning "-32-34"]

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

let dummy_loc =
  Sarek_ast.
    {
      loc_file = "test.ml";
      loc_line = 1;
      loc_col = 0;
      loc_end_line = 1;
      loc_end_col = 10;
    }

(* Test: is_reserved recognizes C keywords *)
let test_is_reserved_c_keywords () =
  Alcotest.(check bool)
    "for is reserved"
    true
    (Sarek_reserved.is_reserved "for") ;
  Alcotest.(check bool)
    "while is reserved"
    true
    (Sarek_reserved.is_reserved "while") ;
  Alcotest.(check bool)
    "int is reserved"
    true
    (Sarek_reserved.is_reserved "int") ;
  Alcotest.(check bool)
    "float is reserved"
    true
    (Sarek_reserved.is_reserved "float") ;
  Alcotest.(check bool)
    "double is reserved"
    true
    (Sarek_reserved.is_reserved "double") ;
  Alcotest.(check bool)
    "return is reserved"
    true
    (Sarek_reserved.is_reserved "return") ;
  Alcotest.(check bool)
    "void is reserved"
    true
    (Sarek_reserved.is_reserved "void") ;
  Alcotest.(check bool)
    "struct is reserved"
    true
    (Sarek_reserved.is_reserved "struct")

let test_is_reserved_cuda_keywords () =
  Alcotest.(check bool)
    "__global__ is reserved"
    true
    (Sarek_reserved.is_reserved "__global__") ;
  Alcotest.(check bool)
    "__device__ is reserved"
    true
    (Sarek_reserved.is_reserved "__device__") ;
  Alcotest.(check bool)
    "__shared__ is reserved"
    true
    (Sarek_reserved.is_reserved "__shared__") ;
  Alcotest.(check bool)
    "__constant__ is reserved"
    true
    (Sarek_reserved.is_reserved "__constant__") ;
  Alcotest.(check bool)
    "threadIdx is reserved"
    true
    (Sarek_reserved.is_reserved "threadIdx") ;
  Alcotest.(check bool)
    "blockIdx is reserved"
    true
    (Sarek_reserved.is_reserved "blockIdx") ;
  Alcotest.(check bool)
    "blockDim is reserved"
    true
    (Sarek_reserved.is_reserved "blockDim") ;
  Alcotest.(check bool)
    "__syncthreads is reserved"
    true
    (Sarek_reserved.is_reserved "__syncthreads")

let test_is_reserved_opencl_keywords () =
  Alcotest.(check bool)
    "__kernel is reserved"
    true
    (Sarek_reserved.is_reserved "__kernel") ;
  Alcotest.(check bool)
    "kernel is reserved"
    true
    (Sarek_reserved.is_reserved "kernel") ;
  Alcotest.(check bool)
    "__global is reserved"
    true
    (Sarek_reserved.is_reserved "__global") ;
  Alcotest.(check bool)
    "global is reserved"
    true
    (Sarek_reserved.is_reserved "global") ;
  Alcotest.(check bool)
    "__local is reserved"
    true
    (Sarek_reserved.is_reserved "__local") ;
  Alcotest.(check bool)
    "local is reserved"
    true
    (Sarek_reserved.is_reserved "local") ;
  Alcotest.(check bool)
    "barrier is reserved"
    true
    (Sarek_reserved.is_reserved "barrier") ;
  Alcotest.(check bool)
    "get_global_id is reserved"
    true
    (Sarek_reserved.is_reserved "get_global_id")

let test_is_reserved_vector_types () =
  Alcotest.(check bool)
    "int2 is reserved"
    true
    (Sarek_reserved.is_reserved "int2") ;
  Alcotest.(check bool)
    "float4 is reserved"
    true
    (Sarek_reserved.is_reserved "float4") ;
  Alcotest.(check bool)
    "double2 is reserved"
    true
    (Sarek_reserved.is_reserved "double2") ;
  Alcotest.(check bool)
    "uint is reserved"
    true
    (Sarek_reserved.is_reserved "uint") ;
  Alcotest.(check bool)
    "ulong is reserved"
    true
    (Sarek_reserved.is_reserved "ulong")

let test_is_reserved_valid_identifiers () =
  Alcotest.(check bool)
    "foo not reserved"
    false
    (Sarek_reserved.is_reserved "foo") ;
  Alcotest.(check bool)
    "my_var not reserved"
    false
    (Sarek_reserved.is_reserved "my_var") ;
  Alcotest.(check bool)
    "calculate not reserved"
    false
    (Sarek_reserved.is_reserved "calculate") ;
  Alcotest.(check bool)
    "result not reserved"
    false
    (Sarek_reserved.is_reserved "result") ;
  Alcotest.(check bool)
    "Vector not reserved"
    false
    (Sarek_reserved.is_reserved "Vector") ;
  Alcotest.(check bool)
    "kernel_ not reserved"
    false
    (Sarek_reserved.is_reserved "kernel_")

(* Test: is_reserved with edge cases *)
let test_is_reserved_case_sensitive () =
  (* Reserved keywords are case-sensitive *)
  Alcotest.(check bool)
    "FOR not reserved"
    false
    (Sarek_reserved.is_reserved "FOR") ;
  Alcotest.(check bool)
    "for is reserved"
    true
    (Sarek_reserved.is_reserved "for") ;
  Alcotest.(check bool)
    "Float not reserved"
    false
    (Sarek_reserved.is_reserved "Float")

let test_is_reserved_partial_match () =
  (* Partial matches should not be reserved *)
  Alcotest.(check bool)
    "kernel_ not reserved"
    false
    (Sarek_reserved.is_reserved "kernel_") ;
  Alcotest.(check bool)
    "_kernel not reserved"
    false
    (Sarek_reserved.is_reserved "_kernel") ;
  Alcotest.(check bool)
    "myint not reserved"
    false
    (Sarek_reserved.is_reserved "myint")

let test_is_reserved_empty () =
  (* Empty string should not be reserved *)
  Alcotest.(check bool)
    "empty string not reserved"
    false
    (Sarek_reserved.is_reserved "")

(* Test: keyword lists are non-empty *)
let test_c_keywords_nonempty () =
  Alcotest.(check bool)
    "c_keywords non-empty"
    true
    (List.length Sarek_reserved.c_keywords > 0)

let test_opencl_keywords_nonempty () =
  Alcotest.(check bool)
    "opencl_keywords non-empty"
    true
    (List.length Sarek_reserved.opencl_keywords > 0)

let test_cuda_keywords_nonempty () =
  Alcotest.(check bool)
    "cuda_keywords non-empty"
    true
    (List.length Sarek_reserved.cuda_keywords > 0)

let test_all_lists_combined () =
  (* Check that lists have been combined into hashtable *)
  let has_c = Sarek_reserved.is_reserved "int" in
  let has_cuda = Sarek_reserved.is_reserved "__device__" in
  let has_opencl = Sarek_reserved.is_reserved "__kernel" in
  Alcotest.(check bool) "has C keyword" true has_c ;
  Alcotest.(check bool) "has CUDA keyword" true has_cuda ;
  Alcotest.(check bool) "has OpenCL keyword" true has_opencl

(* Test suite *)
let () =
  let open Alcotest in
  run
    "Sarek_reserved"
    [
      ( "is_reserved",
        [
          test_case "C keywords" `Quick test_is_reserved_c_keywords;
          test_case "CUDA keywords" `Quick test_is_reserved_cuda_keywords;
          test_case "OpenCL keywords" `Quick test_is_reserved_opencl_keywords;
          test_case "Vector types" `Quick test_is_reserved_vector_types;
          test_case
            "Valid identifiers"
            `Quick
            test_is_reserved_valid_identifiers;
          test_case "Case sensitive" `Quick test_is_reserved_case_sensitive;
          test_case "Partial match" `Quick test_is_reserved_partial_match;
          test_case "Empty string" `Quick test_is_reserved_empty;
        ] );
      ( "keyword_lists",
        [
          test_case "C keywords non-empty" `Quick test_c_keywords_nonempty;
          test_case
            "OpenCL keywords non-empty"
            `Quick
            test_opencl_keywords_nonempty;
          test_case "CUDA keywords non-empty" `Quick test_cuda_keywords_nonempty;
          test_case "All lists combined" `Quick test_all_lists_combined;
        ] );
    ]
