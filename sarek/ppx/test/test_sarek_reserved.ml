(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unit tests for Sarek_reserved module *)

open Alcotest

(** Test C keywords *)
let test_c_keyword_if () =
  check bool "if is reserved" true (Sarek_reserved.is_reserved "if")

let test_c_keyword_while () =
  check bool "while is reserved" true (Sarek_reserved.is_reserved "while")

let test_c_keyword_return () =
  check bool "return is reserved" true (Sarek_reserved.is_reserved "return")

let test_c_keyword_struct () =
  check bool "struct is reserved" true (Sarek_reserved.is_reserved "struct")

let test_c_keyword_sizeof () =
  check bool "sizeof is reserved" true (Sarek_reserved.is_reserved "sizeof")

let test_c_keyword_unsigned () =
  check bool "unsigned is reserved" true (Sarek_reserved.is_reserved "unsigned")

(** Test OpenCL keywords *)
let test_opencl_kernel () =
  check bool "__kernel is reserved" true (Sarek_reserved.is_reserved "__kernel")

let test_opencl_global () =
  check bool "__global is reserved" true (Sarek_reserved.is_reserved "__global")

let test_opencl_local () =
  check bool "__local is reserved" true (Sarek_reserved.is_reserved "__local")

let test_opencl_barrier () =
  check bool "barrier is reserved" true (Sarek_reserved.is_reserved "barrier")

let test_opencl_vector_types () =
  check bool "float4 is reserved" true (Sarek_reserved.is_reserved "float4") ;
  check bool "int2 is reserved" true (Sarek_reserved.is_reserved "int2") ;
  check bool "uchar8 is reserved" true (Sarek_reserved.is_reserved "uchar8")

(** Test CUDA keywords *)
let test_cuda_device () =
  check
    bool
    "__device__ is reserved"
    true
    (Sarek_reserved.is_reserved "__device__")

let test_cuda_global () =
  check
    bool
    "__global__ is reserved"
    true
    (Sarek_reserved.is_reserved "__global__")

let test_cuda_shared () =
  check
    bool
    "__shared__ is reserved"
    true
    (Sarek_reserved.is_reserved "__shared__")

let test_cuda_syncthreads () =
  check
    bool
    "__syncthreads is reserved"
    true
    (Sarek_reserved.is_reserved "__syncthreads")

let test_cuda_builtin_vars () =
  check
    bool
    "threadIdx is reserved"
    true
    (Sarek_reserved.is_reserved "threadIdx") ;
  check bool "blockIdx is reserved" true (Sarek_reserved.is_reserved "blockIdx") ;
  check bool "blockDim is reserved" true (Sarek_reserved.is_reserved "blockDim")

(** Test non-reserved identifiers *)
let test_my_function_not_reserved () =
  check
    bool
    "my_function is not reserved"
    false
    (Sarek_reserved.is_reserved "my_function")

let test_kernel_name_not_reserved () =
  check
    bool
    "vector_add is not reserved"
    false
    (Sarek_reserved.is_reserved "vector_add")

let test_camel_case_not_reserved () =
  check
    bool
    "calculateSum is not reserved"
    false
    (Sarek_reserved.is_reserved "calculateSum")

let test_underscore_prefix_allowed () =
  check bool "_temp is not reserved" false (Sarek_reserved.is_reserved "_temp")

let test_custom_type_not_reserved () =
  check
    bool
    "my_record_t is not reserved"
    false
    (Sarek_reserved.is_reserved "my_record_t")

(** Test case sensitivity *)
let test_uppercase_c_keyword () =
  check
    bool
    "IF is not reserved (case sensitive)"
    false
    (Sarek_reserved.is_reserved "IF")

let test_uppercase_while () =
  check bool "WHILE is not reserved" false (Sarek_reserved.is_reserved "WHILE")

(** Test edge cases *)
let test_empty_string () =
  check
    bool
    "empty string is not reserved"
    false
    (Sarek_reserved.is_reserved "")

let test_keyword_with_suffix () =
  check
    bool
    "float_value not reserved"
    false
    (Sarek_reserved.is_reserved "float_value")

let test_keyword_with_prefix () =
  check bool "my_int not reserved" false (Sarek_reserved.is_reserved "my_int")

(** Test comprehensive keyword coverage *)
let test_all_c_keywords_reserved () =
  List.iter
    (fun kw ->
      check bool (kw ^ " is reserved") true (Sarek_reserved.is_reserved kw))
    Sarek_reserved.c_keywords

let test_multiple_checks () =
  (* Test multiple keywords in sequence *)
  check bool "int reserved" true (Sarek_reserved.is_reserved "int") ;
  check bool "float reserved" true (Sarek_reserved.is_reserved "float") ;
  check bool "double reserved" true (Sarek_reserved.is_reserved "double") ;
  check bool "void reserved" true (Sarek_reserved.is_reserved "void")

let () =
  run
    "Sarek_reserved"
    [
      ( "c_keywords",
        [
          test_case "if" `Quick test_c_keyword_if;
          test_case "while" `Quick test_c_keyword_while;
          test_case "return" `Quick test_c_keyword_return;
          test_case "struct" `Quick test_c_keyword_struct;
          test_case "sizeof" `Quick test_c_keyword_sizeof;
          test_case "unsigned" `Quick test_c_keyword_unsigned;
          test_case "all_c_keywords" `Quick test_all_c_keywords_reserved;
        ] );
      ( "opencl_keywords",
        [
          test_case "kernel" `Quick test_opencl_kernel;
          test_case "global" `Quick test_opencl_global;
          test_case "local" `Quick test_opencl_local;
          test_case "barrier" `Quick test_opencl_barrier;
          test_case "vector_types" `Quick test_opencl_vector_types;
        ] );
      ( "cuda_keywords",
        [
          test_case "device" `Quick test_cuda_device;
          test_case "global" `Quick test_cuda_global;
          test_case "shared" `Quick test_cuda_shared;
          test_case "syncthreads" `Quick test_cuda_syncthreads;
          test_case "builtin_vars" `Quick test_cuda_builtin_vars;
        ] );
      ( "non_reserved",
        [
          test_case "my_function" `Quick test_my_function_not_reserved;
          test_case "kernel_name" `Quick test_kernel_name_not_reserved;
          test_case "camel_case" `Quick test_camel_case_not_reserved;
          test_case "underscore_prefix" `Quick test_underscore_prefix_allowed;
          test_case "custom_type" `Quick test_custom_type_not_reserved;
        ] );
      ( "case_sensitivity",
        [
          test_case "uppercase_c" `Quick test_uppercase_c_keyword;
          test_case "uppercase_while" `Quick test_uppercase_while;
        ] );
      ( "edge_cases",
        [
          test_case "empty_string" `Quick test_empty_string;
          test_case "keyword_with_suffix" `Quick test_keyword_with_suffix;
          test_case "keyword_with_prefix" `Quick test_keyword_with_prefix;
          test_case "multiple_checks" `Quick test_multiple_checks;
        ] );
    ]
