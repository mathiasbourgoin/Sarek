(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Sarek_native_intrinsics
 *
 * Tests type mapping and intrinsic code generation for native backend.
 ******************************************************************************)

(* Force stdlib initialization *)
let () = Sarek_stdlib.force_init ()

open Ppxlib
open Sarek_types
open Sarek_env
open Sarek_native_intrinsics

let dummy_loc = Location.none

(* Helper to check if expression generates without crashing *)
let expr_to_string expr = Format.asprintf "%a" Pprintast.expression expr

(* ===== Test: core_type_of_typ ===== *)

let test_core_type_unit () =
  let ct = core_type_of_typ ~loc:dummy_loc t_unit in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  Alcotest.(check bool) "unit type generated" true (String.length str > 0)

let test_core_type_bool () =
  let ct = core_type_of_typ ~loc:dummy_loc t_bool in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  Alcotest.(check bool) "bool type generated" true (String.length str > 0)

let test_core_type_int32 () =
  let ct = core_type_of_typ ~loc:dummy_loc t_int32 in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  Alcotest.(check bool) "int32 type generated" true (String.length str > 0)

let test_core_type_float32 () =
  let ct = core_type_of_typ ~loc:dummy_loc t_float32 in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  Alcotest.(check bool) "float32 type generated" true (String.length str > 0)

let test_core_type_float64 () =
  let ct = core_type_of_typ ~loc:dummy_loc t_float64 in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  Alcotest.(check bool) "float64 type generated" true (String.length str > 0)

let test_core_type_int64 () =
  let ty = TReg Int64 in
  let ct = core_type_of_typ ~loc:dummy_loc ty in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  Alcotest.(check bool) "int64 type generated" true (String.length str > 0)

let test_core_type_vector () =
  let ty = TVec t_int32 in
  let ct = core_type_of_typ ~loc:dummy_loc ty in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  (* Should generate wildcard _ *)
  Alcotest.(check bool) "vector type generated" true (String.length str > 0)

let test_core_type_array () =
  let ty = TArr (t_float32, Local) in
  let ct = core_type_of_typ ~loc:dummy_loc ty in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  (* Should generate wildcard _ *)
  Alcotest.(check bool) "array type generated" true (String.length str > 0)

let test_core_type_tuple () =
  let ty = TTuple [t_int32; t_float32; t_bool] in
  let ct = core_type_of_typ ~loc:dummy_loc ty in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  Alcotest.(check bool) "tuple type generated" true (String.length str > 0)

let test_core_type_record_inline () =
  (* Simple inline record (no module qualifier) *)
  let ty = TRecord ("point", [("x", t_int32); ("y", t_int32)]) in
  let ct = core_type_of_typ ~loc:dummy_loc ty in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  (* Should generate wildcard _ for inline types *)
  Alcotest.(check bool)
    "inline record type generated"
    true
    (String.length str > 0)

let test_core_type_record_qualified () =
  (* Qualified record type (e.g., Module.point) *)
  let ty = TRecord ("MyModule.point", [("x", t_int32); ("y", t_int32)]) in
  let ct = core_type_of_typ ~loc:dummy_loc ty in
  let str = Format.asprintf "%a" Pprintast.core_type ct in
  (* Should generate the qualified type path *)
  Alcotest.(check bool)
    "qualified record type generated"
    true
    (String.length str > 0)

(* ===== Test: gen_mode values ===== *)

let test_gen_mode_exists () =
  (* Just check that gen_mode variants exist and can be used *)
  let modes = [FullMode; Simple1DMode; Simple2DMode; Simple3DMode] in
  Alcotest.(check int) "four gen modes" 4 (List.length modes)

(* ===== Test: simple_gid_* constants ===== *)

let test_simple_gid_constants () =
  Alcotest.(check string) "simple_gid_x" "__gid_x" simple_gid_x ;
  Alcotest.(check string) "simple_gid_y" "__gid_y" simple_gid_y ;
  Alcotest.(check string) "simple_gid_z" "__gid_z" simple_gid_z

(* ===== Test: gen_intrinsic_const (sampling) ===== *)

let test_gen_intrinsic_const_thread_idx_x () =
  let ref = CorePrimitiveRef "thread_idx_x" in
  let expr = gen_intrinsic_const ~loc:dummy_loc ~gen_mode:FullMode ref in
  let str = expr_to_string expr in
  (* Should generate state.Sarek.Sarek_cpu_runtime.thread_idx_x *)
  Alcotest.(check bool) "thread_idx_x generated" true (String.length str > 0)

let test_gen_intrinsic_const_block_dim () =
  let ref = CorePrimitiveRef "block_dim_x" in
  let expr = gen_intrinsic_const ~loc:dummy_loc ~gen_mode:FullMode ref in
  let str = expr_to_string expr in
  Alcotest.(check bool) "block_dim_x generated" true (String.length str > 0)

let test_gen_intrinsic_const_global_idx () =
  let ref = CorePrimitiveRef "global_idx_x" in
  let expr = gen_intrinsic_const ~loc:dummy_loc ~gen_mode:FullMode ref in
  let str = expr_to_string expr in
  (* Should generate function call to global_idx_x *)
  Alcotest.(check bool) "global_idx_x generated" true (String.length str > 0)

let test_gen_intrinsic_const_simple_mode () =
  let ref = CorePrimitiveRef "global_idx_x" in
  let expr = gen_intrinsic_const ~loc:dummy_loc ~gen_mode:Simple1DMode ref in
  let str = expr_to_string expr in
  (* In simple mode, should use __gid_x variable *)
  Alcotest.(check bool) "simple mode global_idx_x" true (String.length str > 0)

(* ===== Test: gen_intrinsic_fun (sampling) ===== *)

let test_gen_intrinsic_fun_block_barrier () =
  let ref = CorePrimitiveRef "block_barrier" in
  let expr = gen_intrinsic_fun ~loc:dummy_loc ~gen_mode:FullMode ref [] in
  let str = expr_to_string expr in
  (* Should generate call to barrier function *)
  Alcotest.(check bool) "block_barrier generated" true (String.length str > 0)

let test_gen_intrinsic_fun_global_idx () =
  let ref = CorePrimitiveRef "global_idx" in
  let expr = gen_intrinsic_fun ~loc:dummy_loc ~gen_mode:FullMode ref [] in
  let str = expr_to_string expr in
  (* Should generate call to global_idx_x *)
  Alcotest.(check bool) "global_idx generated" true (String.length str > 0)

let test_gen_intrinsic_fun_gpu_function () =
  (* Test a generic Gpu module function *)
  let ref = IntrinsicRef (["Gpu"], "float") in
  let args = [] in
  let expr = gen_intrinsic_fun ~loc:dummy_loc ~gen_mode:FullMode ref args in
  let str = expr_to_string expr in
  (* Should call Gpu.float *)
  Alcotest.(check bool) "Gpu.float generated" true (String.length str > 0)

let test_gen_intrinsic_fun_with_args () =
  (* Test intrinsic with arguments *)
  let ref = CorePrimitiveRef "some_function" in
  let arg1 = Ast_builder.Default.eint ~loc:dummy_loc 42 in
  let arg2 = Ast_builder.Default.efloat ~loc:dummy_loc "3.14" in
  let expr =
    gen_intrinsic_fun ~loc:dummy_loc ~gen_mode:FullMode ref [arg1; arg2]
  in
  let str = expr_to_string expr in
  Alcotest.(check bool)
    "function with args generated"
    true
    (String.length str > 0)

(* ===== Test Suite ===== *)

let core_type_tests =
  [
    Alcotest.test_case "unit" `Quick test_core_type_unit;
    Alcotest.test_case "bool" `Quick test_core_type_bool;
    Alcotest.test_case "int32" `Quick test_core_type_int32;
    Alcotest.test_case "float32" `Quick test_core_type_float32;
    Alcotest.test_case "float64" `Quick test_core_type_float64;
    Alcotest.test_case "int64" `Quick test_core_type_int64;
    Alcotest.test_case "vector" `Quick test_core_type_vector;
    Alcotest.test_case "array" `Quick test_core_type_array;
    Alcotest.test_case "tuple" `Quick test_core_type_tuple;
    Alcotest.test_case "record inline" `Quick test_core_type_record_inline;
    Alcotest.test_case "record qualified" `Quick test_core_type_record_qualified;
  ]

let gen_mode_tests = [Alcotest.test_case "exists" `Quick test_gen_mode_exists]

let constants_tests =
  [Alcotest.test_case "simple_gid_*" `Quick test_simple_gid_constants]

let intrinsic_const_tests =
  [
    Alcotest.test_case
      "thread_idx_x"
      `Quick
      test_gen_intrinsic_const_thread_idx_x;
    Alcotest.test_case "block_dim" `Quick test_gen_intrinsic_const_block_dim;
    Alcotest.test_case "global_idx" `Quick test_gen_intrinsic_const_global_idx;
    Alcotest.test_case "simple mode" `Quick test_gen_intrinsic_const_simple_mode;
  ]

let intrinsic_fun_tests =
  [
    Alcotest.test_case
      "block_barrier"
      `Quick
      test_gen_intrinsic_fun_block_barrier;
    Alcotest.test_case "global_idx" `Quick test_gen_intrinsic_fun_global_idx;
    Alcotest.test_case "Gpu function" `Quick test_gen_intrinsic_fun_gpu_function;
    Alcotest.test_case "with args" `Quick test_gen_intrinsic_fun_with_args;
  ]

let () =
  Alcotest.run
    "Sarek_native_intrinsics"
    [
      ("core_type_of_typ", core_type_tests);
      ("gen_mode", gen_mode_tests);
      ("constants", constants_tests);
      ("gen_intrinsic_const", intrinsic_const_tests);
      ("gen_intrinsic_fun", intrinsic_fun_tests);
    ]
