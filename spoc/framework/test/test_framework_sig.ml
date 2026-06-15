(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Framework_sig
 *
 * Tests dimension helpers and type construction.
 ******************************************************************************)

open Spoc_framework.Framework_sig

(** {1 dims Helper Tests} *)

let test_dims_1d () =
  let d = dims_1d 128 in
  assert (d.x = 128) ;
  assert (d.y = 1) ;
  assert (d.z = 1) ;
  print_endline "  dims_1d: OK"

let test_dims_2d () =
  let d = dims_2d 64 32 in
  assert (d.x = 64) ;
  assert (d.y = 32) ;
  assert (d.z = 1) ;
  print_endline "  dims_2d: OK"

let test_dims_3d () =
  let d = dims_3d 16 8 4 in
  assert (d.x = 16) ;
  assert (d.y = 8) ;
  assert (d.z = 4) ;
  print_endline "  dims_3d: OK"

let test_dims_raw () =
  let d = {x = 256; y = 128; z = 64} in
  assert (d.x = 256) ;
  assert (d.y = 128) ;
  assert (d.z = 64) ;
  print_endline "  dims record: OK"

(** {1 Capabilities Type Tests} *)

let test_capabilities () =
  let caps : capabilities =
    {
      max_threads_per_block = 1024;
      max_block_dims = (1024, 1024, 64);
      max_grid_dims = (65535, 65535, 65535);
      shared_mem_per_block = 49152;
      total_global_mem = Int64.of_int (8 * 1024 * 1024 * 1024);
      compute_capability = (7, 5);
      supports_fp64 = true;
      supports_atomics = true;
      warp_size = 32;
      max_registers_per_block = 65536;
      clock_rate_khz = 1530000;
      multiprocessor_count = 68;
      is_cpu = false;
    }
  in
  assert (caps.max_threads_per_block = 1024) ;
  assert (caps.warp_size = 32) ;
  assert (caps.supports_fp64 = true) ;
  assert (caps.is_cpu = false) ;
  print_endline "  capabilities: OK"

(** {1 Device Type Tests} *)

let test_device () =
  let caps : capabilities =
    {
      max_threads_per_block = 256;
      max_block_dims = (256, 256, 64);
      max_grid_dims = (65535, 65535, 65535);
      shared_mem_per_block = 16384;
      total_global_mem = Int64.of_int (2 * 1024 * 1024 * 1024);
      compute_capability = (0, 0);
      supports_fp64 = true;
      supports_atomics = true;
      warp_size = 64;
      max_registers_per_block = 16384;
      clock_rate_khz = 1000000;
      multiprocessor_count = 8;
      is_cpu = true;
    }
  in
  let dev : device =
    {
      id = 0;
      backend_id = 0;
      name = "Test CPU Device";
      framework = "OpenCL";
      capabilities = caps;
    }
  in
  assert (dev.id = 0) ;
  assert (dev.name = "Test CPU Device") ;
  assert (dev.framework = "OpenCL") ;
  assert (dev.capabilities.is_cpu = true) ;
  print_endline "  device: OK"

(** {1 Execution Model Tests} *)

let test_execution_model () =
  let jit = JIT in
  let direct = Direct in
  let custom = Custom in
  (* Ensure variants are distinct *)
  assert (jit <> direct) ;
  assert (direct <> custom) ;
  assert (jit <> custom) ;
  print_endline "  execution_model: OK"

(** {1 Source Language Tests} *)

let test_source_lang () =
  let langs = [CUDA_Source; OpenCL_Source; PTX; SPIR_V; GLSL_Source] in
  assert (List.length langs = 5) ;
  (* Ensure all are distinct *)
  let distinct = List.sort_uniq compare langs in
  assert (List.length distinct = 5) ;
  print_endline "  source_lang: OK"

(** {1 Convergence Tests} *)

let test_convergence () =
  let uniform = Uniform in
  let divergent = Divergent in
  let sync = Sync in
  assert (uniform <> divergent) ;
  assert (divergent <> sync) ;
  print_endline "  convergence: OK"

(** {1 exec_arg Tests} *)

let test_exec_arg_basic () =
  let i32 = EA_Int32 42l in
  let i64 = EA_Int64 123L in
  let f32 = EA_Float32 3.14 in
  let f64 = EA_Float64 2.718281828 in
  (match i32 with EA_Int32 n -> assert (n = 42l) | _ -> assert false) ;
  (match i64 with EA_Int64 n -> assert (n = 123L) | _ -> assert false) ;
  (match f32 with
  | EA_Float32 f -> assert (abs_float (f -. 3.14) < 0.001)
  | _ -> assert false) ;
  (match f64 with
  | EA_Float64 f -> assert (abs_float (f -. 2.718281828) < 1e-9)
  | _ -> assert false) ;
  print_endline "  exec_arg basic: OK"

(** {1 run_source_arg Tests} *)

let test_run_source_arg () =
  let i32 = RSA_Int32 100l in
  let i64 = RSA_Int64 200L in
  let f32 = RSA_Float32 1.5 in
  let f64 = RSA_Float64 2.5 in
  (match i32 with RSA_Int32 n -> assert (n = 100l) | _ -> assert false) ;
  (match i64 with RSA_Int64 n -> assert (n = 200L) | _ -> assert false) ;
  (match f32 with RSA_Float32 f -> assert (f = 1.5) | _ -> assert false) ;
  (match f64 with RSA_Float64 f -> assert (f = 2.5) | _ -> assert false) ;
  print_endline "  run_source_arg: OK"

(** {1 Main} *)

let () =
  print_endline "Framework_sig tests:" ;
  test_dims_1d () ;
  test_dims_2d () ;
  test_dims_3d () ;
  test_dims_raw () ;
  test_capabilities () ;
  test_device () ;
  test_execution_model () ;
  test_source_lang () ;
  test_convergence () ;
  test_exec_arg_basic () ;
  test_run_source_arg () ;
  print_endline "All Framework_sig tests passed!"
