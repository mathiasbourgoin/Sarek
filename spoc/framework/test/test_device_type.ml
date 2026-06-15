(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Device_type
 *
 * Verifies that Device_type correctly aliases Framework_sig.device.
 ******************************************************************************)

(** {1 Type Alias Tests} *)

let test_device_type_alias () =
  (* Create a device using Device_type.t *)
  let caps : Spoc_framework.Framework_sig.capabilities =
    {
      max_threads_per_block = 512;
      max_block_dims = (512, 512, 64);
      max_grid_dims = (65535, 65535, 65535);
      shared_mem_per_block = 32768;
      total_global_mem = Int64.of_int (4 * 1024 * 1024 * 1024);
      compute_capability = (6, 1);
      supports_fp64 = true;
      supports_atomics = true;
      warp_size = 32;
      max_registers_per_block = 65536;
      clock_rate_khz = 1400000;
      multiprocessor_count = 20;
      is_cpu = false;
    }
  in
  let dev : Spoc_framework.Device_type.t =
    {
      id = 1;
      backend_id = 0;
      name = "GeForce GTX 1080";
      framework = "CUDA";
      capabilities = caps;
    }
  in
  (* Verify Device_type.t and Framework_sig.device are the same *)
  let dev_as_fw : Spoc_framework.Framework_sig.device = dev in
  assert (dev_as_fw.id = dev.id) ;
  assert (dev_as_fw.name = dev.name) ;
  assert (dev_as_fw.framework = dev.framework) ;
  print_endline "  Device_type.t = Framework_sig.device: OK"

let test_device_type_fields () =
  let caps : Spoc_framework.Framework_sig.capabilities =
    {
      max_threads_per_block = 1024;
      max_block_dims = (1024, 1024, 64);
      max_grid_dims = (2147483647, 65535, 65535);
      shared_mem_per_block = 49152;
      total_global_mem = Int64.of_int (11 * 1024 * 1024 * 1024);
      compute_capability = (8, 6);
      supports_fp64 = true;
      supports_atomics = true;
      warp_size = 32;
      max_registers_per_block = 65536;
      clock_rate_khz = 1695000;
      multiprocessor_count = 84;
      is_cpu = false;
    }
  in
  let dev : Spoc_framework.Device_type.t =
    {
      id = 0;
      backend_id = 0;
      name = "NVIDIA RTX 3090";
      framework = "CUDA";
      capabilities = caps;
    }
  in
  assert (dev.id = 0) ;
  assert (dev.backend_id = 0) ;
  assert (dev.name = "NVIDIA RTX 3090") ;
  assert (dev.framework = "CUDA") ;
  assert (dev.capabilities.max_threads_per_block = 1024) ;
  assert (dev.capabilities.warp_size = 32) ;
  assert (dev.capabilities.is_cpu = false) ;
  print_endline "  Device_type fields: OK"

let test_device_type_cpu () =
  let caps : Spoc_framework.Framework_sig.capabilities =
    {
      max_threads_per_block = 1;
      max_block_dims = (1, 1, 1);
      max_grid_dims = (65535, 1, 1);
      shared_mem_per_block = 0;
      total_global_mem = Int64.of_int (32 * 1024 * 1024 * 1024);
      compute_capability = (0, 0);
      supports_fp64 = true;
      supports_atomics = true;
      warp_size = 1;
      max_registers_per_block = 0;
      clock_rate_khz = 3600000;
      multiprocessor_count = 16;
      is_cpu = true;
    }
  in
  let dev : Spoc_framework.Device_type.t =
    {
      id = 2;
      backend_id = 0;
      name = "Intel Core i9-10900K";
      framework = "OpenCL";
      capabilities = caps;
    }
  in
  assert (dev.capabilities.is_cpu = true) ;
  assert (dev.framework = "OpenCL") ;
  print_endline "  Device_type CPU device: OK"

(** {1 Cross-module Compatibility Tests} *)

let test_cross_module_compatibility () =
  (* Test that a function taking Framework_sig.device accepts Device_type.t *)
  let get_device_name (d : Spoc_framework.Framework_sig.device) = d.name in
  let caps : Spoc_framework.Framework_sig.capabilities =
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
      is_cpu = false;
    }
  in
  let dev : Spoc_framework.Device_type.t =
    {
      id = 0;
      backend_id = 0;
      name = "AMD Radeon RX 6800";
      framework = "OpenCL";
      capabilities = caps;
    }
  in
  (* This should compile because Device_type.t = Framework_sig.device *)
  let name = get_device_name dev in
  assert (name = "AMD Radeon RX 6800") ;
  print_endline "  Cross-module compatibility: OK"

(** {1 Main} *)

let () =
  print_endline "Device_type tests:" ;
  test_device_type_alias () ;
  test_device_type_fields () ;
  test_device_type_cpu () ;
  test_cross_module_compatibility () ;
  print_endline "All Device_type tests passed!"
