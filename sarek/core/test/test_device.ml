(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Device module
 *
 * Tests device type, initialization, queries, and predicates.
 * Note: Most operations require backends, so we test structure, pure
 * functions, and error handling paths.
 ******************************************************************************)

(** {1 Helper to create fake capabilities} *)

let make_fake_caps ?(is_cpu = false) ?(supports_fp64 = true)
    ?(supports_atomics = true) ?(warp_size = 32) ?(compute_capability = (0, 0))
    () : Spoc_framework.Framework_sig.capabilities =
  {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability;
    supports_fp64;
    supports_atomics;
    warp_size;
    max_registers_per_block = 16384;
    clock_rate_khz = 1000000;
    multiprocessor_count = 4;
    is_cpu;
  }

let make_fake_device ?(id = 0) ?(backend_id = 0) ?(name = "Fake Device")
    ?(framework = "Test") ?caps () : Spoc_core.Device.t =
  let capabilities =
    match caps with Some c -> c | None -> make_fake_caps ()
  in
  {id; backend_id; name; framework; capabilities}

(** {1 Device Type Tests} *)

let test_device_type_structure () =
  let dev = make_fake_device () in
  assert (dev.id = 0) ;
  assert (dev.backend_id = 0) ;
  assert (dev.name = "Fake Device") ;
  assert (dev.framework = "Test") ;
  print_endline "  device type structure: OK"

let test_capabilities_structure () =
  let caps = make_fake_caps ~supports_fp64:true ~warp_size:64 () in
  assert (caps.max_threads_per_block = 256) ;
  assert (caps.supports_fp64 = true) ;
  assert (caps.warp_size = 64) ;
  print_endline "  capabilities structure: OK"

(** {1 Type Predicate Tests} *)

let test_is_cuda () =
  let cuda_dev = make_fake_device ~framework:"CUDA" () in
  let opencl_dev = make_fake_device ~framework:"OpenCL" () in
  assert (Spoc_core.Device.is_cuda cuda_dev = true) ;
  assert (Spoc_core.Device.is_cuda opencl_dev = false) ;
  print_endline "  is_cuda: OK"

let test_is_opencl () =
  let cuda_dev = make_fake_device ~framework:"CUDA" () in
  let opencl_dev = make_fake_device ~framework:"OpenCL" () in
  assert (Spoc_core.Device.is_opencl opencl_dev = true) ;
  assert (Spoc_core.Device.is_opencl cuda_dev = false) ;
  print_endline "  is_opencl: OK"

let test_is_native () =
  let native_dev = make_fake_device ~framework:"Native" () in
  let cuda_dev = make_fake_device ~framework:"CUDA" () in
  assert (Spoc_core.Device.is_native native_dev = true) ;
  assert (Spoc_core.Device.is_native cuda_dev = false) ;
  print_endline "  is_native: OK"

let test_is_gpu () =
  let cuda_dev = make_fake_device ~framework:"CUDA" () in
  let native_dev = make_fake_device ~framework:"Native" () in
  let opencl_cpu =
    make_fake_device
      ~framework:"OpenCL"
      ~caps:(make_fake_caps ~is_cpu:true ())
      ()
  in
  assert (Spoc_core.Device.is_gpu cuda_dev = true) ;
  assert (Spoc_core.Device.is_gpu native_dev = false) ;
  assert (Spoc_core.Device.is_gpu opencl_cpu = false) ;
  print_endline "  is_gpu: OK"

let test_is_cpu () =
  let native_dev = make_fake_device ~framework:"Native" () in
  let cuda_dev = make_fake_device ~framework:"CUDA" () in
  let cpu_opencl_dev =
    make_fake_device
      ~framework:"OpenCL"
      ~caps:(make_fake_caps ~is_cpu:true ())
      ()
  in
  assert (Spoc_core.Device.is_cpu native_dev = true) ;
  assert (Spoc_core.Device.is_cpu cuda_dev = false) ;
  assert (Spoc_core.Device.is_cpu cpu_opencl_dev = true) ;
  print_endline "  is_cpu: OK"

(** {1 Capability Query Tests} *)

let test_allows_fp64 () =
  let fp64_caps = make_fake_caps ~supports_fp64:true () in
  let no_fp64_caps = make_fake_caps ~supports_fp64:false () in
  let fp64_dev = make_fake_device ~caps:fp64_caps () in
  let no_fp64_dev = make_fake_device ~caps:no_fp64_caps () in
  assert (Spoc_core.Device.allows_fp64 fp64_dev = true) ;
  assert (Spoc_core.Device.allows_fp64 no_fp64_dev = false) ;
  print_endline "  allows_fp64: OK"

let test_supports_atomics () =
  let atomics_caps = make_fake_caps ~supports_atomics:true () in
  let no_atomics_caps = make_fake_caps ~supports_atomics:false () in
  let atomics_dev = make_fake_device ~caps:atomics_caps () in
  let no_atomics_dev = make_fake_device ~caps:no_atomics_caps () in
  assert (Spoc_core.Device.supports_atomics atomics_dev = true) ;
  assert (Spoc_core.Device.supports_atomics no_atomics_dev = false) ;
  print_endline "  supports_atomics: OK"

let test_compute_capability () =
  let sm75_caps = make_fake_caps ~compute_capability:(7, 5) () in
  let sm75_dev = make_fake_device ~caps:sm75_caps () in
  let major, minor = Spoc_core.Device.compute_capability sm75_dev in
  assert (major = 7) ;
  assert (minor = 5) ;
  print_endline "  compute_capability: OK"

let test_warp_size () =
  let warp32_caps = make_fake_caps ~warp_size:32 () in
  let warp64_caps = make_fake_caps ~warp_size:64 () in
  let warp32_dev = make_fake_device ~caps:warp32_caps () in
  let warp64_dev = make_fake_device ~caps:warp64_caps () in
  assert (Spoc_core.Device.warp_size warp32_dev = 32) ;
  assert (Spoc_core.Device.warp_size warp64_dev = 64) ;
  print_endline "  warp_size: OK"

let test_max_threads_per_block () =
  let dev = make_fake_device () in
  assert (Spoc_core.Device.max_threads_per_block dev = 256) ;
  print_endline "  max_threads_per_block: OK"

let test_total_memory () =
  let dev = make_fake_device () in
  assert (Spoc_core.Device.total_memory dev = 1073741824L) ;
  print_endline "  total_memory: OK"

let test_multiprocessor_count () =
  let dev = make_fake_device () in
  assert (Spoc_core.Device.multiprocessor_count dev = 4) ;
  print_endline "  multiprocessor_count: OK"

(** {1 Finder Tests} *)

let test_find_cuda () =
  let devices =
    [|
      make_fake_device ~id:0 ~framework:"OpenCL" ();
      make_fake_device ~id:1 ~framework:"CUDA" ();
      make_fake_device ~id:2 ~framework:"Native" ();
    |]
  in
  let found = Spoc_core.Device.find_cuda devices in
  assert (Option.is_some found) ;
  assert ((Option.get found).id = 1) ;
  print_endline "  find_cuda: OK"

let test_find_opencl () =
  let devices =
    [|
      make_fake_device ~id:0 ~framework:"CUDA" ();
      make_fake_device ~id:1 ~framework:"OpenCL" ();
    |]
  in
  let found = Spoc_core.Device.find_opencl devices in
  assert (Option.is_some found) ;
  assert ((Option.get found).id = 1) ;
  print_endline "  find_opencl: OK"

let test_find_native () =
  let devices =
    [|
      make_fake_device ~id:0 ~framework:"CUDA" ();
      make_fake_device ~id:1 ~framework:"Native" ();
    |]
  in
  let found = Spoc_core.Device.find_native devices in
  assert (Option.is_some found) ;
  assert ((Option.get found).id = 1) ;
  print_endline "  find_native: OK"

let test_find_by_name () =
  let devices =
    [|
      make_fake_device ~id:0 ~name:"GeForce GTX 1080" ();
      make_fake_device ~id:1 ~name:"Tesla V100" ();
    |]
  in
  let found = Spoc_core.Device.find_by_name devices "Tesla V100" in
  assert (Option.is_some found) ;
  assert ((Option.get found).id = 1) ;
  let not_found = Spoc_core.Device.find_by_name devices "RTX 3090" in
  assert (Option.is_none not_found) ;
  print_endline "  find_by_name: OK"

let test_find_by_id () =
  let devices =
    [|
      make_fake_device ~id:0 ();
      make_fake_device ~id:1 ();
      make_fake_device ~id:2 ();
    |]
  in
  let found = Spoc_core.Device.find_by_id devices 1 in
  assert (Option.is_some found) ;
  assert ((Option.get found).id = 1) ;
  let not_found = Spoc_core.Device.find_by_id devices 99 in
  assert (Option.is_none not_found) ;
  print_endline "  find_by_id: OK"

(** {1 to_string Tests} *)

let test_to_string () =
  let caps = make_fake_caps ~compute_capability:(7, 5) () in
  let dev =
    make_fake_device ~id:0 ~name:"Tesla V100" ~framework:"CUDA" ~caps ()
  in
  let s = Spoc_core.Device.to_string dev in
  assert (String.length s > 0) ;
  assert (String.sub s 0 3 = "[0]") ;
  print_endline "  to_string: OK"

(** {1 Reset Tests} *)

let test_reset () =
  Spoc_core.Device.reset () ;
  (* After reset, count should be 0 until reinit *)
  (* Note: This will trigger init again if we call count(), so we just verify
     reset doesn't crash *)
  print_endline "  reset: OK"

(** {1 API Signature Tests} *)

let test_init_signature () =
  let _f : ?frameworks:string list -> unit -> Spoc_core.Device.t array =
    Spoc_core.Device.init
  in
  print_endline "  init signature: OK"

let test_all_signature () =
  let _f : unit -> Spoc_core.Device.t array = Spoc_core.Device.all in
  print_endline "  all signature: OK"

let test_count_signature () =
  let _f : unit -> int = Spoc_core.Device.count in
  print_endline "  count signature: OK"

let test_get_signature () =
  let _f : int -> Spoc_core.Device.t option = Spoc_core.Device.get in
  print_endline "  get signature: OK"

let test_first_signature () =
  let _f : unit -> Spoc_core.Device.t option = Spoc_core.Device.first in
  print_endline "  first signature: OK"

let test_best_signature () =
  let _f : unit -> Spoc_core.Device.t option = Spoc_core.Device.best in
  print_endline "  best signature: OK"

let test_by_framework_signature () =
  let _f : string -> Spoc_core.Device.t array = Spoc_core.Device.by_framework in
  print_endline "  by_framework signature: OK"

let test_synchronize_signature () =
  let _f : Spoc_core.Device.t -> unit = Spoc_core.Device.synchronize in
  print_endline "  synchronize signature: OK"

let test_set_current_signature () =
  let _f : Spoc_core.Device.t -> unit = Spoc_core.Device.set_current in
  print_endline "  set_current signature: OK"

(** {1 Main} *)

let () =
  print_endline "Device module tests:" ;
  (* Type tests *)
  test_device_type_structure () ;
  test_capabilities_structure () ;
  (* Predicate tests *)
  test_is_cuda () ;
  test_is_opencl () ;
  test_is_native () ;
  test_is_gpu () ;
  test_is_cpu () ;
  (* Capability query tests *)
  test_allows_fp64 () ;
  test_supports_atomics () ;
  test_compute_capability () ;
  test_warp_size () ;
  test_max_threads_per_block () ;
  test_total_memory () ;
  test_multiprocessor_count () ;
  (* Finder tests *)
  test_find_cuda () ;
  test_find_opencl () ;
  test_find_native () ;
  test_find_by_name () ;
  test_find_by_id () ;
  (* Utility tests *)
  test_to_string () ;
  test_reset () ;
  (* Signature tests *)
  test_init_signature () ;
  test_all_signature () ;
  test_count_signature () ;
  test_get_signature () ;
  test_first_signature () ;
  test_best_signature () ;
  test_by_framework_signature () ;
  test_synchronize_signature () ;
  test_set_current_signature () ;
  print_endline "All Device module tests passed!"
