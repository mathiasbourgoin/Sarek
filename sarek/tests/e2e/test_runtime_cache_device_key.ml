(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression: Runtime's outer kernel memo must be keyed by DEVICE identity.
 *
 * [Device.t.framework] is the BACKEND NAME, shared by every device of that
 * backend, so a key built from (framework, name, source) aliases all of them.
 * [Kernel.compile_cached] closes the specific backend kernel into the [Kernel.t]
 * it returns, so a hit on that key hands device B a kernel compiled for device
 * A. No exception is raised: the launch targets the wrong device and the caller
 * silently reads back whatever was (not) written.
 *
 * Two checks, both of which fail without a device-keyed cache:
 *   1. structural - the kernel handed back for device 1 is bound to device 1;
 *   2. end-to-end  - a kernel that writes 7.0 into a buffer produces 7.0 on the
 *      second device, both cold (control) and after another device of the same
 *      backend has already compiled the same source (mutation).
 *
 * Requires >= 2 devices of ONE backend. The probe kernel is written in OpenCL C,
 * so the gate is the OpenCL backend specifically; the test skips (exit 0) with a
 * printed reason when fewer than two OpenCL devices are enumerated. A CUDA/HIP
 * variant only needs the corresponding source spelling added here.
 ******************************************************************************)

open Spoc_core

let source =
  "__kernel void cache_key_probe(__global float* a) { a[get_global_id(0)] = \
   7.0f; }"

let kernel_name = "cache_key_probe"

let n = 16

let failures = ref 0

let failf fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      Printf.printf "FAIL: %s\n%!" s)
    fmt

(* Launch the probe on [d] and read the buffer back. Returns every element, so a
   partial write is visible rather than being masked by checking only [0]. *)
let run_and_read (d : Device.t) =
  let buf = Runtime.alloc_float32 d n in
  let host = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  Bigarray.Array1.fill host 0.0 ;
  Memory.host_to_device ~src:host ~dst:buf ;
  Runtime.run
    d
    ~name:kernel_name
    ~source
    ~args:[Runtime.ArgBuffer buf]
    ~grid:(Runtime.dims1d n)
    ~block:(Runtime.dims1d 1)
    () ;
  Device.synchronize d ;
  Memory.device_to_host ~src:buf ~dst:host ;
  host

let check_all_sevens label (host : (float, _, _) Bigarray.Array1.t) =
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if host.{i} <> 7.0 then incr bad
  done ;
  if !bad > 0 then
    failf
      "%s: %d/%d elements were not written by the kernel (got %g %g %g %g ...) \
       - the launch went to the wrong device"
      label
      !bad
      n
      host.{0}
      host.{1}
      host.{2}
      host.{3}
  else Printf.printf "OK: %s -> all %d elements = 7\n%!" label n

let () =
  Test_helpers.Benchmarks.init_backends () ;
  let devs = Device.init () in
  let opencl =
    Array.of_list
      (List.filter
         (fun (d : Device.t) -> d.framework = "OpenCL")
         (Array.to_list devs))
  in
  if Array.length opencl < 2 then begin
    Printf.printf
      "SKIP: needs >= 2 devices of one backend to detect cross-device cache \
       aliasing; found %d OpenCL device(s) (the probe kernel is OpenCL C)\n\
       %!"
      (Array.length opencl) ;
    exit 0
  end ;
  let d0 = opencl.(0) and d1 = opencl.(1) in
  Printf.printf
    "using OpenCL devices [%d] %s and [%d] %s (same framework %S)\n%!"
    d0.id
    d0.name
    d1.id
    d1.name
    d0.framework ;

  (* 1. Structural: compile for d0 first, then d1. Without a device-keyed cache
     the second call is a cache hit and returns d0's kernel. *)
  let k0 = Runtime.compile_kernel d0 ~name:kernel_name ~source in
  let k1 = Runtime.compile_kernel d1 ~name:kernel_name ~source in
  if (Kernel.device k0).Device.id <> d0.id then
    failf
      "requested device %d, got a kernel bound to device %d"
      d0.id
      (Kernel.device k0).Device.id ;
  if (Kernel.device k1).Device.id <> d1.id then
    failf
      "requested device %d, got a kernel bound to device %d (outer cache key \
       omits device identity)"
      d1.id
      (Kernel.device k1).Device.id
  else Printf.printf "OK: per-device kernels kept distinct\n%!" ;

  (* 2a. Control: cold cache, only the second device is ever touched. *)
  Runtime.clear_cache () ;
  check_all_sevens
    (Printf.sprintf "control (cold cache, device %d only)" d1.id)
    (run_and_read d1) ;

  (* 2b. Mutation: same run, but another device of the same backend compiles the
     same source first. Identical device, identical kernel - only the cache
     state differs, so any difference from 2a is the aliasing bug. *)
  Runtime.clear_cache () ;
  ignore (run_and_read d0) ;
  check_all_sevens
    (Printf.sprintf
       "after device %d warmed the cache with the same source"
       d0.id)
    (run_and_read d1) ;

  if !failures > 0 then begin
    Printf.printf "%d check(s) failed\n%!" !failures ;
    exit 1
  end ;
  print_endline "test_runtime_cache_device_key: PASSED"
