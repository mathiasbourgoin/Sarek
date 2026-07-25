(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression: the two holes #86 left open behind [Sarek.Kernel.clear_cache].
 *
 * H1 - a DIRECT backend clear used to bypass the notification. Only
 *      [Sarek.Kernel.clear_cache] fired [Cache_hooks.notify_clear_all]; the
 *      backend's own [Kernel.clear_cache] was a bare [Guarded_cache.clear], so
 *      a caller resolving the backend through [Framework_registry] and calling
 *      it released clReleaseKernel/clReleaseProgram with the outer memo still
 *      holding [Kernel.t] closures over those handles - the exact use-after-free
 *      #86 fixed, reached through a sibling entry point. The notification now
 *      lives in [Cache_hooks.around_clear], which wraps every backend's
 *      [clear_cache] body, so both entry points are covered.
 *
 * H2 - the eviction SCOPE was mixed. Every backend keys its compiled artifact
 *      by the backend-local device index, but only CUDA and HIP passed that
 *      index as [~device_id] to [Guarded_cache.find_or_build]. Without it the
 *      entry is not grouped by device, so [Guarded_cache.evict_device] - and
 *      therefore the whole [Cache_hooks.on_device_destroy] path - could not
 *      reach an OpenCL, Metal or Vulkan entry at all. All backends now pass it
 *      and register the same listener.
 *
 * Both are checked against the BACKEND cache directly (physical identity of the
 * value [compile_cached] returns), not against the outer memo: the outer memo
 * is what the other three tests in this directory cover, and using it here
 * would not distinguish "the backend cache was evicted" from "the memo was".
 *
 * OpenCL only. The Vulkan half of H2 is the same check against the Vulkan
 * backend and lives in test_vulkan_cache_device_scope.ml, in its OWN process:
 * on this host (Mesa 25.x, radeonsi OpenCL + RADV Vulkan) any Vulkan pipeline
 * creation in a process that has already enumerated OpenCL segfaults inside
 * libvulkan_radeon. That is a pre-existing host/driver conflict, not a SPOC
 * bug - test_static_tag_erasure, untouched here, SIGSEGVs the same way on
 * origin/main and passes with OCL_ICD_VENDORS pointed at an empty directory -
 * but it means the two backends must not share a test binary.
 *
 * Requires a real device of the backend under test. Each case skips - as a
 * distinct Alcotest [SKIP] row, not a green [OK] - when that backend enumerates
 * nothing here. A backend that IS present and fails to compile the probe is a
 * failure, never a skip.
 ******************************************************************************)

module Device = Spoc_core.Device

let opencl_source =
  "__kernel void scope_probe(__global float* a) { a[get_global_id(0)] = 7.0f; }"

let n = 16

(* Resolve a backend that has at least one enumerated device, or [None]. Goes
   through [Framework_registry] on purpose: that is precisely the "sibling entry
   point" H1 is about. *)
let backend_with_device family =
  match Spoc_framework_registry.Framework_registry.find_backend family with
  | None -> None
  | Some (module B : Spoc_framework.Framework_sig.BACKEND) ->
      if not (B.is_available ()) then None
      else begin
        B.Device.init () ;
        if B.Device.count () = 0 then None
        else Some (module B : Spoc_framework.Framework_sig.BACKEND)
      end

(* H2, per backend. Two [compile_cached] calls with the same key must be the
   same physical value (the cache is hitting - otherwise the rest proves
   nothing), a device-destroy notification for ANOTHER backend family must
   leave it alone, and one for this backend's own family must evict it. *)
let check_per_device_eviction ~family ~source () =
  match backend_with_device family with
  | None ->
      Printf.printf
        "[SKIP] no %s device enumerated here, so the backend kernel cache \
         cannot be exercised\n\
         %!"
        family ;
      Alcotest.skip ()
  | Some (module B : Spoc_framework.Framework_sig.BACKEND) ->
      let dev = B.Device.get 0 in
      let index = B.Device.id dev in
      Printf.printf
        "%s: using backend-local device %d (%s)\n%!"
        family
        index
        (B.Device.name dev) ;
      let k1 = B.Kernel.compile_cached dev ~name:"scope_probe" ~source in
      let k2 = B.Kernel.compile_cached dev ~name:"scope_probe" ~source in
      if not (k1 == k2) then
        Alcotest.failf
          "%s: two compile_cached calls with the same key returned different \
           values - the backend cache is not hitting at all, so this test \
           cannot tell eviction from a permanent miss"
          family ;
      (* A foreign backend's teardown must not evict: backend-local indices
         collide across backends (OpenCL 0, Vulkan 0, HIP 0 all exist), and
         evict_device does not merely drop memoization, it aborts in-flight
         builds for that index. *)
      let foreign = if String.equal family "HIP" then "OpenCL" else "HIP" in
      Spoc_framework.Cache_hooks.notify_device_destroy ~backend:foreign index ;
      let k3 = B.Kernel.compile_cached dev ~name:"scope_probe" ~source in
      if not (k3 == k1) then
        Alcotest.failf
          "%s: a %s device-%d teardown evicted the %s entry for the same index \
           - the listener matches on the backend-local index alone"
          family
          foreign
          index
          family ;
      Spoc_framework.Cache_hooks.notify_device_destroy ~backend:family index ;
      let k4 = B.Kernel.compile_cached dev ~name:"scope_probe" ~source in
      if k4 == k1 then
        Alcotest.failf
          "%s: the device-%d teardown did NOT evict the backend cache entry. \
           Guarded_cache.evict_device can only reach entries that were \
           installed with ~device_id, and %s's find_or_build does not pass it \
           - so this backend's kernel cache is global where CUDA/HIP are \
           per-device, and a destroyed device's handles survive it"
          family
          index
          family ;
      Printf.printf
        "%s: device-%d teardown evicted the backend cache entry, a foreign \
         backend's did not\n\
         %!"
        family
        index

(* H1. Drive a launch through the PUBLIC runtime API so the outer memo holds a
   [Kernel.t] closing over the backend handles, then release those handles by
   the direct backend entry point - NOT [Sarek.Kernel.clear_cache]. If the
   notification does not reach the outer memo, the next run launches through a
   released cl_kernel: observed as SIGSEGV before the fix, which dune reports as
   a nonzero exit of this executable. *)
let run_and_read (d : Device.t) =
  let buf = Spoc_core.Runtime.alloc_float32 d n in
  let host = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  Bigarray.Array1.fill host 0.0 ;
  Spoc_core.Memory.host_to_device ~src:host ~dst:buf ;
  Spoc_core.Runtime.run
    d
    ~name:"scope_probe"
    ~source:opencl_source
    ~args:[Spoc_core.Runtime.ArgBuffer buf]
    ~grid:(Spoc_core.Runtime.dims1d n)
    ~block:(Spoc_core.Runtime.dims1d 1)
    () ;
  Device.synchronize d ;
  Spoc_core.Memory.device_to_host ~src:buf ~dst:host ;
  host.{0}

let test_direct_backend_clear_invalidates_outer_memo () =
  Test_helpers.Benchmarks.init_backends () ;
  let devs = Device.init () in
  match Array.find_opt (fun (d : Device.t) -> d.framework = "OpenCL") devs with
  | None ->
      Printf.printf
        "[SKIP] needs an OpenCL device: the hazard is a released driver \
         handle, which the CPU backends do not have, and the probe is OpenCL C\n\
         %!" ;
      Alcotest.skip ()
  | Some d -> (
      Printf.printf "using OpenCL device [%d] %s\n%!" d.id d.name ;
      let before = run_and_read d in
      if before <> 7.0 then
        Alcotest.failf
          "baseline launch produced %g, expected 7 - the device is not usable, \
           so nothing below is meaningful"
          before ;
      match
        Spoc_framework_registry.Framework_registry.find_backend d.framework
      with
      | None -> Alcotest.failf "framework %S did not resolve" d.framework
      | Some (module B : Spoc_framework.Framework_sig.BACKEND) ->
          (* The bypass: releases every cl_kernel/cl_program the outer memo's
             Kernel.t closures hold, without going through
             Sarek.Kernel.clear_cache. *)
          (* Structural check first, so the failure NAMES the problem instead
             of only crashing: physical identity of what compile_kernel returns
             is the outer memo's own state. The launch afterwards is still
             required - identity cannot tell whether the handles were really
             released, only whether the memo was dropped. *)
          let memoized =
            Spoc_core.Runtime.compile_kernel
              d
              ~name:"scope_probe"
              ~source:opencl_source
          in
          if
            not
              (Spoc_core.Runtime.compile_kernel
                 d
                 ~name:"scope_probe"
                 ~source:opencl_source
              == memoized)
          then
            Alcotest.failf
              "the outer memo is not hitting before the clear, so this test \
               cannot tell an invalidated memo from one that never held \
               anything" ;
          Printf.printf "calling B.Kernel.clear_cache () directly\n%!" ;
          B.Kernel.clear_cache () ;
          if
            Spoc_core.Runtime.compile_kernel
              d
              ~name:"scope_probe"
              ~source:opencl_source
            == memoized
          then
            Alcotest.failf
              "a direct B.Kernel.clear_cache () released the backend handles \
               without invalidating the outer memo - the notification is in \
               the Sarek.Kernel.clear_cache wrapper only, so this sibling \
               entry point bypasses it and the next launch uses a released \
               cl_kernel" ;
          let after = run_and_read d in
          if after <> 7.0 then
            Alcotest.failf
              "post-clear launch produced %g, expected 7: the outer memo \
               served a Kernel.t whose backend handle a direct \
               B.Kernel.clear_cache () had already released"
              after ;
          Printf.printf
            "direct backend clear invalidated the outer memo (readback 7)\n%!")

let () =
  Alcotest.run
    "Backend cache scope"
    [
      ( "direct backend clear",
        [
          Alcotest.test_case
            "B.Kernel.clear_cache () invalidates the outer memo"
            `Quick
            test_direct_backend_clear_invalidates_outer_memo;
        ] );
      ( "per-device eviction",
        [
          Alcotest.test_case
            "OpenCL kernel cache is evictable per device"
            `Quick
            (check_per_device_eviction ~family:"OpenCL" ~source:opencl_source);
        ] );
    ]
