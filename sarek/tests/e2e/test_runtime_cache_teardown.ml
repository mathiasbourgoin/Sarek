(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression: Runtime's outer kernel memo must be invalidated by backend
 * teardown.
 *
 * [Kernel.clear_cache device] reaches the backend's own guarded cache, whose
 * [destroy] RELEASES the handles ([clReleaseKernel] / [clReleaseProgram],
 * [cuModuleUnload], ...). [Runtime.kernel_cache] sits above it and holds
 * [Kernel.t] closures over exactly those handles. If it is not dropped too, the
 * next [Runtime.run] hits the outer memo and launches through a released
 * driver object.
 *
 * Without the fix this test does not fail an assertion - it SIGSEGVs (observed
 * rc=139). That is why it is its own executable: the crash is reported by dune
 * as a nonzero exit of this one test instead of taking a shared test binary
 * (and every other case in it) down.
 *
 * Requires the OpenCL backend (the probe kernel is OpenCL C) and at least one
 * OpenCL device; reports an Alcotest [SKIP] otherwise. Not a printed "SKIP"
 * from a plain executable exiting 0, which is what this used to be and which is
 * indistinguishable from a pass on a machine with no OpenCL device (i.e. on
 * CI). Native/Interpreter are deliberately not accepted: they hold no
 * releasable driver handle, so the test would pass vacuously.
 ******************************************************************************)

open Spoc_core

let source =
  "__kernel void teardown_probe(__global float* a) { a[get_global_id(0)] = \
   7.0f; }"

let kernel_name = "teardown_probe"

let n = 16

let run_and_read (d : Device.t) label =
  Printf.printf "  [%s] launching\n%!" label ;
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
  Printf.printf "  [%s] readback[0] = %g\n%!" label host.{0} ;
  host.{0}

let test_outer_memo_dropped_by_backend_teardown () =
  Test_helpers.Benchmarks.init_backends () ;
  let devs = Device.init () in
  match Array.find_opt (fun (d : Device.t) -> d.framework = "OpenCL") devs with
  | None ->
      Printf.printf
        "[SKIP] needs an OpenCL device (the probe kernel is OpenCL C and the \
         hazard is a released driver handle, which CPU backends do not have)\n\
         %!" ;
      Alcotest.skip ()
  | Some d ->
      Printf.printf "using OpenCL device [%d] %s\n%!" d.id d.name ;
      let before = run_and_read d "before clear_cache" in
      if before <> 7.0 then
        Alcotest.failf
          "baseline launch produced %g, expected 7 - the device is not usable, \
           so nothing below is meaningful"
          before ;
      (* Structural check FIRST, before anything is launched. Without it this
         test's only signal is the SIGSEGV below, which names nothing: dune
         reports a nonzero exit and the reader has to work out why. Physical
         identity of what [Runtime.compile_kernel] returns is the memo's own
         state, so a stale entry is reported here as a message instead of as a
         crash. The launch that follows still has to happen: this identity check
         cannot tell whether the RELEASE actually occurred, only whether the
         memo was dropped. *)
      let memoized = Runtime.compile_kernel d ~name:kernel_name ~source in
      if not (Runtime.compile_kernel d ~name:kernel_name ~source == memoized)
      then
        Alcotest.failf
          "the outer memo is not hitting at all before the clear, so this test \
           cannot tell an invalidated memo from a memo that never held \
           anything" ;
      (* Releases the backend handles the outer memo closed over. *)
      Printf.printf "calling Kernel.clear_cache\n%!" ;
      Kernel.clear_cache d ;
      Printf.printf "clear_cache returned\n%!" ;
      if Runtime.compile_kernel d ~name:kernel_name ~source == memoized then
        Alcotest.failf
          "Kernel.clear_cache released the backend handles but the outer memo \
           still holds the Kernel.t that closes over them: the next launch \
           goes through a released cl_kernel. Cache_hooks.notify_clear_all did \
           not reach Runtime's on_clear_all listener" ;
      (* Must recompile, not serve the stale closure. SIGSEGV here before the
         fix. *)
      let after = run_and_read d "after clear_cache" in
      if after <> 7.0 then
        Alcotest.failf
          "post-clear launch produced %g, expected 7: the outer memo served a \
           Kernel.t whose backend handle Kernel.clear_cache had already \
           released"
          after ;
      (* And a second clear/run cycle, to confirm the invalidation is not a
         one-shot. *)
      Kernel.clear_cache d ;
      let again = run_and_read d "after a second clear_cache" in
      if again <> 7.0 then
        Alcotest.failf
          "second post-clear launch produced %g, expected 7: the invalidation \
           is one-shot"
          again

let () =
  Alcotest.run
    "Runtime cache teardown"
    [
      ( "outer memo",
        [
          Alcotest.test_case
            "is dropped by Kernel.clear_cache, repeatably"
            `Quick
            test_outer_memo_dropped_by_backend_teardown;
        ] );
    ]
