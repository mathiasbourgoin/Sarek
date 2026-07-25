(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Regression: Runtime's outer kernel memo must not survive [Device.reset].
 *
 * Keying that memo by device id fixes cross-device aliasing only while a given
 * id keeps denoting a given device. It does not: [Device.init] restarts
 * [global_id] at 0 over whichever frameworks it is handed, so a reset followed
 * by an init with a different framework list reassigns the ids. A framework
 * enumerated BEFORE OpenCL shifts every OpenCL device down by its own device
 * count, and an id that named one OpenCL device now names another.
 *
 * The memo key is "<framework>#<id>", so the framework component does not save
 * this case - both spellings are "OpenCL". A surviving entry is served to a
 * physically different GPU with no exception raised: the launch targets the
 * device the cached kernel was built for and the buffer the caller passed is
 * simply never written. That is the same silent-wrong-results class the
 * device-keying fix exists to close, reached through [Device.reset] instead of
 * through two live devices.
 *
 * Only this outer layer is exposed: every per-backend cache keys on its own
 * backend-local device index (e.g. Opencl_plugin_base keys on
 * [device.Opencl_api.Device.id]), which a reset does not perturb. So the fix is
 * a single [Cache_hooks.notify_clear_all] in [Device.reset], and this test goes
 * red when that call is removed.
 *
 * Requires >= 2 OpenCL devices plus at least one other backend that enumerates
 * fewer devices than OpenCL does before it, so that some id actually changes
 * meaning. Skips with a printed reason (exit 0) otherwise - and, importantly,
 * skips only after checking that no id changed meaning, so it cannot pass by
 * silently failing to set up the hazard.
 ******************************************************************************)

open Spoc_core

let source =
  "__kernel void reset_probe(__global float* a) { a[get_global_id(0)] = 7.0f; }"

let kernel_name = "reset_probe"

let n = 16

(* Launch the probe on [d] and return how many elements it failed to write. A
   kernel served from a stale entry runs on the device it was built for, so the
   buffer here stays at its initial 0. *)
let run_and_count_unwritten (d : Device.t) =
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
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if host.{i} <> 7.0 then incr bad
  done ;
  !bad

let snapshot () =
  Array.map (fun (d : Device.t) -> (d.id, d.framework, d.name)) !Device.devices

let () =
  Test_helpers.Benchmarks.init_backends () ;

  (* Init A: OpenCL alone, so its devices occupy ids 0..k-1. *)
  ignore (Device.init ~frameworks:["OpenCL"] ()) ;
  let before = snapshot () in
  let opencl_count = Array.length before in
  if opencl_count < 2 then begin
    Printf.printf
      "SKIP: needs >= 2 OpenCL devices for a reset to be able to move an id \
       between two of them; found %d (the probe kernel is OpenCL C)\n\
       %!"
      opencl_count ;
    exit 0
  end ;

  (* Warm the outer memo for every OpenCL id under init A's numbering. *)
  Array.iter
    (fun (d : Device.t) ->
      let bad = run_and_count_unwritten d in
      if bad > 0 then begin
        Printf.printf
          "FAIL: baseline run on init-A id %d (%s) left %d/%d elements \
           unwritten; the test cannot distinguish a stale hit from a broken \
           device\n\
           %!"
          d.id
          d.name
          bad
          n ;
        exit 1
      end)
    !Device.devices ;
  Printf.printf
    "init A: warmed the memo for %d OpenCL device(s)\n%!"
    opencl_count ;

  (* Init B: put another backend in front so the OpenCL ids shift. *)
  Device.reset () ;
  ignore (Device.init ~frameworks:["Native"; "OpenCL"] ()) ;
  let after = snapshot () in

  (* Find an id that changed meaning: same id, same framework, different
     physical device. That is the collision the memo key cannot see. *)
  let collision =
    Array.to_list after
    |> List.find_opt (fun (id, fw, name) ->
        fw = "OpenCL"
        && Array.exists
             (fun (id', fw', name') -> id' = id && fw' = fw && name' <> name)
             before)
  in
  match collision with
  | None ->
      Printf.printf
        "SKIP: no OpenCL id changed meaning across the reset on this machine, \
         so the hazard is not reachable here (init A ids: %s / init B ids: %s)\n\
         %!"
        (String.concat
           " "
           (List.map
              (fun (id, _, _) -> string_of_int id)
              (Array.to_list before)))
        (String.concat
           " "
           (List.map
              (fun (id, fw, _) -> Printf.sprintf "%d:%s" id fw)
              (Array.to_list after))) ;
      exit 0
  | Some (id, _, name) ->
      Printf.printf
        "id %d denoted a different OpenCL device before the reset; it now \
         denotes %s\n\
         %!"
        id
        name ;
      let d =
        Option.get
          (Array.find_opt (fun (dev : Device.t) -> dev.id = id) !Device.devices)
      in
      let bad = run_and_count_unwritten d in
      if bad > 0 then begin
        Printf.printf
          "FAIL: %d/%d elements were not written - the outer memo served id \
           %d's pre-reset entry, which is bound to a different physical device\n\
           %!"
          bad
          n
          id ;
        exit 1
      end ;
      Printf.printf "OK: all %d elements = 7 after the reset\n%!" n ;
      print_endline "test_runtime_cache_reset_ids: PASSED"
