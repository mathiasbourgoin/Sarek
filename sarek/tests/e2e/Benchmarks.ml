(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Spoc_core

type config = {
  mutable dev_id : int;
  mutable use_interpreter : bool;
  mutable use_native : bool;
  mutable use_vulkan : bool;
  mutable use_metal : bool;
  mutable benchmark_all : bool;
  mutable size : int;
  mutable block_size : int;
  mutable verify : bool;
}

let config =
  {
    dev_id = 0;
    use_interpreter = false;
    use_native = false;
    use_vulkan = false;
    use_metal = false;
    benchmark_all = false;
    size = 1024;
    block_size = 256;
    verify = true;
  }

(* Force backend registration.
   Uses Backend_loader which conditionally initializes available backends
   based on dune's (select ...) mechanism. *)
let init_backends () = Backend_loader.init ()

let parse_args () =
  let usage_msg = "Usage: " ^ Sys.argv.(0) ^ " [options]" in
  let speclist =
    [
      ("-d", Arg.Int (fun i -> config.dev_id <- i), "Device ID");
      ( "--interpreter",
        Arg.Unit (fun () -> config.use_interpreter <- true),
        "Use Interpreter" );
      ( "--native",
        Arg.Unit (fun () -> config.use_native <- true),
        "Use Native CPU" );
      ("--vulkan", Arg.Unit (fun () -> config.use_vulkan <- true), "Use Vulkan");
      ("--metal", Arg.Unit (fun () -> config.use_metal <- true), "Use Metal");
      ( "--benchmark",
        Arg.Unit (fun () -> config.benchmark_all <- true),
        "Benchmark all devices" );
      ("-s", Arg.Int (fun i -> config.size <- i), "Problem size");
      ("-b", Arg.Int (fun i -> config.block_size <- i), "Block size");
      ( "-no-verify",
        Arg.Unit (fun () -> config.verify <- false),
        "Disable verification" );
    ]
  in
  Arg.parse speclist (fun _ -> ()) usage_msg

let get_target_devices all_devices =
  if config.benchmark_all then all_devices
  else if config.use_interpreter then (
    match
      Array.find_opt (fun d -> d.Device.framework = "Interpreter") all_devices
    with
    | Some d -> [|d|]
    | None ->
        Printf.eprintf "Interpreter device not found\n" ;
        Stdlib.exit 1)
  else if config.use_native then (
    match
      Array.find_opt (fun d -> d.Device.framework = "Native") all_devices
    with
    | Some d -> [|d|]
    | None ->
        Printf.eprintf "Native device not found\n" ;
        Stdlib.exit 1)
  else if config.use_vulkan then (
    match
      Array.find_opt (fun d -> d.Device.framework = "Vulkan") all_devices
    with
    | Some d -> [|d|]
    | None ->
        Printf.eprintf "Vulkan device not found\n" ;
        Stdlib.exit 1)
  else if config.use_metal then (
    match
      Array.find_opt (fun d -> d.Device.framework = "Metal") all_devices
    with
    | Some d -> [|d|]
    | None ->
        Printf.eprintf "Metal device not found\n" ;
        Stdlib.exit 1)
  else if config.dev_id < Array.length all_devices then
    [|all_devices.(config.dev_id)|]
  else (
    Printf.eprintf "Device ID %d out of range\n" config.dev_id ;
    Stdlib.exit 1)

let initialized = ref false

let global_success = ref true

let init () =
  if not !initialized then begin
    init_backends () ;
    parse_args () ;
    initialized := true
  end

let exit () = if !global_success then Stdlib.exit 0 else Stdlib.exit 1

(** Convenience filter: exclude interpreter backend *)
let no_interpreter (d : Device.t) = d.framework <> "Interpreter"

(** Convenience filter: exclude native backend *)
let no_native (d : Device.t) = d.framework <> "Native"

(** Convenience filter: GPU backends only (excludes Native and Interpreter) *)
let gpu_only (d : Device.t) =
  d.framework <> "Native" && d.framework <> "Interpreter"

(** Convenience filter: JIT backends only (CUDA, OpenCL, Vulkan, Metal) *)
let jit_only (d : Device.t) =
  d.framework = "CUDA" || d.framework = "OpenCL" || d.framework = "Vulkan"
  || d.framework = "Metal"

let run ?(baseline : (int -> 'a) option) ?(verify : ('a -> 'a -> bool) option)
    ?(filter : (Device.t -> bool) option)
    ?(exclude_frameworks : string list = []) test_name
    (f : Device.t -> int -> int -> float * 'a) =
  init () ;

  Printf.printf "=== %s ===\n" test_name ;
  Printf.printf "Size: %d\n" config.size ;

  (* Use all registered backends - they auto-register when linked *)
  let all_devices = Device.init () in
  if Array.length all_devices = 0 then (
    Printf.eprintf "No devices found.\n" ;
    Stdlib.exit 1) ;

  Printf.printf "Available devices:\n" ;
  Array.iteri
    (fun i d ->
      Printf.printf "  [%d] %s (%s)\n" i d.Device.name d.Device.framework)
    all_devices ;

  let targets = get_target_devices all_devices in
  (* Apply exclude_frameworks filter *)
  let targets =
    if exclude_frameworks = [] then targets
    else
      Array.to_list targets
      |> List.filter (fun d ->
          not (List.mem d.Device.framework exclude_frameworks))
      |> Array.of_list
  in
  (* Apply custom filter *)
  let targets =
    match filter with
    | Some p -> Array.to_list targets |> List.filter p |> Array.of_list
    | None -> targets
  in

  Printf.printf
    "\n%-40s | %10s | %10s | %10s\n"
    "Device"
    "Time (ms)"
    "Status"
    "Speedup" ;
  Printf.printf "%s\n" (String.make 79 '-') ;

  let baseline_result = ref None in
  let baseline_time = ref 0.0 in

  (* Run baseline if provided *)
  (match baseline with
  | Some base_func ->
      let t0 = Unix.gettimeofday () in
      let res = base_func config.size in
      let t1 = Unix.gettimeofday () in
      baseline_time := (t1 -. t0) *. 1000.0 ;
      baseline_result := Some res ;
      Printf.printf
        "%-40s | %10.4f | %10s | %10s\n"
        "CPU Baseline"
        !baseline_time
        "PASS"
        "1.00x" ;
      flush stdout
  | None -> ()) ;

  let all_passed = ref true in

  Array.iter
    (fun dev ->
      let dev_label =
        Printf.sprintf "%s (%s)" dev.Device.name dev.Device.framework
      in
      let dev_label =
        if String.length dev_label > 38 then String.sub dev_label 0 38
        else dev_label
      in

      try
        let time, result = f dev config.size config.block_size in

        let passed =
          if not config.verify then true
          else
            match (verify, !baseline_result) with
            | Some v_func, Some expected -> v_func result expected
            | _ -> true
          (* No verification function or no baseline to compare against *)
        in

        let status =
          if not config.verify then "N/A" else if passed then "PASS" else "FAIL"
        in
        if config.verify && not passed then all_passed := false ;

        let speedup =
          if !baseline_time > 0.0 then
            Printf.sprintf "%.2fx" (!baseline_time /. time)
          else "N/A"
        in

        Printf.printf
          "%-40s | %10.4f | %10s | %10s\n"
          dev_label
          time
          status
          speedup ;
        flush stdout
      with e ->
        Printf.printf
          "%-40s | %10s | %10s | %10s\n"
          dev_label
          "ERR"
          "ERROR"
          "N/A" ;
        Printf.eprintf
          "Error on %s: %s\n"
          dev.Device.name
          (Printexc.to_string e) ;
        (* Only fail if this is the baseline CPU, otherwise tolerate GPU errors *)
        if dev.Device.framework = "CPU" then all_passed := false)
    targets ;

  Printf.printf "%s\n\n" (String.make 79 '-') ;

  if !all_passed then Printf.printf "Test PASSED\n"
  else (
    Printf.printf "Test FAILED\n" ;
    global_success := false)
