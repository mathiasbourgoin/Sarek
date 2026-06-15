(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Test helpers for Sarek E2E tests
 *
 * Shared utilities for device selection, verification, and benchmarking.
 * runtime-only version using Spoc_core.
 ******************************************************************************)

open Spoc_core

(** Command line options *)
type config = {
  mutable dev_id : int;
  mutable use_interpreter : bool;
  mutable use_native : bool;
  mutable use_vulkan : bool;
  mutable use_metal : bool;
  mutable benchmark_all : bool;
  mutable benchmark_devices : int list option;
      (** None = all, Some ids = specific *)
  mutable verify : bool;
  mutable size : int;
  mutable block_size : int;
}

let default_config () =
  {
    dev_id = -1;
    (* -1 means run on all devices *)
    use_interpreter = false;
    use_native = false;
    use_vulkan = false;
    use_metal = false;
    benchmark_all = false;
    benchmark_devices = None;
    verify = true;
    size = 1024;
    block_size = 256;
  }

let usage name extra_opts =
  Printf.printf "Usage: %s [options]\n" name ;
  Printf.printf "Options:\n" ;
  Printf.printf "  -d <id>       Device ID (default: all GPU devices)\n" ;
  Printf.printf "  --interpreter Use CPU interpreter device\n" ;
  Printf.printf "  --native      Use native CPU runtime device\n" ;
  Printf.printf "  --vulkan      Use Vulkan device\n" ;
  Printf.printf "  --metal       Use Metal device\n" ;
  Printf.printf "  --benchmark, --benchmark-all  Run on all devices\n" ;
  Printf.printf "  --benchmark-devices <0,1,4>  Run on specific devices\n" ;
  Printf.printf "  -s, --size <size>  Problem size (default: 1024)\n" ;
  Printf.printf "  -b <size>     Block/work-group size (default: 256)\n" ;
  Printf.printf "  -no-verify    Skip result verification\n" ;
  Printf.printf "  -h            Show this help\n" ;
  extra_opts () ;
  exit 0

let parse_args ?(extra = fun _ _ -> false) ?(extra_usage = fun () -> ()) name =
  let cfg = default_config () in
  let i = ref 1 in
  while !i < Array.length Sys.argv do
    let consumed = extra cfg !i in
    (if not consumed then
       match Sys.argv.(!i) with
       | "-d" ->
           incr i ;
           cfg.dev_id <- int_of_string Sys.argv.(!i)
       | "--interpreter" -> cfg.use_interpreter <- true
       | "--native" -> cfg.use_native <- true
       | "--vulkan" -> cfg.use_vulkan <- true
       | "--metal" -> cfg.use_metal <- true
       | "--benchmark" | "--benchmark-all" -> cfg.benchmark_all <- true
       | "--benchmark-devices" ->
           incr i ;
           let ids =
             String.split_on_char ',' Sys.argv.(!i)
             |> List.map String.trim |> List.map int_of_string
           in
           cfg.benchmark_all <- true ;
           cfg.benchmark_devices <- Some ids
       | "-s" | "--size" ->
           incr i ;
           cfg.size <- int_of_string Sys.argv.(!i)
       | "-b" ->
           incr i ;
           cfg.block_size <- int_of_string Sys.argv.(!i)
       | "-no-verify" -> cfg.verify <- false
       | "-h" | "--help" -> usage name extra_usage
       | _ -> ()) ;
    incr i
  done ;
  cfg

(** Initialize runtime devices - uses all registered backends *)
let init_devices _cfg = Device.init ()

(** Get device based on config *)
let get_device cfg devs =
  if cfg.use_native then (
    match Array.find_opt (fun d -> d.Device.framework = "Native") devs with
    | Some d -> d
    | None ->
        print_endline "No native CPU device found" ;
        exit 1)
  else if cfg.use_interpreter then (
    match Array.find_opt (fun d -> d.Device.framework = "Interpreter") devs with
    | Some d -> d
    | None ->
        print_endline "No interpreter device found" ;
        exit 1)
  else if cfg.use_vulkan then (
    match Array.find_opt (fun d -> d.Device.framework = "Vulkan") devs with
    | Some d -> d
    | None ->
        print_endline "No Vulkan device found" ;
        exit 1)
  else if cfg.use_metal then (
    match Array.find_opt (fun d -> d.Device.framework = "Metal") devs with
    | Some d -> d
    | None ->
        print_endline "No Metal device found" ;
        exit 1)
  else if cfg.dev_id >= 0 then devs.(cfg.dev_id)
  else devs.(0)
(* default to first device *)

(** Print available devices *)
let print_devices devs =
  Printf.printf "Available devices:\n" ;
  Array.iteri
    (fun i d ->
      Printf.printf "  [%d] %s (%s)\n" i d.Device.name d.Device.framework)
    devs ;
  flush stdout

(** Run benchmark on selected devices (None = all, Some ids = specific) *)
let benchmark_all ?(device_ids = None) devs run_test name =
  (match device_ids with
  | None -> Printf.printf "\nBenchmark: %s (all devices)\n" name
  | Some ids ->
      Printf.printf
        "\nBenchmark: %s (devices: %s)\n"
        name
        (String.concat ", " (List.map string_of_int ids))) ;
  Printf.printf "%-40s %12s %10s\n" "Device" "Time (ms)" "Status" ;
  Printf.printf "%s\n" (String.make 64 '-') ;
  Array.iteri
    (fun i dev ->
      let should_run =
        match device_ids with None -> true | Some ids -> List.mem i ids
      in
      if should_run then begin
        let dev_name = dev.Device.name in
        flush stdout ;
        let time_ms, ok = run_test dev in
        let status = if ok then "OK" else "FAIL" in
        Printf.printf "%-40s %12.4f %10s\n%!" dev_name time_ms status
      end)
    devs ;
  print_endline "\nBenchmark complete."

(** Run benchmark with pure OCaml baseline, showing speedups *)
let benchmark_with_baseline ?(device_ids = None) devs ~baseline run_test name =
  (match device_ids with
  | None -> Printf.printf "\nBenchmark: %s (all devices)\n" name
  | Some ids ->
      Printf.printf
        "\nBenchmark: %s (devices: %s)\n"
        name
        (String.concat ", " (List.map string_of_int ids))) ;
  Printf.printf "%-40s %12s %10s %10s\n" "Device" "Time (ms)" "Status" "Speedup" ;
  Printf.printf "%s\n" (String.make 76 '-') ;
  (* Run baseline first *)
  let baseline_time, baseline_ok = baseline () in
  Printf.printf
    "%-40s %12.4f %10s %10s\n%!"
    "Pure OCaml (baseline)"
    baseline_time
    (if baseline_ok then "OK" else "FAIL")
    "1.00x" ;
  (* Run on devices *)
  Array.iteri
    (fun i dev ->
      let should_run =
        match device_ids with None -> true | Some ids -> List.mem i ids
      in
      if should_run then begin
        let dev_name = dev.Device.name in
        flush stdout ;
        let time_ms, ok = run_test dev in
        let status = if ok then "OK" else "FAIL" in
        let speedup = if time_ms > 0.0 then baseline_time /. time_ms else 0.0 in
        Printf.printf
          "%-40s %12.4f %10s %9.2fx\n%!"
          dev_name
          time_ms
          status
          speedup
      end)
    devs ;
  print_endline "\nBenchmark complete."

(** Get appropriate block size for device *)
let get_block_size cfg (dev : Device.t) =
  match dev.framework with
  | "OpenCL" ->
      (* CPU OpenCL can use larger work-groups for barrier-based kernels.
         Use cfg.block_size if specified, otherwise default to reasonable size. *)
      if cfg.block_size > 1 then cfg.block_size else 64
  | _ -> cfg.block_size

(** Verify float arrays are approximately equal *)
let verify_float_array expected actual tolerance =
  let n = Array.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let e = expected.(i) in
    let a = actual.(i) in
    if abs_float (e -. a) > tolerance then begin
      if !errors < 10 then
        Printf.printf "  Mismatch at %d: expected %.6f, got %.6f\n" i e a ;
      incr errors
    end
  done ;
  if !errors > 0 then Printf.printf "  Total errors: %d\n" !errors ;
  !errors = 0

(** Verify int32 arrays are equal *)
let verify_int32_array expected actual =
  let n = Array.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let e = expected.(i) in
    let a = actual.(i) in
    if e <> a then begin
      if !errors < 10 then
        Printf.printf "  Mismatch at %d: expected %ld, got %ld\n" i e a ;
      incr errors
    end
  done ;
  if !errors > 0 then Printf.printf "  Total errors: %d\n" !errors ;
  !errors = 0

(** Time a function and return (result, time_ms) *)
let time_it f =
  let t0 = Unix.gettimeofday () in
  let result = f () in
  let t1 = Unix.gettimeofday () in
  (result, (t1 -. t0) *. 1000.0)

module Benchmarks = Benchmarks
