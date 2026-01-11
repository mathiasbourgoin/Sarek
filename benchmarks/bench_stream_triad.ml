(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** STREAM Triad Benchmark

    Industry-standard memory bandwidth benchmark. The STREAM Triad kernel is the
    most demanding of the STREAM operations: A[i] = B[i] + C[i] * scalar. Used
    worldwide for comparing memory subsystem performance. *)

open Benchmark_common
open Benchmark_backends
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

(** Configuration *)
type config = {
  sizes : int list;
  block_size : int;
  iterations : int;
  warmup : int;
  output_dir : string;
  device_filter : Device.t -> bool;
}

let default_config =
  {
    sizes = [1_000_000; 10_000_000; 50_000_000; 100_000_000; 500_000_000];
    block_size = 256;
    iterations = 20;
    warmup = 5;
    output_dir = "results";
    device_filter =
      (fun dev ->
        dev.Device.framework <> "Native"
        && dev.Device.framework <> "Interpreter");
  }

(** STREAM Triad kernel: A[i] = B[i] + C[i] * scalar *)
let stream_triad_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (scalar : float32)
        (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then a.(tid) <- b.(tid) +. (c.(tid) *. scalar)]
[@@warning "-33"]

(** Run benchmark for one device and size *)
let benchmark_device dev size config =
  Printf.printf
    "  Device: %s (%s), Size: %d\n"
    dev.Device.name
    dev.Device.framework
    size ;
  flush stdout ;

  let n = size in
  let scalar = 3.0 in

  (* Prepare host data *)
  let b = Array.init n (fun i -> float_of_int (i mod 100) /. 100.0) in
  let c = Array.init n (fun i -> float_of_int ((i + 1) mod 100) /. 100.0) in
  let expected = Array.init n (fun i -> b.(i) +. (c.(i) *. scalar)) in

  (* Get kernel IR *)
  let _, kirc = stream_triad_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  (* Allocate device memory *)
  let va = Vector.create Vector.float32 n in
  let vb = Vector.create Vector.float32 n in
  let vc = Vector.create Vector.float32 n in

  (* Copy data to device *)
  for i = 0 to n - 1 do
    Vector.set vb i b.(i) ;
    Vector.set vc i c.(i)
  done ;

  (* Configure 1D grid *)
  let block_sz = config.block_size in
  let grid_sz = (n + block_sz - 1) / block_sz in
  let block = Sarek.Execute.dims1d block_sz in
  let grid = Sarek.Execute.dims1d grid_sz in

  (* Warmup *)
  for _ = 1 to config.warmup do
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Sarek.Execute.Vec va;
          Sarek.Execute.Vec vb;
          Sarek.Execute.Vec vc;
          Sarek.Execute.Float32 scalar;
          Sarek.Execute.Int32 (Int32.of_int n);
        ]
      ~block
      ~grid
      () ;
    Spoc_core.Transfer.flush dev
  done ;

  (* Benchmark iterations *)
  let times = ref [] in
  for _ = 1 to config.iterations do
    let start_time = Unix.gettimeofday () in
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Sarek.Execute.Vec va;
          Sarek.Execute.Vec vb;
          Sarek.Execute.Vec vc;
          Sarek.Execute.Float32 scalar;
          Sarek.Execute.Int32 (Int32.of_int n);
        ]
      ~block
      ~grid
      () ;
    Spoc_core.Transfer.flush dev ;
    let end_time = Unix.gettimeofday () in
    times := (end_time -. start_time) :: !times
  done ;

  (* Copy result back *)
  let result = Vector.to_array va in

  (* Verify correctness *)
  let tolerance = 1e-5 in
  let verify_result () =
    try
      for i = 0 to n - 1 do
        let diff = abs_float (result.(i) -. expected.(i)) in
        if diff > tolerance then
          failwith
            (Printf.sprintf
               "Mismatch at index %d: expected=%f, got=%f, diff=%f"
               i
               expected.(i)
               result.(i)
               diff)
      done ;
      true
    with Failure msg ->
      Printf.eprintf "Verification failed: %s\n" msg ;
      false
  in

  let verified = verify_result () in

  (* Compute statistics *)
  let times_array = Array.of_list !times in
  let mean_ms = Common.mean times_array in
  let stddev_ms = Common.stddev times_array in
  let median_ms = Common.median times_array in
  let min_ms = Common.min times_array in
  let max_ms = Common.max times_array in

  (* Compute bandwidth in GB/s *)
  (* 4 memory operations: 2 reads (B, C) + 1 write (A) + 1 scalar = 3 vectors
     = 3 * n * 4 bytes *)
  let bytes_transferred = 3.0 *. float_of_int n *. 4.0 in
  let bandwidth_gb_s = bytes_transferred /. (median_ms *. 1e9) in

  (* Print results *)
  Printf.printf
    "  Mean: %.3f ms, Median: %.3f ms, StdDev: %.3f ms\n"
    mean_ms
    median_ms
    stddev_ms ;
  Printf.printf "  Bandwidth: %.3f GB/s\n" bandwidth_gb_s ;
  Printf.printf "  Verified: %s\n" (if verified then "✓" else "✗") ;
  flush stdout ;

  Output.
    {
      device_id = dev.Device.id;
      device_name = dev.Device.name;
      framework = dev.Device.framework;
      iterations = times_array;
      mean_ms;
      stddev_ms;
      median_ms;
      min_ms;
      max_ms;
      throughput = Some bandwidth_gb_s;
      verified = Some verified;
    }

(** Main benchmark runner *)
let run_benchmark config =
  Printf.printf "STREAM Triad Benchmark\n" ;
  Printf.printf "======================\n\n" ;

  (* Initialize backends *)
  Backend_loader.init () ;

  (* Initialize devices *)
  let devices = Device.init () in

  Printf.printf "Available devices:\n" ;
  Array.iter
    (fun dev ->
      Printf.printf "  - %s (%s)\n" dev.Device.name dev.Device.framework)
    devices ;
  flush stdout ;

  (* Filter devices *)
  let devices =
    Array.to_list devices |> List.filter config.device_filter |> Array.of_list
  in

  if Array.length devices = 0 then begin
    Printf.eprintf "Error: No devices match filter\n" ;
    exit 1
  end ;

  (* Create output directory *)
  if not (Sys.file_exists config.output_dir) then
    Unix.mkdir config.output_dir 0o755 ;

  (* Collect system info once *)
  let system_info = System_info.collect devices in
  let git_commit = Common.get_git_commit () in

  (* Run benchmarks for each size *)
  List.iter
    (fun size ->
      Printf.printf "\n--- Size: %d elements ---\n" size ;

      (* Benchmark all devices *)
      let results =
        Array.to_list devices
        |> List.mapi (fun device_id dev ->
            try
              let r = benchmark_device dev size config in
              {r with Output.device_id}
            with e ->
              Printf.eprintf
                "Error on %s: %s\n"
                dev.Device.name
                (Printexc.to_string e) ;
              Output.
                {
                  device_id;
                  device_name = dev.Device.name;
                  framework = dev.Device.framework;
                  iterations = [||];
                  mean_ms = 0.0;
                  stddev_ms = 0.0;
                  median_ms = 0.0;
                  min_ms = 0.0;
                  max_ms = 0.0;
                  throughput = None;
                  verified = Some false;
                })
      in

      (* Create benchmark result *)
      let result =
        Output.
          {
            params =
              {
                name = "stream_triad";
                size;
                block_size = config.block_size;
                iterations = config.iterations;
                warmup = config.warmup;
              };
            timestamp = Common.get_timestamp ();
            git_commit;
            system = system_info;
            results;
          }
      in

      (* Write JSON output *)
      let filename =
        Output.make_filename
          ~output_dir:config.output_dir
          ~benchmark_name:"stream_triad"
          ~size
      in
      Output.write_json filename result ;
      Printf.printf "\nResults written to: %s\n" filename ;
      flush stdout)
    config.sizes ;

  Printf.printf "\nBenchmark complete!\n"

(** Parse command line arguments *)
let parse_args () =
  let config = ref default_config in

  let specs =
    [
      ( "--sizes",
        Arg.String
          (fun s ->
            config :=
              {
                !config with
                sizes = List.map int_of_string (String.split_on_char ',' s);
              }),
        "Comma-separated list of sizes (default: 1M,10M,50M,100M,500M)" );
      ( "--iterations",
        Arg.Int (fun n -> config := {!config with iterations = n}),
        "Number of benchmark iterations (default: 20)" );
      ( "--warmup",
        Arg.Int (fun n -> config := {!config with warmup = n}),
        "Number of warmup iterations (default: 5)" );
      ( "--block-size",
        Arg.Int (fun n -> config := {!config with block_size = n}),
        "Block size (default: 256)" );
      ( "--output",
        Arg.String (fun s -> config := {!config with output_dir = s}),
        "Output directory (default: results)" );
      ( "--all-backends",
        Arg.Unit
          (fun () -> config := {!config with device_filter = (fun _ -> true)}),
        "Include all backends (Native, Interpreter)" );
    ]
  in
  Arg.parse specs (fun _ -> ()) "STREAM Triad benchmark" ;
  !config

(** Entry point *)
let () =
  let config = parse_args () in
  run_benchmark config
