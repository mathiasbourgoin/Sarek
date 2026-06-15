(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Shared benchmark runner infrastructure

    Provides common configuration, CLI parsing, and execution flow for all
    benchmarks. Reduces code duplication and ensures consistency. *)

open Spoc_core

(** Standard benchmark configuration *)
type config = {
  sizes : int list;
  block_size : int;
  iterations : int;
  warmup : int;
  output_dir : string;
  device_filter : Device.t -> bool;
}

(** Default configuration suitable for most benchmarks *)
let default_config =
  {
    sizes = [1_000_000; 10_000_000; 50_000_000; 100_000_000];
    block_size = 256;
    iterations = 20;
    warmup = 5;
    output_dir = "results";
    device_filter =
      (fun dev ->
        dev.Device.framework <> "Native"
        && dev.Device.framework <> "Interpreter");
  }

(** Parse standard CLI arguments *)
let parse_args ~benchmark_name ~default_sizes () =
  let config = ref {default_config with sizes = default_sizes} in
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
        "Comma-separated list of sizes" );
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
  Arg.parse specs (fun _ -> ()) (Printf.sprintf "%s benchmark" benchmark_name) ;
  !config

(** Benchmark execution callback type

    @param device Device to run on
    @param size Problem size
    @param config Benchmark configuration
    @return Output.device_result with timing and verification *)
type benchmark_fn =
  device:Device.t -> size:int -> config:config -> Output.device_result

(** Run benchmark across all sizes and devices, generate JSON output

    @param benchmark_name Name of the benchmark (for output files)
    @param config Configuration with sizes, iterations, etc.
    @param run_fn Function to run single benchmark (device + size) *)
let run_benchmark ~benchmark_name ~config ~run_fn =
  Printf.printf "%s Benchmark\n" benchmark_name ;
  Printf.printf "%s\n\n" (String.make (String.length benchmark_name + 10) '=') ;

  (* Initialize backends *)
  Benchmark_backends.Backend_loader.init () ;

  (* Initialize devices *)
  let devices =
    Device.init () |> Array.to_list
    |> List.filter config.device_filter
    |> Array.of_list
  in

  if Array.length devices = 0 then (
    Printf.eprintf "No suitable devices found.\n" ;
    exit 1) ;

  (* Create output directory *)
  if not (Sys.file_exists config.output_dir) then
    Unix.mkdir config.output_dir 0o755 ;

  (* Collect system info once *)
  let system_info = System_info.collect devices in
  let git_commit = Common.get_git_commit () in

  (* Run for each size *)
  List.iter
    (fun size ->
      Printf.printf "\n--- Size: %d ---\n" size ;

      (* Run on each device *)
      let results =
        Array.to_list devices
        |> List.mapi (fun device_id dev ->
            try
              let r = run_fn ~device:dev ~size ~config in
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
                name = benchmark_name;
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
        Output.make_filename ~output_dir:config.output_dir ~benchmark_name ~size
      in
      Output.write_json filename result ;
      Printf.printf "\nResults written to: %s\n" filename ;
      flush stdout)
    config.sizes ;

  Printf.printf "\nBenchmark complete!\n"

(** Simple runner for console-only benchmarks (Sprint 2 pattern)

    For benchmarks that don't yet use the full JSON output infrastructure. Just
    handles CLI arg parsing for size and device selection.

    @param benchmark_name Name for display
    @param default_size Default problem size
    @param run_fn Function (device -> size -> bool) returning success *)
let run_simple ~benchmark_name ~default_size ~run_fn =
  let size = ref default_size in
  let specs =
    [
      ("--size", Arg.Set_int size, "Problem size");
      ("-s", Arg.Set_int size, "Problem size (short form)");
    ]
  in
  Arg.parse specs (fun _ -> ()) (Printf.sprintf "%s benchmark" benchmark_name) ;

  (* Initialize backends *)
  Benchmark_backends.Backend_loader.init () ;

  (* Initialize devices *)
  let devices = Device.init () in
  if Array.length devices = 0 then (
    Printf.eprintf "No GPU devices found. Cannot run benchmark.\n" ;
    exit 1) ;

  (* Run on first available device *)
  let device = devices.(0) in
  let backend_name = device.Device.framework in

  Printf.printf "Using device: %s (%s)\n\n" device.Device.name backend_name ;

  let success = run_fn device !size in
  exit (if success then 0 else 1)
