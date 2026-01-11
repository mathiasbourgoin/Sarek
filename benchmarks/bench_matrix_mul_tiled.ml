(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Matrix Multiplication (Tiled) Benchmark

    Benchmarks tiled matrix multiplication with shared memory optimization. Uses
    16x16 tiles with supersteps (barriers) for synchronization. Demonstrates
    performance improvement over naive version. *)

open Benchmark_common
open Benchmark_backends
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

(** Configuration *)
type config = {
  sizes : int list;
  tile_size : int;
  iterations : int;
  warmup : int;
  output_dir : string;
  device_filter : Device.t -> bool;
}

let default_config =
  {
    sizes = [128; 256; 512; 1024; 2048; 4096];
    tile_size = 16;
    iterations = 20;
    warmup = 5;
    output_dir = "results";
    device_filter =
      (fun dev ->
        (* Exclude backends that don't support shared memory *)
        dev.Device.framework <> "Native"
        && dev.Device.framework <> "Interpreter");
  }

(** Pure OCaml baseline *)
let ocaml_matmul a b c m n k =
  for row = 0 to m - 1 do
    for col = 0 to n - 1 do
      let sum = ref 0.0 in
      for i = 0 to k - 1 do
        sum := !sum +. (a.((row * k) + i) *. b.((i * n) + col))
      done ;
      c.((row * n) + col) <- !sum
    done
  done

(** Tiled matrix multiplication kernel with shared memory *)
let matmul_tiled_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let%shared (tile_a : float32) = 256l in
      let%shared (tile_b : float32) = 256l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let row = ty + (block_dim_y * block_idx_y) in
      let col = tx + (block_dim_x * block_idx_x) in
      let tile_size = 16l in
      let num_tiles = (k + tile_size - 1l) / tile_size in
      let sum = mut 0.0 in
      for t = 0 to num_tiles - 1l do
        let%superstep load_a =
          let a_col = (t * tile_size) + tx in
          if row < m && a_col < k then
            tile_a.((ty * tile_size) + tx) <- a.((row * k) + a_col)
          else tile_a.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep load_b =
          let b_row = (t * tile_size) + ty in
          if b_row < k && col < n then
            tile_b.((ty * tile_size) + tx) <- b.((b_row * n) + col)
          else tile_b.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep _compute =
          for i = 0 to tile_size - 1l do
            sum :=
              sum
              +. (tile_a.((ty * tile_size) + i) *. tile_b.((i * tile_size) + tx))
          done
        in
        ()
      done ;
      if row < m && col < n then c.((row * n) + col) <- sum]
[@@warning "-33"]

(** Run benchmark for one device and size *)
let benchmark_device dev size config =
  Printf.printf
    "  Device: %s (%s), Size: %d\n"
    dev.Device.name
    dev.Device.framework
    size ;
  flush stdout ;

  let dim = int_of_float (sqrt (float_of_int size)) in
  let m, n, k = (dim, dim, dim) in

  (* Prepare host data *)
  let a = Array.init (m * k) (fun i -> float_of_int (i mod 10) /. 10.0) in
  let b = Array.init (k * n) (fun i -> float_of_int ((i + 1) mod 10) /. 10.0) in
  let expected = Array.make (m * n) 0.0 in
  ocaml_matmul a b expected m n k ;

  (* Get kernel IR *)
  let _, kirc = matmul_tiled_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  (* Allocate device memory *)
  let va = Vector.create Vector.float32 (m * k) in
  let vb = Vector.create Vector.float32 (k * n) in
  let vc = Vector.create Vector.float32 (m * n) in

  (* Copy data to device *)
  for i = 0 to (m * k) - 1 do
    Vector.set va i a.(i)
  done ;
  for i = 0 to (k * n) - 1 do
    Vector.set vb i b.(i)
  done ;

  (* Configure 2D grid *)
  let tile_sz = config.tile_size in
  let blocks_x = (n + tile_sz - 1) / tile_sz in
  let blocks_y = (m + tile_sz - 1) / tile_sz in
  let block = Sarek.Execute.dims2d tile_sz tile_sz in
  let grid = Sarek.Execute.dims2d blocks_x blocks_y in
  let shared_mem = 2 * tile_sz * tile_sz * 4 in
  (* 2 tiles * tile_sz^2 elements * 4 bytes *)

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
          Sarek.Execute.Int32 (Int32.of_int m);
          Sarek.Execute.Int32 (Int32.of_int n);
          Sarek.Execute.Int32 (Int32.of_int k);
        ]
      ~block
      ~grid
      ~shared_mem
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
          Sarek.Execute.Int32 (Int32.of_int m);
          Sarek.Execute.Int32 (Int32.of_int n);
          Sarek.Execute.Int32 (Int32.of_int k);
        ]
      ~block
      ~grid
      ~shared_mem
      () ;
    Spoc_core.Transfer.flush dev ;
    let end_time = Unix.gettimeofday () in
    times := (end_time -. start_time) :: !times
  done ;

  (* Copy result back *)
  let result = Vector.to_array vc in

  (* Verify correctness *)
  let tolerance = 0.001 in
  let verify_result () =
    try
      for i = 0 to (m * n) - 1 do
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

  (* Compute throughput in GFLOPS *)
  let ops = 2.0 *. float_of_int m *. float_of_int n *. float_of_int k in
  let throughput_gflops = ops /. (median_ms *. 1e9) in

  (* Print results *)
  Printf.printf
    "  Mean: %.3f ms, Median: %.3f ms, StdDev: %.3f ms\n"
    mean_ms
    median_ms
    stddev_ms ;
  Printf.printf "  Throughput: %.3f GFLOPS\n" throughput_gflops ;
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
      throughput = Some throughput_gflops;
      verified = Some verified;
    }

(** Main benchmark runner *)
let run_benchmark config =
  Printf.printf "Matrix Multiplication (Tiled) Benchmark\n" ;
  Printf.printf "=========================================\n\n" ;

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
      let dim = int_of_float (sqrt (float_of_int size)) in
      Printf.printf "\n--- Size: %dx%d matrix ---\n" dim dim ;

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
                name = "matrix_mul_tiled";
                size;
                block_size = config.tile_size * config.tile_size;
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
          ~benchmark_name:"matrix_mul_tiled"
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
        "Comma-separated list of matrix sizes (default: \
         128,256,512,1024,2048,4096)" );
      ( "--iterations",
        Arg.Int (fun n -> config := {!config with iterations = n}),
        "Number of benchmark iterations (default: 20)" );
      ( "--warmup",
        Arg.Int (fun n -> config := {!config with warmup = n}),
        "Number of warmup iterations (default: 5)" );
      ( "--tile-size",
        Arg.Int (fun n -> config := {!config with tile_size = n}),
        "Tile size for shared memory (default: 16)" );
      ( "--output",
        Arg.String (fun s -> config := {!config with output_dir = s}),
        "Output directory (default: results)" );
      ( "--all-backends",
        Arg.Unit
          (fun () -> config := {!config with device_filter = (fun _ -> true)}),
        "Include all backends (Native, Interpreter)" );
    ]
  in
  Arg.parse specs (fun _ -> ()) "Tiled matrix multiplication benchmark" ;
  !config

(** Entry point *)
let () =
  let config = parse_args () in
  run_benchmark config
