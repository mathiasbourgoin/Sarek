(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Matrix Multiplication Benchmark

    Benchmarks naive and tiled matrix multiplication across all devices. Outputs
    JSON files suitable for multi-machine aggregation. *)

open Sarek
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
    sizes = [256; 512; 1024; 2048];
    block_size = 256;
    iterations = 20;
    warmup = 5;
    output_dir = "results";
    device_filter =
      (fun dev ->
        (* By default, exclude slow CPU backends (Native, Interpreter) *)
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

(** Naive matrix multiplication kernel *)
let matmul_naive_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let open Std in
      let tid = global_thread_id in
      let row = tid / n in
      let col = tid mod n in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum := sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done ;
        c.((row * n) + col) <- sum
      end]
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
  let _, kirc = matmul_naive_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  (* Setup execution parameters *)
  let block = Execute.dims1d config.block_size in
  let grid =
    Execute.dims1d (((m * n) + config.block_size - 1) / config.block_size)
  in

  (* Benchmark using the new API *)
  let init () =
    (* Create and upload device vectors *)
    let va = Vector.create Vector.float32 (m * k) in
    let vb = Vector.create Vector.float32 (k * n) in
    let vc = Vector.create Vector.float32 (m * n) in
    Array.iteri (fun i x -> Vector.set va i x) a ;
    Array.iteri (fun i x -> Vector.set vb i x) b ;
    (va, vb, vc)
  in

  let compute (va, vb, vc) =
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:[Vec va; Vec vb; Vec vc; Int m; Int n; Int k]
      ~block
      ~grid
      ()
  in

  let verify (_va, _vb, vc) =
    let result_array = Vector.to_array vc in
    Common.arrays_equal ~epsilon:0.001 result_array expected
  in

  let times, verified =
    Common.benchmark_gpu
      ~dev
      ~warmup:config.warmup
      ~iterations:config.iterations
      ~init
      ~compute
      ~verify
  in

  (* Compute throughput (GFLOPS) *)
  let ops = float_of_int (2 * m * n * k) in
  (* multiply-add per element *)
  let mean_s = Common.mean times /. 1000.0 in
  let gflops = ops /. mean_s /. 1e9 in

  (* Create result *)
  Output.
    {
      device_id = 0;
      (* Will be set properly by caller *)
      device_name = dev.Device.name;
      framework = dev.Device.framework;
      iterations = times;
      mean_ms = Common.mean times;
      stddev_ms = Common.stddev times;
      median_ms = Common.median times;
      min_ms = Common.min times;
      max_ms = Common.max times;
      throughput = Some gflops;
      verified = Some verified;
    }

(** Run full benchmark suite *)
let run config =
  Printf.printf "=== Matrix Multiplication Benchmark ===\n" ;
  Printf.printf "Iterations: %d, Warmup: %d\n" config.iterations config.warmup ;
  Printf.printf
    "Sizes: %s\n"
    (String.concat ", " (List.map string_of_int config.sizes)) ;
  flush stdout ;

  (* Initialize backends *)
  Backend_loader.init () ;

  (* Initialize devices *)
  let devices = Device.init () in
  let devices =
    Array.to_list devices |> List.filter config.device_filter |> Array.of_list
  in

  if Array.length devices = 0 then (
    Printf.eprintf "No devices available\n" ;
    exit 1) ;

  Printf.printf "Devices: %d\n" (Array.length devices) ;
  Array.iter
    (fun dev ->
      Printf.printf "  - %s (%s)\n" dev.Device.name dev.Device.framework)
    devices ;

  (* Create output directory *)
  if not (Sys.file_exists config.output_dir) then
    Unix.mkdir config.output_dir 0o755 ;

  (* Collect system info once *)
  let system_info = System_info.collect devices in
  let git_commit = Common.get_git_commit () in

  (* Run benchmarks for each size *)
  List.iter
    (fun size ->
      Printf.printf "\n--- Size: %d ---\n" size ;

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
                name = "matrix_mul_naive";
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
          ~benchmark_name:"matrix_mul_naive"
          ~size
      in
      Output.write_json filename result ;
      Printf.printf "Written: %s\n" filename ;

      (* Also append to CSV for quick analysis *)
      let csv_filename =
        Filename.concat config.output_dir "matrix_mul_naive.csv"
      in
      Output.append_csv csv_filename result)
    config.sizes ;

  Printf.printf "\n=== Benchmark Complete ===\n"

(** Main entry point *)
let () =
  let config = ref default_config in

  let speclist =
    [
      ( "--sizes",
        Arg.String
          (fun s ->
            config :=
              {
                !config with
                sizes = String.split_on_char ',' s |> List.map int_of_string;
              }),
        "Comma-separated list of sizes (e.g., 256,512,1024)" );
      ( "--iterations",
        Arg.Int (fun n -> config := {!config with iterations = n}),
        "Number of iterations per benchmark" );
      ( "--warmup",
        Arg.Int (fun n -> config := {!config with warmup = n}),
        "Number of warmup iterations" );
      ( "--block-size",
        Arg.Int (fun n -> config := {!config with block_size = n}),
        "Block size for kernel launch" );
      ( "--output",
        Arg.String (fun s -> config := {!config with output_dir = s}),
        "Output directory for results" );
      ( "--all-devices",
        Arg.Unit
          (fun () -> config := {!config with device_filter = (fun _ -> true)}),
        "Benchmark all devices including slow CPU backends (Native, \
         Interpreter)" );
    ]
  in

  Arg.parse speclist (fun _ -> ()) "Matrix multiplication benchmark" ;
  run !config
