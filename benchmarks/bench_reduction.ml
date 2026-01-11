(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Parallel Reduction Benchmark

    Measures performance of tree-based parallel sum reduction using shared
    memory. This is a fundamental parallel pattern used in many algorithms.

    Demonstrates:
    - Shared memory usage
    - Barrier synchronization (supersteps)
    - Logarithmic tree reduction
    - Multiple kernel launches for large arrays *)

module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Execute = Sarek.Execute

(* Benchmark infrastructure *)
open Benchmark_common
open Benchmark_backends

(** Benchmark configuration *)
type config = {
  sizes : int list;
  iterations : int;
  warmup : int;
  block_size : int;
  output_dir : string;
  device_filter : Device.t -> bool;
}

let default_config =
  {
    sizes = [1_000_000; 10_000_000; 50_000_000; 100_000_000];
    iterations = 20;
    warmup = 5;
    block_size = 256;
    output_dir = "results";
    device_filter =
      (fun dev ->
        (* By default, exclude slow CPU backends (Native, Interpreter) *)
        dev.Device.framework <> "Native"
        && dev.Device.framework <> "Interpreter");
  }

(** Pure OCaml baseline *)
let _ocaml_sum arr n =
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    sum := !sum +. arr.(i)
  done ;
  !sum

(** Tree-based parallel reduction kernel

    Uses shared memory and logarithmic reduction steps. Each block reduces 256
    elements to 1 element. Multiple kernel launches needed for large arrays. *)
let reduce_sum_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let%shared (sdata : float32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      let%superstep load =
        if gid < n then sdata.(tid) <- input.(gid) else sdata.(tid) <- 0.0
      in
      let%superstep reduce128 =
        if tid < 128l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 128l)
      in
      let%superstep reduce64 =
        if tid < 64l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 64l)
      in
      let%superstep reduce32 =
        if tid < 32l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 32l)
      in
      let%superstep reduce16 =
        if tid < 16l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 16l)
      in
      let%superstep reduce8 =
        if tid < 8l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 8l)
      in
      let%superstep reduce4 =
        if tid < 4l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 4l)
      in
      let%superstep reduce2 =
        if tid < 2l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 2l)
      in
      let%superstep reduce1 =
        if tid < 1l then sdata.(tid) <- sdata.(tid) +. sdata.(tid + 1l)
      in
      if tid = 0l then output.(block_idx_x) <- sdata.(0l)]
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

  (* Prepare host data - all ones so sum = n *)
  let input_data = Array.make n 1.0 in
  let expected_sum = float_of_int n in

  (* Get kernel IR *)
  let _, kirc = reduce_sum_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "No IR"
  in

  (* Setup execution parameters - 256 threads per block *)
  let block = Execute.dims1d config.block_size in

  (* Benchmark using the new API *)
  let init () =
    (* Create and upload device vectors *)
    let num_blocks = (n + config.block_size - 1) / config.block_size in
    let input_vec = Vector.create Vector.float32 n in
    let output_vec = Vector.create Vector.float32 num_blocks in

    (* Upload input data *)
    Array.iteri (fun i x -> Vector.set input_vec i x) input_data ;

    (* Initialize output to zero *)
    for i = 0 to num_blocks - 1 do
      Vector.set output_vec i 0.0
    done ;

    (input_vec, output_vec, num_blocks)
  in

  let compute (input_vec, output_vec, num_blocks) =
    (* Single kernel call that produces num_blocks partial sums *)
    let grid = Execute.dims1d num_blocks in
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:[Vec input_vec; Vec output_vec; Int32 (Int32.of_int n)]
      ~block
      ~grid
      ()
  in

  let verify (_input_vec, output_vec, num_blocks) =
    (* Download partial results and sum on CPU *)
    let partial_sums = Vector.to_array output_vec in
    let computed_sum = ref 0.0 in
    for i = 0 to num_blocks - 1 do
      computed_sum := !computed_sum +. partial_sums.(i)
    done ;
    let diff = abs_float (!computed_sum -. expected_sum) in
    let rel_err = if expected_sum > 0.0 then diff /. expected_sum else diff in
    let is_correct = rel_err < 0.001 in
    if not is_correct then
      Printf.eprintf
        "    WARNING: Verification failed! Expected %.0f, got %.6f (error: %.6f)\n"
        expected_sum
        !computed_sum
        rel_err ;
    is_correct
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

  (* Calculate throughput (GB/s for memory bandwidth) *)
  (* Each element is read once and written once in first pass, then logarithmically fewer *)
  let bytes = float_of_int (n * 4) in
  (* Total bytes ~ n * 4 * (1 + 1/256 + 1/256^2 + ...) ≈ n * 4 * 1.004 *)
  let total_bytes = bytes *. 1.004 in
  let mean_s = Common.mean times /. 1000.0 in
  let bandwidth_gbs = total_bytes /. mean_s /. 1e9 in

  Printf.printf
    "    Min time: %.3f ms, Bandwidth: %.3f GB/s, Verified: %s\n"
    (Common.min times)
    bandwidth_gbs
    (if verified then "✓" else "✗") ;
  flush stdout ;

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
      throughput = Some bandwidth_gbs;
      verified = Some verified;
    }

(** Run all benchmarks *)
let run config =
  Printf.printf "=== Parallel Reduction Benchmark (Sum) ===\n" ;
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

  if Array.length devices = 0 then begin
    Printf.eprintf "No devices available\n" ;
    exit 1
  end ;

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
        |> Array.of_list
      in

      (* Save results *)
      let benchmark_params =
        Output.
          {
            name = "reduction_sum";
            size;
            block_size = config.block_size;
            iterations = config.iterations;
            warmup = config.warmup;
          }
      in

      let benchmark_result =
        Output.
          {
            params = benchmark_params;
            timestamp = Common.timestamp_filename ();
            git_commit;
            system = system_info;
            results = Array.to_list results;
          }
      in

      let filename =
        Output.make_filename
          ~output_dir:config.output_dir
          ~benchmark_name:"reduction_sum"
          ~size
      in
      Output.write_json filename benchmark_result ;
      Printf.printf "Written: %s\n" filename ;
      flush stdout)
    config.sizes ;

  Printf.printf "\n=== Benchmark Complete ===\n"

(** Main entry point *)
let () = run default_config
