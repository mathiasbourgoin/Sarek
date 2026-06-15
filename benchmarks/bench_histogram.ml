(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: Histogram Computation
 *
 * Priority: P1 (Sprint 2)
 * Pattern: Atomic operations, shared memory, supersteps
 * Category: Data analysis primitive
 *
 * Description:
 * Implements histogram computation using shared memory and atomic operations.
 * Each block maintains a local histogram in shared memory, then merges to
 * global memory. This is a common pattern in data analysis and image processing.
 *
 * Performance Notes:
 * - Uses shared memory to reduce global memory contention
 * - Atomic operations for thread-safe counting
 * - Three supersteps: initialize local, count, merge to global
 * - Memory bandwidth: dominated by input reads and atomic updates
 * - Performance sensitive to data distribution and number of bins
 *
 * Hardware Tested:
 * - Intel Arc GPU (OpenCL + Vulkan)
 * - Intel Core CPU (OpenCL)
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std
module Gpu = Sarek_stdlib.Gpu
open Benchmark_common
open Benchmark_runner

(** Pure OCaml baseline: simple histogram *)
let cpu_histogram input n bins =
  let hist = Array.make bins 0l in
  for i = 0 to n - 1 do
    let bin = Int32.to_int (Int32.rem input.(i) (Int32.of_int bins)) in
    hist.(bin) <- Int32.add hist.(bin) 1l
  done ;
  hist

(** Sarek kernel: Histogram with shared memory and atomics *)
let histogram_kernel =
  [%kernel
    fun (input : int32 vector)
        (histogram : int32 vector)
        (n : int32)
        (num_bins : int32) ->
      let open Std in
      let open Gpu in
      let%shared (local_hist : int32) = 256l in
      let tid = thread_idx_x in
      let gid = global_thread_id in
      (* Initialize local histogram *)
      let%superstep init = if tid < num_bins then local_hist.(tid) <- 0l in
      (* Count in local histogram using atomic increment *)
      let%superstep[@divergent] count =
        if gid < n then begin
          let bin = input.(gid) mod num_bins in
          let _old = atomic_add_int32 local_hist bin 1l in
          ()
        end
      in
      (* Merge to global histogram using atomics *)
      let%superstep[@divergent] merge =
        if tid < num_bins then begin
          let _old = atomic_add_global_int32 histogram tid local_hist.(tid) in
          ()
        end
      in
      ()]
[@@warning "-33"]

(** Run histogram benchmark on specified device *)
let run_histogram_benchmark ~device ~size ~config =
  let backend_name = device.Device.framework in
  let n = size in
  let num_bins = 256 in

  let _, kirc = histogram_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test data: random values in range [0, num_bins) *)
  Random.init 42 ;
  let input = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    Vector.set input i (Int32.of_int (Random.int num_bins))
  done ;

  let histogram = Vector.create Vector.int32 num_bins in

  (* Launch configuration *)
  let block_size = 256 in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in
  let shared_mem = num_bins * 4 in
  (* 4 bytes per int32 *)

  (* Warmup *)
  for _ = 1 to config.warmup do
    for i = 0 to num_bins - 1 do
      Vector.set histogram i 0l
    done ;
    Sarek.Execute.run_vectors
      ~device
      ~ir
      ~args:
        [
          Sarek.Execute.Vec input;
          Sarek.Execute.Vec histogram;
          Sarek.Execute.Int32 (Int32.of_int n);
          Sarek.Execute.Int32 (Int32.of_int num_bins);
        ]
      ~block
      ~grid
      ~shared_mem
      () ;
    Transfer.flush device
  done ;

  (* Timed runs *)
  let times = ref [] in
  let final_histogram = ref (Vector.create Vector.int32 num_bins) in

  for _ = 1 to config.iterations do
    (* Create fresh histogram *)
    let histogram = Vector.create Vector.int32 num_bins in
    for i = 0 to num_bins - 1 do
      Vector.set histogram i 0l
    done ;

    let t0 = Unix.gettimeofday () in
    Sarek.Execute.run_vectors
      ~device
      ~ir
      ~args:
        [
          Sarek.Execute.Vec input;
          Sarek.Execute.Vec histogram;
          Sarek.Execute.Int32 (Int32.of_int n);
          Sarek.Execute.Int32 (Int32.of_int num_bins);
        ]
      ~block
      ~grid
      ~shared_mem
      () ;
    Transfer.flush device ;
    let t1 = Unix.gettimeofday () in
    times := ((t1 -. t0) *. 1000.0) :: !times ;
    final_histogram := histogram
  done ;

  (* Verify correctness *)
  let gpu_result = Vector.to_array !final_histogram in
  let input_arr = Array.init n (fun i -> Vector.get input i) in
  let cpu_result = cpu_histogram input_arr n num_bins in

  let errors = ref 0 in
  for i = 0 to num_bins - 1 do
    if gpu_result.(i) <> cpu_result.(i) then begin
      if !errors < 5 then
        Printf.printf
          "  ERROR at bin %d: GPU=%ld CPU=%ld\n"
          i
          gpu_result.(i)
          cpu_result.(i) ;
      incr errors
    end
  done ;

  let verified = !errors = 0 in
  let times_array = Array.of_list (List.rev !times) in
  let median_ms = Common.median times_array in

  (* Report results *)
  Printf.printf
    "  %s: size=%d, median=%.3f ms, verified=%s\n"
    device.Device.name
    n
    median_ms
    (if verified then "✓" else "✗") ;

  let throughput_melems = float_of_int n /. (median_ms /. 1000.0 *. 1e6) in

  Output.
    {
      device_id = device.Device.id;
      device_name = device.Device.name;
      framework = backend_name;
      iterations = times_array;
      mean_ms = Common.mean times_array;
      stddev_ms = Common.stddev times_array;
      median_ms;
      min_ms = Common.min times_array;
      max_ms = Common.max times_array;
      throughput = Some throughput_melems;
      verified = Some verified;
    }

(** Main benchmark runner *)
let () =
  let config =
    Benchmark_runner.parse_args
      ~benchmark_name:"histogram"
      ~default_sizes:[1_000_000; 10_000_000; 50_000_000]
      ()
  in
  Benchmark_runner.run_benchmark
    ~benchmark_name:"histogram"
    ~config
    ~run_fn:run_histogram_benchmark
