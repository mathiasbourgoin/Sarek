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
open Benchmark_backends

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

(** Run histogram benchmark on specified device *)
let run_histogram_benchmark device backend_name size =
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

  (* Warmup run *)
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
  Transfer.flush device ;

  (* Timed runs *)
  let num_runs = 100 in
  let times = ref [] in
  let final_histogram = ref (Vector.create Vector.int32 num_bins) in

  for _ = 1 to num_runs do
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
    times := (t1 -. t0) :: !times ;
    final_histogram := histogram
  done ;

  (* Compute statistics *)
  let sorted_times = List.sort compare !times in
  let avg_time =
    List.fold_left ( +. ) 0.0 sorted_times /. float_of_int num_runs
  in
  let min_time = List.hd sorted_times in
  let median_time = List.nth sorted_times (num_runs / 2) in

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

  (* Report results *)
  Printf.printf "\n=== Histogram Benchmark ===\n" ;
  Printf.printf "Backend: %s\n" backend_name ;
  Printf.printf "Array size: %d elements\n" n ;
  Printf.printf "Number of bins: %d\n" num_bins ;
  Printf.printf "Block size: %d\n" block_size ;
  Printf.printf "Grid size: %d blocks\n" num_blocks ;
  Printf.printf "Supersteps: 3 (init, count, merge)\n" ;
  Printf.printf "\nTiming (%d runs):\n" num_runs ;
  Printf.printf "  Min:    %.3f ms\n" (min_time *. 1000.0) ;
  Printf.printf "  Median: %.3f ms\n" (median_time *. 1000.0) ;
  Printf.printf "  Mean:   %.3f ms\n" (avg_time *. 1000.0) ;

  let throughput_melems = float_of_int n /. (median_time *. 1e6) in
  Printf.printf "  Throughput: %.2f M elements/s\n" throughput_melems ;

  Printf.printf "\nVerification: " ;
  if !errors = 0 then Printf.printf "PASS ✓ (all %d bins correct)\n" num_bins
  else Printf.printf "FAIL ✗ (%d/%d errors)\n" !errors num_bins ;

  (* Show first few bins and total count *)
  Printf.printf "First 8 bins: [" ;
  for i = 0 to min 7 (num_bins - 1) do
    Printf.printf "%ld%s" gpu_result.(i) (if i < 7 then "; " else "")
  done ;
  Printf.printf "]\n" ;

  let total_count = Array.fold_left Int32.add 0l gpu_result in
  Printf.printf "Total count: %ld (expected %d)\n" total_count n ;
  Printf.printf "===========================\n" ;

  !errors = 0

(** Main benchmark runner *)
let () =
  Printf.printf "Histogram Computation Benchmark\n" ;
  Printf.printf "Atomic operations with shared memory\n\n" ;

  let size =
    if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 65536
  in

  (* Initialize backends *)
  Backend_loader.init () ;

  (* Initialize SPOC and discover devices *)
  let devices = Device.init () in

  if Array.length devices = 0 then begin
    Printf.printf "No GPU devices found. Cannot run benchmark.\n" ;
    exit 1
  end ;

  (* Run on first available device *)
  let device = devices.(0) in
  let backend_name = device.Device.framework in

  Printf.printf "Using device: %s (%s)\n\n" device.Device.name backend_name ;

  let success = run_histogram_benchmark device backend_name size in
  exit (if success then 0 else 1)
