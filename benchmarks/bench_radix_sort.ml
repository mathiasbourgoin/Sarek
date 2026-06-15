(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: Radix Sort
 *
 * Priority: P1 (Sprint 2)
 * Pattern: Histogram-based sorting with multiple passes
 * Category: Sorting algorithm
 *
 * Description:
 * Implements a single pass of radix sort using 8-bit digits (256 bins). 
 * This benchmarks the core histogram and scatter kernels.
 * Note: A full stable multi-pass radix sort requires stable scattering, 
 * which is complex to implement efficiently with global atomics.
 * This benchmark focuses on the performance of the histogram and scatter steps.
 *
 * Performance Notes:
 * - Complexity: O(n) for single pass
 * - Uses histogram and prefix sum as building blocks
 * - Unstable sort (global atomics) - acceptable for single pass binning
 * - Memory: requires temporary buffer for ping-pong between passes
 * - Good for uniformly distributed integer data
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

(** Pure OCaml baseline: radix sort *)
let cpu_radix_sort input n =
  let bits_per_pass = 4 in
  let num_bins = 1 lsl bits_per_pass in
  let num_passes = 32 / bits_per_pass in
  let mask = Int32.of_int (num_bins - 1) in

  let src = Array.copy input in
  let dst = Array.make n 0l in

  for pass = 0 to num_passes - 1 do
    let shift = pass * bits_per_pass in

    (* Count occurrences of each digit *)
    let counts = Array.make num_bins 0 in
    for i = 0 to n - 1 do
      let digit =
        Int32.to_int
          (Int32.logand (Int32.shift_right_logical src.(i) shift) mask)
      in
      counts.(digit) <- counts.(digit) + 1
    done ;

    (* Compute prefix sum for starting positions *)
    let positions = Array.make num_bins 0 in
    for i = 1 to num_bins - 1 do
      positions.(i) <- positions.(i - 1) + counts.(i - 1)
    done ;

    (* Distribute elements to their positions *)
    for i = 0 to n - 1 do
      let digit =
        Int32.to_int
          (Int32.logand (Int32.shift_right_logical src.(i) shift) mask)
      in
      dst.(positions.(digit)) <- src.(i) ;
      positions.(digit) <- positions.(digit) + 1
    done ;

    (* Swap buffers *)
    Array.blit dst 0 src 0 n
  done ;

  src

(** Sarek kernel: Histogram for one radix pass *)
let radix_histogram_kernel =
  [%kernel
    fun (input : int32 vector)
        (histogram : int32 vector)
        (n : int32)
        (shift : int32)
        (mask : int32) ->
      let open Std in
      let open Gpu in
      let%shared (local_hist : int32) = 256l in
      let tid = thread_idx_x in
      let gid = global_thread_id in
      let num_bins = 256l in
      (* Initialize local histogram *)
      let%superstep init = if tid < num_bins then local_hist.(tid) <- 0l in
      (* Count in local histogram *)
      let%superstep[@divergent] count =
        if gid < n then begin
          let value = input.(gid) in
          let digit = (value lsr shift) land mask in
          let _old = atomic_add_int32 local_hist digit 1l in
          ()
        end
      in
      (* Merge to global histogram *)
      let%superstep[@divergent] merge =
        if tid < num_bins then begin
          let _old = atomic_add_global_int32 histogram tid local_hist.(tid) in
          ()
        end
      in
      ()]
[@@warning "-33"]

(** Sarek kernel: Scatter elements based on prefix sum *)
let radix_scatter_kernel =
  [%kernel
    fun (input : int32 vector)
        (output : int32 vector)
        (counters : int32 vector)
        (n : int32)
        (shift : int32)
        (mask : int32) ->
      let open Std in
      let open Gpu in
      let gid = global_thread_id in
      if gid < n then begin
        let value = input.(gid) in
        let digit = (value lsr shift) land mask in
        (* Atomically get and increment counter for this digit *)
        let pos = atomic_add_global_int32 counters digit 1l in
        output.(pos) <- value
      end]
[@@warning "-33"]

(** Compute prefix sum on CPU (helper) *)
let cpu_prefix_sum arr n =
  let result = Array.make n 0l in
  if n > 0 then begin
    result.(0) <- 0l ;
    for i = 1 to n - 1 do
      result.(i) <- Int32.add result.(i - 1) arr.(i - 1)
    done
  end ;
  result

(** Run radix sort benchmark on specified device *)
let run_radix_sort_benchmark ~device ~size ~config =
  let n = size in
  let bits_per_pass = 8 in
  let num_bins = 1 lsl bits_per_pass in
  let _num_passes = 1 in
  (* Single pass benchmark to avoid stability issues with global atomics *)
  let mask = Int32.of_int (num_bins - 1) in

  let _, hist_kirc = radix_histogram_kernel in
  let hist_ir =
    match hist_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Histogram kernel has no IR"
  in

  let _, scatter_kirc = radix_scatter_kernel in
  let scatter_ir =
    match scatter_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Scatter kernel has no IR"
  in

  (* Launch configuration *)
  let block_size = 256 in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  (* Prepare host data for verification *)
  Random.init 42 ;
  let input_arr = Array.init n (fun _ -> Int32.of_int (Random.int 100000)) in
  let _cpu_result = cpu_radix_sort input_arr n in

  (* init: Create fresh vectors from host data *)
  let init () =
    let input = Vector.create Vector.int32 n in
    let output = Vector.create Vector.int32 n in
    Array.iteri (fun i x -> Vector.set input i x) input_arr ;
    (* Create reusable histogram and counters vectors *)
    let histogram = Vector.create Vector.int32 num_bins in
    let counters = Vector.create Vector.int32 num_bins in
    (input, output, histogram, counters)
  in

  (* compute: Run single radix pass *)
  let compute (input, output, histogram, counters) =
    (* Reinitialize input data at start of each compute call *)
    Array.iteri (fun i x -> Vector.set input i x) input_arr ;
    Transfer.to_device input device ;

    (* Run single pass (shift = 0) *)
    let shift = 0l in

    (* Reset histogram *)
    for i = 0 to num_bins - 1 do
      Vector.set histogram i 0l
    done ;
    (* Transfer histogram to GPU *)
    Transfer.to_device histogram device ;

    (* Compute histogram *)
    Sarek.Execute.run_vectors
      ~device
      ~ir:hist_ir
      ~args:
        [
          Sarek.Execute.Vec input;
          Sarek.Execute.Vec histogram;
          Sarek.Execute.Int32 (Int32.of_int n);
          Sarek.Execute.Int32 shift;
          Sarek.Execute.Int32 mask;
        ]
      ~block
      ~grid
      () ;
    Device.synchronize device ;

    (* Compute prefix sum on CPU - histogram must be transferred back *)
    Transfer.to_cpu ~force:true histogram ;
    let hist_arr = Vector.to_array histogram in

    (* Verify histogram sum *)
    let total_count =
      Array.fold_left (fun acc x -> acc + Int32.to_int x) 0 hist_arr
    in
    if total_count <> n then
      Printf.printf
        "WARNING: Histogram sum %d != n %d (Diff: %d)\n"
        total_count
        n
        (n - total_count) ;

    (* cpu_prefix_sum computes STARTING positions for each bin *)
    let prefix_arr = cpu_prefix_sum hist_arr num_bins in

    for i = 0 to num_bins - 1 do
      Vector.set counters i prefix_arr.(i)
    done ;
    (* Transfer counters to GPU *)
    Transfer.to_device counters device ;

    (* Scatter elements *)
    Sarek.Execute.run_vectors
      ~device
      ~ir:scatter_ir
      ~args:
        [
          Sarek.Execute.Vec input;
          Sarek.Execute.Vec output;
          Sarek.Execute.Vec counters;
          Sarek.Execute.Int32 (Int32.of_int n);
          Sarek.Execute.Int32 shift;
          Sarek.Execute.Int32 mask;
        ]
      ~block
      ~grid
      () ;
    Device.synchronize device ;

    (* WORKAROUND: Force sync counters to reset state for next iteration *)
    Transfer.to_cpu ~force:true counters ;

    (* Result is in output *)
    ()
  in

  (* verify: Check correctness *)
  let verify (_input, output, _histogram, _counters) =
    (* Ensure all GPU work is complete and data is back on host *)
    Transfer.flush device ;
    Device.synchronize device ;
    Transfer.to_cpu ~force:true output ;

    (* Verify sortedness by LSB 8 bits *)
    let output_arr = Vector.to_array output in
    let errors = ref 0 in

    Printf.printf "Checking LSB 8-bit sortedness...\n" ;
    for i = 1 to n - 1 do
      let prev_digit = Int32.to_int output_arr.(i - 1) land 255 in
      let curr_digit = Int32.to_int output_arr.(i) land 255 in
      if curr_digit < prev_digit then begin
        if !errors < 10 then
          Printf.printf
            "Error at %d: prev_digit=%d curr_digit=%d (val=%ld)\n"
            i
            prev_digit
            curr_digit
            output_arr.(i) ;
        incr errors
      end
    done ;

    Printf.printf "Errors: %d\n" !errors ;
    !errors = 0
  in

  (* Run benchmark using Common infrastructure *)
  let times, verified =
    Common.benchmark_gpu
      ~dev:device
      ~warmup:config.warmup
      ~iterations:config.iterations
      ~init
      ~compute
      ~verify
  in

  let times_array = times in
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
      framework = device.Device.framework;
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
      ~benchmark_name:"radix_sort"
      ~default_sizes:[1_000_000; 10_000_000; 50_000_000]
      ()
  in
  Benchmark_runner.run_benchmark
    ~benchmark_name:"radix_sort"
    ~config
    ~run_fn:run_radix_sort_benchmark
