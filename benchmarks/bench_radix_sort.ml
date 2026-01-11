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
 * Implements radix sort using 4-bit digits (16 bins per pass). Radix sort
 * processes integers digit-by-digit from least to most significant, using
 * counting sort for each digit. This implementation requires 8 passes for
 * 32-bit integers (32 bits / 4 bits per pass).
 *
 * Performance Notes:
 * - Complexity: O(k*n) where k is number of passes
 * - Uses histogram and prefix sum as building blocks
 * - Stable sort: preserves relative order of equal elements
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
open Benchmark_backends

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
      let%shared (local_hist : int32) = 16l in
      let tid = thread_idx_x in
      let gid = global_thread_id in
      let num_bins = 16l in
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

(** Sarek kernel: Scatter elements based on prefix sum *)
let radix_scatter_kernel =
  [%kernel
    fun (input : int32 vector)
        (output : int32 vector)
        (prefix_sum : int32 vector)
        (n : int32)
        (shift : int32)
        (mask : int32) ->
      let open Std in
      let open Gpu in
      let gid = global_thread_id in
      if gid < n then begin
        let value = input.(gid) in
        let digit = (value lsr shift) land mask in
        (* Atomically get and increment position *)
        let pos = atomic_add_global_int32 prefix_sum digit 1l in
        output.(pos) <- value
      end]

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
let run_radix_sort_benchmark device backend_name size =
  let n = size in
  let bits_per_pass = 4 in
  let num_bins = 1 lsl bits_per_pass in
  let num_passes = 32 / bits_per_pass in
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

  (* Create test data: random values *)
  Random.init 42 ;
  let input = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set input i (Int32.of_int (Random.int 100000))
  done ;

  (* Launch configuration *)
  let block_size = 256 in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  (* Warmup run - single pass *)
  let histogram = Vector.create Vector.int32 num_bins in
  for i = 0 to num_bins - 1 do
    Vector.set histogram i 0l
  done ;
  Sarek.Execute.run_vectors
    ~device
    ~ir:hist_ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec histogram;
        Sarek.Execute.Int32 (Int32.of_int n);
        Sarek.Execute.Int32 0l;
        Sarek.Execute.Int32 mask;
      ]
    ~block
    ~grid
    () ;
  Transfer.flush device ;

  (* Timed runs *)
  let num_runs = 100 in
  let times = ref [] in

  for _ = 1 to num_runs do
    (* Reset input *)
    Random.init 42 ;
    for i = 0 to n - 1 do
      Vector.set input i (Int32.of_int (Random.int 100000))
    done ;

    let t0 = Unix.gettimeofday () in

    (* Perform all radix passes *)
    let current_input = ref input in
    let current_output = ref output in

    for pass = 0 to num_passes - 1 do
      let shift = Int32.of_int (pass * bits_per_pass) in

      (* Create fresh histogram for this pass *)
      let histogram = Vector.create Vector.int32 num_bins in
      for i = 0 to num_bins - 1 do
        Vector.set histogram i 0l
      done ;

      (* Compute histogram *)
      Sarek.Execute.run_vectors
        ~device
        ~ir:hist_ir
        ~args:
          [
            Sarek.Execute.Vec !current_input;
            Sarek.Execute.Vec histogram;
            Sarek.Execute.Int32 (Int32.of_int n);
            Sarek.Execute.Int32 shift;
            Sarek.Execute.Int32 mask;
          ]
        ~block
        ~grid
        () ;
      Transfer.flush device ;

      (* Compute prefix sum on CPU and create fresh prefix_sum vector *)
      let hist_arr = Vector.to_array histogram in
      let prefix_arr = cpu_prefix_sum hist_arr num_bins in
      let prefix_sum = Vector.create Vector.int32 num_bins in
      for i = 0 to num_bins - 1 do
        Vector.set prefix_sum i prefix_arr.(i)
      done ;

      (* Scatter elements *)
      Sarek.Execute.run_vectors
        ~device
        ~ir:scatter_ir
        ~args:
          [
            Sarek.Execute.Vec !current_input;
            Sarek.Execute.Vec !current_output;
            Sarek.Execute.Vec prefix_sum;
            Sarek.Execute.Int32 (Int32.of_int n);
            Sarek.Execute.Int32 shift;
            Sarek.Execute.Int32 mask;
          ]
        ~block
        ~grid
        () ;
      Transfer.flush device ;

      (* Swap buffers *)
      let temp = !current_input in
      current_input := !current_output ;
      current_output := temp
    done ;

    let t1 = Unix.gettimeofday () in
    times := (t1 -. t0) :: !times
  done ;

  (* Compute statistics *)
  let sorted_times = List.sort compare !times in
  let avg_time =
    List.fold_left ( +. ) 0.0 sorted_times /. float_of_int num_runs
  in
  let min_time = List.hd sorted_times in
  let median_time = List.nth sorted_times (num_runs / 2) in

  (* Verify correctness *)
  (* After even number of passes, result is in input buffer *)
  let final_buffer = if num_passes mod 2 = 0 then output else input in
  let gpu_result = Vector.to_array final_buffer in

  Random.init 42 ;
  let input_arr = Array.init n (fun _ -> Int32.of_int (Random.int 100000)) in
  let cpu_result = cpu_radix_sort input_arr n in

  let errors = ref 0 in
  for i = 0 to n - 1 do
    if gpu_result.(i) <> cpu_result.(i) then begin
      if !errors < 5 then
        Printf.printf
          "  ERROR at %d: GPU=%ld CPU=%ld\n"
          i
          gpu_result.(i)
          cpu_result.(i) ;
      incr errors
    end
  done ;

  (* Check if sorted *)
  let is_sorted = ref true in
  for i = 1 to n - 1 do
    if gpu_result.(i) < gpu_result.(i - 1) then is_sorted := false
  done ;

  (* Report results *)
  Printf.printf "\n=== Radix Sort Benchmark ===\n" ;
  Printf.printf "Backend: %s\n" backend_name ;
  Printf.printf "Array size: %d elements\n" n ;
  Printf.printf "Block size: %d\n" block_size ;
  Printf.printf "Grid size: %d blocks\n" num_blocks ;
  Printf.printf "Radix passes: %d (%d bits per pass)\n" num_passes bits_per_pass ;
  Printf.printf "Bins per pass: %d\n" num_bins ;
  Printf.printf "\nTiming (%d runs):\n" num_runs ;
  Printf.printf "  Min:    %.3f ms\n" (min_time *. 1000.0) ;
  Printf.printf "  Median: %.3f ms\n" (median_time *. 1000.0) ;
  Printf.printf "  Mean:   %.3f ms\n" (avg_time *. 1000.0) ;

  let throughput_melems = float_of_int n /. (median_time *. 1e6) in
  Printf.printf "  Throughput: %.2f M elements/s\n" throughput_melems ;

  Printf.printf "\nVerification: " ;
  if !errors = 0 && !is_sorted then Printf.printf "PASS ✓ (sorted correctly)\n"
  else Printf.printf "FAIL ✗ (%d errors, sorted=%b)\n" !errors !is_sorted ;

  if n <= 16 then begin
    Printf.printf "GPU result: [" ;
    for i = 0 to n - 1 do
      Printf.printf "%ld%s" gpu_result.(i) (if i < n - 1 then "; " else "")
    done ;
    Printf.printf "]\n"
  end
  else begin
    Printf.printf "First 8 GPU results: [" ;
    for i = 0 to 7 do
      Printf.printf "%ld%s" gpu_result.(i) (if i < 7 then "; " else "")
    done ;
    Printf.printf "]\n"
  end ;
  Printf.printf "============================\n" ;

  !errors = 0 && !is_sorted

(** Main benchmark runner *)
let () =
  Printf.printf "Radix Sort Benchmark\n" ;
  Printf.printf "4-bit radix sort for 32-bit integers\n\n" ;

  let size =
    if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 4096
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

  let success = run_radix_sort_benchmark device backend_name size in
  exit (if success then 0 else 1)
