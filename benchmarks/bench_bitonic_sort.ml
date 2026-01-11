(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: Bitonic Sort
 *
 * Priority: P1 (Sprint 2)
 * Pattern: Parallel sorting with regular communication patterns
 * Category: Sorting algorithm
 *
 * Description:
 * Implements bitonic sort, a data-oblivious sorting algorithm well-suited
 * for parallel execution on GPUs. The algorithm performs O(log²(n)) passes
 * with predictable memory access patterns.
 *
 * Performance Notes:
 * - Data-oblivious: same comparisons regardless of input values
 * - Requires n to be a power of 2
 * - O(n log²(n)) comparisons
 * - Regular memory access patterns enable efficient GPU execution
 * - Each pass synchronizes all threads
 *
 * Hardware Tested:
 * - Intel Arc GPU (OpenCL + Vulkan)
 * - Intel Core CPU (OpenCL)
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std
open Benchmark_backends

(** Pure OCaml baseline: standard sort *)
let cpu_sort arr n =
  let a = Array.sub arr 0 n in
  Array.sort Int32.compare a ;
  a

(** Sarek kernel: Bitonic sort step - one comparison/swap pass *)
let bitonic_sort_step_kernel =
  [%kernel
    fun (data : int32 vector) (j : int32) (k : int32) (n : int32) ->
      let open Std in
      let i = global_thread_id in
      if i < n then begin
        let ij = i lxor j in
        if ij > i then begin
          let di = data.(i) in
          let dij = data.(ij) in
          let ascending = i land k = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            data.(i) <- dij ;
            data.(ij) <- di
          end
        end
      end]

(** Run bitonic sort benchmark on specified device *)
let run_bitonic_sort_benchmark device backend_name size =
  (* Round to nearest power of 2 *)
  let log2n = int_of_float (log (float_of_int size) /. log 2.0) in
  let n = 1 lsl log2n in

  let _, kirc = bitonic_sort_step_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test data: random values *)
  Random.init 42 ;
  let input = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    Vector.set input i (Int32.of_int (Random.int 10000))
  done ;

  (* Launch configuration *)
  let block_size = 256 in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  (* Warmup run *)
  for k_val = 2 to n do
    let k_val = k_val * 2 in
    if k_val > n then ()
    else
      for j_val = k_val / 2 downto 1 do
        Sarek.Execute.run_vectors
          ~device
          ~ir
          ~args:
            [
              Sarek.Execute.Vec input;
              Sarek.Execute.Int32 (Int32.of_int j_val);
              Sarek.Execute.Int32 (Int32.of_int k_val);
              Sarek.Execute.Int32 (Int32.of_int n);
            ]
          ~block
          ~grid
          ()
      done
  done ;
  Transfer.flush device ;

  (* Reset data for timed runs *)
  Random.init 42 ;
  for i = 0 to n - 1 do
    Vector.set input i (Int32.of_int (Random.int 10000))
  done ;

  (* Timed runs *)
  let num_runs = 100 in
  let times = ref [] in

  for _ = 1 to num_runs do
    (* Reset input data *)
    Random.init 42 ;
    for i = 0 to n - 1 do
      Vector.set input i (Int32.of_int (Random.int 10000))
    done ;

    let t0 = Unix.gettimeofday () in
    (* Perform full bitonic sort *)
    let rec outer k_val =
      if k_val > n then ()
      else begin
        let rec inner j_val =
          if j_val < 1 then ()
          else begin
            Sarek.Execute.run_vectors
              ~device
              ~ir
              ~args:
                [
                  Sarek.Execute.Vec input;
                  Sarek.Execute.Int32 (Int32.of_int j_val);
                  Sarek.Execute.Int32 (Int32.of_int k_val);
                  Sarek.Execute.Int32 (Int32.of_int n);
                ]
              ~block
              ~grid
              () ;
            inner (j_val / 2)
          end
        in
        inner (k_val / 2) ;
        outer (k_val * 2)
      end
    in
    outer 2 ;
    Transfer.flush device ;
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
  let gpu_result = Vector.to_array input in
  let input_arr = Array.init n (fun _ -> Int32.of_int (Random.int 10000)) in
  Random.init 42 ;
  for i = 0 to n - 1 do
    input_arr.(i) <- Int32.of_int (Random.int 10000)
  done ;
  let cpu_result = cpu_sort input_arr n in

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

  (* Count number of passes *)
  let num_passes = log2n * (log2n + 1) / 2 in

  (* Report results *)
  Printf.printf "\n=== Bitonic Sort Benchmark ===\n" ;
  Printf.printf "Backend: %s\n" backend_name ;
  Printf.printf "Array size: %d elements\n" n ;
  Printf.printf "Block size: %d\n" block_size ;
  Printf.printf "Grid size: %d blocks\n" num_blocks ;
  Printf.printf "Algorithm passes: %d\n" num_passes ;
  Printf.printf "\nTiming (%d runs):\n" num_runs ;
  Printf.printf "  Min:    %.3f ms\n" (min_time *. 1000.0) ;
  Printf.printf "  Median: %.3f ms\n" (median_time *. 1000.0) ;
  Printf.printf "  Mean:   %.3f ms\n" (avg_time *. 1000.0) ;

  let throughput_melems = float_of_int n /. (median_time *. 1e6) in
  Printf.printf "  Throughput: %.2f M elements/s\n" throughput_melems ;

  Printf.printf "\nVerification: " ;
  if !errors = 0 then Printf.printf "PASS ✓ (all %d values correct)\n" n
  else Printf.printf "FAIL ✗ (%d/%d errors)\n" !errors n ;

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
  Printf.printf "==============================\n" ;

  !errors = 0

(** Main benchmark runner *)
let () =
  Printf.printf "Bitonic Sort Benchmark\n" ;
  Printf.printf "Data-oblivious parallel sorting\n\n" ;

  let size =
    if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 1024
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

  let success = run_bitonic_sort_benchmark device backend_name size in
  exit (if success then 0 else 1)
