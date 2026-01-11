(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: Prefix Sum (Inclusive Scan)
 *
 * Priority: P1 (Sprint 2)
 * Pattern: Parallel scan/prefix sum using Hillis-Steele algorithm
 * Category: Fundamental parallel primitive
 *
 * Description:
 * Implements inclusive prefix sum within a single block using shared memory.
 * The Hillis-Steele algorithm performs log(n) supersteps with distance-doubling.
 * This is a fundamental building block for many parallel algorithms.
 *
 * Performance Notes:
 * - Limited to single block (256 elements max)
 * - Uses shared memory for communication between threads
 * - 9 supersteps for 256 elements (log2(256) = 8, plus load)
 * - Memory bandwidth: dominated by shared memory access patterns
 * - For larger arrays, need hierarchical scan or tree-based approach
 *
 * Hardware Tested:
 * - Intel Arc GPU (OpenCL + Vulkan)
 * - Intel Core CPU (OpenCL)
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
open Benchmark_backends

(** Pure OCaml baseline: inclusive prefix sum *)
let cpu_inclusive_scan input n =
  let output = Array.make n 0l in
  if n > 0 then begin
    output.(0) <- input.(0) ;
    for i = 1 to n - 1 do
      output.(i) <- Int32.add output.(i - 1) input.(i)
    done
  end ;
  output

(** Sarek kernel: Inclusive scan using Hillis-Steele algorithm *)
let inclusive_scan_kernel =
  [%kernel
    fun (input : int32 vector) (output : int32 vector) (n : int32) ->
      let%shared (temp : int32) = 512l in
      let%shared (temp2 : int32) = 512l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      (* Load data into shared memory *)
      let%superstep load =
        if gid < n then temp.(tid) <- input.(gid) else temp.(tid) <- 0l
      in
      (* Hillis-Steele: distance 1 *)
      let%superstep step1 =
        let v = temp.(tid) in
        let add = if tid >= 1l then temp.(tid - 1l) else 0l in
        temp2.(tid) <- v + add
      in
      (* Distance 2 *)
      let%superstep step2 =
        let v = temp2.(tid) in
        let add = if tid >= 2l then temp2.(tid - 2l) else 0l in
        temp.(tid) <- v + add
      in
      (* Distance 4 *)
      let%superstep step4 =
        let v = temp.(tid) in
        let add = if tid >= 4l then temp.(tid - 4l) else 0l in
        temp2.(tid) <- v + add
      in
      (* Distance 8 *)
      let%superstep step8 =
        let v = temp2.(tid) in
        let add = if tid >= 8l then temp2.(tid - 8l) else 0l in
        temp.(tid) <- v + add
      in
      (* Distance 16 *)
      let%superstep step16 =
        let v = temp.(tid) in
        let add = if tid >= 16l then temp.(tid - 16l) else 0l in
        temp2.(tid) <- v + add
      in
      (* Distance 32 *)
      let%superstep step32 =
        let v = temp2.(tid) in
        let add = if tid >= 32l then temp2.(tid - 32l) else 0l in
        temp.(tid) <- v + add
      in
      (* Distance 64 *)
      let%superstep step64 =
        let v = temp.(tid) in
        let add = if tid >= 64l then temp.(tid - 64l) else 0l in
        temp2.(tid) <- v + add
      in
      (* Distance 128 *)
      let%superstep step128 =
        let v = temp2.(tid) in
        let add = if tid >= 128l then temp2.(tid - 128l) else 0l in
        temp.(tid) <- v + add
      in
      (* Store result *)
      if gid < n then output.(gid) <- temp.(tid)]

(** Run scan benchmark on specified device *)
let run_scan_benchmark device backend_name size =
  (* Limit to 256 elements (single block) *)
  let n = min size 256 in

  let _, kirc = inclusive_scan_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test data: array [1, 2, 3, ..., n] *)
  let input = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Vector.set input i (Int32.of_int (i + 1))
  done ;

  (* Launch configuration: single block *)
  let block_size = 256 in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d 1 in

  (* Warmup run *)
  Sarek.Execute.run_vectors
    ~device
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec output;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    ~block
    ~grid
    () ;
  Transfer.flush device ;

  (* Timed runs *)
  let num_runs = 100 in
  let times = ref [] in

  for _ = 1 to num_runs do
    let t0 = Unix.gettimeofday () in
    Sarek.Execute.run_vectors
      ~device
      ~ir
      ~args:
        [
          Sarek.Execute.Vec input;
          Sarek.Execute.Vec output;
          Sarek.Execute.Int32 (Int32.of_int n);
        ]
      ~block
      ~grid
      () ;
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
  let gpu_result = Vector.to_array output in
  let input_arr = Array.init n (fun i -> Int32.of_int (i + 1)) in
  let cpu_result = cpu_inclusive_scan input_arr n in

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

  (* Report results *)
  Printf.printf "\n=== Prefix Sum (Scan) Benchmark ===\n" ;
  Printf.printf "Backend: %s\n" backend_name ;
  Printf.printf "Array size: %d elements\n" n ;
  Printf.printf "Block size: %d\n" block_size ;
  Printf.printf "Supersteps: 9 (1 load + 8 scan)\n" ;
  Printf.printf "\nTiming (%d runs):\n" num_runs ;
  Printf.printf "  Min:    %.3f ms\n" (min_time *. 1000.0) ;
  Printf.printf "  Median: %.3f ms\n" (median_time *. 1000.0) ;
  Printf.printf "  Mean:   %.3f ms\n" (avg_time *. 1000.0) ;

  let throughput_melems = float_of_int n /. (median_time *. 1e6) in
  Printf.printf "  Throughput: %.2f M elements/s\n" throughput_melems ;

  Printf.printf "\nVerification: " ;
  if !errors = 0 then Printf.printf "PASS ✓ (all %d values correct)\n" n
  else Printf.printf "FAIL ✗ (%d/%d errors)\n" !errors n ;

  Printf.printf "Expected sum at position %d: %ld\n" (n - 1) cpu_result.(n - 1) ;
  Printf.printf "GPU result at position %d: %ld\n" (n - 1) gpu_result.(n - 1) ;
  Printf.printf "====================================\n" ;

  !errors = 0

(** Main benchmark runner *)
let () =
  Printf.printf "Prefix Sum (Inclusive Scan) Benchmark\n" ;
  Printf.printf "Single-block Hillis-Steele algorithm\n\n" ;

  let size =
    if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 256
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

  let success = run_scan_benchmark device backend_name size in
  exit (if success then 0 else 1)
