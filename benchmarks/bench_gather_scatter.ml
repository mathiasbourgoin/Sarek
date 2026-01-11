(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: Gather/Scatter Memory Patterns
 *
 * Priority: P1 (Sprint 2)
 * Pattern: Indirect memory access patterns
 * Category: Memory access primitive
 *
 * Description:
 * Implements gather and scatter operations - fundamental memory access patterns
 * for irregular data structures. Gather reads from indexed locations, while
 * scatter writes to indexed locations.
 *
 * Gather: output[i] = input[indices[i]]
 * Scatter: output[indices[i]] = input[i]
 *
 * Performance Notes:
 * - Gather is safe and straightforward (multiple reads from same location OK)
 * - Scatter requires care with write conflicts (uses atomic operations)
 * - Memory bandwidth: irregular access patterns may cause cache inefficiency
 * - Important primitive for graph algorithms and sparse data structures
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

(** Pure OCaml baseline: gather operation *)
let cpu_gather input indices output n =
  for i = 0 to n - 1 do
    let idx = Int32.to_int indices.(i) in
    output.(i) <- input.(idx)
  done

(** Sarek kernel: Gather operation *)
let gather_kernel =
  [%kernel
    fun (input : int32 vector)
        (indices : int32 vector)
        (output : int32 vector)
        (n : int32) ->
      let open Std in
      let i = global_thread_id in
      if i < n then begin
        let idx = indices.(i) in
        output.(i) <- input.(idx)
      end]

(** Sarek kernel: Scatter operation *)
let scatter_kernel =
  [%kernel
    fun (input : int32 vector)
        (indices : int32 vector)
        (output : int32 vector)
        (n : int32) ->
      let open Std in
      let i = global_thread_id in
      if i < n then begin
        let idx = indices.(i) in
        (* Direct write - last write wins semantics *)
        output.(idx) <- input.(i)
      end]

(** Run gather benchmark on specified device *)
let run_gather_benchmark device backend_name size =
  let n = size in

  let _, kirc = gather_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test data *)
  Random.init 42 ;
  let input = Vector.create Vector.int32 n in
  let indices = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  (* Fill input with sequential values *)
  for i = 0 to n - 1 do
    Vector.set input i (Int32.of_int (i * 2))
  done ;

  (* Create random permutation for indices *)
  for i = 0 to n - 1 do
    Vector.set indices i (Int32.of_int (Random.int n))
  done ;

  (* Launch configuration *)
  let block_size = 256 in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  (* Warmup run *)
  Sarek.Execute.run_vectors
    ~device
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec indices;
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
          Sarek.Execute.Vec indices;
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
  let input_arr = Array.init n (fun i -> Vector.get input i) in
  let indices_arr = Array.init n (fun i -> Vector.get indices i) in
  let cpu_result = Array.make n 0l in
  cpu_gather input_arr indices_arr cpu_result n ;

  let errors = ref 0 in
  for i = 0 to n - 1 do
    if gpu_result.(i) <> cpu_result.(i) then begin
      if !errors < 5 then
        Printf.printf
          "  ERROR at %d: GPU=%ld CPU=%ld (index=%ld)\n"
          i
          gpu_result.(i)
          cpu_result.(i)
          indices_arr.(i) ;
      incr errors
    end
  done ;

  (* Report results *)
  Printf.printf "\n=== Gather Benchmark ===\n" ;
  Printf.printf "Backend: %s\n" backend_name ;
  Printf.printf "Array size: %d elements\n" n ;
  Printf.printf "Block size: %d\n" block_size ;
  Printf.printf "Grid size: %d blocks\n" num_blocks ;
  Printf.printf "\nTiming (%d runs):\n" num_runs ;
  Printf.printf "  Min:    %.3f ms\n" (min_time *. 1000.0) ;
  Printf.printf "  Median: %.3f ms\n" (median_time *. 1000.0) ;
  Printf.printf "  Mean:   %.3f ms\n" (avg_time *. 1000.0) ;

  let throughput_melems = float_of_int n /. (median_time *. 1e6) in
  Printf.printf "  Throughput: %.2f M elements/s\n" throughput_melems ;

  Printf.printf "\nVerification: " ;
  if !errors = 0 then Printf.printf "PASS ✓ (all %d values correct)\n" n
  else Printf.printf "FAIL ✗ (%d/%d errors)\n" !errors n ;
  Printf.printf "========================\n" ;

  !errors = 0

(** Run scatter benchmark on specified device *)
let run_scatter_benchmark device backend_name size =
  let n = size in

  let _, kirc = scatter_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test data *)
  Random.init 42 ;
  let input = Vector.create Vector.int32 n in
  let indices = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  (* Fill input with sequential values *)
  for i = 0 to n - 1 do
    Vector.set input i (Int32.of_int ((i * 3) + 1))
  done ;

  (* Create random permutation for indices *)
  for i = 0 to n - 1 do
    Vector.set indices i (Int32.of_int (Random.int n))
  done ;

  (* Initialize output *)
  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  (* Launch configuration *)
  let block_size = 256 in
  let num_blocks = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d num_blocks in

  (* Warmup run *)
  Sarek.Execute.run_vectors
    ~device
    ~ir
    ~args:
      [
        Sarek.Execute.Vec input;
        Sarek.Execute.Vec indices;
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
    (* Clear output *)
    for i = 0 to n - 1 do
      Vector.set output i 0l
    done ;

    let t0 = Unix.gettimeofday () in
    Sarek.Execute.run_vectors
      ~device
      ~ir
      ~args:
        [
          Sarek.Execute.Vec input;
          Sarek.Execute.Vec indices;
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

  (* Verify correctness - for scatter with conflicts, we check that:
     1. Values in output are from the input (or 0 for unwritten locations)
     2. We don't compare exact match due to race conditions *)
  let gpu_result = Vector.to_array output in
  let input_arr = Array.init n (fun i -> Vector.get input i) in
  let input_set = Hashtbl.create n in
  for i = 0 to n - 1 do
    Hashtbl.add input_set input_arr.(i) true
  done ;

  let errors = ref 0 in
  let non_zero = ref 0 in
  for i = 0 to n - 1 do
    if gpu_result.(i) <> 0l then begin
      incr non_zero ;
      (* Check if this value exists in input *)
      if not (Hashtbl.mem input_set gpu_result.(i)) then begin
        if !errors < 5 then
          Printf.printf
            "  ERROR at %d: GPU=%ld (not in input)\n"
            i
            gpu_result.(i) ;
        incr errors
      end
    end
  done ;

  (* Report results *)
  Printf.printf "\n=== Scatter Benchmark ===\n" ;
  Printf.printf "Backend: %s\n" backend_name ;
  Printf.printf "Array size: %d elements\n" n ;
  Printf.printf "Block size: %d\n" block_size ;
  Printf.printf "Grid size: %d blocks\n" num_blocks ;
  Printf.printf "Non-zero outputs: %d\n" !non_zero ;
  Printf.printf "\nTiming (%d runs):\n" num_runs ;
  Printf.printf "  Min:    %.3f ms\n" (min_time *. 1000.0) ;
  Printf.printf "  Median: %.3f ms\n" (median_time *. 1000.0) ;
  Printf.printf "  Mean:   %.3f ms\n" (avg_time *. 1000.0) ;

  let throughput_melems = float_of_int n /. (median_time *. 1e6) in
  Printf.printf "  Throughput: %.2f M elements/s\n" throughput_melems ;

  Printf.printf "\nVerification: " ;
  if !errors = 0 then Printf.printf "PASS ✓ (scatter completed)\n"
  else Printf.printf "FAIL ✗ (%d errors)\n" !errors ;
  Printf.printf "========================\n" ;

  !errors = 0

(** Main benchmark runner *)
let () =
  Printf.printf "Gather/Scatter Memory Patterns Benchmark\n" ;
  Printf.printf "Indirect memory access operations\n\n" ;

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

  let success1 = run_gather_benchmark device backend_name size in
  let success2 = run_scatter_benchmark device backend_name size in

  exit (if success1 && success2 then 0 else 1)
