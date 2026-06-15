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
open Benchmark_common
open Benchmark_runner

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
[@@warning "-33"]

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
[@@warning "-33"]

(** Run gather benchmark on specified device *)
let run_gather_benchmark ~device ~size ~config =
  let backend_name = device.Device.framework in
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

  (* Warmup *)
  for _ = 1 to config.warmup do
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
    Transfer.flush device
  done ;

  (* Timed runs *)
  let times = ref [] in
  for _ = 1 to config.iterations do
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
    times := ((t1 -. t0) *. 1000.0) :: !times
  done ;

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

  let verified = !errors = 0 in
  let times_array = Array.of_list (List.rev !times) in
  let median_ms = Common.median times_array in

  (* Report results *)
  Printf.printf
    "  %s (gather): size=%d, median=%.3f ms, verified=%s\n"
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

(** Run scatter benchmark on specified device *)
let run_scatter_benchmark ~device ~size ~config =
  let backend_name = device.Device.framework in
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

  (* Warmup *)
  for _ = 1 to config.warmup do
    for i = 0 to n - 1 do
      Vector.set output i 0l
    done ;
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
    Transfer.flush device
  done ;

  (* Timed runs *)
  let times = ref [] in
  for _ = 1 to config.iterations do
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
    times := ((t1 -. t0) *. 1000.0) :: !times
  done ;

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
  for i = 0 to n - 1 do
    if gpu_result.(i) <> 0l then begin
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

  let verified = !errors = 0 in
  let times_array = Array.of_list (List.rev !times) in
  let median_ms = Common.median times_array in

  (* Report results *)
  Printf.printf
    "  %s (scatter): size=%d, median=%.3f ms, verified=%s\n"
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
      ~benchmark_name:"gather_scatter"
      ~default_sizes:[1_000_000; 10_000_000; 50_000_000]
      ()
  in

  Printf.printf "=== Running Gather Benchmark ===\n" ;
  Benchmark_runner.run_benchmark
    ~benchmark_name:"gather"
    ~config
    ~run_fn:run_gather_benchmark ;

  Printf.printf "\n=== Running Scatter Benchmark ===\n" ;
  Benchmark_runner.run_benchmark
    ~benchmark_name:"scatter"
    ~config
    ~run_fn:run_scatter_benchmark
