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
open Benchmark_common
open Benchmark_runner

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
[@@warning "-33"]

(** Run bitonic sort benchmark on specified device *)
let run_bitonic_sort_benchmark ~device ~size ~config =
  let backend_name = device.Device.framework in
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

  (* Warmup *)
  for _ = 1 to config.warmup do
    Random.init 42 ;
    for i = 0 to n - 1 do
      Vector.set input i (Int32.of_int (Random.int 10000))
    done ;
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
    Transfer.flush device
  done ;

  (* Timed runs *)
  let times = ref [] in
  for _ = 1 to config.iterations do
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
    times := ((t1 -. t0) *. 1000.0) :: !times
  done ;

  (* Verify correctness *)
  let gpu_result = Vector.to_array input in
  Random.init 42 ;
  let input_arr = Array.init n (fun _ -> Int32.of_int (Random.int 10000)) in
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
      ~benchmark_name:"bitonic_sort"
      ~default_sizes:[1024; 4096; 16384]
      ()
  in
  Benchmark_runner.run_benchmark
    ~benchmark_name:"bitonic_sort"
    ~config
    ~run_fn:run_bitonic_sort_benchmark
