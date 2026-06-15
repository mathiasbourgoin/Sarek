(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: 2D Jacobi / 5-Point Stencil
 *
 * Priority: P1 (Sprint 3)
 * Pattern: 5-point stencil computation (up, down, left, right, center)
 * Category: Scientific Computing / Stencil / PDE Solver
 *
 * Description:
 * Implements 2D Jacobi iteration for heat diffusion / Laplace equation.
 * Each cell is updated as the average of its 4 neighbors (5-point stencil).
 * Classic stencil pattern for PDE solvers and image processing.
 *
 * Performance Notes:
 * - Memory bandwidth bound: 5 reads, 1 write per cell
 * - Grid size N×N, processes N² cells
 * - Can be optimized with shared memory tiling
 * - Arithmetic intensity: 4 adds + 1 div = 5 FLOPs, 6 memory ops
 * - Performance: cells/sec, GB/s
 *
 * Hardware Tested:
 * - Intel Arc GPU (OpenCL + Vulkan)
 * - Intel Core CPU (OpenCL)
 ******************************************************************************)

[@@@warning "-33"] (* Suppress unused-open warnings from PPX *)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Std = Sarek_stdlib.Std
open Benchmark_common
open Benchmark_runner

(** Pure OCaml baseline: 2D Jacobi stencil *)
let cpu_stencil_2d input output width height =
  (* Interior points only, boundaries unchanged *)
  for y = 1 to height - 2 do
    for x = 1 to width - 2 do
      let idx = (y * width) + x in
      let up = input.(((y - 1) * width) + x) in
      let down = input.(((y + 1) * width) + x) in
      let left = input.((y * width) + x - 1) in
      let right = input.((y * width) + x + 1) in
      let center = input.(idx) in
      output.(idx) <- (up +. down +. left +. right +. center) /. 5.0
    done
  done

(** Sarek kernel: 2D 5-point stencil *)
let stencil_2d_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      let x = thread_idx_x + (block_idx_x * block_dim_x) in
      let y = thread_idx_y + (block_idx_y * block_dim_y) in
      if x > 0 && x < width - 1 && y > 0 && y < height - 1 then
        let idx = (y * width) + x in
        let up = input.(((y - 1) * width) + x) in
        let down = input.(((y + 1) * width) + x) in
        let left = input.((y * width) + x - 1) in
        let right = input.((y * width) + x + 1) in
        let center = input.(idx) in
        output.(idx) <- (up +. down +. left +. right +. center) /. 5.0]

(** Run 2D stencil benchmark on specified device *)
let run_stencil_2d_benchmark ~device ~size ~config =
  let backend_name = device.Device.framework in
  (* Use square grid *)
  let width = size in
  let height = size in
  let n = width * height in

  let _, kirc = stencil_2d_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test data: sin wave pattern *)
  let input_data =
    Array.init n (fun i ->
        let x = i mod width in
        let y = i / width in
        sin (float_of_int x *. 0.1) *. cos (float_of_int y *. 0.1))
  in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set input i input_data.(i) ;
    Vector.set output i 0.0
  done ;

  (* Launch configuration: 2D grid *)
  let block_x = 16 in
  let block_y = 16 in
  let grid_x = (width + block_x - 1) / block_x in
  let grid_y = (height + block_y - 1) / block_y in
  let block = Sarek.Execute.dims2d block_x block_y in
  let grid = Sarek.Execute.dims2d grid_x grid_y in

  (* Warmup runs *)
  for _ = 1 to config.warmup do
    Sarek.Execute.run_vectors
      ~device
      ~ir
      ~args:
        [
          Sarek.Execute.Vec input;
          Sarek.Execute.Vec output;
          Sarek.Execute.Int32 (Int32.of_int width);
          Sarek.Execute.Int32 (Int32.of_int height);
        ]
      ~block
      ~grid
      ()
  done ;
  Transfer.flush device ;

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
          Sarek.Execute.Vec output;
          Sarek.Execute.Int32 (Int32.of_int width);
          Sarek.Execute.Int32 (Int32.of_int height);
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
  let cpu_result = Array.copy input_data in
  cpu_stencil_2d input_data cpu_result width height ;

  let errors = ref 0 in
  let tolerance = 0.0001 in
  for y = 1 to height - 2 do
    for x = 1 to width - 2 do
      let idx = (y * width) + x in
      let diff = abs_float (gpu_result.(idx) -. cpu_result.(idx)) in
      if diff > tolerance then incr errors
    done
  done ;

  let verified = !errors = 0 in

  (* Compute statistics *)
  let times_array = Array.of_list (List.rev !times) in
  let median_ms = Common.median times_array in

  (* Calculate throughput: M cells/s *)
  let throughput_mcells = float_of_int n /. (median_ms /. 1000.0) /. 1e6 in

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
      throughput = Some throughput_mcells;
      verified = Some verified;
    }

(** Main benchmark runner *)
let () =
  let config =
    Benchmark_runner.parse_args
      ~benchmark_name:"stencil_2d"
      ~default_sizes:[256; 512; 1024; 2048]
      ()
  in
  Benchmark_runner.run_benchmark
    ~benchmark_name:"stencil_2d"
    ~config
    ~run_fn:run_stencil_2d_benchmark
