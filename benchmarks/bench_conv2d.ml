(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: 2D Convolution
 *
 * Priority: P1 (Sprint 3)
 * Pattern: 2D image filtering with 3×3 kernel
 * Category: Image Processing / Stencil
 *
 * Description:
 * Implements 2D convolution for image filtering using 3×3 kernel.
 * Common operation in image processing, computer vision, and CNNs.
 * Tests memory access patterns and arithmetic intensity.
 *
 * Performance Notes:
 * - Memory: 9 reads + 1 write per pixel for 3×3 kernel
 * - Arithmetic: 9 multiplies + 8 adds = 17 FLOPs per pixel
 * - Can be optimized with shared memory tiling
 * - Separable kernels can reduce to 2*(3 reads + 3 muls + 2 adds)
 * - Performance: pixels/sec, GB/s
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

(** Pure OCaml baseline: 2D convolution with 3×3 kernel *)
let cpu_conv2d input output width height kernel =
  (* Interior points only, boundaries set to 0 *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let idx = (y * width) + x in
      if x = 0 || x = width - 1 || y = 0 || y = height - 1 then
        output.(idx) <- 0.0
      else
        let sum = ref 0.0 in
        for ky = 0 to 2 do
          for kx = 0 to 2 do
            let px = x + kx - 1 in
            let py = y + ky - 1 in
            let pidx = (py * width) + px in
            let kidx = (ky * 3) + kx in
            sum := !sum +. (input.(pidx) *. kernel.(kidx))
          done
        done ;
        output.(idx) <- !sum
    done
  done

(** Sarek kernel: 2D convolution with hardcoded 3×3 box blur *)
let conv2d_kernel =
  [%kernel
    fun (input : float32 vector)
        (output : float32 vector)
        (width : int32)
        (height : int32) ->
      let open Std in
      let tid = global_thread_id in
      let n = width * height in
      if tid < n then
        let x = tid mod width in
        let y = tid / width in
        if x = 0 || x = width - 1 || y = 0 || y = height - 1 then
          output.(tid) <- 0.0
        else
          (* Unroll 3×3 box blur (all coefficients = 1/9) *)
          let p00 = input.(((y - 1) * width) + x - 1) in
          let p01 = input.(((y - 1) * width) + x) in
          let p02 = input.(((y - 1) * width) + x + 1) in
          let p10 = input.((y * width) + x - 1) in
          let p11 = input.((y * width) + x) in
          let p12 = input.((y * width) + x + 1) in
          let p20 = input.(((y + 1) * width) + x - 1) in
          let p21 = input.(((y + 1) * width) + x) in
          let p22 = input.(((y + 1) * width) + x + 1) in
          let sum =
            p00 +. p01 +. p02 +. p10 +. p11 +. p12 +. p20 +. p21 +. p22
          in
          output.(tid) <- sum /. 9.0]

(** Run 2D convolution benchmark on specified device *)
let run_conv2d_benchmark ~device ~size ~config =
  let backend_name = device.Device.framework in
  (* Use square image *)
  let width = size in
  let height = size in
  let n = width * height in

  let _, kirc = conv2d_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test image: gradient pattern *)
  let input_data =
    Array.init n (fun i ->
        let x = i mod width in
        let y = i / width in
        float_of_int ((x + y) mod 256) /. 255.0)
  in

  (* Box blur kernel (normalized) *)
  let kernel_data = Array.make 9 (1.0 /. 9.0) in

  let input = Vector.create Vector.float32 n in
  let output = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set input i input_data.(i) ;
    Vector.set output i 0.0
  done ;

  (* Launch configuration: 1D grid *)
  let block_size = 256 in
  let grid_size = (n + block_size - 1) / block_size in
  let block = Sarek.Execute.dims1d block_size in
  let grid = Sarek.Execute.dims1d grid_size in

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
  let cpu_result = Array.make n 0.0 in
  cpu_conv2d input_data cpu_result width height kernel_data ;

  let errors = ref 0 in
  let tolerance = 0.001 in
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

  (* Calculate throughput: M pixels/s *)
  let throughput_mpixels = float_of_int n /. (median_ms /. 1000.0) /. 1e6 in

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
      throughput = Some throughput_mpixels;
      verified = Some verified;
    }

(** Main benchmark runner *)
let () =
  let config =
    Benchmark_runner.parse_args
      ~benchmark_name:"conv2d"
      ~default_sizes:[256; 512; 1024; 2048]
      ()
  in
  Benchmark_runner.run_benchmark
    ~benchmark_name:"conv2d"
    ~config
    ~run_fn:run_conv2d_benchmark
