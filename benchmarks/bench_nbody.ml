(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Benchmark: N-Body Simulation (Naive)
 *
 * Priority: P1 (Sprint 3)
 * Pattern: All-pairs gravitational force calculation
 * Category: Scientific Computing / Physics
 *
 * Description:
 * Implements naive N-body simulation computing all-pairs gravitational forces.
 * O(N²) complexity, embarrassingly parallel. Common HPC benchmark showing
 * high arithmetic intensity (many FLOPs per memory access).
 *
 * Performance Notes:
 * - O(N²) interactions
 * - High arithmetic intensity (~20 FLOPs per interaction)
 * - Embarrassingly parallel - each particle independent
 * - Memory: 6N floats input (x,y,z positions), 3N output (ax,ay,az forces)
 * - GFLOPS = (N² * 20) / time
 *
 * Hardware Tested:
 * - Intel Arc GPU (OpenCL + Vulkan)
 * - Intel Core CPU (OpenCL)
 ******************************************************************************)

[@@@warning "-33"] (* Suppress unused-open warnings from PPX *)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
open Benchmark_common
open Benchmark_runner

(** Pure OCaml baseline: N-body force calculation *)
let cpu_nbody xs ys zs n =
  let ax = Array.make n 0.0 in
  let ay = Array.make n 0.0 in
  let az = Array.make n 0.0 in
  for i = 0 to n - 1 do
    let px = xs.(i) and py = ys.(i) and pz = zs.(i) in
    let rfx = ref 0.0 and rfy = ref 0.0 and rfz = ref 0.0 in
    for j = 0 to n - 1 do
      if j <> i then (
        let qx = xs.(j) and qy = ys.(j) and qz = zs.(j) in
        let dx = qx -. px in
        let dy = qy -. py in
        let dz = qz -. pz in
        let dist2 = (dx *. dx) +. (dy *. dy) +. (dz *. dz) +. 1e-9 in
        let inv = 1.0 /. (sqrt dist2 *. dist2) in
        rfx := !rfx +. (dx *. inv) ;
        rfy := !rfy +. (dy *. inv) ;
        rfz := !rfz +. (dz *. inv))
    done ;
    ax.(i) <- !rfx ;
    ay.(i) <- !rfy ;
    az.(i) <- !rfz
  done ;
  (ax, ay, az)

(** Sarek kernel: N-body acceleration calculation *)
let accel_kernel =
  [%kernel
    let module Types = struct
      type particle = {x : float32; y : float32; z : float32}
    end in
    let make_p (x : float32) (y : float32) (z : float32) : particle =
      {x; y; z}
    in
    fun (xs : float32 vector)
        (ys : float32 vector)
        (zs : float32 vector)
        (ax : float32 vector)
        (ay : float32 vector)
        (az : float32 vector)
        (n : int32)
      ->
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then (
        let p = make_p xs.(tid) ys.(tid) zs.(tid) in
        let fx = mut 0.0 in
        let fy = mut 0.0 in
        let fz = mut 0.0 in
        let i = tid in
        for j = 0 to n - 1 do
          if j <> i then (
            let q = make_p xs.(j) ys.(j) zs.(j) in
            let dx = q.x -. p.x in
            let dy = q.y -. p.y in
            let dz = q.z -. p.z in
            let dist2 = (dx *. dx) +. (dy *. dy) +. (dz *. dz) +. 1e-9 in
            let inv = 1.0 /. (sqrt dist2 *. dist2) in
            fx := fx +. (dx *. inv) ;
            fy := fy +. (dy *. inv) ;
            fz := fz +. (dz *. inv))
        done ;
        ax.(tid) <- fx ;
        ay.(tid) <- fy ;
        az.(tid) <- fz)]

(** Run N-body benchmark on specified device *)
let run_nbody_benchmark ~device ~size ~config =
  let n = size in
  let backend_name = device.Device.framework in

  let _, kirc = accel_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  (* Create test data: simple particle distribution *)
  let input_xs = Array.init n (fun i -> float_of_int (i mod 7) /. 10.0) in
  let input_ys = Array.init n (fun i -> float_of_int (i mod 5) /. 10.0) in
  let input_zs = Array.init n (fun i -> float_of_int (i mod 3) /. 10.0) in

  let xs = Vector.create Vector.float32 n in
  let ys = Vector.create Vector.float32 n in
  let zs = Vector.create Vector.float32 n in
  let ax = Vector.create Vector.float32 n in
  let ay = Vector.create Vector.float32 n in
  let az = Vector.create Vector.float32 n in

  for i = 0 to n - 1 do
    Vector.set xs i input_xs.(i) ;
    Vector.set ys i input_ys.(i) ;
    Vector.set zs i input_zs.(i) ;
    Vector.set ax i 0.0 ;
    Vector.set ay i 0.0 ;
    Vector.set az i 0.0
  done ;

  (* Launch configuration *)
  let block_size = 64 in
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
          Sarek.Execute.Vec xs;
          Sarek.Execute.Vec ys;
          Sarek.Execute.Vec zs;
          Sarek.Execute.Vec ax;
          Sarek.Execute.Vec ay;
          Sarek.Execute.Vec az;
          Sarek.Execute.Int32 (Int32.of_int n);
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
          Sarek.Execute.Vec xs;
          Sarek.Execute.Vec ys;
          Sarek.Execute.Vec zs;
          Sarek.Execute.Vec ax;
          Sarek.Execute.Vec ay;
          Sarek.Execute.Vec az;
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
  let gpu_ax = Vector.to_array ax in
  let gpu_ay = Vector.to_array ay in
  let gpu_az = Vector.to_array az in
  let cpu_ax, cpu_ay, cpu_az = cpu_nbody input_xs input_ys input_zs n in

  let errors = ref 0 in
  for i = 0 to n - 1 do
    let check v1 v2 =
      let diff = abs_float (v1 -. v2) in
      if diff > 1e-3 && diff /. (abs_float v2 +. 1e-9) > 1e-3 then false
      else true
    in
    if
      not
        (check gpu_ax.(i) cpu_ax.(i)
        && check gpu_ay.(i) cpu_ay.(i)
        && check gpu_az.(i) cpu_az.(i))
    then incr errors
  done ;

  let verified = !errors = 0 in

  (* Compute statistics *)
  let times_array = Array.of_list (List.rev !times) in
  let median_ms = Common.median times_array in

  (* Calculate throughput: interactions per second *)
  let interactions = float_of_int (n * n) in
  let throughput_ginteractions = interactions /. (median_ms /. 1000.0) /. 1e9 in

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
      throughput = Some throughput_ginteractions;
      verified = Some verified;
    }

(** Main benchmark runner *)
let () =
  let config =
    Benchmark_runner.parse_args
      ~benchmark_name:"nbody"
      ~default_sizes:[512; 1024; 2048; 4096]
      ()
  in
  Benchmark_runner.run_benchmark
    ~benchmark_name:"nbody"
    ~config
    ~run_fn:run_nbody_benchmark
