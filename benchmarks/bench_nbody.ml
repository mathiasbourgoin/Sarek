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

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
open Benchmark_backends

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
let run_nbody_benchmark device backend_name size =
  let n = size in

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

  (* Warmup run *)
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
    then begin
      if !errors < 5 then
        Printf.printf
          "  ERROR at %d: GPU (%f,%f,%f) CPU (%f,%f,%f)\n"
          i
          gpu_ax.(i)
          gpu_ay.(i)
          gpu_az.(i)
          cpu_ax.(i)
          cpu_ay.(i)
          cpu_az.(i) ;
      incr errors
    end
  done ;

  (* Report results *)
  Printf.printf "\n=== N-Body Simulation Benchmark ===\n" ;
  Printf.printf "Backend: %s\n" backend_name ;
  Printf.printf "Particles: %d\n" n ;
  Printf.printf "Interactions: %d (N²)\n" (n * n) ;
  Printf.printf "Block size: %d\n" block_size ;
  Printf.printf "Grid size: %d blocks\n" grid_size ;
  Printf.printf "\nTiming (%d runs):\n" num_runs ;
  Printf.printf "  Min:    %.3f ms\n" (min_time *. 1000.0) ;
  Printf.printf "  Median: %.3f ms\n" (median_time *. 1000.0) ;
  Printf.printf "  Mean:   %.3f ms\n" (avg_time *. 1000.0) ;

  (* Calculate GFLOPS: ~20 FLOPs per interaction *)
  let flops = float_of_int (n * n) *. 20.0 in
  let gflops = flops /. (median_time *. 1e9) in
  Printf.printf "  Performance: %.2f GFLOPS\n" gflops ;

  let interactions_per_sec = float_of_int (n * n) /. median_time /. 1e9 in
  Printf.printf "  Interactions: %.2f G/s\n" interactions_per_sec ;

  Printf.printf "\nVerification: " ;
  if !errors = 0 then Printf.printf "PASS ✓ (all %d particles correct)\n" n
  else Printf.printf "FAIL ✗ (%d/%d errors)\n" !errors n ;

  Printf.printf
    "Sample forces (particle 0): ax=%.6f ay=%.6f az=%.6f\n"
    gpu_ax.(0)
    gpu_ay.(0)
    gpu_az.(0) ;
  Printf.printf "===================================\n" ;

  !errors = 0

(** Main benchmark runner *)
let () =
  Printf.printf "N-Body Simulation Benchmark\n" ;
  Printf.printf "All-pairs gravitational forces (O(N²))\n\n" ;

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

  let success = run_nbody_benchmark device backend_name size in
  exit (if success then 0 else 1)
