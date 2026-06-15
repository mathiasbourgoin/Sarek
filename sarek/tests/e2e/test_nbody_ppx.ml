(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Simplified N-Body example using Sarek PPX with ktype + helper.
 * Uses let x = mut ... syntax for mutable accumulators.
 * Adapted for Benchmarks runner.
 ******************************************************************************)

(* Shadow SPOC's vector with runtime vector for kernel compatibility *)
type ('a, 'b) vector = ('a, 'b) Spoc_core.Vector.t

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

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

let cpu_nbody n =
  let bax = Array.init n (fun i -> float_of_int (i mod 7) /. 10.0) in
  let bay = Array.init n (fun i -> float_of_int (i mod 5) /. 10.0) in
  let baz = Array.init n (fun i -> float_of_int (i mod 3) /. 10.0) in
  let rax = Array.make n 0.0 in
  let ray = Array.make n 0.0 in
  let raz = Array.make n 0.0 in
  for i = 0 to n - 1 do
    let px = bax.(i) and py = bay.(i) and pz = baz.(i) in
    let rfx = ref 0.0 and rfy = ref 0.0 and rfz = ref 0.0 in
    for j = 0 to n - 1 do
      if j <> i then (
        let qx = bax.(j) and qy = bay.(j) and qz = baz.(j) in
        let dx = qx -. px in
        let dy = qy -. py in
        let dz = qz -. pz in
        let dist2 = (dx *. dx) +. (dy *. dy) +. (dz *. dz) +. 1e-9 in
        let inv = 1.0 /. (sqrt dist2 *. dist2) in
        rfx := !rfx +. (dx *. inv) ;
        rfy := !rfy +. (dy *. inv) ;
        rfz := !rfz +. (dz *. inv))
    done ;
    rax.(i) <- !rfx ;
    ray.(i) <- !rfy ;
    raz.(i) <- !rfz
  done ;
  (rax, ray, raz)

let verify (gax, gay, gaz) (rax, ray, raz) =
  let n = Array.length rax in
  let ok = ref true in
  for i = 0 to n - 1 do
    let check v1 v2 =
      let diff = abs_float (v1 -. v2) in
      if diff > 1e-3 && diff /. (abs_float v2 +. 1e-9) > 1e-3 then false
      else true
    in
    if
      not
        (check gax.(i) rax.(i) && check gay.(i) ray.(i) && check gaz.(i) raz.(i))
    then (
      ok := false ;
      Printf.printf
        "Mismatch at %d: GPU (%f,%f,%f) CPU (%f,%f,%f)\n%!"
        i
        gax.(i)
        gay.(i)
        gaz.(i)
        rax.(i)
        ray.(i)
        raz.(i))
  done ;
  !ok

let () =
  let _, kirc_kernel = accel_kernel in
  print_endline "=== NBody (PPX) IR ===" ;
  print_endline "======================" ;

  Benchmarks.run ~baseline:cpu_nbody ~verify "NBody PPX" (fun dev n _ ->
      match kirc_kernel.Sarek.Kirc_types.body_ir with
      | None ->
          print_endline "No IR available" ;
          (0.0, (Array.make n 0.0, Array.make n 0.0, Array.make n 0.0))
      | Some ir ->
          let threads = 64 in
          let grid_x = (n + threads - 1) / threads in

          let bax = Array.init n (fun i -> float_of_int (i mod 7) /. 10.0) in
          let bay = Array.init n (fun i -> float_of_int (i mod 5) /. 10.0) in
          let baz = Array.init n (fun i -> float_of_int (i mod 3) /. 10.0) in

          let xs = Vector.create Vector.float32 n in
          let ys = Vector.create Vector.float32 n in
          let zs = Vector.create Vector.float32 n in
          let ax = Vector.create Vector.float32 n in
          let ay = Vector.create Vector.float32 n in
          let az = Vector.create Vector.float32 n in

          for i = 0 to n - 1 do
            Vector.set xs i bax.(i) ;
            Vector.set ys i bay.(i) ;
            Vector.set zs i baz.(i) ;
            Vector.set ax i 0.0 ;
            Vector.set ay i 0.0 ;
            Vector.set az i 0.0
          done ;

          let block = Sarek.Execute.dims1d threads in
          let grid = Sarek.Execute.dims1d grid_x in

          let t0 = Unix.gettimeofday () in
          Sarek.Execute.run_vectors
            ~device:dev
            ~ir
            ~args:
              [
                Sarek.Execute.Vec xs;
                Sarek.Execute.Vec ys;
                Sarek.Execute.Vec zs;
                Sarek.Execute.Vec ax;
                Sarek.Execute.Vec ay;
                Sarek.Execute.Vec az;
                Sarek.Execute.Int n;
              ]
            ~block
            ~grid
            () ;
          Transfer.flush dev ;
          let t1 = Unix.gettimeofday () in

          let gax = Vector.to_array ax in
          let gay = Vector.to_array ay in
          let gaz = Vector.to_array az in
          ((t1 -. t0) *. 1000.0, (gax, gay, gaz))) ;
  Benchmarks.exit ()
