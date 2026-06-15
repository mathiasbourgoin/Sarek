(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Complex types
 *
 * Tests custom record types with both SPOC and runtime execution paths.
 * Uses [@@sarek.type] for runtime-compatible custom vector types.
 ******************************************************************************)

[@@@warning "-32"]

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

(* Backends auto-register when linked; Benchmarks.init() ensures initialization *)

let cfg = Test_helpers.default_config ()

type float32 = float

(* Type definitions with [@@sarek.type] for runtime support *)
type point2d = {px : float32; py : float32} [@@sarek.type]

type point3d = {x : float32; y : float32; z : float32} [@@sarek.type]

type particle = {
  pos_x : float32;
  pos_y : float32;
  vel_x : float32;
  vel_y : float32;
  mass : float32;
}
[@@sarek.type]

type color = {r : float32; g : float32; b : float32; a : float32} [@@sarek.type]

(* ============================================================================
   Point2D distance - uses point2d struct vector
   ============================================================================ *)

let point2d_distance_kirc =
  snd
    [%kernel
      fun (points : point2d vector) (distances : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = points.(tid) in
          distances.(tid) <- sqrt ((p.px *. p.px) +. (p.py *. p.py))
        end]

(* ============================================================================
   Point3D normalize - uses point3d struct vector
   ============================================================================ *)

let point3d_normalize_kirc =
  snd
    [%kernel
      fun (points : point3d vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = points.(tid) in
          let len = sqrt ((p.x *. p.x) +. (p.y *. p.y) +. (p.z *. p.z)) in
          if len > 0.0 then
            let inv = 1.0 /. len in
            points.(tid) <- {x = p.x *. inv; y = p.y *. inv; z = p.z *. inv}
        end]

(* ============================================================================
   Particle update - uses particle struct vector
   ============================================================================ *)

let particle_update_kirc =
  snd
    [%kernel
      fun (particles : particle vector) (dt : float32) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p = particles.(tid) in
          particles.(tid) <-
            {
              pos_x = p.pos_x +. (p.vel_x *. dt);
              pos_y = p.pos_y +. (p.vel_y *. dt);
              vel_x = p.vel_x;
              vel_y = p.vel_y;
              mass = p.mass;
            }
        end]

(* ============================================================================
   Color blend - uses color struct vector
   ============================================================================ *)

let color_blend_kirc =
  snd
    [%kernel
      fun (c1 : color vector)
          (c2 : color vector)
          (out : color vector)
          (t : float32)
          (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let p1 = c1.(tid) in
          let p2 = c2.(tid) in
          let inv_t = 1.0 -. t in
          out.(tid) <-
            {
              r = (p1.r *. inv_t) +. (p2.r *. t);
              g = (p1.g *. inv_t) +. (p2.g *. t);
              b = (p1.b *. inv_t) +. (p2.b *. t);
              a = (p1.a *. inv_t) +. (p2.a *. t);
            }
        end]

let run_point2d_test dev n _block_size =
  let points = Vector.create_custom point2d_custom n in
  let distances = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set points i {px = float_of_int i; py = float_of_int (n - i)} ;
    Vector.set distances i 0.0
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match point2d_distance_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:
      [
        Sarek.Execute.Vec points;
        Sarek.Execute.Vec distances;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, (points, distances))

let run_point3d_test dev n _block_size =
  let points = Vector.create_custom point3d_custom n in
  for i = 0 to n - 1 do
    let fi = float_of_int i in
    Vector.set points i {x = fi; y = fi *. 2.0; z = fi *. 3.0}
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match point3d_normalize_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:[Sarek.Execute.Vec points; Sarek.Execute.Int32 (Int32.of_int n)]
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, points)

let run_particle_test dev n _block_size =
  let particles = Vector.create_custom particle_custom n in
  let dt = 0.1 in
  for i = 0 to n - 1 do
    let fi = float_of_int i in
    Vector.set
      particles
      i
      {pos_x = fi; pos_y = fi *. 2.0; vel_x = 1.0; vel_y = 2.0; mass = 1.0}
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match particle_update_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:
      [
        Sarek.Execute.Vec particles;
        Sarek.Execute.Float32 dt;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, particles)

let run_color_test dev n _block_size =
  let c1 = Vector.create_custom color_custom n in
  let c2 = Vector.create_custom color_custom n in
  let output = Vector.create_custom color_custom n in
  let t = 0.5 in
  for i = 0 to n - 1 do
    Vector.set c1 i {r = 1.0; g = 0.0; b = 0.0; a = 1.0} ;
    Vector.set c2 i {r = 0.0; g = 1.0; b = 0.0; a = 1.0} ;
    Vector.set output i {r = 0.0; g = 0.0; b = 0.0; a = 0.0}
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let ir =
    match color_blend_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir
    ~args:
      [
        Sarek.Execute.Vec c1;
        Sarek.Execute.Vec c2;
        Sarek.Execute.Vec output;
        Sarek.Execute.Float32 t;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, output)

(* ============================================================================
   Main
   ============================================================================ *)

let () =
  Benchmarks.init () ;

  (* Point2D *)
  let verify_point2d (points, distances) _ =
    let n = Vector.length points in
    let ok = ref true in
    let error_count = ref 0 in
    let first_error_idx = ref (-1) in
    for i = 0 to n - 1 do
      let p = Vector.get points i in
      let expected = sqrt ((p.px *. p.px) +. (p.py *. p.py)) in
      let got = Vector.get distances i in
      let tol = max 1e-4 (abs_float expected *. 1e-6) in
      if abs_float (got -. expected) > tol then (
        ok := false ;
        incr error_count ;
        if !first_error_idx < 0 then first_error_idx := i ;
        if !error_count <= 3 then
          Printf.printf
            "    Mismatch at %d: got %f expected %f (p={%.1f,%.1f})\n%!"
            i
            got
            expected
            p.px
            p.py)
    done ;
    if !error_count > 0 then
      Printf.printf
        "    Total errors: %d/%d, first at index %d\n%!"
        !error_count
        n
        !first_error_idx ;
    !ok
  in
  (* Complex types are slow on interpreter (3-20s) - exclude it *)
  Benchmarks.run
    ~verify:verify_point2d
    ~filter:Benchmarks.no_interpreter
    "Point2D Distance"
    run_point2d_test ;

  (* Point3D *)
  let verify_point3d points _ =
    let n = Vector.length points in
    let ok = ref true in
    for i = 1 to min (n - 1) 10 do
      let p = Vector.get points i in
      let len = sqrt ((p.x *. p.x) +. (p.y *. p.y) +. (p.z *. p.z)) in
      if abs_float (len -. 1.0) > 1e-3 then (
        ok := false ;
        if i < 3 then
          Printf.printf "    Point %d: len=%.4f (expected 1.0)\n%!" i len)
    done ;
    !ok
  in
  Benchmarks.run
    ~verify:verify_point3d
    ~filter:Benchmarks.no_interpreter
    "Point3D Normalize"
    run_point3d_test ;

  (* Particle *)
  let verify_particle particles _ =
    let n = Vector.length particles in
    let dt = 0.1 in
    let ok = ref true in
    for i = 0 to min (n - 1) 10 do
      let fi = float_of_int i in
      let p = Vector.get particles i in
      let expected_x = fi +. (1.0 *. dt) in
      let expected_y = (fi *. 2.0) +. (2.0 *. dt) in
      if
        abs_float (p.pos_x -. expected_x) > 1e-3
        || abs_float (p.pos_y -. expected_y) > 1e-3
      then (
        ok := false ;
        if i < 3 then
          Printf.printf
            "    Particle %d: pos=(%.2f,%.2f) expected=(%.2f,%.2f)\n%!"
            i
            p.pos_x
            p.pos_y
            expected_x
            expected_y)
    done ;
    !ok
  in
  Benchmarks.run
    ~verify:verify_particle
    ~filter:Benchmarks.no_interpreter
    "Particle Update"
    run_particle_test ;

  (* Color *)
  let verify_color output _ =
    let n = Vector.length output in
    let ok = ref true in
    for i = 0 to min (n - 1) 10 do
      let c = Vector.get output i in
      if abs_float (c.r -. 0.5) > 1e-3 || abs_float (c.g -. 0.5) > 1e-3 then (
        ok := false ;
        if i < 3 then
          Printf.printf
            "    Color %d: (%.2f,%.2f,%.2f,%.2f) expected (0.5,0.5,0.0,1.0)\n%!"
            i
            c.r
            c.g
            c.b
            c.a)
    done ;
    !ok
  in
  Benchmarks.run
    ~verify:verify_color
    ~filter:Benchmarks.no_interpreter
    "Color Blend"
    run_color_test ;
  Benchmarks.exit ()
