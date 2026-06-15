(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: simple ray tracing with PPX Sarek.
 * Renders a small sphere scene to a float32 RGB buffer and checks against CPU.
 * Adapted for Benchmarks runner.
 ******************************************************************************)

(* Shadow SPOC's vector with runtime vector for kernel compatibility *)
type ('a, 'b) vector = ('a, 'b) Spoc_core.Vector.t

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

let ray_kernel =
  [%kernel
    let module Types = struct
      type vec3 = {x : float32; y : float32; z : float32}
    end in
    let make_vec3 (x : float32) (y : float32) (z : float32) : vec3 =
      {x; y; z}
    in
    let dot (a : vec3) (b : vec3) : float32 =
      (a.x *. b.x) +. (a.y *. b.y) +. (a.z *. b.z)
    in
    let normalize (v : vec3) : vec3 =
      let inv = 1.0 /. sqrt (dot v v) in
      make_vec3 (v.x *. inv) (v.y *. inv) (v.z *. inv)
    in
    fun (dirx : float32 vector)
        (diry : float32 vector)
        (dirz : float32 vector)
        (out : float32 vector)
        (n : int32)
      ->
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then (
        let dx = dirx.(tid) in
        let dy = diry.(tid) in
        let dz = dirz.(tid) in
        let dir = normalize (make_vec3 dx dy dz) in
        let dxn = dir.x in
        let dyn = dir.y in
        let dzn = dir.z in
        let a = dot dir dir in
        let half_b = 2.0 *. dzn in
        let c = 3.75 in
        let disc = (half_b *. half_b) -. (a *. c) in
        let idx = tid * 3 in
        if disc > 0.0 then (
          let t = (-.half_b -. sqrt disc) /. a in
          let hx = t *. dxn in
          let hy = t *. dyn in
          let hz = t *. dzn in
          let nx = hx -. 0.0 in
          let ny = hy -. 0.0 in
          let nz = hz +. 2.0 in
          let nrm = normalize (make_vec3 nx ny nz) in
          out.(idx) <- 0.5 *. (nrm.x +. 1.0) ;
          out.(idx + 1) <- 0.5 *. (nrm.y +. 1.0) ;
          out.(idx + 2) <- 0.5 *. (nrm.z +. 1.0))
        else
          let t = 0.5 *. (dyn +. 1.0) in
          let r = 1.0 -. t +. (t *. 0.5) in
          let g = 1.0 -. t +. (t *. 0.7) in
          let b = 1.0 in
          out.(idx) <- r ;
          out.(idx + 1) <- g ;
          out.(idx + 2) <- b)]

let cpu_ray n =
  let w = int_of_float (sqrt (float_of_int n)) in
  let h = n / w in
  let n = w * h in
  (* Adjust n to be rectangular *)

  let bax =
    Array.init n (fun i ->
        let x = i mod w in
        (2.0 *. float_of_int x /. float_of_int (w - 1)) -. 1.0)
  in
  let bay =
    Array.init n (fun i ->
        let y = i / w in
        (2.0 *. float_of_int y /. float_of_int (h - 1)) -. 1.0)
  in
  let baz = Array.make n (-1.5) in
  let expected = Array.make (n * 3) 0.0 in

  for i = 0 to n - 1 do
    let idx = i * 3 in
    let dx = bax.(i) in
    let dy = bay.(i) in
    let dz = baz.(i) in
    let dir_len = sqrt ((dx *. dx) +. (dy *. dy) +. (dz *. dz)) in
    let dxn = dx /. dir_len and dyn = dy /. dir_len and dzn = dz /. dir_len in
    let a = (dxn *. dxn) +. (dyn *. dyn) +. (dzn *. dzn) in
    let half_b = (0.0 *. dxn) +. (0.0 *. dyn) +. (2.0 *. dzn) in
    let c = (2.0 *. 2.0) -. (0.5 *. 0.5) in
    let disc = (half_b *. half_b) -. (a *. c) in
    let r, g, b =
      if disc > 0.0 then
        let t = (-.half_b -. sqrt disc) /. a in
        let hx = t *. dxn and hy = t *. dyn and hz = t *. dzn in
        let nx = hx -. 0.0 and ny = hy -. 0.0 and nz = hz +. 2.0 in
        let inv = 1.0 /. sqrt ((nx *. nx) +. (ny *. ny) +. (nz *. nz)) in
        let nx = nx *. inv and ny = ny *. inv and nz = nz *. inv in
        (0.5 *. (nx +. 1.0), 0.5 *. (ny +. 1.0), 0.5 *. (nz +. 1.0))
      else
        let t = 0.5 *. (dyn +. 1.0) in
        (1.0 -. t +. (t *. 0.5), 1.0 -. t +. (t *. 0.7), 1.0)
    in
    expected.(idx) <- r ;
    expected.(idx + 1) <- g ;
    expected.(idx + 2) <- b
  done ;
  expected

let verify gpu_out expected =
  let len = Array.length expected in
  let n = len / 3 in
  let w = int_of_float (sqrt (float_of_int n)) in
  let h = n / w in
  let ok = ref true in
  let ppm = open_out "/tmp/ray_ppx.ppm" in
  Printf.fprintf ppm "P3\n%d %d\n255\n" w h ;

  for i = 0 to n - 1 do
    let idx = i * 3 in
    let r = expected.(idx) in
    let g = expected.(idx + 1) in
    let b = expected.(idx + 2) in

    let gx = gpu_out.(idx) in
    let gy = gpu_out.(idx + 1) in
    let gz = gpu_out.(idx + 2) in

    if
      abs_float (gx -. r) > 1e-3
      || abs_float (gy -. g) > 1e-3
      || abs_float (gz -. b) > 1e-3
    then (
      ok := false ;
      let x = i mod w in
      let y = i / w in
      Printf.printf
        "Mismatch at %d,%d GPU (%f,%f,%f) CPU (%f,%f,%f)\n%!"
        x
        y
        gx
        gy
        gz
        r
        g
        b) ;

    let to_byte f =
      let v = int_of_float (255.0 *. max 0.0 (min 1.0 f)) in
      max 0 (min 255 v)
    in
    Printf.fprintf ppm "%d %d %d\n" (to_byte r) (to_byte g) (to_byte b)
  done ;
  close_out ppm ;
  if !ok then Printf.printf "Ray PPX PASSED (ppm: /tmp/ray_ppx.ppm)\n%!" ;
  !ok

let () =
  let _, kirc_kernel = ray_kernel in
  print_endline "=== Ray PPX IR ===" ;
  print_endline "==================" ;

  Benchmarks.run ~baseline:cpu_ray ~verify "Ray PPX" (fun dev n _ ->
      match kirc_kernel.Sarek.Kirc_types.body_ir with
      | None ->
          print_endline "No IR available" ;
          (0.0, Array.make (n * 3) 0.0)
      | Some ir ->
          let w = int_of_float (sqrt (float_of_int n)) in
          let h = n / w in
          let n = w * h in
          let threads = 128 in
          let grid_x = (n + threads - 1) / threads in

          (* Initialize ray directions *)
          let bax =
            Array.init n (fun i ->
                let x = i mod w in
                (2.0 *. float_of_int x /. float_of_int (w - 1)) -. 1.0)
          in
          let bay =
            Array.init n (fun i ->
                let y = i / w in
                (2.0 *. float_of_int y /. float_of_int (h - 1)) -. 1.0)
          in
          let baz = Array.make n (-1.5) in

          let dirx = Vector.create Vector.float32 n in
          let diry = Vector.create Vector.float32 n in
          let dirz = Vector.create Vector.float32 n in
          let out = Vector.create Vector.float32 (n * 3) in
          for i = 0 to n - 1 do
            Vector.set dirx i bax.(i) ;
            Vector.set diry i bay.(i) ;
            Vector.set dirz i baz.(i)
          done ;
          for i = 0 to (n * 3) - 1 do
            Vector.set out i 0.0
          done ;

          let block = Sarek.Execute.dims1d threads in
          let grid = Sarek.Execute.dims1d grid_x in

          let t0 = Unix.gettimeofday () in
          Sarek.Execute.run_vectors
            ~device:dev
            ~ir
            ~args:
              [
                Sarek.Execute.Vec dirx;
                Sarek.Execute.Vec diry;
                Sarek.Execute.Vec dirz;
                Sarek.Execute.Vec out;
                Sarek.Execute.Int n;
              ]
            ~block
            ~grid
            () ;
          Transfer.flush dev ;
          let t1 = Unix.gettimeofday () in

          let gpu_out = Vector.to_array out in
          ((t1 -. t0) *. 1000.0, gpu_out)) ;
  Benchmarks.exit ()
