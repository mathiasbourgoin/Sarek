(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: register a Sarek record type outside kernels via [%sarek.type].
 * Adapted for Benchmarks runner.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

[@@@warning "-32"]

type float32 = float

type point = {x : float32; y : float32} [@@sarek.type]

let kernel =
  [%kernel
    fun (xs : float32 vector)
        (ys : float32 vector)
        (dst : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then
        let p = {x = xs.(tid); y = ys.(tid)} in
        dst.(tid) <- sqrt ((p.x *. p.x) +. (p.y *. p.y))]

let run_test dev n _block_size =
  let xv = Vector.create Vector.float32 n in
  let yv = Vector.create Vector.float32 n in
  let dst = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set xv i (float_of_int i) ;
    Vector.set yv i (float_of_int (n - i)) ;
    Vector.set dst i 0.0
  done ;
  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let _, kirc = kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
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
        Sarek.Execute.Vec xv;
        Sarek.Execute.Vec yv;
        Sarek.Execute.Vec dst;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, (xv, yv, dst))

let verify (xv, yv, dst) _ =
  let n = Vector.length xv in
  let ok = ref true in
  for i = 0 to n - 1 do
    let x = Vector.get xv i in
    let y = Vector.get yv i in
    let expected = sqrt ((x *. x) +. (y *. y)) in
    let got = Vector.get dst i in
    if abs_float (got -. expected) > 1e-3 then (
      ok := false ;
      if i < 5 then
        Printf.printf "  Mismatch at %d: got %f expected %f\n%!" i got expected)
  done ;
  !ok

let () =
  let _native, _kirc = kernel in
  print_endline "=== Registered type IR ===" ;
  print_endline "==========================" ;

  Benchmarks.run ~verify "Registered Type" run_test ;
  Benchmarks.exit ()
