(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: register a Sarek variant type outside kernels via [@@sarek.type].
 * Uses GPU runtime only.
 * Adapted for Benchmarks runner.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

[@@@warning "-32"]

type float32 = float

type color = Red | Value of float32 [@@sarek.type]

let variant_kernel =
  snd
    [%kernel
      fun (xs : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let c = if xs.(tid) > 0.0 then Value xs.(tid) else Red in
          match c with
          | Red -> dst.(tid) <- 0.0
          | Value v -> dst.(tid) <- v +. 1.0]

let run_test dev n _block_size =
  let ir =
    match variant_kernel.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in
  let xs = Vector.create Vector.float32 n in
  let dst = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set xs i (if i mod 2 = 0 then -1.0 else float_of_int i) ;
    Vector.set dst i 0.0
  done ;

  let threads = 64 in
  let grid_x = (n + threads - 1) / threads in
  let t0 = Unix.gettimeofday () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:
      [
        Sarek.Execute.Vec xs;
        Sarek.Execute.Vec dst;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    () ;
  Transfer.flush dev ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in
  (time_ms, dst)

let verify dst _ =
  let n = Vector.length dst in
  let ok = ref true in
  for i = 0 to n - 1 do
    let expected = if i mod 2 = 0 then 0.0 else float_of_int i +. 1.0 in
    let got = Vector.get dst i in
    if abs_float (got -. expected) > 1e-3 then (
      ok := false ;
      if i < 5 then
        Printf.printf "  Mismatch at %d: got %f expected %f\n%!" i got expected)
  done ;
  !ok

let () =
  print_endline "=== Registered Variant IR ===" ;
  print_endline "==============================" ;

  Benchmarks.run ~verify "Registered Variant" run_test ;
  Benchmarks.exit ()
