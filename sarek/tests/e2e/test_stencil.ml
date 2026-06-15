(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Stencil operations
 *
 * Tests 1D stencil pattern.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std

(* Module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

(* Backends auto-register when linked; Benchmarks.init() ensures initialization *)

(* ========== Pure OCaml baseline ========== *)

let ocaml_stencil_1d input output n =
  for i = 1 to n - 2 do
    let left = input.(i - 1) in
    let center = input.(i) in
    let right = input.(i + 1) in
    output.(i) <- (left +. center +. right) /. 3.0
  done

(* ========== Sarek kernel ========== *)

let stencil_1d_kernel =
  [%kernel
    fun (input : float32 vector) (output : float32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid > 0 && tid < n - 1 then
        let left = input.(tid - 1) in
        let center = input.(tid) in
        let right = input.(tid + 1) in
        output.(tid) <- (left +. center +. right) /. 3.0]

(* ========== Benchmark Runner ========== *)

let () =
  let run_stencil (dev : Device.t) size _block_size =
    let n = size in
    let _, kirc = stencil_1d_kernel in
    let ir =
      match kirc.Sarek.Kirc_types.body_ir with
      | Some ir -> ir
      | None -> failwith "No IR"
    in

    (* Initialize data *)
    let input_arr = Array.init n (fun i -> sin (float_of_int i *. 0.1)) in
    let input = Vector.create Vector.float32 n in
    let output = Vector.create Vector.float32 n in

    for i = 0 to n - 1 do
      Vector.set input i input_arr.(i) ;
      Vector.set output i 0.0
    done ;

    let block_size = 256 in
    let grid_size = (n + block_size - 1) / block_size in
    let block = Sarek.Execute.dims1d block_size in
    let grid = Sarek.Execute.dims1d grid_size in

    let t0 = Unix.gettimeofday () in
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [Sarek.Execute.Vec input; Sarek.Execute.Vec output; Sarek.Execute.Int n]
      ~block
      ~grid
      () ;
    Transfer.flush dev ;
    let t1 = Unix.gettimeofday () in

    ((t1 -. t0) *. 1000.0, Vector.to_array output)
  in

  let baseline size =
    let input = Array.init size (fun i -> sin (float_of_int i *. 0.1)) in
    let output = Array.make size 0.0 in
    ocaml_stencil_1d input output size ;
    output
  in

  let verify result expected =
    let n = Array.length expected in
    let errors = ref 0 in
    let tolerance = 0.0001 in
    for i = 1 to n - 2 do
      let diff = abs_float (result.(i) -. expected.(i)) in
      if diff > tolerance then begin
        if !errors < 5 then
          Printf.printf
            "  Mismatch at %d: expected %.6f, got %.6f\n"
            i
            expected.(i)
            result.(i) ;
        incr errors
      end
    done ;
    !errors = 0
  in

  Benchmarks.run ~baseline ~verify "1D Stencil" run_stencil ;
  Benchmarks.exit ()
