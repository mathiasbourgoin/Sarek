(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - Test for Polymorphic Module Functions
 *
 * Tests that [@sarek.module] functions can be polymorphic and used at
 * multiple types within kernels.
 * GPU runtime only.
 ******************************************************************************)

(* Module aliases to avoid conflicts *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
open Sarek
module Std = Sarek_stdlib.Std

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

(* Define a polymorphic identity function in module scope *)
let[@sarek.module] identity (x : 'a) : 'a = x

let test_kernel =
  [%kernel
    fun (src_i : int32 vector)
        (src_f : float32 vector)
        (dst_i : int32 vector)
        (dst_f : float32 vector) ->
      let open Std in
      let idx = global_idx_x in
      (* Use identity at int32 *)
      dst_i.(idx) <- identity src_i.(idx) ;
      (* Use identity at float32 *)
      dst_f.(idx) <- identity src_f.(idx)]

(* === runtime Test === *)

let test_poly_identity_v2 () =
  print_endline "=== runtime: Polymorphic module identity ===" ;

  let devs = Device.all () in
  if Array.length devs = 0 then (
    print_endline "No runtime devices - skipping runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "Testing on: %s\n%!" dev.Device.name ;

    let _, kirc = test_kernel in
    let ir =
      match kirc.Sarek.Kirc_types.body_ir with
      | Some ir -> ir
      | None -> failwith "Kernel has no IR"
    in

    let n = 1024 in
    let src_i = Vector.create Vector.int32 n in
    let src_f = Vector.create Vector.float32 n in
    let dst_i = Vector.create Vector.int32 n in
    let dst_f = Vector.create Vector.float32 n in

    for i = 0 to n - 1 do
      Vector.set src_i i (Int32.of_int i) ;
      Vector.set src_f i (float_of_int i)
    done ;

    let block = Execute.dims1d 256 in
    let grid = Execute.dims1d (n / 256) in
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:
        [
          Execute.Vec src_i;
          Execute.Vec src_f;
          Execute.Vec dst_i;
          Execute.Vec dst_f;
        ]
      ~block
      ~grid
      () ;
    Transfer.flush dev ;

    let ok = ref true in
    for i = 0 to n - 1 do
      let expected_i = Int32.of_int i in
      let expected_f = float_of_int i in
      let got_i = Vector.get dst_i i in
      let got_f = Vector.get dst_f i in
      if got_i <> expected_i then (
        Printf.printf "FAIL: dst_i[%d] = %ld, expected %ld\n" i got_i expected_i ;
        ok := false) ;
      if abs_float (got_f -. expected_f) > 0.001 then (
        Printf.printf "FAIL: dst_f[%d] = %f, expected %f\n" i got_f expected_f ;
        ok := false)
    done ;
    if !ok then print_endline "PASS: runtime Polymorphic module identity" ;
    !ok

let () =
  print_endline "=== Polymorphic Module Function Tests (runtime) ===" ;
  print_endline "" ;

  let t1_v2 = test_poly_identity_v2 () in
  print_endline "" ;

  print_endline "=== Summary ===" ;
  Printf.printf
    "Polymorphic module identity (runtime): %s\n"
    (if t1_v2 then "PASS" else "FAIL") ;

  if t1_v2 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
