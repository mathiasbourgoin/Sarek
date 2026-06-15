(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - End-to-End Test for Module Functions and Polymorphism
 *
 * Tests:
 * 1. Basic module functions with concrete types
 * 2. Polymorphic module functions used at multiple types
 * GPU runtime only.
 ******************************************************************************)

module Std = Sarek_stdlib.Std
open Sarek

(* runtime module aliases *)
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init ()

let basic_kernel =
  [%kernel
    let add_one (x : int32) : int32 = x + 1l in
    fun (src : int32 vector) (dst : int32 vector) ->
      let open Std in
      let idx = global_idx_x in
      dst.(idx) <- add_one src.(idx)]

let times_two_kernel =
  [%kernel
    let times_two (x : int32) : int32 = x + x in
    fun (src : int32 vector) (dst : int32 vector) ->
      let open Std in
      let idx = global_idx_x in
      dst.(idx) <- times_two src.(idx)]

let identity_kernel =
  [%kernel
    let identity (x : 'a) : 'a = x in
    fun (src_i : int32 vector)
        (src_f : float32 vector)
        (dst_i : int32 vector)
        (dst_f : float32 vector)
      ->
      let open Std in
      let idx = global_idx_x in
      (* Use identity at int32 *)
      dst_i.(idx) <- identity src_i.(idx) ;
      (* Use identity at float32 *)
      dst_f.(idx) <- identity src_f.(idx)]

(* ========== runtime Tests ========== *)

let test_basic_v2 () =
  let _, kirc = basic_kernel in
  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then (
    print_endline "runtime: No device - SKIPPED" ;
    true)
  else
    let dev = devs.(0) in
    match kirc.Sarek.Kirc_types.body_ir with
    | None ->
        print_endline "runtime: No IR - SKIPPED" ;
        true
    | Some ir ->
        let n = 1024 in
        let src = Vector.create Vector.int32 n in
        let dst = Vector.create Vector.int32 n in
        for i = 0 to n - 1 do
          Vector.set src i (Int32.of_int i)
        done ;

        let block = Execute.dims1d 256 in
        let grid = Execute.dims1d (n / 256) in
        Execute.run_vectors
          ~device:dev
          ~ir
          ~args:[Execute.Vec src; Execute.Vec dst]
          ~block
          ~grid
          () ;
        Transfer.flush dev ;

        let ok = ref true in
        for i = 0 to n - 1 do
          let expected = Int32.of_int (i + 1) in
          if Vector.get dst i <> expected then (
            Printf.printf
              "runtime FAIL: dst[%d] = %ld, expected %ld\n"
              i
              (Vector.get dst i)
              expected ;
            ok := false)
        done ;
        if !ok then print_endline "PASS: Basic module function (runtime)" ;
        !ok

let test_times_two_v2 () =
  let _, kirc = times_two_kernel in
  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then (
    print_endline "runtime: No device - SKIPPED" ;
    true)
  else
    let dev = devs.(0) in
    match kirc.Sarek.Kirc_types.body_ir with
    | None ->
        print_endline "runtime: No IR - SKIPPED" ;
        true
    | Some ir ->
        let n = 1024 in
        let src = Vector.create Vector.int32 n in
        let dst = Vector.create Vector.int32 n in
        for i = 0 to n - 1 do
          Vector.set src i (Int32.of_int i)
        done ;

        let block = Execute.dims1d 256 in
        let grid = Execute.dims1d (n / 256) in
        Execute.run_vectors
          ~device:dev
          ~ir
          ~args:[Execute.Vec src; Execute.Vec dst]
          ~block
          ~grid
          () ;
        Transfer.flush dev ;

        let ok = ref true in
        for i = 0 to n - 1 do
          let expected = Int32.of_int (2 * i) in
          if Vector.get dst i <> expected then (
            Printf.printf
              "runtime FAIL: dst[%d] = %ld, expected %ld\n"
              i
              (Vector.get dst i)
              expected ;
            ok := false)
        done ;
        if !ok then print_endline "PASS: Times two (runtime)" ;
        !ok

let test_identity_v2 () =
  let _, kirc = identity_kernel in
  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then (
    print_endline "runtime: No device - SKIPPED" ;
    true)
  else
    let dev = devs.(0) in
    match kirc.Sarek.Kirc_types.body_ir with
    | None ->
        print_endline "runtime: No IR - SKIPPED" ;
        true
    | Some ir ->
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
          if Vector.get dst_i i <> expected_i then (
            Printf.printf
              "runtime FAIL: dst_i[%d] = %ld, expected %ld\n"
              i
              (Vector.get dst_i i)
              expected_i ;
            ok := false) ;
          if abs_float (Vector.get dst_f i -. expected_f) > 0.001 then (
            Printf.printf
              "runtime FAIL: dst_f[%d] = %f, expected %f\n"
              i
              (Vector.get dst_f i)
              expected_f ;
            ok := false)
        done ;
        if !ok then print_endline "PASS: Polymorphic identity (runtime)" ;
        !ok

(* ========== Interpreter Tests ========== *)

let test_basic_interpreter () =
  let _, kirc = basic_kernel in
  match kirc.Sarek.Kirc_types.body_ir with
  | None ->
      print_endline "Interpreter: No IR - SKIPPED" ;
      true
  | Some ir ->
      let n = 1024 in
      let src =
        Array.init n (fun i -> Sarek_ir_interp.VInt32 (Int32.of_int i))
      in
      let dst = Array.make n (Sarek_ir_interp.VInt32 0l) in

      Sarek_ir_interp.run_kernel
        ir
        ~block:(256, 1, 1)
        ~grid:(n / 256, 1, 1)
        [
          ("src", Sarek_ir_interp.ArgArray src);
          ("dst", Sarek_ir_interp.ArgArray dst);
        ] ;

      let ok = ref true in
      for i = 0 to n - 1 do
        let expected = Int32.of_int (i + 1) in
        let got =
          match dst.(i) with Sarek_ir_interp.VInt32 v -> v | _ -> 0l
        in
        if got <> expected then (
          if !ok then
            Printf.printf
              "Interpreter FAIL: dst[%d] = %ld, expected %ld\n"
              i
              got
              expected ;
          ok := false)
      done ;
      if !ok then print_endline "PASS: Basic module function (Interpreter)" ;
      !ok

let () =
  print_endline "=== Module Functions & Polymorphism Tests (runtime) ===" ;
  print_endline "" ;

  print_endline "=== runtime Tests ===" ;
  let v1 = test_basic_v2 () in
  let v2 = test_times_two_v2 () in
  let v3 = test_identity_v2 () in
  print_endline "" ;

  print_endline "=== Interpreter Tests ===" ;
  let i1 = test_basic_interpreter () in
  print_endline "" ;

  print_endline "=== Summary ===" ;
  Printf.printf
    "Basic module function (runtime): %s\n"
    (if v1 then "PASS" else "FAIL") ;
  Printf.printf "Times two (runtime): %s\n" (if v2 then "PASS" else "FAIL") ;
  Printf.printf
    "Polymorphic identity (runtime): %s\n"
    (if v3 then "PASS" else "FAIL") ;
  Printf.printf
    "Basic module function (Interpreter): %s\n"
    (if i1 then "PASS" else "FAIL") ;

  if v1 && v2 && v3 && i1 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
