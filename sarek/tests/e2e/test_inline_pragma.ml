(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - pragma ["sarek.inline N"] for non-tail recursion
 *
 * Tests that non-tail-recursive functions can be inlined using pragma.
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

(* Minimal test: Power of 2 using non-tail recursion *)
let pow2_kernel =
  [%kernel
    let open Std in
    (* Non-tail recursive power of 2: pow2(n) = 2 * pow2(n-1) *)
    let rec pow2 (n : int32) : int32 =
      pragma ["sarek.inline 6"] (if n <= 0l then 1l else 2l * pow2 (n - 1l))
    in
    fun (output : int32 vector) (n : int32) ->
      let idx = global_idx_x in
      if idx = 0l then output.(idx) <- pow2 n]

(* Fibonacci kernel with sufficient inline depth *)
let fib_kernel =
  [%kernel
    let open Std in
    (* Classic Fibonacci: fib(0)=0, fib(1)=1, fib(n) = fib(n-1) + fib(n-2) *)
    let rec fib (n : int32) : int32 =
      pragma
        ["sarek.inline 6"]
        (if n <= 0l then 0l
         else if n = 1l then 1l
         else fib (n - 1l) + fib (n - 2l))
    in
    fun (output : int32 vector) (n : int32) ->
      let idx = global_idx_x in
      if idx = 0l then output.(idx) <- fib n]

(* === runtime Tests === *)

let test_pow2_v2 () =
  let devs = Device.all () in
  if Array.length devs = 0 then (
    print_endline "No runtime devices - skipping GPU runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "runtime: Testing pow2 on: %s\n%!" dev.Device.name ;

    let _, kirc = pow2_kernel in
    let ir =
      match kirc.Sarek.Kirc_types.body_ir with
      | Some ir -> ir
      | None -> failwith "Kernel has no IR"
    in

    let output = Vector.create Vector.int32 1 in
    Vector.set output 0 0l ;
    let n = 5 in

    let block = Execute.dims1d 1 in
    let grid = Execute.dims1d 1 in
    Execute.run_vectors
      ~device:dev
      ~ir
      ~args:[Execute.Vec output; Execute.Int32 (Int32.of_int n)]
      ~block
      ~grid
      () ;
    Transfer.flush dev ;

    let got = Vector.get output 0 in
    let expected = 32l in
    if got = expected then (
      Printf.printf "  PASS: pow2(%d) = %ld (expected %ld)\n" n got expected ;
      true)
    else (
      Printf.printf "  FAIL: pow2(%d) = %ld, expected %ld\n" n got expected ;
      false)

let test_fib_v2 () =
  let devs = Device.all () in
  if Array.length devs = 0 then (
    print_endline "No runtime devices - skipping GPU runtime test" ;
    true)
  else
    let dev = devs.(0) in
    Printf.printf "runtime: Testing fib on: %s\n%!" dev.Device.name ;

    let _, kirc = fib_kernel in
    let ir =
      match kirc.Sarek.Kirc_types.body_ir with
      | Some ir -> ir
      | None -> failwith "Kernel has no IR"
    in

    let test_cases = [(0, 0); (1, 1); (5, 5); (7, 13)] in
    let all_pass = ref true in
    List.iter
      (fun (n, expected) ->
        let output = Vector.create Vector.int32 1 in
        Vector.set output 0 0l ;
        let block = Execute.dims1d 1 in
        let grid = Execute.dims1d 1 in
        Execute.run_vectors
          ~device:dev
          ~ir
          ~args:[Execute.Vec output; Execute.Int32 (Int32.of_int n)]
          ~block
          ~grid
          () ;
        Transfer.flush dev ;
        let got = Vector.get output 0 in
        if got = Int32.of_int expected then
          Printf.printf "  PASS: fib(%d) = %ld\n" n got
        else (
          Printf.printf "  FAIL: fib(%d) = %ld, expected %d\n" n got expected ;
          all_pass := false))
      test_cases ;
    !all_pass

let () =
  print_endline "=== Pragma Inline Tests (runtime) ===" ;
  print_endline "" ;

  print_endline "--- runtime Path ---" ;
  let t1_v2 = test_pow2_v2 () in
  Printf.printf "pow2 (runtime): %s\n" (if t1_v2 then "PASS" else "FAIL") ;
  let t2_v2 = test_fib_v2 () in
  Printf.printf "fib (runtime): %s\n" (if t2_v2 then "PASS" else "FAIL") ;
  print_endline "" ;

  if t1_v2 && t2_v2 then (
    print_endline "=== All pragma inline tests PASSED ===" ;
    exit 0)
  else (
    print_endline "=== Some tests FAILED ===" ;
    exit 1)
