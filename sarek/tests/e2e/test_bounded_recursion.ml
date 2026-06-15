(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for Sarek PPX - Tail recursion transformation
 *
 * Tests that tail-recursive functions are correctly transformed to loops.
 * Non-tail recursion is not currently supported (requires manual conversion
 * to tail-recursive form using accumulators).
 * GPU runtime only.
 ******************************************************************************)

module Std = Sarek_stdlib.Std
open Sarek
module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

(* Force backend registration *)
let () =
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init ()

(* Extract kirc for runtime - factorial *)
let factorial_kirc =
  snd
    [%kernel
      let open Std in
      let rec fact_aux (acc : int32) (n : int32) : int32 =
        if n <= 1l then acc else fact_aux (acc * n) (n - 1l)
      in
      fun (output : int32 vector) (n : int32) ->
        let idx = global_idx_x in
        if idx = 0l then output.(idx) <- fact_aux 1l n]

(* Extract kirc for runtime - power *)
let power_kirc =
  snd
    [%kernel
      let open Std in
      let rec power_aux (acc : int32) (base : int32) (exp : int32) : int32 =
        if exp <= 0l then acc else power_aux (acc * base) base (exp - 1l)
      in
      fun (output : int32 vector) (base : int32) (exp : int32) ->
        let idx = global_idx_x in
        if idx = 0l then output.(idx) <- power_aux 1l base exp]

(* Extract kirc for runtime - GCD *)
let gcd_kirc =
  snd
    [%kernel
      let open Std in
      let rec gcd (a : int32) (b : int32) : int32 =
        if b = 0l then a else gcd b (a mod b)
      in
      fun (output : int32 vector) (a : int32) (b : int32) ->
        let idx = global_idx_x in
        if idx = 0l then output.(idx) <- gcd a b]

(* runtime test for tail-recursive factorial *)
let test_factorial_v2 dev =
  print_endline "=== Tail-recursive factorial ===" ;
  Printf.printf "runtime Testing on: %s\n%!" dev.Device.name ;

  let output = Vector.create Vector.int32 1 in
  Vector.set output 0 0l ;
  let n = 10l in

  let ir =
    match factorial_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  let t0 = Unix.gettimeofday () in
  try
    Sarek.Execute.run_vectors
      ~device:dev
      ~block:(Sarek.Execute.dims1d 1)
      ~grid:(Sarek.Execute.dims1d 1)
      ~ir
      ~args:[Sarek.Execute.Vec output; Sarek.Execute.Int32 n]
      () ;
    Transfer.flush dev ;
    let t1 = Unix.gettimeofday () in
    Printf.printf "  runtime exec: %.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;

    let got = Vector.get output 0 in
    (* 10! = 3628800 *)
    let expected = 3628800l in
    if got = expected then (
      Printf.printf "runtime PASS: fact(10) = %ld\n%!" got ;
      true)
    else (
      Printf.printf
        "runtime FAIL: fact(10) = %ld, expected %ld\n%!"
        got
        expected ;
      false)
  with e ->
    Printf.printf "runtime FAIL: %s\n%!" (Printexc.to_string e) ;
    false

(* runtime test for tail-recursive power *)
let test_power_v2 dev =
  print_endline "=== Tail-recursive power ===" ;
  Printf.printf "runtime Testing on: %s\n%!" dev.Device.name ;

  let output = Vector.create Vector.int32 1 in
  Vector.set output 0 0l ;
  let base = 2l in
  let exp = 10l in

  let ir =
    match power_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  let t0 = Unix.gettimeofday () in
  try
    Sarek.Execute.run_vectors
      ~device:dev
      ~block:(Sarek.Execute.dims1d 1)
      ~grid:(Sarek.Execute.dims1d 1)
      ~ir
      ~args:
        [
          Sarek.Execute.Vec output;
          Sarek.Execute.Int32 base;
          Sarek.Execute.Int32 exp;
        ]
      () ;
    Transfer.flush dev ;
    let t1 = Unix.gettimeofday () in
    Printf.printf "  runtime exec: %.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;

    let got = Vector.get output 0 in
    (* 2^10 = 1024 *)
    let expected = 1024l in
    if got = expected then (
      Printf.printf "runtime PASS: power(2, 10) = %ld\n%!" got ;
      true)
    else (
      Printf.printf
        "runtime FAIL: power(2, 10) = %ld, expected %ld\n%!"
        got
        expected ;
      false)
  with e ->
    Printf.printf "runtime FAIL: %s\n%!" (Printexc.to_string e) ;
    false

(* runtime test for tail-recursive GCD *)
let test_gcd_v2 dev =
  print_endline "=== Tail-recursive GCD ===" ;
  Printf.printf "runtime Testing on: %s\n%!" dev.Device.name ;

  let output = Vector.create Vector.int32 1 in
  Vector.set output 0 0l ;
  let a = 48l in
  let b = 18l in

  let ir =
    match gcd_kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "Kernel has no IR"
  in

  let t0 = Unix.gettimeofday () in
  try
    Sarek.Execute.run_vectors
      ~device:dev
      ~block:(Sarek.Execute.dims1d 1)
      ~grid:(Sarek.Execute.dims1d 1)
      ~ir
      ~args:
        [Sarek.Execute.Vec output; Sarek.Execute.Int32 a; Sarek.Execute.Int32 b]
      () ;
    Transfer.flush dev ;
    let t1 = Unix.gettimeofday () in
    Printf.printf "  runtime exec: %.2f ms\n%!" ((t1 -. t0) *. 1000.0) ;

    let got = Vector.get output 0 in
    (* gcd(48, 18) = 6 *)
    let expected = 6l in
    if got = expected then (
      Printf.printf "runtime PASS: gcd(48, 18) = %ld\n%!" got ;
      true)
    else (
      Printf.printf
        "runtime FAIL: gcd(48, 18) = %ld, expected %ld\n%!"
        got
        expected ;
      false)
  with e ->
    Printf.printf "runtime FAIL: %s\n%!" (Printexc.to_string e) ;
    false

let () =
  print_endline "=== Tail Recursion Transformation Tests (runtime) ===" ;
  print_endline "" ;

  let devs =
    Device.init ~frameworks:["CUDA"; "OpenCL"; "Native"; "Interpreter"] ()
  in
  if Array.length devs = 0 then (
    print_endline "No runtime devices found - skipping" ;
    exit 0) ;

  let dev = devs.(0) in

  let t1 = test_factorial_v2 dev in
  print_endline "" ;

  let t2 = test_power_v2 dev in
  print_endline "" ;

  let t3 = test_gcd_v2 dev in
  print_endline "" ;

  print_endline "=== Summary ===" ;
  Printf.printf "runtime Factorial: %s\n" (if t1 then "PASS" else "FAIL") ;
  Printf.printf "runtime Power: %s\n" (if t2 then "PASS" else "FAIL") ;
  Printf.printf "runtime GCD: %s\n" (if t3 then "PASS" else "FAIL") ;

  if t1 && t2 && t3 then (
    print_endline "\nAll tests passed!" ;
    exit 0)
  else (
    print_endline "\nSome tests failed!" ;
    exit 1)
