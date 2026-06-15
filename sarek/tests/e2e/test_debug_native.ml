(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Debug test for Native runtime kernel execution *)

[@@@warning "-33"] (* Suppress unused open warnings from generated code *)

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_native.Native_plugin.init ()

(* Super simple kernel that just writes to output *)
let simple_write_kernel =
  [%kernel
    fun (output : float32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then output.(tid) <- Float32.of_float 42.0]

(* Kernel that uses the thread id *)
let tid_kernel =
  [%kernel
    fun (output : float32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then output.(tid) <- Float32.of_int (Int32.to_int tid)]

let () =
  let devs = Device.init ~frameworks:["Native"] () in
  let native_dev = devs.(0) in

  Printf.printf "Testing on %s\n%!" native_dev.Device.name ;

  let n = 16 in

  (* Test 1: simple write kernel *)
  Printf.printf "\n--- Test 1: Simple write (should see 42.0) ---\n%!" ;
  let output1 = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set output1 i 0.0
  done ;

  let _, kirc1 = simple_write_kernel in
  let ir1 =
    match kirc1.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "no ir"
  in

  Execute.run_vectors
    ~device:native_dev
    ~ir:ir1
    ~args:[Execute.Vec output1; Execute.Int32 (Int32.of_int n)]
    ~block:(Execute.dims1d 16)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush native_dev ;

  Printf.printf "Result: [" ;
  for i = 0 to min 10 (n - 1) do
    Printf.printf " %.1f" (Vector.get output1 i)
  done ;
  Printf.printf " ]\n%!" ;

  (* Test 2: tid kernel *)
  Printf.printf "\n--- Test 2: Thread ID (should see 0, 1, 2, ...) ---\n%!" ;
  let output2 = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set output2 i 0.0
  done ;

  let _, kirc2 = tid_kernel in
  let ir2 =
    match kirc2.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "no ir"
  in

  Execute.run_vectors
    ~device:native_dev
    ~ir:ir2
    ~args:[Execute.Vec output2; Execute.Int32 (Int32.of_int n)]
    ~block:(Execute.dims1d 16)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush native_dev ;

  Printf.printf "Result: [" ;
  for i = 0 to min 10 (n - 1) do
    Printf.printf " %.1f" (Vector.get output2 i)
  done ;
  Printf.printf " ]\n%!" ;

  print_endline "\nDone."
