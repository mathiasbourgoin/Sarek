(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Regression test for TECreateArray native codegen (Sarek_native_gen_expr.ml).
   Pre-fix, `create_array` in a kernel body was lowered to
   [Array.make size_e default_e] where [size_e] evaluates at type [int32]
   while [Array.make] expects [int]. This is a native-mode compile-time type
   error ("This expression has type int32 but an expression was expected of
   type int"), i.e. this file fails to build at all pre-fix. Post-fix, the
   size expression is converted with [Int32.to_int] and the kernel compiles
   and runs correctly. *)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_native.Native_plugin.init ()

(* Kernel that creates a local array, fills it, and writes back a computed
   value that depends on the array contents so the test genuinely exercises
   both allocation and use of the created array. *)
let create_array_kernel =
  [%kernel
    fun (output : int32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        let arr = create_array n Local in
        arr.(tid) <- (tid * 2l) + 1l ;
        output.(tid) <- arr.(tid)
      end]

let () =
  let devs = Device.init ~frameworks:["Native"] () in
  let native_dev = devs.(0) in
  let n = 16 in
  let output = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    Vector.set output i 0l
  done ;

  let _, kirc = create_array_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "no ir"
  in

  Execute.run_vectors
    ~device:native_dev
    ~ir
    ~args:[Execute.Vec output; Execute.Int32 (Int32.of_int n)]
    ~block:(Execute.dims1d n)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush native_dev ;

  let ok = ref true in
  for i = 0 to n - 1 do
    let expected = Int32.add (Int32.mul (Int32.of_int i) 2l) 1l in
    let got = Vector.get output i in
    if got <> expected then begin
      Printf.printf "MISMATCH at %d: expected %ld, got %ld\n%!" i expected got ;
      ok := false
    end
  done ;
  if !ok then print_endline "test_native_create_array: PASSED"
  else begin
    print_endline "test_native_create_array: FAILED" ;
    exit 1
  end
