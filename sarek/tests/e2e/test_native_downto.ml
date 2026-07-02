(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Regression test for `downto` loop native codegen (Sarek_native_gen_expr.ml).
   The parser (Sarek_parse.ml) stores `for i = start downto stop` positionally
   as (lo, hi) = (start, stop), regardless of direction. Pre-fix, the Downto
   codegen arm emitted `for i = hi downto lo`, i.e. for `for i = 9 downto 0`
   it emitted the OCaml loop `for i = 0 downto 9`, which never executes any
   iteration (0 is not >= 9). Post-fix it emits `for i = lo downto hi`
   ( = `for i = 9 downto 0`), running the expected 10 descending
   iterations. *)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_native.Native_plugin.init ()

(* Writes the loop counter of `for i = 9 downto 0` into successive output
   slots, so the recorded sequence exposes both iteration count and order. *)
let downto_kernel =
  [%kernel
    fun (output : int32 vector) (count : int32 vector) ->
      let open Std in
      let tid = global_thread_id in
      if tid = 0l then begin
        let idx = mut 0l in
        for i = 9 downto 0 do
          output.(idx) <- i ;
          idx := idx + 1l
        done ;
        count.(0) <- idx
      end]

let () =
  let devs = Device.init ~frameworks:["Native"] () in
  let native_dev = devs.(0) in
  let n = 10 in
  let output = Vector.create Vector.int32 n in
  let count = Vector.create Vector.int32 1 in
  for i = 0 to n - 1 do
    Vector.set output i (-1l)
  done ;
  Vector.set count 0 (-1l) ;

  let _, kirc = downto_kernel in
  let ir =
    match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir
    | None -> failwith "no ir"
  in

  Execute.run_vectors
    ~device:native_dev
    ~ir
    ~args:[Execute.Vec output; Execute.Vec count]
    ~block:(Execute.dims1d 1)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush native_dev ;

  let got_count = Vector.get count 0 in
  let ok = ref (got_count = Int32.of_int n) in
  if not !ok then
    Printf.printf
      "MISMATCH: expected %d iterations, got %ld\n%!"
      n
      got_count ;
  for i = 0 to n - 1 do
    let expected = Int32.of_int (9 - i) in
    let got = Vector.get output i in
    if got <> expected then begin
      Printf.printf
        "MISMATCH at slot %d: expected %ld, got %ld\n%!"
        i
        expected
        got ;
      ok := false
    end
  done ;
  if !ok then print_endline "test_native_downto: PASSED"
  else begin
    print_endline "test_native_downto: FAILED" ;
    exit 1
  end
