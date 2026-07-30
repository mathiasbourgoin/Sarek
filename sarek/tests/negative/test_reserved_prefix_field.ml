(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* backlog-208: this case used to carry `open Spoc` while its dune stanza
   declared only `sarek sarek.stdlib sarek.ppx.lib`. There is no `Spoc` module in
   this tree at all (nor a `Kirc` one -- both are SPOC-v1 vestige), so the file
   only ever compiled as far as the PPX: the refusal below fired before
   type-checking reached the `open`, and `-w -33` hid the unused-open hint. The
   case was therefore red for TWO reasons, and would have looked exactly the same
   if the refusal had stopped working -- an unbound-module error is still a
   non-zero exit with the file's name on it. The `open` and the dead
   `Kirc.print_ast` tail are gone so the red is the refusal and nothing else. *)

(* Test kernel that should FAIL to compile:
   - A record type declared for the kernel has a field named [sarek_len], which
     begins with the [sarek_] prefix reserved by the code generator. The
     reserved-prefix policy rejects it at kernel elaboration with the located
     reservation error. *)

open Sarek

(* The kernel body below writes `let open Std in`, and the native code the PPX
   generates re-emits that open verbatim -- so `Std` has to resolve in THIS file
   for the case to fail on its refusal and nothing else. It did not: measured with
   the refusal trigger removed, this file failed with "Unbound module Std" (a
   third independent cause of red, after `open Spoc` and the `Kirc` tail). The
   alias below is what the e2e suites already do. *)
module Std = Sarek_stdlib.Std

let () =
  let bad_kernel =
    [%kernel
      let module Types = struct
        type box = {sarek_len : int32; value : float32}
      end in
      let make_box (n : int32) (v : float32) : box =
        {sarek_len = n; value = v}
      in
      fun (a : float32 vector) (r : float32 vector) ->
        let open Std in
        let idx = global_idx_x in
        let b = make_box idx a.(idx) in
        r.(idx) <- b.value]
  in
  let _, kirc = bad_kernel in
  ignore kirc ;
  print_endline "This should not print - test should have failed to compile"
