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

(******************************************************************************
 * Sarek PPX - Negative Test for Reserved Keywords
 *
 * This test should FAIL to compile with a clear error message about
 * 'double' being a reserved C/CUDA/OpenCL keyword.
 ******************************************************************************)

open Sarek

(* The kernel body below writes `let open Std in`, and the native code the PPX
   generates re-emits that open verbatim -- so `Std` has to resolve in THIS file
   for the case to fail on its refusal and nothing else. It did not: measured with
   the refusal trigger removed, this file failed with "Unbound module Std" (a
   third independent cause of red, after `open Spoc` and the `Kirc` tail). The
   alias below is what the e2e suites already do. *)
module Std = Sarek_stdlib.Std

let test_kernel =
  [%kernel
    let double (x : int32) : int32 = x + x in
    fun (src : int32 vector) (dst : int32 vector) ->
      let open Std in
      let idx = global_idx_x in
      dst.(idx) <- double src.(idx)]

let () =
  let _, kirc = test_kernel in
  ignore kirc
