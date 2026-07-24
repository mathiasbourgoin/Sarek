(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile:
   - A kernel helper function is named [sarek_foo], which begins with the
     [sarek_] prefix reserved by the code generator. The reserved-prefix policy
     rejects it at kernel elaboration with the located reservation error. *)

open Spoc
open Sarek

let () =
  let bad_kernel =
    [%kernel
      let sarek_foo (x : int32) : int32 = x + x in
      fun (a : int32 vector) (r : int32 vector) ->
        let open Std in
        let idx = global_idx_x in
        r.(idx) <- sarek_foo a.(idx)]
  in
  let _, kirc = bad_kernel in
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "This should not print - test should have failed to compile"
