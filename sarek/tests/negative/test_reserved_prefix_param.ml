(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernel that should FAIL to compile:
   - A kernel parameter is named [sarek_smod], which begins with the [sarek_]
     prefix reserved by the code generator (this is the exact name of the GLSL
     integer-remainder helper, PR #255). The reserved-prefix policy rejects it
     at kernel elaboration with the located reservation error. *)

open Spoc
open Sarek

let () =
  let bad_kernel =
    [%kernel
      fun (a : int32 vector) (r : int32 vector) (sarek_smod : int32) ->
        let open Std in
        let idx = global_idx_x in
        r.(idx) <- a.(idx) mod sarek_smod]
  in
  let _, kirc = bad_kernel in
  Kirc.print_ast kirc.Kirc.body ;
  print_endline "This should not print - test should have failed to compile"
